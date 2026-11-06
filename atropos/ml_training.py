"""Walk-forward ML training & evaluation engine (data/model-agnostic).

:func:`run_ml_kill_protocol` takes a ready dataset -- a feature frame,
a binary target, per-event executable returns and int64-microsecond
timestamps -- walks an expanding-block walk-forward CV over it (purge +
embargo between train and test), fits every requested model from the
open :data:`MODEL_REGISTRY` through the identical path (fit -> OOS
probabilities -> threshold -> EV), and evaluates four configurable
gates per model (plus an opt-in fifth, default off):

* ``incremental``          the model's taken-trade EV must beat a
                           linear-battery logit (features + caller
                           supplied control columns) by a paired
                           bootstrap t;
* ``cost_line``            the merged-OOS EV of taken trades must
                           clear ``profit_bar``;
* ``importance_stability`` per-fold permutation importances must rank
                           consistently (median pairwise Spearman);
* ``deflated_sharpe``      the deflated Sharpe ratio (Bailey & Lopez
                           de Prado 2014) of the daily EV stream must
                           clear ``dsr_min``;
* ``robustness``           OPT-IN (``checks.robustness``, default
                           ``False``): the taken-book EV must keep its
                           sign when every fold threshold is shifted
                           by ``checks.robustness_shift`` up and down
                           (threshold-sensitivity battery).

Evaluation verdicts of a run: ``SURVIVES`` (every requested model
passes all enabled gates with enough independent events),
``DEAD(<gate>)`` (first failing gate of the first non-surviving
model) and ``UNDERPOWERED`` (fewer than ``n_min_events`` declustered
events -- reported, not a gate failure, exactly like the sample-size
gate it descends from).  Per-model verdicts always live in
``report["models"][name]`` so several models can be compared without
the engine selecting one.

The input contract is direction-agnostic: a long/short protocol runs
the engine twice, once per direction, with the caller's own labeling
rule (e.g. ``exec = fwd - cost`` / ``-fwd - cost``,
``y = exec > bar``).  Defaults originate from a production
walk-forward protocol and are reasonable starting points, not frozen
constants of this module.

The kill verdict is anchored or it is not: a run whose config carries
no ``prereg_hash`` issues a :class:`RuntimeWarning` (the verdict is
then unanchored to a frozen preregistration) while the hash itself is
echoed into the report when supplied.  After deployment,
:func:`ml_forward_check` bridges the walk-forward report and the live
taken-book track to :func:`atropos.track_record.live_verdict` -- a
helper, not a gate: forward tracking happens after deployment, never
inside the walk-forward.

The model fleet is open: :data:`MODEL_REGISTRY` ships eight bundled
factories spanning complementary bias axes -- linear (``logit``),
exact-split boosting (``gbm``), histogram boosting (``hist_gbm``),
optimized-split bagging (``rf``), random-split decorrelation
(``extratrees``), neural (``mlp``), local neighbourhood (``knn``)
and generative quadratic (``qda``) -- and accepts caller-registered
entries with no engine changes.

Run ``python -m atropos.ml_training`` for the synthetic verification
cases: a planted nonlinear edge must SURVIVES for the tree models
while the linear battery stays blind (incremental gate), pure noise
must not survive, a thin sample must come back UNDERPOWERED, the
full eight-model registry must complete one comparison pass with
SURVIVES the majority outcome, and a statistics-calibrated
reproduction of a documented tombstone (closed donor corpus: the
VRP-panic conditional short straddle, calibrated to the tombstone's
marginal statistics) must come back ``DEAD(...)`` with both
documented cause-of-death gates triggered (cost line + incremental).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier as _ETClass
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.ensemble import RandomForestClassifier as _RFClass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atropos._utils import (
    clean_1d,
    purge_embargo_mask,
    sane_variance_mask,
    validate_int,
    validate_number,
)
from atropos.evaluation_metrics import deflated_sharpe_from_moments
from atropos.track_record import live_verdict

__all__ = [
    "DEFAULT_TRAINING_CONFIG",
    "MODEL_REGISTRY",
    "assign_folds",
    "decluster_count",
    "deflated_sharpe_value",
    "ev_of_taken",
    "fold_edges",
    "make_extratrees",
    "make_gbm",
    "make_hist_gbm",
    "make_knn",
    "make_logit",
    "make_mlp",
    "make_qda",
    "make_rf",
    "ml_forward_check",
    "paired_bootstrap_t_stat",
    "permutation_importance_vec",
    "pick_threshold",
    "purge_embargo_mask",
    "rolling_z",
    "run_ml_kill_protocol",
    "spearman_median",
    "stream_stats",
]

ModelFactory = Callable[..., Any]
"""Registry entry: ``factory(**model_params[name]) -> estimator``.

The returned estimator must expose sklearn-style ``fit`` /
``predict_proba`` / ``predict`` and be deterministic for a fixed
seed (no free-threading parallelism).  Registering a custom factory
(``MODEL_REGISTRY["x"] = factory``) immediately makes it available to
:data:`DEFAULT_TRAINING_CONFIG` keys ``model`` / ``models`` and to the
full engine path.
"""

_HOUR_US = 3_600 * 1_000_000
_DAY_US = 24 * _HOUR_US
_DEFAULT_GRID = tuple(round(0.30 + 0.05 * step, 2) for step in range(9))

DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    # model selection: primary model (flat report aliases are its
    # metrics) and/or several models evaluated in ONE CV pass
    # (comparison only -- the engine never selects among them)
    "model": "gbm",
    "models": None,  # None -> [model]; list[str] -> run each
    # evaluation-gate toggles and thresholds (see module docstring);
    # robustness is OPT-IN and default OFF: enabling it adds a fifth
    # kills entry (and a possible DEAD(robustness) verdict) without
    # touching the four-gate contract of existing runs
    "checks": {
        "incremental": True,
        "incremental_t": 2.0,
        "cost_line": True,
        "importance_stability": True,
        "importance_rho_min": 0.5,
        "deflated_sharpe": True,
        "dsr_min": 0.95,
        "robustness": False,
        "robustness_shift": 0.20,
    },
    # hash of the frozen preregistration (e.g. the spec hash from
    # generate_prereg); None is LEGAL (backward compat) but warns at
    # run time -- an unanchored kill verdict -- and the value, when
    # given, is echoed into the report
    "prereg_hash": None,
    # defaults originating from a production walk-forward protocol;
    # cost_rt is consumed by the CALLER when building exec returns
    # (exec = fwd - cost), profit_bar is the EV floor of cost_line
    "cost_rt": 0.0020,
    "profit_bar": 0.0040,
    "horizon_h": 1,  # label horizon feeding the purge window
    "purge_h": 2,
    "embargo_h": 24,
    "k_blocks": 6,
    "threshold_grid": tuple(
        round(0.30 + 0.05 * step, 2) for step in range(9)
    ),
    "n_min_events": 500,
    "bootstrap_b": 2000,
    "bootstrap_seed": 42,
    "model_seed": 42,
    "importance_seed": 42,
    "n_trials_ds": 5,
    # provenance of the source feature recipe; carried for caller-side
    # dataset construction, the engine itself never uses them
    "z_window_h": 720,
    "daily_z_window_d": 90,
    # factory keyword arguments per model name (validated in factories)
    "model_params": {
        "extratrees": {
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 20,
        },
        "gbm": {
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_estimators": 400,
            "subsample": 0.8,
        },
        "hist_gbm": {
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_depth": 3,
        },
        "knn": {"n_neighbors": 50, "weights": "distance"},
        "logit": {"max_iter": 1000},
        "mlp": {
            "hidden_layer_sizes": (32, 16),
            "max_iter": 500,
            "early_stopping": False,
        },
        "qda": {"reg_param": 0.5},
        "rf": {
            "n_estimators": 400,
            "max_depth": 5,
            "min_samples_leaf": 50,
        },
    },
}

# integer / non-negative-number scalar parameters of the config
_INT_PARAMS: dict[str, int] = {
    "horizon_h": 1,
    "purge_h": 0,
    "embargo_h": 0,
    "k_blocks": 2,
    "n_min_events": 1,
    "bootstrap_b": 2,
    "bootstrap_seed": 0,
    "model_seed": 0,
    "importance_seed": 0,
    "n_trials_ds": 2,
    "z_window_h": 2,
    "daily_z_window_d": 2,
}
_FLOAT_PARAMS: dict[str, float] = {"cost_rt": 0.0, "profit_bar": 0.0}
_CHECK_BOOLS = (
    "incremental",
    "cost_line",
    "importance_stability",
    "deflated_sharpe",
    "robustness",
)
_CHECK_NUMBERS = (
    "incremental_t",
    "importance_rho_min",
    "dsr_min",
    "robustness_shift",
)

_BASELINE_NAME = "baseline"
_BATTERY_NAME = "battery"

_ROBUSTNESS_MIN_SHIFT_EVENTS = 10
"""A shifted book below this many taken trades cannot power the
sensitivity comparison: the robustness gate degrades to INCONCLUSIVE
instead of reading noise off a handful of trades."""


# ---------------------------------------------------------------------------
# pure mechanics (deterministic primitives)
# ---------------------------------------------------------------------------
def rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing z-score over the last ``window`` observations.

    ``z_i = (v_i - mean(v_{i-w+1..i})) / sd(v_{i-w+1..i})`` with the
    sample sd (ddof=1): NaN until ``window`` observations exist and NaN
    whenever the window's sd is zero (constant window).  Purely causal
    -- appending future values never changes past outputs.

    Args:
        values: 1-D float array (NaNs propagate to every window that
            contains them).
        window: trailing window length, integer >= 2.

    Returns:
        1-D float array of the same length with a NaN warm-up head.

    Raises:
        ValueError: if ``window`` is not an integer >= 2.

    Example::

        >>> vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> zed = rolling_z(vals, 3)
        >>> bool(np.isnan(zed[:2]).all())
        True
        >>> round(float(zed[2]), 12)
        1.0
    """
    window = validate_int("window", window, minimum=2)
    values_arr = np.asarray(values, dtype=float)
    # z is scale-invariant, so on extreme magnitudes (subnormal or
    # ~max-float) the rolling squares inside std under/overflow long
    # before the true z does: renormalize to O(1) first.  Normal-path
    # magnitudes are left untouched (bit-compatible rolling math).
    finite_mask = np.isfinite(values_arr)
    if bool(finite_mask.any()):
        scale = float(np.abs(values_arr[finite_mask]).max())
        if 0.0 < scale < 1e-150 or scale > 1e150:
            values_arr = values_arr / scale
    if values_arr.size < window:
        return np.full(values_arr.size, np.nan)
    # two-pass std (mean first, then squared deviations) instead of
    # pandas' rolling one-pass form: the one-pass variance is
    # catastrophically cancellative -- a mathematically constant
    # window returns positive rounding dust at one magnitude and
    # exactly 0 at another, fabricating a 0.0 z at the first scale and
    # NaN at the second (scale invariance broken).  Two-pass
    # deviations are exactly 0 at every scale; per-window computation
    # also makes outputs independent of appended future rows.
    windows = np.lib.stride_tricks.sliding_window_view(values_arr, window)
    means = windows.mean(axis=1)
    stds = np.sqrt(windows.var(axis=1, ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        zed_tail = (values_arr[window - 1 :] - means) / stds
    usable = sane_variance_mask(means, stds)
    zed_tail = np.where(usable, zed_tail, np.nan)
    warmup = np.full(window - 1, np.nan)
    return np.concatenate([warmup, zed_tail])


def ev_of_taken(exec_vals: np.ndarray, taken: np.ndarray) -> float:
    """Mean executable return over taken rows; NaN if nothing taken.

    Example::

        >>> ev_of_taken(
        ...     np.array([0.01, -0.005, 0.006]),
        ...     np.array([True, True, False]),
        ... )
        0.0025
        >>> float(ev_of_taken(np.array([0.01]), np.array([False])))
        nan
    """
    exec_arr = np.asarray(exec_vals, dtype=np.float64)
    taken_arr = np.asarray(taken, dtype=bool)
    if exec_arr.shape != taken_arr.shape:
        raise ValueError(
            f"exec_vals and taken must have equal length, got "
            f"{exec_arr.size} and {taken_arr.size}"
        )
    if int(taken_arr.sum()) == 0:
        return math.nan
    return float(exec_arr[taken_arr].mean())


def pick_threshold(
    proba: np.ndarray,
    exec_vals: np.ndarray,
    grid: tuple[float, ...] | list[float] = _DEFAULT_GRID,
) -> float:
    """Train-fold decision threshold: argmax EV over a fixed grid.

    ``taken = proba > thr`` (strict); ties resolve to the LOWEST
    threshold reaching the max EV, and a grid where no threshold takes
    any trade falls back to the most conservative (highest) value.

    Args:
        proba: train-fold predicted probabilities of the positive class.
        exec_vals: executable per-event returns of the same rows.
        grid: strictly increasing probabilities in (0, 1).

    Returns:
        The selected threshold as a float.

    Raises:
        ValueError: if the grid is empty, not strictly increasing, or
            contains values outside (0, 1).

    Example::

        >>> pick_threshold(
        ...     np.array([0.31, 0.36, 0.66]),
        ...     np.array([0.005, -0.001, 0.010]),
        ...     grid=(0.30, 0.35, 0.65, 0.70),
        ... )
        0.65
    """
    proba_arr = np.asarray(proba)
    exec_arr = np.asarray(exec_vals, dtype=np.float64)
    if proba_arr.shape != exec_arr.shape:
        raise ValueError(
            f"proba and exec_vals must have equal length, got "
            f"{proba_arr.size} and {exec_arr.size}"
        )
    if not isinstance(grid, (tuple, list)) or len(grid) == 0:
        raise ValueError("grid must be a non-empty tuple or list")
    levels = [validate_number("grid entry", level) for level in grid]
    for low, high in pairwise(levels):
        if not low < high:
            raise ValueError(f"grid must be strictly increasing, got {grid}")
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError("grid entries must lie inside (0, 1)")
    best_threshold = levels[-1]
    best_ev = -math.inf
    for level in levels:
        ev = ev_of_taken(exec_vals, np.asarray(proba) > level)
        if math.isfinite(ev) and ev > best_ev:
            best_ev = ev
            best_threshold = float(level)
    return best_threshold


def decluster_count(times_us: np.ndarray, min_gap_us: int = _HOUR_US) -> int:
    """Count independent events by greedy de-clustering of stamps.

    An event is counted iff its stamp is at least ``min_gap_us`` later
    than the last COUNTED event (duplicates collapse); the result is
    the number of clusters, not rows.

    Args:
        times_us: event timestamps in int64 microseconds (any order).
        min_gap_us: minimum separation between counted events.

    Returns:
        The number of independent events.

    Raises:
        ValueError: if ``min_gap_us`` is not an integer >= 1.

    Example::

        >>> hour = 3_600_000_000
        >>> decluster_count(np.array([0, 0, hour - 1, hour, 2 * hour]))
        3
    """
    min_gap_us = validate_int("min_gap_us", min_gap_us)
    raw_stamps = np.asarray(times_us)
    stamps_finite = bool(np.isfinite(raw_stamps.astype(float)).all())
    if raw_stamps.size and not stamps_finite:
        raise ValueError("times_us must contain finite microsecond stamps")
    stamps = np.unique(raw_stamps)
    if stamps.size == 0:
        return 0
    count = 1
    last = int(stamps[0])
    for stamp in stamps[1:]:
        if int(stamp) - last >= min_gap_us:
            count += 1
            last = int(stamp)
    return count


def fold_edges(times_us: np.ndarray, n_blocks: int) -> list[int]:
    """Chronological block edges over unique sorted stamps.

    Edges are placed by an even linspace over the unique stamps, so
    blocks hold (nearly) equal numbers of distinct stamps; the first
    edge is the first stamp and the last edge the last stamp.

    Args:
        times_us: int64 microsecond stamps (any order, duplicates ok).
        n_blocks: number of blocks, integer >= 2.

    Returns:
        ``n_blocks + 1`` strictly increasing int64 edges.

    Raises:
        ValueError: if ``n_blocks`` is not an integer >= 2, fewer than
            ``2 * n_blocks`` unique stamps exist, or the edges come out
            degenerate.

    Example::

        >>> fold_edges(np.arange(12, dtype=np.int64), 3)
        [0, 3, 7, 11]
    """
    n_blocks = validate_int("n_blocks", n_blocks, minimum=2)
    raw_stamps = np.asarray(times_us)
    stamps_finite = bool(np.isfinite(raw_stamps.astype(float)).all())
    if raw_stamps.size and not stamps_finite:
        raise ValueError("times_us must contain finite microsecond stamps")
    uniq = np.unique(raw_stamps)
    if uniq.size < 2 * n_blocks:
        raise ValueError(
            f"need >= {2 * n_blocks} unique stamps, have {uniq.size}"
        )
    positions = np.linspace(0, uniq.size - 1, n_blocks + 1).astype(int)
    edges = [int(uniq[pos]) for pos in positions]
    if any(edges[slot] >= edges[slot + 1] for slot in range(len(edges) - 1)):
        raise ValueError("degenerate fold edges")
    return edges


def assign_folds(times_us: np.ndarray, edges: list[int]) -> np.ndarray:
    """0-based block index per row for half-open ``[e_i, e_{i+1})``.

    Example::

        >>> assign_folds(
        ...     np.array([0, 3, 4, 11]), [0, 3, 7, 11]
        ... ).tolist()
        [0, 1, 1, 2]
    """
    inner = np.asarray(edges[1:-1], dtype=np.int64)
    return np.searchsorted(
        inner, np.asarray(times_us, dtype=np.int64), side="right"
    )


def paired_bootstrap_t_stat(
    exec_a: np.ndarray,
    taken_a: np.ndarray,
    exec_b: np.ndarray,
    taken_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, int]:
    """Paired bootstrap t of EV(a) - EV(b) on the same OOS rows.

    Row indices are resampled jointly; a draw is skipped when either
    arm takes zero trades in it.  Identical streams yield exactly 0.0.

    Args:
        exec_a: executable returns of arm a (all OOS rows).
        taken_a: arm-a trade flags over the same rows.
        exec_b: executable returns of arm b.
        taken_b: arm-b trade flags.
        n_boot: number of bootstrap draws, integer >= 2.
        seed: RNG seed.

    Returns:
        ``(t_stat, n_skipped_draws)``; NaN t-stat when fewer than two
        valid draws survive.

    Raises:
        ValueError: if ``n_boot`` is not an integer >= 2 or ``seed``
            is not an integer >= 0.

    Example::

        >>> exec_a = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        >>> taken = np.array([True, True, True, True, True])
        >>> t_stat, skipped = paired_bootstrap_t_stat(
        ...     exec_a, taken, exec_a, taken, n_boot=50, seed=42
        ... )
        >>> (t_stat, skipped)
        (0.0, 0)
    """
    n_boot = validate_int("n_boot", n_boot, minimum=2)
    seed = validate_int("seed", seed, minimum=0)
    exec_arr_a = np.asarray(exec_a, dtype=np.float64)
    exec_arr_b = np.asarray(exec_b, dtype=np.float64)
    flags_a = np.asarray(taken_a, dtype=bool)
    flags_b = np.asarray(taken_b, dtype=bool)
    rng = np.random.default_rng(seed)
    n_rows = exec_arr_a.size
    diffs: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, n_rows)
        taken_draw_a = flags_a[idx]
        taken_draw_b = flags_b[idx]
        if int(taken_draw_a.sum()) == 0 or int(taken_draw_b.sum()) == 0:
            skipped += 1
            continue
        diffs.append(
            exec_arr_a[idx][taken_draw_a].mean()
            - exec_arr_b[idx][taken_draw_b].mean()
        )
    if len(diffs) < 2:
        return math.nan, skipped
    draw_arr = np.asarray(diffs, dtype=np.float64)
    spread = float(draw_arr.std(ddof=1))
    if spread == 0.0:
        return 0.0, skipped
    t_stat = float(draw_arr.mean() / (spread / math.sqrt(draw_arr.size)))
    return t_stat, skipped


def stream_stats(stream: np.ndarray) -> dict[str, float]:
    """Moment statistics of a per-day EV stream for the DSR gate.

    Example::

        >>> stats = stream_stats(np.array([0.01, -0.02, 0.03, 0.005]))
        >>> round(stats["mean"], 6)
        0.00625
        >>> sorted(stats)
        ['kurt_pearson', 'mean', 'skew', 'sr', 'std']
    """
    values = np.asarray(stream, dtype=np.float64)
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    mean = float(values.mean())
    if (math.isinf(spread) or math.isinf(mean)) and bool(
        np.isfinite(values).all()
    ):
        # squared deviations overflowed while the true moments are
        # representable: rescale to O(1), compute, rescale back
        scale = float(np.abs(values).max())
        centered = values / scale
        spread = scale * float(centered.std(ddof=1))
        mean = scale * float(centered.mean())
    return {
        "sr": mean / spread if spread > 0 else math.nan,
        "skew": float(skew(values)) if values.size >= 3 else math.nan,
        "kurt_pearson": (
            float(kurtosis(values, fisher=False))
            if values.size >= 3
            else math.nan
        ),
        "mean": mean,
        "std": spread,
    }


def deflated_sharpe_value(
    sr: float,
    n_obs: int,
    skew_val: float,
    kurt_val: float,
    var_sr: float,
    *,
    n_trials: int = 5,
) -> float:
    """Deflated Sharpe ratio (CDF form) of a per-trade EV stream.

    The return value is a probability ``Phi(z)`` in ``[0, 1]``; a value
    of exactly ``1.0`` is the saturation cap (``z`` large enough that
    the normal CDF rounds to one), not an anomaly -- for the frozen
    demo worlds with z ~ 9-12 saturation is the expected reading.  Tiny
    values on dead books are honest extreme CDF-tail values (finite,
    representable), not artifacts.

    Bailey & Lopez de Prado (2014); the value returned is the CDF
    ``Phi(z)``.  Thin validated alias: the deflation math lives in
    :func:`atropos.evaluation_metrics.deflated_sharpe_from_moments`
    (the single home shared with the returns-based
    :func:`atropos.evaluation_metrics.deflated_sharpe`); this wrapper
    keeps the EV-stream call signature and enforces the engine's
    stricter ``n_trials >= 2`` gate contract.

    Benchmark ``sr0 = sqrt(var_sr) * ((1-g)*Z(1-1/N) + g*Z(1-1/(N*e)))``
    with ``g`` the Euler-Mascheroni constant; ``DSR = Phi((sr - sr0) *
    sqrt(T-1) / sqrt(1 - skew*sr + (kurt-1)/4 * sr^2))``.  Non-finite
    or degenerate input (a non-finite moment, ``var_sr < 0``,
    ``n_obs < 3``, a degenerate denominator) yields NaN (never a
    fabricated 0 or 1).

    Distinct from :func:`atropos.evaluation_metrics.deflated_sharpe`,
    which ingests per-period RETURN series, annualizes and reports a
    p-value; this alias consumes already-computed moments of a
    per-trade/per-day executable-value stream and returns the raw CDF
    value.

    Args:
        sr: (per-period) Sharpe ratio of the stream.
        n_obs: number of observations in the stream.
        skew_val: sample skewness of the stream.
        kurt_val: Pearson (non-excess) kurtosis of the stream.
        var_sr: variance of SRs across the sibling trial streams.
        n_trials: declared number of trials, integer >= 2.

    Returns:
        The deflated Sharpe ratio in [0, 1], or NaN.

    Raises:
        ValueError: if ``n_obs`` is not an integer >= 1 or ``n_trials``
            is not an integer >= 2.

    Example::

        >>> round(deflated_sharpe_value(0.0, 100, 0.0, 3.0, 0.0), 6)
        0.5
    """
    n_obs = validate_int("n_obs", n_obs)
    n_trials = validate_int("n_trials", n_trials, minimum=2)
    return deflated_sharpe_from_moments(
        sr, n_obs, skew_val, kurt_val, var_sr, n_trials=n_trials
    )


def spearman_median(
    vectors: list[np.ndarray],
) -> tuple[float, int, int]:
    """Median pairwise Spearman correlation of ranking vectors.

    Args:
        vectors: list of equal-length importance/ranking vectors.

    Returns:
        ``(median_rho, n_valid_pairs, n_nan_pairs)``; the median is NaN
        when no pair produces a finite correlation (e.g. constant
        vectors).

    Example::

        >>> median, n_pairs, n_nan = spearman_median(
        ...     [
        ...         np.array([1.0, 2.0, 3.0, 4.0]),
        ...         np.array([4.0, 3.0, 2.0, 1.0]),
        ...     ]
        ... )
        >>> (median, n_pairs, n_nan)
        (-1.0, 1, 0)
    """
    n_vec = len(vectors)
    rhos: list[float] = []
    nan_pairs = 0
    for first in range(n_vec):
        for second in range(first + 1, n_vec):
            rho = spearmanr(vectors[first], vectors[second]).statistic
            if rho is None or not math.isfinite(float(rho)):
                nan_pairs += 1
            else:
                rhos.append(float(rho))
    if not rhos:
        return math.nan, 0, nan_pairs
    return float(np.median(rhos)), len(rhos), nan_pairs


def permutation_importance_vec(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Per-feature OOS log-loss degradation under column shuffling.

    Positive values mark informative features (shuffling them hurts);
    near-zero values mark noise columns.  Deterministic for a fixed
    ``seed``.

    Args:
        model: fitted estimator exposing ``predict_proba``.
        x_test: 2-D test feature matrix.
        y_test: 1-D 0/1 test labels.
        seed: RNG seed for the column shuffles.

    Returns:
        1-D array of per-feature importance (degradation) values.

    Raises:
        ValueError: if ``seed`` is not an integer >= 0.

    Example::

        >>> from sklearn.tree import DecisionTreeClassifier
        >>> feats = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> tree = DecisionTreeClassifier(random_state=0).fit(
        ...     feats, labels
        ... )
        >>> float(permutation_importance_vec(tree, feats, labels, 0)[0]) > 0
        True
    """
    seed = validate_int("seed", seed, minimum=0)
    base_proba = np.asarray(model.predict_proba(x_test))[:, 1]
    base_loss = float(log_loss(y_test, base_proba, labels=[0, 1]))
    rng = np.random.default_rng(seed)
    importances: list[float] = []
    for column in range(x_test.shape[1]):
        shuffled = x_test.copy()
        column_values = shuffled[:, column].copy()
        rng.shuffle(column_values)
        shuffled[:, column] = column_values
        proba = np.asarray(model.predict_proba(shuffled))[:, 1]
        importances.append(
            float(log_loss(y_test, proba, labels=[0, 1])) - base_loss
        )
    return np.asarray(importances, dtype=np.float64)


# ---------------------------------------------------------------------------
# model factories and the open registry
# ---------------------------------------------------------------------------
def make_extratrees(
    n_estimators: int = 400,
    max_depth: int = 6,
    min_samples_leaf: int = 20,
    seed: int = 42,
) -> Any:
    """Extremely randomized tree ensemble in a Pipeline (deterministic).

    The decorrelation axis of the registry next to :func:`make_rf`:
    the same bagged-forest shape but with RANDOM split thresholds, so
    tree errors decorrelate without the optimized-split bias.  Fit is
    trivially parallel but ``n_jobs=1`` pins single-threaded
    execution so a fixed seed is bit-reproducible across runs.

    Args:
        n_estimators: number of trees, integer >= 1.
        max_depth: max tree depth, integer >= 1.
        min_samples_leaf: minimum leaf size, integer >= 1.
        seed: random seed, integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if any parameter fails its documented contract.

    Example::

        >>> model = make_extratrees(n_estimators=50, max_depth=3,
        ...                         min_samples_leaf=1)
        >>> model.fit(
        ...     np.array([[0.0], [0.1], [0.9], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.05], [0.95]])).tolist()
        [0, 1]
    """
    n_estimators = validate_int("n_estimators", n_estimators)
    max_depth = validate_int("max_depth", max_depth)
    min_samples_leaf = validate_int("min_samples_leaf", min_samples_leaf)
    seed = validate_int("seed", seed, minimum=0)
    return Pipeline(
        [
            (
                "extratrees",
                _ETClass(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=1,
                ),
            )
        ]
    )


def make_gbm(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
    seed: int = 42,
) -> Any:
    """Gradient-boosted trees classifier in a Pipeline.

    Deterministic by construction: single-threaded sklearn GBM with a
    fixed seed; every hyperparameter comes from the caller's config.

    Args:
        n_estimators: number of boosting stages, integer >= 1.
        learning_rate: shrinkage learning rate, finite > 0.
        max_depth: max tree depth, integer >= 1.
        subsample: row subsampling fraction in (0, 1].
        seed: random seed, integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if any parameter fails its documented contract.

    Example::

        >>> model = make_gbm(n_estimators=10)
        >>> model.fit(
        ...     np.array([[0.0], [0.1], [0.9], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.05], [0.95]])).tolist()
        [0, 1]
    """
    n_estimators = validate_int("n_estimators", n_estimators)
    max_depth = validate_int("max_depth", max_depth)
    seed = validate_int("seed", seed, minimum=0)
    learning_rate = validate_number("learning_rate", learning_rate)
    subsample = validate_number("subsample", subsample)
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not 0.0 < subsample <= 1.0:
        raise ValueError(f"subsample must lie in (0, 1], got {subsample}")
    return Pipeline(
        [
            (
                "gbm",
                GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    random_state=seed,
                ),
            )
        ]
    )


def make_hist_gbm(
    max_iter: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    seed: int = 42,
) -> Any:
    """Histogram-based gradient boosting in a Pipeline (deterministic).

    The histogram axis next to the exact-split :func:`make_gbm`:
    features are pre-binned LightGBM-style, so each split costs O(bins)
    instead of O(sorted rows).  The estimator exposes no ``n_jobs``
    (OpenMP internally) yet fits deterministically for a fixed
    ``random_state`` regardless of thread count.  ``early_stopping``
    is pinned ``False``: sklearn auto-enables a seeded validation
    split beyond 10k rows, and an explicit off keeps the iteration
    schedule identical at every dataset size.  The estimator is
    NaN-tolerant (missing values get a dedicated bin), but the engine
    never supplies any: :func:`run_ml_kill_protocol` rejects non-finite
    features up front, so that tolerance is dormant headroom for
    other callers of the factory.

    Args:
        max_iter: number of boosting iterations, integer >= 1.
        learning_rate: shrinkage learning rate, finite > 0.
        max_depth: max tree depth, integer >= 1.
        seed: random seed, integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if any parameter fails its documented contract.

    Example::

        >>> rng = np.random.default_rng(0)
        >>> feats = rng.standard_normal((60, 2))
        >>> labels = (feats[:, 0] > 0.0).astype(np.int64)
        >>> model = make_hist_gbm(max_iter=20).fit(feats, labels)
        >>> bool((model.predict(feats) == labels).mean() > 0.8)
        True
    """
    max_iter = validate_int("max_iter", max_iter)
    max_depth = validate_int("max_depth", max_depth)
    seed = validate_int("seed", seed, minimum=0)
    learning_rate = validate_number("learning_rate", learning_rate)
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    return Pipeline(
        [
            (
                "hist_gbm",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    early_stopping=False,
                    random_state=seed,
                ),
            )
        ]
    )


def make_knn(
    n_neighbors: int = 50,
    weights: str = "distance",
    seed: int = 42,
) -> Any:
    """Distance-weighted k-nearest-neighbour classifier in a Pipeline.

    The local/nonparametric axis of the registry.  Determinism is
    trivial (no training randomness; ``n_jobs=1`` pins the search),
    so ``seed`` exists only for the uniform factory contract -- the
    engine injects ``model_seed`` into every factory -- and is
    validated then unused.

    CONTRACT: ``n_neighbors`` must stay below the SMALLEST training
    fold the engine will ever fit.  sklearn accepts the construction
    and the fit and rejects the first PREDICT with ``ValueError``
    when ``n_neighbors`` exceeds the training rows -- with
    ``k_blocks`` expanding blocks the first fold holds roughly
    ``n_rows / k_blocks`` rows minus the purge and embargo flanks, so
    bound the default accordingly (e.g. pass
    ``n_neighbors=min(expected_fold, 50)`` for thin datasets).

    NOTE: like the linear battery, kNN cannot cherry-pick a
    high-confidence sub-book -- its distance-weighted estimates stay
    compressed near the base rate, so the EV-argmax threshold
    saturates at the grid floor -- and on a mixed linear+nonlinear
    world its broad book roughly TIES the logit tail book.  The
    planted-nonlinearity contract test documents this and skips the
    comparison for ``knn``; determinism, probability-validity and the
    full engine path still cover it end to end.

    Args:
        n_neighbors: number of neighbours, integer >= 1 and < the
            smallest training fold at fit time.
        weights: ``"distance"`` (default) or ``"uniform"``.
        seed: random seed, integer >= 0 (validated, unused -- kNN
            has no stochastic component).

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if ``n_neighbors``/``seed`` fail their contract
            or ``weights`` is not one of the two supported schemes.

    Example::

        >>> model = make_knn(n_neighbors=2)
        >>> model.fit(
        ...     np.array([[0.0], [0.1], [0.9], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.05], [0.95]])).tolist()
        [0, 1]
    """
    n_neighbors = validate_int("n_neighbors", n_neighbors)
    seed = validate_int("seed", seed, minimum=0)
    if weights not in {"uniform", "distance"}:
        raise ValueError(
            f"weights must be 'uniform' or 'distance', got {weights!r}"
        )
    return Pipeline(
        [
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    n_jobs=1,
                ),
            )
        ]
    )


def make_logit(max_iter: int = 1000, seed: int = 42) -> Any:
    """L2 logistic classifier in a standardizing Pipeline.

    The linear comparator of the engine (baseline arm and battery
    arm); lbfgs with a fixed seed is deterministic.  The L2 penalty is
    the sklearn default (left implicit: ``penalty`` is deprecated
    since sklearn 1.8).

    Args:
        max_iter: lbfgs iteration cap, integer >= 1.
        seed: random seed, integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if ``max_iter`` or ``seed`` fails its contract.

    Example::

        >>> model = make_logit()
        >>> model.fit(
        ...     np.array([[-1.0], [-0.5], [0.5], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.9], [-0.9]])).tolist()
        [1, 0]
    """
    max_iter = validate_int("max_iter", max_iter)
    seed = validate_int("seed", seed, minimum=0)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_mlp(
    hidden_layer_sizes: tuple[int, ...] | list[int] = (32, 16),
    max_iter: int = 500,
    early_stopping: bool = False,
    seed: int = 42,
) -> Any:
    """Two-layer perceptron classifier behind a standardizing Pipeline.

    The neural axis of the registry: ``StandardScaler`` (adam needs
    comparably scaled inputs) feeding an adam-trained
    ``MLPClassifier``.  Determinism is empirical fact, not just
    construction: two fresh fits on identical data with the same
    seed produce bit-identical predictions (pinned by the contract
    tests), including in the non-converged regime.  The default
    ``max_iter=500`` converges (no sklearn ``ConvergenceWarning``)
    on engine-scale datasets of a few hundred rows or more; on tiny
    or XOR-like samples the optimizer may stop at the budget, which
    is deterministic, observable via ``loss_curve_`` and only noisy,
    not wrong.  ``early_stopping=False`` keeps every training row in
    the optimizer and the iteration schedule independent of dataset
    size (the auto mode would hold out a seeded validation split
    beyond 10k rows).

    Args:
        hidden_layer_sizes: widths of the hidden layers, non-empty
            sequence of integers >= 1 (a list from JSON-style configs
            is accepted and normalized to a tuple).
        max_iter: adam iteration budget, integer >= 1.
        early_stopping: must stay a bool; keep ``False`` for the
            documented determinism semantics.
        seed: random seed (weight init + shuffling), integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if any parameter fails its documented contract.

    Example::

        >>> rng = np.random.default_rng(0)
        >>> feats = rng.standard_normal((200, 2))
        >>> labels = (feats[:, 0] > 0.0).astype(np.int64)
        >>> model = make_mlp(max_iter=500).fit(feats, labels)
        >>> bool((model.predict(feats) == labels).mean() > 0.8)
        True
    """
    widths = tuple(
        validate_int("hidden_layer_sizes entry", width)
        for width in hidden_layer_sizes
    )
    if not widths:
        raise ValueError("hidden_layer_sizes must be a non-empty sequence")
    max_iter = validate_int("max_iter", max_iter)
    seed = validate_int("seed", seed, minimum=0)
    if not isinstance(early_stopping, bool):
        raise ValueError(
            f"early_stopping must be a bool, got {early_stopping!r}"
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=widths,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_qda(reg_param: float = 0.5, seed: int = 42) -> Any:
    """Quadratic discriminant classifier in a Pipeline (generative).

    The generative axis of the registry -- the only bundled entry
    whose QUADRATIC decision surface sees cross-feature interactions
    with a closed-form fit and no search at all.  ``reg_param``
    shrinks each class covariance towards its scaled diagonal:
    without it, collinear or duplicated features make the covariance
    estimate singular and ``predict_proba`` saturates to exactly 0/1
    (the regularized default keeps probabilities strictly inside the
    open interval -- pinned by a dedicated unit test).  The fit is
    deterministic by construction (no stochastic component), so
    ``seed`` exists only for the uniform factory contract -- the
    engine injects ``model_seed`` into every factory -- and is
    validated then unused.

    Args:
        reg_param: covariance shrinkage strength, finite >= 0
            (sklearn's accepted range; 0.5 is deliberately strong).
        seed: random seed, integer >= 0 (validated, unused -- the
            fit is deterministic by construction).

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if ``reg_param`` is not a finite number >= 0 or
            ``seed`` fails its contract.

    Example::

        >>> model = make_qda()
        >>> model.fit(
        ...     np.array([[0.0], [0.1], [0.9], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.05], [0.95]])).tolist()
        [0, 1]
    """
    reg_param = validate_number("reg_param", reg_param)
    if reg_param < 0.0:
        raise ValueError(f"reg_param must be >= 0, got {reg_param}")
    seed = validate_int("seed", seed, minimum=0)
    return Pipeline(
        [("qda", QuadraticDiscriminantAnalysis(reg_param=reg_param))]
    )


def make_rf(
    n_estimators: int = 400,
    max_depth: int = 5,
    min_samples_leaf: int = 50,
    seed: int = 42,
) -> Any:
    """Random-forest classifier in a Pipeline (deterministic).

    ``n_jobs=1`` pins single-threaded execution: forest building with a
    fixed seed is then bit-reproducible across runs and platforms.

    Args:
        n_estimators: number of trees, integer >= 1.
        max_depth: max tree depth, integer >= 1.
        min_samples_leaf: minimum leaf size, integer >= 1.
        seed: random seed, integer >= 0.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.

    Raises:
        ValueError: if any parameter fails its documented contract.

    Example::

        >>> model = make_rf(n_estimators=10, max_depth=3,
        ...                 min_samples_leaf=1)
        >>> model.fit(
        ...     np.array([[0.0], [0.1], [0.9], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... ).predict(np.array([[0.05], [0.95]])).tolist()
        [0, 1]
    """
    n_estimators = validate_int("n_estimators", n_estimators)
    max_depth = validate_int("max_depth", max_depth)
    min_samples_leaf = validate_int("min_samples_leaf", min_samples_leaf)
    seed = validate_int("seed", seed, minimum=0)
    return Pipeline(
        [
            (
                "rf",
                _RFClass(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=1,
                ),
            )
        ]
    )


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "extratrees": make_extratrees,
    "gbm": make_gbm,
    "hist_gbm": make_hist_gbm,
    "knn": make_knn,
    "logit": make_logit,
    "mlp": make_mlp,
    "qda": make_qda,
    "rf": make_rf,
}
"""Open model registry: name -> factory.

Factories accept keyword arguments from
``DEFAULT_TRAINING_CONFIG["model_params"][name]`` (validated inside
each factory) and return an unfitted sklearn-style estimator with
``fit`` / ``predict_proba``.  Registering an entry
(``MODEL_REGISTRY["x"] = factory``) makes it immediately available to
``run_ml_kill_protocol`` via ``config["model"]`` / ``config["models"]``
with no engine changes.

The eight bundled entries cover complementary bias axes -- linear
(``logit``), exact-split boosting (``gbm``), histogram boosting
(``hist_gbm``), optimized-split bagging (``rf``), random-split
decorrelation (``extratrees``), neural (``mlp``), local neighbourhood
(``knn``) and generative quadratic (``qda``) -- so a comparison run
spans genuinely different ways to be wrong, not eight tuning variants
of one model.
"""


# ---------------------------------------------------------------------------
# config and input validation
# ---------------------------------------------------------------------------
def _merge_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user config over :data:`DEFAULT_TRAINING_CONFIG`.

    Dicts merge key-wise at every level, so an override like
    ``{"model_params": {"gbm": {"n_estimators": 60}}}`` keeps every
    other default of the ``gbm`` parameter block (shallow copying
    here silently swapped sklearn defaults for the frozen ones --
    the exact "another model than documented" trap).
    """
    cfg: dict[str, Any] = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in DEFAULT_TRAINING_CONFIG.items()
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                target = cfg[key]
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and isinstance(
                        target.get(sub_key), dict
                    ):
                        target[sub_key] = {**target[sub_key], **sub_value}
                    else:
                        target[sub_key] = sub_value
            else:
                cfg[key] = value
    return cfg


def _validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the merged config in place; return it.

    Raises:
        ValueError: on any non-integer, non-number, non-boolean or
            structurally invalid parameter, naming the offending key.
    """
    for key, minimum in _INT_PARAMS.items():
        cfg[key] = validate_int(key, cfg[key], minimum=minimum)
    for key, floor in _FLOAT_PARAMS.items():
        cfg[key] = validate_number(key, cfg[key])
        if cfg[key] < floor:
            raise ValueError(f"{key} must be >= {floor}, got {cfg[key]}")
    prereg_hash = cfg["prereg_hash"]
    if prereg_hash is not None and (
        not isinstance(prereg_hash, str) or not prereg_hash
    ):
        raise ValueError(
            "prereg_hash must be None or a non-empty string, got "
            f"{prereg_hash!r}"
        )
    _validate_grid(cfg)
    _validate_checks(cfg)
    _validate_model_selection(cfg)
    _validate_model_params(cfg)
    return cfg


def _validate_grid(cfg: dict[str, Any]) -> None:
    """Validate the threshold grid; store it as a float tuple."""
    grid = cfg["threshold_grid"]
    if not isinstance(grid, (tuple, list)) or len(grid) == 0:
        raise ValueError(
            "threshold_grid must be a non-empty tuple or list of floats"
        )
    levels = [
        validate_number("threshold_grid entry", level) for level in grid
    ]
    for low, high in pairwise(levels):
        if not low < high:
            raise ValueError(
                f"threshold_grid must be strictly increasing, got {grid}"
            )
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError("threshold_grid entries must lie inside (0, 1)")
    cfg["threshold_grid"] = tuple(levels)


def _validate_checks(cfg: dict[str, Any]) -> None:
    """Validate the evaluation-gate toggles and thresholds."""
    checks = cfg["checks"]
    if not isinstance(checks, dict):
        raise ValueError("checks must be a dict of gate settings")
    for key in _CHECK_BOOLS:
        if not isinstance(checks.get(key), bool):
            raise ValueError(
                f"checks.{key} must be a bool, got {checks.get(key)!r}"
            )
    for key in _CHECK_NUMBERS:
        if key not in checks:
            raise ValueError(f"checks.{key} is required")
        checks[key] = validate_number(f"checks.{key}", checks[key])
    # range checks: an out-of-domain threshold silently turns a gate
    # into always-pass (fail-open) or always-kill (fail-closed cap)
    if not 0.0 <= checks["dsr_min"] <= 1.0:
        raise ValueError(
            f"checks.dsr_min must lie in [0, 1], got {checks['dsr_min']}"
        )
    if not -1.0 <= checks["importance_rho_min"] <= 1.0:
        raise ValueError(
            "checks.importance_rho_min must lie in [-1, 1], got "
            f"{checks['importance_rho_min']}"
        )
    if checks["incremental_t"] < 0.0:
        raise ValueError(
            "checks.incremental_t must be >= 0, got "
            f"{checks['incremental_t']}"
        )
    if not 0.05 <= checks["robustness_shift"] <= 0.5:
        raise ValueError(
            "checks.robustness_shift must lie in [0.05, 0.5], got "
            f"{checks['robustness_shift']}"
        )


def _validate_model_selection(cfg: dict[str, Any]) -> None:
    """Validate the model / models selection keys against the registry."""
    if not isinstance(cfg["model"], str):
        raise ValueError(
            f"model must be a registry name string, got {cfg['model']!r}"
        )
    models = cfg["models"]
    if models is None:
        return
    if (
        not isinstance(models, (list, tuple))
        or len(models) == 0
        or not all(isinstance(name, str) for name in models)
    ):
        raise ValueError(
            "models must be None or a non-empty list of registry names"
        )


def _validate_model_params(cfg: dict[str, Any]) -> None:
    """Validate the model_params structure (values reach factories)."""
    params = cfg["model_params"]
    if not isinstance(params, dict) or not all(
        isinstance(entry, dict) for entry in params.values()
    ):
        raise ValueError(
            "model_params must map model names to dicts of kwargs"
        )


def _as_int_labels(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce to int64 labels, rejecting non-exact or non-binary data."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={arr.ndim}")
    try:
        as_int = arr.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain integers, got {arr.dtype}"
        ) from error
    if not np.array_equal(arr, as_int):
        raise ValueError(f"{name} must contain exact integers")
    if not np.isin(as_int, (0, 1)).all():
        found = sorted(np.unique(as_int).tolist())
        raise ValueError(f"{name} must be binary 0/1, got {found}")
    return as_int


def _as_times(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce to int64 microsecond stamps; enforce monotonicity."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={arr.ndim}")
    try:
        as_int = arr.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain int64 microseconds, got {arr.dtype}"
        ) from error
    if not np.array_equal(arr, as_int):
        raise ValueError(f"{name} must contain exact integers")
    if as_int.size > 1 and np.any(np.diff(as_int) < 0):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    return as_int


def _validate_inputs(
    features: Any,
    target: Any,
    exec_returns: Any,
    times_us: Any,
    battery_controls: list[str] | None,
) -> dict[str, Any]:
    """Validate the dataset contract and build the run-data bundle.

    Raises:
        ValueError: when ``features`` is not a finite numeric
            DataFrame, the 1-D arrays disagree in length, ``target``
            is not binary integers, ``exec_returns`` is not finite
            floats, ``times_us`` is not monotone int64 microseconds,
            or a battery-control column is missing.
    """
    if not isinstance(features, pd.DataFrame):
        raise ValueError(
            "features must be a pandas DataFrame of float columns"
        )
    if features.shape[1] < 1 or features.shape[0] < 2:
        raise ValueError("features must have >= 1 column and >= 2 rows")
    labels = _as_int_labels("target", np.asarray(target))
    exec_vals = np.asarray(exec_returns, dtype=np.float64)
    stamps = _as_times("times_us", np.asarray(times_us))
    if exec_vals.ndim != 1 or not np.isfinite(exec_vals).all():
        raise ValueError("exec_returns must be 1-D finite floats")
    n_rows = features.shape[0]
    if labels.size != n_rows or exec_vals.size != n_rows:
        raise ValueError(
            f"target/exec_returns/times_us must match features rows "
            f"({n_rows}), got {labels.size}/{exec_vals.size}/{stamps.size}"
        )
    if stamps.size != n_rows:
        raise ValueError(
            f"times_us must match features rows ({n_rows}), got {stamps.size}"
        )
    controls = list(dict.fromkeys(battery_controls or []))
    missing = [name for name in controls if name not in features.columns]
    if missing:
        raise ValueError(f"battery_controls not found in features: {missing}")
    model_columns = [
        column for column in features.columns if column not in controls
    ]
    if not model_columns:
        raise ValueError("features must keep at least one non-control column")
    used_columns = model_columns + controls
    matrix = features.loc[:, used_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("feature columns must be finite floats (no NaN/inf)")
    model_positions = list(range(len(model_columns)))
    return {
        "x_matrix": matrix,
        "model_positions": model_positions,
        "battery_positions": list(range(len(used_columns))),
        "labels": labels,
        "exec_vals": exec_vals,
        "times_us": stamps,
        "feature_names": used_columns,
        "battery_available": bool(controls),
    }


def _resolve_model_specs(
    cfg: dict[str, Any],
) -> list[tuple[str, ModelFactory, dict[str, Any]]]:
    """Resolve the config's model selection to registry specs.

    Raises:
        ValueError: naming the unknown model and the available ones.
    """
    names = cfg["models"] if cfg["models"] is not None else [cfg["model"]]
    unique_names: list[str] = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    specs: list[tuple[str, ModelFactory, dict[str, Any]]] = []
    for name in unique_names:
        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown model {name!r}: available {sorted(MODEL_REGISTRY)}"
            )
        params = dict(cfg["model_params"].get(name, {}))
        specs.append((name, MODEL_REGISTRY[name], params))
    return specs


# ---------------------------------------------------------------------------
# walk-forward orchestration
# ---------------------------------------------------------------------------
def _fit_predict(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``model`` and return (train, test) positive-class probas."""
    model.fit(x_train, y_train)
    train_proba = np.asarray(model.predict_proba(x_train))[:, 1]
    test_proba = np.asarray(model.predict_proba(x_test))[:, 1]
    return train_proba, test_proba


def _fit_fold_variants(
    fold_data: dict[str, Any],
    model_specs: list[tuple[str, ModelFactory, dict[str, Any]]],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Fit every model + linear comparators for one CV fold.

    Returns a dict ``name -> {"proba_train", "proba_test",
    "threshold", "fitted"}`` covering the registry models, the shared
    logit ``baseline`` and (when battery columns exist) the logit
    ``battery`` on features + controls.
    """
    x_train = fold_data["x_train"]
    x_test = fold_data["x_test"]
    x_bat_train = fold_data["x_bat_train"]
    x_bat_test = fold_data["x_bat_test"]
    y_train = fold_data["y_train"]
    exec_train = fold_data["exec_train"]
    grid = cfg["threshold_grid"]
    variants: dict[str, dict[str, Any]] = {}
    for name, factory, params in model_specs:
        model = factory(**_with_seed(params, cfg))
        proba_train, proba_test = _fit_predict(
            model, x_train, y_train, x_test
        )
        variants[name] = {
            "proba_train": proba_train,
            "proba_test": proba_test,
            "threshold": pick_threshold(proba_train, exec_train, grid),
            "fitted": model,
        }
    logit_params = dict(cfg["model_params"].get("logit", {}))
    for arm, train_matrix, test_matrix in (
        (_BASELINE_NAME, x_train, x_test),
        (_BATTERY_NAME, x_bat_train, x_bat_test),
    ):
        if arm == _BATTERY_NAME and not fold_data["battery_available"]:
            continue
        logit = make_logit(**_with_seed(logit_params, cfg))
        proba_train, proba_test = _fit_predict(
            logit, train_matrix, y_train, test_matrix
        )
        variants[arm] = {
            "proba_train": proba_train,
            "proba_test": proba_test,
            "threshold": pick_threshold(proba_train, exec_train, grid),
            "fitted": logit,
        }
    return variants


def _with_seed(params: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy factory params, defaulting ``seed`` to the config's seed."""
    merged = dict(params)
    merged.setdefault("seed", cfg["model_seed"])
    return merged


def _run_cv_folds(
    run_data: dict[str, Any],
    cfg: dict[str, Any],
    model_specs: list[tuple[str, ModelFactory, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the expanding walk-forward CV over chronological blocks.

    Returns ``(fold_rows, oos)`` where ``fold_rows`` holds one record
    per test fold (rows/edges plus per-variant threshold/taken/EV) and
    ``oos`` holds the merged out-of-sample bundle: per-variant
    probabilities, per-variant per-fold thresholds, exec values,
    stamps, labels and per-model per-fold permutation importances
    (computed on the fly from the still-fitted fold models).
    """
    stamps_all = run_data["times_us"]
    labels = run_data["labels"]
    exec_vals = run_data["exec_vals"]
    x_matrix = run_data["x_matrix"]
    model_pos = run_data["model_positions"]
    battery_pos = run_data["battery_positions"]
    n_blocks = cfg["k_blocks"]
    edges = fold_edges(stamps_all, n_blocks)
    folds = assign_folds(stamps_all, edges)
    fold_rows: list[dict[str, Any]] = []
    proba_chunks: dict[str, list[np.ndarray]] = {}
    importances: dict[str, list[np.ndarray]] = {
        name: [] for name, _, _ in model_specs
    }
    oos_exec: list[np.ndarray] = []
    oos_times: list[np.ndarray] = []
    oos_labels: list[np.ndarray] = []
    fold_sizes: list[int] = []
    for fold_no in range(1, n_blocks):
        train_rows = np.flatnonzero(folds < fold_no)
        test_rows = np.flatnonzero(folds == fold_no)
        keep = purge_embargo_mask(
            stamps_all[train_rows],
            edges[fold_no],
            horizon_h=cfg["horizon_h"],
            purge_h=cfg["purge_h"],
            embargo_h=cfg["embargo_h"],
        )
        train_rows = train_rows[keep]
        if train_rows.size == 0:
            raise ValueError(
                f"fold {fold_no}: no training rows survive the "
                f"purge ({cfg['purge_h']}h) + embargo "
                f"({cfg['embargo_h']}h) before the test block -- "
                "reduce k_blocks or the purge/embargo hours"
            )
        fold_data = {
            "x_train": x_matrix[np.ix_(train_rows, model_pos)],
            "x_test": x_matrix[np.ix_(test_rows, model_pos)],
            "x_bat_train": x_matrix[np.ix_(train_rows, battery_pos)],
            "x_bat_test": x_matrix[np.ix_(test_rows, battery_pos)],
            "y_train": labels[train_rows],
            "exec_train": exec_vals[train_rows],
            "battery_available": run_data["battery_available"],
        }
        variants = _fit_fold_variants(fold_data, model_specs, cfg)
        fold_record: dict[str, Any] = {
            "fold": fold_no,
            "train_rows": int(train_rows.size),
            "test_rows": int(test_rows.size),
            "test_start": int(edges[fold_no]),
            "n_dropped_purge_embargo": int((~keep).sum()),
            "models": {},
        }
        x_test_model = fold_data["x_test"]
        y_test = labels[test_rows]
        exec_test = exec_vals[test_rows]
        for name, variant in variants.items():
            taken = variant["proba_test"] > variant["threshold"]
            ev = ev_of_taken(exec_test, taken)
            fold_record["models"][name] = {
                "threshold": float(variant["threshold"]),
                "taken": int(taken.sum()),
                "ev": ev if math.isfinite(ev) else None,
            }
            proba_chunks.setdefault(name, []).append(
                np.asarray(variant["proba_test"])
            )
        for model_pos_index, (name, _, _) in enumerate(model_specs):
            seed = cfg["importance_seed"] + 100 * fold_no + model_pos_index
            importances[name].append(
                permutation_importance_vec(
                    variants[name]["fitted"],
                    x_test_model,
                    y_test,
                    seed,
                )
            )
        fold_rows.append(fold_record)
        fold_sizes.append(int(test_rows.size))
        oos_exec.append(exec_test)
        oos_times.append(stamps_all[test_rows])
        oos_labels.append(y_test)
    oos = {
        "exec": np.concatenate(oos_exec),
        "times": np.concatenate(oos_times),
        "labels": np.concatenate(oos_labels),
        "proba": {
            name: np.concatenate(chunks)
            for name, chunks in proba_chunks.items()
        },
        "thresholds": {
            name: [row["models"][name]["threshold"] for row in fold_rows]
            for name in proba_chunks
        },
        "fold_sizes": fold_sizes,
        "importances": importances,
    }
    return fold_rows, oos


def _row_thresholds(oos: dict[str, Any], name: str) -> np.ndarray:
    """Per-row fold-local decision thresholds in merged-OOS order.

    Each test fold's rows carry the threshold that fold's train block
    picked (the fold-local discipline the engine evaluates gates on).
    """
    return np.concatenate(
        [
            np.full(size, thr, dtype=float)
            for size, thr in zip(
                oos["fold_sizes"], oos["thresholds"][name], strict=True
            )
        ]
    )


def _taken_flags(oos: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-variant merged-OOS taken flags (fold-local thresholds)."""
    flags: dict[str, np.ndarray] = {}
    for name, proba in oos["proba"].items():
        flags[name] = np.asarray(proba) > _row_thresholds(oos, name)
    return flags


def _daily_ev_stream(
    times_us: np.ndarray,
    exec_vals: np.ndarray,
    taken: np.ndarray,
    day_lo: int,
    day_hi: int,
) -> np.ndarray:
    """Daily executable-value stream over a fixed epoch-day calendar.

    Every UTC day in ``[day_lo, day_hi]`` gets the sum of executable
    returns of taken trades that day (zero when nothing was taken), so
    quiet days count as zero -- not missing -- in the DSR moments.
    """
    day_index = np.asarray(times_us, dtype=np.int64) // _DAY_US
    stream = np.zeros(day_hi - day_lo + 1, dtype=np.float64)
    taken_arr = np.asarray(taken, dtype=bool)
    if taken_arr.any():
        np.add.at(
            stream,
            day_index[taken_arr] - day_lo,
            np.asarray(exec_vals, dtype=np.float64)[taken_arr],
        )
    return stream


def _stream_family(
    oos: dict[str, Any], taken: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Daily streams + moments for every variant and the DSR inputs.

    ``var_sr`` is the sample variance of the finite per-day Sharpe
    ratios across the run's family of sibling streams (all evaluated
    models plus the linear comparators) -- the multiple-testing
    dispersion the deflated Sharpe ratio discounts for.
    """
    day_lo = int(oos["times"].min() // _DAY_US)
    day_hi = int(oos["times"].max() // _DAY_US)
    stats: dict[str, dict[str, float]] = {}
    for name, flags in taken.items():
        stream = _daily_ev_stream(
            oos["times"], oos["exec"], flags, day_lo, day_hi
        )
        stats[name] = stream_stats(stream)
    srs = np.asarray(
        [entry["sr"] for entry in stats.values()], dtype=np.float64
    )
    finite = srs[np.isfinite(srs)]
    var_sr = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    return {
        "n_days": int(day_hi - day_lo + 1),
        "var_sr": var_sr,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# evaluation gates
# ---------------------------------------------------------------------------
def _gate_entry(
    check: str,
    enabled: bool,
    triggered: bool,
    inconclusive: bool,
    reason: str | None,
) -> dict[str, Any]:
    """One uniform gate-evaluation record."""
    return {
        "check": check,
        "enabled": enabled,
        "triggered": triggered,
        "inconclusive": inconclusive,
        "reason": reason,
    }


def _gate_incremental(
    name: str,
    oos: dict[str, Any],
    taken: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Incremental gate: model must beat the linear battery (t > 2)."""
    checks = cfg["checks"]
    battery_on = _BATTERY_NAME in taken
    detail: dict[str, Any] = {
        "t_vs_battery": None,
        "t_vs_baseline": None,
        "skipped_draws": None,
    }
    if not checks["incremental"]:
        return (
            _gate_entry("incremental", False, False, False, None),
            detail,
        )
    t_base, skipped_base = paired_bootstrap_t_stat(
        oos["exec"],
        taken[name],
        oos["exec"],
        taken[_BASELINE_NAME],
        n_boot=cfg["bootstrap_b"],
        seed=cfg["bootstrap_seed"],
    )
    detail["t_vs_baseline"] = t_base
    detail["skipped_draws"] = skipped_base
    if not battery_on:
        reason = (
            "battery controls not supplied: incremental edge over a "
            "linear battery is not decidable in this run"
        )
        return (
            _gate_entry("incremental", True, False, True, reason),
            detail,
        )
    t_battery, skipped = paired_bootstrap_t_stat(
        oos["exec"],
        taken[name],
        oos["exec"],
        taken[_BATTERY_NAME],
        n_boot=cfg["bootstrap_b"],
        seed=cfg["bootstrap_seed"],
    )
    detail["t_vs_battery"] = t_battery
    detail["skipped_draws"] = skipped
    if not math.isfinite(t_battery) or (t_battery <= checks["incremental_t"]):
        reason = (
            f"paired bootstrap t vs linear battery {t_battery:.3f} "
            f"<= {checks['incremental_t']}: no incremental edge"
        )
        return (
            _gate_entry("incremental", True, True, False, reason),
            detail,
        )
    return (_gate_entry("incremental", True, False, False, None), detail)


def _gate_cost_line(ev: float | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Cost-line gate: merged-OOS EV of taken trades must clear the bar."""
    if not cfg["checks"]["cost_line"]:
        return _gate_entry("cost_line", False, False, False, None)
    bar = cfg["profit_bar"]
    if ev is None:
        return _gate_entry(
            "cost_line",
            True,
            True,
            False,
            "model took no out-of-sample trades (EV undefined)",
        )
    if ev < bar:
        return _gate_entry(
            "cost_line",
            True,
            True,
            False,
            f"OOS EV of taken trades {ev:.4f} < profit bar {bar}",
        )
    return _gate_entry("cost_line", True, False, False, None)


def _gate_importance(
    importance_vectors: list[np.ndarray], cfg: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    """Importance-stability gate: consistent per-fold feature rankings."""
    if not cfg["checks"]["importance_stability"]:
        return _gate_entry(
            "importance_stability", False, False, False, None
        ), math.nan
    median, _, _ = spearman_median(importance_vectors)
    floor = cfg["checks"]["importance_rho_min"]
    if not math.isfinite(median) or median < floor:
        reason = (
            f"median pairwise Spearman of per-fold importances "
            f"{median:.3f} < {floor}: feature rankings unstable"
        )
        return (
            _gate_entry("importance_stability", True, True, False, reason),
            median,
        )
    return (
        _gate_entry("importance_stability", True, False, False, None),
        median,
    )


def _gate_deflated_sharpe(
    model_stats: dict[str, float],
    family: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Deflated-Sharpe gate on the model's daily EV stream."""
    if not cfg["checks"]["deflated_sharpe"]:
        return (
            _gate_entry("deflated_sharpe", False, False, False, None),
            math.nan,
        )
    dsr = deflated_sharpe_value(
        model_stats["sr"],
        int(family["n_days"]),
        model_stats["skew"],
        model_stats["kurt_pearson"],
        float(family["var_sr"]),
        n_trials=cfg["n_trials_ds"],
    )
    floor = cfg["checks"]["dsr_min"]
    if not math.isfinite(dsr) or dsr < floor:
        reason = (
            f"deflated Sharpe of the daily EV stream {dsr:.4f} < "
            f"{floor}: edge does not survive multiple-testing discount"
        )
        return (
            _gate_entry("deflated_sharpe", True, True, False, reason),
            dsr,
        )
    return (
        _gate_entry("deflated_sharpe", True, False, False, None),
        dsr,
    )


def _gate_robustness(
    oos: dict[str, Any],
    name: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Threshold-robustness gate: the taken-book EV sign must survive.

    Donor finding: a model can beat the battery at ONE threshold while
    its EV collapses when the threshold moves -- a threshold artifact,
    not an edge.  The gate re-scores the merged OOS book at every fold
    threshold scaled by ``(1 - shift)`` and ``(1 + shift)`` with
    ``shift = checks.robustness_shift`` and TRIGGERs when the EV sign
    at either shifted book is opposite to the base book's (in
    particular, any negative shifted EV against a positive base EV).
    A shifted book under :data:`_ROBUSTNESS_MIN_SHIFT_EVENTS` trades
    cannot power the comparison: the gate degrades to INCONCLUSIVE
    (an empty base book likewise).  The per-fold thresholds are
    shifted AFTER the walk-forward picked them, so the gate reads only
    already-computed OOS probabilities -- no refit, no leakage.

    Returns:
        ``(gate_entry, detail)`` where ``detail`` carries the merged
        taken counts and EVs of the base and both shifted books.
    """
    shift_frac = float(cfg["checks"]["robustness_shift"])
    proba = np.asarray(oos["proba"][name])
    row_thr = _row_thresholds(oos, name)
    exec_vals = oos["exec"]
    detail: dict[str, Any] = {"shift_frac": shift_frac}
    taken_base = proba > row_thr
    n_base = int(taken_base.sum())
    ev_base = ev_of_taken(exec_vals, taken_base)
    detail["n_taken_base"] = n_base
    detail["ev_base"] = ev_base if math.isfinite(ev_base) else None
    if n_base == 0 or not math.isfinite(ev_base):
        reason = (
            "model took no out-of-sample trades at the fold thresholds "
            "(base book empty): no threshold sensitivity to measure"
        )
        return _gate_entry("robustness", True, False, True, reason), detail
    ev_by_book: dict[str, float] = {}
    thin_books: list[str] = []
    shift_books = (
        ("low", 1.0 - shift_frac),
        ("high", 1.0 + shift_frac),
    )
    for label, factor in shift_books:
        taken_shift = proba > row_thr * factor
        n_shift = int(taken_shift.sum())
        ev_shift = ev_of_taken(exec_vals, taken_shift)
        detail[f"n_taken_{label}"] = n_shift
        detail[f"ev_{label}"] = ev_shift if math.isfinite(ev_shift) else None
        if n_shift < _ROBUSTNESS_MIN_SHIFT_EVENTS:
            thin_books.append(f"{label} ({n_shift})")
        else:
            ev_by_book[label] = ev_shift
    if thin_books:
        reason = (
            "shifted book below the "
            f"{_ROBUSTNESS_MIN_SHIFT_EVENTS}-trade floor at "
            f"{', '.join(thin_books)}: threshold sensitivity "
            "undecidable at this book size"
        )
        return _gate_entry("robustness", True, False, True, reason), detail
    flipped: list[str] = []
    for label, ev_shift in ev_by_book.items():
        if (ev_base > 0.0 and ev_shift < 0.0) or (
            ev_base < 0.0 and ev_shift > 0.0
        ):
            flipped.append(label)
    if flipped:
        reason = (
            f"taken-book EV flips sign at threshold shift(s) "
            f"{', '.join(flipped)} (base EV {ev_base:.4f}, "
            f"shift +/-{shift_frac:.2f}): the edge is a threshold "
            "artifact"
        )
        return _gate_entry("robustness", True, True, False, reason), detail
    return _gate_entry("robustness", True, False, False, None), detail


def _model_verdict(gates: list[dict[str, Any]], powered: bool) -> str:
    """Power gate first: an underpowered run cannot trust its gates.

    Fewer than ``n_min_events`` independent taken events -> the sample
    cannot power any gate decision, so the verdict is UNDERPOWERED and
    the gate flags stay informational (the sample-size semantics this
    engine inherited); otherwise the first triggered gate names the
    verdict.
    """
    if not powered:
        return "UNDERPOWERED"
    for gate in gates:
        if gate["triggered"]:
            return f"DEAD({gate['check']})"
    return "SURVIVES"


def _evaluate_models(
    model_names: list[str],
    oos: dict[str, Any],
    taken: dict[str, np.ndarray],
    family: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the gates for every model; per-model reports.

    The four default gates plus the opt-in ``robustness`` gate (its
    detail block lands in ``models[name]["robustness"]`` only when
    enabled).  Returns ``(models_report, incremental_detail)``.
    """
    models_report: dict[str, Any] = {}
    incremental_detail: dict[str, Any] = {}
    for name in model_names:
        ev = ev_of_taken(oos["exec"], taken[name])
        n_events = decluster_count(oos["times"][taken[name]])
        powered = n_events >= cfg["n_min_events"]
        gate_incremental, detail = _gate_incremental(name, oos, taken, cfg)
        gate_cost = _gate_cost_line(ev if math.isfinite(ev) else None, cfg)
        gate_importance, median_rho = _gate_importance(
            oos["importances"][name], cfg
        )
        gate_dsr, dsr = _gate_deflated_sharpe(
            family["stats"][name], family, cfg
        )
        gates = [gate_incremental, gate_cost, gate_importance, gate_dsr]
        robustness_detail: dict[str, Any] | None = None
        if cfg["checks"]["robustness"]:
            # opt-in fifth gate, appended LAST: enabling it is additive
            # (a new entry in the kills list) and never reorders the
            # four-gate verdict contract of existing runs
            gate_robustness, robustness_detail = _gate_robustness(
                oos, name, cfg
            )
            gates.append(gate_robustness)
        incremental_detail[name] = detail
        models_report[name] = {
            "verdict": _model_verdict(gates, powered),
            "ev": ev if math.isfinite(ev) else None,
            "n_events": n_events,
            "deflated_sharpe": dsr if math.isfinite(dsr) else None,
            "importance_stability": (
                median_rho if math.isfinite(median_rho) else None
            ),
            "kills": gates,
            "importance_by_fold": [
                vector.tolist() for vector in oos["importances"][name]
            ],
            # per-trade executable returns of the merged OOS taken
            # book (deployment bridge input for ml_forward_check);
            # plain float list: report stays JSON-safe and dict==able
            "taken_exec": (np.asarray(oos["exec"])[taken[name]].tolist()),
        }
        if robustness_detail is not None:
            models_report[name]["robustness"] = robustness_detail
    return models_report, incremental_detail


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def run_ml_kill_protocol(
    features: pd.DataFrame,
    target: np.ndarray,
    exec_returns: np.ndarray,
    times_us: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
    battery_controls: list[str] | None = None,
) -> dict[str, Any]:
    """Walk-forward training + gate evaluation of one dataset.

    The dataset contract is fully caller-defined (data-agnostic): build
    one dataset per direction of your protocol -- e.g. long
    ``exec = fwd - cost``, ``target = (exec > bar)`` and the mirrored
    short recipe -- and run the engine once per dataset (the two
    variants long/short of a two-sided protocol are two runs).

    Args:
        features: DataFrame of float feature columns, chronological
            rows.  Columns named in ``battery_controls`` are EXCLUDED
            from the models' own feature set and ADDED to the linear
            battery comparator.  Boundary note: the engine cannot tell
            whether a feature column itself peeks into the future --
            feature causality is the caller's discipline.  What the
            walk-forward structure guarantees is that a leaky feature's
            CONSEQUENCES are caught: past-only training with embargo
            means an OOS fold never shares rows with its fit, so a
            look-ahead edge collapses on unseen blocks.
        target: 0/1 integer labels of the run's direction.
        exec_returns: executable per-event returns (costs already
            netted by the caller); they drive thresholds, EV, the
            bootstrap and the daily EV streams.
        times_us: int64 microsecond stamps, monotonically
            non-decreasing, one per row (the walk-forward axis).
        config: overrides over :data:`DEFAULT_TRAINING_CONFIG` (nested
            dicts merge key-wise).  Notable keys: ``model`` /
            ``models`` (registry names; several models share one CV
            pass and are compared, never selected), ``checks`` (gate
            on/off and thresholds), ``profit_bar``, ``k_blocks``,
            ``purge_h``, ``embargo_h``, ``model_params``,
            ``prereg_hash`` (hash of the frozen preregistration; when
            absent, the kill verdict is unanchored and a
            :class:`RuntimeWarning` says so -- a warning, NOT an
            error, so existing callers keep working).
        battery_controls: optional feature columns feeding the linear
            battery of the incremental gate; without them that gate is
            reported inconclusive (not triggered) and everything else
            runs.

    Returns:
        One dict: ``{verdict, kills, n_events, n_rows, folds, models
        (per-model verdicts/EV/gates/importances/taken book),
        baseline, battery, incremental, dsr_detail, ev_model,
        ev_baseline, ev_battery, deflated_sharpe,
        importance_stability, powered, n_min_events, prereg_hash,
        config}``.  ``verdict`` is an evaluation verdict of the run:
        ``SURVIVES`` / ``DEAD(<gate>)`` / ``UNDERPOWERED``.
        ``UNDERPOWERED`` is reported, not a kill: the run is undecided,
        not dead (too few independent events to power any gate
        decision).  A model
        with fewer than ``n_min_events`` declustered taken events is
        UNDERPOWERED (reported, not a gate failure -- gate flags stay
        informational there); otherwise its first triggered gate names
        its verdict, and the run verdict echoes the first non-surviving
        model.  Flat ``ev_model`` / ``deflated_sharpe`` / ``kills`` /
        ... alias the primary model's entries.

    Raises:
        ValueError: if the config or the dataset violates its contract
            (see the validators), before any model is fit.

    Example::

        A planted nonlinear edge (quadrant interaction of f1*f2 plus a
        mild linear f3 term), the DSR gate off because the toy sample
        spans too few days for it::

        >>> rng = np.random.default_rng(0)
        >>> stamps = np.arange(600, dtype=np.int64) * 3_600_000_000
        >>> hidden = rng.standard_normal((600, 3))
        >>> feats = pd.DataFrame(
        ...     {"f1": hidden[:, 0], "f2": hidden[:, 1], "f3": hidden[:, 2]}
        ... )
        >>> exec_ret = (
        ...     0.01 * np.sign(hidden[:, 0] * hidden[:, 1])
        ...     + 0.002 * hidden[:, 2]
        ...     + 0.003 * rng.standard_normal(600)
        ... )
        >>> labels = (exec_ret > 0.004).astype(np.int64)
        >>> report = run_ml_kill_protocol(
        ...     feats,
        ...     labels,
        ...     exec_ret,
        ...     stamps,
        ...     config={
        ...         "k_blocks": 3,
        ...         "bootstrap_b": 100,
        ...         "n_min_events": 5,
        ...         "checks": {"deflated_sharpe": False},
        ...         "model_params": {"gbm": {"n_estimators": 25}},
        ...         "prereg_hash": "b2c3d4e5",
        ...     },
        ... )
        >>> report["verdict"]
        'SURVIVES'
        >>> round(report["ev_model"], 4)
        0.0104
        >>> report["prereg_hash"]
        'b2c3d4e5'
    """
    cfg = _validate_config(_merge_config(config))
    if cfg["prereg_hash"] is None:
        # backward-compat warning, deliberately NOT a ValueError: an
        # unanchored kill verdict still runs, it just says so loudly
        warnings.warn(
            "no prereg_hash in config — the kill verdict is unanchored "
            "to a frozen preregistration",
            RuntimeWarning,
            stacklevel=2,
        )
    run_data = _validate_inputs(
        features, target, exec_returns, times_us, battery_controls
    )
    model_specs = _resolve_model_specs(cfg)
    model_names = [name for name, _, _ in model_specs]
    fold_rows, oos = _run_cv_folds(run_data, cfg, model_specs)
    taken = _taken_flags(oos)
    family = _stream_family(oos, taken)
    models_report, incremental_detail = _evaluate_models(
        model_names, oos, taken, family, cfg
    )
    primary = (
        cfg["model"] if cfg["model"] in models_report else (model_names[0])
    )
    primary_report = models_report[primary]
    run_verdict = primary_report["verdict"]
    for name in model_names:
        if models_report[name]["verdict"] != "SURVIVES":
            run_verdict = models_report[name]["verdict"]
            break
    baseline_ev = ev_of_taken(oos["exec"], taken[_BASELINE_NAME])
    battery_available = run_data["battery_available"]
    battery_entry: dict[str, Any] = {"available": battery_available}
    if _BATTERY_NAME in taken:
        battery_ev = ev_of_taken(oos["exec"], taken[_BATTERY_NAME])
        battery_entry["ev"] = (
            battery_ev if math.isfinite(battery_ev) else None
        )
        battery_entry["n_events"] = decluster_count(
            oos["times"][taken[_BATTERY_NAME]]
        )
    else:
        battery_entry["ev"] = None
        battery_entry["n_events"] = 0
    return {
        "verdict": run_verdict,
        "kills": primary_report["kills"],
        "n_events": primary_report["n_events"],
        "n_rows": int(run_data["times_us"].size),
        "folds": fold_rows,
        "models": models_report,
        "baseline": {
            "ev": baseline_ev if math.isfinite(baseline_ev) else None,
            "n_events": decluster_count(oos["times"][taken[_BASELINE_NAME]]),
        },
        "battery": battery_entry,
        "incremental": incremental_detail,
        "dsr_detail": {
            "n_days": family["n_days"],
            "var_sr": family["var_sr"],
            "sr": {
                name: (entry["sr"] if math.isfinite(entry["sr"]) else None)
                for name, entry in family["stats"].items()
            },
        },
        "ev_model": primary_report["ev"],
        "ev_baseline": (baseline_ev if math.isfinite(baseline_ev) else None),
        "ev_battery": battery_entry["ev"],
        "deflated_sharpe": primary_report["deflated_sharpe"],
        "importance_stability": primary_report["importance_stability"],
        # powered speaks for the WHOLE run: an underpowered secondary
        # model makes the run undecided even when the primary is fine,
        # so it must not contradict a non-SURVIVES run verdict
        "powered": all(
            entry["verdict"] != "UNDERPOWERED"
            for entry in models_report.values()
        ),
        "n_min_events": cfg["n_min_events"],
        "prereg_hash": cfg["prereg_hash"],
        "config": _echo_config(cfg),
    }


# ---------------------------------------------------------------------------
# deployment bridge: walk-forward verdict -> live track record
# ---------------------------------------------------------------------------
_FORWARD_ROUTE = "ml_taken_book → track_record.live_verdict"
"""Value of the ``route`` key: the exact delegation of
:func:`ml_forward_check` -- the walk-forward taken book's live track
routed through the track-record contract test."""


def ml_forward_check(
    report: dict[str, Any],
    live_returns: Any,
    *,
    expected_sharpe: float,
    n_backtest: int,
    periods: int = 252,
    kill_z: float = 1.64,
    dd_floor: float = -0.25,
    min_observations: int = 252,
) -> dict[str, Any]:
    """Route a deployed ML taken book's live track through live_verdict.

    The bridge between the walk-forward verdict and the live track --
    the ML analogue of the rule-based forward gate.  NOT a gate and
    NOT part of the walk-forward: forward tracking happens after
    deployment, when the survived model's taken book has accrued a
    real live record.  The helper extracts the primary model's
    taken-book per-trade stream from the engine report (the
    ``taken_exec`` block of ``report["models"][primary]``),
    normalizes both streams (finite-value filter, flat 1-D), and
    delegates the verdict entirely to
    :func:`atropos.track_record.live_verdict` -- the same two
    preregistered kill rules (drawdown barrier, then the
    prereg-divergence z at the decision point).  No ML-specific kill
    logic is added here: the verdict IS the track-record verdict, so
    a live ML book is held to exactly the standard a rule-based
    strategy is.

    Args:
        report: a report from :func:`run_ml_kill_protocol`; the
            primary model is resolved exactly like the engine's flat
            aliases (``config["model"]`` when it is among the
            evaluated models, else the first evaluated model).  The
            model entry must carry ``taken_exec``.
        live_returns: per-period live returns of the deployed taken
            book since deployment (lists / Series / arrays accepted).
        expected_sharpe: annualized Sharpe promised by the prereg;
            validated by :func:`atropos.track_record.live_verdict`.
        n_backtest: observations behind the preregistered estimate
            (the walk-forward run's independent events are the honest
            choice -- see ``report["n_events"]``).
        periods: periods per year of the live stream (252 = daily).
        kill_z: one-sided kill threshold in sigmas.
        dd_floor: preregistered drawdown barrier, negative.
        min_observations: preregistered decision point length.

    Returns:
        The :func:`atropos.track_record.live_verdict` verdict dict
        (``verdict`` / ``reason`` / ``n_obs`` / ``live_sharpe`` /
        ``z`` / ``mdd`` / ...) plus the bridge keys: ``route`` (the
        delegation, fixed string), ``model`` (the primary model the
        track belongs to), ``bt_n_trades`` / ``bt_ev`` (the
        walk-forward taken book's size and mean per-trade executable
        value -- the anchor the live per-trade statistics can be
        eyeballed against) and ``bt_verdict`` (the walk-forward
        verdict the live track must not contradict).

    Raises:
        ValueError: if ``report`` is not an engine report with a
            non-empty ``models`` dict, if the primary model entry
            carries no ``taken_exec`` stream, or through
            :func:`atropos.track_record.live_verdict` on invalid
            prereg parameters.

    Example::

        >>> import numpy as np
        >>> report = {
        ...     "config": {"model": "gbm"},
        ...     "models": {"gbm": {
        ...         "verdict": "SURVIVES",
        ...         "taken_exec": np.array([0.010, 0.006, 0.012]),
        ...     }},
        ... }
        >>> live = np.linspace(0.0005, 0.0015, 300)
        >>> out = ml_forward_check(
        ...     report, live, expected_sharpe=5.0, n_backtest=843
        ... )
        >>> out["verdict"]
        'LIVE_OK'
        >>> out["route"]
        'ml_taken_book → track_record.live_verdict'
        >>> out["model"]
        'gbm'
        >>> round(out["bt_ev"], 4)
        0.0093
    """
    models = report.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(
            "report must be a run_ml_kill_protocol report with a "
            "non-empty 'models' dict"
        )
    config = report.get("config")
    configured = config.get("model") if isinstance(config, dict) else None
    primary = configured if configured in models else next(iter(models))
    entry = models[primary]
    taken_exec = entry.get("taken_exec") if isinstance(entry, dict) else None
    if taken_exec is None:
        raise ValueError(
            f"report['models']['{primary}'] carries no 'taken_exec' "
            "stream: the bridge needs the walk-forward taken-book "
            "per-trade returns the engine reports"
        )
    book = clean_1d(taken_exec)
    live = clean_1d(live_returns)
    verdict = live_verdict(
        live,
        expected_sharpe,
        n_backtest=n_backtest,
        periods=periods,
        kill_z=kill_z,
        dd_floor=dd_floor,
        min_observations=min_observations,
    )
    book_ev = float(book.mean()) if book.size else math.nan
    out: dict[str, Any] = dict(verdict)
    out["route"] = _FORWARD_ROUTE
    out["model"] = primary
    out["bt_n_trades"] = int(book.size)
    out["bt_ev"] = book_ev if math.isfinite(book_ev) else None
    out["bt_verdict"] = (
        entry.get("verdict") if isinstance(entry, dict) else None
    )
    return out


def _echo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe echo of the merged config (tuples become lists)."""
    echo = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in cfg.items()
    }
    echo["checks"] = dict(cfg["checks"])
    echo["model_params"] = {
        name: dict(params) for name, params in cfg["model_params"].items()
    }
    return echo


# ---------------------------------------------------------------------------
# synthetic verification demo
# ---------------------------------------------------------------------------
def _demo_world(
    kind: str,
    n_rows: int = 7000,
    seed: int = 7,
) -> dict[str, Any]:
    """Synthetic hourly world with (or without) a planted nonlinear edge.

    ``kind='planted'``: the executable return loads on the quadrant
    interaction ``tanh(f1 * f2 / 2)`` -- invisible to any linear model
    -- plus a mild linear ``f3`` component the battery CAN see, noise
    elsewhere.  ``kind='noise'``: executable returns are pure noise.
    Battery controls ``ctrl1`` / ``ctrl2`` are noise columns.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    noise = 0.0024 * rng.standard_normal(n_rows)
    if kind == "planted":
        nonlinear = 0.012 * np.tanh(feats[:, 0] * feats[:, 1] / 2.0)
        linear = 0.0035 * feats[:, 2]
        exec_ret = nonlinear + linear + noise
    elif kind == "noise":
        # symmetric zero-mean outcomes: the positive class sits at 50%
        # so models still trade a lot (a powered, honest DEAD rather
        # than a silent UNDERPOWERED) while every gate sees pure noise
        exec_ret = 0.008 * np.sign(rng.standard_normal(n_rows))
    else:
        raise ValueError(f"unknown demo kind {kind!r}")
    feature_frame = pd.DataFrame(
        {
            "f1": feats[:, 0],
            "f2": feats[:, 1],
            "f3": feats[:, 2],
            "f4": feats[:, 3],
            "ctrl1": controls[:, 0],
            "ctrl2": controls[:, 1],
        }
    )
    stamps = (
        int(np.datetime64("2023-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_rows, dtype=np.int64) * _HOUR_US
    )
    labels = (exec_ret > DEFAULT_TRAINING_CONFIG["profit_bar"]).astype(
        np.int64
    )
    return {
        "features": feature_frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["ctrl1", "ctrl2"],
    }


def _vrp_panic_world(
    n_rows: int = 1600,
    seed: int = 41,
) -> dict[str, Any]:
    """Statistics-calibrated world for the documented VRP-panic kill.

    Honest marking: this is a statistics-calibrated reproduction of a
    documented tombstone (closed donor corpus) -- NOT a re-test of the
    original data, which are not available.  The synthetic hourly
    corpus is calibrated to the tombstone's documented MARGINAL
    statistics so the engine can be shown to reproduce the documented
    verdict from those statistics alone: unconditional per-trade EV
    ``-6.43%`` (the tombstone's control book, n=253); panic events
    (top decile of the VRP proxy) worse by ~1.56pp per trade (the
    documented negative conditional lift -- panic is WORSE than
    average); panic-window median IV30-RV7 ~ ``+12.1`` vol points
    (VRP on paper, never monetized); per-trade sd ``0.15 = |EV| *
    sqrt(n) / 10`` at the demo book size, so the taken book's implied
    t lands near the documented ``-10``.  The return structure is
    LINEAR in the proxy, so the linear battery sees everything the
    model sees -- the tombstone's INCREMENTAL point: conditioning
    adds nothing beyond the known linear structure.  Features:
    ``iv_rv_gap_z`` (causal z of the IV30-RV7 proxy, the conditioning
    signal), ``iv_level_z`` / ``rv7_z`` (noise); battery controls
    ``momentum`` / ``mean_reversion`` (noise).  The panic window is
    marked like the donor screen: corpus-level p90 of the proxy.
    Calibration is on MARGINAL statistics only, so a world seed other
    than the shipped one can echo ``DEAD(cost_line)`` instead of
    ``DEAD(incremental)`` (roughly a quarter of realizations): which
    of the pair the verdict echoes is book noise, the PAIR is the
    contract.

    Returns:
        Dataset dict of the engine contract (``features`` / ``target``
        / ``exec_returns`` / ``times_us`` / ``battery_controls``)
        plus ``calibration`` -- realized world statistics feeding the
        demo's tombstone-vs-engine table.
    """
    warmup = 168  # causal z window; the NaN head is sliced off below
    rng = np.random.default_rng(seed)
    # IV30-RV7 proxy in vol points: N(4.95, 4.3) puts the top-decile
    # median at ~+12.1pt, the tombstone's panic-window figure
    gap_pt = 4.95 + 4.3 * rng.standard_normal(n_rows + warmup)
    gap_z = rolling_z(gap_pt, warmup)[warmup:]
    gap_pt = gap_pt[warmup:]
    panic_mask = gap_z > float(np.quantile(gap_z, 0.90))
    # executable returns: the control-book EV, a LINEAR panic penalty
    # spreading the documented -1.56pp conditional lift over the z
    # range (slope = -0.0156 / E[z | panic window]; the realized
    # rolling-z tail is heavier than a Gaussian's), and straddle-scale
    # per-trade noise calibrated to the tombstone's t ~ -10 at the
    # demo book size: sd = |EV| * sqrt(n) / 10
    exec_ret = -0.0643 - 0.0069 * gap_z + 0.15 * rng.standard_normal(n_rows)
    feature_frame = pd.DataFrame(
        {
            "iv_rv_gap_z": gap_z,
            "iv_level_z": rng.standard_normal(n_rows),
            "rv7_z": rng.standard_normal(n_rows),
            "momentum": rng.standard_normal(n_rows),
            "mean_reversion": rng.standard_normal(n_rows),
        }
    )
    stamps = (
        int(np.datetime64("2024-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_rows, dtype=np.int64) * _HOUR_US
    )
    labels = (exec_ret > DEFAULT_TRAINING_CONFIG["profit_bar"]).astype(
        np.int64
    )
    calibration = {
        "sd_trade": 0.15,
        "n_panic": int(panic_mask.sum()),
        "panic_median_gap_pt": float(np.median(gap_pt[panic_mask])),
        "panic_lift": float(exec_ret[panic_mask].mean() - exec_ret.mean()),
    }
    return {
        "features": feature_frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["momentum", "mean_reversion"],
        "calibration": calibration,
    }


def run_all_examples() -> None:
    """Verify the engine on five synthetic worlds (asserted demo).

    Case 1: planted nonlinear edge, two tree models in one pass --
    both must clear every gate (SURVIVES) while the linear battery
    stays blind to the interaction (incremental gate); case 2: pure
    noise must NOT survive; case 3: a thin sample must come back
    UNDERPOWERED (reported, not a gate failure); case 4: the FULL
    registry in one comparison pass on reduced capacities -- every
    entry reported, every probability valid, SURVIVES the majority
    verdict; case 5: a statistics-calibrated reproduction of a
    documented tombstone (closed donor corpus) must come back
    ``DEAD(...)`` with BOTH documented cause-of-death gates
    triggered (cost line + incremental).
    """
    print("=" * 78)
    print("CASE 1: planted nonlinear edge, models=['gbm', 'rf']")
    print("=" * 78)
    planted = _demo_world("planted")
    report = run_ml_kill_protocol(
        planted["features"],
        planted["target"],
        planted["exec_returns"],
        planted["times_us"],
        config={
            "models": ["gbm", "rf"],
            # sharper leaves so the forest emits usable probabilities
            "model_params": {"rf": {"min_samples_leaf": 1, "max_depth": 7}},
        },
        battery_controls=planted["battery_controls"],
    )
    _print_report(report)
    print()
    print("=" * 78)
    print("CASE 2: pure noise (must NOT survive)")
    print("=" * 78)
    noise = _demo_world("noise")
    noise_report = run_ml_kill_protocol(
        noise["features"],
        noise["target"],
        noise["exec_returns"],
        noise["times_us"],
        # the linear arm trades the whole symmetric book (a powered,
        # honest DEAD); a deep tree would overfit itself under the
        # 500-event floor and die UNDERPOWERED instead
        config={"model": "logit"},
        battery_controls=noise["battery_controls"],
    )
    _print_report(noise_report)
    print()
    print("=" * 78)
    print("CASE 3: thin sample (must be UNDERPOWERED)")
    print("=" * 78)
    thin = _demo_world("planted", n_rows=480, seed=11)
    thin_report = run_ml_kill_protocol(
        thin["features"],
        thin["target"],
        thin["exec_returns"],
        thin["times_us"],
        config={"model_params": {"gbm": {"n_estimators": 60}}},
        battery_controls=thin["battery_controls"],
    )
    _print_report(thin_report)
    print()
    print("=" * 78)
    print("CASE 4: full registry, reduced capacities (one CV pass)")
    print("=" * 78)
    full = _demo_world("planted", n_rows=9000, seed=17)
    full_report = run_ml_kill_protocol(
        full["features"],
        full["target"],
        full["exec_returns"],
        full["times_us"],
        config={
            "models": sorted(MODEL_REGISTRY),
            "bootstrap_b": 400,
            # forest sizes shrunk so the eight-model pass stays fast;
            # the boosting pair keeps default capacity because a
            # 60-tree budget blurs its threshold book enough to fall
            # under the 500-event power floor, which would misreport
            # UNDERPOWERED for a model that trades fine at defaults
            "model_params": {
                "rf": {
                    "n_estimators": 60,
                    "max_depth": 7,
                    "min_samples_leaf": 1,
                },
                "extratrees": {"n_estimators": 60},
            },
        },
        battery_controls=full["battery_controls"],
    )
    _print_report(full_report)
    print()
    print("=" * 78)
    print(
        "CASE 5: documented-kill reproduction (VRP panic) -- "
        "statistics-calibrated reproduction of a documented tombstone "
        "(closed donor corpus)"
    )
    print("=" * 78)
    panic = _vrp_panic_world()
    panic_report = run_ml_kill_protocol(
        panic["features"],
        panic["target"],
        panic["exec_returns"],
        panic["times_us"],
        config={
            # frozen low-side grid: on an all-negative-EV world the
            # default 0.30-0.70 grid lets the argmax chase degenerate
            # thin books (a handful of trades, huge per-book noise);
            # the low grid keeps the taken book powered so the gates
            # can speak -- the same frozen-grid discipline a
            # preregistered short-vol direction run would use
            "threshold_grid": (0.20, 0.25, 0.30),
            "model_params": {"gbm": {"n_estimators": 120}},
        },
        battery_controls=panic["battery_controls"],
    )
    _print_report(panic_report)
    _print_vrp_panic_refs(panic_report, panic)
    print()
    _verify_demo(report, noise_report, thin_report)
    _verify_full_registry_demo(full_report, full)
    _verify_vrp_panic_demo(panic_report, panic)


def _print_report(report: dict[str, Any]) -> None:
    """Human-readable one-run summary (demo output only)."""
    print(f"verdict     : {report['verdict']}")
    print(
        f"n_rows={report['n_rows']}  n_events={report['n_events']} "
        f"(min {report['n_min_events']}, powered={report['powered']})"
    )
    # the powered rule is non-obvious with a multi-model registry: it
    # requires EVERY model to clear the event floor, so a single thin
    # secondary flags the whole run as undecided even when the primary
    # is well-powered -- say so instead of leaving a contradiction
    underpowered = [
        f"{name} ({entry['n_events']})"
        for name, entry in report["models"].items()
        if entry["verdict"] == "UNDERPOWERED"
    ]
    if not report["powered"] and underpowered:
        print(
            "  powered=False because these registry models are below "
            f"the event floor: {', '.join(underpowered)}; "
            "SURVIVES of other models is per-model, not run-level"
        )
    print(
        f"baseline EV={_fmt_bp(report['ev_baseline'])}  "
        f"battery EV={_fmt_bp(report['ev_battery'])}"
    )
    for name, entry in report["models"].items():
        print(
            f"  model {name:<8} verdict={entry['verdict']:<28} "
            f"EV={_fmt_bp(entry['ev'])} "
            f"DSR={entry['deflated_sharpe']} "
            f"rho={entry['importance_stability']} "
            f"events={entry['n_events']}"
        )
        for gate in entry["kills"]:
            state = (
                "off"
                if not gate["enabled"]
                else (
                    "INCONCLUSIVE"
                    if gate["inconclusive"]
                    else ("TRIGGERED" if gate["triggered"] else "pass")
                )
            )
            print(f"    {gate['check']:<20} {state}")


def _fmt_bp(value: float | None) -> str:
    """Basis-point formatting for demo output."""
    if value is None:
        return "n/a"
    return f"{value * 1e4:.1f}bp"


def _fmt_pct(value: float | None) -> str:
    """Percent formatting for the tombstone-reproduction table."""
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _verify_demo(
    report: dict[str, Any],
    noise_report: dict[str, Any],
    thin_report: dict[str, Any],
) -> None:
    """Assert the demo contract (the demo fails loudly on drift)."""
    assert {"gbm", "rf"} <= set(report["models"])
    for name in ("gbm", "rf"):
        entry = report["models"][name]
        assert entry["verdict"] == "SURVIVES", (name, entry["verdict"])
        assert entry["ev"] is not None and entry["ev"] >= 0.004
        assert entry["importance_stability"] is not None and (
            entry["importance_stability"] >= 0.5
        )
        assert entry["deflated_sharpe"] is not None and (
            entry["deflated_sharpe"] >= 1.0
        )
        t_stat = report["incremental"][name]["t_vs_battery"]
        assert t_stat is not None and t_stat > 2.0
    assert report["verdict"] == "SURVIVES"
    assert report["ev_model"] is not None and report["ev_model"] <= 0.02
    assert noise_report["verdict"] != "SURVIVES"
    assert noise_report["verdict"].startswith("DEAD")
    assert thin_report["verdict"] == "UNDERPOWERED"
    assert thin_report["powered"] is False
    print("all verification checks passed")


def _verify_full_registry_demo(
    report: dict[str, Any], world: dict[str, Any]
) -> None:
    """Assert the full-registry demo contract (fails loudly on drift).

    Every bundled registry entry must complete the pass and appear in
    the report; every entry's probabilities must be valid at demo
    scale (refit on a 2 x 2400-row subslice of the world with the
    run's own merged ``model_params``); and the planted edge must
    leave SURVIVES the MAJORITY outcome -- at least 4 of the 8
    bundled models clear all four gates simultaneously.  The bar of 4
    is the honest floor observed across the fleet (the linear
    ``logit`` arm and the threshold-compressed ``knn`` are
    structurally not expected to clear the incremental gate here); it
    is not tuned to whatever the last run happened to produce.
    """
    assert set(report["models"]) == set(MODEL_REGISTRY)
    n_survives = sum(
        entry["verdict"] == "SURVIVES" for entry in report["models"].values()
    )
    controls = list(world["battery_controls"] or [])
    columns = [
        name for name in world["features"].columns if name not in controls
    ]
    matrix = world["features"].loc[:, columns].to_numpy(dtype=np.float64)
    labels = np.asarray(world["target"])
    # demo-scale subslice of the world: refitting the full-capacity
    # boosting pair on the whole train half would double the case's
    # runtime for no extra contractual strength
    train_rows = min(matrix.shape[0] // 2, 2400)
    test_rows = min(matrix.shape[0] - train_rows, 2400)
    train_x = matrix[:train_rows]
    train_y = labels[:train_rows]
    test_x = matrix[train_rows : train_rows + test_rows]
    run_params = report["config"]["model_params"]
    for name in sorted(MODEL_REGISTRY):
        params = {**run_params.get(name, {})}
        params["seed"] = report["config"]["model_seed"]
        model = MODEL_REGISTRY[name](**params).fit(train_x, train_y)
        proba = np.asarray(model.predict_proba(test_x))
        assert proba.shape == (test_rows, 2), name
        assert (proba >= 0.0).all() and (proba <= 1.0).all(), name
        np.testing.assert_allclose(
            proba.sum(axis=1), 1.0, atol=1e-9, err_msg=name
        )
    assert n_survives >= 4, {
        name: entry["verdict"] for name, entry in report["models"].items()
    }
    print("full-registry verification checks passed")


def _print_vrp_panic_refs(
    report: dict[str, Any], world: dict[str, Any]
) -> None:
    """Tombstone-vs-engine table for the reproduction demo (output only).

    Every engine number is printed next to the documented statistic it
    was calibrated against, so the reproduction is auditable at a
    glance; the implied book t is engine arithmetic (EV over the
    calibrated per-trade sd), not a new engine output.
    """
    calibration = world["calibration"]
    sd = float(calibration["sd_trade"])
    ev = report["ev_model"]
    n_events = int(report["n_events"])
    implied_t = (
        report["ev_model"] / sd * math.sqrt(n_events)
        if ev is not None
        else math.nan
    )
    t_battery = report["incremental"]["gbm"]["t_vs_battery"]
    print("tombstone (closed donor corpus, VRP-panic entry):")
    print(
        "  panic straddle -7.98%/trade (t -10.03, n=23, hit 0%); "
        "control -6.43%/trade (n=253)"
    )
    print(
        "  conditional lift -1.56%/trade (Welch t -1.89): panic "
        "WORSE than average"
    )
    print(
        "  panic median IV30-RV7 +12.1pt: VRP on paper, never "
        "monetized; COD cost + incremental"
    )
    print("engine reproduction (statistics-calibrated world):")
    print(
        f"  taken book EV {_fmt_pct(ev)} (n_events {n_events}, "
        f"implied t {implied_t:.2f})"
    )
    print(
        f"  incremental t vs battery {t_battery:.2f} "
        f"(<= 2.0 -> no incremental edge)"
    )
    print(
        "  realized corpus: panic median gap "
        f"{calibration['panic_median_gap_pt']:+.1f}pt, lift "
        f"{calibration['panic_lift'] * 100:+.2f}pp "
        "(same sign as the documented -1.56pp; within panic-window "
        "sample noise)"
    )
    print(
        f"  verdict {report['verdict']}: cost_line and incremental "
        "both triggered -- the documented COD pair"
    )


def _verify_vrp_panic_demo(
    report: dict[str, Any], world: dict[str, Any]
) -> None:
    """Assert the tombstone-reproduction contract (fails loudly on drift).

    The documented kill must reproduce structurally: a POWERED run
    whose taken book carries the documented economics (per-trade EV in
    the tombstone band, not merely any loss) and dies with BOTH
    documented cause-of-death gates triggered -- ``cost_line`` (the
    -7.98%/trade straddle economics never clear the profit bar) and
    ``incremental`` (the conditioning adds nothing over the linear
    battery; the documented lift is negative).  The verdict echoes the
    FIRST triggered gate, so the echoed name can be either of the pair
    (book noise decides at demo scale): the COD match is the pair, not
    the name.  The world carries its own calibration asserts (panic
    worse than average, panic-window proxy median at the documented
    level).
    """
    assert report["verdict"].startswith("DEAD("), report["verdict"]
    assert report["powered"] is True
    entry = report["models"]["gbm"]
    assert entry["ev"] is not None and -0.10 < entry["ev"] < -0.03
    gates = {gate["check"]: gate for gate in entry["kills"]}
    assert gates["cost_line"]["triggered"] is True
    assert gates["incremental"]["triggered"] is True
    assert 300 <= report["n_events"] <= 1200, report["n_events"]
    calibration = world["calibration"]
    assert calibration["panic_lift"] < 0  # panic worse than average
    assert 10.0 < calibration["panic_median_gap_pt"] < 14.0
    print("tombstone-reproduction verification checks passed")


if __name__ == "__main__":
    run_all_examples()
