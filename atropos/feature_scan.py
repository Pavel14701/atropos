"""Feature lab: pre-registration-grade scans around atropos.orthogonal.

This module is ANALYTIC TOOLING, not a protocol stage: it produces
classification cells and diagnostics that help DESIGN a preregistered
run; it never validates a strategy.  Every function returns a fixed
TypedDict schema (every key present on every outcome, same discipline
as :mod:`atropos.orthogonal`, whose ``orthogonal_check`` this module
re-exports into its pipeline).

The functions, in pipeline order:

* :func:`missingness_scan` -- is missingness random, or does the NaN
  pattern carry outcome information (selection bias) or correlate
  with a control (proxy missingness)?
* :func:`correlation_cluster` -- collapses correlated features into
  clusters and picks one representative per cluster, so the family
  for Bonferroni is ``n_effective`` (clusters), not ``n_raw``
  (columns).
* :func:`bucket_scan` -- per-feature bucket profile (n, mean, hit
  rate, normalized mean, bootstrap CI), row-level Spearman
  monotonicity, permutation p-value, and the Q1/Q2 split that asks
  WHERE a bucket effect lives (hit rate vs raw outcome vs normalized
  outcome).
* :func:`interaction_scan` -- 2-D bucket heatmap plus an additive
  vs full OLS F-test: does the pair interact beyond additivity?
* :func:`rule_fit` -- train/test evaluation of caller-built rules
  (masks): delta kept-vs-dropped on both splits, honest
  ``valid / insufficient_n / unstable`` statuses.

The orchestrator :func:`feature_analysis` chains the scans in the
order above and classifies every representative feature into one
cell::

    STRONG              orthogonal pass + monotone + regime pass
                        (verified regime indicator, checkpoint A2)
    REGIME_DEPENDENT    orthogonal pass + monotone, regime flips sign
                        (verified indicator, checkpoint A2)
    STRONG_UNVERIFIED   orthogonal pass + monotone, no (verified)
                        regime indicator supplied
    SHAPE_UNKNOWN       orthogonal pass but monotone fail
    PROXY               orthogonal gate fail (incl. collinear with a
                        control -- carries no information beyond it)
    NOISE               feature not testable (degenerate column)
    EXPLORATORY         auto-discovery output only (checkpoint A3);
                        never assigned to declared features

Recommended actions: ``build_rule`` (STRONG),
``build_rule_with_regime_filter`` (REGIME_DEPENDENT),
``prereg_regime_indicator_then_retest`` (STRONG_UNVERIFIED),
``investigate_interaction`` (SHAPE_UNKNOWN), ``discard``
(PROXY/NOISE), ``manual_review_required`` (EXPLORATORY).

Checkpoint note: this file lands in three verified checkpoints --
A1 (this state: the five functions above + the orchestrator skeleton
through rule_fit + classification), A2 (``regime_scan`` +
``threshold_scan``), A3 (``auto_discovery``, ``feature_importance``,
``auto_rule_generation`` -- all EXPLORATORY by construction).  The
report schema below is FROZEN at A1: the checkpoint-A2/A3 keys exist
from the start and stay ``None`` until their functions land.

Run ``python -m atropos.feature_scan`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import norm

from atropos._utils import align_inputs, frameify, ols_hc0, validate_int
from atropos.orthogonal import OrthogonalReport, orthogonal_check

__all__ = [
    "AutoRuleCandidate",
    "AutoRuleReport",
    "BucketCell",
    "BucketFeatureReport",
    "BucketQSplit",
    "BucketScanReport",
    "ClusterReport",
    "DiscoveryEntry",
    "DiscoveryReport",
    "FeatureAnalysisReport",
    "ImportanceEntry",
    "ImportanceReport",
    "InteractionCell",
    "InteractionScanReport",
    "MissingnessEntry",
    "MissingnessReport",
    "RegimeEntry",
    "RegimeScanReport",
    "RuleFitEntry",
    "RuleFitReport",
    "ThresholdPoint",
    "ThresholdScanReport",
    "bucket_scan",
    "correlation_cluster",
    "feature_analysis",
    "interaction_scan",
    "missingness_scan",
    "rule_fit",
]


# ---------------------------------------------------------------------------
# gates and constants (every number pinned once, named, documented)
# ---------------------------------------------------------------------------
_ALPHA = 0.05
"""Family-wise alpha behind every p-value gate in this module."""

_KS_P_GATE = 0.05
"""Missingness: outcome-distribution shift (KS) significance gate."""

_NAN_RHO_GATE = 0.3
"""Missingness: |pearson| of NaN mask vs control that flags a link."""

_MIN_KS_N = 5
"""Minimum observations per side before the KS test is allowed."""

_MIN_CORR_N = 20
"""Minimum rows before a NaN-mask/control correlation is allowed."""

_T_Q = 2.0
"""Q1/Q2 split significance gate (|t| for means, |z| for hit rates)."""

_SPEAR_P_GATE = 0.05
"""Row-level Spearman p gate behind every ``monotone`` verdict."""

_N_BOOT = 1000
"""Bootstrap draws per bucket CI (chunked, memory-bounded)."""

_T_STRONG_DEFAULT = 2.5
"""Fallback |t| gate when a caller passes ``orthogonal_threshold``."""

_MAX_CLUSTER_THRESHOLD = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# classification cells and actions
# ---------------------------------------------------------------------------
_CELL_STRONG = "STRONG"
_CELL_REGIME = "REGIME_DEPENDENT"
_CELL_UNVERIFIED = "STRONG_UNVERIFIED"
_CELL_SHAPE = "SHAPE_UNKNOWN"
_CELL_PROXY = "PROXY"
_CELL_NOISE = "NOISE"
_CELL_EXPLORATORY = "EXPLORATORY"

_ACTION_BUILD = "build_rule"
_ACTION_REGIME_FILTER = "build_rule_with_regime_filter"
_ACTION_PREREG = "prereg_regime_indicator_then_retest"
_ACTION_INTERACT = "investigate_interaction"
_ACTION_DISCARD = "discard"
_ACTION_MANUAL = "manual_review_required"

_STATUS_VALID = "valid"
_STATUS_INSUFFICIENT = "insufficient_n"
_STATUS_UNSTABLE = "unstable"


# ---------------------------------------------------------------------------
# frozen report schemas (A1: every key exists from the first commit)
# ---------------------------------------------------------------------------
class MissingnessEntry(TypedDict):
    """Missingness profile of one feature over the union index.

    ``ks_p`` compares the outcome distribution where the feature is
    NaN against where it is observed (NaN when either side has fewer
    than :data:`_MIN_KS_N` observations); ``nan_correlates_with`` maps
    every control to the pearson correlation of the NaN indicator
    with that control (NaN when under-sampled or the control is
    constant); ``warn`` is True only when the fraction exceeds the
    caller's threshold AND the pattern is non-random (outcome shift
    or control link).
    """

    nan_fraction: float
    ks_p: float
    nan_outcome_differs: bool
    nan_correlates_with: dict[str, float]
    warn: bool


class MissingnessReport(TypedDict):
    """Per-feature missingness profiles plus the run's notes."""

    label: str
    nan_threshold: float
    features: dict[str, MissingnessEntry]
    notes: list[str]


class ClusterReport(TypedDict):
    """Correlation clusters, representatives and the Bonferroni split.

    ``clusters`` holds the member lists (average linkage over
    ``1 - |pearson|`` distance, cut at ``threshold``);
    ``representatives`` maps EVERY member to its cluster's
    representative (the representative maps to itself); the
    family-wise Bonferroni threshold is ``alpha / n_effective`` where
    ``n_effective`` is the cluster count, not the column count.
    """

    label: str
    threshold: float
    clusters: list[list[str]]
    representatives: dict[str, str]
    n_raw: int
    n_effective: int
    bonferroni_threshold: float
    notes: list[str]


class BucketCell(TypedDict):
    """Statistics of one bucket: n, outcome mean, hit rate, CI.

    ``mean_norm_r`` is the mean of the normalized outcome (the
    caller-supplied ``norm_r`` column, defaulting to the outcome
    itself); ``ci_lo`` / ``ci_hi`` bound the percentile bootstrap CI
    of ``mean_outcome``.
    """

    n: int
    mean_outcome: float
    hit_rate: float
    mean_norm_r: float
    ci_lo: float
    ci_hi: float


class BucketQSplit(TypedDict):
    """Low-half vs high-half bucket comparison (the Q1/Q2 question).

    Q1 pools the low bucket indices, Q2 the high ones (an odd middle
    bucket is excluded).  ``net_r_t`` / ``norm_r_t`` are Welch t
    statistics (Q2 minus Q1), ``hit_rate_z`` the pooled
    two-proportion z; each ``*_signal`` flag is True when the stat is
    finite and its absolute value reaches :data:`_T_Q`.  When the
    caller supplied no ``norm_r`` column, ``norm_r_t`` duplicates
    ``net_r_t`` (an affine transform cannot change Welch t) and the
    report says so.
    """

    n_q1: int
    n_q2: int
    q1_mean: float
    q2_mean: float
    q1_hit_rate: float
    q2_hit_rate: float
    q1_norm_r: float
    q2_norm_r: float
    net_r_t: float
    hit_rate_z: float
    norm_r_t: float
    net_r_signal: bool
    hit_rate_signal: bool
    norm_r_signal: bool


class BucketFeatureReport(TypedDict):
    """Full bucket profile of one feature.

    ``spearman`` / ``spearman_p`` are the row-level rank correlation
    with the outcome; ``p_perm`` is its seeded permutation p-value;
    ``monotone`` requires the Spearman gate AND no low-power bucket
    (every bucket at least ``min_bucket_n`` rows); ``buckets`` is
    ordered by bucket index (low to high).
    """

    n_buckets_kept: int
    spearman: float
    spearman_p: float
    p_perm: float
    monotone: bool
    low_power_buckets: list[int]
    buckets: list[BucketCell]
    q_split: BucketQSplit


class BucketScanReport(TypedDict):
    """Per-feature bucket profiles for one scan."""

    label: str
    n_buckets: int
    min_bucket_n: int
    norm_r_is_outcome: bool
    features: dict[str, BucketFeatureReport]
    notes: list[str]


class InteractionCell(TypedDict):
    """One 2-D cell: bucket-a x bucket-b co-occurrence statistics."""

    n: int
    mean_outcome: float
    hit_rate: float


class InteractionScanReport(TypedDict):
    """Additive-vs-full F-test for one feature pair.

    ``cells`` keys are ``"row x col"`` bucket indices; the F-test
    compares the additive design (bucket dummies) against the design
    plus all dummy products.  ``interaction_present`` is True only
    when the F p-value clears :data:`_ALPHA` AND no cell is
    under-powered (``min_cell_n``) -- a significant F on starved cells
    is not claimed.  ``status`` is ``OK`` or ``INCONCLUSIVE`` (rank
    deficient, too few rows, or a degenerate bucket split).
    """

    label: str
    feature_a: str
    feature_b: str
    n_cells: int
    cells: dict[str, InteractionCell]
    low_power_cells: list[str]
    f_stat: float
    p_f: float
    interaction_present: bool
    status: str
    notes: list[str]


class RuleFitEntry(TypedDict):
    """Train/test evaluation of one caller-built rule mask.

    ``delta_*`` is the kept-minus-dropped outcome mean on that split
    (NaN when one side is empty); ``stable`` means both deltas are
    positive (same sign by construction); ``status`` applies the
    sample-size guards first, then the stability verdict.
    """

    n_train_kept: int
    n_test_kept: int
    delta_train: float
    delta_test: float
    stable: bool
    status: str


class RuleFitReport(TypedDict):
    """Rule masks evaluated on a caller-chosen train/test split."""

    label: str
    rules: dict[str, RuleFitEntry]
    best_valid_rule: str | None
    notes: list[str]


class RegimeEntry(TypedDict):
    """One feature's profile inside one regime value (checkpoint A2)."""

    n: int
    spearman: float
    spearman_p: float
    monotone: bool


class RegimeScanReport(TypedDict):
    """Per-regime feature behaviour against a prereg indicator (A2).

    ``status`` is ``OK`` only when the indicator carries a
    ``prereg_hash``; an unfrozen indicator reads
    ``REGIME_UNVERIFIED`` and can never upgrade a cell to STRONG.
    ``same_sign[feature]`` is the pairwise sign test across regimes;
    ``strongest_regime`` / ``weakest_regime`` name the regimes by
    |spearman|.
    """

    label: str
    regime_name: str
    regime_definition: str
    prereg_hash: str | None
    status: str
    min_regime_n: int
    features: dict[str, dict[str, RegimeEntry]]
    same_sign: dict[str, bool]
    strongest_regime: dict[str, str]
    weakest_regime: dict[str, str]
    notes: list[str]


class ThresholdPoint(TypedDict):
    """One threshold's train/test deltas on the template rule (A2)."""

    threshold: float
    rule_str: str
    n_train_kept: int
    n_test_kept: int
    delta_train: float
    delta_test: float
    status: str


class ThresholdScanReport(TypedDict):
    """Plateau-vs-edge reading of a threshold sweep (checkpoint A2).

    ``plateau_detected``: both edge thresholds reach 70% of the best
    ``delta_test``; ``edge_detected``: an edge falls below 30% of it
    (a spike -- the classic single-threshold overfit signature).
    ``optimal_range`` spans the contiguous >= 70%-of-best run around
    the argmax, or is ``None`` when no point has a positive delta.
    """

    label: str
    feature: str
    op: str
    rule_template: str
    points: list[ThresholdPoint]
    plateau_detected: bool
    edge_detected: bool
    optimal_range: list[float] | None
    notes: list[str]


class DiscoveryEntry(TypedDict):
    """One candidate that cleared the discovery gates (checkpoint A3)."""

    feature: str
    t_stat: float
    monotone: bool
    p_bonferroni: float


class DiscoveryReport(TypedDict):
    """Exploratory sweep over a candidate pool (checkpoint A3).

    EXPLORATORY by construction: the output is a hypothesis list for
    a future preregistration, never a classification upgrade, and it
    is never merged into the declared-feature cells.
    """

    label: str
    n_candidates: int
    t_gate: float
    discovered: list[DiscoveryEntry]
    all_below_threshold: bool
    note: str
    notes: list[str]


class ImportanceEntry(TypedDict):
    """Permutation importance of one feature (checkpoint A3)."""

    importance_mean: float
    importance_std: float
    ci_lo: float
    ci_hi: float


class ImportanceReport(TypedDict):
    """Model-based importance; diagnostics only, never a cell (A3)."""

    label: str
    model: str
    n_permutations: int
    features: dict[str, ImportanceEntry]
    notes: list[str]


class AutoRuleCandidate(TypedDict):
    """One grid-searched rule evaluated train-then-test (A3)."""

    name: str
    rule_str: str
    n_train_kept: int
    n_test_kept: int
    delta_train: float
    delta_test: float
    status: str


class AutoRuleReport(TypedDict):
    """Grid search over a caller-fixed rule shape (checkpoint A3).

    Values are searched on TRAIN only; the top candidates then face
    the untouched test split.  ``overfit_warning`` is True when every
    candidate that worked on train fails on test.
    """

    label: str
    templates: dict[str, str]
    candidates: list[AutoRuleCandidate]
    valid_rules: list[str]
    best_valid_rule: str | None
    overfit_warning: bool
    notes: list[str]


class FeatureAnalysisReport(TypedDict):
    """Frozen top-level schema; optional blocks stay ``None``.

    ``classification`` / ``recommended_actions`` cover the
    REPRESENTATIVE features only (one verdict per cluster);
    ``skipped_features`` records why every other column has no
    independent cell: ``"noise"`` (degenerate), ``"proxy"``
    (orthogonal gate fail) or ``"cluster:<rep>"`` (covered by its
    representative).
    """

    label: str
    seed: int
    n_effective: int
    orthogonal_t_gate: float
    missingness: MissingnessReport
    clustering: ClusterReport
    orthogonal: OrthogonalReport
    buckets: BucketScanReport
    interaction: dict[str, InteractionScanReport]
    rules: RuleFitReport | None
    regime: RegimeScanReport | None
    threshold: dict[str, ThresholdScanReport] | None
    discovery: DiscoveryReport | None
    importance: ImportanceReport | None
    rule_generation: AutoRuleReport | None
    skipped_features: dict[str, str]
    classification: dict[str, str]
    recommended_actions: dict[str, str]
    notes: list[str]


# ---------------------------------------------------------------------------
# small helpers shared by the scans
# ---------------------------------------------------------------------------
def _validate_fraction(name: str, value: float) -> float:
    """Reject a fraction outside the open (0, 1) interval."""
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be a float in (0, 1), got {value!r}")
    return float(value)


def _to_series(values: Any, name: str) -> pd.Series:
    """One named float Series view of any 1-D pandas/numpy input."""
    frame = frameify(values, name)
    if frame.shape[1] != 1:
        raise ValueError(
            f"input {name!r} must be a single column, got"
            f" {frame.shape[1]} columns"
        )
    series = frame.iloc[:, 0].astype(float)
    series.name = name
    return series


def _welch_t(first: np.ndarray, second: np.ndarray) -> float:
    """Welch t of two samples; t = mean(first) - mean(second).

    Fewer than two observations on either side, zero-variance, or a
    non-finite denominator returns NaN (no signal, never a fake 0).
    """
    size_first = float(first.size)
    size_second = float(second.size)
    if first.size < 2 or second.size < 2:
        return float("nan")
    var_first = float(np.var(first, ddof=1))
    var_second = float(np.var(second, ddof=1))
    denom = np.sqrt(var_first / size_first + var_second / size_second)
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float((np.mean(first) - np.mean(second)) / denom)


def _prop_z(
    count_second: float,
    size_second: int,
    count_first: float,
    size_first: int,
) -> float:
    """Pooled two-proportion z: share(second) minus share(first)."""
    if size_first == 0 or size_second == 0:
        return float("nan")
    pooled = (count_first + count_second) / (size_first + size_second)
    if pooled in (0.0, 1.0):
        return float("nan")
    denom = np.sqrt(
        pooled * (1.0 - pooled) * (1.0 / size_first + 1.0 / size_second)
    )
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    share_first = count_first / size_first
    share_second = count_second / size_second
    return float((share_second - share_first) / denom)


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap CI (2.5 / 97.5) of the mean, chunked.

    Draws are generated in chunks of 100 so a large bucket cannot
    allocate ``(n_boot, n_obs)`` at once.
    """
    n_obs = values.size
    if n_obs == 0:
        return float("nan"), float("nan")
    means = np.empty(_N_BOOT)
    for start in range(0, _N_BOOT, 100):
        stop = min(start + 100, _N_BOOT)
        draws = rng.choice(values, size=(stop - start, n_obs), replace=True)
        means[start:stop] = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _spearman_pair(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    """Row-level Spearman rho + p; NaN pair for degenerate input."""
    if first.size < 3 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return float("nan"), float("nan")
    result = stats.spearmanr(first, second)
    return float(result.statistic), float(result.pvalue)


def _perm_p_spearman(
    feature_values: np.ndarray,
    outcome_values: np.ndarray,
    real_rho: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p of the row-level Spearman rho.

    The feature column is shuffled (marginal kept, link broken); the
    share of shuffled draws whose |rho| reaches the real |rho| is the
    p-value.  Non-finite draw rhos never count as exceeding.
    """
    exceed = 0
    for _ in range(n_perm):
        rho_perm = stats.spearmanr(
            rng.permutation(feature_values), outcome_values
        ).statistic
        if np.isfinite(rho_perm) and abs(float(rho_perm)) >= abs(real_rho):
            exceed += 1
    return exceed / float(n_perm)


def _bucket_indices(
    values: np.ndarray,
    n_buckets: int,
    edges: list[float] | None,
    notes: list[str],
    tag: str,
) -> tuple[np.ndarray, int]:
    """Digitize values into buckets from prereg edges or quantiles.

    Returns the per-row bucket index (0-based, ``right=False`` bins)
    and the number of DISTINCT buckets that survived.  Collapsed
    quantile edges shrink the count with a note; empty bins from
    explicit edges are kept as honest empty cells.
    """
    if edges is not None:
        interior = np.unique(
            np.asarray([float(edge) for edge in edges], dtype=float)
        )
        source = "edges"
    else:
        interior = np.unique(
            np.quantile(values, np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
        )
        source = "quantile edges"
    n_kept = int(interior.size) + 1
    if edges is None and n_kept < n_buckets:
        notes.append(
            f"{tag}: only {n_kept} of {n_buckets} buckets distinct"
            f" (duplicate {source}); buckets collapsed"
        )
    return np.digitize(values, interior), n_kept


def _hit_rate(values: np.ndarray) -> float:
    """Share of strictly positive observations; NaN when empty."""
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > 0.0))


def _cell_stats(
    outcome_values: np.ndarray,
    norm_values: np.ndarray,
    rng: np.random.Generator,
) -> BucketCell:
    """Pack one bucket's n / means / hit rate / bootstrap CI."""
    mean_outcome = (
        float(np.mean(outcome_values))
        if outcome_values.size
        else float("nan")
    )
    mean_norm = (
        float(np.mean(norm_values)) if norm_values.size else float("nan")
    )
    ci_low, ci_high = _bootstrap_ci(outcome_values, rng)
    return BucketCell(
        n=int(outcome_values.size),
        mean_outcome=mean_outcome,
        hit_rate=_hit_rate(outcome_values),
        mean_norm_r=mean_norm,
        ci_lo=ci_low,
        ci_hi=ci_high,
    )


def _q_split(
    outcome_values: np.ndarray,
    norm_values: np.ndarray,
    bucket_ids: np.ndarray,
    n_kept: int,
) -> BucketQSplit:
    """Pool low-half vs high-half buckets and run the three signals.

    The low pool takes bucket indices below ``n_kept // 2`` and the
    high pool the indices at ``n_kept - n_kept // 2`` and above; an
    odd middle bucket is deliberately excluded from both pools.
    """
    low_ids = range(n_kept // 2)
    high_start = n_kept - n_kept // 2
    high_ids = range(high_start, n_kept)
    mask_q1 = np.isin(bucket_ids, list(low_ids))
    mask_q2 = np.isin(bucket_ids, list(high_ids))
    out_q1 = outcome_values[mask_q1]
    out_q2 = outcome_values[mask_q2]
    norm_q1 = norm_values[mask_q1]
    norm_q2 = norm_values[mask_q2]
    net_t = _welch_t(out_q2, out_q1)
    norm_t = _welch_t(norm_q2, norm_q1)
    hits_q1 = float(np.sum(out_q1 > 0.0))
    hits_q2 = float(np.sum(out_q2 > 0.0))
    hit_z = _prop_z(hits_q2, int(out_q2.size), hits_q1, int(out_q1.size))
    return BucketQSplit(
        n_q1=int(out_q1.size),
        n_q2=int(out_q2.size),
        q1_mean=float(np.mean(out_q1)) if out_q1.size else float("nan"),
        q2_mean=float(np.mean(out_q2)) if out_q2.size else float("nan"),
        q1_hit_rate=_hit_rate(out_q1),
        q2_hit_rate=_hit_rate(out_q2),
        q1_norm_r=float(np.mean(norm_q1)) if norm_q1.size else float("nan"),
        q2_norm_r=float(np.mean(norm_q2)) if norm_q2.size else float("nan"),
        net_r_t=net_t,
        hit_rate_z=hit_z,
        norm_r_t=norm_t,
        net_r_signal=bool(np.isfinite(net_t) and abs(net_t) >= _T_Q),
        hit_rate_signal=bool(np.isfinite(hit_z) and abs(hit_z) >= _T_Q),
        norm_r_signal=bool(np.isfinite(norm_t) and abs(norm_t) >= _T_Q),
    )


# ---------------------------------------------------------------------------
# 1. missingness_scan
# ---------------------------------------------------------------------------
def missingness_scan(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
    *,
    nan_threshold: float = 0.05,
    label: str = "unnamed",
) -> MissingnessReport:
    """Is each feature's missingness random, or does it carry signal?

    Works on the UNION index (outer join of outcome, features and
    controls) -- the rows that a later inner-join alignment would
    silently drop are exactly the rows this scan must judge.  For
    every feature: the NaN fraction, a KS test of the outcome where
    the feature is NaN vs observed, and the pearson correlation of
    the NaN indicator with every control.  ``warn`` is True only when
    the fraction exceeds ``nan_threshold`` AND the pattern is
    non-random (outcome shift or a control link beyond
    :data:`_NAN_RHO_GATE`) -- random missingness never warns.

    Args:
        outcome: response series (pandas/numpy 1-D array-like).
        features: mapping feature name -> single column.
        controls: mapping control name -> single column.
        nan_threshold: fraction above which a feature's missingness
            becomes eligible for the warn flag; in the open (0, 1).
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`MissingnessReport`; every key is present for every
        feature, with NaN statistics where a test was under-powered.

    Raises:
        ValueError: if ``nan_threshold`` is outside (0, 1), an input
            is not a single column, or feature/control names collide.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(400)
        >>> out = pd.Series(rng.standard_normal(400), index=idx)
        >>> feat = pd.Series(rng.standard_normal(400), index=idx)
        >>> feat[out > 1.0] = np.nan
        >>> report = missingness_scan(
        ...     out, {"feat": feat}, {}, nan_threshold=0.05
        ... )
        >>> report["features"]["feat"]["nan_outcome_differs"]
        True
    """
    _validate_fraction("nan_threshold", nan_threshold)
    feature_names = list(features)
    control_names = list(controls)
    overlap = sorted(set(feature_names) & set(control_names))
    if overlap:
        raise ValueError(f"feature and control names collide: {overlap}")
    reserved = sorted((set(feature_names) | set(control_names)) & {"outcome"})
    if reserved:
        raise ValueError(
            "the name 'outcome' is reserved for the outcome column;"
            f" rename: {reserved}"
        )
    outcome_series = _to_series(outcome, "outcome")
    feature_series = {
        name: _to_series(features[name], name) for name in feature_names
    }
    control_series = {
        name: _to_series(controls[name], name) for name in control_names
    }
    union = pd.concat(
        [outcome_series, *feature_series.values(), *control_series.values()],
        axis=1,
        join="outer",
    )
    outcome_values = union["outcome"].to_numpy(dtype=float)
    entries: dict[str, MissingnessEntry] = {}
    notes: list[str] = []
    for name in feature_names:
        entries[name] = _missingness_entry(
            name,
            union[name].to_numpy(dtype=float),
            outcome_values,
            union,
            control_names,
            nan_threshold=nan_threshold,
            notes=notes,
        )
    return MissingnessReport(
        label=label,
        nan_threshold=nan_threshold,
        features=entries,
        notes=notes,
    )


def _missingness_entry(
    name: str,
    column: np.ndarray,
    outcome_values: np.ndarray,
    union: pd.DataFrame,
    control_names: list[str],
    *,
    nan_threshold: float,
    notes: list[str],
) -> MissingnessEntry:
    """Profile one feature's missingness against outcome and controls.

    Appends under-power notes to ``notes`` (KS skipped, undersampled
    indicator correlations) so no silent NaN reaches the report.
    """
    nan_mask = np.isnan(column)
    nan_fraction = float(np.mean(nan_mask)) if nan_mask.size else 0.0
    observed = ~nan_mask & np.isfinite(outcome_values)
    missing = nan_mask & np.isfinite(outcome_values)
    ks_p = float("nan")
    differs = False
    if missing.sum() >= _MIN_KS_N and observed.sum() >= _MIN_KS_N:
        ks_p = float(
            stats.ks_2samp(
                outcome_values[missing], outcome_values[observed]
            ).pvalue
        )
        differs = bool(ks_p < _KS_P_GATE)
    else:
        notes.append(
            f"{name}: KS not run (fewer than {_MIN_KS_N} outcome"
            " observations on one side)"
        )
    corr_map: dict[str, float] = {}
    mask_used = nan_mask.astype(float)
    for control_name in control_names:
        control_values = union[control_name].to_numpy(dtype=float)
        usable = np.isfinite(control_values)
        if (
            usable.sum() < _MIN_CORR_N
            or np.ptp(control_values[usable]) == 0.0
            or np.ptp(mask_used[usable]) == 0.0
        ):
            # under-sampled, constant control, or no NaN rows at
            # all: the indicator correlation is undefined (NaN)
            corr_map[control_name] = float("nan")
            continue
        corr_map[control_name] = float(
            stats.pearsonr(
                mask_used[usable], control_values[usable]
            ).statistic
        )
    linked = any(
        np.isfinite(value) and abs(value) >= _NAN_RHO_GATE
        for value in corr_map.values()
    )
    warn = bool(nan_fraction > nan_threshold and (differs or linked))
    return MissingnessEntry(
        nan_fraction=nan_fraction,
        ks_p=ks_p,
        nan_outcome_differs=differs,
        nan_correlates_with=corr_map,
        warn=warn,
    )


# ---------------------------------------------------------------------------
# 2. correlation_cluster
# ---------------------------------------------------------------------------
def correlation_cluster(
    features: dict[str, Any],
    outcome: Any,
    *,
    threshold: float = 0.8,
    representatives: dict[str, str] | None = None,
    label: str = "unnamed",
) -> ClusterReport:
    """Cluster correlated features; pick one representative each.

    Builds the pearson matrix over the aligned complete rows, converts
    it to ``1 - |rho|`` distance and cuts an average-linkage tree at
    ``threshold``.  The representative of each cluster is the member
    with the largest |pearson| against the outcome (insertion order
    breaks ties), unless ``representatives`` explicitly pins the
    choice: the mapping ``{member: representative}`` must name the
    representative and every OTHER member of its cluster (clusters
    with no entry auto-pick), point INTO the same cluster, and never
    conflict.  ``bonferroni_threshold`` is
    ``alpha / n_effective`` -- the family for downstream gates is the
    CLUSTER count, not the raw column count.

    Args:
        features: mapping feature name -> single column.
        outcome: response series; only used to rank representatives.
        threshold: cut level of |pearson|; in the open (0, 1).
        representatives: optional explicit member -> representative
            map (validated strictly, see above).
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`ClusterReport` with clusters in stable insertion
        order; every member maps to its representative.

    Raises:
        ValueError: if ``threshold`` is outside (0, 1), ``features``
            is empty, no rows survive alignment, a representative
            name is unknown, or the explicit map violates the
            every-member-or-none / same-cluster / no-conflict rules.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(200)
        >>> base = pd.Series(rng.standard_normal(200), index=idx)
        >>> twin = 0.9 * base + 0.1 * rng.standard_normal(200)
        >>> lone = pd.Series(rng.standard_normal(200), index=idx)
        >>> out = pd.Series(base + 0.3 * rng.standard_normal(200),
        ...                 index=idx)
        >>> report = correlation_cluster(
        ...     {"base": base, "twin": twin, "lone": lone}, out
        ... )
        >>> report["n_effective"]
        2
        >>> report["representatives"]["twin"]
        'base'
    """
    checked_threshold = _validate_fraction("threshold", threshold)
    if not features:
        raise ValueError("features dict must not be empty")
    names = list(features)
    frames = [
        _to_series(outcome, "outcome"),
        *(_to_series(features[name], name) for name in names),
    ]
    aligned = pd.concat(frames, axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("no rows survive alignment; cannot correlate")
    notes: list[str] = []
    rho_values = aligned[names].corr().to_numpy(dtype=float)
    degenerate = [
        name
        for position, name in enumerate(names)
        if not np.isfinite(rho_values[position, position])
    ]
    notes.extend(
        f"{name}: degenerate column (constant or all-NaN after"
        " alignment); treated as uncorrelated"
        for name in degenerate
    )
    rho_values = np.where(np.isfinite(rho_values), rho_values, 0.0)
    distance_values = 1.0 - np.abs(rho_values)
    np.fill_diagonal(distance_values, 0.0)
    distance_values = np.clip(
        (distance_values + distance_values.T) / 2.0, 0.0, 1.0
    )
    if len(names) == 1:
        labels = np.array([1])
    else:
        tree = hierarchy.linkage(
            squareform(distance_values, checks=False), method="average"
        )
        labels = hierarchy.fcluster(
            tree, t=1.0 - checked_threshold, criterion="distance"
        )
    groups: dict[int, list[str]] = {}
    for name, cluster_id in zip(names, labels, strict=True):
        groups.setdefault(int(cluster_id), []).append(name)
    order = sorted(groups, key=lambda key: names.index(groups[key][0]))
    clusters = [groups[cluster_id] for cluster_id in order]
    outcome_values = aligned["outcome"].to_numpy(dtype=float)
    rep_map = _resolve_representatives(
        clusters, aligned, outcome_values, representatives or {}
    )
    n_effective = len(clusters)
    return ClusterReport(
        label=label,
        threshold=checked_threshold,
        clusters=clusters,
        representatives=rep_map,
        n_raw=len(names),
        n_effective=n_effective,
        bonferroni_threshold=_ALPHA / n_effective,
        notes=notes,
    )


def _resolve_representatives(
    clusters: list[list[str]],
    aligned: pd.DataFrame,
    outcome_values: np.ndarray,
    explicit: dict[str, str],
) -> dict[str, str]:
    """Pick (or validate the caller-pinned) representative per cluster.

    Auto rule: the member with the largest |pearson| against the
    outcome, insertion order breaking ties.  Explicit rule: the map
    must name the representative and every OTHER member of its
    cluster, point INTO the same cluster, and never conflict; clusters
    with no entry auto-pick.  Every violation raises ValueError.
    """
    unknown = sorted(set(explicit) - set(aligned.columns) - {"outcome"})
    if unknown:
        raise ValueError(f"representative map has unknown names: {unknown}")
    rep_map: dict[str, str] = {}
    for members in clusters:
        pinned = [member for member in members if member in explicit]
        if pinned:
            targets = sorted({explicit[member] for member in pinned})
            if len(targets) != 1:
                raise ValueError(
                    f"cluster {members}: conflicting representatives"
                    f" {targets}"
                )
            chosen = targets[0]
            if chosen not in members:
                raise ValueError(
                    f"cluster {members}: representative {chosen!r} is"
                    " not a member of its own cluster"
                )
            unmapped = sorted(set(members) - {chosen} - set(pinned))
            if unmapped:
                raise ValueError(
                    f"cluster {members}: representatives must name the"
                    " representative and every other member; unmapped:"
                    f" {unmapped}"
                )
        else:
            chosen = max(
                members,
                key=lambda member: _outcome_abs_corr(
                    aligned, member, outcome_values
                ),
            )
        for member in members:
            rep_map[member] = chosen
    return rep_map


def _outcome_abs_corr(
    aligned: pd.DataFrame, member: str, outcome_values: np.ndarray
) -> float:
    """|pearson| of one member column vs the outcome (-1 for constant).

    The -1 floor ranks a degenerate column below any real candidate,
    so a constant member can never win the auto-pick.
    """
    column = aligned[member].to_numpy(dtype=float)
    if np.ptp(column) == 0.0:
        return -1.0
    return abs(float(stats.pearsonr(column, outcome_values).statistic))


# ---------------------------------------------------------------------------
# 3. bucket_scan
# ---------------------------------------------------------------------------
def bucket_scan(
    outcome: Any,
    features: dict[str, Any],
    *,
    n_buckets: int = 5,
    bucket_edges: dict[str, list[float]] | None = None,
    min_bucket_n: int = 0,
    norm_r: Any | None = None,
    n_perm: int = 200,
    seed: int = 42,
    label: str = "unnamed",
) -> BucketScanReport:
    """Bucket profile, monotonicity and the Q1/Q2 signal split.

    For every feature: buckets come from the feature's prereg
    ``bucket_edges`` when supplied (else interior quantiles), each
    bucket reports n / mean / hit rate / normalized mean / bootstrap
    CI, the row-level Spearman gives the monotone verdict (p gate AND
    no bucket below ``min_bucket_n``), and a seeded permutation gives
    ``p_perm``.  The Q1/Q2 split pools the low and high buckets and
    asks where the effect lives: raw outcome (``net_r``), win
    frequency (``hit_rate``) or the normalized outcome (``norm_r``).

    Args:
        outcome: response series.
        features: mapping feature name -> single column.
        n_buckets: requested bucket count (>= 2).
        bucket_edges: optional per-feature interior edges (prereg);
            features without an entry fall back to quantiles.
        min_bucket_n: power floor per bucket for the monotone
            verdict; 0 disables the floor.
        norm_r: optional normalized-outcome column (index-aligned
            with the outcome; the inner join is performed per
            feature); ``None`` uses the outcome itself, in which case
            ``norm_r_signal`` mirrors ``net_r_signal`` (Welch t is
            invariant to affine scaling -- the report notes this).
        n_perm: permutation draws per feature; at least 1.
        seed: non-negative seed of the pinned rng.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`BucketScanReport`; every feature key is present.

    Raises:
        ValueError: if ``n_buckets`` < 2, ``n_perm`` < 1, ``seed`` is
            negative, an input is not a single column, or ``norm_r``
            is supplied but does not align with the outcome.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(400)
        >>> feat = pd.Series(rng.uniform(0.0, 1.0, 400), index=idx)
        >>> out = pd.Series(2.0 * feat + rng.standard_normal(400),
        ...                 index=idx)
        >>> report = bucket_scan(
        ...     out, {"feat": feat}, n_perm=25, seed=42
        ... )
        >>> report["features"]["feat"]["monotone"]
        True
    """
    validate_int("n_buckets", n_buckets, minimum=2)
    validate_int("n_perm", n_perm)
    validate_int("seed", seed, minimum=0)
    validate_int("min_bucket_n", min_bucket_n, minimum=0)
    notes: list[str] = []
    outcome_series = _to_series(outcome, "outcome")
    norm_is_outcome = norm_r is None
    norm_series = (
        outcome_series if norm_is_outcome else _to_series(norm_r, "norm_r")
    )
    entries: dict[str, BucketFeatureReport] = {}
    rng = np.random.default_rng(seed)
    for name, feature_value in features.items():
        entries[name] = _bucket_profile(
            name,
            _to_series(feature_value, name),
            outcome_series,
            norm_series,
            n_buckets=n_buckets,
            feature_edges=(bucket_edges or {}).get(name),
            min_bucket_n=min_bucket_n,
            n_perm=n_perm,
            rng=rng,
            notes=notes,
        )
    if norm_is_outcome:
        notes.append(
            "norm_r defaults to the outcome: norm_r_signal mirrors"
            " net_r_signal (Welch t is invariant to affine scaling)"
        )
    return BucketScanReport(
        label=label,
        n_buckets=n_buckets,
        min_bucket_n=min_bucket_n,
        norm_r_is_outcome=norm_is_outcome,
        features=entries,
        notes=notes,
    )


def _bucket_profile(
    name: str,
    feature_series: pd.Series,
    outcome_series: pd.Series,
    norm_series: pd.Series,
    *,
    n_buckets: int,
    feature_edges: list[float] | None,
    min_bucket_n: int,
    n_perm: int,
    rng: np.random.Generator,
    notes: list[str],
) -> BucketFeatureReport:
    """Full bucket profile of one feature (see :func:`bucket_scan`).

    Collapsed-quantile notes go to ``notes``; the rng is shared so a
    fixed seed reproduces the whole scan bit for bit.
    """
    aligned_out, aligned_feat, aligned_norm = align_inputs(
        outcome_series, feature_series, norm_series
    )
    rho, rho_p = _spearman_pair(aligned_feat, aligned_out)
    bucket_ids, n_kept = _bucket_indices(
        aligned_feat, n_buckets, feature_edges, notes, name
    )
    low_power: list[int] = []
    cells: list[BucketCell] = []
    for bucket_id in range(n_kept):
        mask = bucket_ids == bucket_id
        cells.append(_cell_stats(aligned_out[mask], aligned_norm[mask], rng))
        if min_bucket_n > 0 and cells[-1]["n"] < min_bucket_n:
            low_power.append(bucket_id)
    p_perm = (
        _perm_p_spearman(aligned_feat, aligned_out, rho, n_perm, rng)
        if np.isfinite(rho)
        else float("nan")
    )
    monotone = bool(
        np.isfinite(rho)
        and np.isfinite(rho_p)
        and rho_p <= _SPEAR_P_GATE
        and not low_power
    )
    return BucketFeatureReport(
        n_buckets_kept=n_kept,
        spearman=rho,
        spearman_p=rho_p,
        p_perm=p_perm,
        monotone=monotone,
        low_power_buckets=low_power,
        buckets=cells,
        q_split=_q_split(aligned_out, aligned_norm, bucket_ids, n_kept),
    )


# ---------------------------------------------------------------------------
# 4. interaction_scan
# ---------------------------------------------------------------------------
def interaction_scan(
    outcome: Any,
    feature_a: Any,
    feature_b: Any,
    *,
    n_buckets_a: int = 3,
    n_buckets_b: int = 4,
    min_cell_n: int = 30,
    name_a: str = "feature_a",
    name_b: str = "feature_b",
    label: str = "unnamed",
) -> InteractionScanReport:
    """Does one pair interact beyond additivity (bucket dummies)?

    Quantile-buckets both features, reports the 2-D cell heatmap, then
    fits the additive design (intercepts + bucket dummies) and the
    full design (plus every dummy product) with OLS and compares them
    with the classical nested F-test on the residual sums of squares.
    ``interaction_present`` requires the F p-value below
    :data:`_ALPHA` AND no cell below ``min_cell_n`` rows -- a
    significant F over starved cells is flagged, not claimed.

    Args:
        outcome: response series.
        feature_a: first feature column (heatmap rows).
        feature_b: second feature column (heatmap columns).
        n_buckets_a: bucket count for ``feature_a`` (>= 2).
        n_buckets_b: bucket count for ``feature_b`` (>= 2).
        min_cell_n: power floor per cell for the verdict.
        name_a: feature name recorded under ``report["feature_a"]``.
        name_b: feature name recorded under ``report["feature_b"]``.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`InteractionScanReport`; ``status`` is ``OK`` or
        ``INCONCLUSIVE`` (rank-deficient design, too few rows, or a
        degenerate bucket split), with the F statistics NaN in the
        latter case.

    Raises:
        ValueError: if bucket counts are below 2 or an input is not a
            single column.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(400)
        >>> feat_a = pd.Series(rng.uniform(0, 1, 400), index=idx)
        >>> feat_b = pd.Series(rng.uniform(0, 1, 400), index=idx)
        >>> out = pd.Series(
        ...     feat_a + feat_b + rng.standard_normal(400), index=idx
        ... )
        >>> report = interaction_scan(out, feat_a, feat_b)
        >>> report["interaction_present"]
        False
    """
    validate_int("n_buckets_a", n_buckets_a, minimum=2)
    validate_int("n_buckets_b", n_buckets_b, minimum=2)
    validate_int("min_cell_n", min_cell_n, minimum=1)
    notes: list[str] = []
    aligned_out, aligned_a, aligned_b = align_inputs(
        _to_series(outcome, "outcome"),
        _to_series(feature_a, "feature_a"),
        _to_series(feature_b, "feature_b"),
    )
    ids_a, n_a = _bucket_indices(aligned_a, n_buckets_a, None, notes, name_a)
    ids_b, n_b = _bucket_indices(aligned_b, n_buckets_b, None, notes, name_b)
    cells: dict[str, InteractionCell] = {}
    low_power: list[str] = []
    for row_id in range(n_a):
        for col_id in range(n_b):
            mask = (ids_a == row_id) & (ids_b == col_id)
            values = aligned_out[mask]
            key = f"{row_id}x{col_id}"
            cells[key] = InteractionCell(
                n=int(values.size),
                mean_outcome=(
                    float(np.mean(values)) if values.size else float("nan")
                ),
                hit_rate=_hit_rate(values),
            )
            if values.size < min_cell_n:
                low_power.append(key)
    f_stat, p_f, present, status = _interaction_f_test(
        aligned_out,
        ids_a,
        ids_b,
        n_a,
        n_b,
        low_power=low_power,
        notes=notes,
    )
    if low_power and status == "OK":
        notes.append(
            f"{len(low_power)} of {len(cells)} cells below min_cell_n"
            f" ({min_cell_n}); verdict power-guarded"
        )
    return InteractionScanReport(
        label=label,
        feature_a=name_a,
        feature_b=name_b,
        n_cells=len(cells),
        cells=cells,
        low_power_cells=low_power,
        f_stat=f_stat,
        p_f=p_f,
        interaction_present=present,
        status=status,
        notes=notes,
    )


def _interaction_f_test(
    aligned_out: np.ndarray,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    n_a: int,
    n_b: int,
    *,
    low_power: list[str],
    notes: list[str],
) -> tuple[float, float, bool, str]:
    """Nested OLS F-test: additive bucket design vs full (with products).

    Returns ``(f_stat, p_f, present, status)``; every non-testable
    branch reads INCONCLUSIVE with a named note, and ``present``
    additionally demands no low-power cell.
    """
    n_rows = int(aligned_out.size)
    if n_a < 2 or n_b < 2:
        notes.append(
            "degenerate bucket split (fewer than 2 distinct buckets on"
            " one side); design cannot be built"
        )
        return float("nan"), float("nan"), False, "INCONCLUSIVE"
    if n_rows <= (n_a - 1) + (n_b - 1) + (n_a - 1) * (n_b - 1) + 1:
        notes.append(
            f"too few rows ({n_rows}) for the full design; F-test is"
            " NaN and must not be read as no-interaction"
        )
        return float("nan"), float("nan"), False, "INCONCLUSIVE"
    dummies_a = np.column_stack(
        [(ids_a == bucket_id).astype(float) for bucket_id in range(1, n_a)]
    )
    dummies_b = np.column_stack(
        [(ids_b == bucket_id).astype(float) for bucket_id in range(1, n_b)]
    )
    products = np.column_stack(
        [
            dummies_a[:, pos_a] * dummies_b[:, pos_b]
            for pos_a in range(dummies_a.shape[1])
            for pos_b in range(dummies_b.shape[1])
        ]
    )
    ones = np.ones(n_rows)
    additive = np.column_stack([ones, dummies_a, dummies_b])
    full = np.column_stack([ones, dummies_a, dummies_b, products])
    fit_additive = ols_hc0(aligned_out, additive)
    fit_full = ols_hc0(aligned_out, full)
    if fit_full["rank_deficient"] or fit_additive["rank_deficient"]:
        notes.append(
            "design rank-deficient; F-test is NaN and must not be"
            " read as no-interaction"
        )
        return float("nan"), float("nan"), False, "INCONCLUSIVE"
    rss_additive = float(
        np.sum((aligned_out - additive @ fit_additive["beta"]) ** 2)
    )
    rss_full = float(np.sum((aligned_out - full @ fit_full["beta"]) ** 2))
    df_full = n_rows - full.shape[1]
    df_diff = int(full.shape[1] - additive.shape[1])
    if df_full <= 0 or df_diff <= 0 or rss_full <= 0.0:
        notes.append("degenerate residual sums; F-test is NaN")
        return float("nan"), float("nan"), False, "INCONCLUSIVE"
    f_stat = ((rss_additive - rss_full) / df_diff) / (rss_full / df_full)
    p_f = float(stats.f.sf(f_stat, df_diff, df_full))
    present = bool(np.isfinite(p_f) and p_f < _ALPHA and not low_power)
    if low_power and np.isfinite(p_f) and p_f < _ALPHA:
        notes.append(
            "F significant but low_power_cells present;"
            " interaction_present stays False (power guard)"
        )
    return f_stat, p_f, present, "OK"


# ---------------------------------------------------------------------------
# 5. rule_fit
# ---------------------------------------------------------------------------
def rule_fit(
    outcome: Any,
    rules: dict[str, Any],
    *,
    train_idx: Any,
    test_idx: Any,
    min_n_train: int = 100,
    min_n_test: int = 50,
    label: str = "unnamed",
) -> RuleFitReport:
    """Evaluate caller-built rule masks on a frozen train/test split.

    For every rule (a boolean mask over the outcome positions) the
    kept-minus-dropped outcome mean is computed on the train indices
    and on the test indices.  ``status`` applies the sample-size
    guards first (``insufficient_n``), then reads the deltas: both
    positive -> ``valid``, anything else -> ``unstable`` (a rule that
    works on train but not on test lands here -- the honest overfit
    signature).  ``best_valid_rule`` names the valid rule with the
    largest ``delta_test``, or ``None`` when no rule is valid.

    Args:
        outcome: response series WITHOUT NaN (NaN rows would corrupt
            the positional mask semantics; align and drop first).
        rules: mapping rule name -> boolean mask (Series or array)
            over the outcome positions.
        train_idx: positional indices of the train split.
        test_idx: positional indices of the test split; must not
            overlap ``train_idx``.
        min_n_train: minimum kept rows on train for ``valid``.
        min_n_test: minimum kept rows on test for ``valid``.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`RuleFitReport` with one entry per rule.

    Raises:
        ValueError: if ``outcome`` contains NaN, the splits are empty
            or overlapping, an index is out of range, a mask's length
            differs from the outcome, or ``rules`` is empty.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> rows = np.arange(300)
        >>> out = pd.Series(
        ...     np.where(rows % 2 == 0, -1.0, 1.0)
        ...     + 0.1 * rng.standard_normal(300)
        ... )
        >>> rule = rows % 2 == 0
        >>> report = rule_fit(
        ...     out, {"parity": rule},
        ...     train_idx=np.arange(200), test_idx=np.arange(200, 300),
        ...     min_n_train=10, min_n_test=5,
        ... )
        >>> report["rules"]["parity"]["status"]
        'unstable'
    """
    validate_int("min_n_train", min_n_train, minimum=1)
    validate_int("min_n_test", min_n_test, minimum=1)
    if not rules:
        raise ValueError("rules dict must not be empty")
    outcome_series = _to_series(outcome, "outcome")
    if bool(outcome_series.isna().any()):
        raise ValueError(
            "outcome contains NaN; align and drop rows before rule_fit"
        )
    outcome_values = outcome_series.to_numpy(dtype=float)
    train_positions = np.asarray(train_idx, dtype=int).reshape(-1)
    test_positions = np.asarray(test_idx, dtype=int).reshape(-1)
    if train_positions.size == 0 or test_positions.size == 0:
        raise ValueError("train_idx and test_idx must both be non-empty")
    if set(train_positions.tolist()) & set(test_positions.tolist()):
        raise ValueError("train_idx and test_idx overlap")
    total = outcome_values.size
    splits = (
        ("train_idx", train_positions),
        ("test_idx", test_positions),
    )
    for tag, positions in splits:
        if positions.min() < 0 or positions.max() >= total:
            raise ValueError(
                f"{tag} out of range for outcome of {total} rows"
            )
    entries: dict[str, RuleFitEntry] = {}
    notes: list[str] = []
    for name, mask_value in rules.items():
        entries[name] = _rule_entry(
            name,
            mask_value,
            outcome_values,
            train_positions,
            test_positions,
            min_n_train=min_n_train,
            min_n_test=min_n_test,
            notes=notes,
        )
    valid_names = [
        name
        for name, entry in entries.items()
        if entry["status"] == _STATUS_VALID
    ]
    best_valid: str | None = None
    if valid_names:
        best_valid = max(
            valid_names, key=lambda name: entries[name]["delta_test"]
        )
    return RuleFitReport(
        label=label,
        rules=entries,
        best_valid_rule=best_valid,
        notes=notes,
    )


def _rule_entry(
    name: str,
    mask_value: Any,
    outcome_values: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    *,
    min_n_train: int,
    min_n_test: int,
    notes: list[str],
) -> RuleFitEntry:
    """Evaluate one rule mask on the frozen split (see :func:`rule_fit`).

    Empty split sides read NaN deltas with an explicit note; the
    status ladder is n-guard first, then the both-deltas-positive
    stability verdict.
    """
    total = outcome_values.size
    mask = np.asarray(mask_value, dtype=bool).reshape(-1)
    if mask.size != total:
        raise ValueError(
            f"rule {name!r}: mask length {mask.size} != outcome"
            f" length {total}"
        )
    kept_train = outcome_values[train_positions][mask[train_positions]]
    drop_train = outcome_values[train_positions][~mask[train_positions]]
    kept_test = outcome_values[test_positions][mask[test_positions]]
    drop_test = outcome_values[test_positions][~mask[test_positions]]
    delta_train = (
        float(np.mean(kept_train) - np.mean(drop_train))
        if kept_train.size and drop_train.size
        else float("nan")
    )
    delta_test = (
        float(np.mean(kept_test) - np.mean(drop_test))
        if kept_test.size and drop_test.size
        else float("nan")
    )
    n_train_kept = int(kept_train.size)
    n_test_kept = int(kept_test.size)
    stable = bool(
        np.isfinite(delta_train)
        and np.isfinite(delta_test)
        and delta_train > 0.0
        and delta_test > 0.0
    )
    if not (np.isfinite(delta_train) and np.isfinite(delta_test)):
        status = _STATUS_INSUFFICIENT
        notes.append(
            f"rule {name!r}: one split side is empty; deltas are NaN"
        )
    elif n_train_kept < min_n_train or n_test_kept < min_n_test:
        status = _STATUS_INSUFFICIENT
    elif stable:
        status = _STATUS_VALID
    else:
        status = _STATUS_UNSTABLE
    return RuleFitEntry(
        n_train_kept=n_train_kept,
        n_test_kept=n_test_kept,
        delta_train=delta_train,
        delta_test=delta_test,
        stable=stable,
        status=status,
    )


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _classify_features(
    orth: OrthogonalReport,
    buckets: BucketScanReport,
    clustering: ClusterReport,
    rep_names: list[str],
    notes: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Cells, actions and skip reasons for the representative features.

    Ladder: not testable -> NOISE; orthogonal fail (gate or
    collinearity screen) -> PROXY; pass + monotone ->
    STRONG_UNVERIFIED (checkpoint A2 upgrades this cell through the
    regime gate); pass + non-monotone -> SHAPE_UNKNOWN.  Cluster
    members are pointed at their representative in ``skipped_features``
    and get no independent cell.
    """
    classification: dict[str, str] = {}
    actions: dict[str, str] = {}
    skipped: dict[str, str] = {}
    orth_kept = set(orth["features"])
    detected = orth["per_feature_detected"]
    for name in rep_names:
        if name not in orth_kept:
            skipped[name] = "noise"
            classification[name] = _CELL_NOISE
            actions[name] = _ACTION_DISCARD
        elif not detected.get(name, False):
            # covers both gate failures and the collinearity screen:
            # either way the column carries nothing beyond controls
            skipped[name] = "proxy"
            classification[name] = _CELL_PROXY
            actions[name] = _ACTION_DISCARD
        elif name not in buckets["features"]:
            skipped[name] = "not_bucketed"
            classification[name] = _CELL_SHAPE
            actions[name] = _ACTION_INTERACT
        elif buckets["features"][name]["monotone"]:
            classification[name] = _CELL_UNVERIFIED
            actions[name] = _ACTION_PREREG
        else:
            classification[name] = _CELL_SHAPE
            actions[name] = _ACTION_INTERACT
    for member, chosen in clustering["representatives"].items():
        if member != chosen:
            skipped[member] = f"cluster:{chosen}"
    notes.append(
        "classification covers representatives only; cluster members"
        " are pointed at their representative in skipped_features"
    )
    return classification, actions, skipped


def feature_analysis(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
    *,
    nan_threshold: float = 0.05,
    cluster_threshold: float = 0.8,
    representatives: dict[str, str] | None = None,
    orthogonal_threshold: float | None = None,
    n_perm_orth: int = 200,
    n_buckets: int = 5,
    bucket_edges: dict[str, list[float]] | None = None,
    min_bucket_n: int = 50,
    norm_r: Any | None = None,
    n_perm_bucket: int = 200,
    interaction_pairs: list[tuple[str, str]] | None = None,
    min_cell_n: int = 30,
    candidate_rules: dict[str, Any] | None = None,
    train_idx: Any | None = None,
    test_idx: Any | None = None,
    seed: int = 42,
    label: str = "unnamed",
) -> FeatureAnalysisReport:
    """Chain the scans, then classify every representative feature.

    Order (each stage feeds the next): missingness flags ->
    correlation clusters (Bonferroni family = ``n_effective``) ->
    orthogonal check on the REPRESENTATIVES with the derived t-gate
    (``alpha / n_effective`` two-sided normal quantile unless
    ``orthogonal_threshold`` overrides it) -> bucket scan on the
    orthogonally detected features -> interaction scans for the
    requested pairs -> rule fit when candidate rules AND a frozen
    split are supplied -> classification -> recommended actions.

    Classification (per representative): not testable -> ``NOISE``;
    orthogonal gate fail -> ``PROXY``; pass + monotone ->
    ``STRONG_UNVERIFIED`` (no verified regime indicator in this
    checkpoint); pass + non-monotone -> ``SHAPE_UNKNOWN``.  Cluster
    members get no independent cell: ``skipped_features`` points them
    at their representative.  Optional blocks (regime, threshold,
    discovery, importance, rule generation) land in checkpoints A2/A3
    and read ``None`` until then.

    Args:
        outcome: response series.
        features: mapping feature name -> single column.
        controls: mapping control name -> single column.
        nan_threshold: missingness warn gate; in the open (0, 1).
        cluster_threshold: cluster cut level of |pearson|.
        representatives: optional explicit member -> representative
            map (see :func:`correlation_cluster`).
        orthogonal_threshold: explicit |t| gate for the orthogonal
            check; ``None`` derives ``alpha / n_effective``.
        n_perm_orth: permutation draws for the orthogonal check.
        n_buckets: requested bucket count for :func:`bucket_scan`.
        bucket_edges: optional per-feature prereg bucket edges.
        min_bucket_n: power floor per bucket (0 disables).
        norm_r: optional normalized-outcome column (see
            :func:`bucket_scan`).
        n_perm_bucket: permutation draws per bucket feature.
        interaction_pairs: optional list of ``(name_a, name_b)``
            pairs to test; both names must be declared features.
        min_cell_n: power floor per interaction cell.
        candidate_rules: optional rule masks for :func:`rule_fit`;
            requires ``train_idx`` AND ``test_idx``.
        train_idx: positional train split for ``candidate_rules``.
        test_idx: positional test split for ``candidate_rules``.
        seed: non-negative seed threaded into every seeded step.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`FeatureAnalysisReport`; every schema key is present.

    Raises:
        ValueError: if gate fractions are outside (0, 1), a pair name
            is not a declared feature, ``candidate_rules`` is given
            without both split indices, or any sub-scan raises.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(500)
        >>> feat = pd.Series(rng.standard_normal(500), index=idx)
        >>> ctrl = pd.Series(rng.standard_normal(500), index=idx)
        >>> out = pd.Series(0.5 * feat + rng.standard_normal(500),
        ...                 index=idx)
        >>> report = feature_analysis(
        ...     out, {"feat": feat}, {"ctrl": ctrl},
        ...     n_perm_orth=25, n_perm_bucket=25, min_bucket_n=0,
        ...     label="demo",
        ... )
        >>> report["classification"]["feat"]
        'STRONG_UNVERIFIED'
    """
    _validate_fraction("nan_threshold", nan_threshold)
    _validate_fraction("cluster_threshold", cluster_threshold)
    validate_int("seed", seed, minimum=0)
    notes: list[str] = []
    if (candidate_rules is not None) != (train_idx is not None) or (
        candidate_rules is not None
    ) != (test_idx is not None):
        raise ValueError(
            "candidate_rules requires BOTH train_idx and test_idx"
            " (rule_fit has no default split)"
        )
    feature_names = list(features)

    missing = missingness_scan(
        outcome, features, controls, nan_threshold=nan_threshold, label=label
    )
    clustering = correlation_cluster(
        features,
        outcome,
        threshold=cluster_threshold,
        representatives=representatives,
        label=label,
    )
    rep_features = {
        name: features[name]
        for name in feature_names
        if clustering["representatives"].get(name) == name
    }
    n_effective = clustering["n_effective"]
    if orthogonal_threshold is None:
        alpha_family = _ALPHA / n_effective
        orth_gate = float(norm.isf(alpha_family / 2.0))
        notes.append(
            f"orthogonal t-gate derived from alpha/{n_effective}:"
            f" |t| >= {orth_gate:.3f}"
        )
    else:
        if not np.isfinite(orthogonal_threshold) or orthogonal_threshold <= 0:
            raise ValueError(
                "orthogonal_threshold must be a positive finite float,"
                f" got {orthogonal_threshold!r}"
            )
        orth_gate = float(orthogonal_threshold)
    orth = orthogonal_check(
        outcome,
        rep_features,
        controls,
        label=label,
        n_perm=n_perm_orth,
        seed=seed,
        t_threshold=orth_gate,
    )
    detected = orth["per_feature_detected"]
    bucket_targets = [name for name in rep_features if detected.get(name)]
    bucket_edges_kept = {
        name: edges
        for name, edges in (bucket_edges or {}).items()
        if name in bucket_targets
    }
    buckets = bucket_scan(
        outcome,
        {name: features[name] for name in bucket_targets},
        n_buckets=n_buckets,
        bucket_edges=bucket_edges_kept or None,
        min_bucket_n=min_bucket_n,
        norm_r=norm_r,
        n_perm=n_perm_bucket,
        seed=seed,
        label=label,
    )
    interactions: dict[str, InteractionScanReport] = {}
    for first_name, second_name in interaction_pairs or []:
        if first_name not in features or second_name not in features:
            raise ValueError(
                f"interaction pair ({first_name!r}, {second_name!r}):"
                " both names must be declared features"
            )
        if first_name == second_name:
            raise ValueError(
                f"interaction pair ({first_name!r}, {second_name!r}):"
                " a feature cannot interact with itself"
            )
        pair_key = f"{first_name}x{second_name}"
        interactions[pair_key] = interaction_scan(
            outcome,
            features[first_name],
            features[second_name],
            min_cell_n=min_cell_n,
            name_a=first_name,
            name_b=second_name,
            label=pair_key,
        )
    rules = None
    if candidate_rules is not None:
        rules = rule_fit(
            outcome,
            candidate_rules,
            train_idx=train_idx,
            test_idx=test_idx,
            label=label,
        )

    classification, actions, skipped = _classify_features(
        orth, buckets, clustering, list(rep_features), notes
    )
    notes.append(
        "checkpoint-A2/A3 blocks (regime, threshold, discovery,"
        " importance, rule generation) not requested; keys stay None"
    )
    return FeatureAnalysisReport(
        label=label,
        seed=seed,
        n_effective=n_effective,
        orthogonal_t_gate=orth_gate,
        missingness=missing,
        clustering=clustering,
        orthogonal=orth,
        buckets=buckets,
        interaction=interactions,
        rules=rules,
        regime=None,
        threshold=None,
        discovery=None,
        importance=None,
        rule_generation=None,
        skipped_features=skipped,
        classification=classification,
        recommended_actions=actions,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: one planted world through the whole A1 chain, asserted.

    Every printed verdict is asserted, so a regression crashes the
    demo instead of printing a wrong number quietly.
    """
    n_rows = 800
    rng = np.random.default_rng(42)
    real = rng.standard_normal(n_rows)
    null = rng.standard_normal(n_rows)
    control = rng.standard_normal(n_rows)
    outcome_values = 0.5 * real + 0.4 * control + rng.standard_normal(n_rows)
    idx = pd.RangeIndex(n_rows)
    outcome = pd.Series(outcome_values, index=idx)
    features = {
        "alpha_real": pd.Series(real, index=idx),
        "beta_null": pd.Series(null, index=idx),
    }
    controls = {"gamma_mom": pd.Series(control, index=idx)}

    print("=" * 72)
    print("feature_scan demo: planted world (alpha real, beta null)")
    print("=" * 72)
    report = feature_analysis(
        outcome,
        features,
        controls,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="feature_scan demo",
    )
    print(f"n_effective={report['n_effective']}")
    print(
        "orthogonal t-gate:",
        round(report["orthogonal_t_gate"], 3),
    )
    print("classification:", report["classification"])
    print("actions:", report["recommended_actions"])
    print("skipped:", report["skipped_features"])
    assert report["n_effective"] == 2
    assert report["classification"]["alpha_real"] == _CELL_UNVERIFIED
    assert report["classification"]["beta_null"] == _CELL_PROXY
    assert report["recommended_actions"]["alpha_real"] == _ACTION_PREREG
    assert report["rules"] is None and report["regime"] is None

    missing = report["missingness"]
    assert sorted(missing["features"]) == ["alpha_real", "beta_null"]
    assert missing["features"]["alpha_real"]["warn"] is False

    bucket_report = bucket_scan(
        outcome,
        {"alpha_real": features["alpha_real"]},
        n_perm=25,
        seed=42,
    )
    profile = bucket_report["features"]["alpha_real"]
    print(
        "bucket profile: monotone=",
        profile["monotone"],
        " spearman=",
        round(profile["spearman"], 3),
        " p_perm=",
        profile["p_perm"],
    )
    assert profile["monotone"] is True
    assert profile["p_perm"] <= 0.05
    assert profile["q_split"]["net_r_signal"] is True
    assert bucket_report["norm_r_is_outcome"] is True

    interaction = interaction_scan(
        outcome,
        features["alpha_real"],
        features["beta_null"],
        min_cell_n=30,
    )
    print(
        "interaction (additive world): present=",
        interaction["interaction_present"],
        " p_f=",
        round(interaction["p_f"], 3),
    )
    assert interaction["status"] == "OK"
    assert interaction["interaction_present"] is False

    rule_mask = (real > 0.0).astype(bool)
    rule_report = rule_fit(
        outcome,
        {"sign_rule": rule_mask},
        train_idx=np.arange(0, 500),
        test_idx=np.arange(500, n_rows),
        min_n_train=50,
        min_n_test=20,
    )
    print(
        "rule_fit sign_rule:",
        rule_report["rules"]["sign_rule"]["status"],
        " delta_test=",
        round(rule_report["rules"]["sign_rule"]["delta_test"], 4),
    )
    assert rule_report["rules"]["sign_rule"]["status"] == _STATUS_VALID
    assert rule_report["best_valid_rule"] == "sign_rule"

    nan_outcome = outcome.copy()
    nan_feature = features["beta_null"].copy()
    nan_feature[nan_outcome > 1.0] = np.nan
    missing_shift = missingness_scan(
        nan_outcome, {"flagged": nan_feature}, {}, nan_threshold=0.05
    )
    print(
        "missingness (outcome-linked NaN): warn=",
        missing_shift["features"]["flagged"]["warn"],
        " differs=",
        missing_shift["features"]["flagged"]["nan_outcome_differs"],
    )
    assert missing_shift["features"]["flagged"]["warn"] is True


if __name__ == "__main__":
    run_all_examples()
