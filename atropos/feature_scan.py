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
* :func:`regime_scan` -- per-regime feature profiles against a
  PREREGISTERED indicator: an unfrozen indicator reads
  ``REGIME_UNVERIFIED`` and can never upgrade a cell to STRONG.
* :func:`threshold_scan` -- a threshold sweep over a parsed (never
  evaluated) rule template, with the plateau-vs-edge reading that
  separates a wide ridge from a pick-one-number spike.
* :func:`auto_discovery` -- Bonferroni-gated sweep over an explicit
  candidate pool; the output is a hypothesis list, EXPLORATORY by
  construction, never a classification.
* :func:`feature_importance` -- permutation importance for one
  linear/tree model; a diagnostic that feeds no cell and rescues no
  PROXY.
* :func:`auto_rule_generation` -- the caller fixes the rule SHAPE,
  the tool grid-searches values on TRAIN only, and the untouched
  test split judges the winners; all-fail-on-test reads
  ``overfit_warning``.

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

Checkpoint note: this file landed in three verified checkpoints --
A1 (the five defensive/analysis functions + the orchestrator
skeleton), A2 (``regime_scan`` + ``threshold_scan``), A3
(``auto_discovery``, ``feature_importance``, ``auto_rule_generation``
-- all EXPLORATORY by construction, this state).  The report schema
was FROZEN at A1: every key above exists from the first commit.
If a future wave wants auto-discovery -> prereg path, that is a
separate protocol change, not a tool feature.

Run ``python -m atropos.feature_scan`` for a synthetic demo.
"""

from __future__ import annotations

import re
from itertools import product
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from atropos._utils import align_inputs, frameify, ols_hc0, validate_int
from atropos.data_contract import MaterializedData
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
    "auto_discovery",
    "auto_rule_generation",
    "bucket_scan",
    "correlation_cluster",
    "feature_analysis",
    "feature_importance",
    "interaction_scan",
    "missingness_scan",
    "regime_scan",
    "rule_fit",
    "threshold_scan",
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

_MAX_THRESHOLDS = 7
"""Prereg cap on a threshold sweep length (the pick-a-number guard)."""

_PLATEAU_SHARE = 0.7
"""Edge delta (vs the sweep max) that reads as plateau, not spike."""

_EDGE_SHARE = 0.3
"""Edge delta (vs the sweep max) below which the sweep reads edge."""

_MAX_COMBOS = 200
"""Grid-search cap for auto_rule_generation (exploration hygiene)."""

_DISCOVERY_NOTE = (
    "exploratory -- validation requires prereg + run_full_pipeline"
)
"""The one-line doctrine every discovery report must carry."""

_ALLOWED_OPS = ("<", "<=", ">", ">=")
"""The only comparison operators a rule template may use."""

_OP_FUNCS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


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
    return _to_raw_series(values, name).astype(float)


def _to_raw_series(values: Any, name: str) -> pd.Series:
    """One named Series view of any 1-D input, dtype preserved.

    Regime labels may be strings ("trend"/"chop") -- only the scans'
    numeric inputs go through the float cast in :func:`_to_series`.
    """
    frame = frameify(values, name)
    if frame.shape[1] != 1:
        raise ValueError(
            f"input {name!r} must be a single column, got"
            f" {frame.shape[1]} columns"
        )
    series = frame.iloc[:, 0]
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
# 6. regime_scan (checkpoint A2)
# ---------------------------------------------------------------------------
def regime_scan(
    outcome: Any,
    features: dict[str, Any],
    regime_indicator: dict[str, Any],
    *,
    min_regime_n: int = 100,
    label: str = "unnamed",
) -> RegimeScanReport:
    """Is a feature's effect regime-stable, or does the sign flip?

    Splits the aligned sample by the caller's regime indicator (a
    dict carrying ``name``, ``definition``, ``prereg_hash`` and the
    ``values`` Series) and profiles every feature inside every regime
    value: n, row-level Spearman, and the monotone verdict.  The
    indicator MUST be preregistered (``prereg_hash``): an unfrozen
    indicator reads ``REGIME_UNVERIFIED`` -- cells are computed but
    can never upgrade a classification to STRONG.  Regimes with fewer
    than ``min_regime_n`` rows are excluded from the sign comparison
    with a note.

    Args:
        outcome: response series.
        features: mapping feature name -> single column.
        regime_indicator: ``{name, definition, prereg_hash, values}``;
            ``values`` is the regime Series (labels may be numeric or
            string), index-aligned with the outcome.
        min_regime_n: minimum rows for a regime to qualify for the
            sign comparison; at least 1.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`RegimeScanReport`; ``same_sign[feature]`` is the
        pairwise sign test over qualifying regimes, ``strongest_regime``
        / ``weakest_regime`` name the regimes by |spearman|.

    Raises:
        ValueError: if the indicator dict misses a key, has an empty
            name/definition, ``min_regime_n`` < 1, or an input is not
            a single column.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(400)
        >>> feat = pd.Series(rng.standard_normal(400), index=idx)
        >>> out = pd.Series(2.0 * feat + rng.standard_normal(400),
        ...                 index=idx)
        >>> regime = pd.Series(np.where(rng.random(400) < 0.5, 0, 1),
        ...                    index=idx)
        >>> report = regime_scan(
        ...     out, {"feat": feat},
        ...     {"name": "coin", "definition": "independent coin",
        ...      "prereg_hash": "hash", "values": regime},
        ...     min_regime_n=50,
        ... )
        >>> report["same_sign"]["feat"]
        True
    """
    for key in ("name", "definition", "prereg_hash", "values"):
        if key not in regime_indicator:
            raise ValueError(
                f"regime_indicator must carry {key!r}; got keys"
                f" {sorted(regime_indicator)}"
            )
    if not regime_indicator["name"] or not regime_indicator["definition"]:
        raise ValueError(
            "regime_indicator name and definition must be non-empty"
        )
    validate_int("min_regime_n", min_regime_n, minimum=1)
    prereg_hash = regime_indicator["prereg_hash"]
    notes: list[str] = []
    if prereg_hash:
        status = "OK"
    else:
        status = "REGIME_UNVERIFIED"
        notes.append(
            "regime indicator carries no prereg_hash: status"
            " REGIME_UNVERIFIED -- cells can never upgrade a"
            " classification to STRONG"
        )
    outcome_series = _to_series(outcome, "outcome")
    values_series = _to_raw_series(regime_indicator["values"], "regime_value")
    entries: dict[str, dict[str, RegimeEntry]] = {}
    same_sign: dict[str, bool] = {}
    strongest_regime: dict[str, str] = {}
    weakest_regime: dict[str, str] = {}
    for name, feature_value in features.items():
        feature_series = _to_series(feature_value, name)
        frame = pd.concat(
            [outcome_series, feature_series, values_series],
            axis=1,
            join="inner",
        ).dropna()
        aligned_out = frame["outcome"].to_numpy(dtype=float)
        aligned_feat = frame[name].to_numpy(dtype=float)
        labels = frame["regime_value"].to_numpy()
        feature_entries: dict[str, RegimeEntry] = {}
        for label_value in sorted(set(labels.tolist()), key=str):
            mask = labels == label_value
            slice_rho, slice_p = _spearman_pair(
                aligned_feat[mask], aligned_out[mask]
            )
            feature_entries[str(label_value)] = RegimeEntry(
                n=int(mask.sum()),
                spearman=slice_rho,
                spearman_p=slice_p,
                monotone=bool(
                    np.isfinite(slice_rho)
                    and np.isfinite(slice_p)
                    and slice_p <= _SPEAR_P_GATE
                ),
            )
        qualifying = [
            regime_key
            for regime_key, entry in feature_entries.items()
            if entry["n"] >= min_regime_n
        ]
        excluded = [
            regime_key
            for regime_key, entry in feature_entries.items()
            if entry["n"] < min_regime_n
        ]
        if excluded:
            notes.append(
                f"{name}: regimes excluded from the sign comparison"
                f" (below min_regime_n = {min_regime_n}): {excluded}"
            )
        if len(qualifying) < 2:
            notes.append(
                f"{name}: fewer than 2 qualifying regimes; same_sign"
                " reads False (not testable, not a flip)"
            )
            same_sign[name] = False
            strongest_regime[name] = ""
            weakest_regime[name] = ""
        else:
            rhos = {
                regime_key: feature_entries[regime_key]["spearman"]
                for regime_key in qualifying
            }
            signs = {
                np.sign(value)
                for value in rhos.values()
                if np.isfinite(value)
            }
            same_sign[name] = bool(len(signs) == 1 and signs)
            strongest_regime[name] = max(
                qualifying, key=lambda key: abs(rhos[key])
            )
            weakest_regime[name] = min(
                qualifying, key=lambda key: abs(rhos[key])
            )
        entries[name] = feature_entries
    return RegimeScanReport(
        label=label,
        regime_name=str(regime_indicator["name"]),
        regime_definition=str(regime_indicator["definition"]),
        prereg_hash=prereg_hash if prereg_hash else None,
        status=status,
        min_regime_n=min_regime_n,
        features=entries,
        same_sign=same_sign,
        strongest_regime=strongest_regime,
        weakest_regime=weakest_regime,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 7. threshold_scan (checkpoint A2)
# ---------------------------------------------------------------------------
def _parse_rule_template(
    rule_template: str, feature_name: str
) -> tuple[str, Any]:
    """Parse ``"<feature> <op> {t}"`` without ever evaluating a string.

    Returns the comparison operator's numpy function and the rule's
    feature token.  Every deviation (missing ``{t}``, unknown
    operator, feature token mismatch) raises ValueError -- a template
    is parsed, never ``eval``-ed.
    """
    if "{t}" not in rule_template:
        raise ValueError(
            f"rule_template must contain the {{t}} placeholder:"
            f" {rule_template!r}"
        )
    prefix_tokens = rule_template.partition("{t}")[0].split()
    if len(prefix_tokens) < 2:
        raise ValueError(
            "rule_template must read '<feature> <op> {t}':"
            f" {rule_template!r}"
        )
    op = prefix_tokens[-1]
    feature_token = prefix_tokens[-2]
    if op not in _ALLOWED_OPS:
        raise ValueError(
            f"rule_template operator must be one of {list(_ALLOWED_OPS)},"
            f" got {op!r}"
        )
    if feature_token != feature_name:
        raise ValueError(
            f"rule_template names feature {feature_token!r} but the"
            f" supplied feature is {feature_name!r}"
        )
    return op, _OP_FUNCS[op]


def threshold_scan(
    outcome: Any,
    feature: Any,
    rule_template: str,
    threshold_values: list[float],
    *,
    feature_name: str,
    train_idx: Any,
    test_idx: Any,
    min_n_kept: int = 100,
    label: str = "unnamed",
) -> ThresholdScanReport:
    """Sweep one threshold over a fixed rule shape, train then test.

    The template pins the shape (``"<feature> <op> {t}"``, parsed,
    never evaluated); each threshold builds the rule mask on the
    aligned sample and reuses the :func:`rule_fit` ladder
    (``min_n_kept`` floors both splits).  The sweep-level reading:
    ``plateau_detected`` when both edge deltas hold
    :data:`_PLATEAU_SHARE` of the best ``delta_test`` (a wide ridge,
    safe to ship), ``edge_detected`` when an edge falls below
    :data:`_EDGE_SHARE` of it (a spike -- the pick-one-number
    overfit signature), and ``optimal_range`` the contiguous
    >= share run around the argmax.

    Args:
        outcome: response series WITHOUT NaN (aligned with feature).
        feature: the feature column the template compares against.
        rule_template: format string ``"<feature> <op> {t}"``.
        threshold_values: the preregistered thresholds; at most
            :data:`_MAX_THRESHOLDS`, at least 2.
        feature_name: the feature's declared name (validated against
            the template's feature token).
        train_idx: positional train split.
        test_idx: positional test split (must not overlap train).
        min_n_kept: minimum kept rows per split for ``valid``.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`ThresholdScanReport` with one point per threshold.

    Raises:
        ValueError: on template deviations (see
            :func:`_parse_rule_template`), a sweep outside
            [2, _MAX_THRESHOLDS], NaN outcome, or bad splits (the
            :func:`rule_fit` guards).

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> rows = np.arange(600)
        >>> feat = pd.Series(rng.uniform(0, 1, 600), index=rows)
        >>> out = pd.Series(
        ...     np.where(feat < 0.65, 1.0, 0.0)
        ...     + 0.1 * rng.standard_normal(600), index=rows
        ... )
        >>> report = threshold_scan(
        ...     out, feat, "feat < {t}",
        ...     [0.7, 0.75, 0.8, 0.85, 0.9],
        ...     feature_name="feat",
        ...     train_idx=np.arange(400), test_idx=np.arange(400, 600),
        ...     min_n_kept=30,
        ... )
        >>> report["plateau_detected"], report["edge_detected"]
        (True, False)
    """
    op, op_func = _parse_rule_template(rule_template, feature_name)
    if len(threshold_values) > _MAX_THRESHOLDS:
        raise ValueError(
            f"threshold sweep length {len(threshold_values)} exceeds"
            f" the prereg cap {_MAX_THRESHOLDS}; freeze fewer values"
        )
    if len(threshold_values) < 2:
        raise ValueError(
            "threshold sweep needs at least 2 values to read plateau/edge"
        )
    validate_int("min_n_kept", min_n_kept, minimum=1)
    notes: list[str] = []
    outcome_series = _to_series(outcome, "outcome")
    feature_series = _to_series(feature, feature_name)
    frame = pd.concat(
        [outcome_series, feature_series], axis=1, join="inner"
    ).dropna()
    aligned_out = frame["outcome"].to_numpy(dtype=float)
    aligned_feat = frame[feature_name].to_numpy(dtype=float)
    train_positions = np.asarray(train_idx, dtype=int).reshape(-1)
    test_positions = np.asarray(test_idx, dtype=int).reshape(-1)
    points: list[ThresholdPoint] = []
    for value in threshold_values:
        mask = op_func(aligned_feat, float(value))
        rule_str = rule_template.format(t=repr(float(value)))
        entry = _rule_entry(
            rule_str,
            mask,
            aligned_out,
            train_positions,
            test_positions,
            min_n_train=min_n_kept,
            min_n_test=min_n_kept,
            notes=notes,
        )
        points.append(
            ThresholdPoint(
                threshold=float(value),
                rule_str=rule_str,
                n_train_kept=entry["n_train_kept"],
                n_test_kept=entry["n_test_kept"],
                delta_train=entry["delta_train"],
                delta_test=entry["delta_test"],
                status=entry["status"],
            )
        )
    finite_deltas = [
        point["delta_test"]
        for point in points
        if np.isfinite(point["delta_test"])
    ]
    plateau = False
    edge = False
    optimal_range: list[float] | None = None
    if len(finite_deltas) < 2:
        notes.append(
            "fewer than 2 points have a finite delta_test;"
            " plateau/edge read False (not testable)"
        )
    else:
        max_delta = max(finite_deltas)
        if max_delta <= 0:
            notes.append(
                "no positive delta_test on the sweep; plateau/edge"
                " read False (the rule shape itself fails on test)"
            )
        else:
            first_delta = finite_deltas[0]
            last_delta = finite_deltas[-1]
            plateau = bool(
                first_delta >= _PLATEAU_SHARE * max_delta
                and last_delta >= _PLATEAU_SHARE * max_delta
            )
            edge = bool(
                first_delta < _EDGE_SHARE * max_delta
                or last_delta < _EDGE_SHARE * max_delta
            )
            best_pos = finite_deltas.index(max_delta)
            low_pos = best_pos
            while (
                low_pos > 0
                and finite_deltas[low_pos - 1] >= _PLATEAU_SHARE * max_delta
            ):
                low_pos -= 1
            high_pos = best_pos
            while (
                high_pos < len(finite_deltas) - 1
                and finite_deltas[high_pos + 1] >= _PLATEAU_SHARE * max_delta
            ):
                high_pos += 1
            finite_points = [
                point for point in points if np.isfinite(point["delta_test"])
            ]
            optimal_range = [
                finite_points[low_pos]["threshold"],
                finite_points[high_pos]["threshold"],
            ]
    return ThresholdScanReport(
        label=label,
        feature=feature_name,
        op=op,
        rule_template=rule_template,
        points=points,
        plateau_detected=plateau,
        edge_detected=edge,
        optimal_range=optimal_range,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 8. auto_discovery (checkpoint A3; EXPLORATORY by construction)
# ---------------------------------------------------------------------------
def auto_discovery(
    outcome: Any,
    candidate_pool: dict[str, Any],
    controls: dict[str, Any],
    *,
    max_features: int = 20,
    min_abs_spearman: float = 0.15,
    orthogonal_threshold: float | None = None,
    label: str = "unnamed",
) -> DiscoveryReport:
    """Sweep a candidate pool for effects; the output is a HYPOTHESIS list.

    Every candidate that clears the Bonferroni t-gate (derived from
    the number of TESTED candidates: ``alpha / n_tested`` two-sided
    normal quantile) and a light spearman pre-filter is listed with
    its t, its monotone flag and its Bonferroni-adjusted p.  The
    report is EXPLORATORY by construction: it never assigns cells,
    never names a best feature, and its doctrine note requires
    preregistration + the full pipeline before any candidate can be
    believed.  Candidates whose |spearman| against the outcome is
    below ``min_abs_spearman`` are not tested (and shrink the
    Bonferroni family accordingly, with a note).

    Args:
        outcome: response series.
        candidate_pool: EXPLICIT mapping candidate name -> single
            column (an auto-scrape of a frame is deliberately not
            supported; the caller owns the pool).
        controls: mapping control name -> single column (same rules).
        max_features: pool-size cap (exploration hygiene); a larger
            pool raises.
        min_abs_spearman: pre-filter floor on |spearman| vs the
            outcome; 0 disables the filter.
        orthogonal_threshold: explicit |t| gate; ``None`` derives the
            Bonferroni gate from ``alpha / n_tested``.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`DiscoveryReport` with ``note`` fixed to
        :data:`_DISCOVERY_NOTE`.

    Raises:
        ValueError: if the pool exceeds ``max_features``, is empty,
            ``min_abs_spearman`` is negative, a control collides with
            a candidate name, or an input is not a single column.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(500)
        >>> planted = pd.Series(rng.standard_normal(500), index=idx)
        >>> null = pd.Series(rng.standard_normal(500), index=idx)
        >>> out = pd.Series(1.5 * planted + rng.standard_normal(500),
        ...                 index=idx)
        >>> report = auto_discovery(
        ...     out, {"planted": planted, "null": null}, {},
        ...     max_features=5, min_abs_spearman=0.0,
        ... )
        >>> [entry["feature"] for entry in report["discovered"]]
        ['planted']
        >>> report["all_below_threshold"]
        False
    """
    if not candidate_pool:
        raise ValueError("candidate_pool must not be empty")
    if len(candidate_pool) > max_features:
        raise ValueError(
            f"candidate pool size {len(candidate_pool)} exceeds"
            f" max_features ({max_features}); the caller must shrink"
            " the pool explicitly"
        )
    if min_abs_spearman < 0.0 or not np.isfinite(min_abs_spearman):
        raise ValueError(
            f"min_abs_spearman must be a finite float >= 0, got"
            f" {min_abs_spearman!r}"
        )
    candidate_names = list(candidate_pool)
    control_names = list(controls)
    overlap = sorted(set(candidate_names) & set(control_names))
    if overlap:
        raise ValueError(f"candidate and control names collide: {overlap}")
    outcome_series = _to_series(outcome, "outcome")
    control_series = [
        _to_series(controls[name], name) for name in control_names
    ]
    notes: list[str] = []
    tested: dict[str, pd.Series] = {}
    for name in candidate_names:
        candidate_series = _to_series(candidate_pool[name], name)
        aligned_out, aligned_feat = align_inputs(
            outcome_series, candidate_series
        )
        rho, rho_p = _spearman_pair(aligned_feat, aligned_out)
        if (
            min_abs_spearman > 0.0
            and np.isfinite(rho)
            and abs(rho) < min_abs_spearman
        ):
            notes.append(
                f"{name}: below the spearman floor"
                f" (|rho| = {abs(rho):.3f} < {min_abs_spearman}); not"
                " tested (shrinks the Bonferroni family)"
            )
            continue
        tested[name] = candidate_series
    n_tested = len(tested)
    if n_tested == 0:
        return DiscoveryReport(
            label=label,
            n_candidates=0,
            t_gate=float("nan"),
            discovered=[],
            all_below_threshold=True,
            note=_DISCOVERY_NOTE,
            notes=[
                *notes,
                "every candidate fell below the spearman floor",
            ],
        )
    if orthogonal_threshold is None:
        t_gate = float(norm.isf(_ALPHA / n_tested / 2.0))
        notes.append(
            f"Bonferroni t-gate from alpha/{n_tested}: |t| >= {t_gate:.3f}"
        )
    else:
        if not np.isfinite(orthogonal_threshold) or (
            orthogonal_threshold <= 0
        ):
            raise ValueError(
                "orthogonal_threshold must be a positive finite float,"
                f" got {orthogonal_threshold!r}"
            )
        t_gate = float(orthogonal_threshold)
    discovered: list[DiscoveryEntry] = []
    for name, candidate_series in tested.items():
        aligned = align_inputs(
            outcome_series, candidate_series, *control_series
        )
        aligned_out = aligned[0]
        aligned_feat = aligned[1]
        design = np.column_stack([np.ones(aligned_out.size), *aligned[1:]])
        fit = ols_hc0(aligned_out, design)
        t_stat = float(fit["t"][1])
        rho, rho_p = _spearman_pair(aligned_feat, aligned_out)
        monotone = bool(
            np.isfinite(rho) and np.isfinite(rho_p) and rho_p <= _SPEAR_P_GATE
        )
        if np.isfinite(t_stat) and abs(t_stat) >= t_gate:
            p_two_sided = (
                2.0 * float(stats.t.sf(abs(t_stat), fit["df"]))
                if np.isfinite(t_stat)
                else 1.0
            )
            discovered.append(
                DiscoveryEntry(
                    feature=name,
                    t_stat=t_stat,
                    monotone=monotone,
                    p_bonferroni=float(min(1.0, p_two_sided * n_tested)),
                )
            )
    return DiscoveryReport(
        label=label,
        n_candidates=n_tested,
        t_gate=t_gate,
        discovered=discovered,
        all_below_threshold=len(discovered) == 0,
        note=_DISCOVERY_NOTE,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 9. feature_importance (checkpoint A3; diagnostics only)
# ---------------------------------------------------------------------------
def feature_importance(
    outcome: Any,
    features: dict[str, Any],
    *,
    model: str = "linear",
    n_permutations: int = 100,
    seed: int = 42,
    label: str = "unnamed",
) -> ImportanceReport:
    """Permutation importance of each feature for ONE fitted model.

    ``model="linear"`` fits OLS on standardized features;
    ``model="tree"`` fits a seeded RandomForest (200 trees, depth 4).
    Importance = sklearn permutation importance against R2 with
    ``n_repeats=n_permutations``; ``ci_lo`` / ``ci_hi`` are
    ``mean +/- 1.96 * std`` of the permutation repeats.  This is a
    DIAGNOSTIC: it feeds no classification cell, names no top
    feature, and a high importance never rescues a PROXY verdict.

    Args:
        outcome: response series.
        features: mapping feature name -> single column.
        model: ``"linear"`` or ``"tree"``.
        n_permutations: permutation repeats; at least 1.
        seed: non-negative seed for the permutation draws (and the
            forest's random_state).
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`ImportanceReport` with one entry per feature.

    Raises:
        ValueError: if ``model`` is unknown, ``n_permutations`` < 1,
            ``seed`` is negative, ``features`` is empty, or fewer
            rows than features + 1 survive alignment.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(400)
        >>> feat_a = pd.Series(rng.standard_normal(400), index=idx)
        >>> feat_b = pd.Series(rng.standard_normal(400), index=idx)
        >>> out = pd.Series(2.0 * feat_a + rng.standard_normal(400),
        ...                 index=idx)
        >>> report = feature_importance(
        ...     out, {"a": feat_a, "b": feat_b}, n_permutations=20
        ... )
        >>> report["features"]["a"]["importance_mean"] > (
        ...     report["features"]["b"]["importance_mean"]
        ... )
        True
    """
    if model not in ("linear", "tree"):
        raise ValueError(f"model must be 'linear' or 'tree', got {model!r}")
    validate_int("n_permutations", n_permutations)
    validate_int("seed", seed, minimum=0)
    if not features:
        raise ValueError("features dict must not be empty")
    names = list(features)
    aligned = align_inputs(
        _to_series(outcome, "outcome"),
        *(_to_series(features[name], name) for name in names),
    )
    outcome_values = aligned[0]
    design = np.column_stack(aligned[1:])
    n_rows = int(outcome_values.size)
    if n_rows <= len(names) + 1:
        raise ValueError(
            f"too few rows ({n_rows}) for {len(names)} features;"
            " importance is undefined"
        )
    notes: list[str] = []
    means = design.mean(axis=0)
    stds = design.std(axis=0)
    stds[stds == 0.0] = 1.0
    design_std = (design - means) / stds
    if model == "linear":
        fitted = LinearRegression().fit(design_std, outcome_values)
    else:
        fitted = RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=seed, n_jobs=1
        ).fit(design_std, outcome_values)
    result = permutation_importance(
        fitted,
        design_std,
        outcome_values,
        n_repeats=n_permutations,
        random_state=seed,
        scoring="r2",
    )
    entries: dict[str, ImportanceEntry] = {}
    for position, name in enumerate(names):
        mean_value = float(result.importances_mean[position])
        std_value = float(result.importances_std[position])
        entries[name] = ImportanceEntry(
            importance_mean=mean_value,
            importance_std=std_value,
            ci_lo=mean_value - 1.96 * std_value,
            ci_hi=mean_value + 1.96 * std_value,
        )
    notes.append(
        "diagnostic only: importance feeds no classification cell"
        " (see the module docstring)"
    )
    return ImportanceReport(
        label=label,
        model=model,
        n_permutations=n_permutations,
        features=entries,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 10. auto_rule_generation (checkpoint A3; EXPLORATORY by construction)
# ---------------------------------------------------------------------------
def _parse_multi_rule_template(
    rule_template: str,
    feature_names: list[str],
    max_depth: int,
) -> list[tuple[str, Any, str]]:
    """Parse an AND-of-comparisons template, clause by clause.

    Returns one ``(placeholder, op_func, feature_name)`` triple per
    clause (template order).  Every clause must read
    ``"<feature> <op> {tN}"`` with its own placeholder; unknown
    features, unknown operators, duplicate placeholders and depths
    beyond ``max_depth`` all raise ValueError.  Templates are
    parsed, never ``eval``-ed.
    """
    clauses = [clause.strip() for clause in rule_template.split("&")]
    if not clauses or any(not clause for clause in clauses):
        raise ValueError(
            f"rule template must be non-empty AND-clauses: {rule_template!r}"
        )
    if len(clauses) > max_depth:
        raise ValueError(
            f"rule template has {len(clauses)} clauses, above the"
            f" max_depth {max_depth}"
        )
    placeholders: list[str] = []
    parsed: list[tuple[str, Any, str]] = []
    for clause in clauses:
        tokens = clause.split()
        if len(tokens) != 3:
            raise ValueError(
                f"clause {clause!r} must read '<feature> <op> {{tN}}'"
            )
        feature_token, op, placeholder_token = tokens
        if op not in _ALLOWED_OPS:
            raise ValueError(
                f"clause {clause!r}: operator must be one of"
                f" {list(_ALLOWED_OPS)}, got {op!r}"
            )
        if feature_token not in feature_names:
            raise ValueError(
                f"clause {clause!r}: feature {feature_token!r} is not"
                " a declared feature"
            )
        match = re.fullmatch(r"\{(t\d*)\}", placeholder_token)
        if match is None:
            raise ValueError(
                f"clause {clause!r}: placeholder must look like"
                " {{tN}}, got"
                f" {placeholder_token!r}"
            )
        placeholder = match.group(1)
        if placeholder in placeholders:
            raise ValueError(
                f"rule template reuses placeholder {placeholder!r};"
                " each clause needs its own"
            )
        placeholders.append(placeholder)
        parsed.append((placeholder, _OP_FUNCS[op], feature_token))
    return parsed


def auto_rule_generation(
    outcome: Any,
    features: dict[str, Any],
    rule_specs: dict[str, dict[str, Any]],
    *,
    max_depth: int = 2,
    top_k: int = 5,
    min_n_train_kept: int = 100,
    train_idx: Any,
    test_idx: Any,
    label: str = "unnamed",
) -> AutoRuleReport:
    """Grid-search rule VALUES on train; the test split judges them.

    The caller fixes the rule SHAPE (an AND-of-comparisons template
    like ``"feat_a < {t1} & feat_b <= {t2}"`` plus one grid per
    clause); the tool enumerates value combinations ON THE TRAIN
    SPLIT ONLY, keeps the ``top_k`` by ``delta_train``, and only then
    looks at the untouched test split.  Candidates whose
    ``delta_test`` is positive land in ``valid_rules``;
    ``overfit_warning`` fires when every kept candidate fails on
    test.  EXPLORATORY by construction: generation without a frozen
    downstream preregistration is exactly the winner's curse the
    framework exists to kill.

    Args:
        outcome: response series WITHOUT NaN.
        features: mapping feature name -> single column (every
            template feature must be declared here).
        rule_specs: ``{spec_name: {template: str, grids:
            [[values], ...]}}``; ``len(grids)`` must equal the
            template's clause count.
        max_depth: clause-count cap (at least 1).
        top_k: how many train winners face the test split; at
            least 1.
        min_n_train_kept: minimum kept rows on train for ``valid``.
        train_idx: positional train split.
        test_idx: positional test split (must not overlap).
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`AutoRuleReport` with ``candidates`` ordered by
        ``delta_train`` (best first).

    Raises:
        ValueError: on template deviations (see
            :func:`_parse_multi_rule_template`), a grid count or
            combo count mismatch, an over-cap grid product
            (:data:`_MAX_COMBOS`), NaN outcome, or bad splits.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> rows = np.arange(600)
        >>> feat = pd.Series(rng.uniform(0, 1, 600), index=rows)
        >>> out = pd.Series(
        ...     np.where(feat < 0.5, 1.0, 0.0)
        ...     + 0.1 * rng.standard_normal(600), index=rows
        ... )
        >>> report = auto_rule_generation(
        ...     out, {"feat": feat},
        ...     {"low": {"template": "feat < {t1}",
        ...              "grids": [[0.4, 0.5, 0.6]]}},
        ...     min_n_train_kept=30,
        ...     train_idx=np.arange(400), test_idx=np.arange(400, 600),
        ... )
        >>> report["best_valid_rule"] is not None
        True
        >>> report["overfit_warning"]
        False
    """
    if not rule_specs:
        raise ValueError("rule_specs must not be empty")
    validate_int("max_depth", max_depth, minimum=1)
    validate_int("top_k", top_k, minimum=1)
    validate_int("min_n_train_kept", min_n_train_kept, minimum=1)
    feature_names = list(features)
    notes: list[str] = []
    outcome_series = _to_series(outcome, "outcome")
    if bool(outcome_series.isna().any()):
        raise ValueError(
            "outcome contains NaN; align and drop rows before"
            " auto_rule_generation"
        )
    outcome_values = outcome_series.to_numpy(dtype=float)
    train_positions = np.asarray(train_idx, dtype=int).reshape(-1)
    test_positions = np.asarray(test_idx, dtype=int).reshape(-1)
    if train_positions.size == 0 or test_positions.size == 0:
        raise ValueError("train_idx and test_idx must both be non-empty")
    if set(train_positions.tolist()) & set(test_positions.tolist()):
        raise ValueError("train_idx and test_idx overlap")
    total = outcome_values.size
    for tag, positions in (
        ("train_idx", train_positions),
        ("test_idx", test_positions),
    ):
        if positions.min() < 0 or positions.max() >= total:
            raise ValueError(
                f"{tag} out of range for outcome of {total} rows"
            )
    feature_columns = {
        name: _to_series(features[name], name).to_numpy(dtype=float)
        for name in feature_names
    }
    train_out = outcome_values[train_positions]
    train_rows = train_out.size
    evaluated: list[tuple[AutoRuleCandidate, str, tuple[float, ...]]] = []
    templates: dict[str, str] = {}
    for spec_name, spec in rule_specs.items():
        template, parsed, grid_lists = _validate_rule_spec(
            spec_name, spec, feature_names, max_depth
        )
        templates[spec_name] = template
        evaluated.extend(
            _train_combo_entry(
                spec_name,
                template,
                parsed,
                combo,
                feature_columns=feature_columns,
                train_positions=train_positions,
                train_out=train_out,
                train_rows=train_rows,
            )
            for combo in product(*grid_lists)
        )
    finite = [
        entry for entry in evaluated if np.isfinite(entry[0]["delta_train"])
    ]
    if not finite:
        return AutoRuleReport(
            label=label,
            templates=templates,
            candidates=[],
            valid_rules=[],
            best_valid_rule=None,
            overfit_warning=True,
            notes=[
                *notes,
                "every grid combo left one train side empty",
            ],
        )
    finite.sort(key=lambda entry: entry[0]["delta_train"], reverse=True)
    top_entries = finite[:top_k]
    top_candidates = [entry[0] for entry in top_entries]
    _judge_test_entries(
        top_entries,
        templates,
        feature_columns,
        feature_names,
        max_depth,
        outcome_values=outcome_values,
        test_positions=test_positions,
        min_n_train_kept=min_n_train_kept,
    )
    valid_names = [
        candidate["name"]
        for candidate in top_candidates
        if candidate["status"] == "valid"
    ]
    best_valid: str | None = None
    if valid_names:
        by_name = {
            candidate["name"]: candidate for candidate in top_candidates
        }
        best_valid = max(
            valid_names, key=lambda name: by_name[name]["delta_test"]
        )
    overfit_warning = all(
        candidate["status"] != "valid" for candidate in top_candidates
    )
    return AutoRuleReport(
        label=label,
        templates=templates,
        candidates=top_candidates,
        valid_rules=valid_names,
        best_valid_rule=best_valid,
        overfit_warning=overfit_warning,
        notes=notes,
    )


def _validate_rule_spec(
    spec_name: str,
    spec: Any,
    feature_names: list[str],
    max_depth: int,
) -> tuple[str, list[tuple[str, Any, str]], list[list[float]]]:
    """Validate one rule spec: shape, grids, and the combo cap.

    Returns the template, its parsed clauses and the float grids;
    every deviation (missing keys, clause/grid mismatch, an empty
    grid, a product over :data:`_MAX_COMBOS`) raises ValueError.
    """
    if (
        not isinstance(spec, dict)
        or "template" not in spec
        or ("grids" not in spec)
    ):
        raise ValueError(
            f"rule spec {spec_name!r} must carry 'template' and 'grids'"
        )
    template = str(spec["template"])
    grids = spec["grids"]
    parsed = _parse_multi_rule_template(template, feature_names, max_depth)
    if len(grids) != len(parsed):
        raise ValueError(
            f"rule spec {spec_name!r}: {len(parsed)} clauses but"
            f" {len(grids)} grids"
        )
    grid_lists = [[float(value) for value in grid] for grid in grids]
    if any(not grid for grid in grid_lists):
        raise ValueError(
            f"rule spec {spec_name!r}: every grid must be non-empty"
        )
    combo_count = 1
    for grid in grid_lists:
        combo_count *= len(grid)
    if combo_count > _MAX_COMBOS:
        raise ValueError(
            f"rule spec {spec_name!r}: grid product {combo_count}"
            f" exceeds the cap {_MAX_COMBOS}; shrink the grids"
        )
    return template, parsed, grid_lists


def _train_combo_entry(
    spec_name: str,
    template: str,
    parsed: list[tuple[str, Any, str]],
    combo: tuple[float, ...],
    *,
    feature_columns: dict[str, np.ndarray],
    train_positions: np.ndarray,
    train_out: np.ndarray,
    train_rows: int,
) -> tuple[AutoRuleCandidate, str, tuple[float, ...]]:
    """Build one grid combo's TRAIN-side candidate (generation only).

    The mask is evaluated on the train rows exclusively; the test
    fields stay empty (``delta_test`` NaN, status ``pending``) until
    the top-k judging pass.
    """
    mask = np.ones(train_rows, dtype=bool)
    for value, (_placeholder, op_func, feature_token) in zip(
        combo, parsed, strict=True
    ):
        mask &= op_func(
            feature_columns[feature_token][train_positions], value
        )
    kept = train_out[mask]
    dropped = train_out[~mask]
    delta_train = (
        float(np.mean(kept) - np.mean(dropped))
        if kept.size and dropped.size
        else float("nan")
    )
    render = {
        parsed[position][0]: combo[position]
        for position in range(len(parsed))
    }
    candidate = AutoRuleCandidate(
        name=f"{spec_name}:{combo}",
        rule_str=template.format(**render),
        n_train_kept=int(kept.size),
        n_test_kept=0,
        delta_train=delta_train,
        delta_test=float("nan"),
        status="pending",
    )
    return candidate, spec_name, combo


def _judge_test_entries(
    top_entries: list[tuple[AutoRuleCandidate, str, tuple[float, ...]]],
    templates: dict[str, str],
    feature_columns: dict[str, np.ndarray],
    feature_names: list[str],
    max_depth: int,
    *,
    outcome_values: np.ndarray,
    test_positions: np.ndarray,
    min_n_train_kept: int,
) -> None:
    """Face the top train winners with the untouched test split.

    Mutates each candidate in place: fills ``n_test_kept`` /
    ``delta_test`` and applies the status ladder (n guard first,
    then both-deltas-positive = valid, anything else = unstable).
    """
    total = outcome_values.size
    for candidate, spec_name, combo in top_entries:
        parsed = _parse_multi_rule_template(
            templates[spec_name], feature_names, max_depth
        )
        mask = np.ones(total, dtype=bool)
        for value, (_placeholder, op_func, feature_token) in zip(
            combo, parsed, strict=True
        ):
            mask &= op_func(feature_columns[feature_token], value)
        kept = outcome_values[test_positions][mask[test_positions]]
        dropped = outcome_values[test_positions][~mask[test_positions]]
        delta_test = (
            float(np.mean(kept) - np.mean(dropped))
            if kept.size and dropped.size
            else float("nan")
        )
        candidate["n_test_kept"] = int(kept.size)
        candidate["delta_test"] = delta_test
        if (
            not np.isfinite(delta_test)
            or candidate["n_train_kept"] < min_n_train_kept
            or kept.size == 0
            or dropped.size == 0
        ):
            candidate["status"] = "insufficient_n"
        elif delta_test > 0.0 and candidate["delta_train"] > 0.0:
            candidate["status"] = "valid"
        else:
            candidate["status"] = "unstable"


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _run_interactions(
    outcome: Any,
    features: dict[str, Any],
    interaction_pairs: list[tuple[str, str]] | None,
    min_cell_n: int,
) -> dict[str, InteractionScanReport]:
    """Run every requested pair through :func:`interaction_scan`.

    Both pair names must be declared features and must differ; the
    report key is ``"nameAxnameB"``.
    """
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
    return interactions


def _run_threshold_sweeps(
    outcome: Any,
    features: dict[str, Any],
    threshold_specs: dict[str, list[float]] | None,
    threshold_template: str | None,
    *,
    train_idx: Any | None,
    test_idx: Any | None,
    min_n_kept: int,
) -> dict[str, ThresholdScanReport] | None:
    """Run every spec'd feature through :func:`threshold_scan`.

    The template's ``{feature}`` token is substituted textually (the
    ``{t}`` placeholder stays for the sweep); ``None`` specs return
    ``None``; specs without a frozen split raise.
    """
    if threshold_specs is None:
        return None
    if train_idx is None or test_idx is None:
        raise ValueError(
            "threshold_specs requires BOTH train_idx and test_idx"
            " (the sweep deltas are train/test deltas)"
        )
    template_format = threshold_template or "{feature} < {t}"
    reports: dict[str, ThresholdScanReport] = {}
    for name, values in threshold_specs.items():
        if name not in features:
            raise ValueError(
                f"threshold_specs names {name!r}, which is not a"
                " declared feature"
            )
        sweep_template = template_format.replace("{feature}", name)
        reports[name] = threshold_scan(
            outcome,
            features[name],
            sweep_template,
            values,
            feature_name=name,
            train_idx=train_idx,
            test_idx=test_idx,
            min_n_kept=min_n_kept,
            label=name,
        )
    return reports


def _run_exploratory_blocks(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
    *,
    discovery_pool: dict[str, Any] | None,
    discovery_min_abs_spearman: float,
    importance_model: str | None,
    importance_n_permutations: int,
    rule_generation_specs: dict[str, dict[str, Any]] | None,
    rule_generation_top_k: int,
    train_idx: Any | None,
    test_idx: Any | None,
    seed: int,
    label: str,
) -> tuple[
    DiscoveryReport | None, ImportanceReport | None, AutoRuleReport | None
]:
    """Run the three EXPLORATORY blocks when (and only when) requested.

    Discovery and importance need no split; rule generation requires
    BOTH split indices (generation is train-only, judging is
    test-only).  The outputs never touch the classification.
    """
    discovery = None
    if discovery_pool is not None:
        discovery = auto_discovery(
            outcome,
            discovery_pool,
            controls,
            min_abs_spearman=discovery_min_abs_spearman,
            label=label,
        )
    importance = None
    if importance_model is not None:
        importance = feature_importance(
            outcome,
            features,
            model=importance_model,
            n_permutations=importance_n_permutations,
            seed=seed,
            label=label,
        )
    rule_generation = None
    if rule_generation_specs is not None:
        if train_idx is None or test_idx is None:
            raise ValueError(
                "rule_generation_specs requires BOTH train_idx and"
                " test_idx (generation is train-only, judging is"
                " test-only)"
            )
        rule_generation = auto_rule_generation(
            outcome,
            features,
            rule_generation_specs,
            top_k=rule_generation_top_k,
            train_idx=train_idx,
            test_idx=test_idx,
            label=label,
        )
    return discovery, importance, rule_generation


def _classify_features(
    orth: OrthogonalReport,
    buckets: BucketScanReport,
    clustering: ClusterReport,
    rep_names: list[str],
    regime: RegimeScanReport | None,
    *,
    notes: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Cells, actions and skip reasons for the representative features.

    Ladder: not testable -> NOISE; orthogonal fail (gate or
    collinearity screen) -> PROXY; pass + non-monotone ->
    SHAPE_UNKNOWN; pass + monotone goes through the regime gate -- a
    VERIFIED indicator (status ``OK``, two-plus qualifying regimes)
    splits monotone features into STRONG (monotone in every
    qualifying regime) vs REGIME_DEPENDENT (monotone in some); any
    weaker regime setup (absent, UNVERIFIED, under-powered) keeps the
    cell STRONG_UNVERIFIED.  Cluster members are pointed at their
    representative in ``skipped_features`` and get no independent
    cell.
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
            classification[name], actions[name] = _monotone_cell(
                name, regime, notes
            )
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


def _monotone_cell(
    name: str,
    regime: RegimeScanReport | None,
    notes: list[str],
) -> tuple[str, str]:
    """Regime-gated cell for one orthogonally-detected monotone feature.

    STRONG demands a VERIFIED indicator (prereg_hash -> status OK)
    with at least two qualifying regimes in which the feature is
    monotone; monotone-in-some reads REGIME_DEPENDENT; monotone-in-
    none (a global effect no single regime carries) falls back to
    SHAPE_UNKNOWN for interaction investigation; everything weaker
    stays STRONG_UNVERIFIED.
    """
    if regime is None:
        return _CELL_UNVERIFIED, _ACTION_PREREG
    if regime["status"] != "OK":
        notes.append(
            f"{name}: regime indicator {regime['regime_name']!r} is"
            f" {regime['status']}; cell stays STRONG_UNVERIFIED"
        )
        return _CELL_UNVERIFIED, _ACTION_PREREG
    regime_flags = regime["features"].get(name, {})
    qualifying = [
        regime_key
        for regime_key, entry in regime_flags.items()
        if entry["n"] >= regime["min_regime_n"]
    ]
    if len(qualifying) < 2:
        notes.append(
            f"{name}: fewer than 2 qualifying regimes in verified"
            f" indicator {regime['regime_name']!r}; cell stays"
            " STRONG_UNVERIFIED"
        )
        return _CELL_UNVERIFIED, _ACTION_PREREG
    flags = [regime_flags[key]["monotone"] for key in qualifying]
    if all(flags):
        return _CELL_STRONG, _ACTION_BUILD
    if any(flags):
        return _CELL_REGIME, _ACTION_REGIME_FILTER
    notes.append(
        f"{name}: globally monotone but monotone in no single regime;"
        " cell falls back to SHAPE_UNKNOWN for interaction"
        " investigation"
    )
    return _CELL_SHAPE, _ACTION_INTERACT


def _resolve_inputs(
    outcome: MaterializedData | pd.Series,
    features: dict[str, Any] | pd.DataFrame | None,
    controls: dict[str, Any] | pd.DataFrame | None,
    notes: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the outcome union into plain name -> column maps.

    A :class:`~atropos.data_contract.MaterializedData` bundle is the
    contract path: its manifest is VALIDATED here (failed lookahead
    checks or a failed validation battery are fatal; a missing
    manifest is a loud warning recorded in ``notes``) and its frames
    supply every column -- ``features``/``controls`` must then be
    ``None``.  The legacy path keeps the bare-Series semantics.
    """
    if isinstance(outcome, MaterializedData):
        if features is not None or controls is not None:
            raise ValueError(
                "outcome is a MaterializedData: features/controls"
                " come from the bundle itself and must be None"
            )
        manifest = outcome.manifest
        if manifest is None:
            notes.append(
                "unverified data, contract guarantees not applied"
                " (manifest is None)"
            )
        else:
            failed_lookahead = sorted(
                name
                for name, ok in manifest["lookahead_checks"].items()
                if not ok
            )
            if failed_lookahead:
                raise ValueError(
                    "MaterializedData lookahead check FAILED for"
                    f" {failed_lookahead}; refusing analysis"
                )
            if not manifest["validation"]["passed"]:
                raise ValueError(
                    "MaterializedData validation failed:"
                    " " + "; ".join(manifest["validation"]["fatal"])
                )
        feature_frame = outcome.features
        control_frame = outcome.controls
        features_map = {
            name: feature_frame[name] for name in feature_frame.columns
        }
        controls_map = {
            name: control_frame[name] for name in control_frame.columns
        }
        return features_map, controls_map
    features_arg = features if features is not None else {}
    controls_arg = controls if controls is not None else {}
    if isinstance(features_arg, pd.DataFrame):
        features_map = {
            name: features_arg[name] for name in features_arg.columns
        }
    else:
        features_map = dict(features_arg)
    if isinstance(controls_arg, pd.DataFrame):
        controls_map = {
            name: controls_arg[name] for name in controls_arg.columns
        }
    else:
        controls_map = dict(controls_arg)
    return features_map, controls_map


def feature_analysis(
    outcome: MaterializedData | pd.Series,
    features: dict[str, Any] | pd.DataFrame | None = None,
    controls: dict[str, Any] | pd.DataFrame | None = None,
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
    regime_indicator: dict[str, Any] | None = None,
    min_regime_n: int = 100,
    threshold_specs: dict[str, list[float]] | None = None,
    threshold_template: str | None = None,
    min_n_kept: int = 100,
    discovery_pool: dict[str, Any] | None = None,
    discovery_min_abs_spearman: float = 0.15,
    importance_model: str | None = None,
    importance_n_permutations: int = 100,
    rule_generation_specs: dict[str, dict[str, Any]] | None = None,
    rule_generation_top_k: int = 5,
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
    split are supplied -> regime scan when an indicator is supplied
    (on the detected features) -> threshold sweeps when specs are
    supplied (requires the frozen split) -> classification ->
    recommended actions.

    Classification (per representative): not testable -> ``NOISE``;
    orthogonal gate fail -> ``PROXY``; pass + non-monotone ->
    ``SHAPE_UNKNOWN``; pass + monotone goes through the regime gate:
    a VERIFIED indicator (``prereg_hash``, 2+ qualifying regimes)
    upgrades monotone-in-every-regime features to ``STRONG`` and
    monotone-in-some to ``REGIME_DEPENDENT``; anything weaker stays
    ``STRONG_UNVERIFIED``.  Cluster members get no independent cell:
    ``skipped_features`` points them at their representative.  The
    discovery/importance/rule-generation blocks land in checkpoint
    A3 and read ``None`` until then.

    Args:
        outcome: a :class:`~atropos.data_contract.MaterializedData`
            bundle (the contract path -- its manifest is VALIDATED:
            a failed lookahead check or a failed validation battery
            is fatal, a missing manifest is a loud warning in the
            report notes) or the response series (legacy path).
        features: mapping feature name -> single column, or a
            DataFrame; ``None`` when ``outcome`` is a
            MaterializedData (the bundle carries the columns).
        controls: mapping control name -> single column, or a
            DataFrame; ``None`` when ``outcome`` is a
            MaterializedData.
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
        train_idx: positional train split for ``candidate_rules`` and
            the threshold sweeps.
        test_idx: positional test split (see ``train_idx``).
        regime_indicator: optional ``{name, definition, prereg_hash,
            values}`` dict for :func:`regime_scan` (see there); the
            gate demands ``prereg_hash`` for a STRONG cell.
        min_regime_n: minimum rows per regime for the sign/monotone
            comparison.
        threshold_specs: optional ``{feature: [thresholds]}`` sweeps
            for :func:`threshold_scan`; requires the frozen split.
        threshold_template: optional format string with ``{feature}``
            and ``{t}`` placeholders (default ``"{feature} < {t}"``).
        min_n_kept: minimum kept rows per split for a ``valid``
            threshold point.
        discovery_pool: optional explicit candidate dict for
            :func:`auto_discovery` -- EXPLORATORY output, never
            merged into the classification.
        discovery_min_abs_spearman: spearman pre-filter floor for
            the discovery pool.
        importance_model: optional ``"linear"`` / ``"tree"`` for
            :func:`feature_importance` -- diagnostic only, feeds no
            cell.
        importance_n_permutations: permutation repeats for the
            importance runs.
        rule_generation_specs: optional ``{name: {template, grids}}``
            for :func:`auto_rule_generation`; requires the frozen
            split.  EXPLORATORY output.
        rule_generation_top_k: how many train winners face the test
            split.
        seed: non-negative seed threaded into every seeded step.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`FeatureAnalysisReport`; every schema key is present.

    Raises:
        ValueError: if gate fractions are outside (0, 1), a pair name
            is not a declared feature, ``candidate_rules`` is given
            without both split indices, ``outcome`` is a
            MaterializedData whose manifest fails (lookahead check or
            validation battery), ``features``/``controls`` are passed
            alongside a MaterializedData, or any sub-scan raises.

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
    feature_map, control_map = _resolve_inputs(
        outcome, features, controls, notes
    )
    outcome_series = (
        outcome.outcome if isinstance(outcome, MaterializedData) else outcome
    )
    if candidate_rules is not None and (
        train_idx is None or test_idx is None
    ):
        raise ValueError(
            "candidate_rules requires BOTH train_idx and test_idx"
            " (rule_fit has no default split)"
        )
    feature_names = list(feature_map)

    missing = missingness_scan(
        outcome_series,
        feature_map,
        control_map,
        nan_threshold=nan_threshold,
        label=label,
    )
    clustering = correlation_cluster(
        feature_map,
        outcome_series,
        threshold=cluster_threshold,
        representatives=representatives,
        label=label,
    )
    rep_features = {
        name: feature_map[name]
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
        outcome_series,
        rep_features,
        control_map,
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
        outcome_series,
        {name: feature_map[name] for name in bucket_targets},
        n_buckets=n_buckets,
        bucket_edges=bucket_edges_kept or None,
        min_bucket_n=min_bucket_n,
        norm_r=norm_r,
        n_perm=n_perm_bucket,
        seed=seed,
        label=label,
    )
    interactions = _run_interactions(
        outcome_series, feature_map, interaction_pairs, min_cell_n
    )
    rules = None
    if candidate_rules is not None:
        rules = rule_fit(
            outcome_series,
            candidate_rules,
            train_idx=train_idx,
            test_idx=test_idx,
            label=label,
        )
    regime = None
    if regime_indicator is not None:
        regime = regime_scan(
            outcome_series,
            {name: feature_map[name] for name in bucket_targets},
            regime_indicator,
            min_regime_n=min_regime_n,
            label=label,
        )
    threshold_reports = _run_threshold_sweeps(
        outcome_series,
        feature_map,
        threshold_specs,
        threshold_template,
        train_idx=train_idx,
        test_idx=test_idx,
        min_n_kept=min_n_kept,
    )
    discovery, importance, rule_generation = _run_exploratory_blocks(
        outcome_series,
        feature_map,
        control_map,
        discovery_pool=discovery_pool,
        discovery_min_abs_spearman=discovery_min_abs_spearman,
        importance_model=importance_model,
        importance_n_permutations=importance_n_permutations,
        rule_generation_specs=rule_generation_specs,
        rule_generation_top_k=rule_generation_top_k,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=seed,
        label=label,
    )

    classification, actions, skipped = _classify_features(
        orth, buckets, clustering, list(rep_features), regime, notes=notes
    )
    if discovery is None and importance is None and (rule_generation is None):
        notes.append(
            "exploratory blocks (discovery, importance, rule"
            " generation) not requested; keys stay None"
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
        regime=regime,
        threshold=threshold_reports,
        discovery=discovery,
        importance=importance,
        rule_generation=rule_generation,
        skipped_features=skipped,
        classification=classification,
        recommended_actions=actions,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _demo_orchestrator(
    outcome: pd.Series,
    features: dict[str, pd.Series],
    controls: dict[str, pd.Series],
) -> None:
    """Demo block: the orchestrator classification on the planted world."""
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


def _demo_bucket(outcome: pd.Series, features: dict[str, pd.Series]) -> None:
    """Demo block: the bucket profile + Q1/Q2 split."""
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


def _demo_interaction(
    outcome: pd.Series, features: dict[str, pd.Series]
) -> None:
    """Demo block: the additive world reads no interaction."""
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


def _demo_rule_fit(outcome: pd.Series, real: np.ndarray, n_rows: int) -> None:
    """Demo block: a planted rule reads valid with a positive test delta."""
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


def _demo_missingness(
    outcome: pd.Series, features: dict[str, pd.Series]
) -> None:
    """Demo block: outcome-linked NaN warns, random NaN stays silent."""
    nan_feature = features["beta_null"].copy()
    nan_feature[outcome > 1.0] = np.nan
    missing_shift = missingness_scan(
        outcome, {"flagged": nan_feature}, {}, nan_threshold=0.05
    )
    print(
        "missingness (outcome-linked NaN): warn=",
        missing_shift["features"]["flagged"]["warn"],
        " differs=",
        missing_shift["features"]["flagged"]["nan_outcome_differs"],
    )
    assert missing_shift["features"]["flagged"]["warn"] is True


def _demo_discovery(
    outcome: pd.Series, features: dict[str, pd.Series]
) -> None:
    """Demo block: the planted pool yields one EXPLORATORY hypothesis."""
    discovery = auto_discovery(
        outcome,
        {
            "alpha_real": features["alpha_real"],
            "beta_null": features["beta_null"],
        },
        {},
        min_abs_spearman=0.0,
        label="discovery demo",
    )
    print(
        "discovery (planted pool):",
        [entry["feature"] for entry in discovery["discovered"]],
        "| all_below:",
        discovery["all_below_threshold"],
    )
    assert discovery["n_candidates"] == 2
    assert [entry["feature"] for entry in discovery["discovered"]] == [
        "alpha_real"
    ]
    assert "prereg + run_full_pipeline" in discovery["note"]


def _demo_importance(
    outcome: pd.Series, features: dict[str, pd.Series]
) -> None:
    """Demo block: the planted feature dominates permutation importance."""
    importance = feature_importance(
        outcome,
        features,
        model="linear",
        n_permutations=20,
        seed=42,
    )
    print(
        "importance (linear): alpha_real =",
        round(importance["features"]["alpha_real"]["importance_mean"], 4),
        "beta_null =",
        round(importance["features"]["beta_null"]["importance_mean"], 4),
    )
    assert (
        importance["features"]["alpha_real"]["importance_mean"]
        > importance["features"]["beta_null"]["importance_mean"]
    )


def _demo_rule_generation(
    outcome: pd.Series, features: dict[str, pd.Series], n_rows: int
) -> None:
    """Demo block: grid-searched rule shape validates on test."""
    rule_gen = auto_rule_generation(
        outcome,
        features,
        {
            "sign_rule": {
                "template": "alpha_real > {t1}",
                "grids": [[0.0, 0.3]],
            }
        },
        min_n_train_kept=50,
        train_idx=np.arange(0, 500),
        test_idx=np.arange(500, n_rows),
    )
    print(
        "rule generation: best =",
        rule_gen["best_valid_rule"],
        "| overfit_warning:",
        rule_gen["overfit_warning"],
    )
    assert rule_gen["best_valid_rule"] is not None
    assert rule_gen["overfit_warning"] is False


def run_all_examples() -> None:
    """Demo: one planted world through the whole chain, asserted.

    Every printed verdict is asserted (in the per-block helpers), so
    a regression crashes the demo instead of printing a wrong number
    quietly.
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
    _demo_orchestrator(outcome, features, controls)
    _demo_bucket(outcome, features)
    _demo_interaction(outcome, features)
    _demo_rule_fit(outcome, real, n_rows)
    _demo_missingness(outcome, features)
    _demo_discovery(outcome, features)
    _demo_importance(outcome, features)
    _demo_rule_generation(outcome, features, n_rows)


if __name__ == "__main__":
    run_all_examples()
