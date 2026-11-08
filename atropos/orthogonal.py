"""Orthogonal checks: what does a feature add that controls do not know?

Two public checks share ONE fixed report schema
(:class:`OrthogonalReport`, every key present on every outcome):

* :func:`orthogonal_check` (``kind="orthogonal"``) decomposes a set of
  candidate features against a control block: block delta-R2, per-column
  HC0 statistics, per-feature partial delta-R2, a three-gate detection
  rule and a per-feature permutation p-value (seed pinned, so the
  report is reproducible bit for bit).
* :func:`signal_vs_controls` (``kind="signal_check"``) answers the
  stage-1 question -- does ONE signal survive its controls? -- mapped
  onto the same schema (one feature entry, EMPTY ``permutation`` dict:
  signal acceptance is deterministic, not permuted).

Statuses are explicit, never silent:

* ``OK`` -- identifiable design, numbers are evidence;
* ``NO_FEATURES`` -- nothing testable was supplied: degenerate feature
  columns are dropped with a note, and an empty / fully degenerate
  feature set returns early with the reason in ``facts["notes"]``
  (controls-only statistics are still computed);
* ``INCONCLUSIVE`` -- rank-deficient or saturated design, constant
  outcome or empty alignment: the statistics come back NaN and must
  not be read as evidence either way.

The one detection flag is ``facts["detected"]`` (single field name for
both kinds); the reading per kind:

* ``kind="orthogonal"`` -- True when AT LEAST ONE feature passes all
  three gates: ``|t| >= 2.5`` (Bonferroni-style single-feature gate,
  recorded as ``facts["t_threshold"]``), thirds-of-sample sign
  stability (every third's feature effect has the full-sample sign AND
  ``|t_third| >= 1`` in at least 2 of 3 thirds), and the collinearity
  screen (a feature with ``|pearson| >= 0.8`` against any control is
  flagged "collinear ..., cannot separate" and reads NOT detected --
  False here means "the check could not separate", never "the feature
  does not work").  Per-feature verdicts reconstruct from
  ``full["t_stat"]`` (the gate), ``facts["sign_consistent"]`` (the
  thirds rule for the STRONGEST feature, the headline claim) and the
  collinear notes.
* ``kind="signal_check"`` -- True when the signal's full-sample
  ``t >= 2.0`` in its STATED orientation after controls (orientation is
  pinned: a signal with ``t <= -2`` survives controls INVERTED and says
  so in a note, but ``detected`` stays False for the stated
  orientation).

Import note (Minor 1): imports atropos._utils.ols_hc0 (intra-package);
promote ols_hc0 to a public path if _utils is refactored.

The per-feature gate verdicts are surfaced verbatim under
``per_feature_detected`` (added for atropos.feature_scan, whose
classification needs each feature's verdict, not only the aggregate);
``orthogonal_check`` additionally accepts an explicit ``t_threshold``
override so a caller can derive the Bonferroni gate from the number
of tested clusters (``None`` keeps the house 2.5).

Run ``python -m atropos.orthogonal`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from atropos._utils import align_inputs, frameify, ols_hc0, validate_int

__all__ = [
    "OrthogonalReport",
    "orthogonal_check",
    "signal_vs_controls",
]

_T_STRONG = 2.5
"""Single-feature |t| gate for ``kind="orthogonal"`` (Bonferroni-style)."""

_T_SIGNAL = 2.0
"""Orientation-pinned survival gate for ``kind="signal_check"``."""

_T_THIRD = 1.0
"""Per-third |t| floor of the thirds sign-stability rule."""

_COLLINEAR_RHO = 0.8
"""|pearson| against a control that blocks feature separation."""

_MAX_DROPNA_SHARE = 0.10
"""Fraction of aligned rows dropped before the excessive-dropna note."""

_KIND_DECOMP = "orthogonal"
_KIND_SIGNAL = "signal_check"
_OK = "OK"
_NO_FEATURES = "NO_FEATURES"
_INCONCLUSIVE = "INCONCLUSIVE"


class FitBlock(TypedDict):
    """HC0 statistics of one fitted specification, keyed by column.

    Keys: ``r2`` (R-squared of the fit), ``beta`` / ``se`` / ``t_stat``
    / ``p_value`` (per-column maps; ``intercept`` first, then features
    in insertion order, then controls in insertion order; the map names
    carry the repo's two-letter minimum, the underlying ``ols_hc0``
    payload keys stay ``t`` / ``p``).
    """

    r2: float
    beta: dict[str, float]
    se: dict[str, float]
    t_stat: dict[str, float]
    p_value: dict[str, float]


class CorrelationBlock(TypedDict):
    """Pearson and Spearman matrices over outcome + features + controls.

    Keys ``pearson`` and ``spearman`` each hold a row -> column -> float
    matrix; the row/column keys are ``"outcome"`` plus the surviving
    feature names plus the control names.
    """

    pearson: dict[str, dict[str, float]]
    spearman: dict[str, dict[str, float]]


class FactsBlock(TypedDict):
    """Reading of the check: verdict flag, strongest column and notes.

    ``detected`` is the single verdict field for both kinds (see the
    module docstring for the per-kind reading); ``sign_consistent`` is
    the thirds sign-stability of ``strongest_feature`` (the headline
    claim); ``t_threshold`` is the gate the kind applies (2.5
    orthogonal, 2.0 signal); ``notes`` carries every explicit reason
    (dropna, degenerate drops, collinearity, INCONCLUSIVE causes) so no
    outcome is silent.
    """

    detected: bool
    sign_consistent: bool
    strongest_feature: str
    strongest_t: float
    t_threshold: float
    notes: list[str]


class OrthogonalReport(TypedDict):
    """Fixed report schema; every key is present on every outcome."""

    kind: str
    label: str
    status: str
    n_before_dropna: int
    n_after_dropna: int
    n_dropped: int
    controls: list[str]
    features: list[str]
    correlation: CorrelationBlock
    controls_only: FitBlock
    full: FitBlock
    delta_r2_block: float
    per_feature_delta_r2: dict[str, float]
    per_feature_detected: dict[str, bool]
    permutation: dict[str, float]
    facts: FactsBlock


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _is_degenerate(values: np.ndarray) -> bool:
    """True when a column cannot carry any statistical test.

    Same semantics as ``atropos.ensemble_pipeline._is_degenerate``:
    empty, all-NaN, no finite entry, or a zero peak-to-peak range over
    the finite slice.  Judged on the ALIGNED column (post-dropna), the
    values that would actually enter the design.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return True
    if np.all(np.isnan(arr)):
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.ptp(finite) == 0.0)


def _as_named_frame(value: Any, name: str) -> pd.DataFrame:
    """One-column frame view of a dict input, named for the join.

    A multi-column value (DataFrame or 2-D array) cannot map onto one
    report name, so it is rejected loudly instead of being silently
    flattened under a synthetic label.
    """
    frame = frameify(value, name)
    if frame.shape[1] != 1:
        raise ValueError(
            f"input {name!r} must be a single column, got"
            f" {frame.shape[1]} columns"
        )
    return frame


def _check_input_names(
    feature_names: list[str], control_names: list[str]
) -> None:
    """Reject reserved / colliding input names before any fitting.

    ``"outcome"`` names the outcome column of the correlation matrices;
    a feature or control stealing it (or a name shared between the two
    dicts) would silently overwrite a column, so both fail loudly.
    """
    overlap = sorted(set(feature_names) & set(control_names))
    if overlap:
        raise ValueError(f"feature and control names collide: {overlap}")
    reserved = sorted((set(feature_names) | set(control_names)) & {"outcome"})
    if reserved:
        raise ValueError(
            "the name 'outcome' is reserved for the report's outcome"
            f" column; rename: {reserved}"
        )


def _column_map(values: Any, names: list[str]) -> dict[str, float]:
    """Zip one ``ols_hc0`` statistic vector with its column names."""
    array = np.asarray(values, dtype=float)
    return {name: float(array[pos]) for pos, name in enumerate(names)}


def _fit_block(fit: dict[str, Any], names: list[str]) -> FitBlock:
    """Pack one ``ols_hc0`` payload into the report's fit block."""
    return FitBlock(
        r2=float(fit["r2"]),
        beta=_column_map(fit["beta"], names),
        se=_column_map(fit["se"], names),
        t_stat=_column_map(fit["t"], names),
        p_value=_column_map(fit["p"], names),
    )


def _nan_block() -> FitBlock:
    """Empty fit block for an outcome with no identifiable model."""
    return FitBlock(
        r2=float("nan"),
        beta={},
        se={},
        t_stat={},
        p_value={},
    )


def _corr_matrix(
    frame: pd.DataFrame, method: Literal["pearson", "spearman"]
) -> dict[str, dict[str, float]]:
    """Full correlation matrix of ``frame`` as plain nested floats."""
    matrix = frame.corr(method=method)
    return {
        row: {col: float(matrix.loc[row, col]) for col in matrix.columns}
        for row in matrix.index
    }


def _third_slices(n_rows: int) -> list[slice]:
    """Contiguous thirds of the aligned sample (as equal as ints allow).

    Fewer than 3 rows: three empty slices, whose fits come back
    rank-deficient (NaN), which fails the sign-stability rule honestly.
    """
    if n_rows < 3:
        return [slice(0, 0), slice(0, 0), slice(0, 0)]
    edges = [0, n_rows // 3, (2 * n_rows) // 3, n_rows]
    return [slice(edges[slot], edges[slot + 1]) for slot in range(3)]


def _feature_third_ts(
    target: np.ndarray,
    column: np.ndarray,
    controls: list[np.ndarray],
    slices: list[slice],
) -> list[float]:
    """Per-third t of one feature in the full (feature + controls) spec."""
    third_ts: list[float] = []
    for part in slices:
        width = part.stop - part.start
        third_design = np.column_stack(
            [
                np.ones(width),
                column[part],
                *(ctrl[part] for ctrl in controls),
            ]
        )
        fit = ols_hc0(target[part], third_design)
        third_ts.append(float(fit["t"][1]))
    return third_ts


def _sign_stable(t_full: float, third_ts: list[float]) -> bool:
    """Thirds rule: every sign matches the full sample, 2 of 3 strong.

    A third whose t is non-finite (tiny / collinear slice) carries no
    sign information and fails the rule rather than passing silently.
    """
    if not np.isfinite(t_full):
        return False
    full_sign = np.sign(t_full)
    signs_match = all(
        np.isfinite(t_third) and np.sign(t_third) == full_sign
        for t_third in third_ts
    )
    strong_thirds = sum(abs(t_third) >= _T_THIRD for t_third in third_ts)
    return bool(signs_match and strong_thirds >= 2)


def _permutation_p(
    target: np.ndarray,
    design: np.ndarray,
    *,
    position: int,
    real_t: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p of one design column's t.

    The column is shuffled in place (its marginal distribution is kept,
    its link to the outcome broken); the share of shuffled fits whose
    ``|t|`` reaches the real ``|t|`` is the p-value.  Non-finite
    permuted t (degenerate shuffle draw) never counts as exceeding.
    """
    column = design[:, position]
    exceed = 0
    for _ in range(n_perm):
        trial = design.copy()
        trial[:, position] = rng.permutation(column)
        t_perm = float(ols_hc0(target, trial)["t"][position])
        if np.isfinite(t_perm) and abs(t_perm) >= abs(real_t):
            exceed += 1
    return exceed / float(n_perm)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def _report(
    *,
    kind: str,
    label: str,
    status: str,
    n_before: int,
    n_after: int,
    n_dropped: int,
    control_names: list[str],
    feature_names: list[str],
    correlation: CorrelationBlock,
    controls_only: FitBlock,
    full: FitBlock,
    delta_r2_block: float,
    per_feature_delta_r2: dict[str, float],
    per_feature_detected: dict[str, bool],
    permutation: dict[str, float],
    detected: bool,
    sign_consistent: bool,
    strongest_feature: str,
    strongest_t: float,
    t_threshold: float,
    notes: list[str],
) -> OrthogonalReport:
    """Assemble the fixed-schema report (every key, every outcome)."""
    facts = FactsBlock(
        detected=detected,
        sign_consistent=sign_consistent,
        strongest_feature=strongest_feature,
        strongest_t=strongest_t,
        t_threshold=t_threshold,
        notes=notes,
    )
    return OrthogonalReport(
        kind=kind,
        label=label,
        status=status,
        n_before_dropna=n_before,
        n_after_dropna=n_after,
        n_dropped=n_dropped,
        controls=control_names,
        features=feature_names,
        correlation=correlation,
        controls_only=controls_only,
        full=full,
        delta_r2_block=delta_r2_block,
        per_feature_delta_r2=per_feature_delta_r2,
        per_feature_detected=per_feature_detected,
        permutation=permutation,
        facts=facts,
    )


def _align_count(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
) -> tuple[int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Index-align all inputs; count rows before/after the NaN drop.

    The counters replicate the inner-join ``align_inputs`` performs
    (both go through the same concat + dropna path), so
    ``n_before - n_after`` is exactly the number of dropped rows.
    """
    frames = [_as_named_frame(outcome, "outcome")]
    frames.extend(
        _as_named_frame(value, name) for name, value in features.items()
    )
    frames.extend(
        _as_named_frame(value, name) for name, value in controls.items()
    )
    n_before = len(pd.concat(frames, axis=1, join="inner"))
    aligned = align_inputs(outcome, *features.values(), *controls.values())
    feature_arrs = aligned[1 : 1 + len(features)]
    control_arrs = aligned[1 + len(features) :]
    return n_before, aligned[0], feature_arrs, control_arrs


def _screen_degenerate(
    feature_names: list[str],
    feature_arrs: list[np.ndarray],
    control_names: list[str],
    control_arrs: list[np.ndarray],
    notes: list[str],
) -> tuple[list[str], list[np.ndarray]]:
    """Drop degenerate FEATURES with a note; flag degenerate controls.

    A degenerate control cannot be dropped (it would silently change
    the requested specification), so it stays and the rank-deficiency
    guard turns the run INCONCLUSIVE downstream.
    """
    kept: list[tuple[str, np.ndarray]] = []
    for name, arr in zip(feature_names, feature_arrs, strict=True):
        if _is_degenerate(arr):
            notes.append(
                f"feature {name} dropped: degenerate column (constant"
                " or all-NaN after alignment)"
            )
        else:
            kept.append((name, arr))
    for name, arr in zip(control_names, control_arrs, strict=True):
        if _is_degenerate(arr):
            notes.append(
                f"control {name} is degenerate (constant or all-NaN);"
                f" expect status {_INCONCLUSIVE}"
            )
    kept_names = [name for name, _ in kept]
    kept_arrs = [arr for _, arr in kept]
    return kept_names, kept_arrs


def _gate_verdicts(
    fit_full: dict[str, Any],
    correlation: CorrelationBlock,
    *,
    kept_names: list[str],
    kept_arrs: list[np.ndarray],
    control_names: list[str],
    control_arrs: list[np.ndarray],
    outcome_arr: np.ndarray,
    kind: str,
    t_threshold: float,
    identifiable: bool,
    n_after: int,
    notes: list[str],
) -> tuple[dict[str, bool], bool, bool, str, float]:
    """Apply the kind's detection gates to every surviving feature.

    Returns the per-feature verdict map, the aggregate ``detected``
    flag, the strongest feature's thirds-stability, and the strongest
    feature / its t (headline claim of the report).
    """
    slices = _third_slices(n_after)
    stability_by_name: dict[str, bool] = {}
    detected_map: dict[str, bool] = {}
    strongest_feature = ""
    strongest_t = float("nan")
    for position, name in enumerate(kept_names, start=1):
        t_column = float(fit_full["t"][position])
        stable = _sign_stable(
            t_column,
            _feature_third_ts(
                outcome_arr, kept_arrs[position - 1], control_arrs, slices
            ),
        )
        stability_by_name[name] = stable
        collinear_with = ""
        for control_name in control_names:
            rho = correlation["pearson"][name][control_name]
            if abs(rho) >= _COLLINEAR_RHO:
                collinear_with = control_name
                notes.append(
                    f"feature {name} is collinear with control"
                    f" {control_name} (|pearson| = {abs(rho):.2f}),"
                    " cannot separate"
                )
                break
        gate_passes = (
            identifiable
            and np.isfinite(t_column)
            and abs(t_column) >= t_threshold
        )
        detected_map[name] = bool(
            gate_passes and stable and not collinear_with
        )
        if np.isfinite(t_column) and (
            not strongest_feature or abs(t_column) > abs(strongest_t)
        ):
            strongest_feature = name
            strongest_t = t_column

    sign_consistent = bool(
        identifiable
        and strongest_feature != ""
        and stability_by_name[strongest_feature]
    )
    if kind == _KIND_SIGNAL:
        signal_name = kept_names[0]
        t_signal = float(fit_full["t"][1])
        if identifiable:
            detected_map[signal_name] = bool(t_signal >= t_threshold)
            if t_signal <= -t_threshold:
                notes.append(
                    "signal survives controls with INVERTED sign"
                    f" (t = {t_signal:.2f}); detected stays False for"
                    " the stated orientation"
                )
        notes.append(
            "signal-vs-controls check (stage-1 semantics): not a"
            " feature decomposition"
        )
        detected = bool(detected_map[signal_name])
    else:
        notes.append(
            f"feature gate: |t| >= {t_threshold:g} (Bonferroni-style),"
            " thirds sign stability (|t_third| >= 1 in >= 2 of 3"
            f" thirds), collinearity screen (|pearson| >="
            f" {_COLLINEAR_RHO})"
        )
        detected = bool(identifiable and any(detected_map.values()))
    return (
        detected_map,
        detected,
        sign_consistent,
        strongest_feature,
        strongest_t,
    )


def _no_features_report(
    *,
    kind: str,
    label: str,
    t_threshold: float,
    n_before: int,
    n_after: int,
    n_dropped: int,
    outcome_arr: np.ndarray,
    control_names: list[str],
    control_arrs: list[np.ndarray],
    notes: list[str],
) -> OrthogonalReport:
    """Early NO_FEATURES report: explicit reason, controls still fitted.

    The full-model block stays honestly empty (NaN r2, no column maps):
    without features there is no full model, and the schema key must
    still be present so readers never branch on a missing key.
    """
    ones = np.ones(n_after)
    design_restricted = np.column_stack([ones, *control_arrs])
    fit_restricted = ols_hc0(outcome_arr, design_restricted)
    controls_only = _fit_block(fit_restricted, ["intercept", *control_names])
    corr_frame = pd.DataFrame(
        {
            "outcome": outcome_arr,
            **dict(zip(control_names, control_arrs, strict=True)),
        }
    )
    correlation = CorrelationBlock(
        pearson=_corr_matrix(corr_frame, "pearson"),
        spearman=_corr_matrix(corr_frame, "spearman"),
    )
    notes.append(
        f"status {_NO_FEATURES}: no usable feature columns; the"
        " controls-only statistics above are still computed"
    )
    return _report(
        kind=kind,
        label=label,
        status=_NO_FEATURES,
        n_before=n_before,
        n_after=n_after,
        n_dropped=n_dropped,
        control_names=control_names,
        feature_names=[],
        correlation=correlation,
        controls_only=controls_only,
        full=_nan_block(),
        delta_r2_block=float("nan"),
        per_feature_delta_r2={},
        per_feature_detected={},
        permutation={},
        detected=False,
        sign_consistent=False,
        strongest_feature="",
        strongest_t=float("nan"),
        t_threshold=t_threshold,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# shared worker
# ---------------------------------------------------------------------------
def _permutation_block(
    fit_full: dict[str, Any],
    design_full: np.ndarray,
    outcome_arr: np.ndarray,
    kept_names: list[str],
    *,
    kind: str,
    identifiable: bool,
    n_perm: int,
    seed: int,
) -> dict[str, float]:
    """Permutation p-values (orthogonal kind) or the honest fallback.

    ``kind="signal_check"`` returns the EMPTY dict (signal acceptance
    is deterministic, not permuted); an unidentifiable design returns
    NaN per feature so the keys stay present but read as no evidence.
    One pinned rng serves every feature in insertion order, so a fixed
    seed reproduces the whole block bit for bit.
    """
    permutation: dict[str, float] = {}
    if kind == _KIND_SIGNAL:
        return permutation
    if not identifiable:
        return {name: float("nan") for name in kept_names}
    rng = np.random.default_rng(seed)
    for position, name in enumerate(kept_names, start=1):
        permutation[name] = _permutation_p(
            outcome_arr,
            design_full,
            position=position,
            real_t=float(fit_full["t"][position]),
            n_perm=n_perm,
            rng=rng,
        )
    return permutation


def _run_check(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
    *,
    label: str,
    kind: str,
    t_threshold: float,
    n_perm: int = 0,
    seed: int = 0,
) -> OrthogonalReport:
    """Align inputs, fit restricted/full OLS, apply the kind's gates.

    ``kind="orthogonal"`` runs the three-gate feature decomposition and
    the permutation block; ``kind="signal_check"`` maps the single
    signal through the same path with the orientation-pinned gate and
    an EMPTY permutation dict.
    """
    feature_names = list(features)
    control_names = list(controls)
    _check_input_names(feature_names, control_names)
    n_before, outcome_arr, feature_arrs, control_arrs = _align_count(
        outcome, features, controls
    )
    n_after = int(outcome_arr.size)
    n_dropped = n_before - n_after

    notes: list[str] = []
    if n_before > 0 and n_dropped / n_before > _MAX_DROPNA_SHARE:
        pct = 100.0 * n_dropped / n_before
        notes.append(
            f"excessive dropna — {pct:.0f}% removed, check missingness"
            " pattern"
        )
    kept_names, kept_arrs = _screen_degenerate(
        feature_names, feature_arrs, control_names, control_arrs, notes
    )
    if not kept_names:
        if not feature_names:
            notes.append("no features supplied (features dict is empty)")
        else:
            notes.append(
                "all supplied features are degenerate (constant or"
                " all-NaN after alignment)"
            )
        return _no_features_report(
            kind=kind,
            label=label,
            t_threshold=t_threshold,
            n_before=n_before,
            n_after=n_after,
            n_dropped=n_dropped,
            outcome_arr=outcome_arr,
            control_names=control_names,
            control_arrs=control_arrs,
            notes=notes,
        )

    ones = np.ones(n_after)
    design_restricted = np.column_stack([ones, *control_arrs])
    design_full = np.column_stack([ones, *kept_arrs, *control_arrs])
    fit_restricted = ols_hc0(outcome_arr, design_restricted)
    fit_full = ols_hc0(outcome_arr, design_full)
    controls_only = _fit_block(fit_restricted, ["intercept", *control_names])
    full = _fit_block(fit_full, ["intercept", *kept_names, *control_names])
    r2_full = float(fit_full["r2"])
    delta_r2_block = r2_full - float(fit_restricted["r2"])

    per_feature_delta_r2: dict[str, float] = {}
    for position, name in enumerate(kept_names, start=1):
        loo_design = np.delete(design_full, position, axis=1)
        r2_loo = float(ols_hc0(outcome_arr, loo_design)["r2"])
        per_feature_delta_r2[name] = r2_full - r2_loo

    corr_frame = pd.DataFrame(
        {
            "outcome": outcome_arr,
            **dict(
                zip(
                    [*kept_names, *control_names],
                    [*kept_arrs, *control_arrs],
                    strict=True,
                )
            ),
        }
    )
    correlation = CorrelationBlock(
        pearson=_corr_matrix(corr_frame, "pearson"),
        spearman=_corr_matrix(corr_frame, "spearman"),
    )

    status = _OK
    rank_ok = not bool(fit_full["rank_deficient"])
    constant_outcome = bool(n_after > 0 and np.ptp(outcome_arr) == 0.0)
    identifiable = rank_ok and not constant_outcome and np.isfinite(r2_full)
    if n_after == 0:
        status = _INCONCLUSIVE
        notes.append("no rows survived alignment - cannot fit any design")
    elif not rank_ok:
        status = _INCONCLUSIVE
        notes.append(
            "full design is rank-deficient or saturated (collinear"
            " columns or too few rows) - statistics are NaN and must"
            " not be read as evidence"
        )
    elif constant_outcome:
        status = _INCONCLUSIVE
        notes.append("outcome is constant after alignment - nothing to fit")

    (
        detected_map,
        detected,
        sign_consistent,
        strongest_feature,
        strongest_t,
    ) = _gate_verdicts(
        fit_full,
        correlation,
        kept_names=kept_names,
        kept_arrs=kept_arrs,
        control_names=control_names,
        control_arrs=control_arrs,
        outcome_arr=outcome_arr,
        kind=kind,
        t_threshold=t_threshold,
        identifiable=identifiable,
        n_after=n_after,
        notes=notes,
    )
    permutation = _permutation_block(
        fit_full,
        design_full,
        outcome_arr,
        kept_names,
        kind=kind,
        identifiable=identifiable,
        n_perm=n_perm,
        seed=seed,
    )

    return _report(
        kind=kind,
        label=label,
        status=status,
        n_before=n_before,
        n_after=n_after,
        n_dropped=n_dropped,
        control_names=control_names,
        feature_names=kept_names,
        correlation=correlation,
        controls_only=controls_only,
        full=full,
        delta_r2_block=delta_r2_block,
        per_feature_delta_r2=per_feature_delta_r2,
        per_feature_detected=dict(detected_map),
        permutation=permutation,
        detected=detected,
        sign_consistent=sign_consistent,
        strongest_feature=strongest_feature,
        strongest_t=strongest_t,
        t_threshold=t_threshold,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# public checks
# ---------------------------------------------------------------------------
def orthogonal_check(
    outcome: Any,
    features: dict[str, Any],
    controls: dict[str, Any],
    *,
    label: str,
    n_perm: int = 200,
    seed: int = 42,
    t_threshold: float | None = None,
) -> OrthogonalReport:
    """Decompose candidate features against a control block.

    Fits the controls-only design (restricted) and the features +
    controls design (full) with HC0 standard errors, and reports the
    block delta-R2, per-column statistics, per-feature leave-one-out
    delta-R2, the three-gate detection verdicts and per-feature
    permutation p-values (one pinned rng, features in insertion order).
    See the module docstring for the verdict semantics; degenerate
    features are dropped with notes, and a feature set that is empty or
    fully degenerate returns status ``NO_FEATURES`` (controls-only
    statistics still computed) instead of failing silently.

    Args:
        outcome: response series (pandas/numpy 1-D array-like),
            index-aligned with every column, NaN rows dropped.
        features: mapping feature name -> single column (Series or
            1-D array-like); insertion order fixes the design order.
        controls: mapping control name -> single column; same rules as
            ``features``.
        label: free-text tag stored under ``report["label"]``.
        n_perm: permutation draws per feature; at least 1.
        seed: non-negative seed of the pinned permutation rng.
        t_threshold: |t| gate of the detection rule; ``None`` keeps the
            house default 2.5.  Callers that derive a Bonferroni gate
            from the number of tested clusters (feature_scan) pass the
            derived value explicitly; it is recorded in
            ``facts["t_threshold"]`` either way.

    Returns:
        :class:`OrthogonalReport` with ``kind="orthogonal"``; the
        status is ``OK``, ``NO_FEATURES`` or ``INCONCLUSIVE`` and every
        schema key is present on every outcome.

    Raises:
        ValueError: if ``n_perm`` is not an integer >= 1, ``seed`` is
            not a non-negative integer, ``t_threshold`` is not a
            positive finite float, a feature/control value is not a
            single column, a name collides between the two dicts, or
            a name uses the reserved key ``"outcome"``.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(240)
        >>> feat = pd.Series(rng.standard_normal(240), index=idx)
        >>> ctrl = pd.Series(rng.standard_normal(240), index=idx)
        >>> out = pd.Series(
        ...     0.4 * feat + 0.3 * ctrl + rng.standard_normal(240),
        ...     index=idx,
        ... )
        >>> report = orthogonal_check(
        ...     out, {"feat": feat}, {"ctrl": ctrl},
        ...     label="demo", n_perm=25,
        ... )
        >>> report["status"], report["facts"]["detected"]
        ('OK', True)
        >>> report["permutation"]["feat"] == 0.0
        True
    """
    validate_int("n_perm", n_perm)
    validate_int("seed", seed, minimum=0)
    if t_threshold is None:
        t_threshold = _T_STRONG
    if not np.isfinite(t_threshold) or t_threshold <= 0:
        raise ValueError(
            f"t_threshold must be a positive finite float, got"
            f" {t_threshold!r}"
        )
    return _run_check(
        outcome,
        features,
        controls,
        label=label,
        kind=_KIND_DECOMP,
        t_threshold=t_threshold,
        n_perm=n_perm,
        seed=seed,
    )


def signal_vs_controls(
    outcome: Any,
    signal: dict[str, Any],
    controls: dict[str, Any],
    *,
    label: str,
) -> OrthogonalReport:
    """Does ONE signal survive its controls (stage-1 question)?

    The signal is treated as a single feature column, so the shared
    schema holds: ``per_feature_delta_r2`` carries exactly one key and
    ``permutation`` is EMPTY -- signal acceptance is deterministic, not
    permuted.  ``facts["detected"]`` is orientation-pinned: True when
    the signal's ``t >= 2.0`` after controls in its stated orientation;
    a ``t <= -2`` signal survives controls inverted (noted as such) but
    is NOT detected as stated.  This is the signal-vs-controls check
    (what stage-1 already does), NOT a feature decomposition.

    Args:
        outcome: response series (pandas/numpy 1-D array-like),
            index-aligned with every column, NaN rows dropped.
        signal: mapping with EXACTLY one entry ``{name: column}``.
        controls: mapping control name -> single column; same rules as
            the signal entry.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`OrthogonalReport` with ``kind="signal_check"``; every
        schema key is present on every outcome.

    Raises:
        ValueError: if ``signal`` does not hold exactly one entry, a
            signal/control value is not a single column, a name
            collides between the dicts, or a name uses the reserved
            key ``"outcome"``.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> idx = pd.RangeIndex(240)
        >>> feat = pd.Series(rng.standard_normal(240), index=idx)
        >>> ctrl = pd.Series(rng.standard_normal(240), index=idx)
        >>> out = pd.Series(
        ...     0.4 * feat + 0.3 * ctrl + rng.standard_normal(240),
        ...     index=idx,
        ... )
        >>> report = signal_vs_controls(
        ...     out, {"feat": feat}, {"ctrl": ctrl}, label="stage-1"
        ... )
        >>> report["kind"], report["facts"]["detected"]
        ('signal_check', True)
        >>> report["permutation"]
        {}
    """
    if len(signal) != 1:
        raise ValueError(
            "signal must hold exactly one {name: column} entry, got"
            f" {len(signal)} entries: {sorted(signal)}"
        )
    return _run_check(
        outcome,
        signal,
        controls,
        label=label,
        kind=_KIND_SIGNAL,
        t_threshold=_T_SIGNAL,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: a planted 600-row world splits real effect from null noise.

    The demo doubles as the layer-0 verification: every printed verdict
    is asserted, so a regression crashes the demo instead of printing a
    wrong number quietly.
    """
    n_rows = 600
    rng = np.random.default_rng(42)
    real = rng.standard_normal(n_rows)
    null = rng.standard_normal(n_rows)
    control = rng.standard_normal(n_rows)
    outcome = pd.Series(
        0.5 * real + 0.4 * control + rng.standard_normal(n_rows)
    )

    print("=" * 72)
    print("orthogonal demo: planted world (alpha real, beta null)")
    print("=" * 72)
    report = orthogonal_check(
        outcome,
        {"alpha_real": pd.Series(real), "beta_null": pd.Series(null)},
        {"gamma_mom": pd.Series(control)},
        label="orthogonal demo",
    )
    facts = report["facts"]
    print(f"status={report['status']}  detected={facts['detected']}")
    print(f"delta_r2_block={report['delta_r2_block']:.4f}")
    print(
        f"strongest={facts['strongest_feature']}"
        f" t={facts['strongest_t']:.2f}"
        f" threshold={facts['t_threshold']}"
    )
    print(
        "per-feature delta_r2:",
        {
            name: round(value, 4)
            for name, value in report["per_feature_delta_r2"].items()
        },
    )
    print("permutation p:", report["permutation"])
    # layer-0 verification: everything printed above must hold
    assert report["status"] == "OK"
    assert report["n_before_dropna"] == n_rows
    assert report["n_dropped"] == 0
    assert facts["detected"] is True
    assert facts["sign_consistent"] is True
    assert facts["strongest_feature"] == "alpha_real"
    assert facts["strongest_t"] == report["full"]["t_stat"]["alpha_real"]
    assert report["delta_r2_block"] > 0.0
    assert report["full"]["t_stat"]["alpha_real"] > facts["t_threshold"]
    assert report["full"]["t_stat"]["beta_null"] < facts["t_threshold"]
    assert (
        report["per_feature_delta_r2"]["alpha_real"]
        > report["per_feature_delta_r2"]["beta_null"]
    )
    assert report["permutation"]["alpha_real"] <= 0.05


if __name__ == "__main__":
    run_all_examples()
