"""Incremental (HC0-robust) regression: is a new signal anything NEW?

Two kill rules, deliberately different in strictness:

* :func:`incremental_test` is the STRICT gate: one canonical controlled
  spec with a four-way verdict (full table in its docstring):
  ``SURVIVES`` (finite ``|t_new| >= 2``), ``REPACKAGING-DEATH``
  (controls present, ``|t_new| < 2`` - the signal repackages what the
  controls already know), ``NULL-DEATH`` (no controls, ``|t_new| < 2``
  - no detectable effect at all, nothing is being repackaged) and
  ``INCONCLUSIVE`` (degenerate full model: rank-deficient / saturated
  design, non-finite ``t_new``, constant target, or no rows surviving
  the NaN alignment - unidentifiable, no evidence either way).
* :func:`specification_curve` / :func:`curve_summary` are the BREADTH
  diagnostic: the curve spans control sets x rank / z-score / winsorized
  / demeaned transforms, and the signal dies only if NO spec in the grid
  reaches |t| >= 2.  ``n_surviving_controlled`` counts the specs that
  survive WITH controls - a curve kept alive purely by uncontrolled
  specs is a repackaging flag even when the lenient verdict says
  SURVIVES.  A spec whose model is unidentifiable (collinear design,
  zero-sd z-score, constant outcome, empty alignment) gets a per-spec
  verdict of ``INCONCLUSIVE``; its non-finite ``t_new`` is excluded
  from the curve statistics instead of silently counting as a death.

p-values use the t distribution with ``df = n - k`` degrees of freedom
-- an asymptotic approximation for HC0 standard errors, kept as the
house convention.

Functions
---------
``incremental_test``     one canonical spec: controls vs controls + new
                         signal.
``specification_curve``  grid of specs; returns a DataFrame with ``t_new``
                         and a per-spec verdict.
``curve_summary``        overall SURVIVES / REPACKAGING-DEATH call for the
                         curve.

All pandas inputs are index-aligned (inner join); numpy inputs align by
position.  NaN rows are dropped pairwise; when NO row survives the
drop, public functions return an honest INCONCLUSIVE instead of
crashing or killing the hypothesis on zero observations.

Run ``python -m atropos.incremental_regression`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from atropos._utils import align_inputs, ols_hc0

__all__ = [
    "DEFAULT_TRANSFORMS",
    "curve_summary",
    "incremental_test",
    "specification_curve",
]

DEFAULT_TRANSFORMS: tuple[str, ...] = (
    "raw",
    "rank",
    "zscore",
    "winsorize",
    "demean",
)
_T_RULE = 2.0
_SURVIVES = "SURVIVES"
_DEATH = "REPACKAGING-DEATH"
_NULL_DEATH = "NULL-DEATH"
_INCONCLUSIVE = "INCONCLUSIVE"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _fit(
    target: np.ndarray, design: np.ndarray, robust: bool = True
) -> dict[str, Any]:
    """OLS fit; HC0 (``robust``) or classical (non-robust) standard errors.

    The HC0 path delegates to :func:`atropos._utils.ols_hc0`; the
    classical path mirrors its identification contract: when the design
    is saturated (``n_rows <= n_cols``) or rank-deficient, every
    statistic is NaN and ``rank_deficient`` is True.  A pseudo-inverse
    alone would fabricate plausible-looking minimum-norm t-stats
    exactly in the collinear cases the framework must flag, so callers
    must treat ``rank_deficient`` (or a non-finite ``t``) as
    INCONCLUSIVE, never as evidence for or against the hypothesis.
    """
    if robust:
        return ols_hc0(target, design)
    n_rows, n_cols = design.shape
    # Deliberate duplication of the rank gate in
    # atropos._utils.ols_hc0 (kept in sync by hand): the classical
    # path must return the SAME NaN-payload contract as the HC0 path,
    # and the shared helper stays robust-only rather than growing a
    # classical branch used solely by this caller.
    if n_rows <= n_cols or np.linalg.matrix_rank(design) < n_cols:
        nan_vector = np.full(n_cols, np.nan)
        return {
            "beta": nan_vector,
            "se": nan_vector.copy(),
            "t": nan_vector.copy(),
            "p": nan_vector.copy(),
            "df": max(n_rows - n_cols, 1),
            "n": n_rows,
            "r2": float("nan"),
            "rank_deficient": True,
        }
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    s2 = float(resid @ resid) / (n_rows - n_cols)
    cov = s2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    df = n_rows - n_cols
    pvalues = 2.0 * stats.t.sf(np.abs(t_stats), df)
    sst = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {
        "beta": np.asarray(beta, dtype=float),
        "se": np.asarray(se, dtype=float),
        "t": np.asarray(t_stats, dtype=float),
        "p": np.asarray(pvalues, dtype=float),
        "df": df,
        "n": n_rows,
        "r2": r2,
        "rank_deficient": False,
    }


def _validate_winsor_pct(value: float, label: str) -> None:
    """Reject ``winsor_pct`` outside the open interval (0, 0.5).

    The winsorize transform clips to ``[quantile(p), quantile(1-p)]``;
    at ``p >= 0.5`` those bounds invert (``lo > hi``) and ``np.clip``
    silently collapses the signal onto the upper bound - a degenerate
    transform that must fail loudly instead of fabricating a spec.
    """
    if not 0.0 < value < 0.5:
        raise ValueError(
            f"{label} must satisfy 0 < winsor_pct < 0.5, got {value}"
            " (at >= 0.5 the winsorize quantile bounds invert)"
        )


def _transform_signal(
    signal: np.ndarray, kind: str, winsor_pct: float
) -> np.ndarray:
    if kind == "raw":
        return signal.copy()
    if kind == "rank":
        ranks: np.ndarray = stats.rankdata(signal, method="average")
        return ranks / signal.size
    if kind == "zscore":
        sd = float(np.std(signal))
        if sd > 0:
            centered: np.ndarray = (signal - float(np.mean(signal))) / sd
            return centered
        # sd == 0 (constant signal): returning zeros would turn the
        # spec into a regression on a constant - a dummy spec that
        # silently changes the hypothesis.  NaN marks the spec as
        # degenerate so callers report INCONCLUSIVE.
        return np.full(signal.shape, np.nan)
    if kind == "winsorize":
        lo = float(np.quantile(signal, winsor_pct))
        hi = float(np.quantile(signal, 1.0 - winsor_pct))
        clipped: np.ndarray = np.clip(signal, lo, hi)
        return clipped
    if kind == "demean":
        return signal - float(np.mean(signal))
    raise ValueError(
        f"unknown transform {kind!r}; expected one of {DEFAULT_TRANSFORMS}"
    )


# ----------------------------------------------------------------------------
# canonical incremental test
# ----------------------------------------------------------------------------
def _inconclusive_result(robust: bool, n_obs: int = 0) -> dict[str, Any]:
    """NaN payload with verdict INCONCLUSIVE for degenerate inputs.

    ``n_obs`` is the aligned row count to report; it defaults to 0
    (the historical empty-alignment case) so future callers cannot
    accidentally report observations they never fitted.
    """
    return {
        "t_new": float("nan"),
        "delta_r2": float("nan"),
        "verdict": _INCONCLUSIVE,
        "coef_new": float("nan"),
        "p_new": float("nan"),
        "r2_restricted": float("nan"),
        "r2_full": float("nan"),
        "n": int(n_obs),
        "robust": bool(robust),
    }


def incremental_test(
    fwd_ret: Any,
    known_controls_df: Any,
    new_signal: Any,
    robust: bool = True,
) -> dict[str, Any]:
    """Does the new signal add anything after the known controls?

    Fits restricted ``fwd_ret ~ const + controls`` and full
    ``fwd_ret ~ const + controls + new_signal``.  Verdict table:

    * ``SURVIVES`` -- finite ``|t_new| >= 2`` in the full model.
    * ``REPACKAGING-DEATH`` -- known controls are present (at least one
      aligned control column) and ``|t_new| < 2``: the signal adds
      nothing beyond what the controls already know; it is a
      repackaging of known information.
    * ``NULL-DEATH`` -- no controls (``known_controls_df=None`` or an
      empty control set) and ``|t_new| < 2``: there is no detectable
      effect at all - nothing is being repackaged, the signal is simply
      null.
    * ``INCONCLUSIVE`` -- the full model is degenerate and the test is
      unidentifiable: rank-deficient / saturated design (e.g. the
      signal is EXACTLY collinear with a control - a pseudo-inverse
      would fabricate a plausible arbitrary ``t_new`` here), non-finite
      ``t_new``, constant ``fwd_ret``, or an empty NaN alignment
      (``n == 0``).  No evidence either way; do not treat as a kill.

    Note: ``p``-values use the t distribution with ``df = n - k``
    degrees of freedom -- an asymptotic approximation when HC0 robust
    standard errors are used, kept as the house convention.

    Returns ``{t_new, delta_r2, verdict, coef_new, p_new, r2_restricted,
    r2_full, n, robust}``; degenerate cases return NaN statistics with
    ``verdict="INCONCLUSIVE"``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 800  # doctest: +SKIP
        >>> c = rng.standard_normal((n, 2))  # doctest: +SKIP
        >>> z = rng.standard_normal(n)  # doctest: +SKIP
        >>> y = (0.3 * z + 0.5 * c[:, 0]  # doctest: +SKIP
        ...      + rng.standard_normal(n))
        >>> ctrl = pd.DataFrame(c, columns=["c0", "c1"])  # doctest: +SKIP
        >>> incremental_test(y, ctrl, z)["verdict"]  # doctest: +SKIP
        'SURVIVES'
    """
    if known_controls_df is None:
        outcome, sig = align_inputs(fwd_ret, new_signal)
        controls = np.empty((outcome.size, 0))
    else:
        outcome, sig, controls = align_inputs(
            fwd_ret, new_signal, known_controls_df
        )
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

    has_controls = controls.shape[1] > 0
    if outcome.size == 0:
        # nothing survived the NaN alignment: honest INCONCLUSIVE
        # instead of a crash or a silent zero-row kill
        return _inconclusive_result(robust, n_obs=int(outcome.size))

    if has_controls:
        design_reduced = np.column_stack([np.ones(outcome.size), controls])
        design_full = np.column_stack([np.ones(outcome.size), sig, controls])
    else:
        design_reduced = np.ones((outcome.size, 1))
        design_full = np.column_stack([np.ones(outcome.size), sig])

    res_r = _fit(outcome, design_reduced, robust=robust)
    res_f = _fit(outcome, design_full, robust=robust)
    t_new = float(res_f["t"][1])
    p_new = float(res_f["p"][1])
    delta_r2 = float(res_f["r2"] - res_r["r2"])
    constant_target = float(np.ptp(outcome)) == 0.0
    if (
        bool(res_f["rank_deficient"])
        or not np.isfinite(t_new)
        or constant_target
    ):
        verdict = _INCONCLUSIVE
    elif abs(t_new) >= _T_RULE:
        verdict = _SURVIVES
    else:
        verdict = _DEATH if has_controls else _NULL_DEATH
    return {
        "t_new": t_new,
        "delta_r2": delta_r2,
        "verdict": verdict,
        "coef_new": float(res_f["beta"][1]),
        "p_new": p_new,
        "r2_restricted": float(res_r["r2"]),
        "r2_full": float(res_f["r2"]),
        "n": int(outcome.size),
        "robust": bool(robust),
    }


# ----------------------------------------------------------------------------
# specification curve
# ----------------------------------------------------------------------------
def _resolve_controls(
    controls_index: Any, controls_list: list[Any], spec_no: int
) -> Any:
    """Map a spec's ``controls`` key to a control set or None.

    ``None`` means no controls; a valid integer indexes
    ``controls_list``.  Anything else - out-of-range, negative, boolean
    or non-integer values - raises ``ValueError``: silently falling
    back to an uncontrolled specification would quietly change the
    hypothesis being tested.
    """
    if controls_index is None:
        return None
    if (
        isinstance(controls_index, int)
        and not isinstance(controls_index, bool)
        and 0 <= controls_index < len(controls_list)
    ):
        return controls_list[controls_index]
    raise ValueError(
        f"spec {spec_no}: controls={controls_index!r} is not a valid "
        f"index into controls_list (len={len(controls_list)}); use "
        "None for no controls or an in-range integer index"
    )


def specification_curve(
    fwd_ret: Any,
    controls_list: Any,
    signal: Any,
    *,
    specs_to_try: list[dict[str, Any]] | None = None,
    robust: bool = True,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Table of ``t_new`` across a grid of reasonable specifications.

    Parameters
    ----------
    controls_list : list of control sets; each entry is a DataFrame /
        (n, m) array (or None).  A spec references one entry by integer
        index, or ``None`` for no controls; an out-of-range or
        non-integer index raises ``ValueError`` (a mistyped index must
        not silently become an uncontrolled spec).
    specs_to_try : list of ``{"controls": int | None, "transform": str,
        "winsor_pct": float}`` dicts.  ``transform`` is one of
        ``raw | rank | zscore | winsorize | demean``.  When None, the
        full grid ``transforms x (None + each control set)`` is used.
    robust, winsor_pct : keyword-only tuning knobs shared by every
        spec.  ``winsor_pct`` (top level and per spec) must satisfy
        ``0 < winsor_pct < 0.5``, else ``ValueError``: at ``>= 0.5``
        the winsorize quantile bounds invert and the clip degenerates.

    Per-spec verdicts are ``SURVIVES`` / ``REPACKAGING-DEATH`` by the
    ``|t_new| >= 2`` rule, or ``INCONCLUSIVE`` when the spec's full
    model is unidentifiable (rank-deficient / saturated design,
    degenerate transform such as a zero-sd z-score, constant outcome,
    or empty NaN alignment); degenerate specs carry a NaN ``t_new``
    which :func:`curve_summary` excludes from the curve statistics.

    The kill rule: if ``max |t_new| < 2`` across every spec in the curve,
    the signal is a repackaging and the hypothesis dies
    (see :func:`curve_summary`).

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 800  # doctest: +SKIP
        >>> y = rng.standard_normal(n)  # doctest: +SKIP
        >>> z = rng.standard_normal(n)  # doctest: +SKIP
        >>> c = rng.standard_normal(n)  # doctest: +SKIP
        >>> ctrl = [pd.DataFrame({"c": c})]  # doctest: +SKIP
        >>> table = specification_curve(y, ctrl, z)  # doctest: +SKIP
        >>> set(table.columns) >= {"t_new", "verdict"}  # doctest: +SKIP
        True
    """
    controls_list = list(controls_list) if controls_list is not None else []
    _validate_winsor_pct(float(winsor_pct), "winsor_pct")
    if specs_to_try is None:
        specs_to_try = [
            {
                "controls": ctrl_idx,
                "transform": transform,
                "winsor_pct": winsor_pct,
            }
            for transform in DEFAULT_TRANSFORMS
            for ctrl_idx in [None, *range(len(controls_list))]
        ]

    rows: list[dict[str, Any]] = []
    for spec_no, spec in enumerate(specs_to_try):
        sp = dict(spec)
        ci = sp.get("controls")
        tr = sp.get("transform", "raw")
        wp = float(sp.get("winsor_pct", winsor_pct))
        _validate_winsor_pct(wp, f"spec {spec_no}: winsor_pct")
        ctrl = _resolve_controls(ci, controls_list, spec_no)

        if ctrl is None:
            outcome, sig = align_inputs(fwd_ret, signal)
            controls = np.empty((outcome.size, 0))
        else:
            outcome, sig, controls = align_inputs(fwd_ret, signal, ctrl)
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)

        sig_t = _transform_signal(sig, tr, wp)
        row: dict[str, Any] = {
            "spec": spec_no,
            "transform": tr,
            "controls": "none" if ctrl is None else f"set{ci}",
            "n": int(outcome.size),
        }
        if (
            outcome.size == 0
            or not np.isfinite(sig_t).all()
            or float(np.ptp(outcome)) == 0.0
        ):
            # degenerate spec (empty alignment, constant signal after
            # the transform, or constant outcome): honest INCONCLUSIVE
            # row, never a fabricated death
            row.update(
                t_new=float("nan"),
                p_new=float("nan"),
                delta_r2=float("nan"),
                verdict=_INCONCLUSIVE,
            )
            rows.append(row)
            continue
        if controls.shape[1] > 0:
            design_full = np.column_stack(
                [np.ones(outcome.size), sig_t, controls]
            )
            design_reduced = np.column_stack(
                [np.ones(outcome.size), controls]
            )
        else:
            design_full = np.column_stack([np.ones(outcome.size), sig_t])
            design_reduced = np.ones((outcome.size, 1))

        res_f = _fit(outcome, design_full, robust=robust)
        res_r = _fit(outcome, design_reduced, robust=robust)
        t_new = float(res_f["t"][1])
        if bool(res_f["rank_deficient"]) or not np.isfinite(t_new):
            spec_verdict = _INCONCLUSIVE
        else:
            spec_verdict = _SURVIVES if abs(t_new) >= _T_RULE else _DEATH
        row.update(
            t_new=t_new,
            p_new=float(res_f["p"][1]),
            delta_r2=float(res_f["r2"] - res_r["r2"]),
            verdict=spec_verdict,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def curve_summary(spec_table: pd.DataFrame) -> dict[str, Any]:
    """Overall SURVIVES / REPACKAGING-DEATH call for a specification curve.

    The rule (lenient by design - the strict gate is
    :func:`incremental_test`): if NO reasonable spec pushes
    ``|t_new| >= 2``, the signal is a repackaging of known information ->
    death.  ``n_surviving_controlled`` additionally counts the specs that
    survive WITH known controls in the model; 0 surviving controlled
    specs means the curve is kept alive by uncontrolled specs only - a
    repackaging flag even when the verdict says SURVIVES.  A spec curve
    where specs flip sign is reported via ``sign_consistent``.

    ``sign_consistent`` is judged on the NONZERO t-values: an exact 0.0
    t-stat carries no sign information (``np.sign(0.0) == 0.0`` would
    otherwise force any curve containing one to look inconsistent), so
    zeros are ignored and a curve with no nonzero t at all is trivially
    consistent.

    Non-finite ``t_new`` entries (degenerate specs, see
    :func:`specification_curve`) are excluded from every statistic.

    Example::

        >>> table = specification_curve(y, ctrl_sets, z)   # doctest: +SKIP
        >>> curve_summary(table)["verdict"]                # doctest: +SKIP
        'SURVIVES'
    """
    t_vals = spec_table["t_new"].to_numpy(dtype=float)
    finite = np.isfinite(t_vals)
    t_vals = t_vals[finite]
    n_specs = int(t_vals.size)
    max_abs_t = float(np.max(np.abs(t_vals))) if n_specs else float("nan")
    min_abs_t = float(np.min(np.abs(t_vals))) if n_specs else float("nan")
    n_surviving = int(np.sum(np.abs(t_vals) >= _T_RULE)) if n_specs else 0
    controls_col = (
        spec_table["controls"].to_numpy()
        if "controls" in spec_table.columns
        else np.full(len(spec_table), "none", dtype=object)
    )
    controlled = (
        (controls_col != "none")
        & finite
        & (np.abs(spec_table["t_new"].to_numpy(dtype=float)) >= _T_RULE)
    )
    n_surviving_controlled = int(np.sum(controlled))
    nonzero_t = t_vals[t_vals != 0.0]
    sign_consistent = bool(
        n_specs > 0
        and (
            nonzero_t.size == 0
            or np.all(np.sign(nonzero_t) == np.sign(nonzero_t[0]))
        )
    )
    verdict = _SURVIVES if (n_specs > 0 and max_abs_t >= _T_RULE) else _DEATH
    return {
        "verdict": verdict,
        "n_specs": n_specs,
        "n_surviving": n_surviving,
        "n_surviving_controlled": n_surviving_controlled,
        "max_abs_t": max_abs_t,
        "min_abs_t": min_abs_t,
        "sign_consistent": sign_consistent,
        "t_rule": _T_RULE,
    }


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _verify_genuine_edge_case(
    result: dict[str, Any],
    table: pd.DataFrame,
    summary: dict[str, Any],
    n_days: int,
) -> None:
    """Assert the demo-A contract: a genuine edge survives everywhere.

    Seed-independent properties only: the exact seeded numbers
    (``t_new`` ~ 4.95, all 10 specs surviving, 5 controlled survivors)
    are locked in ``tests/test_incremental_regression.py``
    (``test_seeded_regression_demo_*``).
    """
    assert result["verdict"] == "SURVIVES"
    assert np.isfinite(result["t_new"])
    assert result["delta_r2"] > 0.0
    assert result["n"] == n_days - 12  # the lagged control's NaN head drops
    assert summary["verdict"] == "SURVIVES"
    assert summary["n_specs"] == 10  # 5 transforms x (none + set0)
    assert summary["n_surviving"] >= 1
    # the edge survives WITH controls, not on uncontrolled specs alone
    assert summary["n_surviving_controlled"] >= 1
    assert summary["max_abs_t"] >= 2.0
    assert int(table.loc[table["controls"] == "set0", "n"].min()) == (
        n_days - 12
    )


def _verify_repackaged_death_case(
    result: dict[str, Any],
    table: pd.DataFrame,
    summary: dict[str, Any],
    n_days: int,
) -> None:
    """Assert the demo-B contract: strict gate kills, curve flags it.

    Seed-independent properties only: the exact seeded numbers
    (``t_new`` ~ 0.71, exactly 5 uncontrolled survivors) are locked in
    ``tests/test_incremental_regression.py``
    (``test_seeded_regression_demo_*``).
    """
    assert result["verdict"] == "REPACKAGING-DEATH"
    assert np.isfinite(result["t_new"])
    assert result["n"] == n_days - 12
    # the lenient curve stays alive on UNCONTROLLED specs only ...
    assert summary["verdict"] == "SURVIVES"  # lenient by design
    assert summary["n_surviving"] >= 1  # kept alive by uncontrolled specs
    # ... zero controlled survivors is the repackaging flag
    assert summary["n_surviving_controlled"] == 0
    assert (
        table.loc[table["controls"] == "set0", "verdict"]
        == "REPACKAGING-DEATH"
    ).all()


def run_all_examples() -> None:
    """Demo: a genuine edge survives the curve; repackaged momentum dies.

    The demo doubles as the layer-0 verification test: every printed
    outcome below is asserted, so a regression crashes the demo instead
    of printing a wrong number that nobody notices.

    Controls are two honest momentum measurements:

    * ``mom_noisy`` - contemporaneous momentum observed with a little
      measurement noise (NOT an independent 7-day momentum series);
    * ``mom12`` - the SAME momentum lagged 12 periods via a NaN-headed
      shift.  ``np.roll`` would wrap the last 12 (future) values into
      the head of the series - a lookahead leak; the NaN head here is
      dropped by the NaN alignment instead.
    """
    rng = np.random.default_rng(3)
    n_days = 900
    mom = np.zeros(n_days)
    shock = rng.standard_normal(n_days)
    for day in range(1, n_days):
        mom[day] = 0.6 * mom[day - 1] + 0.4 * shock[day]
    driver = rng.standard_normal(n_days)
    mom12 = np.concatenate([np.full(12, np.nan), mom[:-12]])

    # Case A: real edge, signal contaminated with momentum but it survives.
    fwd_a = 0.20 * driver + 0.30 * mom + rng.standard_normal(n_days)
    sig_a = driver + 0.5 * mom
    mom_df = pd.DataFrame(
        {
            "mom_noisy": mom + rng.standard_normal(n_days) * 0.1,
            "mom12": mom12,
        },
        index=range(n_days),
    )

    print("=" * 72)
    print("incremental_regression demo A: genuine edge")
    print("=" * 72)
    res = incremental_test(fwd_a, mom_df, sig_a, robust=True)
    print(
        f"incremental_test: t_new={res['t_new']:.2f}"
        f" delta_r2={res['delta_r2']:.4f}"
        f" verdict={res['verdict']}"
    )
    table = specification_curve(fwd_a, [mom_df], sig_a, robust=True)
    with pd.option_context("display.width", 120, "display.max_columns", 10):
        print(
            table.to_string(
                index=False, float_format=lambda val: f"{val:.3f}"
            )
        )
    summary_a = curve_summary(table)
    print("curve_summary:", summary_a)
    # layer-0 verification: everything printed above must hold
    _verify_genuine_edge_case(res, table, summary_a, n_days)

    # Case B: "new" signal is 95% repackaged momentum + noise -> death.
    fwd_b = 0.35 * mom + rng.standard_normal(n_days)
    sig_b = mom + rng.standard_normal(n_days) * 0.35

    print()
    print("=" * 72)
    print(
        "incremental_regression demo B: repackaged momentum (expected death)"
    )
    print("=" * 72)
    res_b = incremental_test(fwd_b, mom_df, sig_b, robust=True)
    print(
        f"incremental_test: t_new={res_b['t_new']:.2f}"
        f" delta_r2={res_b['delta_r2']:.4f}"
        f" verdict={res_b['verdict']}"
    )
    table_b = specification_curve(fwd_b, [mom_df], sig_b, robust=True)
    summary_b = curve_summary(table_b)
    print("curve_summary:", summary_b)
    print(
        "note: the curve keeps uncontrolled specs alive (momentum predicts"
        " itself);"
    )
    print(
        f"      the kill comes from the CONTROLLED test above"
        f" (t_new={res_b['t_new']:.2f} -> {res_b['verdict']})."
    )
    # layer-0 verification: the strict gate kills, the lenient curve
    # exposes the repackaging via zero controlled survivors
    _verify_repackaged_death_case(res_b, table_b, summary_b, n_days)


if __name__ == "__main__":
    run_all_examples()
