"""Incremental (HC0-robust) regression: is a new signal anything NEW?

Two kill rules, deliberately different in strictness:

* :func:`incremental_test` is the STRICT gate: one canonical controlled
  spec - if the new signal's |t| < 2 once the known controls are in the
  model, the signal is a REPACKAGING of what the model already knows and
  the hypothesis dies.
* :func:`specification_curve` / :func:`curve_summary` are the BREADTH
  diagnostic: the curve spans control sets x rank / z-score / winsorized
  / demeaned transforms, and the signal dies only if NO spec in the grid
  reaches |t| >= 2.  ``n_surviving_controlled`` counts the specs that
  survive WITH controls - a curve kept alive purely by uncontrolled
  specs is a repackaging flag even when the lenient verdict says
  SURVIVES.

Functions
---------
``incremental_test``     one canonical spec: controls vs controls + new
                         signal.
``specification_curve``  grid of specs; returns a DataFrame with ``t_new``
                         and a per-spec verdict.
``curve_summary``        overall SURVIVES / REPACKAGING-DEATH call for the
                         curve.

All pandas inputs are index-aligned (inner join); numpy inputs align by
position.  NaN rows are dropped pairwise.

Run ``python -m atropos.incremental_regression`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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


# ----------------------------------------------------------------------------
# helpers (self-contained: duplicated so the module imports alone)
# ----------------------------------------------------------------------------
def _frameify(source: Any, tag: str) -> pd.DataFrame:
    if isinstance(source, pd.Series):
        return source.to_frame(name=tag)
    if isinstance(source, pd.DataFrame):
        return source.add_prefix(tag + "::")
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 1:
        return pd.DataFrame({tag: arr})
    if arr.ndim != 2:
        raise ValueError(
            f"input {tag!r} must be 1-D or 2-D, got ndim={arr.ndim}"
        )
    return pd.DataFrame(
        arr, columns=[f"{tag}::{col}" for col in range(arr.shape[1])]
    )


def _align(*arrays: Any) -> list[np.ndarray]:
    """Align pandas inputs by index, numpy by position; drop NaN rows.

    Mixing pandas and numpy inputs is allowed: numpy arrays inherit the
    index of the first pandas input when lengths match (positional
    alignment).
    """
    frames = [_frameify(item, f"v{slot}") for slot, item in enumerate(arrays)]
    ref_index = None
    for item in arrays:
        if isinstance(item, (pd.Series, pd.DataFrame)) and not isinstance(
            item.index, pd.RangeIndex
        ):
            ref_index = item.index
            break
    if ref_index is not None:
        # frames is built 1:1 from arrays, so the strict zip is safe
        for slot, (item, frame) in enumerate(
            zip(arrays, frames, strict=True)
        ):
            if not isinstance(item, (pd.Series, pd.DataFrame)):
                if len(frame) == len(ref_index):
                    frame.index = ref_index
                else:
                    raise ValueError(
                        f"input {slot} (numpy, len={len(frame)}) cannot"
                        " be aligned with the pandas index"
                        f" (len={len(ref_index)}); pass pandas inputs or"
                        " equal-length arrays"
                    )
    comb = pd.concat(frames, axis=1, join="inner").dropna()
    out: list[np.ndarray] = []
    pos = 0
    for frame in frames:
        width = frame.shape[1]
        block = comb.iloc[:, pos : pos + width].to_numpy(dtype=float)
        out.append(block.reshape(-1) if width == 1 else block)
        pos += width
    return out


def _fit(
    target: np.ndarray, design: np.ndarray, robust: bool = True
) -> dict[str, Any]:
    n_rows, n_cols = design.shape
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    if robust:
        design_res = design * resid[:, None]  # X' diag(e^2) X = (X*e)'(X*e)
        meat = design_res.T @ design_res
        cov = xtx_inv @ meat @ xtx_inv
    else:
        s2 = float(resid @ resid) / max(n_rows - n_cols, 1)
        cov = s2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    df = max(n_rows - n_cols, 1)
    pvalues = 2.0 * stats.t.sf(np.abs(t_stats), df)
    sst = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {
        "beta": beta,
        "se": se,
        "t": t_stats,
        "p": pvalues,
        "n": n_rows,
        "r2": r2,
    }


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
        return (
            (signal - float(np.mean(signal))) / sd
            if sd > 0
            else np.zeros_like(signal)
        )
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
def incremental_test(
    fwd_ret: Any,
    known_controls_df: Any,
    new_signal: Any,
    robust: bool = True,
) -> dict[str, Any]:
    """Does the new signal add anything after the known controls?

    Fits restricted ``fwd_ret ~ const + controls`` and full
    ``fwd_ret ~ const + controls + new_signal``.  Verdict is ``SURVIVES``
    when ``|t_new| >= 2`` and ``REPACKAGING-DEATH`` otherwise (the kill
    rule).

    Returns ``{t_new, delta_r2, verdict, coef_new, p_new, r2_restricted,
    r2_full, n, robust}``.

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
        outcome, sig = _align(fwd_ret, new_signal)
        controls = np.empty((outcome.size, 0))
    else:
        outcome, sig, controls = _align(
            fwd_ret, new_signal, known_controls_df
        )
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

    if controls.shape[1] > 0:
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
    verdict = _SURVIVES if abs(t_new) >= _T_RULE else _DEATH
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
        index, or ``None`` for no controls.
    specs_to_try : list of ``{"controls": int | None, "transform": str,
        "winsor_pct": float}`` dicts.  ``transform`` is one of
        ``raw | rank | zscore | winsorize | demean``.  When None, the
        full grid ``transforms x (None + each control set)`` is used.
    robust, winsor_pct : keyword-only tuning knobs shared by every spec.

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
        ctrl = (
            controls_list[ci]
            if isinstance(ci, int) and 0 <= ci < len(controls_list)
            else None
        )

        if ctrl is None:
            outcome, sig = _align(fwd_ret, signal)
            controls = np.empty((outcome.size, 0))
        else:
            outcome, sig, controls = _align(fwd_ret, signal, ctrl)
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)

        sig_t = _transform_signal(sig, tr, wp)
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
        rows.append(
            {
                "spec": spec_no,
                "transform": tr,
                "controls": "none" if ctrl is None else f"set{ci}",
                "n": int(outcome.size),
                "t_new": t_new,
                "p_new": float(res_f["p"][1]),
                "delta_r2": float(res_f["r2"] - res_r["r2"]),
                "verdict": _SURVIVES if abs(t_new) >= _T_RULE else _DEATH,
            }
        )
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
    sign_consistent = bool(
        n_specs > 0 and np.all(np.sign(t_vals) == np.sign(t_vals[0]))
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
def run_all_examples() -> None:
    """Demo: a genuine edge survives the curve; repackaged momentum dies."""
    rng = np.random.default_rng(3)
    n_days = 900
    mom = np.zeros(n_days)
    shock = rng.standard_normal(n_days)
    for day in range(1, n_days):
        mom[day] = 0.6 * mom[day - 1] + 0.4 * shock[day]
    driver = rng.standard_normal(n_days)

    # Case A: real edge, signal contaminated with momentum but it survives.
    fwd_a = 0.20 * driver + 0.30 * mom + rng.standard_normal(n_days)
    sig_a = driver + 0.5 * mom
    mom_df = pd.DataFrame(
        {
            "mom7": mom + rng.standard_normal(n_days) * 0.1,
            "mom30": np.roll(mom, 12),
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
    print("curve_summary:", curve_summary(table))

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
    print("curve_summary:", curve_summary(table_b))
    print(
        "note: the curve keeps uncontrolled specs alive (momentum predicts"
        " itself);"
    )
    print(
        "      the kill comes from the CONTROLLED test above (t_new=0.53"
        " -> DEATH)."
    )


if __name__ == "__main__":
    run_all_examples()
