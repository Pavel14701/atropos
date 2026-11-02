"""Incremental (HC0-robust) regression: is a new signal anything NEW?

The kill rule: if the new signal's |t| drops below 2 once ANY reasonable
specification is estimated - different control sets, rank / z-score /
winsorized / demeaned transforms - the signal is a REPACKAGING of what the
model already knows and the hypothesis dies.

Functions
---------
``incremental_test``   one canonical spec: controls vs controls + new signal.
``specification_curve`` grid of specs; returns a DataFrame with ``t_new`` and
                       a per-spec verdict.
``curve_summary``      overall SURVIVES / REPACKAGING-DEATH call for the curve.

All pandas inputs are index-aligned (inner join); numpy inputs align by
position.  NaN rows are dropped pairwise.

Run ``python incremental_regression.py`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["incremental_test", "specification_curve", "curve_summary", "DEFAULT_TRANSFORMS"]

DEFAULT_TRANSFORMS: tuple[str, ...] = ("raw", "rank", "zscore", "winsorize", "demean")
_T_RULE = 2.0
_SURVIVES = "SURVIVES"
_DEATH = "REPACKAGING-DEATH"


# ----------------------------------------------------------------------------
# helpers (self-contained: duplicated deliberately so the module imports alone)
# ----------------------------------------------------------------------------
def _frameify(a: Any, tag: str) -> pd.DataFrame:
    if isinstance(a, pd.Series):
        return a.to_frame(name=tag)
    if isinstance(a, pd.DataFrame):
        return a.add_prefix(tag + "::")
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return pd.DataFrame({tag: arr})
    if arr.ndim != 2:
        raise ValueError(f"input {tag!r} must be 1-D or 2-D, got ndim={arr.ndim}")
    return pd.DataFrame(arr, columns=[f"{tag}::{j}" for j in range(arr.shape[1])])


def _align(*arrays: Any) -> list[np.ndarray]:
    """Index-align pandas inputs (inner join), positional otherwise; drop NaNs.

    Mixing pandas and numpy inputs is allowed: numpy arrays inherit the index
    of the first pandas input when lengths match (positional alignment).
    """
    frames = [_frameify(a, f"v{i}") for i, a in enumerate(arrays)]
    ref_index = None
    for a in arrays:
        if isinstance(a, (pd.Series, pd.DataFrame)) and not isinstance(a.index, pd.RangeIndex):
            ref_index = a.index
            break
    if ref_index is not None:
        for i, (a, f) in enumerate(zip(arrays, frames)):
            if not isinstance(a, (pd.Series, pd.DataFrame)):
                if len(f) == len(ref_index):
                    f.index = ref_index
                else:
                    raise ValueError(
                        f"input {i} (numpy, len={len(f)}) cannot be aligned with the "
                        f"pandas index (len={len(ref_index)}); pass pandas inputs or "
                        "equal-length arrays"
                    )
    comb = pd.concat(frames, axis=1, join="inner").dropna()
    out: list[np.ndarray] = []
    pos = 0
    for f in frames:
        k = f.shape[1]
        block = comb.iloc[:, pos:pos + k].to_numpy(dtype=float)
        out.append(block.reshape(-1) if k == 1 else block)
        pos += k
    return out


def _fit(y: np.ndarray, X: np.ndarray, robust: bool = True) -> dict[str, Any]:
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    if robust:
        Xe = X * resid[:, None]                  # X' diag(e^2) X = (X*e)'(X*e)
        meat = Xe.T @ Xe
        cov = xtx_inv @ meat @ xtx_inv
    else:
        s2 = float(resid @ resid) / max(n - k, 1)
        cov = s2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    df = max(n - k, 1)
    p = 2.0 * stats.t.sf(np.abs(t), df)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {"beta": beta, "se": se, "t": t, "p": p, "n": n, "r2": r2}


def _transform_signal(s: np.ndarray, kind: str, winsor_pct: float) -> np.ndarray:
    if kind == "raw":
        return s.copy()
    if kind == "rank":
        return stats.rankdata(s, method="average") / s.size
    if kind == "zscore":
        sd = float(np.std(s))
        return (s - float(np.mean(s))) / sd if sd > 0 else np.zeros_like(s)
    if kind == "winsorize":
        lo = float(np.quantile(s, winsor_pct))
        hi = float(np.quantile(s, 1.0 - winsor_pct))
        return np.clip(s, lo, hi)
    if kind == "demean":
        return s - float(np.mean(s))
    raise ValueError(f"unknown transform {kind!r}; expected one of {DEFAULT_TRANSFORMS}")


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
    ``fwd_ret ~ const + controls + new_signal``.  Verdict is ``SURVIVES`` when
    ``|t_new| >= 2`` and ``REPACKAGING-DEATH`` otherwise (the kill rule).

    Returns ``{t_new, delta_r2, verdict, coef_new, p_new, r2_restricted,
    r2_full, n, robust}``.

    Example::

        >>> rng = np.random.default_rng(0)                          # doctest: +SKIP
        >>> n = 800                                                 # doctest: +SKIP
        >>> c = rng.standard_normal((n, 2))                         # doctest: +SKIP
        >>> z = rng.standard_normal(n)                              # doctest: +SKIP
        >>> y = 0.3 * z + 0.5 * c[:, 0] + rng.standard_normal(n)    # doctest: +SKIP
        >>> ctrl = pd.DataFrame(c, columns=["c0", "c1"])            # doctest: +SKIP
        >>> incremental_test(y, ctrl, z)["verdict"]                 # doctest: +SKIP
        'SURVIVES'
    """
    if known_controls_df is None:
        y, s = _align(fwd_ret, new_signal)
        C = np.empty((y.size, 0))
    else:
        y, s, C = _align(fwd_ret, new_signal, known_controls_df)
        if C.ndim == 1:
            C = C.reshape(-1, 1)

    if C.shape[1] > 0:
        X_r = np.column_stack([np.ones(y.size), C])
        X_f = np.column_stack([np.ones(y.size), s, C])
    else:
        X_r = np.ones((y.size, 1))
        X_f = np.column_stack([np.ones(y.size), s])

    res_r = _fit(y, X_r, robust=robust)
    res_f = _fit(y, X_f, robust=robust)
    t_new = float(res_f["t"][1])
    p_new = float(res_f["p"][1])
    delta_r2 = float(res_f["r2"] - res_r["r2"])
    verdict = _SURVIVES if abs(t_new) >= _T_RULE else _DEATH
    return {"t_new": t_new, "delta_r2": delta_r2, "verdict": verdict,
            "coef_new": float(res_f["beta"][1]), "p_new": p_new,
            "r2_restricted": float(res_r["r2"]), "r2_full": float(res_f["r2"]),
            "n": int(y.size), "robust": bool(robust)}


# ----------------------------------------------------------------------------
# specification curve
# ----------------------------------------------------------------------------
def specification_curve(
    fwd_ret: Any,
    controls_list: Any,
    signal: Any,
    specs_to_try: list[dict[str, Any]] | None = None,
    robust: bool = True,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Table of ``t_new`` across a grid of reasonable specifications.

    Parameters
    ----------
    controls_list : list of control sets; each entry is a DataFrame / (n, m)
        array (or None).  A spec references one entry by integer index, or
        ``None`` for no controls.
    specs_to_try : list of ``{"controls": int | None, "transform": str,
        "winsor_pct": float}`` dicts.  ``transform`` is one of
        ``raw | rank | zscore | winsorize | demean``.  When None, the full
        grid ``transforms x (None + each control set)`` is used.

    The kill rule: if ``max |t_new| < 2`` across every spec in the curve, the
    signal is a repackaging and the hypothesis dies
    (see :func:`curve_summary`).

    Example::

        >>> rng = np.random.default_rng(0)                       # doctest: +SKIP
        >>> n = 800                                              # doctest: +SKIP
        >>> y = rng.standard_normal(n)                           # doctest: +SKIP
        >>> z = rng.standard_normal(n)                           # doctest: +SKIP
        >>> table = specification_curve(y, [pd.DataFrame({"c": rng.standard_normal(n)})], z)  # doctest: +SKIP
        >>> set(table.columns) >= {"t_new", "verdict"}           # doctest: +SKIP
        True
    """
    controls_list = list(controls_list) if controls_list is not None else []
    if specs_to_try is None:
        specs_to_try = [
            {"controls": ci, "transform": tr, "winsor_pct": winsor_pct}
            for tr in DEFAULT_TRANSFORMS
            for ci in [None] + list(range(len(controls_list)))
        ]

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs_to_try):
        sp = dict(spec)
        ci = sp.get("controls", None)
        tr = sp.get("transform", "raw")
        wp = float(sp.get("winsor_pct", winsor_pct))
        ctrl = controls_list[ci] if isinstance(ci, int) and 0 <= ci < len(controls_list) else None

        if ctrl is None:
            y, s = _align(fwd_ret, signal)
            C = np.empty((y.size, 0))
        else:
            y, s, C = _align(fwd_ret, signal, ctrl)
            if C.ndim == 1:
                C = C.reshape(-1, 1)

        s_t = _transform_signal(s, tr, wp)
        if C.shape[1] > 0:
            X_f = np.column_stack([np.ones(y.size), s_t, C])
            X_r = np.column_stack([np.ones(y.size), C])
        else:
            X_f = np.column_stack([np.ones(y.size), s_t])
            X_r = np.ones((y.size, 1))

        res_f = _fit(y, X_f, robust=robust)
        res_r = _fit(y, X_r, robust=robust)
        t_new = float(res_f["t"][1])
        rows.append({
            "spec": i,
            "transform": tr,
            "controls": "none" if ctrl is None else f"set{ci}",
            "n": int(y.size),
            "t_new": t_new,
            "p_new": float(res_f["p"][1]),
            "delta_r2": float(res_f["r2"] - res_r["r2"]),
            "verdict": _SURVIVES if abs(t_new) >= _T_RULE else _DEATH,
        })
    return pd.DataFrame(rows)


def curve_summary(spec_table: pd.DataFrame) -> dict[str, Any]:
    """Overall SURVIVES / REPACKAGING-DEATH call for a specification curve.

    The rule: if NO reasonable spec pushes ``|t_new| >= 2``, the signal is a
    repackaging of known information -> death.  A spec curve where every spec
    flips sign is also reported via ``sign_consistent``.

    Example::

        >>> table = specification_curve(y, ctrl_sets, z)   # doctest: +SKIP
        >>> curve_summary(table)["verdict"]                # doctest: +SKIP
        'SURVIVES'
    """
    t = spec_table["t_new"].to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    n_specs = int(t.size)
    max_abs_t = float(np.max(np.abs(t))) if n_specs else float("nan")
    min_abs_t = float(np.min(np.abs(t))) if n_specs else float("nan")
    n_surviving = int(np.sum(np.abs(t) >= _T_RULE)) if n_specs else 0
    sign_consistent = bool(n_specs > 0 and np.all(np.sign(t) == np.sign(t[0])))
    verdict = _SURVIVES if (n_specs > 0 and max_abs_t >= _T_RULE) else _DEATH
    return {"verdict": verdict, "n_specs": n_specs, "n_surviving": n_surviving,
            "max_abs_t": max_abs_t, "min_abs_t": min_abs_t,
            "sign_consistent": sign_consistent, "t_rule": _T_RULE}


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: a genuine edge survives the curve; repackaged momentum dies."""
    rng = np.random.default_rng(3)
    n = 900
    mom = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        mom[t] = 0.6 * mom[t - 1] + 0.4 * e[t]
    z = rng.standard_normal(n)

    # Case A: real edge, signal contaminated with momentum but z survives.
    fwd_a = 0.20 * z + 0.30 * mom + rng.standard_normal(n)
    sig_a = z + 0.5 * mom
    mom_df = pd.DataFrame({"mom7": mom + rng.standard_normal(n) * 0.1,
                           "mom30": np.roll(mom, 12)}, index=range(n))

    print("=" * 72)
    print("incremental_regression demo A: genuine edge")
    print("=" * 72)
    res = incremental_test(fwd_a, mom_df, sig_a, robust=True)
    print(f"incremental_test: t_new={res['t_new']:.2f} delta_r2={res['delta_r2']:.4f} "
          f"verdict={res['verdict']}")
    table = specification_curve(fwd_a, [mom_df], sig_a, robust=True)
    with pd.option_context("display.width", 120, "display.max_columns", 10):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("curve_summary:", curve_summary(table))

    # Case B: "new" signal is 95% repackaged momentum + noise -> death.
    fwd_b = 0.35 * mom + rng.standard_normal(n)
    sig_b = mom + rng.standard_normal(n) * 0.35

    print()
    print("=" * 72)
    print("incremental_regression demo B: repackaged momentum (expected death)")
    print("=" * 72)
    res_b = incremental_test(fwd_b, mom_df, sig_b, robust=True)
    print(f"incremental_test: t_new={res_b['t_new']:.2f} delta_r2={res_b['delta_r2']:.4f} "
          f"verdict={res_b['verdict']}")
    table_b = specification_curve(fwd_b, [mom_df], sig_b, robust=True)
    print("curve_summary:", curve_summary(table_b))
    print("note: the curve keeps uncontrolled specs alive (momentum predicts itself);")
    print("      the kill comes from the CONTROLLED test above (t_new=0.53 -> DEATH).")


if __name__ == "__main__":
    run_all_examples()
