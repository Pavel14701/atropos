"""Null-control suite: six adversarial controls for any signal/event hypothesis.

Every test follows the same contract::

    {"stat": float, "p_value": float, "verdict": "PASS" | "FAIL", ...extras}

PASS means the hypothesis SURVIVES that control.  The six controls:

1. ``shuffled_permutation``    - is the real alignment of signal->outcome more
                                 extreme than random shuffles of the signal?
2. ``placebo_placebo``         - are the claimed event days special, or would
                                 random day sets show the same effect?
3. ``momentum_control``        - does the new signal keep |t| >= 2 after
                                 controlling for 7d and 30d momentum?
4. ``known_features_control``  - does it keep |t| >= 2 after all known features?
5. ``subperiod_stability``     - is the effect consistent across subperiods or
                                 concentrated in one?
6. ``calendar_control``        - does it survive year fixed effects?

All pandas inputs are index-aligned (inner join); numpy inputs are aligned by
position.  NaN rows are dropped pairwise.

Run ``python null_control_suite.py`` for a synthetic demo of all six.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "shuffled_permutation",
    "placebo_placebo",
    "momentum_control",
    "known_features_control",
    "subperiod_stability",
    "calendar_control",
]

_ALPHA = 0.05


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _frameify(a: Any, tag: str) -> pd.DataFrame:
    """Convert Series / DataFrame / ndarray into a named DataFrame."""
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

    Returns one float ndarray per input; 1-D inputs stay 1-D, wide inputs come
    back as (n, m) matrices.
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


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ols(y: np.ndarray, X: np.ndarray, robust: bool = True) -> dict[str, Any]:
    """OLS with HC0 (default) or classical standard errors."""
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
    return {"beta": beta, "se": se, "t": t, "p": p, "df": df, "n": n, "r2": r2}


def _pass_fail(p: float, alpha: float = _ALPHA) -> str:
    return "PASS" if (np.isfinite(p) and p < alpha) else "FAIL"


# ----------------------------------------------------------------------------
# 1. permutation null
# ----------------------------------------------------------------------------
def shuffled_permutation(
    signal: Any, outcome: Any, n_draws: int = 200, seed: int | None = None
) -> dict[str, Any]:
    """Percentile of the real |corr(signal, outcome)| among shuffled signals.

    If random shuffles of the signal reproduce the observed correlation, the
    "effect" is just an alignment artifact -> FAIL.

    Returns ``{stat, p_value, verdict, percentile, n, n_draws}``.

    Example::

        >>> rng = np.random.default_rng(0)                       # doctest: +SKIP
        >>> s = rng.standard_normal(500)                         # doctest: +SKIP
        >>> y = 0.3 * s + rng.standard_normal(500)               # doctest: +SKIP
        >>> shuffled_permutation(s, y, seed=0)["verdict"]        # doctest: +SKIP
        'PASS'
    """
    s, y = _align(signal, outcome)
    n = int(s.size)
    r_real = _pearson_r(s, y)
    if not np.isfinite(r_real):
        return {"stat": r_real, "p_value": 1.0, "verdict": "FAIL",
                "percentile": 0.0, "n": n, "n_draws": int(n_draws)}
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_draws)):
        if abs(_pearson_r(rng.permutation(s), y)) >= abs(r_real):
            ge += 1
    p = (1.0 + ge) / (n_draws + 1.0)
    percentile = 100.0 * (1.0 - ge / float(n_draws))
    return {"stat": r_real, "p_value": float(p), "verdict": "PASS" if p <= _ALPHA else "FAIL",
            "percentile": percentile, "n": n, "n_draws": int(n_draws)}


# ----------------------------------------------------------------------------
# 2. placebo events
# ----------------------------------------------------------------------------
def placebo_placebo(
    event_days: Any, outcome_series: Any, n_draws: int = 200, seed: int | None = None
) -> dict[str, Any]:
    """Are the claimed event days special vs random placebo day sets?

    ``event_days`` are integer positions; if ``outcome_series`` is a pandas
    Series and the values are not integers, they are looked up in its index.
    The test statistic is the mean outcome on the event days; the p-value is
    the probability that a random same-size day set has a mean at least as
    extreme in absolute value.

    Returns ``{stat, p_value, verdict, n_events, n, placebo_mean_abs}``.

    Example::

        >>> rng = np.random.default_rng(0)                        # doctest: +SKIP
        >>> y = rng.standard_normal(300)                          # doctest: +SKIP
        >>> y[[10, 20, 30]] += 5.0                                # doctest: +SKIP
        >>> placebo_placebo([10, 20, 30], y, seed=0)["verdict"]   # doctest: +SKIP
        'PASS'
    """
    if isinstance(outcome_series, pd.Series):
        out = outcome_series.to_numpy(dtype=float)
        index = outcome_series.index
    else:
        out = np.asarray(outcome_series, dtype=float).reshape(-1)
        index = None
    n = int(out.size)

    ed = np.asarray(event_days)
    pos: np.ndarray
    resolved = False
    if index is not None and ed.size and not np.issubdtype(ed.dtype, np.integer):
        try:
            found = index.get_indexer(pd.Index(ed))
            if (found >= 0).all():
                pos = found
                resolved = True
        except (TypeError, ValueError):
            pass
    if not resolved:
        try:
            pos = ed.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("event_days must be integer positions or index labels") from exc
    pos = pos[(pos >= 0) & (pos < n)]

    k = int(pos.size)
    if k == 0 or n < 3:
        return {"stat": float("nan"), "p_value": 1.0, "verdict": "FAIL",
                "n_events": 0, "n": n, "placebo_mean_abs": float("nan")}

    actual = float(np.mean(out[pos]))
    rng = np.random.default_rng(seed)
    ge = 0
    placebo_abs = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        idx = rng.choice(n, size=k, replace=False)
        m = float(np.mean(out[idx]))
        placebo_abs[i] = abs(m)
        if abs(m) >= abs(actual):
            ge += 1
    p = (1.0 + ge) / (n_draws + 1.0)
    return {"stat": actual, "p_value": float(p), "verdict": "PASS" if p <= _ALPHA else "FAIL",
            "n_events": k, "n": n, "placebo_mean_abs": float(np.mean(placebo_abs))}


# ----------------------------------------------------------------------------
# 3. momentum control
# ----------------------------------------------------------------------------
def momentum_control(
    fwd_ret: Any,
    new_signal: Any,
    mom7: Any,
    mom30: Any,
    HC0: bool = True,
) -> dict[str, Any]:
    """|t| of the new signal after controlling for 7d and 30d momentum.

    Regresses ``fwd_ret ~ const + new_signal + mom7 + mom30``.  PASS requires
    the new signal to stay significant (p < 0.05) with momentum in the model;
    otherwise the "edge" was repackaged momentum -> FAIL.

    Returns ``{stat, p_value, verdict, t_mom7, t_mom30, coef_new, n, r2, HC0}``.

    Example::

        >>> rng = np.random.default_rng(0)                            # doctest: +SKIP
        >>> n = 600                                                   # doctest: +SKIP
        >>> mom = np.random.default_rng(1).standard_normal(n)         # doctest: +SKIP
        >>> z = rng.standard_normal(n)                                # doctest: +SKIP
        >>> y = 0.4 * z + 0.3 * mom + rng.standard_normal(n)          # doctest: +SKIP
        >>> momentum_control(y, z, mom, np.roll(mom, 7), HC0=True)["verdict"]  # doctest: +SKIP
        'PASS'
    """
    y, s, m7, m30 = _align(fwd_ret, new_signal, mom7, mom30)
    X = np.column_stack([np.ones(y.size), s, m7, m30])
    res = _ols(y, X, robust=bool(HC0))
    t_new = float(res["t"][1])
    p_new = float(res["p"][1])
    return {"stat": t_new, "p_value": p_new, "verdict": _pass_fail(p_new),
            "t_mom7": float(res["t"][2]), "t_mom30": float(res["t"][3]),
            "coef_new": float(res["beta"][1]), "n": int(res["n"]),
            "r2": float(res["r2"]), "HC0": bool(HC0)}


# ----------------------------------------------------------------------------
# 4. known-features control
# ----------------------------------------------------------------------------
def known_features_control(
    fwd_ret: Any, new_signal: Any, known_features_df: Any
) -> dict[str, Any]:
    """|t| of the new signal after controlling for ALL known features.

    Regresses ``fwd_ret ~ const + new_signal + known_features``.  Zero-variance
    feature columns are dropped.  PASS requires the new signal to remain
    significant (p < 0.05); if it dies, it was a repackaging of what was
    already on disk.

    Returns ``{stat, p_value, verdict, n_features, n, r2}``.

    Example::

        >>> rng = np.random.default_rng(0)                         # doctest: +SKIP
        >>> n = 600                                                # doctest: +SKIP
        >>> f1 = rng.standard_normal(n)                            # doctest: +SKIP
        >>> z = 0.8 * f1 + rng.standard_normal(n)                  # doctest: +SKIP
        >>> y = 0.5 * z + rng.standard_normal(n)                   # doctest: +SKIP
        >>> feats = pd.DataFrame({"f1": f1, "f2": rng.standard_normal(n)})  # doctest: +SKIP
        >>> known_features_control(y, z, feats)["verdict"]         # doctest: +SKIP
        'FAIL'
    """
    y, s, F = _align(fwd_ret, new_signal, known_features_df)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    keep = F.var(axis=0) > 1e-12
    F = F[:, keep]
    m = int(F.shape[1])
    if m > 0:
        X = np.column_stack([np.ones(y.size), s, F])
    else:
        X = np.column_stack([np.ones(y.size), s])
    res = _ols(y, X, robust=True)
    t_new = float(res["t"][1])
    p_new = float(res["p"][1])
    return {"stat": t_new, "p_value": p_new, "verdict": _pass_fail(p_new),
            "n_features": m, "n": int(res["n"]), "r2": float(res["r2"])}


# ----------------------------------------------------------------------------
# 5. subperiod stability
# ----------------------------------------------------------------------------
def subperiod_stability(
    daily_ev: Any,
    periods: int = 3,
    max_share_allowed: float = 0.60,
    n_perm: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Is the per-day evidence consistent across subperiods, or concentrated?

    ``daily_ev`` is a 1-D series of per-day evidence contributions (e.g.
    position * forward return).  The series is split into ``periods``
    contiguous chunks; the test statistic is the share of total evidence held
    by the largest chunk (``max_share`` in [1/periods, 1]).  ``p_value`` is the
    permutation probability that a random shuffle of the same values shows at
    least this concentration - so a LOW p_value means the concentration is
    unusual (a bad sign), while PASS requires sign consistency across chunks
    and ``max_share <= max_share_allowed``.

    Returns ``{stat, p_value, verdict, period_effects, period_means,
    sign_consistent, concentration_p, n}``.

    Example::

        >>> rng = np.random.default_rng(0)                        # doctest: +SKIP
        >>> ev = 0.2 + rng.standard_normal(600) * 0.05            # doctest: +SKIP
        >>> subperiod_stability(ev, periods=3, seed=0)["verdict"] # doctest: +SKIP
        'PASS'
    """
    ev = np.asarray(daily_ev, dtype=float).reshape(-1)
    ev = ev[np.isfinite(ev)]
    n = int(ev.size)
    P = int(periods)
    if P < 2 or n < 10 * P:
        return {"stat": float("nan"), "p_value": 1.0, "verdict": "FAIL",
                "period_effects": [], "period_means": [], "sign_consistent": False,
                "n": n, "note": "insufficient data"}

    splits = np.array_split(np.arange(n), P)
    lens = np.array([ix.size for ix in splits], dtype=float)
    sums = np.array([float(ev[ix].sum()) for ix in splits])
    denom = float(np.abs(sums).sum())
    max_share = float(np.abs(sums).max() / denom) if denom > 0 else float("nan")
    overall_sign = np.sign(float(sums.sum()))
    nonzero = sums[sums != 0]
    sign_consistent = bool(nonzero.size > 0 and overall_sign != 0
                           and np.all(np.sign(nonzero) == overall_sign))

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        ev_p = rng.permutation(ev)
        sums_p = np.array([float(ev_p[ix].sum()) for ix in splits])
        d = float(np.abs(sums_p).sum())
        c = float(np.abs(sums_p).max() / d) if d > 0 else 0.0
        if c >= max_share:
            ge += 1
    p_conc = (1.0 + ge) / (n_perm + 1.0)

    verdict = "PASS" if (sign_consistent and np.isfinite(max_share)
                         and max_share <= float(max_share_allowed)) else "FAIL"
    return {"stat": max_share, "p_value": float(p_conc), "verdict": verdict,
            "period_effects": sums.tolist(), "period_means": (sums / lens).tolist(),
            "sign_consistent": sign_consistent, "concentration_p": float(p_conc),
            "max_share_allowed": float(max_share_allowed), "n": n,
            "p_value_note": "low p = concentration more extreme than shuffled baseline"}


# ----------------------------------------------------------------------------
# 6. calendar control
# ----------------------------------------------------------------------------
def calendar_control(fwd_ret: Any, signal: Any, year_dummies: Any) -> dict[str, Any]:
    """Does the signal survive year fixed effects?

    Fits ``fwd_ret ~ const + signal`` and ``fwd_ret ~ const + signal + year
    dummies``.  PASS requires the signal coefficient to stay significant with
    the fixed effects (p < 0.05) AND keep its sign; a sign flip means the
    "edge" was a calendar regime wearing a signal costume.

    Returns ``{stat, p_value, verdict, t_plain, p_plain, n_year_dummies, n}``.

    Example::

        >>> rng = np.random.default_rng(0)                        # doctest: +SKIP
        >>> n = 600                                               # doctest: +SKIP
        >>> s = rng.standard_normal(n)                            # doctest: +SKIP
        >>> y = 0.3 * s + rng.standard_normal(n)                  # doctest: +SKIP
        >>> yrs = pd.get_dummies(np.repeat([0, 1, 2, 3, 4], 120)).to_numpy(float)  # doctest: +SKIP
        >>> calendar_control(y, s, yrs)["verdict"]                # doctest: +SKIP
        'PASS'
    """
    y, s, D = _align(fwd_ret, signal, year_dummies)
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    D = D[:, D.var(axis=0) > 1e-12]
    X0 = np.column_stack([np.ones(y.size), s])
    X1 = np.column_stack([np.ones(y.size), s, D]) if D.shape[1] > 0 else X0
    r0 = _ols(y, X0, robust=True)
    r1 = _ols(y, X1, robust=True)
    t0, p0 = float(r0["t"][1]), float(r0["p"][1])
    t1, p1 = float(r1["t"][1]), float(r1["p"][1])
    same_sign = bool(t1 != 0 and np.sign(t0) == np.sign(t1))
    verdict = "PASS" if (np.isfinite(p1) and p1 < _ALPHA and same_sign) else "FAIL"
    return {"stat": t1, "p_value": p1, "verdict": verdict,
            "t_plain": t0, "p_plain": p0,
            "n_year_dummies": int(D.shape[1]), "n": int(r1["n"])}


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _show(name: str, res: dict[str, Any]) -> None:
    extras = {k: v for k, v in res.items() if k not in ("stat", "p_value", "verdict")}
    print(f"\n[{name}] stat={res['stat']:.4f} p={res['p_value']:.4f} verdict={res['verdict']}")
    print("   " + str(extras))


def run_all_examples() -> None:
    """Demo all six controls on one synthetic dataset with a planted edge."""
    print("=" * 72)
    print("null_control_suite demo: planted edge + momentum confound")
    print("=" * 72)
    rng = np.random.default_rng(42)
    n = 800
    mom = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        mom[t] = 0.7 * mom[t - 1] + 0.3 * e[t]
    z = rng.standard_normal(n)                     # the true (hidden) driver
    mom7 = mom + rng.standard_normal(n) * 0.1
    mom30 = np.roll(mom, 12) + rng.standard_normal(n) * 0.1
    new_signal = z + 0.5 * mom                     # contaminated with momentum
    fwd = 0.20 * z + 0.25 * mom + rng.standard_normal(n)

    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    fwd_s = pd.Series(fwd, index=idx)
    sig_s = pd.Series(new_signal, index=idx)

    _show("1 shuffled_permutation", shuffled_permutation(sig_s, fwd_s, n_draws=200, seed=0))
    event_days = idx[np.argsort(z)[-40:]]
    _show("2 placebo_placebo", placebo_placebo(event_days, fwd_s, n_draws=200, seed=0))
    _show("3 momentum_control", momentum_control(fwd_s, sig_s, mom7, mom30, HC0=True))
    feats = pd.DataFrame({"mom7": mom7, "mom30": mom30}, index=idx)
    _show("4 known_features_control", known_features_control(fwd_s, sig_s, feats))
    daily_ev = new_signal * fwd
    _show("5 subperiod_stability", subperiod_stability(daily_ev, periods=3, n_perm=200, seed=0))
    years = pd.get_dummies(idx.year).to_numpy(dtype=float)
    _show("6 calendar_control", calendar_control(fwd_s, sig_s, years))


if __name__ == "__main__":
    run_all_examples()
