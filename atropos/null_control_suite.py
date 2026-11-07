"""Null-control suite: six adversarial controls for a hypothesis.

Every test follows the same contract::

    {"stat": float, "p_value": float,
     "verdict": "PASS" | "FAIL" | "INCONCLUSIVE", ...extras}

PASS means the hypothesis SURVIVES that control; FAIL means the control
killed it.  INCONCLUSIVE means the control could not be evaluated at
all: the three regression controls return it (plus a ``reason`` field,
currently ``"collinear with controls"``) when the design matrix is
rank-deficient -- e.g. the signal is EXACTLY a known control -- because
a pseudo-inverse would fabricate a plausible-looking minimum-norm
t-statistic precisely in the repackaging cases this suite exists to
catch.

Every numeric threshold in this module is STRICT: PASS requires
``p_value < 0.05`` (``p_value == 0.05`` is a FAIL) and the subperiod
concentration must be strictly below its cap.

The six controls:

1. ``shuffled_permutation``   - is the real alignment of signal->outcome
                                more extreme than random shuffles?
2. ``placebo_placebo``        - are the claimed event days special, or
                                would random day sets show the same effect?
3. ``momentum_control``       - does the new signal keep |t| >= 2 after
                                controlling for 7d and 30d momentum?
4. ``known_features_control`` - does it keep |t| >= 2 after all known
                                features?
5. ``subperiod_stability``    - is the effect consistent across subperiods
                                or concentrated in one?
6. ``calendar_control``       - does it survive year fixed effects?

All pandas inputs are index-aligned (inner join); numpy inputs are aligned
by position.  NaN rows are dropped pairwise.  Build lagged inputs with
``shift`` (NaN head is dropped here), never with ``np.roll`` which wraps
the tail to the head and leaks future data into the lag.

Run ``python -m atropos.null_control_suite`` for a synthetic demo of all
six.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from atropos._utils import align_inputs, ols_hc0

__all__ = [
    "calendar_control",
    "known_features_control",
    "momentum_control",
    "placebo_placebo",
    "shuffled_permutation",
    "subperiod_stability",
]

_ALPHA = 0.05

# known_features_control: a feature column is unusable as a control when
# its variance is below this fraction of the STRONGEST feature's
# variance (relative floor) -- or exactly constant.  The former absolute
# cut (``var > 1e-12``) failed open on small-scale feature sets.
_VAR_FLOOR = 1e-12


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pearson_r(signal: np.ndarray, outcome: np.ndarray) -> float:
    if signal.size < 3 or np.std(signal) <= 0 or np.std(outcome) <= 0:
        return float("nan")
    return float(np.corrcoef(signal, outcome)[0, 1])


def _ols(
    outcome: np.ndarray, design: np.ndarray, robust: bool = True
) -> dict[str, Any]:
    """OLS via ``ols_hc0`` (HC0 errors) or classical standard errors.

    Delegates the fit to :func:`atropos._utils.ols_hc0`, which returns
    NaN statistics with ``rank_deficient=True`` whenever the design is
    not full column rank or has ``n_rows <= n_cols`` -- a pseudo-inverse
    must not fabricate minimum-norm t-stats under exact collinearity.
    ``robust=False`` swaps the HC0 covariance for the classical
    ``s^2 (X'X)^+`` one on the same (already verified non-deficient)
    fit; rank-deficient inputs still get the NaN payload.
    """
    result = ols_hc0(outcome, design)
    if robust or result["rank_deficient"]:
        return result
    beta = result["beta"]
    resid = outcome - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    dof = int(result["df"])
    s2 = float(resid @ resid) / dof
    cov = s2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    pvalues = 2.0 * stats.t.sf(np.abs(t_stats), dof)
    return {**result, "se": se, "t": t_stats, "p": pvalues}


def _pass_fail(pvalue: float, alpha: float = _ALPHA) -> str:
    """Strict rule: PASS needs a finite ``pvalue`` strictly below alpha."""
    return "PASS" if (np.isfinite(pvalue) and pvalue < alpha) else "FAIL"


# ----------------------------------------------------------------------------
# 1. permutation null
# ----------------------------------------------------------------------------
def shuffled_permutation(
    signal: Any, outcome: Any, n_draws: int = 200, seed: int | None = None
) -> dict[str, Any]:
    """Percentile of the real |corr(signal, outcome)| among shuffles.

    If random shuffles of the signal reproduce the observed correlation,
    the "effect" is just an alignment artifact -> FAIL.  The verdict
    threshold is strict: PASS requires ``p_value < 0.05``.

    On degenerate input (fewer than 3 observations, or zero variance in
    either series) the baseline is undefined: the verdict is FAIL and
    ``percentile`` is NaN with a ``note`` -- a plain ``0.0`` would read
    as "measured and not extreme" when nothing could be measured.

    Returns ``{stat, p_value, verdict, percentile, n, n_draws}`` plus a
    ``note`` field on degenerate input.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> sig = rng.standard_normal(500)  # doctest: +SKIP
        >>> out = 0.3 * sig + rng.standard_normal(500)  # doctest: +SKIP
        >>> res = shuffled_permutation(sig, out, seed=0)  # doctest: +SKIP
        >>> res["verdict"]  # doctest: +SKIP
        'PASS'
    """
    sig, out = align_inputs(signal, outcome)
    n_obs = int(sig.size)
    r_real = _pearson_r(sig, out)
    if not np.isfinite(r_real):
        return {
            "stat": r_real,
            "p_value": 1.0,
            "verdict": "FAIL",
            "percentile": float("nan"),
            "n": n_obs,
            "n_draws": int(n_draws),
            "note": (
                "degenerate input: baseline correlation undefined"
                " (n < 3 or zero variance); percentile not meaningful"
            ),
        }
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_draws)):
        if abs(_pearson_r(rng.permutation(sig), out)) >= abs(r_real):
            ge += 1
    pvalue = (1.0 + ge) / (n_draws + 1.0)
    percentile = 100.0 * (1.0 - ge / float(n_draws))
    return {
        "stat": r_real,
        "p_value": float(pvalue),
        "verdict": "PASS" if pvalue < _ALPHA else "FAIL",
        "percentile": percentile,
        "n": n_obs,
        "n_draws": int(n_draws),
    }


# ----------------------------------------------------------------------------
# 2. placebo events
# ----------------------------------------------------------------------------
def placebo_placebo(
    event_days: Any,
    outcome_series: Any,
    n_draws: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Are the claimed event days special vs random placebo day sets?

    ``event_days`` are integer positions; if ``outcome_series`` is a
    pandas Series and the values are not integers, they are looked up in
    its index.  Event days outside the outcome range (negative positions
    or past the end) are dropped from the test but counted in the
    ``n_out_of_range`` key -- silently losing them would hide a broken
    upstream event definition.  The test statistic is the mean outcome
    on the event days; the p-value is the probability that a random
    same-size day set has a mean at least as extreme in absolute value.
    The verdict threshold is strict: PASS requires ``p_value < 0.05``.

    Returns ``{stat, p_value, verdict, n_events, n_out_of_range, n,
    placebo_mean_abs}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> out = rng.standard_normal(300)  # doctest: +SKIP
        >>> out[[10, 20, 30]] += 5.0  # doctest: +SKIP
        >>> res = placebo_placebo(  # doctest: +SKIP
        ...     [10, 20, 30], out, seed=0)
        >>> res["verdict"]  # doctest: +SKIP
        'PASS'
    """
    if isinstance(outcome_series, pd.Series):
        out = outcome_series.to_numpy(dtype=float)
        index = outcome_series.index
    else:
        out = np.asarray(outcome_series, dtype=float).reshape(-1)
        index = None
    n_obs = int(out.size)

    day_values = np.asarray(event_days)
    pos: np.ndarray
    resolved = False
    if (
        index is not None
        and day_values.size
        and not np.issubdtype(day_values.dtype, np.integer)
    ):
        try:
            found = index.get_indexer(pd.Index(day_values))
            if (found >= 0).all():
                pos = found
                resolved = True
        except (TypeError, ValueError):
            pass
    if not resolved:
        try:
            pos = day_values.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "event_days must be integer positions or index labels"
            ) from exc
    pos = pos[(pos >= 0) & (pos < n_obs)]

    n_events = int(pos.size)
    n_out_of_range = int(day_values.size) - n_events
    if n_events == 0 or n_obs < 3:
        return {
            "stat": float("nan"),
            "p_value": 1.0,
            "verdict": "FAIL",
            "n_events": 0,
            "n_out_of_range": n_out_of_range,
            "n": n_obs,
            "placebo_mean_abs": float("nan"),
        }

    actual = float(np.mean(out[pos]))
    rng = np.random.default_rng(seed)
    ge = 0
    placebo_abs = np.empty(int(n_draws), dtype=float)
    for draw in range(int(n_draws)):
        sel = rng.choice(n_obs, size=n_events, replace=False)
        draw_mean = float(np.mean(out[sel]))
        placebo_abs[draw] = abs(draw_mean)
        if abs(draw_mean) >= abs(actual):
            ge += 1
    pvalue = (1.0 + ge) / (n_draws + 1.0)
    return {
        "stat": actual,
        "p_value": float(pvalue),
        "verdict": "PASS" if pvalue < _ALPHA else "FAIL",
        "n_events": n_events,
        "n_out_of_range": n_out_of_range,
        "n": n_obs,
        "placebo_mean_abs": float(np.mean(placebo_abs)),
    }


# ----------------------------------------------------------------------------
# 3. momentum control
# ----------------------------------------------------------------------------
def momentum_control(
    fwd_ret: Any,
    new_signal: Any,
    mom7: Any,
    mom30: Any,
    robust: bool = True,
) -> dict[str, Any]:
    """|t| of the new signal after controlling for 7d and 30d momentum.

    Regresses ``fwd_ret ~ const + new_signal + mom7 + mom30``.  PASS
    requires the new signal to stay significant (p < 0.05, strict) with
    momentum in the model; otherwise the "edge" was repackaged momentum
    -> FAIL.  If the signal is EXACTLY collinear with the momentum
    controls (e.g. ``new_signal is mom7``) the design is rank-deficient
    and no t-stat is meaningful: the verdict is INCONCLUSIVE with
    ``reason = "collinear with controls"`` instead of a pseudo-inverse
    PASS.

    Lagged controls must be built with ``shift`` (the NaN head is
    dropped by the alignment), never with ``np.roll``, which wraps the
    tail to the head and leaks future data into the lag.

    Returns ``{stat, p_value, verdict, t_mom7, t_mom30, coef_new, n, r2,
    HC0}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 600  # doctest: +SKIP
        >>> mom = rng.standard_normal(n)  # doctest: +SKIP
        >>> z = rng.standard_normal(n)  # doctest: +SKIP
        >>> y = 0.4 * z + 0.3 * mom + rng.standard_normal(n)  # doctest: +SKIP
        >>> mom_lag7 = pd.Series(mom).shift(7)  # doctest: +SKIP
        >>> momentum_control(y, z, mom, mom_lag7,  # doctest: +SKIP
        ...                  robust=True)["verdict"]
        'PASS'
    """
    out, sig, m7, m30 = align_inputs(fwd_ret, new_signal, mom7, mom30)
    design = np.column_stack([np.ones(out.size), sig, m7, m30])
    res = _ols(out, design, robust=bool(robust))
    t_new = float(res["t"][1])
    p_new = float(res["p"][1])
    collinear = bool(res["rank_deficient"]) or not np.isfinite(t_new)
    result: dict[str, Any] = {
        "stat": t_new,
        "p_value": p_new,
        "verdict": "INCONCLUSIVE" if collinear else _pass_fail(p_new),
        "t_mom7": float(res["t"][2]),
        "t_mom30": float(res["t"][3]),
        "coef_new": float(res["beta"][1]),
        "n": int(res["n"]),
        "r2": float(res["r2"]),
        "HC0": bool(robust),
    }
    if collinear:
        result["reason"] = "collinear with controls"
    return result


# ----------------------------------------------------------------------------
# 4. known-features control
# ----------------------------------------------------------------------------
def known_features_control(
    fwd_ret: Any, new_signal: Any, known_features_df: Any
) -> dict[str, Any]:
    """|t| of the new signal after controlling for ALL known features.

    Regresses ``fwd_ret ~ const + new_signal + known_features``.
    Feature columns whose variance is negligible relative to the
    strongest feature (below ``1e-12`` of it) or that are exactly
    constant are dropped -- a relative floor, so genuinely informative
    small-scale features (decimal returns, sd ~1e-7) are kept.  PASS
    requires the new signal to remain significant (p < 0.05, strict);
    if it dies, it was a repackaging of what was already on disk.  If
    the signal is EXACTLY collinear with the known features the design
    is rank-deficient and no t-stat is meaningful: the verdict is
    INCONCLUSIVE with ``reason = "collinear with controls"`` instead of
    a fabricated t-stat.

    Returns ``{stat, p_value, verdict, n_features, n, r2}``.

    Example (``y`` depends only on ``f1``; ``z`` is a noisy clone of
    ``f1``, so after controlling for ``[f1, f2]`` it adds no incremental
    information -> FAIL; verified with seed 0)::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 600  # doctest: +SKIP
        >>> f1 = rng.standard_normal(n)  # doctest: +SKIP
        >>> z = 0.8 * f1 + rng.standard_normal(n)  # doctest: +SKIP
        >>> y = 0.4 * f1 + rng.standard_normal(n)  # doctest: +SKIP
        >>> feats = pd.DataFrame(  # doctest: +SKIP
        ...     {"f1": f1, "f2": rng.standard_normal(n)})
        >>> known_features_control(y, z, feats)  # doctest: +SKIP
        {'stat': -1.5055885..., 'p_value': 0.1327026...,
         'verdict': 'FAIL', 'n_features': 2, 'n': 600,
         'r2': 0.1036350...}
    """
    out, sig, feats = align_inputs(fwd_ret, new_signal, known_features_df)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    variances = feats.var(axis=0)
    var_max = float(variances.max()) if variances.size else 0.0
    if var_max > 0.0:
        # relative floor anchored at the strongest feature: an absolute
        # 1e-12 cut failed OPEN on small-scale feature sets (a panel of
        # decimal returns carries variances ~1e-8, so every genuine
        # control was dropped and the check silently degenerated into
        # the very univariate regression it was meant to gate)
        keep = variances > _VAR_FLOOR * var_max
    else:
        # every column is constant at any scale: the scale-free ptp
        # check makes that explicit (nothing varies -> no controls)
        keep = np.ptp(feats, axis=0) > 0.0
    feats = feats[:, keep]
    n_feat = int(feats.shape[1])
    if n_feat > 0:
        design = np.column_stack([np.ones(out.size), sig, feats])
    else:
        design = np.column_stack([np.ones(out.size), sig])
    res = _ols(out, design, robust=True)
    t_new = float(res["t"][1])
    p_new = float(res["p"][1])
    collinear = bool(res["rank_deficient"]) or not np.isfinite(t_new)
    result: dict[str, Any] = {
        "stat": t_new,
        "p_value": p_new,
        "verdict": "INCONCLUSIVE" if collinear else _pass_fail(p_new),
        "n_features": n_feat,
        "n": int(res["n"]),
        "r2": float(res["r2"]),
    }
    if collinear:
        result["reason"] = "collinear with controls"
    return result


# ----------------------------------------------------------------------------
# 5. subperiod stability
# ----------------------------------------------------------------------------
def subperiod_stability(
    daily_ev: Any,
    periods: int = 3,
    max_share_allowed: float | None = None,
    n_perm: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Is the per-day evidence consistent across subperiods, or concentrated?

    ``daily_ev`` is a 1-D series of per-day evidence contributions (e.g.
    position * forward return).  The series is split into ``periods``
    contiguous chunks; the test statistic is the share of total evidence
    held by the largest chunk (``max_share`` in [1/periods, 1]).
    ``p_value`` is the permutation probability that a random shuffle of
    the same values shows at least this concentration - so a LOW p_value
    means the concentration is unusual (a bad sign), while PASS requires
    sign consistency across chunks AND ``max_share`` STRICTLY below
    the resolved cap (equality fails, like every threshold here).

    ``max_share_allowed``: the DEFAULT is ``None`` = auto-scale, which
    sets the cap to ``1.0 / periods + 0.15`` -- the uniform share plus
    a fixed 0.15 concentration allowance (0.483 for 3 periods, 0.350
    for 5, 0.317 for 6).  A kill gate must be conservative by default:
    a FIXED value does not scale with ``periods`` and is calibrated
    only for ``periods=3`` (the legacy 0.60).  For ``periods >= 4`` a
    fixed cap is anti-conservative (with periods=5 the fair share is
    0.20, so a 0.60 cap tolerates a chunk carrying three times its
    fair weight, and the planted review case R4 showed such
    concentrations slipping through), so passing one explicitly raises
    ``RuntimeWarning``; keep it only as a deliberate legacy opt-in.
    For ``periods < 4`` a fixed value is what the cap was calibrated
    for and passes silently.  The resolved cap is echoed in the result
    key ``max_share_allowed``.

    Returns ``{stat, p_value, verdict, period_effects, period_means,
    sign_consistent, concentration_p, max_share_allowed, n}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> ev = 0.2 + rng.standard_normal(600) * 0.05  # doctest: +SKIP
        >>> res = subperiod_stability(  # doctest: +SKIP
        ...     ev, periods=3, seed=0)
        >>> res["verdict"]  # doctest: +SKIP
        'PASS'
    """
    ev = np.asarray(daily_ev, dtype=float).reshape(-1)
    ev = ev[np.isfinite(ev)]
    n_obs = int(ev.size)
    n_periods = int(periods)
    if max_share_allowed is None:
        # max() guards only against periods < 2, rejected right below;
        # the echoed cap is then irrelevant because the test never runs.
        share_cap = 1.0 / max(n_periods, 1) + 0.15
    else:
        share_cap = float(max_share_allowed)
        if n_periods >= 4:
            # a fixed cap cannot scale with periods: at periods >= 4
            # it is more lenient than the auto cap and lets through
            # multi-fair-share concentrations (review case R4).
            warnings.warn(
                f"fixed max_share_allowed={share_cap:g} is "
                f"anti-conservative for periods>=4 "
                f"(fair share={1.0 / n_periods:.3f}); "
                f"consider None (auto-scale)",
                RuntimeWarning,
                stacklevel=2,
            )
    if n_periods < 2 or n_obs < 10 * n_periods:
        return {
            "stat": float("nan"),
            "p_value": 1.0,
            "verdict": "FAIL",
            "period_effects": [],
            "period_means": [],
            "sign_consistent": False,
            "max_share_allowed": share_cap,
            "n": n_obs,
            "note": "insufficient data",
        }

    splits = np.array_split(np.arange(n_obs), n_periods)
    lens = np.array([chunk.size for chunk in splits], dtype=float)
    sums = np.array([float(ev[chunk].sum()) for chunk in splits])
    denom = float(np.abs(sums).sum())
    max_share = (
        float(np.abs(sums).max() / denom) if denom > 0 else float("nan")
    )
    overall_sign = np.sign(float(sums.sum()))
    nonzero = sums[sums != 0]
    sign_consistent = bool(
        nonzero.size > 0
        and overall_sign != 0
        and np.all(np.sign(nonzero) == overall_sign)
    )

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        ev_p = rng.permutation(ev)
        sums_p = np.array([float(ev_p[chunk].sum()) for chunk in splits])
        denom_p = float(np.abs(sums_p).sum())
        share_p = (
            float(np.abs(sums_p).max() / denom_p) if denom_p > 0 else 0.0
        )
        if share_p >= max_share:
            ge += 1
    p_conc = (1.0 + ge) / (n_perm + 1.0)

    verdict = (
        "PASS"
        if (
            sign_consistent
            and np.isfinite(max_share)
            and max_share < share_cap
        )
        else "FAIL"
    )
    return {
        "stat": max_share,
        "p_value": float(p_conc),
        "verdict": verdict,
        "period_effects": sums.tolist(),
        "period_means": (sums / lens).tolist(),
        "sign_consistent": sign_consistent,
        "concentration_p": float(p_conc),
        "max_share_allowed": share_cap,
        "n": n_obs,
        "p_value_note": (
            "low p = concentration more extreme than shuffled baseline"
        ),
    }


# ----------------------------------------------------------------------------
# 6. calendar control
# ----------------------------------------------------------------------------
def calendar_control(
    fwd_ret: Any, signal: Any, year_dummies: Any
) -> dict[str, Any]:
    """Does the signal survive year fixed effects?

    Fits ``fwd_ret ~ const + signal`` and ``fwd_ret ~ const + signal +
    year dummies``.  ``year_dummies`` columns must be 0/1 indicator
    columns; any other values raise ``ValueError`` ("columns must be 0/1
    dummies") because a continuous or scaled column silently changes the
    test into something else.  Rows summing to one across columns is
    NOT required: each column is an independent regime indicator, and
    partial dummy sets are used as-is (see the trap note below for the
    complete one-hot case).

    PASS requires the signal coefficient to stay significant with the
    fixed effects (p < 0.05, strict) AND keep its sign; a sign flip
    means the "edge" was a calendar regime wearing a signal costume.
    If the signal is EXACTLY collinear with the dummies the FE design
    is rank-deficient: verdict INCONCLUSIVE with ``reason =
    "collinear with controls"`` instead of a fabricated t-stat.

    Dummy-trap handling: a complete one-hot dummy block (every row
    sums to 1, e.g. ``pd.get_dummies(years)``) plus the intercept is
    exactly collinear, so the LAST dummy is dropped as the reference
    category before fitting.  The signal's t-statistic is invariant to
    which redundant column is dropped, and ``n_year_dummies`` still
    reports the dummy count kept after the zero-variance filter (i.e.
    before the trap drop).  Rows summing to one is therefore allowed
    but NOT required: partial dummy sets are used as-is.

    Returns ``{stat, p_value, verdict, t_plain, p_plain, n_year_dummies,
    n}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 600  # doctest: +SKIP
        >>> s = rng.standard_normal(n)  # doctest: +SKIP
        >>> y = 0.3 * s + rng.standard_normal(n)  # doctest: +SKIP
        >>> yrs = pd.get_dummies(  # doctest: +SKIP
        ...     np.repeat([0, 1, 2, 3, 4], 120)).to_numpy(float)
        >>> calendar_control(y, s, yrs)["verdict"]  # doctest: +SKIP
        'PASS'
    """
    out, sig, dummies = align_inputs(fwd_ret, signal, year_dummies)
    if dummies.ndim == 1:
        dummies = dummies.reshape(-1, 1)
    if not np.isin(dummies, (0.0, 1.0)).all():
        raise ValueError(
            "year_dummies columns must be 0/1 dummies (found values"
            " other than 0 or 1); continuous or scaled columns are not"
            " regime indicators"
        )
    dummies = dummies[:, dummies.var(axis=0) > 1e-12]
    n_dummies = int(dummies.shape[1])
    if n_dummies > 1 and np.allclose(dummies.sum(axis=1), 1.0):
        # classic dummy trap: the full one-hot block sums to the
        # intercept column, so drop the last dummy (reference category);
        # the signal's t-stat does not depend on which one is dropped.
        dummies = dummies[:, :-1]
    design_plain = np.column_stack([np.ones(out.size), sig])
    design_fe = (
        np.column_stack([np.ones(out.size), sig, dummies])
        if dummies.shape[1] > 0
        else design_plain
    )
    r0 = _ols(out, design_plain, robust=True)
    r1 = _ols(out, design_fe, robust=True)
    t0, p0 = float(r0["t"][1]), float(r0["p"][1])
    t1, p1 = float(r1["t"][1]), float(r1["p"][1])
    collinear = bool(r1["rank_deficient"]) or not np.isfinite(t1)
    if collinear:
        verdict = "INCONCLUSIVE"
    else:
        same_sign = bool(t1 != 0 and np.sign(t0) == np.sign(t1))
        verdict = "PASS" if (p1 < _ALPHA and same_sign) else "FAIL"
    result: dict[str, Any] = {
        "stat": t1,
        "p_value": p1,
        "verdict": verdict,
        "t_plain": t0,
        "p_plain": p0,
        "n_year_dummies": n_dummies,
        "n": int(r1["n"]),
    }
    if collinear:
        result["reason"] = "collinear with controls"
    return result


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _show(name: str, res: dict[str, Any]) -> None:
    extras = {
        key: val
        for key, val in res.items()
        if key not in ("stat", "p_value", "verdict")
    }
    print(
        f"\n[{name}] stat={res['stat']:.4f} p={res['p_value']:.4f}"
        f" verdict={res['verdict']}"
    )
    print("   " + str(extras))


def _verify_planted_edge(
    *,
    res_perm: dict[str, Any],
    res_plac: dict[str, Any],
    res_mom: dict[str, Any],
    res_feat: dict[str, Any],
    res_sub: dict[str, Any],
    res_cal: dict[str, Any],
    n_days: int,
) -> None:
    """Assert the demo contract: the planted edge passes all six.

    Seed-independent properties only (every threshold mirrors the
    module's strict ``p_value < 0.05`` contract); the exact seeded
    permutation stat (~0.274) is locked in
    ``tests/test_null_control_suite.py``
    (``test_seeded_regression_demo_permutation_stat``).
    """
    assert res_perm["verdict"] == "PASS"
    assert res_perm["p_value"] < 0.05  # strict, like the module rule
    # stat is a Pearson r: finite, inside (0, 1) for the planted
    # positive edge - weak but real
    assert np.isfinite(res_perm["stat"])
    assert 0.0 < res_perm["stat"] < 1.0
    assert res_perm["n"] == n_days
    assert res_plac["verdict"] == "PASS"
    assert res_plac["p_value"] < 0.05  # strict, like the module rule
    assert res_plac["n_events"] == 40  # the planted driver's top days
    assert res_plac["n_out_of_range"] == 0
    assert res_mom["verdict"] == "PASS"
    assert abs(res_mom["stat"]) >= 2.0  # seeded run: ~7.6
    assert res_mom["n"] == n_days - 12  # shifted mom12's NaN head drops
    assert res_mom["HC0"] is True
    assert res_feat["verdict"] == "PASS"
    assert abs(res_feat["stat"]) >= 2.0
    assert res_feat["n_features"] == 2
    assert res_feat["n"] == n_days - 12
    assert res_sub["verdict"] == "PASS"
    assert res_sub["sign_consistent"] is True
    assert res_sub["stat"] < res_sub["max_share_allowed"]  # strict cap
    assert res_sub["n"] == n_days
    assert res_cal["verdict"] == "PASS"
    assert abs(res_cal["stat"]) >= 2.0  # seeded run: ~7.9
    assert res_cal["n_year_dummies"] == 4  # 2022..2025 business days
    assert res_cal["n"] == n_days


def _verify_trash_cases(
    fwd_s: pd.Series,
    mom7: np.ndarray,
    mom12: np.ndarray,
    feats: pd.DataFrame,
    n_days: int,
) -> None:
    """Assert the demo trash cases: collinear INCONCLUSIVE, noise FAIL."""
    # exact collinearity: signal == a control column -> rank-deficient
    # design -> INCONCLUSIVE, never a pseudo-inverse PASS
    res_collinear = momentum_control(fwd_s, mom7, mom7, mom12, robust=True)
    assert res_collinear["verdict"] == "INCONCLUSIVE"
    assert res_collinear["reason"] == "collinear with controls"
    res_clone = known_features_control(fwd_s, mom7, feats)
    assert res_clone["verdict"] == "INCONCLUSIVE"
    assert res_clone["reason"] == "collinear with controls"
    # pure noise (fresh seeded stream, unrelated to fwd) must FAIL the
    # permutation control
    noise = np.random.default_rng(2024).standard_normal(n_days)
    res_noise = shuffled_permutation(noise, fwd_s, n_draws=200, seed=0)
    assert res_noise["verdict"] == "FAIL"
    assert res_noise["p_value"] > 0.05


def _verify_auto_scale_cap(
    res_fixed: dict[str, Any], res_auto: dict[str, Any]
) -> None:
    """Assert the auto-scale cap contract on planted concentration.

    The planted series concentrates max_share 0.5 of the evidence in
    one of five chunks.  The legacy fixed 0.60 cap (calibrated for
    periods=3, now an explicit opt-in that raises RuntimeWarning)
    waves that through as PASS; the auto cap ``1/periods + 0.15 =
    0.35`` - the default - correctly fails it.
    """
    assert res_fixed["max_share_allowed"] == 0.60
    assert res_fixed["stat"] == 0.5
    assert res_fixed["verdict"] == "PASS"  # meaninglessly lenient at p=5
    assert res_auto["max_share_allowed"] == 0.35  # 1/5 + 0.15
    assert res_auto["stat"] == 0.5
    assert res_auto["verdict"] == "FAIL"


def run_all_examples() -> None:
    """Demo all six controls on one synthetic dataset with a planted edge.

    The demo doubles as the layer-0 verification test: the planted edge
    must print PASS on all six controls, and the trailing block asserts
    it (plus FAIL / INCONCLUSIVE on trash inputs and the auto-scale
    subperiod cap catching a concentration the legacy 0.60 cap
    tolerates), so a regression crashes the demo instead of printing a
    wrong verdict quietly.
    """
    print("=" * 72)
    print("null_control_suite demo: planted edge + momentum confound")
    print("=" * 72)
    rng = np.random.default_rng(42)
    n_days = 800
    mom = np.zeros(n_days)
    shock = rng.standard_normal(n_days)
    for day in range(1, n_days):
        mom[day] = 0.7 * mom[day - 1] + 0.3 * shock[day]
    driver = rng.standard_normal(n_days)  # the true (hidden) driver
    mom7 = mom + rng.standard_normal(n_days) * 0.1
    # lagged control via shift, NOT np.roll: roll wraps the tail to the
    # head, leaking future momentum into the lag.  The NaN head is
    # dropped by the controls' pairwise alignment.
    mom12 = (
        pd.Series(mom).shift(12).to_numpy(dtype=float)
        + rng.standard_normal(n_days) * 0.1
    )
    new_signal = driver + 0.5 * mom  # contaminated with momentum
    fwd = 0.20 * driver + 0.25 * mom + rng.standard_normal(n_days)

    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    fwd_s = pd.Series(fwd, index=idx)
    sig_s = pd.Series(new_signal, index=idx)

    res_perm = shuffled_permutation(sig_s, fwd_s, n_draws=200, seed=0)
    _show("1 shuffled_permutation", res_perm)
    event_days = idx[np.argsort(driver)[-40:]]
    res_plac = placebo_placebo(event_days, fwd_s, n_draws=200, seed=0)
    _show("2 placebo_placebo", res_plac)
    res_mom = momentum_control(fwd_s, sig_s, mom7, mom12, robust=True)
    _show("3 momentum_control", res_mom)
    feats = pd.DataFrame({"mom7": mom7, "mom12": mom12}, index=idx)
    res_feat = known_features_control(fwd_s, sig_s, feats)
    _show("4 known_features_control", res_feat)
    daily_ev = new_signal * fwd
    res_sub = subperiod_stability(daily_ev, periods=3, n_perm=200, seed=0)
    _show("5 subperiod_stability", res_sub)
    years = pd.get_dummies(idx.year).to_numpy(dtype=float)
    res_cal = calendar_control(fwd_s, sig_s, years)
    _show("6 calendar_control", res_cal)

    # layer-0 verification: assert everything the demo just printed,
    # then check the trash cases
    _verify_planted_edge(
        res_perm=res_perm,
        res_plac=res_plac,
        res_mom=res_mom,
        res_feat=res_feat,
        res_sub=res_sub,
        res_cal=res_cal,
        n_days=n_days,
    )
    _verify_trash_cases(fwd_s, mom7, mom12, feats, n_days)

    # subperiod_stability: the DEFAULT (max_share_allowed=None) auto-
    # scales the cap to 1/periods + 0.15; the legacy fixed 0.60 is now
    # an explicit opt-in that warns at periods >= 4.  A planted
    # concentration the legacy cap tolerates at periods=5 gets killed
    # by the auto cap 1/5 + 0.15 = 0.35.
    print()
    print("=" * 72)
    print(
        "subperiod_stability: auto-scale default vs legacy 0.60 opt-in"
        " at periods=5"
    )
    print("=" * 72)
    # chunk sums [120, 30, 30, 30, 30]: one of five chunks holds
    # max_share 0.5 of the total evidence
    conc_ev = np.concatenate([np.full(60, 2.0), np.full(240, 0.5)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_fixed = subperiod_stability(
            conc_ev, periods=5, max_share_allowed=0.60, n_perm=200, seed=0
        )
    _show("5b subperiod_stability legacy 0.60 (opt-in)", res_fixed)
    assert caught, "explicit fixed cap at periods>=4 must raise a warning"
    print(f"   warning: {caught[0].message}")
    res_auto = subperiod_stability(conc_ev, periods=5, n_perm=200, seed=0)
    _show("5b subperiod_stability auto cap (default)", res_auto)
    _verify_auto_scale_cap(res_fixed, res_auto)


if __name__ == "__main__":
    run_all_examples()
