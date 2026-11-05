"""Standard evaluation metrics for any backtest / strategy return stream.

Every function is a pure mapping: data in -> result dict out.  Sharpe CIs
use a circular moving-block bootstrap (n_boot=1000, block=20) because
daily returns overlap and autocorrelate; the deflated Sharpe ratio
follows Bailey & Lopez de Prado (2014) to punish multiple-testing across
``n_trials`` strategies (``n_trials=1`` carries no penalty: DSR equals
the probabilistic Sharpe ratio).

Functions
---------
``sharpe_ratio``     annualized SR + block-bootstrap CI
``max_drawdown``     max peak-to-trough drawdown + underwater duration
``calmar_ratio``     CAGR / |max drawdown|
``deflated_sharpe``  DSR + p-value under ``n_trials`` strategy trials
``profit_factor``    gross wins / gross losses
``win_rate``         win rate + Wilson score CI
``turnover``         annualized total (two-sided) turnover of positions

Run ``python -m atropos.evaluation_metrics`` for a synthetic demo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from atropos._utils import (
    block_indices,
    clean_1d,
    sane_variance_mask,
    validate_int,
    validate_number,
)

__all__ = [
    "calmar_ratio",
    "deflated_sharpe",
    "deflated_sharpe_from_moments",
    "equity_from_returns",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "turnover",
    "win_rate",
]

_EULER_GAMMA = 0.5772156649015329


def _validate_unit_interval(name: str, value: float) -> None:
    """Raise ValueError unless ``value`` lies strictly inside (0, 1).

    Non-numeric input (``object()``, strings) is re-raised as
    ``ValueError`` naming the parameter, per the shared
    ``validate_number`` contract.
    """
    number = validate_number(name, value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def _validate_finite(name: str, value: float) -> None:
    """Raise ValueError unless ``value`` is a finite number.

    Non-numeric input (``object()``, strings) is re-raised as
    ``ValueError`` naming the parameter, per the shared
    ``validate_number`` contract.
    """
    validate_number(name, value)


def equity_from_returns(returns: Any, initial: float = 1.0) -> np.ndarray:
    """Compound a return series into an equity curve.

    Example::

        >>> equity_from_returns([0.1, -0.05])          # doctest: +SKIP
        array([1.1  , 1.045])
    """
    rets = clean_1d(returns)
    return initial * np.cumprod(1.0 + rets)


# ----------------------------------------------------------------------------
# Sharpe
# ----------------------------------------------------------------------------
def sharpe_ratio(
    returns: Any,
    rf: float = 0.0,
    *,
    periods: int = 252,
    n_boot: int = 1000,
    block: int = 20,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Annualized Sharpe ratio with a block-bootstrap CI.

    ``rf`` is an ANNUAL rate (decimal); it is de-periodized internally.
    The CI is computed on the per-period SR with a circular moving-block
    bootstrap, then rescaled by ``sqrt(periods)``.

    Returns ``{sr, sr_ci, sr_per_period, ci_method, n, rf}``.

    Example::

        >>> rng = np.random.default_rng(0)                  # doctest: +SKIP
        >>> rets = 0.0005 + rng.standard_normal(1000)*0.01  # doctest: +SKIP
        >>> out = sharpe_ratio(rets)                        # doctest: +SKIP
        >>> out["sr_ci"][0] < out["sr"] < out["sr_ci"][1]   # doctest: +SKIP
        True

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1) or not a finite
            number, ``periods`` / ``n_boot`` / ``block`` is not an
            integer >= 1, or ``rf`` is not finite; raised before the
            data is touched.
    """
    _validate_unit_interval("ci_level", ci_level)
    periods = validate_int("periods", periods)
    n_boot = validate_int("n_boot", n_boot)
    block = validate_int("block", block)
    _validate_finite("rf", rf)
    rets = clean_1d(returns)
    n_obs = int(rets.size)
    excess = rets - float(rf) / float(periods)
    stdev = float(np.std(excess, ddof=1))
    sr_pp = float(np.mean(excess) / stdev) if stdev > 0 else float("nan")
    sr_ann = (
        sr_pp * float(np.sqrt(periods))
        if np.isfinite(sr_pp)
        else float("nan")
    )

    ci: tuple[float, float] = (float("nan"), float("nan"))
    method = "insufficient_data"
    if n_obs >= 8 and stdev > 0:
        rng = np.random.default_rng(seed)
        idx = block_indices(n_obs, int(n_boot), block, rng)
        sds = excess[idx].std(axis=1, ddof=1)
        mus = excess[idx].mean(axis=1)
        usable = sane_variance_mask(mus, sds)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr_boot = np.where(usable, mus / sds, np.nan) * float(
                np.sqrt(periods)
            )
        sr_boot = sr_boot[np.isfinite(sr_boot)]
        if sr_boot.size >= 10:
            tail = (1.0 - ci_level) / 2.0
            ci = (
                float(np.quantile(sr_boot, tail)),
                float(np.quantile(sr_boot, 1.0 - tail)),
            )
            method = "moving_block"
    return {
        "sr": sr_ann,
        "sr_ci": ci,
        "sr_per_period": sr_pp,
        "ci_method": method,
        "n": n_obs,
        "rf": float(rf),
    }


# ----------------------------------------------------------------------------
# drawdown / calmar
# ----------------------------------------------------------------------------
def max_drawdown(equity_curve: Any) -> dict[str, Any]:
    """Max peak-to-trough drawdown and longest underwater duration.

    ``equity_curve`` must be strictly positive (multiplicative
    compounding) and 1-D; higher-dimensional input raises instead of
    being silently flattened.  ``mdd_duration`` is measured in index
    units (days for daily data): the longest span between a running peak
    and the eventual recovery (or the end of the sample if still
    underwater).

    Returns ``{mdd, mdd_duration, peak_index, trough_index,
    still_underwater, n}``.

    Example::

        >>> out = max_drawdown([1.0, 1.2, 0.9, 1.1, 1.3])   # doctest: +SKIP
        >>> out["mdd"] < -0.2                               # doctest: +SKIP
        True
    """
    raw = np.asarray(equity_curve, dtype=float)
    # np.asarray pins the ndarray type: stubs resolve this to Any.
    if raw.ndim > 1:
        raise ValueError(
            f"equity_curve must be 1-D, got ndim={raw.ndim}; flatten "
            "explicitly instead of relying on silent reshaping"
        )
    eq = clean_1d(raw)
    n_obs = int(eq.size)
    if n_obs == 0:
        return {
            "mdd": float("nan"),
            "mdd_duration": 0,
            "peak_index": -1,
            "trough_index": -1,
            "still_underwater": False,
            "n": 0,
        }
    if np.any(eq <= 0):
        raise ValueError(
            "equity_curve must be strictly positive for multiplicative "
            "drawdown"
        )
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    trough_index = int(np.argmin(dd))
    peak_index = int(np.argmax(eq[: trough_index + 1]))
    mdd = float(dd[trough_index])
    positions = np.arange(n_obs)
    recent_peak_idx = np.maximum.accumulate(
        np.where(eq >= peak, positions, 0)
    )
    underwater = dd < 0
    duration = (
        int(np.max(positions[underwater] - recent_peak_idx[underwater]))
        if underwater.any()
        else 0
    )
    return {
        "mdd": mdd,
        "mdd_duration": duration,
        "peak_index": peak_index,
        "trough_index": trough_index,
        "still_underwater": bool(dd[-1] < 0),
        "n": n_obs,
    }


def calmar_ratio(returns: Any, periods: int = 252) -> dict[str, Any]:
    """Calmar ratio: annualized geometric return over |max drawdown|.

    ``mdd <= 0`` always holds (drawdown is computed as ``eq/peak - 1``),
    so the zero-drawdown branch below fires when ``mdd == 0``: a curve
    that never fell below a running peak earns an infinite Calmar for
    positive CAGR.  ``mdd >= 0`` is kept as a defensive guard only.

    Returns ``{calmar, cagr, mdd, n}``.

    Raises:
        ValueError: if ``periods`` is not an integer >= 1; raised
            before the data is touched.

    Example::

        >>> rng = np.random.default_rng(0)                  # doctest: +SKIP
        >>> rets = rng.normal(0.0007, 0.01, 1500)           # doctest: +SKIP
        >>> calmar_ratio(rets)["calmar"] > 0                # doctest: +SKIP
        True
    """
    periods = validate_int("periods", periods)
    rets = clean_1d(returns)
    n_obs = int(rets.size)
    if n_obs < 2:
        return {
            "calmar": float("nan"),
            "cagr": float("nan"),
            "mdd": float("nan"),
            "n": n_obs,
        }
    eq = np.cumprod(1.0 + rets)
    years = n_obs / float(periods)
    if eq[-1] <= 0:
        cagr = float("nan")
    else:
        cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    mdd = max_drawdown(eq)["mdd"]
    # nan mdd (defensive) or mdd == 0 (no drawdown at all): the ratio
    # divides by zero -> +inf for a positive CAGR, NaN otherwise.
    if not np.isfinite(mdd) or mdd >= 0:
        calmar = (
            float("inf") if (np.isfinite(cagr) and cagr > 0) else float("nan")
        )
    else:
        calmar = float(cagr / abs(mdd))
    return {"calmar": calmar, "cagr": cagr, "mdd": float(mdd), "n": n_obs}


# ----------------------------------------------------------------------------
# deflated Sharpe (Bailey & Lopez de Prado, 2014)
# ----------------------------------------------------------------------------
def _expected_max_normal(n_trials: int) -> float:
    """E[max of T iid standard normals]; T <= 1 returns 0.0.

    For T >= 2 the Bailey & Lopez de Prado approximation
    ``(1-g)*Z(1-1/T) + g*Z(1-1/(T*e))`` is used.  T <= 1 means a single
    trial: there is no selection among strategies, the expected maximum
    of one N(0,1) draw is its mean 0.0, and the multiple-testing penalty
    vanishes (DSR degenerates to the probabilistic Sharpe ratio).  The
    old ``max(T, 2)`` clamp charged a single strategy for two trials and
    understated its DSR.
    """
    trials = int(n_trials)
    if trials <= 1:
        return 0.0
    z1 = float(stats.norm.ppf(1.0 - 1.0 / trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (trials * np.e)))
    return (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2


def _boot_sr_var_annual(
    returns: np.ndarray,
    n_boot: int,
    block: int,
    periods: int,
    seed: int | None,
) -> float | None:
    """Bootstrap variance of the annualized-SR ESTIMATOR.

    This resamples ONE strategy's return stream, so it measures how
    noisily the SR of that single strategy is estimated -- it is NOT the
    variance of Sharpe ratios ACROSS trials that Bailey & Lopez de
    Prado's DSR strictly wants.  :func:`deflated_sharpe` accepts it as a
    proxy via ``sr_var_annual="bootstrap"``; treat results from that
    mode accordingly.
    """
    n_obs = int(returns.size)
    rng = np.random.default_rng(seed)
    idx = block_indices(n_obs, int(n_boot), block, rng)
    mu = returns[idx].mean(axis=1)
    sd = returns[idx].std(axis=1, ddof=1)
    usable = sane_variance_mask(mu, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr_boot = np.where(usable, mu / sd * np.sqrt(periods), np.nan)
    sr_boot = sr_boot[np.isfinite(sr_boot)]
    if sr_boot.size < 10:
        return None
    return float(np.var(sr_boot, ddof=1))


def deflated_sharpe_from_moments(
    sr: float,
    n_obs: int,
    skew_val: float,
    kurt_val: float,
    var_sr: float,
    *,
    n_trials: int,
) -> float:
    """Deflated Sharpe ratio (CDF form) from pre-computed moments.

    THE single computational home of the Bailey & Lopez de Prado (2014)
    deflation formula shared by :func:`deflated_sharpe` (per-period
    return stream -> moments here) and the ML engine's
    ``atropos.ml_training.deflated_sharpe_value`` (already-computed
    moments of a per-trade/per-day EV stream).  Everything is in
    PER-PERIOD units: ``sr`` and ``var_sr`` must use the same
    de-annualization convention (the returns-based wrapper converts
    ``sr_var_annual / periods`` before calling).

    ``DSR = Phi((sr - sr0) * sqrt(n_obs - 1) / sqrt(denom))`` with
    ``denom = 1 - skew*sr + (kurt - 1)/4 * sr^2`` and the benchmark
    ``sr0 = sqrt(var_sr) * E[max of n_trials unit normals]``
    (``n_trials <= 1`` carries no multiple-testing penalty:
    ``sr0 = 0``, DSR degenerates to the probabilistic Sharpe ratio).

    NaN policy (degenerate input is never a fabricated 0 or 1): a
    non-finite moment, ``n_obs < 3`` (no honest DSR below three
    observations), a negative trial variance (sign error) or a
    degenerate denominator (``denom <= 0``: the normal approximation
    leaves its domain at the vertex of that quadratic) all yield NaN.

    Args:
        sr: (per-period) Sharpe ratio of the stream.
        n_obs: number of observations in the stream, integer >= 0
            (an empty stream is the returns-based caller's degenerate
            case; anything below 3 yields NaN, not an error).
        skew_val: sample skewness of the stream.
        kurt_val: Pearson (non-excess) kurtosis of the stream.
        var_sr: variance of SRs across the sibling trial streams, in
            the same per-period SR units as ``sr``.
        n_trials: declared number of trials, integer >= 1 (values
            <= 1 get a zero benchmark; callers may enforce a stricter
            minimum before calling, as the ML engine does).

    Returns:
        The deflated Sharpe ratio ``Phi(z)`` in [0, 1], or NaN.

    Raises:
        ValueError: if ``n_obs`` is not an integer >= 0 or ``n_trials``
            is not an integer >= 1.

    Example::

        >>> round(deflated_sharpe_from_moments(
        ...     0.0, 100, 0.0, 3.0, 0.0, n_trials=5
        ... ), 6)
        0.5
    """
    n_obs = validate_int("n_obs", n_obs, minimum=0)
    n_trials = validate_int("n_trials", n_trials)
    if not all(
        math.isfinite(float(value))
        for value in (sr, skew_val, kurt_val, var_sr)
    ):
        return math.nan
    if n_obs < 3 or var_sr < 0.0:
        return math.nan
    expected_max = _expected_max_normal(n_trials)
    sr0 = math.sqrt(var_sr) * expected_max
    denom2 = 1.0 - skew_val * sr + ((kurt_val - 1.0) / 4.0) * sr * sr
    if not np.isfinite(denom2) or denom2 <= 0.0:
        return math.nan
    denom = float(np.sqrt(max(denom2, 1e-12)))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))


def deflated_sharpe(
    returns: Any,
    n_trials: int,
    *,
    periods: int = 252,
    rf: float = 0.0,
    sr_annual: float | None = None,
    sr_var_annual: Any = 1.0,
    n_boot: int = 200,
    block: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    Probabilistically shrinks a Sharpe ratio for having tried
    ``n_trials`` strategies.  ``DSR = Phi( (SR - SR0) * sqrt(n-1) /
    sqrt(1 - g3*SR + ((g4-1)/4)*SR^2) )`` with per-period SR and ``SR0``
    the expected max SR of ``n_trials`` unit-variance trials.
    ``n_trials <= 1`` carries no multiple-testing penalty (``SR0 = 0``)
    and DSR equals the probabilistic Sharpe ratio.

    The formula's single computational home is
    :func:`deflated_sharpe_from_moments`; this wrapper estimates the
    moments from ``returns`` and delegates.

    Args:
        returns: 1-D array-like of per-period returns (NaNs dropped).
        n_trials: number of strategy trials the SR was selected over.
        periods: periods per year (252 for daily data).
        rf: ANNUAL risk-free rate (decimal), de-periodized by
            ``periods`` and subtracted from ``returns``.  Default 0.0
            leaves the input untouched: like the rest of the DSR
            literature this function expects EXCESS returns, so pass
            ``rf`` when the stream is gross.
        sr_annual: annualized Sharpe ratio of the excess stream, in the
            same ``periods`` convention.  When given, it DRIVES the
            test: DSR is computed from its de-annualized per-period
            value ``sr_annual / sqrt(periods)`` instead of the sample
            SR (the output field reports the passed value unchanged).
            When ``None`` (default) the per-period SR is estimated from
            ``returns``.
        sr_var_annual: variance of SRs across trials, in ANNUALIZED SR
            units.  Default 1.0 (unit-variance trials).  Pass the
            string ``"bootstrap"`` to estimate it from the
            block-bootstrap dispersion of the SR estimator (a proxy -
            it is estimator variance, not trial variance; see
            ``_boot_sr_var_annual``).
        n_boot: bootstrap replicates when ``sr_var_annual`` is
            ``"bootstrap"``.
        block: block length when ``sr_var_annual`` is ``"bootstrap"``.
        seed: seed of the bootstrap RNG.

    Applicability: the normal-approximation denominator ``1 - g3*SR +
    (g4-1)/4*SR^2`` is non-negative in exact arithmetic for any valid
    (skew, kurtosis) pair.  A computed value ``<= 0`` means the inputs
    sit at the degenerate vertex of that quadratic (two-point-like
    samples under float rounding, or an ``sr_annual`` far from the
    sample): the approximation is out of its domain, and ``dsr`` /
    ``p_value`` are NaN rather than a fabricated 0 or 1.

    Returns:
        Dict ``{dsr, p_value, sr_annual, sr_benchmark_annual,
        sr_var_annual, sr_var_source, skew, kurtosis, n_trials, n}``.
        ``p_value = 1 - DSR``; reject the "best of N trials" null when
        ``p_value < 0.05``.

    Raises:
        ValueError: if ``n_trials``, ``periods``, ``n_boot`` or
            ``block`` is not an integer >= 1, ``rf`` is not finite,
            or ``sr_annual`` is given and not finite; raised before
            the data is touched.

    Example::

        >>> rng = np.random.default_rng(0)                  # doctest: +SKIP
        >>> rets = rng.normal(0.0004, 0.01, 1500)           # doctest: +SKIP
        >>> deflated_sharpe(rets, n_trials=100)["dsr"] > 0  # doctest: +SKIP
        True
    """
    n_trials = validate_int("n_trials", n_trials)
    periods = validate_int("periods", periods)
    n_boot = validate_int("n_boot", n_boot)
    block = validate_int("block", block)
    _validate_finite("rf", rf)
    if sr_annual is not None:
        _validate_finite("sr_annual", sr_annual)
    rets = clean_1d(returns)
    n_obs = int(rets.size)
    excess = rets - float(rf) / float(periods)
    stdev = float(np.std(excess, ddof=1)) if n_obs > 1 else float("nan")
    sr_pp_sample = (
        float(np.mean(excess) / stdev)
        if (n_obs > 1 and stdev > 0)
        else float("nan")
    )
    if sr_annual is not None:
        # The caller-supplied annualized SR drives the test; the sample
        # SR would silently contradict the reported figure otherwise.
        sr_pp_used = float(sr_annual) / float(np.sqrt(periods))
        sr_a = float(sr_annual)
    else:
        sr_pp_used = sr_pp_sample
        sr_a = (
            sr_pp_sample * float(np.sqrt(periods))
            if np.isfinite(sr_pp_sample)
            else float("nan")
        )

    if (
        isinstance(sr_var_annual, str)
        and sr_var_annual.lower() == "bootstrap"
    ):
        est = _boot_sr_var_annual(excess, n_boot, block, periods, seed)
        sr_var_used = est if (est is not None and est > 0) else 1.0
        var_source = "bootstrap" if est is not None else "fallback=1.0"
    else:
        sr_var_used = float(sr_var_annual)
        var_source = "given"

    skew = float(stats.skew(excess)) if n_obs >= 8 else float("nan")
    kurt = (
        float(stats.kurtosis(excess, fisher=False))
        if n_obs >= 8
        else float("nan")
    )

    sr0_pp = float(
        np.sqrt(max(sr_var_used, 1e-12) / periods)
    ) * _expected_max_normal(n_trials)
    # The deflation math itself lives in
    # :func:`deflated_sharpe_from_moments` (single home shared with the
    # ML engine's moment-based wrapper).  The variance unit-floor stays
    # HERE in annual units, applied before the per-period conversion,
    # so this wrapper's numbers stay bit-compatible with the
    # pre-unification implementation; the reported ``sr_var_annual``
    # field stays the raw input.
    var_sr_pp = max(sr_var_used, 1e-12) / periods
    dsr = deflated_sharpe_from_moments(
        sr_pp_used, n_obs, skew, kurt, var_sr_pp, n_trials=n_trials
    )
    return {
        "dsr": dsr,
        "p_value": float(1.0 - dsr) if np.isfinite(dsr) else float("nan"),
        "sr_annual": sr_a,
        "sr_benchmark_annual": sr0_pp * float(np.sqrt(periods)),
        "sr_var_annual": sr_var_used,
        "sr_var_source": var_source,
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": int(n_trials),
        "n": n_obs,
    }


# ----------------------------------------------------------------------------
# trade-level metrics
# ----------------------------------------------------------------------------
def profit_factor(trades: Any) -> dict[str, Any]:
    """Gross profit / |gross loss| over a series of trade PnLs.

    Returns ``{pf, gross_win, gross_loss, n_trades, n_winners,
    n_losers}``.  ``pf`` is ``inf`` when there are losses-free trades
    with positive PnL and NaN when there is no PnL at all.

    Example::

    >>> profit_factor([100, -40, 60, -10])["pf"]
    3.2
    """
    pnl = clean_1d(trades)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = float("nan")
    return {
        "pf": float(pf),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "n_trades": int(pnl.size),
        "n_winners": int(wins.size),
        "n_losers": int(losses.size),
    }


def win_rate(trades: Any, ci_level: float = 0.95) -> dict[str, Any]:
    """Win rate with a Wilson score CI (robust for small trade counts).

    Returns ``{wr, ci_low, ci_high, ci_level, n}``.

    Example::

        >>> win_rate([1, -1, 1, 1, -1, 1, 1, 1])["wr"]
        0.75

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1); raised before
            the data is touched.
    """
    _validate_unit_interval("ci_level", ci_level)
    pnl = clean_1d(trades)
    n_obs = int(pnl.size)
    if n_obs == 0:
        return {
            "wr": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_level": float(ci_level),
            "n": 0,
        }
    wr = float(np.mean(pnl > 0))
    z_crit = float(stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    denom = 1.0 + z_crit * z_crit / n_obs
    center = wr + z_crit * z_crit / (2.0 * n_obs)
    half = z_crit * np.sqrt(
        wr * (1.0 - wr) / n_obs + z_crit * z_crit / (4.0 * n_obs * n_obs)
    )
    lo = max(0.0, (center - half) / denom)
    hi = min(1.0, (center + half) / denom)
    return {
        "wr": wr,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": float(ci_level),
        "n": n_obs,
    }


def turnover(positions: Any, periods: int = 252) -> dict[str, Any]:
    """Annualized turnover of a position series -- TOTAL (two-sided).

    ``positions`` is the per-period position/weight series; each step
    contributes ``|w_t - w_{t-1}|``, which counts BOTH legs of a
    rebalance (the sell and the buy), so the metric is total
    (two-sided) turnover.  The one-sided convention used by some
    venues is HALF of the values reported here.  ``rt_per_year`` scales
    the mean by ``periods`` (252 for daily data).

    Returns ``{rt_per_year, turnover_per_period, n_rebalances,
    periods}``.

    Example::

        >>> out = turnover(
        ...     [0.0, 1.0, 1.0, -1.0, 0.0])
        >>> out["turnover_per_period"]
        1.0

    Raises:
        ValueError: if ``periods`` is not an integer >= 1; raised
            before the data is touched.
    """
    periods = validate_int("periods", periods)
    pos = clean_1d(positions)
    n_obs = int(pos.size)
    if n_obs < 2:
        return {
            "rt_per_year": float("nan"),
            "turnover_per_period": float("nan"),
            "n_rebalances": 0,
            "periods": int(periods),
        }
    diffs = np.abs(np.diff(pos))
    tvr = float(np.mean(diffs))
    return {
        "rt_per_year": tvr * float(periods),
        "turnover_per_period": tvr,
        # actual position CHANGES, not the raw number of diff steps
        "n_rebalances": int(np.count_nonzero(diffs > 1e-12)),
        "periods": int(periods),
    }


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo every metric on one synthetic 3-year daily backtest."""
    print("=" * 72)
    print("evaluation_metrics demo: synthetic 3-year daily strategy")
    print("=" * 72)
    rng = np.random.default_rng(9)
    n_obs = 756
    shocks = rng.standard_normal(n_obs)
    ar = np.zeros(n_obs)
    for step in range(1, n_obs):
        ar[step] = 0.3 * ar[step - 1] + 0.95 * shocks[step]
    returns = 0.0006 + 0.01 * ar

    sr = sharpe_ratio(returns, rf=0.02, seed=0)
    print(
        f"sharpe_ratio : sr={sr['sr']:.2f} "
        f"CI=[{sr['sr_ci'][0]:.2f}, {sr['sr_ci'][1]:.2f}] "
        f"({sr['ci_method']}, n={sr['n']})"
    )
    # Layer-0 verification: the demo doubles as a regression test
    sr_low, sr_high = sr["sr_ci"]
    # seeded regression; percentile CI containment of the point
    # estimate is not a theorem -- it holds at this fixed data/seed,
    # and the exact sr / sr_ci values are pinned as a seeded test in
    # tests/test_evaluation_metrics.py
    assert (
        sr["ci_method"] == "moving_block"
        and sr["n"] == n_obs
        and sr_low < sr["sr"] < sr_high
        and sr["sr"] > 0.0
    )

    eq = equity_from_returns(returns)
    dd = max_drawdown(eq)
    print(
        f"max_drawdown : mdd={dd['mdd']:.2%} "
        f"duration={dd['mdd_duration']}d "
        f"peak_i={dd['peak_index']} trough_i={dd['trough_index']}"
    )
    assert (
        dd["n"] == n_obs
        and dd["trough_index"] > dd["peak_index"] >= 0
        and -1.0 < dd["mdd"] < 0.0
        and dd["mdd_duration"] >= 1
    )

    cal = calmar_ratio(returns)
    print(
        f"calmar_ratio : calmar={cal['calmar']:.2f} "
        f"cagr={cal['cagr']:.2%} mdd={cal['mdd']:.2%}"
    )
    # same equity curve -> identical drawdown, and calmar = cagr/|mdd|
    assert cal["cagr"] > 0.0 and cal["calmar"] > 0.0
    assert cal["mdd"] == dd["mdd"]
    assert abs(cal["calmar"] - cal["cagr"] / abs(cal["mdd"])) < 1e-9

    dsr_by_trials: dict[int, dict[str, Any]] = {}
    for trials in (1, 10, 100):
        dsr_by_trials[trials] = deflated_sharpe(
            returns, n_trials=trials, seed=0
        )
        print(
            f"deflated_sharpe(trials={trials:3d}): "
            f"dsr={dsr_by_trials[trials]['dsr']:.3f} "
            f"p={dsr_by_trials[trials]['p_value']:.4f} "
            f"sr0_annual={dsr_by_trials[trials]['sr_benchmark_annual']:.2f}"
        )
    single = dsr_by_trials[1]
    # a single trial carries no multiple-testing penalty: the benchmark
    # is exactly zero and DSR degenerates to the probabilistic SR;
    # compared with an epsilon instead of == to stay float-robust
    assert abs(float(single["sr_benchmark_annual"])) < 1e-15
    assert abs(single["p_value"] - (1.0 - single["dsr"])) < 1e-12
    sr_pp = float(single["sr_annual"]) / float(np.sqrt(252))
    psr_denom = float(
        np.sqrt(
            1.0
            - single["skew"] * sr_pp
            + (single["kurtosis"] - 1.0) / 4.0 * sr_pp * sr_pp
        )
    )
    psr_value = float(
        stats.norm.cdf(
            sr_pp * float(np.sqrt(int(single["n"]) - 1)) / psr_denom
        )
    )
    assert abs(float(single["dsr"]) - psr_value) < 1e-12
    # the deflation penalty grows monotonically with the trial count
    assert (
        dsr_by_trials[1]["dsr"]
        > dsr_by_trials[10]["dsr"]
        > dsr_by_trials[100]["dsr"]
    )
    assert (
        dsr_by_trials[100]["sr_benchmark_annual"]
        > dsr_by_trials[10]["sr_benchmark_annual"]
        > 0.0
    )

    trades = returns * 100.0 + rng.standard_normal(n_obs) * 2.0
    pf = profit_factor(trades)
    wr = win_rate(trades)
    print(
        f"profit_factor: pf={pf['pf']:.2f} (wins={pf['gross_win']:.0f}, "
        f"losses={pf['gross_loss']:.0f})"
    )
    print(
        f"win_rate     : wr={wr['wr']:.3f} "
        f"CI=[{wr['ci_low']:.3f}, {wr['ci_high']:.3f}]"
    )
    assert (
        pf["pf"] > 1.0
        and pf["n_trades"] == n_obs
        and pf["gross_win"] > 0.0
        and pf["gross_loss"] > 0.0
    )
    assert 0.0 < wr["wr"] < 1.0 and wr["n"] == n_obs
    assert wr["ci_low"] <= wr["wr"] <= wr["ci_high"]
    # win_rate and profit_factor count the same positive-PnL trades, and
    # a sign strategy on this drift earns only a mild win-rate edge
    assert (
        abs(wr["wr"] - pf["n_winners"] / n_obs) < 1e-12
        and abs(wr["wr"] - 0.5) < 0.15
    )

    positions = np.where(returns > 0, 1.0, -1.0)
    to = turnover(positions)
    print(
        f"turnover     : {to['rt_per_year']:.1f} per year "
        f"({to['turnover_per_period']:.3f} per day)"
    )
    # +-1 positions change by at most 2 per period
    assert 0.0 < to["turnover_per_period"] <= 2.0
    assert to["n_rebalances"] > 0
    assert abs(to["rt_per_year"] - to["turnover_per_period"] * 252) < 1e-9


if __name__ == "__main__":
    run_all_examples()
