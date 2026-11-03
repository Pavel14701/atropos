"""Standard evaluation metrics for any backtest / strategy return stream.

Every function is a pure mapping: data in -> result dict out.  Sharpe CIs
use a circular moving-block bootstrap (n_boot=1000, block=20) because
daily returns overlap and autocorrelate; the deflated Sharpe ratio
follows Bailey & Lopez de Prado (2014) to punish multiple-testing across
``n_trials`` strategies.

Functions
---------
``sharpe_ratio``     annualized SR + block-bootstrap CI
``max_drawdown``     max peak-to-trough drawdown + underwater duration
``calmar_ratio``     CAGR / |max drawdown|
``deflated_sharpe``  DSR + p-value under ``n_trials`` strategy trials
``profit_factor``    gross wins / gross losses
``win_rate``         win rate + Wilson score CI
``turnover``         annualized one-sided turnover from a position series

Run ``python -m atropos.evaluation_metrics`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "calmar_ratio",
    "deflated_sharpe",
    "equity_from_returns",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "turnover",
    "win_rate",
]

_EULER_GAMMA = 0.5772156649015329


def _clean(values: Any) -> np.ndarray:
    """Coerce array-like input to a flat 1-D float array, NaNs dropped."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def equity_from_returns(returns: Any, initial: float = 1.0) -> np.ndarray:
    """Compound a return series into an equity curve.

    Example::

        >>> equity_from_returns([0.1, -0.05])          # doctest: +SKIP
        array([1.1  , 1.045])
    """
    rets = _clean(returns)
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
    """
    rets = _clean(returns)
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
        block = int(min(max(block, 1), n_obs))
        n_blocks = int(np.ceil(n_obs / block))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n_obs, size=(int(n_boot), n_blocks))
        idx = (starts[:, :, None] + np.arange(block)) % n_obs
        idx = idx.reshape(int(n_boot), n_blocks * block)[:, :n_obs]
        sds = excess[idx].std(axis=1, ddof=1)
        mus = excess[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr_boot = np.where(sds > 0, mus / sds, np.nan) * float(
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
    compounding).  ``mdd_duration`` is measured in index units (days for
    daily data): the longest span between a running peak and the eventual
    recovery (or the end of the sample if still underwater).

    Returns ``{mdd, mdd_duration, peak_index, trough_index,
    still_underwater, n}``.

    Example::

        >>> out = max_drawdown([1.0, 1.2, 0.9, 1.1, 1.3])   # doctest: +SKIP
        >>> out["mdd"] < -0.2                               # doctest: +SKIP
        True
    """
    eq = _clean(equity_curve)
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

    Returns ``{calmar, cagr, mdd, n}``.

    Example::

        >>> rng = np.random.default_rng(0)                  # doctest: +SKIP
        >>> rets = rng.normal(0.0007, 0.01, 1500)           # doctest: +SKIP
        >>> calmar_ratio(rets)["calmar"] > 0                # doctest: +SKIP
        True
    """
    rets = _clean(returns)
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
    """E[max of T iid standard normals]: (1-g)*Z(1-1/T) + g*Z(1-1/(T*e))."""
    trials = max(int(n_trials), 2)
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
    """Bootstrap variance of the annualized-SR estimator.

    Used as a proxy for the across-trial SR variance in
    :func:`deflated_sharpe`.
    """
    n_obs = int(returns.size)
    block = int(min(max(block, 1), n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs, size=(int(n_boot), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    idx = idx.reshape(int(n_boot), n_blocks * block)[:, :n_obs]
    mu = returns[idx].mean(axis=1)
    sd = returns[idx].std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr_boot = np.where(sd > 0, mu / sd * np.sqrt(periods), np.nan)
    sr_boot = sr_boot[np.isfinite(sr_boot)]
    if sr_boot.size < 10:
        return None
    return float(np.var(sr_boot, ddof=1))


def deflated_sharpe(
    returns: Any,
    n_trials: int,
    *,
    periods: int = 252,
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

    Parameters
    ----------
    sr_var_annual : variance of SRs across trials, in ANNUALIZED SR
        units.  Default 1.0 (unit-variance trials).  Pass the string
        ``"bootstrap"`` to estimate it from the block-bootstrap
        dispersion of the SR estimator (a proxy - it is estimator
        variance, not trial variance).

    Returns ``{dsr, p_value, sr_annual, sr_benchmark_annual,
    sr_var_annual, skew, kurtosis, n_trials, n}``.  ``p_value = 1 - DSR``;
    reject the "best of N trials" null when ``p_value < 0.05``.

    Example::

        >>> rng = np.random.default_rng(0)                  # doctest: +SKIP
        >>> rets = rng.normal(0.0004, 0.01, 1500)           # doctest: +SKIP
        >>> deflated_sharpe(rets, n_trials=100)["dsr"] > 0  # doctest: +SKIP
        True
    """
    rets = _clean(returns)
    n_obs = int(rets.size)
    stdev = float(np.std(rets, ddof=1)) if n_obs > 1 else float("nan")
    sr_pp = (
        float(np.mean(rets) / stdev)
        if (n_obs > 1 and stdev > 0)
        else float("nan")
    )
    sr_a = (
        float(sr_annual)
        if sr_annual is not None
        else (
            sr_pp * float(np.sqrt(periods))
            if np.isfinite(sr_pp)
            else float("nan")
        )
    )

    if (
        isinstance(sr_var_annual, str)
        and sr_var_annual.lower() == "bootstrap"
    ):
        est = _boot_sr_var_annual(rets, n_boot, block, periods, seed)
        sr_var_used = est if (est is not None and est > 0) else 1.0
        var_source = "bootstrap" if est is not None else "fallback=1.0"
    else:
        sr_var_used = float(sr_var_annual)
        var_source = "given"

    skew = float(stats.skew(rets)) if n_obs >= 8 else float("nan")
    kurt = (
        float(stats.kurtosis(rets, fisher=False))
        if n_obs >= 8
        else float("nan")
    )

    sr0_pp = float(
        np.sqrt(max(sr_var_used, 1e-12) / periods)
    ) * _expected_max_normal(n_trials)
    denom2 = 1.0 - skew * sr_pp + ((kurt - 1.0) / 4.0) * sr_pp * sr_pp
    denom = float(np.sqrt(max(denom2, 1e-12)))
    if np.isfinite(sr_pp) and n_obs > 1:
        dsr = float(
            stats.norm.cdf((sr_pp - sr0_pp) * np.sqrt(n_obs - 1) / denom)
        )
    else:
        dsr = float("nan")
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

        >>> profit_factor([100, -40, 60, -10])["pf"]     # doctest: +SKIP
        3.2
    """
    pnl = _clean(trades)
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

        >>> win_rate([1, -1, 1, 1, -1, 1, 1, 1])["wr"]   # doctest: +SKIP
        0.75
    """
    pnl = _clean(trades)
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
    """Annualized turnover of a position series (one-sided: sum of |dw|).

    ``positions`` is the per-period position/weight series; each
    rebalance contributes ``|w_t - w_{t-1}|``.  ``rt_per_year`` scales
    the mean by ``periods`` (252 for daily data).

    Returns ``{rt_per_year, turnover_per_period, n_rebalances,
    periods}``.

    Example::

        >>> out = turnover(                                # doctest: +SKIP
        ...     [0.0, 1.0, 1.0, -1.0, 0.0])
        >>> out["turnover_per_period"]                     # doctest: +SKIP
        1.0
    """
    pos = _clean(positions)
    n_obs = int(pos.size)
    if n_obs < 2:
        return {
            "rt_per_year": float("nan"),
            "turnover_per_period": float("nan"),
            "n_rebalances": max(n_obs - 1, 0),
            "periods": int(periods),
        }
    diffs = np.abs(np.diff(pos))
    tvr = float(np.mean(diffs))
    return {
        "rt_per_year": tvr * float(periods),
        "turnover_per_period": tvr,
        "n_rebalances": int(diffs.size),
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

    eq = equity_from_returns(returns)
    dd = max_drawdown(eq)
    print(
        f"max_drawdown : mdd={dd['mdd']:.2%} "
        f"duration={dd['mdd_duration']}d "
        f"peak_i={dd['peak_index']} trough_i={dd['trough_index']}"
    )

    cal = calmar_ratio(returns)
    print(
        f"calmar_ratio : calmar={cal['calmar']:.2f} "
        f"cagr={cal['cagr']:.2%} mdd={cal['mdd']:.2%}"
    )

    for trials in (1, 10, 100):
        dsr = deflated_sharpe(returns, n_trials=trials, seed=0)
        print(
            f"deflated_sharpe(trials={trials:3d}): dsr={dsr['dsr']:.3f} "
            f"p={dsr['p_value']:.4f} "
            f"sr0_annual={dsr['sr_benchmark_annual']:.2f}"
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

    positions = np.where(returns > 0, 1.0, -1.0)
    to = turnover(positions)
    print(
        f"turnover     : {to['rt_per_year']:.1f} per year "
        f"({to['turnover_per_period']:.3f} per day)"
    )


if __name__ == "__main__":
    run_all_examples()
