"""Monte Carlo toolkit: simulated paths, bootstrap distributions, risk tails.

Five building blocks for stress-testing any candidate edge before it is
trusted:

* :func:`monte_carlo_paths`         - geometric Brownian motion price
  paths.
* :func:`bootstrap_resample`        - distribution of ANY statistic under
  iid or circular moving-block resampling of the data.
* :func:`max_drawdown_distribution` - p5 / p50 / p95 of max drawdown
  across block-bootstrap resampled return paths (percentile-based risk
  envelope).
* :func:`sharpe_distribution`       - p5 / p50 / p95 of the annualized
  Sharpe ratio across the same resampled paths.
* :func:`minimum_detectable_effect` - smallest partial R-squared
  detectable at a given sample size (delegates to
  ``power_calculator.detectable_effect``; imported, not duplicated).

Drawdowns and Sharpe percentiles use the CIRCULAR MOVING-BLOCK bootstrap
(Kunsch, 1989) so autocorrelation and overlapping-return variance survive
resampling.  ``p5`` of a max-drawdown distribution is the ADVERSE tail (the
worst 5% of simulated paths); ``p95`` is the benign tail.

Run ``python -m atropos.monte_carlo`` for a synthetic demo of all five.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from atropos.power_calculator import detectable_effect

__all__ = [
    "bootstrap_resample",
    "max_drawdown_distribution",
    "minimum_detectable_effect",
    "monte_carlo_paths",
    "sharpe_distribution",
]


# ----------------------------------------------------------------------------
# simulated paths
# ----------------------------------------------------------------------------
def monte_carlo_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    *,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
) -> np.ndarray:
    """Geometric-Brownian-motion price paths, shape ``(n_paths, n_steps)``.

    Simulated in exact log form: ``S_{t+1} = S_t * exp((mu - sigma^2/2)*dt +
    sigma*sqrt(dt)*Z)`` so prices stay strictly positive.  Column ``j`` is the
    price AFTER step ``j+1`` (``s0`` itself is not a column).

    Parameters
    ----------
    s0 : initial price (must be > 0)
    mu, sigma : annualized drift and volatility
    n_paths, n_steps : simulation dimensions (both >= 1)
    dt : time step in years (default one trading day)
    seed : optional RNG seed for reproducibility

    Example::

        >>> paths = monte_carlo_paths(  # doctest: +SKIP
        ...     100, 0.05, 0.2, 1000, 252, seed=0
        ... )
        >>> paths.shape  # doctest: +SKIP
        (1000, 252)
    """
    s0f, muf, sig = float(s0), float(mu), float(sigma)
    if s0f <= 0:
        raise ValueError(f"s0 must be > 0, got {s0f}")
    if sig < 0:
        raise ValueError(f"sigma must be >= 0, got {sig}")
    n_p, n_s = int(n_paths), int(n_steps)
    if n_p < 1 or n_s < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_p, n_s))
    drift = (muf - 0.5 * sig * sig) * float(dt)
    vol = sig * np.sqrt(float(dt))
    increments = drift + vol * shocks
    paths = s0f * np.exp(np.cumsum(increments, axis=1))
    return np.asarray(paths, dtype=float)


# ----------------------------------------------------------------------------
# resampling machinery
# ----------------------------------------------------------------------------
def _block_index_matrix(
    n_obs: int, block: int, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular moving-block resample indices, shape ``(n_sims, n_obs)``."""
    block = int(min(max(int(block), 1), n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=(int(n_sims), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    return idx.reshape(int(n_sims), n_blocks * block)[:, :n_obs]


def _apply_stat(
    samples: np.ndarray, statistic: Callable[..., Any]
) -> np.ndarray:
    """Apply ``statistic`` row-wise (vectorized ``axis=1``, loop fallback)."""
    try:
        out = np.asarray(statistic(samples, axis=1), dtype=float)
        if out.ndim == 1 and out.shape[0] == samples.shape[0]:
            return out
    except Exception:
        pass
    return np.asarray([float(statistic(row)) for row in samples], dtype=float)


def bootstrap_resample(
    data: Any,
    statistic: Callable[..., Any],
    n_boot: int = 1000,
    block: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Distribution of ``statistic`` under resampling of ``data``.

    Parameters
    ----------
    data : 1-D array-like (NaNs dropped before resampling)
    statistic : scalar statistic; tried vectorized (``axis=1``) first, then
        per-row.  Must map a 1-D float array to a float.
    n_boot : number of bootstrap replicates (>= 1)
    block : ``None`` -> iid bootstrap; integer >= 1 -> circular moving-block
        resampling that preserves autocorrelation up to lag ~``block``.

    Returns a ``(n_boot,)`` float array; entries are NaN where the statistic
    failed on a replicate.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> draws = bootstrap_resample(  # doctest: +SKIP
        ...     rng.standard_normal(400), np.mean, n_boot=500, seed=0
        ... )
        >>> draws.shape  # doctest: +SKIP
        (500,)
    """
    values = np.asarray(data, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n_obs = int(values.size)
    nb = int(n_boot)
    if nb < 1:
        raise ValueError(f"n_boot must be >= 1, got {nb}")
    if n_obs == 0:
        return np.full(nb, np.nan)
    rng = np.random.default_rng(seed)
    if block is None:
        idx = rng.integers(0, n_obs, size=(nb, n_obs))
    else:
        idx = _block_index_matrix(n_obs, int(block), nb, rng)
    return _apply_stat(values[idx], statistic)


def _resampled_return_matrix(
    returns: Any, n_sim: int, block: int, seed: int | None
) -> tuple[np.ndarray, int]:
    """Block-resample cleaned returns into an ``(n_sim, n_obs)`` matrix."""
    rets = np.asarray(returns, dtype=float).reshape(-1)
    rets = rets[np.isfinite(rets)]
    n_obs = int(rets.size)
    if n_obs < 20:
        raise ValueError(f"need at least 20 valid returns, got {n_obs}")
    ns = int(n_sim)
    if ns < 1:
        raise ValueError(f"n_sim must be >= 1, got {ns}")
    rng = np.random.default_rng(seed)
    return rets[_block_index_matrix(n_obs, int(block), ns, rng)], n_obs


def _percentiles(valid: np.ndarray) -> tuple[float, float, float]:
    if valid.size < 20:
        return float("nan"), float("nan"), float("nan")
    quantiles = np.quantile(valid, [0.05, 0.50, 0.95])
    return (
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


# ----------------------------------------------------------------------------
# risk / performance distributions
# ----------------------------------------------------------------------------
def max_drawdown_distribution(
    returns: Any,
    n_sim: int = 5000,
    block: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    """Percentiles of max drawdown across block-bootstrap resampled paths.

    Each of ``n_sim`` paths is a circular moving-block resample of the return
    series (block length ``block`` preserves autocorrelation); the max
    peak-to-trough drawdown is computed on the compounded equity of each path.
    Paths whose equity goes non-positive (a return <= -1 resampled) are
    dropped and counted in ``n_dropped_paths``.

    Returns ``{mdd_p5, mdd_p50, mdd_p95, n_valid_paths, n_dropped_paths,
    n_sim, n, block, note}`` where ``mdd_p5`` is the ADVERSE tail (worst 5% of
    paths, most negative) and ``mdd_p95`` the benign tail.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> rets = rng.normal(0.0005, 0.01, 1000)  # doctest: +SKIP
        >>> res = max_drawdown_distribution(  # doctest: +SKIP
        ...     rets, n_sim=1000, seed=0
        ... )
        >>> ok = (  # doctest: +SKIP
        ...     res["mdd_p5"] < res["mdd_p50"] < res["mdd_p95"] < 0
        ... )
        >>> ok  # doctest: +SKIP
        True
    """
    paths, n_obs = _resampled_return_matrix(returns, n_sim, block, seed)
    eq = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / peak - 1.0
    mdd = dd.min(axis=1)
    valid = mdd[np.isfinite(mdd)]
    p5, p50, p95 = _percentiles(valid)
    return {
        "mdd_p5": p5,
        "mdd_p50": p50,
        "mdd_p95": p95,
        "n_valid_paths": int(valid.size),
        "n_dropped_paths": int(mdd.size - valid.size),
        "n_sim": int(n_sim),
        "n": n_obs,
        "block": int(block),
        "note": (
            "mdd_p5 = adverse tail (worst 5% of paths); mdd_p95 = benign tail"
        ),
    }


def sharpe_distribution(
    returns: Any,
    n_sim: int = 5000,
    block: int = 20,
    periods: int = 252,
    seed: int | None = None,
) -> dict[str, Any]:
    """Percentiles of the annualized Sharpe ratio across resampled paths.

    Same circular moving-block machinery as :func:`max_drawdown_distribution`;
    each replicate's Sharpe is ``mean/sd * sqrt(periods)`` on per-period
    returns.

    Returns ``{sr_p5, sr_p50, sr_p95, n_valid_paths, n_sim, n, block,
    periods}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> rets = rng.normal(0.0005, 0.01, 1000)  # doctest: +SKIP
        >>> res = sharpe_distribution(  # doctest: +SKIP
        ...     rets, n_sim=1000, seed=0
        ... )
        >>> res["sr_p5"] < res["sr_p50"] < res["sr_p95"]  # doctest: +SKIP
        True
    """
    paths, n_obs = _resampled_return_matrix(returns, n_sim, block, seed)
    sd = paths.std(axis=1, ddof=1)
    mu = paths.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan) * float(
            np.sqrt(float(periods))
        )
    valid = sr[np.isfinite(sr)]
    p5, p50, p95 = _percentiles(valid)
    return {
        "sr_p5": p5,
        "sr_p50": p50,
        "sr_p95": p95,
        "n_valid_paths": int(valid.size),
        "n_sim": int(n_sim),
        "n": n_obs,
        "block": int(block),
        "periods": int(periods),
    }


def minimum_detectable_effect(
    n_obs: int,
    power: float = 0.80,
    alpha: float = 0.05,
    n_predictors: int = 1,
) -> float:
    """Smallest detectable partial R-squared at sample size ``n_obs``.

    Effect size is expressed as partial R-squared, consistent with the
    rest of the framework.  Delegates to
    ``power_calculator.detectable_effect`` (single new coefficient in a
    regression with ``n_predictors`` other regressors).

    Example::

        >>> minimum_detectable_effect(  # doctest: +SKIP
        ...     2000, n_predictors=25
        ... )
        0.0023...
    """
    if int(n_obs) < 2:
        raise ValueError(f"n must be >= 2, got {n_obs}")
    return float(
        detectable_effect(
            n_obs=int(n_obs),
            n_predictors=int(n_predictors),
            alpha=float(alpha),
            power=float(power),
        )
    )


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo all five tools on synthetic data."""
    print("=" * 72)
    print("monte_carlo demo 1: GBM paths vs analytic terminal mean")
    print("=" * 72)
    paths = monte_carlo_paths(
        100.0, 0.05, 0.20, n_paths=20000, n_steps=252, seed=0
    )
    term = paths[:, -1]
    analytic = 100.0 * np.exp(0.05)
    print(
        f"shape={paths.shape}  mean terminal={term.mean():.2f} "
        f"(analytic {analytic:.2f})  min terminal={term.min():.2f}"
    )

    print()
    print("=" * 72)
    print("monte_carlo demo 2: bootstrap distribution of the mean")
    print("=" * 72)
    rng = np.random.default_rng(1)
    sample = 0.02 * rng.standard_normal(500) + 0.005
    dist = bootstrap_resample(sample, np.mean, n_boot=2000, seed=2)
    print(
        f"iid    mean={dist.mean():+.5f} se={dist.std(ddof=1):.5f} "
        f"CI=[{np.quantile(dist, 0.025):+.5f}, "
        f"{np.quantile(dist, 0.975):+.5f}]"
    )
    ar = np.zeros(500)
    shocks = rng.standard_normal(500)
    for pos in range(1, 500):
        ar[pos] = 0.6 * ar[pos - 1] + shocks[pos]
    dist_b = bootstrap_resample(ar, np.mean, n_boot=2000, block=20, seed=3)
    print(
        f"block  AR(1) mean={ar.mean():+.4f} se={dist_b.std(ddof=1):.4f} "
        f"(block se >> iid se for autocorrelated data)"
    )

    print()
    print("=" * 72)
    print(
        "monte_carlo demo 3: max drawdown + Sharpe distributions (weak edge)"
    )
    print("=" * 72)
    strat = 0.0005 + 0.01 * rng.standard_normal(1000)
    dd = max_drawdown_distribution(strat, n_sim=2000, block=20, seed=4)
    sr = sharpe_distribution(strat, n_sim=2000, block=20, seed=5)
    print(
        f"mdd p5/p50/p95 = {dd['mdd_p5']:.1%} / "
        f"{dd['mdd_p50']:.1%} / {dd['mdd_p95']:.1%} "
        f"(valid paths {dd['n_valid_paths']}/{dd['n_sim']})"
    )
    print(
        f"SR  p5/p50/p95 = {sr['sr_p5']:.2f} / {sr['sr_p50']:.2f} / "
        f"{sr['sr_p95']:.2f}"
    )

    print()
    print("=" * 72)
    print(
        "monte_carlo demo 4: minimum detectable effect "
        "(partial R2, power=0.80)"
    )
    print("=" * 72)
    for n_obs in (500, 2000, 5000):
        mde = minimum_detectable_effect(
            n_obs, power=0.80, alpha=0.05, n_predictors=10
        )
        print(f"N={n_obs:5d}  min detectable partial R2 = {mde:.5f}")


if __name__ == "__main__":
    run_all_examples()
