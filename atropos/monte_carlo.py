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

from atropos._utils import (
    apply_stat_rowwise,
    block_indices,
    sane_variance_mask,
    validate_int,
    validate_number,
)
from atropos.power_calculator import detectable_effect, power_for_effect

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
    n_paths, n_steps : simulation dimensions (integers >= 1;
        non-integers such as ``2.5`` are rejected, not truncated)
    dt : time step in years (must be > 0; default one trading day)
    seed : optional RNG seed for reproducibility

    Example::

        >>> paths = monte_carlo_paths(  # doctest: +SKIP
        ...     100, 0.05, 0.2, 1000, 252, seed=0
        ... )
        >>> paths.shape  # doctest: +SKIP
        (1000, 252)
    """
    s0f = validate_number("s0", s0)
    muf = float(mu)
    sig = validate_number("sigma", sigma)
    if s0f <= 0:
        raise ValueError(f"s0 must be > 0, got {s0f}")
    if sig < 0:
        raise ValueError(f"sigma must be >= 0, got {sig}")
    if int(n_paths) < 1 or int(n_steps) < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    n_p = validate_int("n_paths", n_paths)
    n_s = validate_int("n_steps", n_steps)
    dtf = validate_number("dt", dt)
    if dtf <= 0:
        raise ValueError(f"dt must be > 0, got {dtf}")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_p, n_s))
    drift = (muf - 0.5 * sig * sig) * dtf
    vol = sig * np.sqrt(dtf)
    increments = drift + vol * shocks
    # annotation pins the ndarray type: np.exp's stub return is Any.
    paths: np.ndarray = s0f * np.exp(np.cumsum(increments, axis=1))
    return paths


# ----------------------------------------------------------------------------
# resampling machinery
# ----------------------------------------------------------------------------
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
    n_boot : number of bootstrap replicates (integer >= 1; floats such
        as ``2.5`` are rejected, not truncated)
    block : ``None`` -> iid bootstrap; integer >= 1 -> circular moving-block
        resampling that preserves autocorrelation up to lag ~``block``.

    Any ``n_obs >= 1`` is accepted, however small: the bootstrap of an
    arbitrary statistic is well-defined even on a handful of points, so
    judging the sample size is left to the caller.  The distribution
    helpers below deliberately impose a 20-observation floor instead,
    because their risk percentiles are noise on shorter series.

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
    if not callable(statistic):
        raise TypeError(
            f"statistic must be callable, got {type(statistic).__name__}"
        )
    nb = validate_int("n_boot", n_boot)
    block_size = validate_int("block", block) if block is not None else None
    values = np.asarray(data, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n_obs = int(values.size)
    if n_obs == 0:
        return np.full(nb, np.nan)
    rng = np.random.default_rng(seed)
    if block_size is None:
        idx = rng.integers(0, n_obs, size=(nb, n_obs))
    else:
        idx = block_indices(n_obs, nb, block_size, rng)
    return apply_stat_rowwise(values[idx], statistic)


def _resampled_return_matrix(
    returns: Any, n_sim: int, block: int, seed: int | None
) -> tuple[np.ndarray, int]:
    """Block-resample cleaned returns into an ``(n_sim, n_obs)`` matrix.

    Requires at least 20 valid returns -- the same floor the percentile
    helper below trusts -- because drawdown and Sharpe estimates resampled
    from shorter series are noise.  (:func:`bootstrap_resample` accepts any
    ``n_obs >= 1`` by design; the two thresholds are intentionally
    different.)
    """
    rets = np.asarray(returns, dtype=float).reshape(-1)
    rets = rets[np.isfinite(rets)]
    n_obs = int(rets.size)
    if n_obs < 20:
        raise ValueError(f"need at least 20 valid returns, got {n_obs}")
    ns = int(n_sim)
    if ns < 1:
        raise ValueError(f"n_sim must be >= 1, got {ns}")
    rng = np.random.default_rng(seed)
    return rets[block_indices(n_obs, ns, int(block), rng)], n_obs


# Minimum number of valid paths for tail percentiles to carry information;
# below this floor _percentiles reports NaN and the distributions flag it.
_MIN_PATHS_FOR_PERCENTILES = 20


def _percentiles(valid: np.ndarray) -> tuple[float, float, float]:
    """p5 / p50 / p95 of ``valid``, or NaNs below the trust floor.

    With fewer than ``_MIN_PATHS_FOR_PERCENTILES`` (20) observations the
    5%/95% tail quantiles are pure noise, so NaN is reported instead of a
    number that looks authoritative.
    """
    if valid.size < _MIN_PATHS_FOR_PERCENTILES:
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
    paths, most negative) and ``mdd_p95`` the benign tail.  When fewer
    than 20 paths survive, the percentiles are NaN and ``note`` carries
    ``"insufficient paths for percentiles"``.

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
    n_sim = validate_int("n_sim", n_sim)
    block = validate_int("block", block)
    paths, n_obs = _resampled_return_matrix(returns, n_sim, block, seed)
    eq = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / peak - 1.0
    mdd = dd.min(axis=1)
    # A resampled return <= -1 drives the compounded equity non-positive at
    # some step; "drawdown" is then meaningless (sign flips can even look
    # like gains), so such paths are dropped rather than reported.
    nonsense = (eq <= 0).any(axis=1) | ~np.isfinite(mdd)
    valid = mdd[~nonsense]
    p5, p50, p95 = _percentiles(valid)
    note = "mdd_p5 = adverse tail (worst 5% of paths); mdd_p95 = benign tail"
    if valid.size < _MIN_PATHS_FOR_PERCENTILES:
        note += "; insufficient paths for percentiles"
    return {
        "mdd_p5": p5,
        "mdd_p50": p50,
        "mdd_p95": p95,
        "n_valid_paths": int(valid.size),
        "n_dropped_paths": int(nonsense.sum()),
        "n_sim": int(n_sim),
        "n": n_obs,
        "block": int(block),
        "note": note,
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
    returns.  Replicates with a non-finite Sharpe (zero variance, NaNs)
    are dropped and counted in ``n_dropped_paths``.

    Returns ``{sr_p5, sr_p50, sr_p95, n_valid_paths, n_dropped_paths,
    n_sim, n, block, periods}``, plus a ``note`` key when fewer than 20
    paths survive: ``"insufficient paths for percentiles"`` (percentiles
    are then NaN).

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> rets = rng.normal(0.0005, 0.01, 1000)  # doctest: +SKIP
        >>> res = sharpe_distribution(  # doctest: +SKIP
        ...     rets, n_sim=1000, seed=0
        ... )
        >>> res["sr_p5"] < res["sr_p50"] < res["sr_p95"]  # doctest: +SKIP
        True
    """
    periods = validate_int("periods", periods)
    n_sim = validate_int("n_sim", n_sim)
    block = validate_int("block", block)
    paths, n_obs = _resampled_return_matrix(returns, n_sim, block, seed)
    sd = paths.std(axis=1, ddof=1)
    mu = paths.mean(axis=1)
    usable = sane_variance_mask(mu, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(usable, mu / sd, np.nan) * float(
            np.sqrt(float(periods))
        )
    valid = sr[np.isfinite(sr)]
    p5, p50, p95 = _percentiles(valid)
    result: dict[str, Any] = {
        "sr_p5": p5,
        "sr_p50": p50,
        "sr_p95": p95,
        "n_valid_paths": int(valid.size),
        "n_dropped_paths": int(sr.size - valid.size),
        "n_sim": int(n_sim),
        "n": n_obs,
        "block": int(block),
        "periods": int(periods),
    }
    if valid.size < _MIN_PATHS_FOR_PERCENTILES:
        result["note"] = "insufficient paths for percentiles"
    return result


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
        0.0039...
    """
    n_obs = validate_int("n", n_obs, minimum=2)
    n_pred = validate_int("n_predictors", n_predictors, minimum=0)
    return float(
        detectable_effect(
            n_obs=n_obs,
            n_predictors=n_pred,
            alpha=float(alpha),
            power=float(power),
        )
    )


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _demo_gbm_paths() -> None:
    """Demo 1: GBM paths reproduce the analytic terminal mean."""
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
    assert abs(term.mean() - analytic) < 0.02 * analytic
    assert term.min() > 0.0  # log-form GBM keeps prices strictly positive


def _demo_bootstrap_mean(rng: np.random.Generator) -> None:
    """Demo 2: iid vs block bootstrap of the mean on AR(1) data."""
    print()
    print("=" * 72)
    print("monte_carlo demo 2: bootstrap distribution of the mean")
    print("=" * 72)
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
    ci_low, ci_high = (
        np.quantile(dist, 0.025),
        np.quantile(dist, 0.975),
    )
    assert ci_low < sample.mean() < ci_high
    assert dist_b.std(ddof=1) > dist.std(ddof=1)


def _demo_distributions(rng: np.random.Generator) -> None:
    """Demo 3: drawdown/Sharpe distributions + fat-tail path drop."""
    print()
    print("=" * 72)
    print(
        "monte_carlo demo 3: max drawdown + Sharpe distributions "
        "(constant drift)"
    )
    print("=" * 72)
    # 5bp/day drift with 1% noise: a constant-drift synthetic series, NOT
    # a demonstrated edge -- the demo only exercises the machinery.
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
    print("-" * 72)
    print("fat-left-tail stress: non-positive equity paths are dropped")
    fat = 0.002 + 0.008 * rng.standard_normal(400)
    fat[80] = -1.5  # two catastrophic days: a -150% "return" cannot be
    fat[240] = -1.5  # compounded into a meaningful drawdown
    dd_tail = max_drawdown_distribution(fat, n_sim=2000, block=20, seed=6)
    total = dd_tail["n_valid_paths"] + dd_tail["n_dropped_paths"]
    print(
        f"mdd p5 = {dd_tail['mdd_p5']:.1%}  "
        f"(dropped {dd_tail['n_dropped_paths']}/{total} "
        f"non-positive-equity paths, p5 kept above -100%)"
    )
    assert -1.0 <= dd["mdd_p5"] <= dd["mdd_p50"] <= dd["mdd_p95"] <= 0.0
    assert dd["n_valid_paths"] + dd["n_dropped_paths"] == dd["n_sim"]
    assert sr["sr_p5"] <= sr["sr_p50"] <= sr["sr_p95"]
    assert dd_tail["n_dropped_paths"] >= 1  # the -150% days are dropped
    assert total == dd_tail["n_sim"]
    assert dd_tail["mdd_p5"] >= -1.0  # drawdown can never breach -100%


def _demo_detectable_effect() -> None:
    """Demo 4: minimum detectable partial R-squared shrinks with N."""
    print()
    print("=" * 72)
    print(
        "monte_carlo demo 4: minimum detectable effect "
        "(partial R2, power=0.80)"
    )
    print("=" * 72)
    effects = []
    for n_obs in (500, 2000, 5000):
        mde = minimum_detectable_effect(
            n_obs, power=0.80, alpha=0.05, n_predictors=10
        )
        effects.append(mde)
        print(f"N={n_obs:5d}  min detectable partial R2 = {mde:.5f}")
    assert all(0.0 < value < 1.0 for value in effects)
    assert effects[0] > effects[1] > effects[2]  # more N, smaller effect
    # pinned N=2000 / k=10 value: the bisection is deterministic and
    # seed-free, so any drift beyond 1e-4 is a real regression
    assert abs(effects[1] - 0.0039129) <= 1e-4
    # round-trip consistency with power_calculator: the effect that is
    # just detectable at N=2000 must require ~N=2000 observations again
    n_back = power_for_effect(
        effects[1], n_predictors=10, alpha=0.05, power=0.80
    )
    print(
        f"round trip: power_for_effect({effects[1]:.5f}, k=10) "
        f"-> N = {n_back}"
    )
    assert 1900 <= n_back <= 2100


def run_all_examples() -> None:
    """Demo all five tools on synthetic data (asserts on every case)."""
    demo_rng = np.random.default_rng(1)
    _demo_gbm_paths()
    _demo_bootstrap_mean(demo_rng)
    _demo_distributions(demo_rng)
    _demo_detectable_effect()


if __name__ == "__main__":
    run_all_examples()
