"""Block bootstrap confidence intervals for dependent data.

Two estimators for autocorrelated / heavy-tailed series:

* :func:`block_bootstrap_ci` - circular moving-block bootstrap (Kunsch,
  1989).  Handles overlapping returns and autocorrelation by resampling
  contiguous blocks of length ``block`` with wrap-around.
* :func:`stationary_bootstrap_ci` - Politis & Romano (1994) stationary
  bootstrap with geometric block lengths (mean ``mean_block``).  Prefer
  this for heavy-tailed series or when the dependence length is unknown.

Small-sample handling: with fewer than 8 observations the functions
return NaN bounds with ``method = "insufficient_data"`` rather than
silently producing garbage.

Usage::

    from atropos.bootstrap_ci import block_bootstrap_ci

    out = block_bootstrap_ci(returns, n_boot=1000, block=20, stat=np.mean)
    print(out["ci_low"], out["ci_high"])

Run ``python -m atropos.bootstrap_ci`` for a synthetic demo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["block_bootstrap_ci", "percentile_ci", "stationary_bootstrap_ci"]


def _clean(values: Any) -> np.ndarray:
    """Coerce array-like input to a flat 1-D float array, NaNs dropped."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    # np.asarray pins the ndarray type: stubs resolve this to Any.
    return np.asarray(arr[np.isfinite(arr)])


def _apply_stat(samples: np.ndarray, stat: Callable[..., Any]) -> np.ndarray:
    """Apply ``stat`` row-wise; vectorized ``axis=1`` path with a loop."""
    try:
        out = np.asarray(stat(samples, axis=1), dtype=float)
        if out.ndim == 1 and out.shape[0] == samples.shape[0]:
            return out
    except Exception:
        pass
    return np.asarray([float(stat(row)) for row in samples], dtype=float)


def _base_result(
    data: np.ndarray, stat: Callable[..., Any]
) -> dict[str, Any]:
    """Insufficient-data scaffold returned by both bootstrap variants."""
    n_obs = int(data.size)
    point = float(stat(data)) if n_obs > 0 else float("nan")
    return {
        "point_estimate": point,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "se": float("nan"),
        "n": n_obs,
        "B": 0,
        "method": "insufficient_data",
    }


def percentile_ci(
    boot_stats: Any, ci_level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of a bootstrap statistic distribution.

    Example::

        >>> percentile_ci(np.arange(1000) / 1000)   # doctest: +SKIP
        (0.025, 0.975)
    """
    stats_arr = _clean(boot_stats)
    if stats_arr.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(ci_level)) / 2.0
    return (
        float(np.quantile(stats_arr, tail)),
        float(np.quantile(stats_arr, 1.0 - tail)),
    )


def block_bootstrap_ci(
    series: Any,
    *,
    n_boot: int = 1000,
    block: int = 20,
    stat: Callable[..., Any] = np.mean,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Circular moving-block bootstrap CI for ``stat(series)``.

    Parameters
    ----------
    series : 1-D array-like (NaNs dropped)
    n_boot : number of bootstrap replicates
    block : block length; resampling with wrap-around preserves
        short-range autocorrelation and the variance of overlapping sums
    stat : scalar statistic (must accept ``axis=1`` for the vectorized
        path; otherwise a per-row loop is used)

    Returns dict with keys ``point_estimate, ci_low, ci_high, se, n,
    B, block, method``.

    Example::

        >>> rng = np.random.default_rng(0)              # doctest: +SKIP
        >>> data = rng.standard_normal(500)             # doctest: +SKIP
        >>> ci = block_bootstrap_ci(                    # doctest: +SKIP
        ...     data, n_boot=500, block=10, seed=0)
        >>> ci["ci_low"] < 0 < ci["ci_high"]            # doctest: +SKIP
        True

    Notes:
    -----
    Memory is ``O(n_boot * (n + block))`` (an int64 index matrix); for
    very large series and ``n_boot`` reduce ``n_boot`` or chunk the call.
    """
    data = _clean(series)
    n_obs = int(data.size)
    if n_obs < 8:
        return _base_result(data, stat)
    block = int(min(max(int(block), 1), n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs, size=(int(n_boot), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    idx = idx.reshape(int(n_boot), n_blocks * block)[:, :n_obs]
    boot = _apply_stat(data[idx], stat)
    lo, hi = percentile_ci(boot, ci_level)
    se = float(np.std(boot, ddof=1)) if int(n_boot) > 1 else float("nan")
    return {
        "point_estimate": float(stat(data)),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n": n_obs,
        "B": int(n_boot),
        "block": block,
        "method": "moving_block",
    }


def stationary_bootstrap_ci(
    series: Any,
    *,
    n_boot: int = 1000,
    mean_block: float = 20.0,
    stat: Callable[..., Any] = np.mean,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Politis-Romano stationary bootstrap CI (geometric block lengths).

    Preferred over the moving-block bootstrap for heavy-tailed series or
    when the dependence length is unknown: random geometric block lengths
    (mean ``mean_block``) make the resampled series stationary by
    construction.

    Returns dict with keys ``point_estimate, ci_low, ci_high, se, n,
    B, mean_block, method``.

    Example::

        >>> rng = np.random.default_rng(0)             # doctest: +SKIP
        >>> data = rng.standard_t(3, size=400)         # doctest: +SKIP
        >>> ci = stationary_bootstrap_ci(               # doctest: +SKIP
        ...     data, n_boot=300, mean_block=15, seed=0)
        >>> ci["ci_low"] < 0 < ci["ci_high"]           # doctest: +SKIP
        True
    """
    data = _clean(series)
    n_obs = int(data.size)
    if n_obs < 8:
        return _base_result(data, stat)
    renew_prob = 1.0 / max(float(mean_block), 1.0)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for rep in range(int(n_boot)):
        pieces: list[np.ndarray] = []
        total = 0
        while total < n_obs:
            start = int(rng.integers(0, n_obs))
            length = int(rng.geometric(renew_prob))
            pieces.append((start + np.arange(length)) % n_obs)
            total += length
        idx = np.concatenate(pieces)[:n_obs]
        boot[rep] = float(stat(data[idx]))
    lo, hi = percentile_ci(boot, ci_level)
    se = float(np.std(boot, ddof=1)) if int(n_boot) > 1 else float("nan")
    return {
        "point_estimate": float(stat(data)),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n": n_obs,
        "B": int(n_boot),
        "mean_block": float(mean_block),
        "method": "stationary",
    }


def run_all_examples() -> None:
    """Demo both bootstraps on autocorrelated / heavy-tailed data."""
    rng = np.random.default_rng(7)

    print("=" * 72)
    print("bootstrap_ci demo 1: autocorrelated returns (moving block)")
    print("=" * 72)
    shocks = rng.standard_normal(500)
    ar = np.zeros(500)
    for step in range(1, 500):
        ar[step] = 0.6 * ar[step - 1] + shocks[step]
    ci_mb = block_bootstrap_ci(
        ar, n_boot=1000, block=20, stat=np.mean, seed=1
    )
    print(
        f"point={ci_mb['point_estimate']:+.4f} "
        f"CI=[{ci_mb['ci_low']:+.4f}, {ci_mb['ci_high']:+.4f}] "
        f"se={ci_mb['se']:.4f} method={ci_mb['method']}"
    )

    print()
    print("=" * 72)
    print(
        "bootstrap_ci demo 2: heavy-tailed series "
        "(stationary bootstrap, median)"
    )
    print("=" * 72)
    heavy = rng.standard_t(3, size=400) * 1.5
    ci_st = stationary_bootstrap_ci(
        heavy, n_boot=300, mean_block=12, stat=np.median, seed=2
    )
    print(
        f"point={ci_st['point_estimate']:+.4f} "
        f"CI=[{ci_st['ci_low']:+.4f}, {ci_st['ci_high']:+.4f}] "
        f"se={ci_st['se']:.4f} method={ci_st['method']}"
    )

    print()
    print("=" * 72)
    print("bootstrap_ci demo 3: small-n guard")
    print("=" * 72)
    tiny = block_bootstrap_ci(
        [1.0, 2.0, float("nan")], n_boot=100, block=5, seed=3
    )
    print(
        f"n=2 valid points -> method={tiny['method']} ci_low={tiny['ci_low']}"
    )


if __name__ == "__main__":
    run_all_examples()
