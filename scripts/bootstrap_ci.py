"""Block bootstrap confidence intervals for autocorrelated / heavy-tailed series.

Two estimators:

* :func:`block_bootstrap_ci` - circular moving-block bootstrap (Kunsch, 1989).
  Handles overlapping returns and autocorrelation by resampling contiguous
  blocks of length ``block`` with wrap-around.
* :func:`stationary_bootstrap_ci` - Politis & Romano (1994) stationary
  bootstrap with geometric block lengths (mean ``mean_block``).  Prefer this
  for heavy-tailed series or when the dependence length is unknown.

Small-n handling: if ``n < 8`` the functions return NaN bounds with
``method = "insufficient_data"`` rather than silently producing garbage.

Usage::

    from bootstrap_ci import block_bootstrap_ci, stationary_bootstrap_ci

    out = block_bootstrap_ci(returns, B=1000, block=20, stat=np.mean)
    print(out["ci_low"], out["ci_high"])

Run ``python bootstrap_ci.py`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["block_bootstrap_ci", "stationary_bootstrap_ci", "percentile_ci"]


def _clean(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _apply_stat(samples: np.ndarray, stat: Callable) -> np.ndarray:
    """Apply ``stat`` row-wise; vectorized ``axis=1`` path with a loop fallback."""
    try:
        out = np.asarray(stat(samples, axis=1), dtype=float)
        if out.ndim == 1 and out.shape[0] == samples.shape[0]:
            return out
    except Exception:
        pass
    return np.asarray([float(stat(row)) for row in samples], dtype=float)


def _base_result(x: np.ndarray, stat: Callable) -> dict[str, Any]:
    n = int(x.size)
    point = float(stat(x)) if n > 0 else float("nan")
    return {
        "point_estimate": point,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "se": float("nan"),
        "n": n,
        "B": 0,
        "method": "insufficient_data",
    }


def percentile_ci(boot_stats: Any, ci_level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap statistic distribution.

    Example::

        >>> percentile_ci(np.arange(1000) / 1000)   # doctest: +SKIP
        (0.025, 0.975)
    """
    s = _clean(boot_stats)
    if s.size == 0:
        return float("nan"), float("nan")
    a = (1.0 - float(ci_level)) / 2.0
    return float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))


def block_bootstrap_ci(
    series: Any,
    B: int = 1000,
    block: int = 20,
    stat: Callable = np.mean,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Circular moving-block bootstrap CI for ``stat(series)``.

    Parameters
    ----------
    series : 1-D array-like (NaNs dropped)
    B : number of bootstrap replicates
    block : block length; resampling with wrap-around preserves short-range
        autocorrelation and the variance of overlapping sums
    stat : scalar statistic (must accept ``axis=1`` for the vectorized path;
        otherwise a per-row loop is used)

    Returns dict with keys ``point_estimate, ci_low, ci_high, se, n, B, block,
    method``.

    Example::

        >>> rng = np.random.default_rng(0)                    # doctest: +SKIP
        >>> x = rng.standard_normal(500)                      # doctest: +SKIP
        >>> ci = block_bootstrap_ci(x, B=500, block=10, seed=0)   # doctest: +SKIP
        >>> ci["ci_low"] < 0 < ci["ci_high"]                  # doctest: +SKIP
        True

    Notes
    -----
    Memory is ``O(B * (n + block))`` (an int64 index matrix); for very large
    ``n`` and ``B`` reduce ``B`` or chunk the call.
    """
    x = _clean(series)
    n = int(x.size)
    if n < 8:
        return _base_result(x, stat)
    block = int(min(max(int(block), 1), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(int(B), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    idx = idx.reshape(int(B), n_blocks * block)[:, :n]
    boot = _apply_stat(x[idx], stat)
    lo, hi = percentile_ci(boot, ci_level)
    se = float(np.std(boot, ddof=1)) if int(B) > 1 else float("nan")
    return {
        "point_estimate": float(stat(x)),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n": n,
        "B": int(B),
        "block": block,
        "method": "moving_block",
    }


def stationary_bootstrap_ci(
    series: Any,
    B: int = 1000,
    mean_block: float = 20.0,
    stat: Callable = np.mean,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Politis-Romano stationary bootstrap CI (geometric block lengths).

    Preferred over the moving-block bootstrap for heavy-tailed series or when
    the dependence length is unknown: random geometric block lengths (mean
    ``mean_block``) make the resampled series stationary by construction.

    Returns dict with keys ``point_estimate, ci_low, ci_high, se, n, B,
    mean_block, method``.

    Example::

        >>> rng = np.random.default_rng(0)                          # doctest: +SKIP
        >>> x = rng.standard_t(3, size=400)   # heavy tails          # doctest: +SKIP
        >>> ci = stationary_bootstrap_ci(x, B=300, mean_block=15, seed=0)  # doctest: +SKIP
        >>> ci["ci_low"] < 0 < ci["ci_high"]                        # doctest: +SKIP
        True
    """
    x = _clean(series)
    n = int(x.size)
    if n < 8:
        return _base_result(x, stat)
    p = 1.0 / max(float(mean_block), 1.0)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        pieces: list[np.ndarray] = []
        total = 0
        while total < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p))
            pieces.append((start + np.arange(length)) % n)
            total += length
        idx = np.concatenate(pieces)[:n]
        boot[b] = float(stat(x[idx]))
    lo, hi = percentile_ci(boot, ci_level)
    se = float(np.std(boot, ddof=1)) if int(B) > 1 else float("nan")
    return {
        "point_estimate": float(stat(x)),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n": n,
        "B": int(B),
        "mean_block": float(mean_block),
        "method": "stationary",
    }


def run_all_examples() -> None:
    """Demo both bootstraps on autocorrelated and heavy-tailed synthetic data."""
    rng = np.random.default_rng(7)

    print("=" * 72)
    print("bootstrap_ci demo 1: autocorrelated returns (moving block)")
    print("=" * 72)
    e = rng.standard_normal(500)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    ci_mb = block_bootstrap_ci(ar, B=1000, block=20, stat=np.mean, seed=1)
    print(f"point={ci_mb['point_estimate']:+.4f} "
          f"CI=[{ci_mb['ci_low']:+.4f}, {ci_mb['ci_high']:+.4f}] "
          f"se={ci_mb['se']:.4f} method={ci_mb['method']}")

    print()
    print("=" * 72)
    print("bootstrap_ci demo 2: heavy-tailed series (stationary bootstrap, median)")
    print("=" * 72)
    heavy = rng.standard_t(3, size=400) * 1.5
    ci_st = stationary_bootstrap_ci(heavy, B=300, mean_block=12, stat=np.median, seed=2)
    print(f"point={ci_st['point_estimate']:+.4f} "
          f"CI=[{ci_st['ci_low']:+.4f}, {ci_st['ci_high']:+.4f}] "
          f"se={ci_st['se']:.4f} method={ci_st['method']}")

    print()
    print("=" * 72)
    print("bootstrap_ci demo 3: small-n guard")
    print("=" * 72)
    tiny = block_bootstrap_ci([1.0, 2.0, float("nan")], B=100, block=5, seed=3)
    print(f"n=2 valid points -> method={tiny['method']} ci_low={tiny['ci_low']}")


if __name__ == "__main__":
    run_all_examples()
