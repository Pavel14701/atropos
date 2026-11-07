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

import math
from collections.abc import Callable
from typing import Any, Literal, TypedDict

import numpy as np

from atropos._utils import (
    apply_stat_rowwise,
    block_indices,
    clean_1d,
    validate_int,
    validate_number,
)

__all__ = [
    "InsufficientDataResult",
    "MovingBlockResult",
    "StationaryResult",
    "block_bootstrap_ci",
    "percentile_ci",
    "stationary_bootstrap_ci",
]


# Single-letter TypedDict KEYS (``n``, ``B``) are the historical public
# API of the result dicts (users index them with ``out["n"]``); the
# conventions AST test allows EXACTLY these two keys inside TypedDict
# class bodies -- any other single-letter field stays banned, as do
# single-letter variables everywhere.
class InsufficientDataResult(TypedDict):
    """Scaffold returned when fewer than 8 finite observations remain."""

    point_estimate: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    B: int
    method: Literal["insufficient_data"]


class MovingBlockResult(TypedDict):
    """Result of the circular moving-block bootstrap."""

    point_estimate: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    B: int
    block: int
    method: Literal["moving_block"]


class StationaryResult(TypedDict):
    """Result of the Politis-Romano stationary bootstrap."""

    point_estimate: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    B: int
    mean_block: float
    method: Literal["stationary"]


def _validate_ci_level(ci_level: float) -> None:
    """Raise ValueError unless ``ci_level`` lies strictly inside (0, 1).

    Non-numeric input (``object()``, strings) is re-raised as
    ``ValueError`` naming the parameter, per the shared
    ``validate_number`` contract.
    """
    level = validate_number("ci_level", ci_level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")


def _base_result(
    data: np.ndarray, stat: Callable[[np.ndarray], float]
) -> InsufficientDataResult:
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


def _stationary_indices(
    n_obs: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Circular index row of length ``n_obs`` for one stationary replicate.

    Draws geometric block lengths with mean ``mean_block`` until ``n_obs``
    indices are covered; blocks past the end wrap around. The ``while``
    loop is unavoidable because the number of blocks per replicate is
    itself random. The per-replicate RNG draw order (one
    ``rng.integers`` then one ``rng.geometric`` per block) is frozen: it
    must stay identical to the historical inline loop or fixed-seed
    results would change.
    """
    renew_prob = 1.0 / max(float(mean_block), 1.0)
    pieces: list[np.ndarray] = []
    total = 0
    while total < n_obs:
        start = int(rng.integers(0, n_obs))
        length = int(rng.geometric(renew_prob))
        pieces.append((start + np.arange(length)) % n_obs)
        total += length
    return np.concatenate(pieces)[:n_obs]


def _boot_se(boot: np.ndarray, n_boot: int) -> float:
    """Bootstrap SE, recomputed on a scaled copy when squares overflow.

    ``np.std`` squares the deviations, so a bootstrap distribution of
    ~1e305-magnitude statistics overflows to inf even though its true
    spread (~1e305) is representable; dividing by the max magnitude
    first keeps the arithmetic in range.  The normal path is the
    untouched one-step ``np.std`` (bit-compatible with history).
    """
    if int(n_boot) <= 1:
        return float("nan")
    se = float(np.std(boot, ddof=1))
    if math.isinf(se) and bool(np.all(np.isfinite(boot))):
        scale = float(np.max(np.abs(boot)))
        se = scale * float(np.std(boot / scale, ddof=1))
    return se


def percentile_ci(
    boot_stats: Any, ci_level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of a bootstrap statistic distribution.

    Example::

        >>> percentile_ci(np.arange(1000) / 1000)
        (0.024975000000000025, 0.9740249999999999)

    Args:
        boot_stats: 1-D array-like of bootstrap statistics (non-finite
            entries dropped).
        ci_level: confidence level, strictly inside (0, 1).

    Returns:
        Tuple ``(ci_low, ci_high)``; ``(nan, nan)`` when nothing finite
        remains.

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1); raised before the
            data is touched.
    """
    _validate_ci_level(ci_level)
    stats_arr = clean_1d(boot_stats)
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
) -> MovingBlockResult | InsufficientDataResult:
    """Circular moving-block bootstrap CI for ``stat(series)``.

    Args:
        series: 1-D array-like (NaNs dropped).
        n_boot: number of bootstrap replicates.
        block: block length; resampling with wrap-around preserves
            short-range autocorrelation and the variance of overlapping
            sums.
        stat: scalar statistic (must accept ``axis=1`` for the
            vectorized path; otherwise a per-row loop is used).
        ci_level: confidence level, strictly inside (0, 1).
        seed: seed of the bootstrap RNG.

    Returns:
        Dict with keys ``point_estimate, ci_low, ci_high, se, n, B,
        block, method`` (a :class:`MovingBlockResult`), or the
        ``method = "insufficient_data"`` scaffold
        (:class:`InsufficientDataResult`) when fewer than 8 finite
        observations remain.

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1) or not a finite
            number, or ``n_boot`` / ``block`` is not an integer >= 1;
            raised before the data is touched.

    Notes:
        Memory is ``O(n_boot * (n + block))`` (an int64 index matrix);
        for very large series and ``n_boot`` reduce ``n_boot`` or chunk
        the call.

    Example::

        >>> rng = np.random.default_rng(0)              # doctest: +SKIP
        >>> data = rng.standard_normal(500)             # doctest: +SKIP
        >>> ci = block_bootstrap_ci(                    # doctest: +SKIP
        ...     data, n_boot=500, block=10, seed=0)
        >>> ci["ci_low"] < 0 < ci["ci_high"]            # doctest: +SKIP
        True
    """
    _validate_ci_level(ci_level)
    n_boot = validate_int("n_boot", n_boot)
    block = validate_int("block", block)
    data = clean_1d(series)
    n_obs = int(data.size)
    if n_obs < 8:
        return _base_result(data, stat)
    block = int(min(block, n_obs))
    rng = np.random.default_rng(seed)
    idx = block_indices(n_obs, int(n_boot), block, rng)
    boot = apply_stat_rowwise(data[idx], stat)
    lo, hi = percentile_ci(boot, ci_level)
    se = _boot_se(boot, n_boot)
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
) -> StationaryResult | InsufficientDataResult:
    """Politis-Romano stationary bootstrap CI (geometric block lengths).

    Preferred over the moving-block bootstrap for heavy-tailed series or
    when the dependence length is unknown: random geometric block lengths
    (mean ``mean_block``) make the resampled series stationary by
    construction.

    Returns dict with keys ``point_estimate, ci_low, ci_high, se, n,
    B, mean_block, method`` (a :class:`StationaryResult`), or the
    ``method = "insufficient_data"`` scaffold
    (:class:`InsufficientDataResult`) when fewer than 8 finite
    observations remain.

    Example::

        >>> rng = np.random.default_rng(0)             # doctest: +SKIP
        >>> data = rng.standard_t(3, size=400)         # doctest: +SKIP
        >>> ci = stationary_bootstrap_ci(               # doctest: +SKIP
        ...     data, n_boot=300, mean_block=15, seed=0)
        >>> ci["ci_low"] < 0 < ci["ci_high"]           # doctest: +SKIP
        True

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1) or not a finite
            number, ``n_boot`` is not an integer >= 1, or ``mean_block``
            is not a finite number >= 1 (a sub-unity mean block length
            degenerates to the iid bootstrap and NaN reached
            ``rng.geometric`` as a raw numpy error); raised before the
            data is touched.
    """
    _validate_ci_level(ci_level)
    n_boot = validate_int("n_boot", n_boot)
    mean_block_val = validate_number("mean_block", mean_block)
    if mean_block_val < 1.0:
        raise ValueError(
            "mean_block must be >= 1 (a sub-unity mean block length "
            f"degenerates to the iid bootstrap), got {mean_block}"
        )
    data = clean_1d(series)
    n_obs = int(data.size)
    if n_obs < 8:
        return _base_result(data, stat)
    rng = np.random.default_rng(seed)
    idx_matrix = np.vstack(
        [
            _stationary_indices(n_obs, mean_block_val, rng)
            for _ in range(int(n_boot))
        ]
    )
    boot = apply_stat_rowwise(data[idx_matrix], stat)
    lo, hi = percentile_ci(boot, ci_level)
    se = _boot_se(boot, n_boot)
    return {
        "point_estimate": float(stat(data)),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n": n_obs,
        "B": int(n_boot),
        "mean_block": mean_block_val,
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
    # Layer-0 verification: the demo doubles as a regression test.
    # Structural facts plus estimator properties that hold for ANY
    # input; the exact seeded numbers are pinned in
    # tests/test_bootstrap_ci.py::test_seeded_regression_*
    assert ci_mb["method"] == "moving_block"
    assert ci_mb["n"] == 500 and ci_mb["B"] == 1000
    assert ci_mb["block"] == 20
    point_mb = float(ci_mb["point_estimate"])
    ci_mb_low = float(ci_mb["ci_low"])
    ci_mb_high = float(ci_mb["ci_high"])
    se_mb = float(ci_mb["se"])
    # properties of the estimator, not of this seed's draw: every
    # reported number is finite, se is strictly positive, and the
    # point estimate IS exactly the sample mean of the input series
    assert (
        np.isfinite(point_mb)
        and np.isfinite(ci_mb_low)
        and np.isfinite(ci_mb_high)
        and np.isfinite(se_mb)
        and se_mb > 0.0
    )
    assert point_mb == float(np.mean(ar))
    # seeded regression; percentile CI containment of the point
    # estimate is not a theorem, it holds at this fixed seed
    assert ci_mb_low < point_mb < ci_mb_high
    # percentile CI of the mean is roughly symmetric around the point
    # estimate, and each half-width tracks the ~1.96 * se scale
    assert abs((point_mb - ci_mb_low) - (ci_mb_high - point_mb)) <= 0.5 * (
        ci_mb_high - ci_mb_low
    )
    # the 95% normal-approximation half-width is 1.96 * se, but a
    # bootstrap percentile interval of a skewed/skewed-tailed statistic
    # can sit wider than the normal band, hence the 3.5 * se ceiling
    assert se_mb <= ci_mb_high - point_mb <= 3.5 * se_mb
    assert se_mb <= point_mb - ci_mb_low <= 3.5 * se_mb

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
    assert ci_st["method"] == "stationary"
    assert ci_st["n"] == 400 and ci_st["B"] == 300
    assert ci_st["mean_block"] == 12.0
    point_st = float(ci_st["point_estimate"])
    # property, not seed statistics: the point estimate IS exactly the
    # sample median of the input series; exact seeded CI numbers are
    # pinned in tests/test_bootstrap_ci.py::test_seeded_regression_*
    assert (
        np.isfinite(point_st)
        and np.isfinite(float(ci_st["ci_low"]))
        and np.isfinite(float(ci_st["ci_high"]))
        and np.isfinite(float(ci_st["se"]))
        and float(ci_st["se"]) > 0.0
    )
    assert point_st == float(np.median(heavy))
    # seeded regression; percentile CI containment of the point
    # estimate is not a theorem, it holds at this fixed seed
    assert float(ci_st["ci_low"]) < point_st < float(ci_st["ci_high"])

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
    # the small-n guard refuses to bootstrap instead of producing garbage
    assert tiny["method"] == "insufficient_data"
    assert tiny["n"] == 2 and tiny["B"] == 0
    assert tiny["point_estimate"] == 1.5  # exact mean of the finite pair
    assert bool(np.isnan(tiny["ci_low"]))
    assert bool(np.isnan(tiny["ci_high"]))
    assert bool(np.isnan(tiny["se"]))


if __name__ == "__main__":
    run_all_examples()
