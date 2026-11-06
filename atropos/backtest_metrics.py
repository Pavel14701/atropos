"""Performance metrics on per-trade R sequences.

The ranker-head / walk-forward view of performance: every input is net
R per trade, and chronological order matters where documented.

Why this lives separately from :mod:`atropos.evaluation_metrics`: those
metrics assume a PER-PERIOD return stream (a daily series with a fixed
calendar grid, annualized by ``sqrt(periods)``).  A trade sequence has
no fixed period: trades overlap, pool across assets and cluster in
active weeks, so a naive ``sqrt(n_trades)`` annualization inflates the
Sharpe ratio by an order of magnitude.  The per-trade semantics here
are deliberately different:

* the per-trade Sharpe (``mean / std``) is reported UNannualized --
  dimensionless, healthy band ~0.1-0.6; near 1+ per trade is a red
  flag for a computation error (contrast
  :func:`atropos.evaluation_metrics.sharpe_ratio`, whose annualized
  ``sr`` would read 2-5 on the same stream);
* the honest annualization for trade R is calendar bucketing
  (:func:`bucketed_sharpe`): aggregate R per time bucket first, then
  scale by ``sqrt(buckets_per_year)``;
* the t-statistic of the mean (:func:`trade_curve_stats`) is reported
  next to the per-trade Sharpe so the independence assumption behind
  ``mean / std * sqrt(n)`` stays visible.

Degenerate input follows the house NaN policy (honest NaN, never a
fabricated 0): fewer than two usable observations, zero dispersion or
an empty stream yield NaN.  The one deliberate exception is
``max_dd_r`` on an all-winning stream, which is a true 0.0 (the equity
curve simply never fell below its peak).

Run ``python -m atropos.backtest_metrics`` for a synthetic demo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from atropos._utils import align_inputs, clean_1d, validate_int
from atropos.evaluation_metrics import profit_factor, sharpe_ratio

__all__ = [
    "bucketed_sharpe",
    "per_trade_sharpe",
    "pooled_stats",
    "trade_curve_stats",
]

_DAY_MS = 86_400_000.0
"""Milliseconds in one day (the timestamp unit of the bucketing API)."""


def trade_curve_stats(r_net: Any) -> dict[str, float]:
    """Risk profile of a trade sequence (net R per trade).

    Returns ``{n, t_stat, max_dd_r, profit_factor}``:

    * ``n`` -- number of finite trades (NaN/inf entries are dropped);
    * ``t_stat`` -- per-trade t-statistic ``mean / std * sqrt(n)`` under
      an INDEPENDENCE assumption; pooled cross-asset or overlapping
      trades inflate it, so do not read it as a Sharpe ratio (the
      per-trade Sharpe is :func:`per_trade_sharpe` ~0.3-0.6 whenever
      this reads 10+);
    * ``max_dd_r`` -- max peak-to-trough drawdown of the cumulative
      R equity curve, in R units (input MUST be chronological for the
      drawdown to be meaningful; unlike
      :func:`atropos.evaluation_metrics.max_drawdown` this is additive
      R drawdown, not a multiplicative equity fraction);
    * ``profit_factor`` -- gross wins / gross losses, delegated to
      :func:`atropos.evaluation_metrics.profit_factor` (single home:
      ``inf`` when losses are absent, NaN when there is no PnL at all).

    NaN policy: an empty (or all-non-finite) stream returns ``n = 0``
    and NaN for every statistic; a single trade or a zero-dispersion
    stream has no honest t-statistic (NaN) while ``max_dd_r`` stays a
    real, computable number.

    Example::

        >>> curve = trade_curve_stats([1.0, -0.5, 1.5, -1.0, 0.5])
        >>> curve["n"]
        5
        >>> round(curve["t_stat"], 6)
        0.646997
        >>> curve["max_dd_r"]
        1.0
        >>> curve["profit_factor"]
        2.0
    """
    trades = clean_1d(r_net)
    trade_count = int(trades.size)
    if trade_count == 0:
        return {
            "n": 0,
            "t_stat": math.nan,
            "max_dd_r": math.nan,
            "profit_factor": math.nan,
        }
    equity = np.cumsum(trades)
    max_dd_r = float(np.max(np.maximum.accumulate(equity) - equity))
    # overflow-scale streams square to inf inside std; the finite
    # check below already reads that as "no honest t-statistic"
    with np.errstate(over="ignore"):
        spread = float(trades.std(ddof=1)) if trade_count > 1 else math.nan
    if not math.isfinite(spread) or spread <= 0.0:
        t_stat = math.nan
    else:
        t_stat = float(trades.mean() / spread * math.sqrt(trade_count))
    return {
        "n": trade_count,
        "t_stat": t_stat,
        "max_dd_r": max_dd_r,
        "profit_factor": float(profit_factor(trades)["pf"]),
    }


def per_trade_sharpe(r_net: Any) -> float:
    """Per-trade Sharpe ratio: ``mean / std`` (ddof=1).

    Dimensionless and NEVER annualized -- the annualization question is
    deferred to :func:`bucketed_sharpe`.  For a healthy strategy this
    lands in the 0.1-0.6 band; a value near 1+ per trade is a red flag
    for a computation error (e.g. lookahead or duplicated trades).

    Thin alias over
    :func:`atropos.evaluation_metrics.sharpe_ratio`: its ``sr_per_period``
    with ``periods=1`` is exactly this statistic, so the ``mean / std``
    formula keeps a single home and this wrapper only fixes the
    per-trade reading of the input unit (contrast the annualized ``sr``
    field there, which must never be quoted for trade streams; the
    DSR-gate ``sr`` inside ``atropos.ml_training.stream_stats`` is the
    same moment formula on a per-day EV stream).  The bootstrap CI of
    the delegating call is not consumed: ``n_boot=1`` keeps the wasted
    work at zero and ``seed=0`` keeps the output bit-deterministic.

    NaN policy: fewer than two finite trades or zero dispersion yields
    NaN (honest, not 0).

    Example::

        >>> round(per_trade_sharpe([1.0, -0.5, 1.5, -1.0, 0.5]), 6)
        0.289346
        >>> math.isnan(per_trade_sharpe([0.0, 0.0, 0.0]))
        True
    """
    trades = clean_1d(r_net)
    if trades.size < 2:
        return math.nan
    delegated = sharpe_ratio(trades, periods=1, n_boot=1, block=1, seed=0)
    return float(delegated["sr_per_period"])


def bucketed_sharpe(r_net: Any, ts_ms: Any, bucket_days: int = 7) -> float:
    """Annualized Sharpe from calendar-bucketed R sums.

    The honest annualization for trade-R sequences: aggregate R per
    time bucket (default week), then
    ``mean_bucket / std_bucket * sqrt(buckets_per_year)``.  Bucketing
    absorbs intra-bucket overlap (one slot per asset) and cross-asset
    pooling, which otherwise inflate a naive ``sqrt(n)`` annualization
    by an order of magnitude.

    Bucket boundaries follow ``floor((ts - ts_min) / bucket_ms)``: a
    trade stamped exactly on a boundary falls into the NEWER bucket.
    Interior buckets with no trades contribute a 0.0 sum (a genuine
    flat week that the dispersion must see); trailing empty time after
    the last trade creates no bucket.  Contrast
    :func:`atropos.evaluation_metrics.sharpe_ratio`, which annualizes a
    fixed-grid per-period stream by ``sqrt(periods)`` and takes no
    timestamps at all.

    NaN policy: fewer than two finite (trade, timestamp) rows, fewer
    than two buckets, or zero bucket dispersion yields NaN (honest,
    not 0).

    Args:
        r_net: net R per trade; non-finite entries are dropped.
        ts_ms: trade timestamps in milliseconds, aligned positionally
            with ``r_net``; non-finite stamps drop their rows.
        bucket_days: bucket width in days, integer >= 1.

    Returns:
        The annualized Sharpe ratio, or NaN.

    Raises:
        ValueError: if ``bucket_days`` is not an integer >= 1; raised
            before the data is touched.

    Example::

        >>> day = 86_400_000
        >>> stamps = np.arange(10) * day
        >>> stream = np.array([1.0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        >>> round(bucketed_sharpe(stream, stamps), 6)
        66.400974
        >>> math.isnan(bucketed_sharpe([1.0], [0.0]))
        True
    """
    bucket_days = validate_int("bucket_days", bucket_days)
    trades, stamps = align_inputs(r_net, ts_ms)
    if trades.size < 2:
        return math.nan
    bucket_ms = float(bucket_days) * _DAY_MS
    bucket_idx = np.floor((stamps - stamps.min()) / bucket_ms).astype(
        np.int64
    )
    # bincount keeps interior empty buckets as 0.0: a flat week is a
    # real observation for the dispersion, not a hole to skip
    bucket_sums = np.bincount(bucket_idx, weights=trades)
    if bucket_sums.size < 2:
        return math.nan
    with np.errstate(over="ignore"):
        spread = float(bucket_sums.std(ddof=1))
    if not math.isfinite(spread) or spread <= 0.0:
        return math.nan
    per_year = 365.25 / bucket_days
    return float(bucket_sums.mean() / spread * math.sqrt(per_year))


def pooled_stats(r_net: Any, ts_ms: Any) -> dict[str, float]:
    """Chronology-restored pooled trade stats (the common report row).

    Aligns the (net R, timestamp) rows, drops rows where either side is
    non-finite, restores chronological order (stable sort, so equal
    timestamps keep their input order) and computes the pooled row:
    ``{mean, n, dd, t_stat_naive, sharpe_per_trade,
    sharpe_ann_bucketed}``.  ``n`` counts the surviving rows, ``dd``
    and ``t_stat_naive`` come from :func:`trade_curve_stats` (drawdown
    in R units, independence-assuming t-stat), the two Sharpe fields
    from :func:`per_trade_sharpe` (never annualized) and
    :func:`bucketed_sharpe` (the honest annualization).  Degenerate
    rows follow the NaN policy of the underlying functions.

    Example::

        >>> pooled = pooled_stats([1, 0.5, -0.5, -1], [3, 0, 2, 1])
        >>> pooled["mean"], pooled["n"], pooled["dd"]
        (0.0, 4, 1.5)
        >>> pooled["t_stat_naive"], pooled["sharpe_per_trade"]
        (0.0, 0.0)
        >>> math.isnan(pooled["sharpe_ann_bucketed"])
        True
    """
    trades, stamps = align_inputs(r_net, ts_ms)
    order = np.argsort(stamps, kind="stable")
    chrono = trades[order]
    curve = trade_curve_stats(chrono)
    mean_r = float(trades.mean()) if trades.size else math.nan
    return {
        "mean": mean_r,
        "n": int(trades.size),
        "dd": curve["max_dd_r"],
        "t_stat_naive": curve["t_stat"],
        "sharpe_per_trade": per_trade_sharpe(chrono),
        "sharpe_ann_bucketed": bucketed_sharpe(chrono, stamps),
    }


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo every metric on one seeded manual trade stream."""
    print("=" * 72)
    print("backtest_metrics demo: seeded 40-week manual trade stream")
    print("=" * 72)
    day_ms = 86_400_000
    rng = np.random.default_rng(7)
    trade_count = 260
    winners = rng.random(trade_count) < 0.44
    sizes = 0.5 + rng.random(trade_count)
    # asymmetric payoff: fewer winners but each one is 1.8x a loss,
    # so the seeded world carries a genuine positive per-trade edge
    stream = np.where(winners, 1.8 * sizes, -sizes)
    stamps = np.sort(rng.integers(0, 40 * 7, trade_count)) * day_ms

    curve = trade_curve_stats(stream)
    print(
        f"trade_curve_stats : n={curve['n']} "
        f"t_stat={curve['t_stat']:.2f} "
        f"max_dd_r={curve['max_dd_r']:.2f} "
        f"pf={curve['profit_factor']:.2f}"
    )
    assert curve["n"] == trade_count
    assert curve["t_stat"] > 0.0 and curve["profit_factor"] > 1.0
    assert curve["max_dd_r"] > 0.0

    trade_sharpe = per_trade_sharpe(stream)
    print(f"per_trade_sharpe  : {trade_sharpe:.3f} (never annualized)")
    manual = float(stream.mean() / stream.std(ddof=1))
    assert math.isclose(trade_sharpe, manual, rel_tol=1e-12)
    assert math.isclose(
        trade_sharpe * math.sqrt(trade_count),
        curve["t_stat"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    bucketed = bucketed_sharpe(stream, stamps)
    print(f"bucketed_sharpe   : {bucketed:.2f} per year (weekly buckets)")
    assert math.isfinite(bucketed)
    # shifting the later trades one full week forward inserts a flat
    # week; a 0.0-sum bucket changes the dispersion deterministically
    gap_start = 20 * 7 * day_ms
    shift = (stamps >= gap_start) * (7 * day_ms)
    gapped = bucketed_sharpe(stream, stamps + shift)
    assert math.isfinite(gapped) and gapped != bucketed

    # reversing BOTH sides keeps every (trade, stamp) pair intact but
    # scrambles the time order; pooled_stats must restore chronology
    pooled = pooled_stats(stream[::-1], stamps[::-1])
    print(
        f"pooled_stats      : n={pooled['n']} mean={pooled['mean']:.4f} "
        f"dd={pooled['dd']:.2f}"
    )
    assert pooled["n"] == trade_count
    assert math.isclose(pooled["dd"], curve["max_dd_r"], rel_tol=1e-12)
    assert math.isclose(
        pooled["sharpe_per_trade"], trade_sharpe, rel_tol=1e-12
    )


if __name__ == "__main__":
    run_all_examples()
