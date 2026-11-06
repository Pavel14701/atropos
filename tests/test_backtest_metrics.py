"""Known-answer, differential and chaos tests for the per-trade metrics.

Covers :mod:`atropos.backtest_metrics` (ported from the donor engine's
``engine/metrics/trade.py``): known-answer tests on hand-computed
streams, the contract table for constant / empty / thin input, bucket
boundaries and absence handling for the calendar bucketing, the
delegation differentials against
:mod:`atropos.evaluation_metrics`, and bitwise determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos.backtest_metrics import (
    bucketed_sharpe,
    per_trade_sharpe,
    pooled_stats,
    trade_curve_stats,
)
from atropos.evaluation_metrics import profit_factor as eval_profit_factor
from atropos.evaluation_metrics import sharpe_ratio, win_rate

_DAY_MS = 86_400_000

# hand-checked stream: mean=0.3, std(ddof=1)=1.036822..., eq curve
# [1, 0.5, 2.0, 1.0, 1.5] -> max drawdown 1.0 R, pf = 3.0 / 1.5 = 2.0
_MANUAL_STREAM = np.array([1.0, -0.5, 1.5, -1.0, 0.5])
_MANUAL_T_STAT = 0.6469966392206304
_MANUAL_SHARPE = 0.28934569330224724


# ---------------------------------------------------------------------------
# trade_curve_stats: known answers and the degenerate contract
# ---------------------------------------------------------------------------
def test_trade_curve_stats_known_stream() -> None:
    """Hand-computed t-stat, drawdown and profit factor reproduce."""
    curve = trade_curve_stats(_MANUAL_STREAM)
    assert curve["n"] == 5
    assert math.isclose(curve["t_stat"], _MANUAL_T_STAT, rel_tol=1e-12)
    assert curve["max_dd_r"] == 1.0
    assert curve["profit_factor"] == 2.0


def test_trade_curve_stats_constant_stream_contract() -> None:
    """Constant stream: NaN t-stat, true-zero drawdown, honest pf."""
    positive = trade_curve_stats([2.0, 2.0, 2.0, 2.0])
    assert positive["n"] == 4
    assert math.isnan(positive["t_stat"])  # zero dispersion, no t-stat
    assert positive["max_dd_r"] == 0.0  # equity never fell: true 0.0
    assert positive["profit_factor"] == float("inf")
    zero = trade_curve_stats([0.0, 0.0, 0.0])
    assert math.isnan(zero["t_stat"])
    assert zero["max_dd_r"] == 0.0
    assert math.isnan(zero["profit_factor"])  # no PnL at all


def test_trade_curve_stats_profit_factor_manual_win_loss() -> None:
    """Manual W/L bookkeeping: pf counts winners vs losers only."""
    trades = np.array([2.0, -1.0, 3.0, -1.0, 0.0])
    curve = trade_curve_stats(trades)
    gross_win = float(trades[trades > 0].sum())
    gross_loss = float(-trades[trades < 0].sum())
    assert curve["profit_factor"] == gross_win / gross_loss == 2.5
    # single home: identical to the evaluation_metrics pf, and the
    # winner count agrees with the house win_rate bookkeeping
    assert curve["profit_factor"] == eval_profit_factor(trades)["pf"]
    winners = int((trades > 0).sum())
    assert win_rate(trades)["wr"] == winners / 5
    assert eval_profit_factor(trades)["n_winners"] == winners


def test_trade_curve_stats_empty_and_thin() -> None:
    """Empty -> all-NaN payload; single trade -> no t-stat."""
    empty = trade_curve_stats([])
    assert empty["n"] == 0
    for value in empty.values():
        assert value == 0 or math.isnan(value)
    thin = trade_curve_stats([1.5])
    assert thin["n"] == 1
    assert math.isnan(thin["t_stat"])
    assert thin["max_dd_r"] == 0.0
    assert thin["profit_factor"] == float("inf")
    loss_only = trade_curve_stats([-1.5])
    assert loss_only["max_dd_r"] == 0.0  # trough == the running peak
    assert loss_only["profit_factor"] == 0.0
    # deeper pin of the donor drawdown semantics: measured against the
    # best-so-far equity, so a hole dug below an early peak counts
    dug_deeper = trade_curve_stats([-1.0, -2.0])
    assert dug_deeper["max_dd_r"] == 2.0


def test_trade_curve_stats_drops_non_finite() -> None:
    """NaN/inf trades are dropped before any statistic."""
    dirty = np.array([np.nan, np.inf, 1.0, -1.0, -np.inf])
    curve = trade_curve_stats(dirty)
    assert curve["n"] == 2
    assert curve["profit_factor"] == 1.0
    assert curve["max_dd_r"] == 1.0


# ---------------------------------------------------------------------------
# per_trade_sharpe: differential vs manual formula and delegation home
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stream",
    [
        _MANUAL_STREAM,
        np.array([3.0, -1.0, 2.0, -2.0, 1.0, -0.5, 4.0, -1.0]),
        np.array([1e-8, -1e-8, 2e-8, -2e-8, 1e-8, -1e-8, 3e-8, -3e-8]),
    ],
)
def test_per_trade_sharpe_matches_manual_formula(stream: np.ndarray) -> None:
    """Differential: the function equals mean / std(ddof=1) by hand."""
    manual = float(np.mean(stream) / np.std(stream, ddof=1))
    assert math.isclose(per_trade_sharpe(stream), manual, rel_tol=1e-12)


def test_per_trade_sharpe_matches_random_seeded_stream() -> None:
    """Seeded random stream: same differential on realistic data."""
    rng = np.random.default_rng(11)
    stream = 0.3 + 1.2 * rng.standard_normal(200)
    manual = float(np.mean(stream) / np.std(stream, ddof=1))
    assert math.isclose(per_trade_sharpe(stream), manual, rel_tol=1e-12)


def test_per_trade_sharpe_delegates_to_per_period_sharpe() -> None:
    """Identical to evaluation_metrics sr_per_period with periods=1."""
    delegated = sharpe_ratio(_MANUAL_STREAM, periods=1)["sr_per_period"]
    assert per_trade_sharpe(_MANUAL_STREAM) == float(delegated)


def test_per_trade_sharpe_degenerate_is_nan() -> None:
    """Thin and constant streams yield NaN, never 0 or an error."""
    assert math.isnan(per_trade_sharpe([]))
    assert math.isnan(per_trade_sharpe([2.5]))
    assert math.isnan(per_trade_sharpe([0.0, 0.0, 0.0]))
    assert math.isnan(per_trade_sharpe([np.nan, np.inf]))


# ---------------------------------------------------------------------------
# bucketed_sharpe: boundaries, absence handling, known answer
# ---------------------------------------------------------------------------
def test_bucketed_sharpe_known_answer_two_buckets() -> None:
    """Week sums [7, 6] -> hand-computed annualized Sharpe."""
    stamps = np.arange(10) * _DAY_MS
    stream = np.array([1.0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    expected = (6.5 / math.sqrt(0.5)) * math.sqrt(365.25 / 7)
    assert math.isclose(
        bucketed_sharpe(stream, stamps), expected, rel_tol=1e-12
    )


def test_bucketed_sharpe_boundary_falls_into_newer_bucket() -> None:
    """A stamp exactly on the boundary lands in the newer bucket."""
    stream = np.array([1.0, 2.0])
    inside = bucketed_sharpe(stream, np.array([0, 6]) * _DAY_MS)
    on_edge = bucketed_sharpe(stream, np.array([0, 7]) * _DAY_MS)
    # inside: one bucket [3.0] -> NaN; on edge: buckets [1.0, 2.0]
    assert math.isnan(inside)
    expected = (1.5 / math.sqrt(0.5)) * math.sqrt(365.25 / 7)
    assert math.isclose(on_edge, expected, rel_tol=1e-12)


def test_bucketed_sharpe_interior_gap_is_a_zero_observation() -> None:
    """A flat week between active weeks is a real 0.0-sum bucket."""
    dense = np.array([1.0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    stamps_dense = np.arange(14) * _DAY_MS
    stamps_gapped = np.concatenate([np.arange(7), np.arange(7) + 14])
    stream_gapped = np.array([1.0] * 7 + [2.0] * 7)
    dense_sharpe = bucketed_sharpe(dense, stamps_dense)
    gapped_sharpe = bucketed_sharpe(stream_gapped, stamps_gapped * _DAY_MS)
    # dense: sums [7, 14]; gapped: sums [7, 0, 14] - the flat week
    # enters the dispersion, so the values differ deterministically
    assert math.isfinite(dense_sharpe) and math.isfinite(gapped_sharpe)
    assert dense_sharpe != gapped_sharpe
    bucket_sums = np.bincount(
        np.floor(stamps_gapped / 7).astype(np.int64), weights=stream_gapped
    )
    assert bucket_sums.tolist() == [7.0, 0.0, 14.0]


def test_bucketed_sharpe_trailing_silence_adds_no_bucket() -> None:
    """Trailing empty time after the last trade changes nothing."""
    stream = np.array([1.0, -0.5, 2.0])
    base = bucketed_sharpe(stream, np.array([0, 1, 8]) * _DAY_MS)
    stamps_later = np.array([0, 1, 8]) * _DAY_MS + 30 * _DAY_MS
    later = bucketed_sharpe(stream, stamps_later)
    assert math.isclose(base, later, rel_tol=1e-12)


def test_bucketed_sharpe_translation_invariant() -> None:
    """Shifting all stamps by a constant leaves the value unchanged."""
    rng = np.random.default_rng(5)
    stream = rng.standard_normal(50)
    stamps = rng.integers(0, 100, 50) * _DAY_MS
    base = bucketed_sharpe(stream, stamps)
    shifted = bucketed_sharpe(stream, stamps + 1000 * _DAY_MS)
    assert math.isclose(base, shifted, rel_tol=1e-12)


def test_bucketed_sharpe_degenerate_is_nan() -> None:
    """Thin input, one bucket, flat buckets -> honest NaN."""
    stamps = np.arange(6) * _DAY_MS
    stream = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert math.isnan(bucketed_sharpe([], []))
    assert math.isnan(bucketed_sharpe([1.0], [0.0]))
    assert math.isnan(bucketed_sharpe(stream[:5], stamps[:5]))  # one week
    assert math.isnan(bucketed_sharpe(np.zeros(8), np.arange(8) * _DAY_MS))
    dirty = np.array([1.0, np.nan, -1.0])
    assert math.isnan(bucketed_sharpe(dirty, stamps[:3]))


def test_bucketed_sharpe_bucket_days_validation() -> None:
    """bucket_days must be an integer >= 1 (ValueError otherwise)."""
    with pytest.raises(ValueError, match="bucket_days"):
        bucketed_sharpe([1.0, 2.0], [0.0, 1.0], bucket_days=0)
    with pytest.raises(ValueError, match="bucket_days"):
        bucketed_sharpe(
            [1.0, 2.0],
            [0.0, 1.0],
            bucket_days=2.5,  # type: ignore[arg-type]
        )


def test_bucketed_sharpe_pandas_inputs_align() -> None:
    """Pandas Series inputs align positionally like numpy arrays."""
    pd = pytest.importorskip("pandas")
    stream = pd.Series([1.0, -0.5, 2.0])
    stamps = pd.Series([0.0, 1.0, 8.0]) * _DAY_MS
    from_pandas = bucketed_sharpe(stream, stamps)
    from_numpy = bucketed_sharpe(stream.to_numpy(), stamps.to_numpy())
    assert math.isclose(from_pandas, from_numpy, rel_tol=1e-15)


# ---------------------------------------------------------------------------
# pooled_stats: chronology restoration and composition consistency
# ---------------------------------------------------------------------------
def test_pooled_stats_restores_chronology() -> None:
    """Unsorted input reports the chronological-curve statistics."""
    stream = [1.0, 0.5, -0.5, -1.0]
    stamps = [3.0, 0.0, 2.0, 1.0]
    pooled = pooled_stats(stream, stamps)
    chrono = np.array([0.5, -1.0, -0.5, 1.0])
    assert pooled["mean"] == 0.0
    assert pooled["n"] == 4
    assert pooled["dd"] == 1.5
    assert pooled["t_stat_naive"] == 0.0
    assert pooled["sharpe_per_trade"] == 0.0
    assert math.isnan(pooled["sharpe_ann_bucketed"])
    assert pooled["dd"] == trade_curve_stats(chrono)["max_dd_r"]
    assert pooled["sharpe_per_trade"] == per_trade_sharpe(chrono)


def test_pooled_stats_matches_part_functions_on_random_stream() -> None:
    """Every field equals the corresponding single-stream function."""
    rng = np.random.default_rng(3)
    trade_count = 120
    stream = rng.standard_normal(trade_count)
    stamps = np.sort(rng.integers(0, 60, trade_count)) * _DAY_MS
    order = np.argsort(stamps, kind="stable")
    chrono = stream[order]
    pooled = pooled_stats(stream, stamps)
    assert pooled["mean"] == float(np.mean(stream))
    assert pooled["n"] == trade_count
    assert pooled["dd"] == trade_curve_stats(chrono)["max_dd_r"]
    assert math.isclose(
        pooled["t_stat_naive"],
        trade_curve_stats(chrono)["t_stat"],
        rel_tol=1e-12,
    )
    assert pooled["sharpe_per_trade"] == per_trade_sharpe(chrono)
    assert math.isclose(
        pooled["sharpe_ann_bucketed"],
        bucketed_sharpe(chrono, stamps),
        rel_tol=1e-12,
    )


def test_pooled_stats_drops_rows_with_bad_side() -> None:
    """A non-finite trade OR stamp drops the whole row."""
    stream = np.array([1.0, np.nan, -1.0, 2.0])
    stamps = np.array([0.0, 1.0, np.nan, 3.0])
    pooled = pooled_stats(stream, stamps)
    assert pooled["n"] == 2
    assert pooled["mean"] == 1.5


def test_pooled_stats_empty_is_all_nan() -> None:
    """Empty input: n=0 and honest NaN statistics."""
    pooled = pooled_stats([], [])
    assert pooled["n"] == 0
    assert math.isnan(pooled["mean"])
    assert math.isnan(pooled["dd"])
    assert math.isnan(pooled["sharpe_per_trade"])
    assert math.isnan(pooled["sharpe_ann_bucketed"])


# ---------------------------------------------------------------------------
# determinism and chaos minimum
# ---------------------------------------------------------------------------
def test_all_functions_are_deterministic() -> None:
    """Repeated calls reproduce results bit for bit (no hidden RNG)."""
    rng = np.random.default_rng(17)
    stream = rng.standard_normal(80)
    stamps = rng.integers(0, 50, 80) * _DAY_MS
    assert trade_curve_stats(stream) == trade_curve_stats(stream)
    assert pooled_stats(stream, stamps) == pooled_stats(stream, stamps)
    assert bucketed_sharpe(stream, stamps) == bucketed_sharpe(stream, stamps)
    assert per_trade_sharpe(stream) == per_trade_sharpe(stream)


def test_chaos_extreme_scales_stay_finite_or_nan() -> None:
    """Overflow-scale streams degrade to NaN, not inf garbage."""
    huge = np.array([1e300, -1e300, 1e300, -1e300, 1e300])
    curve = trade_curve_stats(huge)
    assert curve["n"] == 5
    # equity [1e300, 0, 1e300, 0, 1e300]: a real, huge drawdown
    assert curve["max_dd_r"] == 1e300
    # squared deviations overflowed: no fabricated t-stat
    assert math.isnan(curve["t_stat"])
    assert curve["profit_factor"] == 1.5
    # all four aligned rows land in one weekly bucket: honest NaN
    huge_stamps = np.arange(5) * _DAY_MS
    assert math.isnan(bucketed_sharpe(huge, huge_stamps))
