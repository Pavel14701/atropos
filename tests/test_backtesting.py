"""Tests for the atropos.backtesting event-simulator engine.

Hand-computed R worlds pin the engine arithmetic bit-by-bit: the taker
cost stack (TP / SL / time / gap-through exits), the state machine's
slot discipline with force-close reconstruction, the admission policies
(including the executable FCFS-loses-the-best-trade finding), maker vs
market execution mechanics, per-trade metrics and the chaos guards.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.backtesting import (
    DEFAULT_COST,
    CostModel,
    bucketed_sharpe,
    maker_entry_sim,
    maker_sim,
    market_sim,
    per_trade_sharpe,
    pessimistic_r,
    pooled_stats,
    rank_normalize_group,
    run_admission_policy,
    run_position_state_machine,
    simulate_signal,
    trade_curve_stats,
)


def _tp_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long world: entry fill 100, target touched at bar 2, no stop."""
    opens = [100.0] * 6
    highs = [100.0, 100.0, 102.4, 100.0, 100.0, 100.0]
    lows = [100.0] * 6
    closes = [100.0] * 6
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


def _sl_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long world: entry fill 100, stop touched at bar 2, no target."""
    opens = [100.0] * 6
    highs = [100.0] * 6
    lows = [100.0, 99.6, 99.4, 100.0, 100.0, 100.0]
    closes = [100.0] * 6
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


def _zone_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maker-entry geometry: dip into the zone at bar 2, TP at bar 10."""
    opens = [100.0] * 12
    highs = [100.0] * 12
    lows = [100.0] * 12
    lows[2] = 99.5
    highs[10] = 102.0
    closes = [100.0] * 12
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


def _time_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quiet world with a single up-move close at bar 3 (time exit)."""
    opens = [100.0] * 8
    highs = [100.0] * 8
    highs[3] = 101.0
    lows = [100.0] * 8
    closes = [100.0] * 8
    closes[3] = 101.0
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


def _gap_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panic world: bar 1 opens 1.5 points below the prior close."""
    opens = [100.0, 98.5, 98.5, 98.5, 98.5, 98.5]
    highs = [100.0, 98.6, 98.6, 98.6, 98.6, 98.6]
    lows = [100.0, 98.4, 98.4, 98.4, 98.4, 98.4]
    closes = [100.0, 98.5, 98.5, 98.5, 98.5, 98.5]
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


def _rally_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Six-bar rally used by the state-machine force-close checks."""
    opens = [100.0, 100.0, 100.0, 102.5, 103.5, 103.8]
    highs = [100.0, 100.0, 102.9, 104.0, 104.0, 104.0]
    lows = [100.0, 99.9, 100.0, 102.0, 103.0, 103.5]
    closes = [100.0, 100.0, 102.5, 103.5, 103.8, 104.0]
    return (
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
    )


# ---------------------------------------------------------------------------
# 1. simulate_signal: the taker exit stack
# ---------------------------------------------------------------------------
def test_tp_hit_exact_taker_costs() -> None:
    """Clean TP hit: pess = opt - e_mult*gen_slip*fill/risk (no more)."""
    opens, highs, lows, closes = _tp_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 48, 1.0
    )
    # fill 100, risk 1: cost_r = (2*comm + slip)*100 = 0.25; gross = 2R
    assert out["exit_idx"] == 2
    assert out["r_opt"] == pytest.approx(2.0 - 0.25)
    # the ONLY pessimism on a target exit is the entry-slippage term:
    entry_slip_r = DEFAULT_COST.e_mult * DEFAULT_COST.gen_slip * 100.0
    assert out["r_opt"] - out["r_pess"] == pytest.approx(entry_slip_r)
    assert out["r_pess"] == pytest.approx(1.65)


def test_sl_hit_pessimism_stack_exact() -> None:
    """SL hit: pess - opt == -(pe + xtr + gap) to the last member."""
    opens, highs, lows, closes = _sl_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "long", 99.5, 200.0, 48, 1.0
    )
    assert out["exit_idx"] == 2
    # optimistic: slipped stop exit net of costs, risk distance 0.5
    assert out["r_opt"] == pytest.approx(
        (99.5 * (1 - 0.0005) - 100.0) / 0.5
        - (2 * 0.001 + 0.0005) * 100.0 / 0.5
    )
    pe = DEFAULT_COST.e_mult * DEFAULT_COST.gen_slip * 100.0 / 0.5
    xtr = (DEFAULT_COST.x_mult - 1) * DEFAULT_COST.gen_slip * 99.5 / 0.5
    gap_r = DEFAULT_COST.gap * 1.0 / 0.5
    assert out["r_pess"] - out["r_opt"] == pytest.approx(-(pe + xtr + gap_r))


def test_gap_through_stop_entry_scratch() -> None:
    """Entry bar opens beyond the stop: immediate scratch, never a win."""
    opens, highs, lows, closes = _gap_world()
    out = simulate_signal(
        opens, highs, lows, closes, 1, "long", 99.0, 110.0, 48, 1.0
    )
    assert out["exit_idx"] == 1  # scratched on the entry bar itself
    fill = 98.5
    denom = abs(fill - 99.0)
    cost_r = (
        2 * DEFAULT_COST.comm * fill + DEFAULT_COST.gen_slip * fill
    ) / denom
    assert out["r_opt"] == pytest.approx(-cost_r)
    pe = DEFAULT_COST.e_mult * DEFAULT_COST.gen_slip * fill / denom
    xtr = (
        (DEFAULT_COST.x_mult - 1) * DEFAULT_COST.gen_slip * abs(fill) / denom
    )
    assert out["r_pess"] == pytest.approx(-cost_r - pe - xtr)
    assert out["r_pess"] < out["r_opt"]


def test_gap_through_short_side_mirrors() -> None:
    """Short entry above its stop scratches the same way."""
    opens, highs, lows, closes = _gap_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "short", 98.0, 90.0, 48, 1.0
    )
    assert out["exit_idx"] == 0
    assert out["r_opt"] == pytest.approx(
        -((2 * 0.001 + 0.0005) * 100.0 / 2.0)
    )
    assert out["r_pess"] < out["r_opt"]


def test_time_exit_on_hold_limit() -> None:
    """Time exit lands on the final hold bar at its slipped close."""
    opens, highs, lows, closes = _time_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "long", 90.0, 200.0, 4, 1.0
    )
    assert out["exit_idx"] == 3  # entry + hold_bars - 1
    exit_px = 101.0 * (1 - 0.0005)
    denom = 10.0  # |100 - 90|
    cost_r = (2 * 0.001 + 0.0005) * 100.0 / denom
    assert out["r_opt"] == pytest.approx((exit_px - 100.0) / denom - cost_r)
    pe = 2 * 0.0005 * 100.0 / denom
    xtr = (DEFAULT_COST.x_mult - 1) * DEFAULT_COST.gen_slip * exit_px / denom
    assert out["r_pess"] == pytest.approx(out["r_opt"] - pe - xtr)


def test_no_exit_when_data_ends_inside_window() -> None:
    """Data ending before the hold window: NaN pair, exit_idx -1."""
    opens, highs, lows, closes = _tp_world()
    out = simulate_signal(
        opens, highs, lows, closes, 3, "long", 99.0, 102.4, 48, 1.0
    )
    assert out["exit_idx"] == -1
    assert math.isnan(out["r_opt"]) and math.isnan(out["r_pess"])


def test_risk_nonpositive_returns_nan_scaffold() -> None:
    """Stop at the fill price: zero risk -> NaN pair, exit_idx -1."""
    opens, highs, lows, closes = _tp_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "long", 100.0, 102.0, 48, 1.0
    )
    assert out["exit_idx"] == -1
    assert math.isnan(out["r_opt"]) and math.isnan(out["r_pess"])


def test_risk_ref_overrides_denominator() -> None:
    """risk_ref replaces |fill - stop| as the R denominator."""
    opens, highs, lows, closes = _tp_world()
    ref = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
        risk_ref=2.0,
    )
    plain = simulate_signal(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 48, 1.0
    )
    # every per-risk R term halves with the doubled denominator
    assert ref["r_opt"] == pytest.approx(2.0 / 2.0 - 0.25 / 2.0)
    assert ref["r_pess"] == pytest.approx(plain["r_pess"] / 2.0)
    # non-positive risk_ref falls back to the actual risk distance
    fallback = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
        risk_ref=0.0,
    )
    assert fallback == plain


def test_sl_first_pessimism_inside_one_bar() -> None:
    """A bar touching BOTH levels resolves as the stop (SL-first)."""
    opens = [100.0, 100.0]
    highs = [100.0, 103.0]
    lows = [100.0, 98.5]
    closes = [100.0, 100.0]
    both = simulate_signal(
        np.asarray(opens),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(closes),
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
    )
    stop_only = simulate_signal(
        np.asarray(opens),
        np.asarray(highs),
        np.asarray([100.0, 98.5]),
        np.asarray(closes),
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
    )
    assert both == stop_only  # same bar as a pure stop bar
    assert both["r_pess"] < both["r_opt"] < 0.0


def test_short_tp_hit_mirrors_long_exactly() -> None:
    """Mirror world: short TP R == long TP R (level-free exit slip)."""
    opens = [100.0] * 6
    long_highs = [100.0, 100.0, 102.4, 100.0, 100.0, 100.0]
    long_lows = [100.0] * 6
    short_highs = [100.0] * 6
    short_lows = [100.0, 100.0, 97.6, 100.0, 100.0, 100.0]
    closes = [100.0] * 6
    long_out = simulate_signal(
        np.asarray(opens),
        np.asarray(long_highs),
        np.asarray(long_lows),
        np.asarray(closes),
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
    )
    short_out = simulate_signal(
        np.asarray(opens),
        np.asarray(short_highs),
        np.asarray(short_lows),
        np.asarray(closes),
        0,
        "short",
        101.0,
        98.0,
        48,
        1.0,
    )
    assert short_out["r_opt"] == long_out["r_opt"]
    assert short_out["r_pess"] == long_out["r_pess"]
    assert short_out["exit_idx"] == long_out["exit_idx"] == 2


def test_short_sl_hit_mirror_within_level_slip() -> None:
    """Mirror SL worlds differ ONLY by the level-proportional slip."""
    long_out = simulate_signal(*_sl_world(), 0, "long", 99.5, 200.0, 48, 1.0)
    opens = [100.0] * 6
    short_highs = [100.0, 100.4, 100.6, 100.0, 100.0, 100.0]
    short_lows = [100.0] * 6
    short_out = simulate_signal(
        np.asarray(opens),
        np.asarray(short_highs),
        np.asarray(short_lows),
        np.asarray([100.0] * 6),
        0,
        "short",
        100.5,
        50.0,  # a level the path can never reach downward
        48,
        1.0,
    )
    # both sides lose ~1.6R opt / ~2.4R pess; the residual asymmetry
    # is gen_slip*(sl_long - sl_short)/risk = 0.0005*(-1.0)/0.5 = -0.001
    assert long_out["r_opt"] == pytest.approx(-1.5995)
    assert short_out["r_opt"] == pytest.approx(-1.6005)
    assert short_out["r_opt"] - long_out["r_opt"] == pytest.approx(-0.001)
    assert long_out["r_pess"] == pytest.approx(-2.399)
    assert short_out["r_pess"] == pytest.approx(-2.401)


def test_cost_model_overrides_move_r_predictably() -> None:
    """comm=0 removes exactly the round-trip fee; zero cost == gross."""
    opens, highs, lows, closes = _tp_world()
    no_comm = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
        cost=CostModel(comm=0.0),
    )
    # cost_r drops by exactly 2 * comm * fill / risk = 0.2
    assert no_comm["r_opt"] == pytest.approx(2.0 - 0.05)
    frictionless = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        48,
        1.0,
        cost=CostModel(
            comm=0.0, gen_slip=0.0, gap=0.0, e_mult=0.0, x_mult=1.0
        ),
    )
    assert frictionless["r_opt"] == pytest.approx(2.0)
    assert frictionless["r_pess"] == pytest.approx(2.0)  # no pessimism left
    # x_mult=1 zeroes the extra exit slippage on the SL stack
    sl_opens, sl_highs, sl_lows, sl_closes = _sl_world()
    soft = simulate_signal(
        sl_opens,
        sl_highs,
        sl_lows,
        sl_closes,
        0,
        "long",
        99.5,
        200.0,
        48,
        1.0,
        cost=CostModel(gap=0.0, x_mult=1.0),
    )
    pe = DEFAULT_COST.e_mult * DEFAULT_COST.gen_slip * 100.0 / 0.5
    assert soft["r_pess"] == pytest.approx(soft["r_opt"] - pe)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"side": "up"}, "side must be"),
        ({"hold_bars": 0}, "hold_bars"),
        ({"entry_idx": 6}, "entry_idx must be <"),
        ({"entry_idx": -1}, "entry_idx must be"),
        ({"stop_price": float("nan")}, "stop_price must be a finite"),
        ({"atr_value": float("inf")}, "atr_value must be a finite"),
        ({"risk_ref": float("nan")}, "risk_ref must be a finite"),
    ],
)
def test_simulate_signal_rejects_bad_scalars(
    kwargs: dict[str, Any], match: str
) -> None:
    """Invalid side / hold / index / scalar params raise ValueError."""
    opens, highs, lows, closes = _tp_world()
    base: dict[str, Any] = {
        "side": "long",
        "stop_price": 99.0,
        "target_price": 102.0,
        "hold_bars": 48,
        "atr_value": 1.0,
        "entry_idx": 0,
    }
    with pytest.raises(ValueError, match=match):
        simulate_signal(opens, highs, lows, closes, **{**base, **kwargs})


def test_simulate_signal_rejects_bad_arrays() -> None:
    """NaN/inf prices, ragged lengths and 2-D input all raise."""
    opens, highs, lows, closes = _tp_world()
    nan_opens = opens.copy()
    nan_opens[2] = np.nan
    with pytest.raises(ValueError, match="opens must be finite"):
        simulate_signal(
            nan_opens, highs, lows, closes, 0, "long", 99.0, 102.0, 48, 1.0
        )
    inf_closes = closes.copy()
    inf_closes[0] = np.inf
    with pytest.raises(ValueError, match="closes must be finite"):
        simulate_signal(
            opens, highs, lows, inf_closes, 0, "long", 99.0, 102.0, 48, 1.0
        )
    with pytest.raises(ValueError, match="same length"):
        simulate_signal(
            opens, highs, lows, closes[:-1], 0, "long", 99.0, 102.0, 48, 1.0
        )
    with pytest.raises(ValueError, match="must be 1-D"):
        simulate_signal(
            np.ones((6, 2)),
            highs,
            lows,
            closes,
            0,
            "long",
            99.0,
            102.0,
            48,
            1.0,
        )
    with pytest.raises(ValueError, match="hold_bars"):
        simulate_signal(
            opens,
            highs,
            lows,
            closes,
            0,
            "long",
            99.0,
            102.0,
            # fractional hold: rejected by validate_int with ValueError
            0.5,  # type: ignore[arg-type]
            1.0,
        )


# ---------------------------------------------------------------------------
# 2. pessimistic_r: the per-row pessimism stack
# ---------------------------------------------------------------------------
def test_pessimistic_r_sl_row_bitwise_with_engine() -> None:
    """SL row: r_net minus pe/xtr/gap exactly at the default cost."""
    value = pessimistic_r(
        risk_unit=1.0,
        fill_price=100.0,
        sl_price=99.0,
        exit_reason="sl",
        exit_price=99.0,
        atr_i=0.4,
        r_net=2.0,
    )
    expected = 2.0 - (
        2.0 * 0.0005 * 100.0 + (2.0 - 1.0) * 0.0005 * 99.0 + 0.25 * 0.4
    )
    assert value == pytest.approx(expected, abs=1e-15)
    assert value == pytest.approx(1.7505)


def test_pessimistic_r_tp_row_entry_term_only() -> None:
    """TP row: only the entry-slippage term is charged."""
    value = pessimistic_r(
        risk_unit=0.5,
        fill_price=100.0,
        sl_price=99.0,
        exit_reason="tp",
        exit_price=102.0,
        atr_i=1.0,
        r_net=4.0,
    )
    assert value == pytest.approx(4.0 - 2.0 * 0.0005 * 100.0 / 0.5)


def test_pessimistic_r_time_row_exit_slippage() -> None:
    """Time row: pe + (x_mult-1)*slip on the exit price."""
    value = pessimistic_r(
        risk_unit=1.0,
        fill_price=100.0,
        sl_price=99.0,
        exit_reason="time",
        exit_price=101.0,
        atr_i=1.0,
        r_net=0.7,
    )
    expected = 0.7 - (2.0 * 0.0005 * 100.0 + (2.0 - 1.0) * 0.0005 * 101.0)
    assert value == pytest.approx(expected)


def test_pessimistic_r_matches_simulate_signal_sl() -> None:
    """pessimistic_r(r_net=r_opt) reproduces the engine's pess leg."""
    opens, highs, lows, closes = _sl_world()
    out = simulate_signal(
        opens, highs, lows, closes, 0, "long", 99.5, 200.0, 48, 1.0
    )
    row = pessimistic_r(
        risk_unit=0.5,
        fill_price=100.0,
        sl_price=99.5,
        exit_reason="sl",
        exit_price=99.5,
        atr_i=1.0,
        r_net=out["r_opt"],
    )
    assert row == pytest.approx(out["r_pess"], abs=1e-12)


def test_pessimistic_r_validates_inputs() -> None:
    """Bad risk_unit / exit_reason / non-finite prices raise."""
    with pytest.raises(ValueError, match="risk_unit must be > 0"):
        pessimistic_r(
            risk_unit=0.0,
            fill_price=100.0,
            sl_price=99.0,
            exit_reason="sl",
            exit_price=99.0,
            atr_i=1.0,
            r_net=1.0,
        )
    with pytest.raises(ValueError, match="exit_reason must be one of"):
        pessimistic_r(
            risk_unit=1.0,
            fill_price=100.0,
            sl_price=99.0,
            exit_reason="gap",
            exit_price=99.0,
            atr_i=1.0,
            r_net=1.0,
        )
    with pytest.raises(ValueError, match="fill_price must be a finite"):
        pessimistic_r(
            risk_unit=1.0,
            fill_price=float("nan"),
            sl_price=99.0,
            exit_reason="tp",
            exit_price=102.0,
            atr_i=1.0,
            r_net=1.0,
        )


# ---------------------------------------------------------------------------
# 3. run_position_state_machine: slot discipline
# ---------------------------------------------------------------------------
def _signals_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a signal frame from row dicts (columns in stable order)."""
    return pd.DataFrame(rows)


def test_machine_takes_only_first_of_overlapping_signals() -> None:
    """Three in-position candidates: exactly one slot, rest skipped."""
    opens, highs, lows, closes = _rally_world()
    long_exit = simulate_signal(
        opens, highs, lows, closes, 1, "long", 95.0, 103.0, 48, 1.0
    )["exit_idx"]
    frame = _signals_frame(
        [
            {"decision_idx": 0, "side": "long", "exit_idx": int(long_exit)},
            {"decision_idx": 1, "side": "long", "exit_idx": 9},
            {"decision_idx": 2, "side": "long", "exit_idx": 9},
        ]
    )
    replay = run_position_state_machine(frame)
    assert replay["status"].tolist() == ["taken", "skipped", "skipped"]
    assert replay["reason"].tolist() == ["", "in_position", "in_position"]


def test_force_close_flags_and_recomputed_pnl() -> None:
    """Reverse force-closes at d: flag + force_close_r == hand recompute."""
    opens, highs, lows, closes = _rally_world()
    long_exit = simulate_signal(
        opens, highs, lows, closes, 1, "long", 95.0, 103.0, 48, 1.0
    )["exit_idx"]
    frame = _signals_frame(
        [
            {
                "decision_idx": 0,
                "side": "long",
                "exit_idx": int(long_exit),
                "stop_price": 95.0,
                "target_price": 103.0,
                "atr_entry": 1.0,
            },
            {
                "decision_idx": 2,
                "side": "short",
                "exit_idx": 5,
                "stop_price": 104.0,
                "target_price": 98.0,
                "atr_entry": 1.0,
            },
        ]
    )
    replay = run_position_state_machine(
        frame,
        allow_reverse=True,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
    )
    closed = replay.iloc[0]
    reopened = replay.iloc[1]
    assert closed["force_exit_idx"] == 2.0
    assert reopened["status"] == "taken"
    # hand recompute: fill at bar 1 (100), truncated hold to bar 2.
    # Neither level is touched by bar 2 (high 102.9 < 103), so the
    # time exit fires at the slipped close of bar 2.
    px = 102.5 * (1 - 0.0005)
    denom = 5.0
    expected = (
        (px - 100.0) / denom
        - (2 * 0.001 + 0.0005) * 100.0 / denom
        - 2.0 * 0.0005 * 100.0 / denom
        - 0.0005 * px / denom
    )
    assert closed["force_close_r"] == pytest.approx(expected, abs=1e-12)
    # the stale isolated value (TP exit at bar 3) differs from the
    # recompute (time exit at bar 2)
    assert closed["exit_idx"] == long_exit  # untouched stale column
    stale = simulate_signal(
        opens, highs, lows, closes, 1, "long", 95.0, 103.0, 48, 1.0
    )["r_pess"]
    assert stale != pytest.approx(closed["force_close_r"])


def test_force_close_before_fill_is_zero_scratch() -> None:
    """A reverse decided before the open trade's fill: scratch 0.0."""
    opens, highs, lows, closes = _rally_world()
    frame = _signals_frame(
        [
            {
                "decision_idx": 1,
                "side": "long",
                "exit_idx": 4,
                "stop_price": 95.0,
                "target_price": 103.0,
            },
            {
                "decision_idx": 1,
                "side": "short",
                "exit_idx": 5,
                "stop_price": 104.0,
                "target_price": 98.0,
            },
        ]
    )
    replay = run_position_state_machine(
        frame,
        allow_reverse=True,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
    )
    # same decision bar: the long wins the bar (stable priority order),
    # the short force-closes BEFORE the long's fill (bar 2 > close 1?)
    # -> the long never existed at bar 1: scratch 0.0
    closed = replay.iloc[0]
    assert closed["force_exit_idx"] == 1.0
    assert closed["force_close_r"] == 0.0
    assert replay.iloc[1]["status"] == "taken"


def test_force_close_requires_ohlc_and_columns() -> None:
    """Reverse without price series / stop columns fails loudly."""
    opens, highs, lows, closes = _rally_world()
    frame = _signals_frame(
        [
            {
                "decision_idx": 0,
                "side": "long",
                "exit_idx": 4,
                "stop_price": 95.0,
                "target_price": 103.0,
            },
            {
                "decision_idx": 2,
                "side": "short",
                "exit_idx": 5,
                "stop_price": 104.0,
                "target_price": 98.0,
            },
        ]
    )
    with pytest.raises(ValueError, match="opens/highs/lows/closes"):
        run_position_state_machine(frame, allow_reverse=True)
    no_stop = frame.drop(columns=["stop_price"])
    with pytest.raises(ValueError, match="stop_price"):
        run_position_state_machine(
            no_stop,
            allow_reverse=True,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
        )


def test_cooldown_blocks_bars_after_exit() -> None:
    """cooldown=2 blocks the exit bar and the next decision bar."""
    frame = _signals_frame(
        [
            {"decision_idx": 0, "side": "long", "exit_idx": 3},
            {"decision_idx": 3, "side": "long", "exit_idx": 9},
            {"decision_idx": 4, "side": "long", "exit_idx": 9},
            {"decision_idx": 5, "side": "long", "exit_idx": 9},
        ]
    )
    replay = run_position_state_machine(frame, cooldown=2)
    assert replay["status"].tolist() == [
        "taken",
        "skipped",
        "skipped",
        "taken",
    ]
    assert replay["reason"].tolist() == [
        "",
        "same_bar_after_exit",
        "cooldown",
        "",
    ]


def test_same_bar_reentry_flag_opens_the_slot() -> None:
    """same_bar_reentry=True lets the exit-bar signal back in."""
    frame = _signals_frame(
        [
            {"decision_idx": 0, "side": "long", "exit_idx": 3},
            {"decision_idx": 3, "side": "long", "exit_idx": 9},
        ]
    )
    replay = run_position_state_machine(frame, same_bar_reentry=True)
    assert replay["status"].tolist() == ["taken", "taken"]
    assert replay["reason"].tolist() == ["", ""]


def test_cooldown_zero_matches_isolated_simulator() -> None:
    """cooldown=0: the very next decision bar after the exit enters."""
    frame = _signals_frame(
        [
            {"decision_idx": 0, "side": "long", "exit_idx": 3},
            {"decision_idx": 4, "side": "long", "exit_idx": 9},
        ]
    )
    replay = run_position_state_machine(frame)
    assert replay["status"].tolist() == ["taken", "taken"]


def test_priority_breaks_same_bar_ties_deterministically() -> None:
    """Same bar: highest priority wins; ties -> side, then input order."""
    rows = [
        {"decision_idx": 0, "side": "long", "exit_idx": 3, "priority": 0.3},
        {"decision_idx": 5, "side": "long", "exit_idx": 9, "priority": 0.7},
        {"decision_idx": 5, "side": "long", "exit_idx": 9, "priority": 0.9},
    ]
    replay = run_position_state_machine(_signals_frame(rows))
    assert replay.iloc[1]["status"] == "taken"  # the 0.9 candidate
    assert replay.iloc[2]["status"] == "skipped"
    # equal priority + equal side: FIRST input row wins (stable order)
    tied = _signals_frame(
        [
            {
                "decision_idx": 0,
                "side": "long",
                "exit_idx": 9,
                "priority": 0.5,
            },
            {
                "decision_idx": 0,
                "side": "long",
                "exit_idx": 9,
                "priority": 0.5,
            },
        ]
    )
    replay_tied = run_position_state_machine(tied)
    assert replay_tied["status"].tolist() == ["taken", "skipped"]
    assert replay_tied["reason"].tolist() == ["", "in_position"]
    # two runs agree bit-for-bit
    again = run_position_state_machine(_signals_frame(rows))
    assert again.equals(replay)


def test_side_breaks_equal_priority_same_bar() -> None:
    """Equal priority on one bar: 'long' sorts before 'short'."""
    rows = [
        {"decision_idx": 0, "side": "short", "exit_idx": 9, "priority": 1.0},
        {"decision_idx": 0, "side": "long", "exit_idx": 9, "priority": 1.0},
    ]
    replay = run_position_state_machine(_signals_frame(rows))
    # 'long' < 'short' alphabetically -> the long candidate is first
    # in processed order and takes the slot
    assert replay.iloc[0]["side"] == "long"
    assert replay.iloc[0]["status"] == "taken"
    assert replay.iloc[1]["status"] == "skipped"


def test_open_ended_trade_holds_slot_to_end() -> None:
    """exit_idx == -1 occupies the slot; later signals cannot enter."""
    rows = [
        {
            "decision_idx": 0,
            "side": "long",
            "exit_idx": -1,
            "stop_price": 95.0,
            "target_price": 103.0,
        },
        {"decision_idx": 2, "side": "long", "exit_idx": 9},
        {"decision_idx": 3, "side": "short", "exit_idx": 9},
    ]
    replay = run_position_state_machine(_signals_frame(rows))
    assert replay["status"].tolist() == ["taken", "skipped", "skipped"]
    assert replay["reason"].tolist() == [
        "",
        "data_ended_open",
        "data_ended_open",
    ]
    # allow_reverse cannot rescue an open-ended slot either
    reverse_replay = run_position_state_machine(
        _signals_frame(rows),
        allow_reverse=True,
        opens=np.asarray([100.0] * 6),
        highs=np.asarray([100.0] * 6),
        lows=np.asarray([100.0] * 6),
        closes=np.asarray([100.0] * 6),
    )
    assert reverse_replay["reason"].tolist() == [
        "",
        "data_ended_open",
        "data_ended_open",
    ]


def test_machine_validates_frame_contract() -> None:
    """Missing columns / bad values / generator rule violations raise."""
    good = _signals_frame(
        [{"decision_idx": 0, "side": "long", "exit_idx": 3}]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        run_position_state_machine(good.drop(columns=["side"]))
    bad_side = _signals_frame(
        [{"decision_idx": 0, "side": "up", "exit_idx": 3}]
    )
    with pytest.raises(ValueError, match="side must be"):
        run_position_state_machine(bad_side)
    nan_decision = _signals_frame(
        [{"decision_idx": float("nan"), "side": "long", "exit_idx": 3}]
    )
    with pytest.raises(ValueError, match="decision_idx must be an integer"):
        run_position_state_machine(nan_decision)
    # generator rule: exit must be >= decision_idx + 1 or exactly -1
    same_bar_exit = _signals_frame(
        [{"decision_idx": 3, "side": "long", "exit_idx": 3}]
    )
    with pytest.raises(ValueError, match="generator rule"):
        run_position_state_machine(same_bar_exit)
    negative_exit = _signals_frame(
        [{"decision_idx": 0, "side": "long", "exit_idx": -2}]
    )
    with pytest.raises(ValueError, match="exit_idx must be >= -1"):
        run_position_state_machine(negative_exit)
    nan_priority = _signals_frame(
        [
            {
                "decision_idx": 0,
                "side": "long",
                "exit_idx": 3,
                "priority": float("nan"),
            }
        ]
    )
    with pytest.raises(ValueError, match="priority must be a finite"):
        run_position_state_machine(nan_priority)
    with pytest.raises(ValueError, match="cooldown must be >= 0"):
        run_position_state_machine(good, cooldown=-1)


def test_machine_empty_frame_returns_empty_scaffold() -> None:
    """Empty signals frame: empty output with the full column set."""
    empty = pd.DataFrame(columns=["decision_idx", "side", "exit_idx"])
    replay = run_position_state_machine(empty)
    assert len(replay) == 0
    assert list(replay.columns) == [
        "decision_idx",
        "side",
        "exit_idx",
        "status",
        "reason",
        "force_exit_idx",
        "force_close_r",
    ]


def test_machine_does_not_mutate_input_frame() -> None:
    """The signal frame is read-only input."""
    frame = _signals_frame(
        [{"decision_idx": 0, "side": "long", "exit_idx": 3}]
    )
    before = frame.copy()
    run_position_state_machine(frame)
    assert frame.equals(before)


# ---------------------------------------------------------------------------
# 4. run_admission_policy: FCFS vs REPLACE on a capped slot stream
# ---------------------------------------------------------------------------
def _admission_stream() -> list[dict[str, Any]]:
    """4 trades, cap 2: FCFS keeps the first two and loses the best."""
    return [
        {
            "ts_entry": 0,
            "ts_exit": 30,
            "s": 0.6,
            "r_pess": 0.2,
            "candidate_id": 1,
        },
        {
            "ts_entry": 5,
            "ts_exit": 40,
            "s": 0.4,
            "r_pess": -0.3,
            "candidate_id": 2,
        },
        {
            "ts_entry": 10,
            "ts_exit": 50,
            "s": 0.9,
            "r_pess": 1.2,
            "candidate_id": 3,
        },
        {
            "ts_entry": 20,
            "ts_exit": 90,
            "s": 0.2,
            "r_pess": -0.1,
            "candidate_id": 4,
        },
    ]


def test_admission_fcfs_accepts_first_and_rejects_capacity() -> None:
    """cap=2 FCFS: first two trades enter, the rest get 'capacity'."""
    report = run_admission_policy(_admission_stream(), cap=2, policy="fcfs")
    assert report["policy"] == "fcfs" and report["cap"] == 2
    assert [trade["candidate_id"] for trade in report["accepted"]] == [1, 2]
    assert [trade["candidate_id"] for trade in report["rejected"]] == [3, 4]
    assert all(trade["reason"] == "capacity" for trade in report["rejected"])
    assert report["replaced"] == []
    summary = report["summary"]
    assert summary["n"] == 2 and summary["replaced"] == 0
    assert summary["ev"] == pytest.approx(-0.1)
    assert summary["dd"] == pytest.approx(0.3)


def test_admission_fcfs_loses_the_better_trade() -> None:
    """The executable FCFS finding, pinned.

    The BEST candidate arrives into a full slot and is rejected while
    its inferior peers keep theirs.
    """
    best_arrives_second = [
        {"ts_entry": 0, "ts_exit": 30, "s": 0.4, "r_pess": 0.1},
        {"ts_entry": 5, "ts_exit": 40, "s": 0.9, "r_pess": 1.0},
    ]
    fcfs = run_admission_policy(best_arrives_second, cap=1, policy="fcfs")
    assert [trade["s"] for trade in fcfs["accepted"]] == [0.4]
    assert [trade["s"] for trade in fcfs["rejected"]] == [0.9]
    accepted_mean = fcfs["accepted"][0]["s"]
    rejected_mean = fcfs["rejected"][0]["s"]
    assert rejected_mean > accepted_mean  # the measured finding, in R too
    assert fcfs["summary"]["ev"] == pytest.approx(0.1)
    replace_low = run_admission_policy(
        best_arrives_second, cap=1, policy="replace_low"
    )
    # REPLACE-low closes the worst open holder and takes the new one
    assert [trade["s"] for trade in replace_low["replaced"]] == [0.4]
    assert replace_low["accepted"][-1]["s"] == 0.9
    assert replace_low["summary"]["ev"] == pytest.approx(1.0)
    assert replace_low["summary"]["ev"] > fcfs["summary"]["ev"]


def test_admission_replace_low_vs_high_credit() -> None:
    """Displaced trade credits 0 under low, full r_pess under high."""
    stream = _admission_stream()
    low = run_admission_policy(stream, cap=2, policy="replace_low")
    high = run_admission_policy(stream, cap=2, policy="replace_high")
    assert [trade["candidate_id"] for trade in low["replaced"]] == [2]
    assert [trade["candidate_id"] for trade in high["replaced"]] == [2]
    # low: replaced contributes 0 -> equity [0.2, 0.2, 1.4]
    assert low["equity_curve"].tolist() == pytest.approx([0.2, 0.2, 1.4])
    assert low["summary"]["ev"] == pytest.approx(1.4)
    assert low["max_drawdown_r"] == 0.0
    # high: replaced keeps its full isolated r_pess (-0.3)
    assert high["equity_curve"].tolist() == pytest.approx([0.2, -0.1, 1.1])
    assert high["summary"]["ev"] == pytest.approx(1.1)
    assert high["max_drawdown_r"] == pytest.approx(0.3)
    # the two policies differ by exactly the replaced trade's r_pess
    assert low["summary"]["ev"] - high["summary"]["ev"] == pytest.approx(0.3)
    assert [trade["reason"] for trade in low["replaced"]] == ["replaced"]
    # the 4th candidate (score 0.2) cannot beat the worst open score
    assert low["rejected"][0]["candidate_id"] == 4
    assert low["rejected"][0]["reason"] == "inferior_score"
    assert high["rejected"][0]["reason"] == "inferior_score"


def test_admission_score_tie_breaks_by_candidate_id() -> None:
    """Equal scores displace the LOWER candidate_id, not the earliest.

    Deterministic across replays.
    """
    tied = [
        {
            "ts_entry": 0,
            "ts_exit": 100,
            "s": 0.5,
            "r_pess": 1.0,
            "candidate_id": 7,
        },
        {
            "ts_entry": 1,
            "ts_exit": 100,
            "s": 0.5,
            "r_pess": 2.0,
            "candidate_id": 3,
        },
        {
            "ts_entry": 2,
            "ts_exit": 100,
            "s": 0.9,
            "r_pess": 0.5,
            "candidate_id": 9,
        },
    ]
    first = run_admission_policy(tied, cap=2, policy="replace_low")
    second = run_admission_policy(tied, cap=2, policy="replace_low")
    # candidate_id 3 < 7 displaces the SECOND arrival, not the first
    assert [trade["candidate_id"] for trade in first["replaced"]] == [3]
    assert first["accepted"] == second["accepted"]  # bit-for-bit
    assert first["replaced"] == second["replaced"]
    assert first["rejected"] == second["rejected"]
    assert first["equity_curve"].tolist() == second["equity_curve"].tolist()
    # without candidate_id the tie falls back to arrival position
    bare = [
        {"ts_entry": 0, "ts_exit": 100, "s": 0.5, "r_pess": 1.0},
        {"ts_entry": 1, "ts_exit": 100, "s": 0.5, "r_pess": 2.0},
        {"ts_entry": 2, "ts_exit": 100, "s": 0.9, "r_pess": 0.5},
    ]
    fallback = run_admission_policy(bare, cap=2, policy="replace_low")
    assert fallback["replaced"][0]["r_pess"] == 1.0  # earliest arrival


def test_admission_validates_inputs_and_never_mutates() -> None:
    """Bad cap/policy/trades raise; input dicts are never touched."""
    stream = _admission_stream()
    with pytest.raises(ValueError, match="cap must be"):
        run_admission_policy(stream, cap=0)
    with pytest.raises(ValueError, match="policy must be one of"):
        run_admission_policy(stream, cap=2, policy="round_robin")
    with pytest.raises(ValueError, match="non-empty list"):
        run_admission_policy([], cap=2)
    with pytest.raises(ValueError, match="must be a dict"):
        run_admission_policy(["trade"], cap=2)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="missing required key"):
        run_admission_policy([{"ts_entry": 0}], cap=2)
    with pytest.raises(ValueError, match="ts_exit must be >= ts_entry"):
        run_admission_policy(
            [{"ts_entry": 10, "ts_exit": 5, "s": 0.5, "r_pess": 1.0}], cap=2
        )
    snapshot = [dict(trade) for trade in stream]
    run_admission_policy(stream, cap=2, policy="replace_high")
    assert stream == snapshot


def test_admission_single_trade_equity_and_drawdown() -> None:
    """One trade: equity is the single exit credit; dd of a win is 0."""
    report = run_admission_policy(
        [{"ts_entry": 0, "ts_exit": 7, "s": 0.5, "r_pess": 1.5}],
        cap=3,
        policy="fcfs",
    )
    assert report["equity_curve"].tolist() == pytest.approx([1.5])
    assert report["max_drawdown_r"] == 0.0
    assert report["final_equity_r"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 5. maker / market execution mechanics
# ---------------------------------------------------------------------------
def test_market_sim_matches_engine_baseline_formula() -> None:
    """market_sim = simulate_signal's pessimistic leg (donor claim)."""
    opens, highs, lows, closes = _zone_world()
    value = market_sim(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 1.0
    )
    engine = simulate_signal(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 48, 1.0
    )
    assert value == engine["r_pess"]
    assert value == pytest.approx(1.65)  # (102-100)/1 - 0.25 - 0.1


def test_maker_sim_better_entry_widens_risk() -> None:
    """Known geometry: limit 99.5 halves the risk -> ~5R gross win."""
    opens, highs, lows, closes = _zone_world()
    value = maker_sim(
        opens, highs, lows, closes, 2, "long", 99.0, 102.0, 99.5, 1.0
    )
    gross = (102.0 - 99.5) / 0.5
    cost_r = (0.001 * (1 + 0.3) + 0.0005) * 99.5 / 0.5
    entry_slip_r = 2 * 0.0005 * 99.5 / 0.5
    assert value == pytest.approx(gross - cost_r - entry_slip_r)
    assert value == pytest.approx(4.4428)
    market_r = market_sim(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 1.0
    )
    assert value > market_r  # the widened stop makes this win larger


def test_maker_fee_fraction_monotone_in_r() -> None:
    """R(maker_frac=0) - R(maker_frac=1) == comm * fill / risk."""
    opens, highs, lows, closes = _zone_world()
    free = maker_sim(
        opens,
        highs,
        lows,
        closes,
        2,
        "long",
        99.0,
        102.0,
        99.5,
        1.0,
        maker_frac=0.0,
    )
    taker = maker_sim(
        opens,
        highs,
        lows,
        closes,
        2,
        "long",
        99.0,
        102.0,
        99.5,
        1.0,
        maker_frac=1.0,
    )
    assert free - taker == pytest.approx(0.001 * 99.5 / 0.5)


def test_maker_sim_gap_through_is_scratch() -> None:
    """A limit filled beyond the stop scratches immediately."""
    opens, highs, lows, closes = _zone_world()
    value = maker_sim(
        opens, highs, lows, closes, 2, "long", 99.0, 102.0, 98.5, 1.0
    )
    denom = 0.5
    cost_r = (0.001 * (1 + 0.3) + 0.0005) * 98.5 / denom
    entry_slip_r = 2 * 0.0005 * 98.5 / denom
    xtr = (2.0 - 1.0) * 0.0005 * 98.5 / denom
    assert value == pytest.approx(-cost_r - entry_slip_r - xtr)


def test_maker_entry_sim_deterministic_and_zero_on_miss() -> None:
    """Same inputs replay bit-for-bit; a missed zone pays exactly 0."""
    opens, highs, lows, closes = _zone_world()
    first = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        100.0,
        1.0,
        delta_atr=0.5,
        wait_bars=4,
    )
    second = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        100.0,
        1.0,
        delta_atr=0.5,
        wait_bars=4,
    )
    assert first == second  # bit-for-bit, no seed needed
    assert first == pytest.approx(4.4428)  # filled at the bar-2 dip
    missed = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        100.0,
        1.0,
        delta_atr=0.5,
        wait_bars=1,
    )
    assert missed == 0.0  # maker-or-skip: missed signal == 0
    unfillable = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.96,
        102.0,
        100.0,
        1.0,
        delta_atr=0.03,
        wait_bars=4,
    )
    assert math.isnan(unfillable)  # zone closer than 5% ATR to the stop


def test_maker_entry_equals_market_at_zero_delta_full_fee() -> None:
    """delta=0 + maker_frac=1 reduces maker_entry_sim to market_sim."""
    opens, highs, lows, closes = _zone_world()
    maker = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        0,
        "long",
        99.0,
        102.0,
        100.0,
        1.0,
        delta_atr=0.0,
        maker_frac=1.0,
    )
    market = market_sim(
        opens, highs, lows, closes, 0, "long", 99.0, 102.0, 1.0
    )
    assert maker == pytest.approx(market, rel=1e-12)


def test_maker_sim_rejects_negative_maker_frac() -> None:
    """A negative maker fee fraction is a contract breach."""
    opens, highs, lows, closes = _zone_world()
    with pytest.raises(ValueError, match="maker_frac must be >= 0"):
        maker_sim(
            opens,
            highs,
            lows,
            closes,
            2,
            "long",
            99.0,
            102.0,
            99.5,
            1.0,
            maker_frac=-0.1,
        )
    with pytest.raises(ValueError, match="delta_atr must be >= 0"):
        maker_entry_sim(
            opens,
            highs,
            lows,
            closes,
            0,
            "long",
            99.0,
            102.0,
            100.0,
            1.0,
            delta_atr=-0.5,
        )


# ---------------------------------------------------------------------------
# 6. rank normalization at the admission seam
# ---------------------------------------------------------------------------
def test_rank_normalize_group_average_ties_and_edges() -> None:
    """Average ties; singleton -> 0.5; monotone; empty -> empty."""
    assert rank_normalize_group(np.array([3.0, 1.0, 2.0])).tolist() == [
        1.0,
        0.0,
        0.5,
    ]
    assert rank_normalize_group(np.array([5.0, 5.0, 1.0])).tolist() == [
        0.75,
        0.75,
        0.0,
    ]
    assert rank_normalize_group(np.array([7.0])).tolist() == [0.5]
    assert rank_normalize_group(np.zeros(4)).tolist() == [0.5, 0.5, 0.5, 0.5]
    empty = rank_normalize_group(np.zeros(0))
    assert empty.size == 0 and empty.dtype == np.float64
    ranks = rank_normalize_group(np.array([0.2, 0.9, 0.4, 0.1]))
    assert ranks.tolist() == [
        0.3333333333333333,
        1.0,
        0.6666666666666666,
        0.0,
    ]


def test_rank_normalize_group_rejects_bad_input() -> None:
    """Non-1-D and non-finite scores raise ValueError."""
    with pytest.raises(ValueError, match="scores must be 1-D"):
        rank_normalize_group(np.ones((2, 2)))
    with pytest.raises(ValueError, match="scores must be finite"):
        rank_normalize_group(np.array([0.5, float("nan")]))


# ---------------------------------------------------------------------------
# 7. per-trade metrics
# ---------------------------------------------------------------------------
def test_trade_curve_stats_known_answer() -> None:
    """Exact n / dd / pf on a hand-checked stream; NaN rows dropped."""
    stats_row = trade_curve_stats([1.0, -0.5, 2.0, -1.0])
    assert stats_row["n"] == 4
    assert stats_row["max_dd_r"] == pytest.approx(1.0)
    assert stats_row["profit_factor"] == pytest.approx(2.0)
    # identity with the per-trade Sharpe: t = sharpe * sqrt(n)
    assert stats_row["t_stat"] == pytest.approx(
        per_trade_sharpe([1.0, -0.5, 2.0, -1.0]) * 2.0
    )
    with_nan = trade_curve_stats([float("nan"), 1.0, -1.0])
    assert with_nan["n"] == 2
    assert with_nan["profit_factor"] == 1.0


def test_trade_curve_stats_degenerate_inputs() -> None:
    """House NaN policy: empty -> NaN stats; zero dispersion -> NaN t.

    The engine donor returned 0.0 scaffolds; the metric home
    (atropos.backtest_metrics) is deliberately honest-NaN instead.
    """
    empty = trade_curve_stats([])
    assert empty["n"] == 0
    assert math.isnan(empty["t_stat"])
    assert math.isnan(empty["max_dd_r"])
    assert math.isnan(empty["profit_factor"])
    winners = trade_curve_stats([0.5, 0.7])
    assert math.isinf(winners["profit_factor"]) and winners["n"] == 2
    constant = trade_curve_stats([0.7] * 4)
    assert math.isnan(constant["t_stat"])  # no honest t at dispersion 0
    assert constant["max_dd_r"] == 0.0  # dd stays real and computable
    single = trade_curve_stats([1.5])
    assert single["n"] == 1 and math.isnan(single["t_stat"])


def test_per_trade_sharpe_scale_and_guards() -> None:
    """mean/std (ddof=1); NaN below 2 trades or at zero dispersion."""
    assert per_trade_sharpe([1.0, 2.0, 3.0, 4.0]) == pytest.approx(
        1.9364916731037085
    )
    assert math.isnan(per_trade_sharpe([0.5]))
    assert math.isnan(per_trade_sharpe([]))
    assert math.isnan(per_trade_sharpe([3.0, 3.0]))
    # a zero MEAN with real dispersion is a real 0.0, not NaN
    assert per_trade_sharpe([1.0, -1.0] * 50) == pytest.approx(0.0)


def test_bucketed_sharpe_bucket_boundaries_and_guards() -> None:
    """Edge timestamps land in the right bucket; degenerate -> NaN."""
    day = 86_400_000
    same_bucket = bucketed_sharpe(
        [1.0, -1.0, 1.0, 1.0], [0, day, 2 * day, 7 * day - 1]
    )
    assert math.isnan(same_bucket)  # a single bucket has no dispersion
    crossed = bucketed_sharpe(
        [1.0, -1.0, 3.0, 1.0], [0, day, 2 * day, 7 * day]
    )
    assert crossed > 0.0  # the 4th trade opens bucket #2
    # hand value for bucket sums [3.0, 1.0]: mean 2, std ddof=1 sqrt(2)
    expected = 2.0 / math.sqrt(2.0) * math.sqrt(365.25 / 7)
    assert crossed == pytest.approx(expected, rel=1e-12)
    # interior empty buckets are present as zeros (bincount), not dropped
    sparse = bucketed_sharpe([1.0, 1.0], [0, 21 * day])
    sums_sparse = np.bincount(np.array([0, 3]))
    manual = (
        sums_sparse.mean() / sums_sparse.std(ddof=1) * math.sqrt(365.25 / 7)
    )
    assert sparse == pytest.approx(float(manual), rel=1e-12)
    assert math.isnan(bucketed_sharpe([0.5], [0]))  # single trade
    # one bucket, zero dispersion between... two same-sum buckets
    assert math.isnan(bucketed_sharpe([1.0, -1.0], [0, day]))
    with pytest.raises(ValueError, match="bucket_days"):
        bucketed_sharpe([1.0, 1.0], [0, day], bucket_days=0)
    # mismatched lengths align positionally down to the shorter input
    # (atropos._utils.align_inputs contract) -> 1 usable row -> NaN
    assert math.isnan(bucketed_sharpe([1.0, 1.0], [0]))


def test_pooled_stats_restores_chronology_before_dd() -> None:
    """The drawdown runs on timestamp order, not on input order."""
    day = 86_400_000
    # input order [win, loss] but timestamps put the loss FIRST
    pooled = pooled_stats(np.array([1.0, -1.0]), np.array([day, 0.0]))
    assert pooled["mean"] == pytest.approx(0.0)
    assert pooled["dd"] == 0.0  # chronological [-1, +1] never draws down
    assert pooled["n"] == 2
    flipped = pooled_stats(np.array([-1.0, 1.0]), np.array([day, 0.0]))
    assert flipped["dd"] == pytest.approx(1.0)  # chronological [+1, -1]
    keys = set(pooled.keys())
    assert keys == {
        "mean",
        "n",
        "dd",
        "t_stat_naive",
        "sharpe_per_trade",
        "sharpe_ann_bucketed",
    }
    # mismatched lengths align down to the shorter side (align_inputs):
    # 1 usable row -> no honest t-stat (NaN), dd computable
    ragged = pooled_stats(np.array([1.0]), np.array([0.0, 1.0]))
    assert ragged["n"] == 1
    assert ragged["mean"] == pytest.approx(1.0)
    assert math.isnan(ragged["t_stat_naive"])
    assert ragged["dd"] == 0.0


# ---------------------------------------------------------------------------
# 8. cost model
# ---------------------------------------------------------------------------
def test_cost_model_validation() -> None:
    """Defaults match the engine constants; bad fields raise."""
    assert DEFAULT_COST.gen_slip == 0.0005
    assert DEFAULT_COST.comm == 0.001
    assert DEFAULT_COST.gap == 0.25
    assert DEFAULT_COST.e_mult == 2.0
    assert DEFAULT_COST.x_mult == 2.0
    for field in ("gen_slip", "comm", "gap", "e_mult"):
        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            CostModel(**{field: -0.1})
    with pytest.raises(ValueError, match="x_mult must be >= 1"):
        CostModel(x_mult=0.5)
    with pytest.raises(ValueError, match="must be a finite number"):
        CostModel(comm=float("nan"))
