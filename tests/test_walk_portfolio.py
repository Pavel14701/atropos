"""Walk-portfolio pins: phases, netting, priority, caps, exposure.

The ORDERING PIN (review refinement 1): every bar resolves
exit-resolution -> slot-release -> netting -> entry-competition.
Netting precedes priority (refinement 2); the tuple declaration
order is the tiebreak (refinement 3); displacement does NOT exist
(negative pin).  Each test names its mutation (applied -> fail ->
restored before commit).
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pytest
from atropos.book import Book, _demo_md
from atropos.data_contract import MaterializedData
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    SizingSpec,
    StopSpec,
)

_BARS = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")


def _held_md() -> MaterializedData:
    """A grid where a 20-ATR-stop long holds while shorts fire.

    mom = [-1, 1, 1, 1, 1, -1, -1, -1]: the long fires bar 1 and
    holds (stop ~81 never touched); the short gate fires bar 5+.
    """
    features = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0, 100.0, 100.5, 100.5, 100.5, 100.0, 99.5, 99.0],
            "mom": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            "atr": [0.5] * 8,
        },
        index=_BARS,
    )
    return MaterializedData(
        outcome=pd.Series(0.0, index=_BARS),
        features=features,
        controls=pd.DataFrame(index=_BARS),
        index=pd.DataFrame(
            {"ts": _BARS, "entity": "", "split": "train"}, index=_BARS
        ),
    )


def _leg(
    name: str,
    direction: Literal["long", "short"],
    op: Literal[">", "<"],
    *,
    priority: float = 0.0,
    size_r: float = 1.0,
) -> LegSpec:
    """A 20-ATR-stop leg (holds through the whole grid)."""
    return LegSpec(
        name=name,
        direction=direction,
        entry=EntrySpec(Compare("mom", op, 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 20.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        sizing=SizingSpec(mode="fixed", param=size_r),
        features_used=("mom", "atr", "close"),
        priority=priority,
        rationale="walk pin fixture",
    )


def _walk(
    legs: tuple[LegSpec, ...] | list[LegSpec],
    combination: str,
    *,
    cap: int = 2,
    md: MaterializedData | None = None,
) -> dict[str, Any]:
    """The demo carry policy + the pinned grid."""
    book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    return book.walk_portfolio(
        legs,
        md if md is not None else _held_md(),
        combination=combination,
        max_concurrent_positions=cap,
    )


def test_netting_close_all_and_remainder() -> None:
    """An opposite entry nets EVERY position on the instrument.

    World: long 1R holds; short 2R fires bar 5 -> the long closes
    "netted", remainder = max(2 - 1, 0) = 1 (signal loss recorded),
    the short intent consumed, warning collected.  Mutation facts:
    closing only the oldest (not all), dropping the clamp (a
    reduction would warn), dropping the remainder field, or
    suppressing the warning each break a distinct assert.
    """
    legs = (
        _leg("long_a", "long", ">", size_r=1.0),
        _leg("short_b", "short", "<", size_r=2.0),
    )
    result = _walk(legs, "netting")
    netted = [
        record
        for leg_records in result["records"].values()
        for record in leg_records
        if record["exit_reason"] == "netted"
    ]
    assert len(netted) == 1
    assert netted[0]["leg_name"] == "long_a"
    assert netted[0]["netted_remainder"] == pytest.approx(1.0)
    assert netted[0]["diagnostics"]["netted_remainder_dropped"] is True
    # the colon anchors the FULL prefix (a "dropped-silent" relabel
    # must fail this pin, not substring-sneak through)
    assert any(
        "netted remainder dropped:" in warning
        for warning in result["warnings"]
    )


def test_netted_reduction_no_warning() -> None:
    """A reduction (new < old) is not a loss: no remainder warning.

    World: long 2R holds; short 1R nets it -> remainder = max(1 - 2,
    0) = 0 (clamped), the reduction recorded in diagnostics, and NO
    warning.  Mutation fact: removing the max(..., 0) clamp makes
    netted_remainder -1.0 and fires a false warning -- both asserts
    fail.
    """
    legs = (
        _leg("long_big", "long", ">", size_r=2.0),
        _leg("short_small", "short", "<", size_r=1.0),
    )
    result = _walk(legs, "netting")
    netted = [
        record
        for leg_records in result["records"].values()
        for record in leg_records
        if record["exit_reason"] == "netted"
    ]
    assert len(netted) == 1
    assert netted[0]["netted_remainder"] == 0.0
    assert not any(
        "netted remainder dropped" in warning
        for warning in result["warnings"]
    )


def test_netting_closes_all_positions() -> None:
    """Two longs + one short-entry: BOTH longs close (close-all).

    Mutation fact: closing only the oldest position leaves the
    second long open and the count assert fails.
    """
    legs = (
        _leg("long_a", "long", ">", size_r=1.0),
        _leg("long_b", "long", ">", size_r=0.5),
        _leg("short_c", "short", "<", size_r=2.0),
    )
    result = _walk(legs, "netting", cap=3)
    netted = [
        record
        for leg_records in result["records"].values()
        for record in leg_records
        if record["exit_reason"] == "netted"
    ]
    assert len(netted) == 2
    assert {record["leg_name"] for record in netted} == {"long_a", "long_b"}
    # remainder = max(2 - 1.5, 0) = 0.5 -> warned
    assert all(
        record["netted_remainder"] == pytest.approx(0.5) for record in netted
    )


def test_netting_mode_ignores_priority() -> None:
    """Netting walks in DECLARATION order; priorities are inert.

    World: cap 1, two same-direction legs firing the SAME bar with
    unequal priorities -- competition for the one slot is exactly
    where priority ordering WOULD show if netting mode applied it.
    Declaration order admits z_declared_first; a priority-sorted
    netting walk admits high_prio instead.  Mutation fact: applying
    the (-priority, decl) sort inside netting mode flips the
    admitted name and fails the assert.
    """
    md = _held_md()
    legs = (
        _leg("z_declared_first", "long", ">", priority=1.0),
        _leg("high_prio", "long", ">", priority=9.0),
    )
    result = _walk(legs, "netting", cap=1, md=md)
    assert result["records"]["z_declared_first"] != []
    assert result["records"]["high_prio"] == []


def test_slot_after_exit_not_before() -> None:
    """A slot freed on bar b admits on bar b (Phase A before Phase D).

    World: cap 1, leg A's stop exits at bar 4, leg B's gate fires
    bar 4 -> B enters bar 4 AFTER A's exit resolved.  Mutation fact:
    running entries BEFORE exits (or freezing the cap check at bar
    start) makes B skip bar 4 and the entry-bar assert fails.
    """
    features = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 89.0, 89.0, 89.0, 89.0],
            "close": [100.0] * 8,
            "mom": [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            "atr": [0.5] * 8,
        },
        index=_BARS,
    )
    md = MaterializedData(
        outcome=pd.Series(0.0, index=_BARS),
        features=features,
        controls=pd.DataFrame(index=_BARS),
        index=pd.DataFrame(
            {"ts": _BARS, "entity": "", "split": "train"}, index=_BARS
        ),
    )
    leg_exit = _leg("exits_at_4", "long", ">", size_r=1.0)
    leg_exit = LegSpec(
        name="exits_at_4",
        direction="long",
        entry=EntrySpec(
            Compare("mom", ">", 0.0), cooldown=pd.Timedelta(days=0)
        ),
        exit=ExitSpec(stop=StopSpec("price", 89.5)),
        instrument=InstrumentRef("field", {"field": "close"}),
        sizing=SizingSpec(mode="fixed", param=1.0),
        features_used=("mom", "close"),
        rationale="exits bar 4 through the tight stop",
    )
    leg_enter = _leg("enters_at_4", "long", ">", size_r=1.0)
    result = _walk((leg_exit, leg_enter), "sequential", cap=1, md=md)
    exits_records = result["records"]["exits_at_4"]
    enters_records = result["records"]["enters_at_4"]
    assert exits_records != []
    assert exits_records[0]["exit_bar"] == 4
    assert enters_records != []
    # admitted on bar 4 (the freed-slot bar), FILLED on bar 5 (the
    # next bar's open -- the house decision_idx+1 convention)
    assert enters_records[0]["entry_bar"] == 5


def test_tiebreak_is_declaration_order() -> None:
    """Equal priorities: the tuple declaration order wins.

    Mutation fact: sorting by NAME (alphabetical) admits b_first
    over a_second on the cap-1 bar and fails the admitted-name
    assert.  records MUST still carry every leg (empty lists for
    the skipped ones -- the run_book contract).
    """
    md = _held_md()
    # names chosen so the NAME sort contradicts the DECLARATION sort:
    # z_first is declared FIRST (tuple position 0), a_second SECOND
    z_first = _leg("z_first", "long", ">")  # declared FIRST
    a_second = _leg("a_second", "long", ">")  # declared SECOND
    result = _walk((z_first, a_second), "priority", cap=1, md=md)
    # both fire bar 0 (equal priority 0.0); cap 1 -> the FIRST
    # DECLARED leg (z_first) wins, and records still carries the
    # skipped leg (empty list).  A name-based tiebreak would admit
    # a_second instead -- the exact mutation this pin kills.
    assert sorted(result["records"]) == ["a_second", "z_first"]
    assert result["records"]["z_first"] != []
    assert result["records"]["a_second"] == []


def test_no_displacement_in_priority() -> None:
    """A higher-priority entry NEVER closes a lower-priority holder.

    World: cap 2; BOTH legs fire bar 0 and hold to the end -- the
    higher-priority admission does not close, net, or displace the
    lower-priority holder.  Mutation fact: an implementation that
    displaces (or nets) the holder produces a netted/closed leg_low
    record -- the exit_reason assert fails.
    """
    legs = (
        _leg("leg_low", "long", ">", priority=1.0),
        _leg("leg_high", "long", ">", priority=9.0),
    )
    result = _walk(legs, "priority", cap=2)
    assert result["records"]["leg_low"] != []
    assert result["records"]["leg_high"] != []
    for leg_records in result["records"].values():
        assert all(
            record["exit_reason"] == "forced" for record in leg_records
        )
    assert result["exposure"]["net"][-1] == pytest.approx(2.0)


def test_sequential_ignores_priority() -> None:
    """Sequential walks declaration order even when priorities say otherwise.

    Mutation fact: applying the priority sort inside sequential mode
    admits b_high over a_low and fails the admitted-name assert.
    """
    md = _demo_md()
    a_low = _leg("a_low", "long", ">", priority=1.0)
    b_high = _leg("b_high", "long", ">", priority=9.0)
    result = _walk((a_low, b_high), "sequential", cap=1, md=md)
    assert result["records"]["a_low"] != []
    assert result["records"]["b_high"] == []


def test_priority_ordering_admits_high_first() -> None:
    """Priority mode admits the higher-priority leg on the contested bar.

    Mutation fact: dropping the -priority sort (declaration order)
    admits a_low instead of b_high and fails the assert.
    """
    md = _demo_md()
    a_low = _leg("a_low", "long", ">", priority=1.0)
    b_high = _leg("b_high", "long", ">", priority=9.0)
    result = _walk((b_high, a_low), "priority", cap=1, md=md)
    assert result["records"]["b_high"] != []
    assert result["records"]["a_low"] == []


def test_exposure_series_shapes_and_values() -> None:
    """concurrent/net/equity_curve are per-bar and direction-signed.

    Mutation fact: dropping the direction*size_r sign from the net
    series (unsigned exposure) breaks the net-value assert.
    """
    md = _held_md()
    legs = (
        _leg("long_hold", "long", ">", size_r=2.0),
        _leg("short_fire", "short", "<", size_r=1.0),
    )
    result = _walk(legs, "sequential", cap=2, md=md)
    exposure = result["exposure"]
    n_bars = 8
    assert len(exposure["concurrent"]) == n_bars
    assert len(exposure["net"]) == n_bars
    assert len(result["equity_curve"]) == n_bars
    # FILL-BAR accounting: the 2R long fills bar 1 (decision bar 0),
    # the 1R short fills bar 6 (decision bar 5)
    assert exposure["net"][0] == pytest.approx(0.0)
    assert exposure["net"][2] == pytest.approx(2.0)
    # bars 6-7: the long holds AND the short filled -> net = +2 - 1
    assert exposure["net"][6] == pytest.approx(1.0)
    assert result["equity_r"] == pytest.approx(result["equity_curve"][-1])


def test_walk_validation_ladder() -> None:
    """Empty/duplicate legs, unknown combinations, cap < 1 reject.

    Mutation fact: relaxing any guard accepts garbage silently.
    """
    md = _held_md()
    book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    with pytest.raises(ValueError, match="at least one leg"):
        book.walk_portfolio(
            [], md, combination="sequential", max_concurrent_positions=1
        )
    with pytest.raises(ValueError, match="duplicate"):
        book.walk_portfolio(
            (_leg("dup", "long", ">"), _leg("dup", "short", "<")),
            md,
            combination="sequential",
            max_concurrent_positions=1,
        )
    with pytest.raises(ValueError, match="independent"):
        book.walk_portfolio(
            (_leg("solo", "long", ">"),),
            md,
            combination="independent",
            max_concurrent_positions=1,
        )
    with pytest.raises(ValueError, match="combination"):
        book.walk_portfolio(
            (_leg("solo", "long", ">"),),
            md,
            combination="random",
            max_concurrent_positions=1,
        )
    with pytest.raises(ValueError, match="max_concurrent_positions"):
        book.walk_portfolio(
            (_leg("solo", "long", ">"),),
            md,
            combination="sequential",
            max_concurrent_positions=0,
        )


def test_sequential_opposites_coexist() -> None:
    """Sequential/priority have NO offsetting: opposites both hold.

    A signal-exit leg ALSO walks here: it must exit BY SIGNAL inside
    the shared walk (bar 5, where mom flips negative) -- a
    Phase-A-ignoring mutation turns that into a forced close at the
    last bar.  Mutation fact: applying the netting rule inside
    sequential mode closes one side and breaks the both-hold assert;
    ignoring the signal exit breaks the signal-reason assert.
    """
    md = _held_md()
    signal_leg = LegSpec(
        name="signal_leg",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(
            stop=StopSpec("atr", 20.0), signal=Compare("mom", "<", 0.0)
        ),
        instrument=InstrumentRef("field", {"field": "close"}),
        sizing=SizingSpec(mode="fixed", param=1.0),
        features_used=("mom", "atr", "close"),
        rationale="signal exit inside the shared walk",
    )
    legs = (
        _leg("long_side", "long", ">", size_r=2.0),
        _leg("short_side", "short", "<", size_r=1.0),
        signal_leg,
    )
    result = _walk(legs, "sequential", cap=3, md=md)
    long_trades = result["records"]["long_side"]
    short_trades = result["records"]["short_side"]
    signal_trades = result["records"]["signal_leg"]
    assert long_trades != []
    assert short_trades != []
    assert signal_trades != []
    assert any(record["exit_reason"] == "signal" for record in signal_trades)
    assert all(
        record["exit_reason"] != "netted"
        for record in long_trades + short_trades
    )
