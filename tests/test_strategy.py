"""Strategy-layer pins: composition, isolation, attribution, bridge.

Each test names its mutation (applied -> fail -> restored before
commit).
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pytest
from atropos.book import _demo_md
from atropos.data_contract import MaterializedData
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    StopSpec,
)
from atropos.strategy import (
    PortfolioPolicy,
    StrategySpec,
    UniverseSpec,
    _daily_ev,
    records_to_features,
    run_strategy,
)


def _policy() -> PortfolioPolicy:
    """The demo carry policy (declared, never defaulted).

    ``periods_override=252.0`` pins the explicit path: the demo
    bundle is manifest-less, and the carry accrual refuses to guess
    a calendar (loud, by design).
    """
    return PortfolioPolicy(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )


def _held_md() -> MaterializedData:
    """The 8-bar world where a 20-ATR long holds while shorts fire."""
    bars = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0, 100.0, 100.5, 100.5, 100.5, 100.0, 99.5, 99.0],
            "mom": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            "atr": [0.5] * 8,
        },
        index=bars,
    )
    return MaterializedData(
        outcome=pd.Series(0.0, index=bars),
        features=features,
        controls=pd.DataFrame(index=bars),
        index=pd.DataFrame(
            {"ts": bars, "entity": "", "split": "train"}, index=bars
        ),
    )


def _leg(name: str, op: str, *, stop_multiple: float = 2.0) -> LegSpec:
    """A demo-geometry leg with a configurable entry gate."""
    gate_op: Literal[">", "<"] = ">" if op == ">" else "<"
    return LegSpec(
        name=name,
        direction="long" if op == ">" else "short",
        entry=EntrySpec(Compare("mom", gate_op, 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", stop_multiple)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="strategy pin fixture",
    )


def test_independent_composition_and_isolation() -> None:
    """Two opposite-gate legs compose; neither touches the other.

    independent means positions COEXIST (NOT netting) -- here the
    two legs see disjoint entry bars and both trade.  Mutation fact:
    a shared Book/state across legs (or cross-leg signal leakage)
    changes one leg's trades and breaks the isolation asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(
        name="mirror",
        legs=(_leg("long_gate", ">"), _leg("short_gate", "<")),
        portfolio=_policy(),
        universe=UniverseSpec(symbols=("TEST",)),
    )
    result = run_strategy(strategy, md)
    summary = result["summary"]
    assert summary["n_legs"] == 2
    assert summary["combination"] == "independent"
    assert summary["leg_isolation_check"] is True
    assert sorted(result["records"]) == ["long_gate", "short_gate"]
    for leg_name, records in result["records"].items():
        for record in records:
            assert record["leg_name"] == leg_name
    long_entries = {
        record["entry_bar"] for record in result["records"]["long_gate"]
    }
    short_entries = {
        record["entry_bar"] for record in result["records"]["short_gate"]
    }
    assert not long_entries & short_entries
    assert summary["equity_r"] == result["equity_r"]


def test_all_four_combinations_accepted() -> None:
    """The four wave-3 combinations construct; unknown rejects (W3-3).

    Mutation fact: dropping the unknown-combination guard accepts
    "random" silently and the pytest.raises fails.
    """
    for mode in ("independent", "netting", "sequential", "priority"):
        policy = PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
            combination=mode,
        )
        assert policy.combination == mode
    with pytest.raises(ValueError, match="combination"):
        PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
            combination="random",
        )
    policy = _policy()
    assert policy.combination == "independent"


def test_leg_attribution_and_recount() -> None:
    """Every record is attributed and counts reconcile with per_leg.

    Mutation fact: losing the leg_name attribution (or a per_leg
    recount drift) breaks the attribution/recount asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(
        name="solo",
        legs=(_leg("solo", ">"),),
        portfolio=_policy(),
    )
    result = run_strategy(strategy, md)
    records = result["records"]["solo"]
    summary = result["summary"]
    assert summary["n_trades"] == len(records)
    assert summary["per_leg"]["solo"]["n_trades"] == len(records)
    expected_equity = sum(record["pnl_net_r"] for record in records)
    assert result["equity_r"] == pytest.approx(expected_equity)
    assert any("deferred" in note for note in summary["policy_notes"])
    with pytest.raises(ValueError, match="unique"):
        StrategySpec(
            name="dup",
            legs=(_leg("same", ">"), _leg("same", "<")),
            portfolio=_policy(),
        )


def test_records_to_features_bridge() -> None:
    """The TradeRecord stream flattens into an analysis-ready frame.

    This is the bridge from "what the strategy did" to "what Atropos
    analyzes": scalar record fields become columns, the entry feature
    snapshot expands with the "entry::" prefix.  Mutation fact:
    dropping the entry:: expansion (or the dict-valued keys) breaks
    the column asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(
        name="bridge",
        legs=(_leg("b1", ">"),),
        portfolio=_policy(),
    )
    result = run_strategy(strategy, md)
    records = result["records"]["b1"]
    assert records != []
    frame = records_to_features(records)
    assert len(frame) == len(records)
    for column in ("leg_name", "entry_ts", "pnl_net_r", "exit_reason"):
        assert column in frame.columns
    snapshot_keys = set(records[0]["features_at_entry"])
    expanded = {f"entry::{key}" for key in snapshot_keys}
    assert expanded <= set(frame.columns)
    # the expansion carries the SNAPSHOT VALUES, not just columns
    for key in snapshot_keys:
        expected = records[0]["features_at_entry"][key]
        assert frame["entry::" + key].iloc[0] == expected
    # dict-valued keys never leak raw dicts into the frame
    for column in frame.columns:
        assert not isinstance(frame[column].iloc[0], dict)
    empty = records_to_features([])
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# W3-3 -- dispatch, gates, calendar-day daily_ev
# ---------------------------------------------------------------------------
def test_dispatch_shared_modes_and_unified_contract() -> None:
    """All three shared modes return the UNIFIED result contract.

    Mutation fact: a dispatch that loses a combination (routing it
    to run_book) still returns records -- the distinguishing pin is
    exposure/equity_curve PRESENCE plus the warnings forwarding.
    """
    md = _held_md()
    legs = (
        _leg("long_side", ">", stop_multiple=20.0),
        _leg("short_side", "<", stop_multiple=20.0),
    )
    for mode in ("sequential", "priority", "netting"):
        strategy = StrategySpec(
            name=f"s_{mode}",
            legs=legs,
            portfolio=PortfolioPolicy(
                borrow_bps_annual=500.0,
                funding_bps_annual=100.0,
                periods_override=252.0,
                combination=mode,
            ),
            universe=UniverseSpec(symbols=("TEST",)),
        )
        result = run_strategy(strategy, md)
        for key in (
            "records",
            "equity_r",
            "per_leg",
            "exposure",
            "equity_curve",
            "policy_notes",
            "gates",
            "summary",
        ):
            assert key in result, (mode, key)
        assert len(result["exposure"]["net"]) == len(md.index)
        assert len(result["equity_curve"]) == len(md.index)
        assert result["gates"] is None  # gates=False default
        if mode == "netting":
            netted = [
                record
                for leg_records in result["records"].values()
                for record in leg_records
                if record["exit_reason"] == "netted"
            ]
            assert netted != []  # the short side nets the held long


def test_gates_wiring_and_unverified_note() -> None:
    """gates=True runs portfolio_gates and marks the report UNVERIFIED.

    Mutation fact: dropping the portfolio_gates call (or the
    UNVERIFIED note) breaks the GateReport/note asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(
        name="gated",
        legs=(_leg("g1", ">"),),
        portfolio=PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
        ),
        universe=UniverseSpec(symbols=("TEST",)),
    )
    result = run_strategy(strategy, md, gates=True)
    gates = result["gates"]
    assert gates is not None
    assert gates["prereg_hash"] is None  # UNVERIFIED by construction
    assert any("UNVERIFIED" in note for note in result["policy_notes"])


def test_daily_ev_is_calendar_day() -> None:
    """daily_ev buckets by DAY, not by bar (the W3-3 pin).

    Two trades exiting the same calendar day collapse into ONE
    daily_ev entry.  Mutation fact: per-bar bucketing produces two
    entries and fails the length assert.
    """
    day = pd.Timestamp("2024-01-05", tz="UTC")
    same_day_other_hour = day + pd.Timedelta(hours=5)
    records = {
        "leg": [
            {
                "leg_name": "leg",
                "entry_ts": day - pd.Timedelta(days=1),
                "exit_ts": day,
                "direction": 1,
                "instrument_id": "close",
                "entry_bar": 0,
                "exit_bar": 1,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "size_r": 1.0,
                "pnl_gross_r": 1.5,
                "pnl_net_r": 1.0,
                "costs_r": 0.5,
                "exit_reason": "target",
                "features_at_entry": {},
                "features_at_exit": {},
                "diagnostics": {},
            },
            {
                "leg_name": "leg",
                "entry_ts": day,
                "exit_ts": same_day_other_hour,
                "direction": 1,
                "instrument_id": "close",
                "entry_bar": 1,
                "exit_bar": 2,
                "entry_price": 101.0,
                "exit_price": 102.0,
                "size_r": 1.0,
                "pnl_gross_r": 0.8,
                "pnl_net_r": 0.5,
                "costs_r": 0.3,
                "exit_reason": "target",
                "features_at_entry": {},
                "features_at_exit": {},
                "diagnostics": {},
            },
        ]
    }
    typed_records: dict[str, list[dict[str, Any]]] = {
        "leg": [
            {
                "leg_name": record["leg_name"],
                "entry_ts": record["entry_ts"],
                "exit_ts": record["exit_ts"],
                "direction": record["direction"],
                "instrument_id": record["instrument_id"],
                "entry_bar": record["entry_bar"],
                "exit_bar": record["exit_bar"],
                "entry_price": record["entry_price"],
                "exit_price": record["exit_price"],
                "size_r": record["size_r"],
                "pnl_gross_r": record["pnl_gross_r"],
                "pnl_net_r": record["pnl_net_r"],
                "costs_r": record["costs_r"],
                "exit_reason": record["exit_reason"],
                "features_at_entry": record["features_at_entry"],
                "features_at_exit": record["features_at_exit"],
                "diagnostics": record["diagnostics"],
            }
            for record in records["leg"]
        ]
    }
    ev = _daily_ev(typed_records)
    assert len(ev) == 1  # same calendar day -> ONE bucket
    assert ev.iloc[0] == pytest.approx(1.0 + 0.5)
