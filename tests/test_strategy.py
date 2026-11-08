"""Strategy-layer pins: composition, isolation, attribution, bridge.

Each test names its mutation (applied -> fail -> restored before
commit).
"""

from __future__ import annotations

import pytest
from atropos.book import _demo_md
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
    records_to_features,
    run_strategy,
)


def _leg(name: str, op: str) -> LegSpec:
    """A demo-geometry leg with a configurable entry gate."""
    return LegSpec(
        name=name,
        direction="long" if op == ">" else "short",
        entry=EntrySpec(Compare("mom", op, 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 2.0)),
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


def test_deferred_combinations_rejected() -> None:
    """netting/sequential/priority are loud deferrals (C6 mutation).

    Mutation fact: accepting any deferred combination makes the first
    pytest.raises fail.
    """
    for mode in ("netting", "sequential", "priority"):
        with pytest.raises(ValueError, match="section 14"):
            PortfolioPolicy(combination=mode)  # type: ignore[arg-type]
    policy = PortfolioPolicy()
    assert policy.combination == "independent"
    assert "NOT netting" in (PortfolioPolicy.__doc__ or "")


def test_leg_attribution_and_recount() -> None:
    """Every record is attributed and counts reconcile with per_leg.

    Mutation fact: losing the leg_name attribution (or a per_leg
    recount drift) breaks the attribution/recount asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(name="solo", legs=(_leg("solo", ">"),))
    result = run_strategy(strategy, md)
    records = result["records"]["solo"]
    summary = result["summary"]
    assert summary["n_trades"] == len(records)
    assert summary["per_leg"]["solo"]["n_trades"] == len(records)
    expected_equity = sum(record["pnl_net_r"] for record in records)
    assert result["equity_r"] == pytest.approx(expected_equity)
    assert any("deferred" in note for note in summary["policy_notes"])
    with pytest.raises(ValueError, match="unique"):
        StrategySpec(name="dup", legs=(_leg("same", ">"), _leg("same", "<")))


def test_records_to_features_bridge() -> None:
    """The TradeRecord stream flattens into an analysis-ready frame.

    This is the bridge from "what the strategy did" to "what Atropos
    analyzes": scalar record fields become columns, the entry feature
    snapshot expands with the "entry::" prefix.  Mutation fact:
    dropping the entry:: expansion (or the dict-valued keys) breaks
    the column asserts.
    """
    md = _demo_md()
    strategy = StrategySpec(name="bridge", legs=(_leg("b1", ">"),))
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
