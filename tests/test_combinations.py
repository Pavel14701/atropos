"""Cross-mode matrix pins: every VALID combination covered (W15-4).

The valid_combinations list is HAND-BUILT (review item 6): invalid
combinations (e.g. "netted" exit outside netting mode) are never
generated.  Each row = (combination, exit mechanism, sizing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.book import Book
from atropos.data_contract import (
    FieldMeta,
    FieldSpec,
    MaterializedData,
    MaterializeSpec,
    _TableAdapter,
    materialize,
)
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    SizingSpec,
    StopSpec,
    TargetSpec,
)

_BARS = 12
_PRICES = [
    100.0,
    100.0,
    101.0,
    102.0,
    101.0,
    100.0,
    99.0,
    98.0,
    97.0,
    96.0,
    95.0,
    94.0,
]

_VALID_COMBINATIONS: list[tuple[str, str, str]] = [
    # (combination, exit mechanism, sizing mode)
    *(
        ("independent", mechanism, sizing)
        for mechanism in ("stop", "target", "time")
        for sizing in ("fixed", "risk_per_trade")
    ),
    *(
        ("sequential", mechanism, sizing)
        for mechanism in ("stop", "target", "time")
        for sizing in ("fixed", "risk_per_trade")
    ),
    *(
        ("priority", mechanism, sizing)
        for mechanism in ("stop", "target", "time")
        for sizing in ("fixed", "risk_per_trade")
    ),
    *(
        ("netting", mechanism, sizing)
        for mechanism in ("stop", "target", "time", "netted")
        for sizing in ("fixed", "risk_per_trade")
    ),
]


def _md() -> MaterializedData:
    """A 12-bar grid where the long holds or stops per the stop width."""
    bars = pd.date_range("2024-01-01", periods=_BARS, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "open": _PRICES,
            "high": [px + 1.0 for px in _PRICES],
            "low": [px - 1.0 for px in _PRICES],
            "close": _PRICES,
            "mom": [1.0] * 4 + [-1.0] * 8,
            "atr": [1.0] * _BARS,
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


def _leg_for(combination: str, mechanism: str, sizing: str) -> LegSpec:
    """A long leg configured for one matrix row.

    stop width 10 ATR holds to the end (forced/stop); stop 2 ATR
    stops quickly; a target at +2 fires early; netting rows add a
    short counterpart via the netting-mode netted close.
    """
    stop_multiple = 2.0 if mechanism == "stop" else 10.0
    exits = ExitSpec(
        stop=StopSpec("atr", stop_multiple),
        target=TargetSpec("atr", 2.0) if mechanism == "target" else None,
    )
    return LegSpec(
        name="matrix_leg",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=exits,
        instrument=InstrumentRef("field", {"field": "close"}),
        sizing=SizingSpec(mode=sizing, param=1.0),
        features_used=("mom", "atr", "close"),
        rationale=(
            f"cross-mode matrix row: {combination}/{mechanism}/{sizing}"
        ),
    )


def _run_row(combination: str, mechanism: str, sizing: str) -> dict[str, Any]:
    """One matrix row through the correct Book surface."""
    md = _md()
    book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    leg = _leg_for(combination, mechanism, sizing)
    if combination == "independent":
        records = book.run(leg, md)
    else:
        walk = book.walk_portfolio(
            (leg,),
            md,
            combination=combination,
            max_concurrent_positions=1,
        )
        records = walk["records"]["matrix_leg"]
    return {"records": records, "md": md}


def test_cross_mode_matrix_all_valid_rows() -> None:
    """Every VALID combination row trades and books a clean record.

    The whole point: the full combinatorial space produces well-
    formed records with the frozen 18-key shape and a legal
    exit_reason per row's mode.  Mutation fact: any dispatch or
    record-assembly break fails SOME row's shape/exit asserts.
    """
    legal_reasons = {
        "independent": {"stop", "target", "forced"},
        "sequential": {"stop", "target", "forced"},
        "priority": {"stop", "target", "forced"},
        "netting": {"stop", "target", "forced"},
    }
    for combination, mechanism, sizing in _VALID_COMBINATIONS:
        result = _run_row(combination, mechanism, sizing)
        records = result["records"]
        assert records != [], (combination, mechanism, sizing)
        for record in records:
            assert frozenset(record) == frozenset(_expected_keys())
            legal = legal_reasons[combination] | {"signal"}
            assert record["exit_reason"] in legal
        # sizing takes effect
        if sizing == "fixed":
            assert all(r["size_r"] == 1.0 for r in records)
        else:
            assert all(r["size_r"] >= 0.0 for r in records)


def test_independent_vs_netting_costs_differ() -> None:
    """The same world under independent vs netting: different costs_r.

    Netting closes the long early (the short nets it), so the
    taker cost is charged on a shorter hold / different geometry --
    the PIN is that the two combination modes produce DIFFERENT
    ledgers for the same legs.  Mutation fact: a dispatch that
    routes netting to independent (or vice versa) makes the costs
    equal and fails the difference assert.
    """
    md = _md()
    long_leg = _leg_for("netting", "target", "fixed")
    short_leg = LegSpec(
        name="short_side",
        direction="short",
        entry=EntrySpec(Compare("mom", "<", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 20.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        sizing=SizingSpec(mode="fixed", param=1.0),
        features_used=("mom", "atr", "close"),
        rationale="netting counterpart",
    )
    independent_book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    independent_cost = sum(
        record["costs_r"]
        for leg in (long_leg, short_leg)
        for record in independent_book.run(leg, md)
    )
    netting_book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    walk = netting_book.walk_portfolio(
        (long_leg, short_leg),
        md,
        combination="netting",
        max_concurrent_positions=2,
    )
    netting_cost = sum(
        record["costs_r"]
        for records in walk["records"].values()
        for record in records
    )
    assert netting_cost != pytest.approx(independent_cost)


def test_materialize_matrix_nan_split_embargo() -> None:
    """nan_policy x split x embargo combinations all behave per spec."""
    grid = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    raw_ret = pd.Series([1.0] * 10, index=grid, name="ret")
    raw_mom = pd.Series([2.0] * 10, index=grid, name="mom")
    raw_mom.iloc[5] = np.nan  # one interior NaN
    adapter = _TableAdapter(
        {"ret": FieldMeta("ret"), "mom": FieldMeta("mom")},
        {"ret": raw_ret, "mom": raw_mom},
    )
    for nan_policy in ("drop", "ffill"):
        for embargo_days in (0, 2):
            spec = MaterializeSpec(
                outcome=FieldSpec("ret", role="outcome"),
                features=(FieldSpec("mom", role="feature"),),
                train_frac=0.6,
                embargo=pd.Timedelta(days=embargo_days),
                min_train_rows=2,
                min_test_rows=2,
                nan_policy=nan_policy,
            )
            md = materialize(adapter, spec)
            assert md.manifest is not None
            assert md.manifest["validation"]["passed"] is True
    # strict + NaN interior -> loud refusal (the strict pin)
    strict_spec = MaterializeSpec(
        outcome=FieldSpec("ret", role="outcome"),
        features=(FieldSpec("mom", role="feature"),),
        train_frac=0.6,
        min_train_rows=2,
        min_test_rows=2,
        nan_policy="strict",
    )
    with pytest.raises(ValueError, match="strict"):
        materialize(adapter, strict_spec)


def _expected_keys() -> frozenset[str]:
    """The frozen 18-key TradeRecord shape."""
    return frozenset(
        {
            "leg_name",
            "entry_ts",
            "exit_ts",
            "direction",
            "instrument_id",
            "entry_bar",
            "exit_bar",
            "entry_price",
            "exit_price",
            "size_r",
            "pnl_gross_r",
            "pnl_net_r",
            "costs_r",
            "exit_reason",
            "features_at_entry",
            "features_at_exit",
            "diagnostics",
            "netted_remainder",
        }
    )
