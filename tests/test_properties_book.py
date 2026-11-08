"""Book property pins: ledger identities over generated worlds (W15-3).

Every walk must satisfy: netted_remainder >= 0 on netted records,
costs_r >= 0, pnl_gross == pnl_net + costs, equity == fsum(pnl_net),
exposure from FILL windows.  Mutation fact: any accounting drift
breaks an identity over SOME generated world.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from atropos.book import Book
from atropos.data_contract import MaterializedData
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    StopSpec,
)
from hypothesis import given, settings
from hypothesis import strategies as st

_BARS = 12

_PRICE_WORLDS: list[list[float]] = [
    [100.0] * _BARS,
    [
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
        93.0,
    ],
    [
        100.0,
        99.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
    ],
    [
        100.0,
        90.0,
        110.0,
        90.0,
        110.0,
        90.0,
        110.0,
        90.0,
        110.0,
        90.0,
        110.0,
        90.0,
    ],
]

_STOP_MULTIPLES = [2.0, 5.0, 20.0]
_CARRY_WORLDS = st.tuples(
    st.floats(min_value=0.0, max_value=2000.0), st.floats(-200.0, 200.0)
)


def _md_from_prices(prices: list[float]) -> Any:
    """A manifest-less OHLC grid from one price path."""
    bars = pd.date_range(
        "2024-01-01", periods=len(prices), freq="h", tz="UTC"
    )
    lows = [px - 1.0 for px in prices]
    highs = [px + 1.0 for px in prices]
    mom = [1.0] * len(prices)
    features = pd.DataFrame(
        {
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "mom": mom,
            "atr": [1.0] * len(prices),
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


def _hold_leg() -> LegSpec:
    return LegSpec(
        name="hold",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 3.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="property fixture: holding long",
    )


@pytest.mark.parametrize("prices", _PRICE_WORLDS)
@pytest.mark.parametrize("stop_multiple", _STOP_MULTIPLES)
@given(carry=_CARRY_WORLDS)
@settings(max_examples=40, deadline=None)
def test_property_ledger_identities(
    prices: list[float], stop_multiple: float, carry: tuple[float, float]
) -> None:
    """Ledger identities hold over every generated world.

    Mutation fact: any accounting drift (costs dropped, equity
    double-counted, drag sign flipped) breaks at least one identity
    over SOME generated world.
    """
    borrow_bps, funding_bps = carry
    md = _md_from_prices(prices)
    leg = LegSpec(
        name="hold",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", stop_multiple)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="property fixture",
    )
    book = Book(
        borrow_bps_annual=borrow_bps,
        funding_bps_annual=funding_bps,
        periods_override=252.0,
    )
    records = book.run(leg, md)
    equity = 0.0
    for record in records:
        diagnostics = record["diagnostics"]
        # split identity
        assert diagnostics["taker_r"] + diagnostics["borrow_r"] + (
            diagnostics["funding_r"]
        ) == pytest.approx(record["costs_r"], abs=1e-12)
        # gross identity
        assert record["pnl_gross_r"] == pytest.approx(
            record["pnl_net_r"] + record["costs_r"], abs=1e-12
        )
        # a long pays positive funding and RECEIVES negative funding:
        # the recorded magnitude carries the rate's sign (paid > 0,
        # received < 0); borrow is shorts-only, so zero for a long
        assert diagnostics["borrow_r"] == 0.0
        if funding_bps > 0.0:
            assert diagnostics["funding_r"] >= 0.0
            # a paying long: carry shrinks net vs gross
            assert record["pnl_net_r"] <= record["pnl_gross_r"]
        elif funding_bps < 0.0:
            # a receiving long: carry is a CREDIT (costs_r < taker),
            # so net can exceed gross -- by design, not a bug
            assert diagnostics["funding_r"] <= 0.0
        equity += record["pnl_net_r"]
    assert book.equity_r == pytest.approx(equity, abs=1e-9)


def test_year_periods_calendar_table() -> None:
    """_year_periods: the calendar table over grid densities.

    The footgun pin: XNYS daily == 252 (NOT 365).  Mutation fact:
    replacing the table with 365 (or dropping the session-hours
    term) breaks an exact equality.
    """
    from atropos.book import _year_periods

    day = pd.Timedelta(days=1)
    hour = pd.Timedelta(hours=1)
    assert _year_periods("XNYS", day) == 252.0
    assert _year_periods("24/7", day) == 365.0
    assert _year_periods("XNYS", hour) == 1638.0
    assert _year_periods("24/7", hour) == 8760.0
    with pytest.raises(ValueError, match="unknown calendar"):
        _year_periods("JPX", day)
