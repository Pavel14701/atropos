"""Contract property pins: manifest keys, TradeRecord keys (W15-3).

Every materialize/run must produce the FROZEN key sets -- verified
over generated worlds, not just hand-built ones.
"""

from __future__ import annotations

import pandas as pd
from atropos.book import Book, _demo_leg, _demo_md
from atropos.data_contract import (
    FieldMeta,
    FieldSpec,
    MaterializeSpec,
    _TableAdapter,
    materialize,
)
from hypothesis import given, settings
from hypothesis import strategies as st

_MANIFEST_KEYS = frozenset(
    {
        "bundle_hash",
        "source_versions",
        "calendar",
        "row_count",
        "train_rows",
        "test_rows",
        "dropped_embargo_gap",
        "date_range",
        "nan_fraction",
        "dropped",
        "warnings",
        "feature_hashes",
        "lookahead_checks",
        "validation",
    }
)


_CALENDARS = st.sampled_from(["XNYS", "24/7"])
_PERIODS = st.integers(min_value=5, max_value=30)


@given(periods=_PERIODS, calendar=_CALENDARS)
@settings(max_examples=30, deadline=None)
def test_property_manifest_keys_stable(periods: int, calendar: str) -> None:
    """Every materialization produces the frozen manifest key set.

    Mutation fact: adding/removing a DataManifest key breaks the
    frozenset equality regardless of the generated world.
    """
    grid = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    raw_ret = pd.Series(1.0, index=grid, name="ret")
    raw_mom = pd.Series(2.0, index=grid, name="mom")
    adapter = _TableAdapter(
        {"ret": FieldMeta("ret"), "mom": FieldMeta("mom")},
        {"ret": raw_ret, "mom": raw_mom},
        calendar=calendar,
    )
    spec = MaterializeSpec(
        outcome=FieldSpec("ret", role="outcome"),
        features=(FieldSpec("mom", role="feature"),),
        min_train_rows=2,
        min_test_rows=2,
        nan_policy="drop",
        calendar=calendar,
    )
    md = materialize(adapter, spec)
    assert md.manifest is not None
    assert frozenset(md.manifest) == _MANIFEST_KEYS
    assert md.manifest["calendar"] == calendar
    assert all(md.manifest["lookahead_checks"].values())


def test_property_trade_record_keys_stable() -> None:
    """Every Book.run emits the frozen 17-key TradeRecord shape.

    Mutation fact: adding/removing a TradeRecord key breaks the
    frozenset equality.
    """
    book = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    records = book.run(_demo_leg(), _demo_md())
    assert records != []
    expected = frozenset(
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
            "netted_remainder",
            "features_at_entry",
            "features_at_exit",
            "diagnostics",
        }
    )
    for record in records:
        assert frozenset(record) == expected


def test_property_materialize_lag_shift() -> None:
    """lag=n shifts the outcome column by exactly n bars (W15-2 pin).

    Mutation fact: dropping the lag term from the as-of cutoff makes
    lag=0 and lag=n identical and this test fails.
    """
    grid = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    raw_ret = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=grid[:6], name="ret"
    )
    raw_mom = pd.Series([7.0, 8.0, 9.0, 10.0], index=grid[:4], name="mom")
    adapter = _TableAdapter(
        {"ret": FieldMeta("ret"), "mom": FieldMeta("mom")},
        {"ret": raw_ret, "mom": raw_mom},
    )

    def _materialize(lag: int) -> pd.Series:
        spec = MaterializeSpec(
            outcome=FieldSpec("ret", role="outcome", lag=lag),
            features=(FieldSpec("mom", role="feature"),),
            min_train_rows=1,
            min_test_rows=1,
            nan_policy="drop",
        )
        md = materialize(adapter, spec)
        return md.outcome

    unlagged = _materialize(0)
    lagged = _materialize(1)
    # lag=1: the first decision row drops (no raw value one bar
    # before it), and every remaining row reads raw[t-1]
    assert len(unlagged) == 6
    assert len(lagged) == 5
    assert lagged.iloc[0] == 1.0  # raw[Jan 1] (the shifted read)
    assert unlagged.iloc[0] == 1.0  # raw[Jan 1] (unshifted)
