"""Adapter pins: offset conventions, as-of semantics, dumb-adapter law.

Worlds are deterministic tz-aware UTC fixtures; each test names the
mutation it must catch (mutation plan; applied -> fail -> restored
before commit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.adapters import OHLCVAdapter, TidyAdapter
from atropos.data_contract import (
    FieldMeta,
    FieldSpec,
    MaterializeSpec,
    materialize,
)

_BAR = pd.Timedelta(hours=1)
_DAY = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")


def _bars_frame() -> pd.DataFrame:
    """Six hourly bars for one symbol with distinctive closes."""
    return pd.DataFrame(
        {
            "ts": _DAY,
            "symbol": "TEST",
            "open": np.full(6, 100.0),
            "high": np.full(6, 101.0),
            "low": np.full(6, 99.0),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": np.full(6, 1000.0),
        }
    )


def test_ohlcv_close_convention_offsets() -> None:
    """Close-labelled bars: close-offset 0, open-offset -bar_delta.

    Pins the offset table exactly.  Mutation fact: dropping the open
    offset (or zeroing the close offset) changes the table and fails
    the assert.
    """
    adapter = OHLCVAdapter(
        _bars_frame(),
        "close",
        _BAR,
        symbol="TEST",
        fields_to_serve=("open", "close", "range_pct"),
    )
    declared = adapter.fields()
    assert declared["close"].t_known_offset == pd.Timedelta(0)
    assert declared["open"].t_known_offset == -_BAR
    # derived fields inherit the close offset, never better
    assert declared["range_pct"].t_known_offset == pd.Timedelta(0)


def test_ohlcv_open_convention_offsets() -> None:
    """Open-labelled bars: open-offset 0, close-offset +bar_delta.

    Mutation fact: mirroring the conventions backwards swaps the two
    tables and fails both offset asserts.
    """
    adapter = OHLCVAdapter(
        _bars_frame(),
        "open",
        _BAR,
        symbol="TEST",
        fields_to_serve=("open", "close", "range_pct"),
    )
    declared = adapter.fields()
    assert declared["open"].t_known_offset == pd.Timedelta(0)
    assert declared["close"].t_known_offset == _BAR
    # derived range_pct inherits close's offset, never better
    assert declared["range_pct"].t_known_offset == _BAR


def test_ohlcv_as_of_between_bars() -> None:
    """as-of returns the LAST value known at ts, not an exact match.

    Half a bar after bar 2's close, that close is already known.
    Mutation fact: an exact-ts equality lookup returns the PREVIOUS
    bar's value and the 103.0 assert fails.
    """
    adapter = OHLCVAdapter(
        _bars_frame(),
        "close",
        _BAR,
        symbol="TEST",
        fields_to_serve=("close",),
    )
    middle = _DAY[2] + _BAR / 2
    assert adapter.as_of("close", middle) == 102.0
    assert adapter.as_of("close", _DAY[2]) == 102.0
    assert adapter.as_of("close", _DAY[2] - pd.Timedelta(hours=1)) == 101.0
    # before the first bar's close nothing is known
    assert np.isnan(adapter.as_of("close", _DAY[0] - _BAR))


def test_ohlcv_fetch_window_and_integration() -> None:
    """Fetch slices the label window; materialize consumes the adapter.

    Integration with the C1 contract: one materialize round trip on
    the served close series proves the adapter satisfies the
    protocol the resolver expects.
    """
    adapter = OHLCVAdapter(
        _bars_frame(),
        "close",
        _BAR,
        symbol="TEST",
        fields_to_serve=("close", "range_pct"),
    )
    window = adapter.fetch("close", _DAY[1], _DAY[3])
    assert list(window.index) == [_DAY[1], _DAY[2], _DAY[3]]
    assert window.tolist() == [101.0, 102.0, 103.0]
    md = materialize(
        adapter,
        MaterializeSpec(
            outcome=FieldSpec("close", role="outcome"),
            features=(FieldSpec("range_pct", role="feature"),),
            min_train_rows=2,
            min_test_rows=1,
        ),
    )
    assert md.manifest is not None
    assert md.manifest["lookahead_checks"] == {
        "close": True,
        "range_pct": True,
    }


def test_deferred_multi_entity() -> None:
    """symbol=None keeps the multi-entity path loudly deferred.

    Mutation fact: serving the pooled frame instead of raising makes
    every deferred assert fail.
    """
    adapter = OHLCVAdapter(_bars_frame(), "close", _BAR, symbol=None)
    with pytest.raises(ValueError, match="deferred"):
        adapter.fetch("close", _DAY[0], _DAY[-1])
    with pytest.raises(ValueError, match="deferred"):
        adapter.as_of("close", _DAY[0])


def test_tidy_pivot_and_duplicate_guard() -> None:
    """Tidy pivot serves flat fields; duplicate (ts, field) raises.

    Mutation fact: removing the duplicate-key guard lets the pivot
    silently average/last rows and the raise assert fails.
    """
    tidy = pd.DataFrame(
        {
            "ts": [*_DAY[:3], *_DAY[:3]],
            "field": ["mom"] * 3 + ["atr"] * 3,
            "value": [1.0, 2.0, 3.0, 0.5, 0.5, 0.6],
        }
    )
    adapter = TidyAdapter(
        tidy,
        fields={
            "mom": FieldMeta("mom"),
            "atr": FieldMeta("atr"),
        },
    )
    assert adapter.fetch("mom", _DAY[0], _DAY[2]).tolist() == [1.0, 2.0, 3.0]
    assert adapter.as_of("atr", _DAY[2]) == 0.6
    duplicated = pd.concat([tidy, tidy.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        TidyAdapter(
            duplicated,
            fields={
                "mom": FieldMeta("mom"),
                "atr": FieldMeta("atr"),
            },
        )
    # a tidy field without metadata (or vice versa) is a loud error
    with pytest.raises(ValueError, match="field mismatch"):
        TidyAdapter(tidy, fields={"mom": FieldMeta("mom")})


def test_tidy_offset_respected() -> None:
    """A declared offset shifts the adapter's own as-of cutoff.

    Mutation fact: as_of ignoring the field's t_known_offset fails
    the shifted-world assert.
    """
    tidy = pd.DataFrame(
        {
            "ts": _DAY[:4],
            "field": ["sig"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    shifted = TidyAdapter(
        tidy,
        fields={"sig": FieldMeta("sig", t_known_offset=_BAR)},
    )
    # value labelled t is known at t + 1h: at exactly _DAY[1] the
    # label _DAY[1] is NOT yet known, label _DAY[0] is
    assert shifted.as_of("sig", _DAY[1]) == 1.0
    assert shifted.as_of("sig", _DAY[1] + _BAR) == 2.0
    plain = TidyAdapter(tidy, fields={"sig": FieldMeta("sig")})
    assert plain.as_of("sig", _DAY[1]) == 2.0
