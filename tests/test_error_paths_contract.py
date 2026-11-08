"""Error-path ladder pins: data_contract + adapters (W15-2).

Coverage-driven: every case here targets an UNCOVERED raise line
from the V-0 coverage.json (validation-ladder rungs).  One param
case per rung, match on a distinctive fragment.  Mutation fact:
deleting the pinned rung's raise fails the case.
"""

from __future__ import annotations

import pandas as pd
import pytest
from atropos.adapters import OHLCVAdapter, TidyAdapter
from atropos.data_contract import (
    FieldMeta,
    FieldSpec,
    MaterializeSpec,
    _TableAdapter,
)

_ZERO = pd.Timedelta(0)
_BAR_H = pd.Timedelta(hours=1)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "symbol": "TEST",
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0] * 4,
        }
    )


def test_fieldmeta_ladder() -> None:
    """Every FieldMeta validation rung raises with the field named."""
    with pytest.raises(ValueError, match="must be a non-empty str"):
        FieldMeta(name="")
    with pytest.raises(ValueError, match="dtype must be one of"):
        FieldMeta(name="f", dtype="complex")
    with pytest.raises(ValueError, match="kind must be one of"):
        FieldMeta(name="f", kind="exotic")
    with pytest.raises(ValueError, match="t_known_offset must be a"):
        FieldMeta(name="f", t_known_offset=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entity_axis must be one of"):
        FieldMeta(name="f", entity_axis="exchange")
    with pytest.raises(ValueError, match="adjustment must be one of"):
        FieldMeta(name="f", adjustment="survivorship")


def test_fieldspec_ladder() -> None:
    """Every FieldSpec rung: name, role, lag, aggregate, fill, staleness."""
    with pytest.raises(ValueError, match="name must be a non-empty str"):
        FieldSpec(name="", role="feature")
    with pytest.raises(ValueError, match="role must be one of"):
        FieldSpec(name="f", role="oracle")
    with pytest.raises(ValueError, match="lag"):
        FieldSpec(name="f", role="feature", lag=-1)
    with pytest.raises(ValueError, match="aggregate values beyond"):
        FieldSpec(name="f", role="feature", aggregate="mean")
    with pytest.raises(ValueError, match="fill must be one of"):
        FieldSpec(name="f", role="feature", fill="interpolate")
    with pytest.raises(ValueError, match="staleness_limit"):
        FieldSpec(
            name="f", role="feature", staleness_limit=pd.Timedelta(days=-1)
        )


def test_materialize_spec_ladder() -> None:
    """Every MaterializeSpec rung raises with its fragment."""
    outcome = FieldSpec("ret", role="outcome")
    feature = FieldSpec("mom", role="feature")
    with pytest.raises(ValueError, match="role='outcome'"):
        MaterializeSpec(outcome=FieldSpec("ret", role="feature"))
    with pytest.raises(ValueError, match="role='feature'"):
        MaterializeSpec(
            outcome=outcome,
            features=(FieldSpec("mom", role="control"),),
            min_train_rows=1,
            min_test_rows=1,
        )
    with pytest.raises(ValueError, match="role='control'"):
        MaterializeSpec(
            outcome=outcome,
            controls=(FieldSpec("mom", role="feature"),),
            min_train_rows=1,
            min_test_rows=1,
        )
    with pytest.raises(ValueError, match="names must be unique"):
        MaterializeSpec(
            outcome=outcome,
            features=(feature, feature),
            min_train_rows=1,
            min_test_rows=1,
        )
    with pytest.raises(ValueError, match="train_frac"):
        MaterializeSpec(outcome=outcome, train_frac=1.5)
    with pytest.raises(ValueError, match="train_end must be"):
        MaterializeSpec(outcome=outcome, train_end="2024-01-01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embargo"):
        MaterializeSpec(outcome=outcome, embargo=pd.Timedelta(days=-2))
    with pytest.raises(ValueError, match="nan_policy"):
        MaterializeSpec(outcome=outcome, nan_policy="ignore")
    with pytest.raises(ValueError, match="calendar must be non-empty"):
        MaterializeSpec(outcome=outcome, calendar="")


def test_table_adapter_guards() -> None:
    """_TableAdapter: series-without-meta and unknown fetch fail loudly."""
    series = pd.Series(
        [1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
    )
    with pytest.raises(ValueError, match="no declared FieldMeta"):
        _TableAdapter({}, {"ret": series})
    adapter = _TableAdapter({"ret": FieldMeta("ret")}, {"ret": series})
    with pytest.raises(ValueError, match="unknown field"):
        adapter.fetch("ghost", series.index.min(), series.index.max())
    with pytest.raises(ValueError, match="unknown field"):
        adapter.as_of("ghost", series.index.max())


def _naive_bars() -> pd.DataFrame:
    bars = _bars()
    bars["ts"] = bars["ts"].dt.tz_localize(None)
    return bars


def test_ohlcv_adapter_ladder() -> None:
    """OHLCV constructor/fetch guards: columns, empty, repeats, naive."""
    with pytest.raises(ValueError, match="columns"):
        OHLCVAdapter(
            _bars().rename(columns={"close": "px"}),
            "close",
            _BAR_H,
            symbol="TEST",
        )
    empty = _bars().iloc[0:0]
    with pytest.raises(ValueError, match="at least one row"):
        OHLCVAdapter(empty, "close", _BAR_H, symbol="TEST")
    bars = _bars()
    dup = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        OHLCVAdapter(dup, "close", _BAR_H, symbol="TEST")
    with pytest.raises(ValueError, match="symbol"):
        OHLCVAdapter(bars, "close", _BAR_H, symbol="NOPE")
    adapter = OHLCVAdapter(
        bars, "close", _BAR_H, symbol="TEST", fields_to_serve=("close",)
    )
    with pytest.raises(ValueError, match="unknown field"):
        adapter.fetch("ghost", bars["ts"].min(), bars["ts"].max())


def test_tidy_adapter_ladder() -> None:
    """TidyAdapter guards: columns, field/meta mismatch, unknown fetch."""
    tidy = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "field": ["mom"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    adapter = TidyAdapter(tidy, fields={"mom": FieldMeta("mom")})
    with pytest.raises(ValueError, match="unknown field"):
        adapter.fetch("ghost", tidy["ts"].min(), tidy["ts"].max())
    with pytest.raises(ValueError, match="field mismatch"):
        TidyAdapter(
            tidy, fields={"mom": FieldMeta("mom"), "x": FieldMeta("x")}
        )
