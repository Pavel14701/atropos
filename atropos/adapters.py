"""Dumb source normalizers: OHLCV bars and tidy (ts, field, value).

Adapters DECLARE metadata and serve RAW fields; the resolver
(:func:`atropos.data_contract.materialize`) does as-of cuts, causal
transforms, lag alignment, splits and validation.  No lag or
alignment logic lives here: an adapter that "helped" by shifting
labels would break the very contract the manifest verifies.

Every field is declared through a
:class:`atropos.data_contract.FieldMeta`: the value labelled
``t`` is known at ``t + t_known_offset``.  An adapter must say
which moment of the bar its ``ts`` column labels:

* ``convention="close"``: close/high/low/volume/range_pct are
  known exactly at the label (offset 0); the OPEN was already
  known one bar earlier, so it carries ``-bar_delta``.
* ``convention="open"``: open is known at the label (offset 0);
  close/high/low/volume/range_pct only exist once the bar has
  closed, so they carry ``+bar_delta``.

Derived-field rule (pinned): a derived field must NEVER carry a
BETTER (smaller) ``t_known_offset`` than any of its inputs;
``range_pct = (high - low) / close`` inherits the CLOSE offset
exactly, never the open's.

Normalization here (dumb, not careless): ``ts`` must be
timezone-aware (naive is fatal, non-UTC converted to UTC); rows
are sorted by key and duplicate keys are fatal.  Serving is
single-series (``entity_axis="none"``); multi-entity
materialization is deferred (KNOWN-ISSUES section 14), so a
symbol-less ``OHLCVAdapter`` declares metadata but every
data-serving method raises.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from atropos.data_contract import FieldMeta

__all__ = ["OHLCVAdapter", "TidyAdapter"]

_DEFERRED_NOTE = "deferred (KNOWN-ISSUES section 14)"
"""Shared phrase marking features the MVP does not serve yet."""

_OHLCV_COLUMNS = ("ts", "symbol", "open", "high", "low", "close", "volume")
"""The exact bars-frame contract (missing AND extra columns are fatal)."""

_TIDY_COLUMNS = ("ts", "field", "value")
"""The exact tidy-frame contract (entity is pre-collapsed by callers)."""

_ALLOWED_CONVENTIONS = ("open", "close")
"""Which moment of the bar the ``ts`` column labels."""

_SERVED_FIELDS = ("open", "high", "low", "close", "volume", "range_pct")
"""Fields an OHLCVAdapter can serve (``range_pct`` is derived here)."""


# ---------------------------------------------------------------------------
# shared validation / as-of bodies (kept out of the constructors)
# ---------------------------------------------------------------------------
def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], tag: str
) -> None:
    """Fatal unless the frame carries exactly ``required``."""
    missing = [col for col in required if col not in frame.columns]
    extra = [str(col) for col in frame.columns if col not in required]
    if missing or extra:
        raise ValueError(
            f"{tag}: columns must be exactly {list(required)};"
            f" missing {missing}, unexpected {extra}"
        )


def _to_utc_labels(column: pd.Series, tag: str) -> pd.DatetimeIndex:
    """UTC DatetimeIndex from a timestamp column; naive is fatal."""
    labels = pd.DatetimeIndex(column)
    if labels.tz is None:
        raise ValueError(f"{tag}: ts must be timezone-aware (UTC)")
    return labels.tz_convert("UTC")


def _offset_map(
    convention: str, bar_delta: pd.Timedelta
) -> dict[str, pd.Timedelta]:
    """Per-field ``t_known_offset`` for one bar convention."""
    if convention == "close":
        open_offset = -bar_delta
        close_offset = pd.Timedelta(0)
    else:
        open_offset = pd.Timedelta(0)
        close_offset = bar_delta
    return {
        "open": open_offset,
        "close": close_offset,
        "high": close_offset,
        "low": close_offset,
        "volume": close_offset,
        "range_pct": close_offset,
    }


def _validate_ohlcv_args(
    convention: str,
    bar_delta: pd.Timedelta,
    symbol: str | None,
    fields_to_serve: tuple[str, ...],
) -> None:
    """Loud ``__init__`` validation: convention, delta, symbol, fields."""
    if convention not in _ALLOWED_CONVENTIONS:
        raise ValueError(
            f"OHLCVAdapter: convention must be one of"
            f" {list(_ALLOWED_CONVENTIONS)}, got {convention!r}"
        )
    if not isinstance(bar_delta, pd.Timedelta) or (
        bar_delta <= pd.Timedelta(0)
    ):
        raise ValueError(
            "OHLCVAdapter: bar_delta must be a positive pd.Timedelta,"
            f" got {bar_delta!r}"
        )
    if symbol is not None and not isinstance(symbol, str):
        raise ValueError(
            "OHLCVAdapter: symbol must be a str or None, got"
            f" {type(symbol).__name__}"
        )
    if len(set(fields_to_serve)) != len(fields_to_serve):
        raise ValueError(
            f"OHLCVAdapter: fields_to_serve repeats entries:"
            f" {list(fields_to_serve)}"
        )
    unknown = [item for item in fields_to_serve if item not in _SERVED_FIELDS]
    if unknown:
        raise ValueError(
            f"OHLCVAdapter: fields_to_serve must be a subset of"
            f" {list(_SERVED_FIELDS)}, got unknown {unknown}"
        )


def _validate_tidy_fields(
    fields: dict[str, FieldMeta], present: set[str]
) -> None:
    """Bidirectional fields <-> tidy-field coverage check."""
    if not fields:
        raise ValueError(
            "TidyAdapter: fields must declare at least one field"
        )
    declared = set(fields)
    missing_meta = sorted(present - declared)
    missing_data = sorted(declared - present)
    if missing_meta or missing_data:
        raise ValueError(
            f"TidyAdapter: field mismatch -- no FieldMeta for"
            f" {missing_meta}, no tidy data for {missing_data}"
        )
    misnamed = sorted(key for key, meta in fields.items() if meta.name != key)
    if misnamed:
        raise ValueError(
            "TidyAdapter: FieldMeta.name must match its mapping key;"
            f" mismatched keys: {misnamed}"
        )


def _as_of_value(
    series: pd.Series, offset: pd.Timedelta, ts: pd.Timestamp
) -> float:
    """Last value whose label ``+ offset`` lands at/below ``ts``."""
    cutoff = ts - offset
    position = int(series.index.searchsorted(cutoff, side="right"))
    position -= 1
    if position < 0:
        return float("nan")
    return float(series.iloc[position])


# ---------------------------------------------------------------------------
# OHLCV bars adapter
# ---------------------------------------------------------------------------
class OHLCVAdapter:
    """Adapter over one OHLCV bars frame (single symbol, flat fields).

    One :class:`atropos.data_contract.FieldMeta` per served field
    (dtype float, kind ``"bar"``, ``entity_axis="none"``,
    ``adjustment="none"``) with the convention-dependent
    ``t_known_offset`` (module conventions table).  ``range_pct =
    (high - low) / close`` is computed ONCE here (NaN where
    ``close == 0``, never an exception) and never claims a better
    offset than its inputs.  With ``symbol=None`` metadata still
    declares, but every data-serving method raises the deferred
    multi-entity ``ValueError``.

    Args:
        bars: frame with exactly ``["ts", "symbol", "open", "high",
            "low", "close", "volume"]``; tz-aware ``ts`` (naive is
            fatal, non-UTC converted); sorted here by ``(symbol,
            ts)``; duplicate ``(ts, symbol)`` rows are fatal.
        convention: which moment ``ts`` labels, ``"close"`` or
            ``"open"`` -- see the module conventions table.
        bar_delta: bar length; a positive ``pd.Timedelta``.
        symbol: the ONE symbol served as flat series; ``None``
            defers serving (data methods raise).
        name: adapter identity recorded by the resolver.
        version: adapter version recorded by the resolver.
        calendar: trading-calendar label (informational).
        fields_to_serve: subset of ``{"open", "high", "low",
            "close", "volume", "range_pct"}`` (no repeats).

    Raises:
        ValueError: on malformed ``bars`` (wrong columns, naive
            ``ts``, duplicates, emptiness, non-numeric values), a
            ``symbol`` absent from ``bars``, a bad ``convention`` /
            ``bar_delta``, or a repeated/unknown
            ``fields_to_serve`` entry.

    Example::

        >>> import pandas as pd
        >>> ts = pd.date_range(
        ...     "2024-01-02 14:00", periods=3, freq="1h", tz="UTC"
        ... )
        >>> bars = pd.DataFrame(
        ...     {"ts": ts, "symbol": "SPY", "open": [99.5, 100.5, 101.5],
        ...      "high": [101.0, 102.0, 103.0], "low": [100.0, 101.0, 102.0],
        ...      "close": [100.0, 101.0, 102.0],
        ...      "volume": [1000.0, 1100.0, 900.0]}
        ... )
        >>> adapter = OHLCVAdapter(
        ...     bars, "close", pd.Timedelta("1h"), symbol="SPY",
        ...     fields_to_serve=("open", "close", "range_pct"),
        ... )
        >>> adapter.fields()["open"].t_known_offset == -pd.Timedelta("1h")
        True
        >>> adapter.fields()["range_pct"].t_known_offset == pd.Timedelta(0)
        True
        >>> adapter.fetch("close", ts[0], ts[2]).tolist()
        [100.0, 101.0, 102.0]
        >>> adapter.as_of("open", ts[0])  # open known a bar earlier
        100.5
        >>> adapter.as_of("close", ts[0] - pd.Timedelta("1h"))
        nan
        >>> adapter.validate_self()
        []
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        convention: Literal["open", "close"],
        bar_delta: pd.Timedelta,
        *,
        symbol: str | None,
        name: str = "ohlcv",
        version: str = "1.0",
        calendar: str = "24/7",
        fields_to_serve: tuple[str, ...] = ("close",),
    ) -> None:
        """Validate, normalize, compute range_pct and declare fields."""
        _validate_ohlcv_args(convention, bar_delta, symbol, fields_to_serve)
        if not isinstance(bars, pd.DataFrame):
            raise ValueError(
                "OHLCVAdapter: bars must be a pd.DataFrame, got"
                f" {type(bars).__name__}"
            )
        _require_columns(bars, _OHLCV_COLUMNS, "OHLCVAdapter bars")
        if bars.empty:
            raise ValueError(
                "OHLCVAdapter: bars must contain at least one row"
            )
        frame = bars.loc[:, list(_OHLCV_COLUMNS)].copy()
        frame["ts"] = _to_utc_labels(frame["ts"], "OHLCVAdapter bars")
        frame = frame.sort_values(["symbol", "ts"], kind="stable")
        duplicates = int(frame.duplicated(subset=["ts", "symbol"]).sum())
        if duplicates:
            raise ValueError(
                f"OHLCVAdapter: {duplicates} duplicate (ts, symbol) row(s)"
            )
        symbol_set = set(frame["symbol"])
        if symbol is not None and symbol not in symbol_set:
            raise ValueError(
                f"OHLCVAdapter: symbol {symbol!r} is not in bars;"
                f" available: {sorted({str(item) for item in symbol_set})}"
            )
        self.name = name
        self.version = version
        self.calendar = calendar
        self._symbol: str | None = symbol
        self._served: tuple[str, ...] = fields_to_serve
        offsets = _offset_map(convention, bar_delta)
        self._fields = {
            field_name: FieldMeta(
                field_name, t_known_offset=offsets[field_name]
            )
            for field_name in self._served
        }
        closes = frame["close"].to_numpy(dtype=float)
        frame["range_pct"] = (
            frame["high"].to_numpy(dtype=float)
            - frame["low"].to_numpy(dtype=float)
        ) / np.where(closes != 0.0, closes, np.nan)
        self._rows = (
            frame if symbol is None else frame[frame["symbol"] == symbol]
        )
        self._labels = pd.DatetimeIndex(self._rows["ts"])
        self._series: dict[str, pd.Series] = {}
        if symbol is None:
            return
        self._series = {
            field_name: pd.Series(
                self._rows[field_name].to_numpy(dtype=float),
                index=self._labels,
                name=field_name,
            )
            for field_name in self._served
        }

    def _require_symbol(self) -> str:
        """Serving guard: multi-entity is deferred while ``None``."""
        if self._symbol is None:
            raise ValueError(
                "OHLCVAdapter: multi-entity serving is"
                f" {_DEFERRED_NOTE}; construct with symbol=<name>"
            )
        return self._symbol

    def _lookup(self, name: str) -> pd.Series:
        """The served Series for ``name`` (unknown field is fatal)."""
        self._require_symbol()
        if name not in self._series:
            raise ValueError(
                f"OHLCVAdapter: unknown field {name!r}; serves"
                f" {list(self._served)}"
            )
        return self._series[name]

    def fields(self) -> dict[str, FieldMeta]:
        """Declared metadata for every served field."""
        return dict(self._fields)

    def extent(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Inclusive ``(min ts, max ts)`` of the served symbol."""
        self._require_symbol()
        first: pd.Timestamp = self._labels.min()
        last: pd.Timestamp = self._labels.max()
        return first, last

    def fetch(
        self, name: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Raw float values of ``name`` with labels in ``[start, end]``."""
        series = self._lookup(name)
        return series.loc[start:end]

    def as_of(self, name: str, ts: pd.Timestamp) -> float:
        """Value of ``name`` known at ``ts`` (NaN when none is).

        Cutoff ``= ts - t_known_offset``, last label ``<=`` cutoff.
        """
        series = self._lookup(name)
        offset = self._fields[name].t_known_offset
        return _as_of_value(series, offset, ts)

    def validate_self(self) -> list[str]:
        """Adapter-internal consistency issues ([] when clean).

        NaN closes, nonpositive volume, bars-per-day irregularity.
        """
        issues: list[str] = []
        rows = self._rows
        nan_closes = int(rows["close"].isna().sum())
        if nan_closes:
            issues.append(f"{nan_closes} close value(s) are NaN")
        nonpositive = int((rows["volume"] <= 0).sum())
        if nonpositive:
            issues.append(f"{nonpositive} volume value(s) are nonpositive")
        per_day = rows.groupby(rows["ts"].dt.floor("D"))["ts"].count()
        distinct = sorted({int(count) for count in per_day})
        if len(distinct) > 1:
            issues.append(
                f"bars per day varies across distinct counts: {distinct}"
            )
        return issues


# ---------------------------------------------------------------------------
# tidy (ts, field, value) adapter
# ---------------------------------------------------------------------------
class TidyAdapter:
    """Adapter over a long-format ``(ts, field, value)`` frame.

    The MVP tidy contract has NO entity column: callers pre-collapse
    entity levels to one series per field.  Every ``field`` value in
    the frame must have a declared
    :class:`atropos.data_contract.FieldMeta` here and vice versa
    (mismatch is fatal, naming the diff).  The frame is normalized
    to one float Series per field (sorted by ``ts``) in ``__init__``;
    ``as_of`` uses each field's own ``t_known_offset`` and
    ``validate_self`` reports non-finite values per field as
    non-fatal warnings.

    Args:
        tidy: frame with exactly ``["ts", "field", "value"]``;
            tz-aware ``ts`` (naive is fatal, non-UTC converted);
            sorted here by ``(field, ts)``; duplicate ``(ts, field)``
            pairs are fatal; ``value`` must be float-coercible and
            every ``field`` value must be a ``str``.
        fields: one FieldMeta per tidy field (and no more), keyed by
            the ``field`` value it describes.
        name: adapter identity recorded by the resolver.
        version: adapter version recorded by the resolver.
        calendar: trading-calendar label (informational).

    Raises:
        ValueError: on malformed ``tidy`` (wrong columns, naive
            ``ts``, duplicate ``(ts, field)`` pairs, non-str field
            values, a non-numeric ``value`` column), a non-dict or
            empty ``fields``, fields/data set mismatches, or a
            FieldMeta whose ``name`` differs from its key.

    Example::

        >>> import pandas as pd
        >>> from atropos.data_contract import FieldMeta
        >>> ts = pd.date_range(
        ...     "2024-01-02 14:00", periods=3, freq="1h", tz="UTC"
        ... )
        >>> tidy = pd.DataFrame(
        ...     {
        ...         "ts": list(ts) + list(ts[:2]),
        ...         "field": ["spread"] * 3 + ["imbalance"] * 2,
        ...         "value": [0.01, 0.02, 0.03, 1.5, -2.5],
        ...     }
        ... )
        >>> fields = {
        ...     "spread": FieldMeta("spread"),
        ...     "imbalance": FieldMeta(
        ...         "imbalance", t_known_offset=pd.Timedelta("30min")
        ...     ),
        ... }
        >>> adapter = TidyAdapter(tidy, fields=fields)
        >>> adapter.extent() == (ts[0], ts[2])
        True
        >>> adapter.as_of("imbalance", ts[1] + pd.Timedelta("30min"))
        -2.5
        >>> adapter.validate_self()
        []
    """

    def __init__(
        self,
        tidy: pd.DataFrame,
        *,
        fields: dict[str, FieldMeta],
        name: str = "tidy",
        version: str = "1.0",
        calendar: str = "24/7",
    ) -> None:
        """Validate, normalize and pivot to one Series per field."""
        if not isinstance(tidy, pd.DataFrame):
            raise ValueError(
                "TidyAdapter: tidy must be a pd.DataFrame, got"
                f" {type(tidy).__name__}"
            )
        if not isinstance(fields, dict):
            raise ValueError(
                "TidyAdapter: fields must be a dict[str, FieldMeta],"
                f" got {type(fields).__name__}"
            )
        _require_columns(tidy, _TIDY_COLUMNS, "TidyAdapter tidy")
        frame = tidy.loc[:, list(_TIDY_COLUMNS)].copy()
        frame["ts"] = _to_utc_labels(frame["ts"], "TidyAdapter tidy")
        frame = frame.sort_values(["field", "ts"], kind="stable")
        duplicates = int(frame.duplicated(subset=["ts", "field"]).sum())
        if duplicates:
            raise ValueError(
                f"TidyAdapter: {duplicates} duplicate (ts, field) pair(s)"
            )
        present = set(frame["field"])
        non_str = sorted(
            {str(item) for item in present if not isinstance(item, str)}
        )
        if non_str:
            raise ValueError(
                f"TidyAdapter: field values must be str, got {non_str}"
            )
        _validate_tidy_fields(fields, present)
        self.name = name
        self.version = version
        self.calendar = calendar
        self._fields = dict(fields)
        values = frame["value"].to_numpy(dtype=float)
        self._series: dict[str, pd.Series] = {}
        for field_name in sorted(present):
            mask = (frame["field"] == field_name).to_numpy()
            labels = pd.DatetimeIndex(frame.loc[mask, "ts"])
            self._series[field_name] = pd.Series(
                values[mask], index=labels, name=field_name
            )

    def _lookup(self, name: str) -> pd.Series:
        """The served Series for ``name`` (unknown field is fatal)."""
        if name not in self._series:
            raise ValueError(
                f"TidyAdapter: unknown field {name!r}; serves"
                f" {sorted(self._series)}"
            )
        return self._series[name]

    def fields(self) -> dict[str, FieldMeta]:
        """Declared metadata (verbatim copy of ``fields``)."""
        return dict(self._fields)

    def extent(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Union ``(min ts, max ts)`` over every served field."""
        first: pd.Timestamp = min(
            series.index.min() for series in self._series.values()
        )
        last: pd.Timestamp = max(
            series.index.max() for series in self._series.values()
        )
        return first, last

    def fetch(
        self, name: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Raw float values of ``name`` with labels in ``[start, end]``."""
        series = self._lookup(name)
        return series.loc[start:end]

    def as_of(self, name: str, ts: pd.Timestamp) -> float:
        """Value of ``name`` known at ``ts`` (NaN when none is).

        Uses ``name``'s OWN ``t_known_offset`` as the cutoff shift.
        """
        series = self._lookup(name)
        offset = self._fields[name].t_known_offset
        return _as_of_value(series, offset, ts)

    def validate_self(self) -> list[str]:
        """Adapter-internal consistency issues ([] when clean).

        Per-field non-finite (NaN/inf) value counts.
        """
        issues: list[str] = []
        for field_name, series in self._series.items():
            values = series.to_numpy(dtype=float)
            non_finite = int(np.count_nonzero(~np.isfinite(values)))
            if non_finite:
                issues.append(
                    f"field {field_name!r}: {non_finite} non-finite value(s)"
                )
        return issues
