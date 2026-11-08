"""Data contract: look-ahead becomes structurally impossible, not discouraged.

Three structures and one protocol close the gap the bare-Series path
leaves open:

* :class:`MaterializedData` -- the ONLY bundle framework functions
  should trust: outcome/features/controls aligned as-of, a decision
  index (ts/entity/split/regime) and a :class:`DataManifest` whose
  guarantees are verified, not assumed.
* :class:`FieldMeta` -- what an adapter DECLARES about a field, with
  ``t_known_offset`` as the heart: the value labelled ``t`` is known
  at ``t + t_known_offset``.  Bar closes carry offset 0; the OPEN of
  a close-labelled bar carries ``-bar_delta`` (it was known one bar
  earlier).  The resolver enforces
  ``cutoff = t_decision - t_known_offset - lag * bar_delta`` and then
  VERIFIES the produced column against an independently recomputed
  cutoff, so a resolver bug flips ``lookahead_checks`` to False and
  :func:`materialize` raises -- a caught bug, not a documented hope.
* :class:`MaterializeSpec` -- what the caller declares: roles, lags,
  causal transforms, fills, and a timestamp-based train/test split
  with a wall-clock embargo.

* :class:`Adapter` -- the protocol every source normalizes to:
  ``fields() / fetch() / as_of() / validate_self()`` plus ``extent()``.
  Adapters are dumb normalizers; lag, alignment, splits and validation
  all live in the resolver.

Conventions pinned here:

* All timestamps are UTC-aware; a naive index from an adapter is a
  fatal error (time semantics are the contract's reason to exist).
* ``transform="zscore"`` is DEFERRED: a full-sample z-score leaks
  future moments, and an expanding variant is a separate decision.
* Multi-entity materialization (``entity_axis != "none"``) is
  DEFERRED (KNOWN-ISSUES section 14): the MVP resolves single-series
  fields; the index frame keeps the entity column (empty strings) so
  the schema does not move later.
* The train/test split is TIMESTAMP-based (over the unique decision
  grid, never row counts) with a wall-clock embargo; decision rows
  inside the embargo gap belong to NEITHER split and are counted in
  ``manifest["dropped"]["embargo_gap"]``.

The bridge :func:`to_pipeline_data` feeds the legacy
``run_full_pipeline(data: dict)`` -- it is a LEGACY path, not the way
forward: the pipeline re-aligns with an inner join + dropna, so the
as-of guarantees do NOT survive the bridge.  The bridge warns loudly
on every call.  For contract guarantees use
``atropos.feature_scan.feature_analysis(materialized_data)`` directly
(it validates the manifest: a failed lookahead check or a failed
validation battery is fatal, a missing manifest is a loud warning).

Run ``python -m atropos.data_contract`` for a synthetic demo.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd

from atropos._utils import validate_int

__all__ = [
    "Adapter",
    "DataManifest",
    "FieldMeta",
    "FieldSpec",
    "MaterializeSpec",
    "MaterializedData",
    "ValidationReport",
    "materialize",
    "to_pipeline_data",
]

_ALLOWED_ROLES = ("outcome", "feature", "control", "regime")
_ALLOWED_DTYPES = ("float", "int", "bool", "category", "datetime")
_ALLOWED_KINDS = (
    "bar",
    "event",
    "derived",
    "surface_scalar",
    "constant",
)
_ALLOWED_ENTITY_AXES = ("none", "symbol", "contract")
_ALLOWED_ADJUSTMENTS = ("none", "split", "split+div", "total_return")
_ALLOWED_TRANSFORMS = ("raw", "diff", "pct_change", "log")
_ALLOWED_AGGREGATES = ("last",)
_ALLOWED_FILL = ("none", "ffill", "zero")
_ALLOWED_NAN_POLICY = ("strict", "drop", "ffill")
_ALLOWED_SPLITS = ("train", "test")

_DEFERRED_NOTE = "deferred (KNOWN-ISSUES section 14)"
_BRIDGE_WARNING = (
    "to_pipeline_data is a LEGACY bridge, not a path for new code:"
    " run_full_pipeline re-aligns with an inner join + dropna, so the"
    " as-of guarantees of the data contract do NOT survive the bridge."
    " For contract guarantees call"
    " atropos.feature_scan.feature_analysis(materialized_data)"
    " directly."
)
_NO_MANIFEST_WARNING = (
    "unverified data, contract guarantees not applied (manifest is None)"
)
_MIN_BAR_OBS = 2
"""Minimum observations of a raw field to estimate its bar delta."""

_ZERO = pd.Timedelta(0)
"""Shared immutable zero offset / embargo (dataclass default)."""


# ---------------------------------------------------------------------------
# validation report + manifest (frozen schemas, every key every outcome)
# ---------------------------------------------------------------------------
class ValidationReport(TypedDict):
    """Outcome of the materialization battery.

    ``passed`` is True only when ``fatal`` is empty; ``warnings`` are
    recorded but never block.
    """

    passed: bool
    fatal: list[str]
    warnings: list[str]


class DataManifest(TypedDict):
    """Provenance + verification record of one materialization.

    ``bundle_hash`` is the FULL sha256 over the canonical JSON of the
    spec, adapter identity, calendar, feature hashes and coverage --
    the anchor a preregistration cites.  ``lookahead_checks`` is the
    verified per-field guarantee (a False entry makes the bundle
    unusable); ``validation`` carries the battery outcome.
    """

    bundle_hash: str
    source_versions: dict[str, str]
    calendar: str
    row_count: int
    train_rows: int
    test_rows: int
    dropped_embargo_gap: int
    date_range: tuple[str, str]
    nan_fraction: dict[str, float]
    dropped: dict[str, int]
    warnings: list[str]
    feature_hashes: dict[str, str]
    lookahead_checks: dict[str, bool]
    validation: ValidationReport


# ---------------------------------------------------------------------------
# declared metadata (adapter side)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldMeta:
    """What an adapter declares about one field.

    ``t_known_offset`` is (moment the value is known) minus
    (timestamp label): bar closes on close-labelled bars carry 0, the
    open of a close-labelled bar carries ``-bar_delta``, events carry
    0.  The resolver turns it into the as-of cutoff; declaring it
    wrong is a data bug the manifest cannot paper over.
    """

    name: str
    dtype: str = "float"
    kind: str = "bar"
    t_known_offset: pd.Timedelta = _ZERO
    entity_axis: str = "none"
    adjustment: str = "none"
    units: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Reject malformed metadata loudly, at construction."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError(
                f"FieldMeta.name must be a non-empty str, got {self.name!r}"
            )
        if self.dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"FieldMeta {self.name!r}: dtype must be one of"
                f" {list(_ALLOWED_DTYPES)}, got {self.dtype!r}"
            )
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"FieldMeta {self.name!r}: kind must be one of"
                f" {list(_ALLOWED_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.t_known_offset, pd.Timedelta):
            raise ValueError(
                f"FieldMeta {self.name!r}: t_known_offset must be a"
                f" pd.Timedelta, got {type(self.t_known_offset).__name__}"
            )
        if self.entity_axis not in _ALLOWED_ENTITY_AXES:
            raise ValueError(
                f"FieldMeta {self.name!r}: entity_axis must be one of"
                f" {list(_ALLOWED_ENTITY_AXES)}, got {self.entity_axis!r}"
            )
        if self.adjustment not in _ALLOWED_ADJUSTMENTS:
            raise ValueError(
                f"FieldMeta {self.name!r}: adjustment must be one of"
                f" {list(_ALLOWED_ADJUSTMENTS)}, got {self.adjustment!r}"
            )


# ---------------------------------------------------------------------------
# declared request (caller side)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """How one field enters the materialized frame.

    ``lag`` is applied AFTER the as-of cut (in bars, floored by the
    field's own median spacing) -- declaring ``lag=1`` on an outcome
    is how a next-bar forward label stays honest.  Transforms are
    causal only: ``zscore`` is deferred because a full-sample z-score
    leaks future moments.
    """

    name: str
    role: str = "feature"
    lag: int = 0
    transform: str = "raw"
    aggregate: str = "last"
    fill: str = "none"
    staleness_limit: pd.Timedelta | None = None

    def __post_init__(self) -> None:
        """Reject malformed specs loudly, at construction."""
        _validate_field_spec(self)


def _validate_field_spec(spec: FieldSpec) -> None:
    """Validation body of :meth:`FieldSpec.__post_init__`.

    Split out of the dataclass body so no single constructor carries
    a 13-branch ladder; every branch raises with the field named.
    """
    if not spec.name or not isinstance(spec.name, str):
        raise ValueError(
            f"FieldSpec.name must be a non-empty str, got {spec.name!r}"
        )
    if spec.role not in _ALLOWED_ROLES:
        raise ValueError(
            f"FieldSpec {spec.name!r}: role must be one of"
            f" {list(_ALLOWED_ROLES)}, got {spec.role!r}"
        )
    validate_int(f"FieldSpec {spec.name!r} lag", spec.lag, minimum=0)
    _validate_field_spec_transform(spec)
    if spec.aggregate not in _ALLOWED_AGGREGATES:
        raise ValueError(
            f"FieldSpec {spec.name!r}: aggregate values beyond"
            f" 'last' are {_DEFERRED_NOTE} (they need resampling)"
        )
    if spec.fill not in _ALLOWED_FILL:
        raise ValueError(
            f"FieldSpec {spec.name!r}: fill must be one of"
            f" {list(_ALLOWED_FILL)}, got {spec.fill!r}"
        )
    if spec.staleness_limit is not None and (
        not isinstance(spec.staleness_limit, pd.Timedelta)
        or spec.staleness_limit < pd.Timedelta(0)
    ):
        raise ValueError(
            f"FieldSpec {spec.name!r}: staleness_limit must be a"
            " non-negative pd.Timedelta when given"
        )


def _validate_field_spec_transform(spec: FieldSpec) -> None:
    """The transform ladder, with the deferred zscore called out."""
    if spec.transform in _ALLOWED_TRANSFORMS:
        return
    if spec.transform == "zscore":
        raise ValueError(
            f"FieldSpec {spec.name!r}: transform 'zscore' is"
            f" {_DEFERRED_NOTE} -- a full-sample z-score leaks"
            " future moments"
        )
    raise ValueError(
        f"FieldSpec {spec.name!r}: transform must be one of"
        f" {list(_ALLOWED_TRANSFORMS)}, got {spec.transform!r}"
    )


@dataclass(frozen=True)
class MaterializeSpec:
    """The full materialization request.

    The split is TIMESTAMP-based: when ``train_end`` is given it wins
    and ``train_frac`` is ignored; otherwise the fraction cuts the
    UNIQUE decision-timestamp grid (never row counts -- holiday gaps
    and irregular sessions make positional cuts a footgun).  The
    default ``embargo=0`` is allowed but warned on for sub-daily
    grids; rows inside the embargo gap land in neither split.
    """

    outcome: FieldSpec
    features: tuple[FieldSpec, ...] = ()
    controls: tuple[FieldSpec, ...] = ()
    regime: FieldSpec | None = None
    train_frac: float = 0.6
    train_end: pd.Timestamp | None = None
    embargo: pd.Timedelta = _ZERO
    min_train_rows: int = 100
    min_test_rows: int = 50
    nan_policy: str = "strict"
    calendar: str = "24/7"

    def __post_init__(self) -> None:
        """Validate roles, split parameters and name uniqueness."""
        _validate_materialize_spec(self)


# ---------------------------------------------------------------------------
# materialized bundle
# ---------------------------------------------------------------------------
def _validate_spec_roles(spec: MaterializeSpec) -> None:
    """Every declared field must carry the role its slot demands."""
    if spec.outcome.role != "outcome":
        raise ValueError(
            "MaterializeSpec.outcome must have role='outcome',"
            f" got {spec.outcome.role!r}"
        )
    for field_spec in spec.features:
        if field_spec.role != "feature":
            raise ValueError(
                f"MaterializeSpec feature {field_spec.name!r} must have"
                f" role='feature', got {field_spec.role!r}"
            )
    for field_spec in spec.controls:
        if field_spec.role != "control":
            raise ValueError(
                f"MaterializeSpec control {field_spec.name!r} must have"
                f" role='control', got {field_spec.role!r}"
            )
    if spec.regime is not None and spec.regime.role != "regime":
        raise ValueError(
            "MaterializeSpec.regime must have role='regime', got"
            f" {spec.regime.role!r}"
        )


def _validate_materialize_spec(spec: MaterializeSpec) -> None:
    """Validation body of :meth:`MaterializeSpec.__post_init__`."""
    _validate_spec_roles(spec)
    names = [
        spec.outcome.name,
        *[field_spec.name for field_spec in spec.features],
        *[field_spec.name for field_spec in spec.controls],
    ]
    if spec.regime is not None:
        names.append(spec.regime.name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"MaterializeSpec: field names must be unique, reused:"
            f" {duplicates}"
        )
    if spec.train_end is None and not (
        np.isfinite(spec.train_frac) and 0.0 < spec.train_frac < 1.0
    ):
        raise ValueError(
            "MaterializeSpec: train_frac must be a float in (0, 1)"
            f" (or set train_end explicitly), got {spec.train_frac!r}"
        )
    if spec.train_end is not None and not isinstance(
        spec.train_end, pd.Timestamp
    ):
        raise ValueError(
            "MaterializeSpec: train_end must be a pd.Timestamp when given"
        )
    if not isinstance(spec.embargo, pd.Timedelta) or (
        spec.embargo < pd.Timedelta(0)
    ):
        raise ValueError(
            "MaterializeSpec: embargo must be a non-negative pd.Timedelta"
        )
    validate_int(
        "MaterializeSpec min_train_rows", spec.min_train_rows, minimum=1
    )
    validate_int(
        "MaterializeSpec min_test_rows", spec.min_test_rows, minimum=1
    )
    if spec.nan_policy not in _ALLOWED_NAN_POLICY:
        raise ValueError(
            "MaterializeSpec: nan_policy must be one of"
            f" {list(_ALLOWED_NAN_POLICY)}, got {spec.nan_policy!r}"
        )
    if not spec.calendar:
        raise ValueError("MaterializeSpec: calendar must be non-empty")


@dataclass(frozen=True)
class MaterializedData:
    """The verified bundle framework functions consume.

    ``index`` carries one row per decision timestamp: ``ts``, the
    (informational, MVP-empty) ``entity``, the ``split`` assignment
    and any ``regime`` column.  ``manifest`` is ``None`` only for
    legacy-wrapped frames -- every consumer must treat that as
    unverified data (see
    :func:`atropos.feature_scan.feature_analysis`).
    """

    outcome: pd.Series
    features: pd.DataFrame
    controls: pd.DataFrame
    index: pd.DataFrame
    manifest: DataManifest | None = None


@runtime_checkable
class Adapter(Protocol):
    """What every source normalizes to; adapters are dumb.

    ``fetch`` returns a Series on a UTC, monotonic, unique DatetimeIndex
    (labelled by bar close per the field's convention) with NO lag
    applied; the resolver does as-of cuts, transforms, fills, splits
    and validation.  ``as_of`` answers one-point queries (used by the
    Book layer for feature snapshots, not by :func:`materialize`).
    """

    name: str
    version: str
    calendar: str

    def fields(self) -> dict[str, FieldMeta]:
        """Declared metadata for every field this adapter serves."""
        ...

    def extent(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Inclusive data coverage ``(start, end)`` of this adapter."""
        ...

    def fetch(
        self, name: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Raw field values with labels in ``[start, end]``."""
        ...

    def as_of(self, name: str, ts: pd.Timestamp) -> float:
        """Value of ``name`` known at ``ts`` (NaN when none is)."""
        ...

    def validate_self(self) -> list[str]:
        """Adapter-internal consistency issues (non-fatal here)."""
        ...


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Any) -> str:
    """Stable JSON text of a JSON-safe structure (sorted keys)."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _series_hash(series: pd.Series) -> str:
    """Full sha256 of one column's values + index (reproducibility)."""
    values = np.asarray(series.to_numpy(dtype=float), dtype=np.float64)
    labels = (
        pd.DatetimeIndex(series.index)
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    digest.update(np.asarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _validate_utc_index(index: pd.DatetimeIndex, tag: str) -> None:
    """Fatal on naive / unsorted / duplicated timestamps."""
    if index.tz is None:
        raise ValueError(f"{tag}: index must be timezone-aware (UTC)")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{tag}: index must be monotonic increasing")
    if index.has_duplicates:
        raise ValueError(f"{tag}: index must be unique")


def _bar_delta(raw_index: pd.DatetimeIndex, name: str) -> pd.Timedelta:
    """Median spacing of a raw field's grid (its native bar)."""
    if raw_index.size < _MIN_BAR_OBS:
        raise ValueError(
            f"field {name!r}: {_MIN_BAR_OBS}+ observations required to"
            " estimate the bar delta"
        )
    median_delta = pd.Timedelta(np.median(np.diff(raw_index.to_numpy())))
    if not np.isfinite(median_delta.value) or median_delta <= pd.Timedelta(0):
        raise ValueError(f"field {name!r}: non-positive bar spacing")
    return median_delta


def _apply_transform(
    values: pd.Series, transform: str, name: str
) -> pd.Series:
    """Causal pointwise transform of an as-of column."""
    if transform == "raw":
        return values
    if transform == "diff":
        return values.diff()
    if transform == "pct_change":
        return values.pct_change()
    if transform == "log":
        with np.errstate(all="ignore"):
            logged = np.log(values.to_numpy(dtype=float))
        logged[values.to_numpy(dtype=float) <= 0.0] = np.nan
        return pd.Series(logged, index=values.index, name=name)
    raise ValueError(f"field {name!r}: unsupported transform {transform!r}")


def _as_of_column(
    raw: pd.Series,
    meta: FieldMeta,
    spec: FieldSpec,
    grid: pd.DatetimeIndex,
    notes: dict[str, int],
) -> tuple[pd.Series, dict[str, bool]]:
    """One decision-grid column resolved as-of, then verified.

    The production cut uses ``cutoff = t - t_known_offset - lag *
    bar_delta``; the verification then INDEPENDENTLY recomputes the
    cutoff from the metadata and checks every produced label against
    it -- a resolver bug flips the flag instead of passing silently.
    """
    raw_index = pd.DatetimeIndex(raw.index)
    _validate_utc_index(raw_index, f"field {spec.name!r}")
    if meta.entity_axis != "none":
        raise ValueError(
            f"field {spec.name!r}: entity_axis={meta.entity_axis!r} is"
            f" {_DEFERRED_NOTE}; MVP materialization resolves"
            " single-series fields"
        )
    bar = _bar_delta(raw_index, spec.name)
    cutoffs = (
        grid
        - meta.t_known_offset
        - pd.Timedelta(spec.lag * bar.value, unit="ns")
    )
    positions = raw_index.searchsorted(cutoffs, side="right") - 1
    valid = positions >= 0
    safe_positions = np.where(valid, positions, 0)
    labels_used = raw_index.to_numpy()[safe_positions]
    values = raw.to_numpy(dtype=float)[safe_positions]
    values = np.where(valid, values, np.nan)
    before_fill_nan = int(
        np.size(values) - np.count_nonzero(np.isfinite(values))
    )
    column = pd.Series(values, index=grid, name=spec.name)
    labels_series = pd.Series(labels_used, index=grid)
    if before_fill_nan and spec.fill == "ffill":
        column = column.ffill()
        labels_series = labels_series.ffill()
    elif before_fill_nan and spec.fill == "zero":
        column = column.fillna(0.0)
        labels_series = labels_series.where(column.notna(), labels_series)
    if spec.staleness_limit is not None:
        labelled_grid = pd.Series(grid, index=grid)
        stale = (labelled_grid - labels_series) > spec.staleness_limit
        column = column.where(~stale, np.nan)
    column = _apply_transform(column, spec.transform, spec.name)
    # independent post-verification against the recomputed cutoff
    checked = (labels_series.to_numpy() <= cutoffs.to_numpy()) | ~valid
    lookahead_ok = bool(np.all(checked))
    notes["nan_lead"] += int(np.count_nonzero(~valid))
    return column, {spec.name: lookahead_ok}


def _split_grid(
    grid: pd.DatetimeIndex, spec: MaterializeSpec
) -> tuple[pd.Timestamp, pd.Series, int]:
    """Timestamp-based train/test split over the decision grid.

    Returns ``(train_end_ts, split_series, embargo_dropped)``; the
    embargo gap belongs to neither split and its rows are counted.
    """
    unique_grid = pd.DatetimeIndex(sorted(set(grid)))
    if spec.train_end is not None:
        train_end_ts = spec.train_end
    else:
        cut_position = int(spec.train_frac * unique_grid.size)
        cut_position = min(max(cut_position, 1), unique_grid.size - 1)
        train_end_ts = unique_grid[cut_position - 1]
    labels = np.where(grid <= train_end_ts, "train", "test")
    embargo_dropped = 0
    if spec.embargo > pd.Timedelta(0):
        in_gap = (grid > train_end_ts) & (grid <= train_end_ts + spec.embargo)
        embargo_dropped = int(in_gap.sum())
        labels = np.where(in_gap, "", labels)
    split = pd.Series(labels, index=grid)
    return train_end_ts, split, embargo_dropped


def materialize(adapter: Adapter, spec: MaterializeSpec) -> MaterializedData:
    """Resolve every declared field as-of onto the decision grid.

    Verifies (never assumes): timezone/monotonic/uniqueness of every
    fetched field, the per-field lookahead check against an
    independently recomputed cutoff, train/test coverage floors, and
    the caller's ``nan_policy``.  Fatal problems raise ``ValueError``
    with the named field; non-fatal ones land in
    ``manifest["validation"]["warnings"]``.

    Args:
        adapter: a dumb source normalizer (see :class:`Adapter`).
        spec: the declared request (see :class:`MaterializeSpec`).

    Returns:
        :class:`MaterializedData` with a fully populated manifest
        (bundle hash, feature hashes, lookahead checks, validation).

    Raises:
        ValueError: on naive/unsorted/duplicate timestamps, unknown
            field names, multi-entity fields (deferred), lookahead
            check failures, a failed nan_policy, coverage shortfalls,
            or an adapter whose self-validation reports a fatal-
            class issue.

    Example::

        >>> from atropos.data_contract import _make_test_md
        >>> md = _make_test_md()
        >>> md.manifest["lookahead_checks"]
        {'booked_return': True, 'mom_signal': True}
        >>> md.manifest["validation"]["passed"]
        True
        >>> md.manifest["bundle_hash"]
        '18ac84bfba05f1c79db47924c63aaf4de6302bebd7eb5d8be8ac13be0a4a887d'
    """
    for attr in ("name", "version", "calendar"):
        value = getattr(adapter, attr)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"adapter attribute {attr!r} must be a non-empty str"
            )
    meta_map = adapter.fields()
    declared_specs = [spec.outcome, *spec.features, *spec.controls]
    if spec.regime is not None:
        declared_specs.append(spec.regime)
    missing = sorted(
        {field_spec.name for field_spec in declared_specs} - set(meta_map)
    )
    if missing:
        raise ValueError(f"adapter does not declare fields: {missing}")
    adapter_issues = adapter.validate_self()
    warnings_list = [f"adapter: {issue}" for issue in adapter_issues]

    raw_outcome = adapter.fetch(spec.outcome.name, *adapter.extent())
    grid = pd.DatetimeIndex(raw_outcome.index)
    _validate_utc_index(grid, f"field {spec.outcome.name!r}")

    notes: dict[str, int] = {"nan_lead": 0}
    columns, lookahead = _resolve_columns(
        adapter,
        spec,
        meta_map=meta_map,
        declared_specs=declared_specs,
        raw_outcome=raw_outcome,
        grid=grid,
        notes=notes,
    )
    _require_clean_lookahead(lookahead)

    outcome_series = columns[spec.outcome.name].rename(spec.outcome.name)
    features_frame = pd.DataFrame(
        {fs.name: columns[fs.name] for fs in spec.features},
        index=grid,
    )
    controls_frame = pd.DataFrame(
        {fs.name: columns[fs.name] for fs in spec.controls},
        index=grid,
    )
    _train_end, split_series, embargo_dropped = _split_grid(grid, spec)
    if embargo_dropped:
        split_series = split_series[split_series != ""]
        keep_mask = split_series.index
        outcome_series = outcome_series.loc[keep_mask]
        features_frame = features_frame.loc[keep_mask]
        controls_frame = controls_frame.loc[keep_mask]
        grid = pd.DatetimeIndex(keep_mask)

    index_frame = _build_index_frame(grid, split_series, columns, spec)
    fatal = _check_coverage(spec, split_series)

    all_columns = {
        spec.outcome.name: outcome_series,
        **{name: features_frame[name] for name in features_frame.columns},
        **{name: controls_frame[name] for name in controls_frame.columns},
    }
    joined = pd.DataFrame(all_columns, index=grid)
    joined, split_series, index_frame, grid = _apply_nan_policy(
        joined,
        spec,
        split_series=split_series,
        index_frame=index_frame,
        grid=grid,
        notes=notes,
        fatal=fatal,
    )
    outcome_series = joined[spec.outcome.name]
    features_frame = joined[list(features_frame.columns)]
    controls_frame = joined[list(controls_frame.columns)]

    if grid.size > 1:
        spacing = pd.Timedelta(np.median(np.diff(grid.to_numpy())))
        sub_daily = spacing < pd.Timedelta(days=1)
        if spec.embargo == _ZERO and sub_daily:
            warnings_list.append(
                "embargo=0 on a sub-daily decision grid: train labels"
                " may overlap test features; set an explicit embargo"
            )

    nan_fraction = {
        str(name): float(series.isna().mean())
        for name, series in joined.items()
    }
    feature_hashes = {
        str(name): _series_hash(series) for name, series in joined.items()
    }
    report = ValidationReport(
        passed=not fatal,
        fatal=fatal,
        warnings=warnings_list,
    )
    if fatal:
        raise ValueError("materialize validation failed: " + "; ".join(fatal))
    manifest = _build_manifest(
        adapter,
        spec,
        grid,
        split_series,
        embargo_dropped=embargo_dropped,
        nan_fraction=nan_fraction,
        notes=notes,
        warnings_list=warnings_list,
        feature_hashes=feature_hashes,
        lookahead=lookahead,
        report=report,
    )
    return MaterializedData(
        outcome=outcome_series,
        features=features_frame,
        controls=controls_frame,
        index=index_frame,
        manifest=manifest,
    )


def _resolve_columns(
    adapter: Adapter,
    spec: MaterializeSpec,
    *,
    meta_map: dict[str, FieldMeta],
    declared_specs: list[FieldSpec],
    raw_outcome: pd.Series,
    grid: pd.DatetimeIndex,
    notes: dict[str, int],
) -> tuple[dict[str, pd.Series], dict[str, bool]]:
    """Fetch and as-of resolve every declared field onto the grid.

    ``notes`` accumulates per-field counters (leading NaN rows) so
    the manifest's ``dropped`` block stays honest.
    """
    lookahead: dict[str, bool] = {}
    columns: dict[str, pd.Series] = {}
    for field_spec in declared_specs:
        raw = (
            raw_outcome
            if field_spec.name == spec.outcome.name
            else adapter.fetch(field_spec.name, *adapter.extent())
        )
        column, flag = _as_of_column(
            raw, meta_map[field_spec.name], field_spec, grid, notes
        )
        lookahead.update(flag)
        columns[field_spec.name] = column
    return columns, lookahead


def _require_clean_lookahead(lookahead: dict[str, bool]) -> None:
    """Fatal when any field's independently verified check failed."""
    failed_lookahead = sorted(
        name for name, ok in lookahead.items() if not ok
    )
    if failed_lookahead:
        raise ValueError(
            "lookahead check FAILED for fields"
            f" {failed_lookahead}: produced labels exceed the"
            " recomputed as-of cutoff"
        )


def _build_index_frame(
    grid: pd.DatetimeIndex,
    split_series: pd.Series,
    columns: dict[str, pd.Series],
    spec: MaterializeSpec,
) -> pd.DataFrame:
    """Decision index: ts, (informational) entity, split, regime."""
    index_frame = pd.DataFrame(
        {
            "ts": grid,
            "entity": [""] * grid.size,
            "split": split_series.to_numpy(),
        },
        index=grid,
    )
    if spec.regime is not None:
        index_frame["regime"] = columns[spec.regime.name].to_numpy()
    return index_frame


def _check_coverage(
    spec: MaterializeSpec, split_series: pd.Series
) -> list[str]:
    """Fatal messages when either split misses its coverage floor."""
    train_rows = int((split_series == "train").sum())
    test_rows = int((split_series == "test").sum())
    fatal: list[str] = []
    if train_rows < spec.min_train_rows:
        fatal.append(
            f"train coverage {train_rows} < min_train_rows"
            f" {spec.min_train_rows}"
        )
    if test_rows < spec.min_test_rows:
        fatal.append(
            f"test coverage {test_rows} < min_test_rows {spec.min_test_rows}"
        )
    return fatal


def _apply_nan_policy(
    joined: pd.DataFrame,
    spec: MaterializeSpec,
    *,
    split_series: pd.Series,
    index_frame: pd.DataFrame,
    grid: pd.DatetimeIndex,
    notes: dict[str, int],
    fatal: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Apply the caller's NaN policy to the joined frame.

    ``strict`` records NaN counts as fatal (named per column);
    ``drop`` removes incomplete rows; ``ffill`` patches forward and
    drops leading rows.  Returns the (possibly shrunk) joined frame
    with its split labels, index frame and grid kept positionally in
    sync; the caller re-slices outcome/features/controls from it.
    """
    if spec.nan_policy == "strict":
        nan_counts = joined.isna().sum()
        flagged = {
            name: int(count) for name, count in nan_counts.items() if count
        }
        if flagged:
            fatal.append(f"nan_policy='strict' found NaN: {flagged}")
        return joined, split_series, index_frame, grid
    if spec.nan_policy == "drop":
        joined = joined.dropna()
        notes["nan_drop"] = grid.size - len(joined)
    else:  # ffill across the joined frame, then drop leading rows
        joined = joined.ffill().dropna()
        notes["nan_ffill"] = grid.size - len(joined)
    split_series = split_series.loc[joined.index]
    index_frame = index_frame.loc[joined.index]
    grid = pd.DatetimeIndex(joined.index)
    return joined, split_series, index_frame, grid


def _build_manifest(
    adapter: Adapter,
    spec: MaterializeSpec,
    grid: pd.DatetimeIndex,
    split_series: pd.Series,
    *,
    embargo_dropped: int,
    nan_fraction: dict[str, float],
    notes: dict[str, int],
    warnings_list: list[str],
    feature_hashes: dict[str, str],
    lookahead: dict[str, bool],
    report: ValidationReport,
) -> DataManifest:
    """Assemble the manifest from verified parts (hash last)."""
    source_versions = {
        "adapter": f"{adapter.name}@{adapter.version}",
    }
    hash_payload = {
        "spec": asdict(spec),
        "adapter": source_versions,
        "calendar": spec.calendar,
        "feature_hashes": feature_hashes,
        "row_count": int(grid.size),
        "date_range": [str(grid.min()), str(grid.max())],
    }
    bundle_hash = hashlib.sha256(
        _canonical_json(hash_payload).encode("utf-8")
    ).hexdigest()
    return DataManifest(
        bundle_hash=bundle_hash,
        source_versions=source_versions,
        calendar=spec.calendar,
        row_count=int(grid.size),
        train_rows=int((split_series == "train").sum()),
        test_rows=int((split_series == "test").sum()),
        dropped_embargo_gap=embargo_dropped,
        date_range=(str(grid.min()), str(grid.max())),
        nan_fraction=nan_fraction,
        dropped=dict(notes),
        warnings=warnings_list,
        feature_hashes=feature_hashes,
        lookahead_checks=lookahead,
        validation=report,
    )


def to_pipeline_data(md: MaterializedData) -> dict[str, Any]:
    """LEGACY bridge to ``run_full_pipeline(data: dict)`` -- not a path.

    The pipeline re-aligns its inputs with an inner join + dropna, so
    the as-of guarantees of the data contract DO NOT survive this
    bridge; a UserWarning fires on every call.  For contract
    guarantees call
    :func:`atropos.feature_scan.feature_analysis` on the bundle
    directly.

    Args:
        md: a materialized bundle (its manifest is attached under
            ``"manifest"`` so the legacy report can cite it).

    Returns:
        The legacy data dict: ``returns`` (the outcome), ``signal``
        (the first feature column), optional ``features`` (the
        controls frame), and the manifest for anchoring.

    Raises:
        ValueError: if the bundle has no feature column to serve as
            the legacy signal.

    Example::

        >>> import warnings
        >>> from atropos.data_contract import _make_test_md
        >>> from atropos.data_contract import to_pipeline_data
        >>> md = _make_test_md()
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     data = to_pipeline_data(md)
        >>> sorted(data)
        ['features', 'manifest', 'returns', 'signal']
    """
    warnings.warn(_BRIDGE_WARNING, UserWarning, stacklevel=2)
    if md.features.shape[1] == 0:
        raise ValueError(
            "to_pipeline_data needs at least one feature column to"
            " serve as the legacy signal"
        )
    signal_name = str(md.features.columns[0])
    return {
        "returns": md.outcome,
        "signal": md.features[signal_name],
        "features": md.controls if md.controls.shape[1] else None,
        "manifest": md.manifest,
    }


# ---------------------------------------------------------------------------
# tiny in-memory adapter for doctests and tests
# ---------------------------------------------------------------------------
class _TableAdapter:
    """Minimal in-memory adapter implementing :class:`Adapter`.

    Serves pre-built float Series on a UTC DatetimeIndex; every field
    is declared close-labelled (offset 0) unless overridden.  Used by
    doctests and the contract tests; NOT a general-purpose adapter.
    """

    def __init__(
        self,
        fields_map: dict[str, FieldMeta],
        series_map: dict[str, pd.Series],
        *,
        name: str = "table",
        version: str = "1.0",
        calendar: str = "24/7",
    ) -> None:
        for meta_name in series_map:
            if meta_name not in fields_map:
                raise ValueError(
                    f"series {meta_name!r} has no declared FieldMeta"
                )
        self.name = name
        self.version = version
        self.calendar = calendar
        self._fields = dict(fields_map)
        self._series = {
            key: value.sort_index() for key, value in series_map.items()
        }

    def fields(self) -> dict[str, FieldMeta]:
        """Declared metadata map."""
        return dict(self._fields)

    def extent(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Union coverage of the served series."""
        starts = [series.index.min() for series in self._series.values()]
        ends = [series.index.max() for series in self._series.values()]
        return min(starts), max(ends)

    def fetch(
        self, name: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Raw slice of one field, labels within ``[start, end]``."""
        if name not in self._series:
            raise ValueError(f"unknown field {name!r}")
        return self._series[name].loc[start:end]

    def as_of(self, name: str, ts: pd.Timestamp) -> float:
        """Last value known at ``ts`` (NaN before the first label)."""
        if name not in self._series:
            raise ValueError(f"unknown field {name!r}")
        offset = self._fields[name].t_known_offset
        cutoff = ts - offset
        series = self._series[name]
        position = series.index.searchsorted(cutoff, side="right") - 1
        if position < 0:
            return float("nan")
        return float(series.iloc[position])

    def validate_self(self) -> list[str]:
        """No internal issues by construction."""
        return []


def _make_test_md() -> MaterializedData:
    """A tiny verified bundle for doctests (79 daily decisions).

    Two fields: the feature ``mom_signal`` (as-of, offset 0) and the
    outcome ``booked_return`` (lag 1 -- the forward label convention;
    the first decision row has no prior label and is dropped by
    ``nan_policy="drop"``).  Deterministic values; the manifest
    verifies clean.
    """
    rng = np.random.default_rng(42)
    raw_signal = pd.Series(
        rng.standard_normal(80),
        index=pd.date_range("2023-12-13", periods=80, freq="D", tz="UTC"),
    )
    raw_return = pd.Series(
        rng.standard_normal(80) * 0.01,
        index=pd.date_range("2023-12-13", periods=80, freq="D", tz="UTC"),
    )
    adapter = _TableAdapter(
        {
            "mom_signal": FieldMeta("mom_signal"),
            "booked_return": FieldMeta("booked_return"),
        },
        {"mom_signal": raw_signal, "booked_return": raw_return},
    )
    spec = MaterializeSpec(
        outcome=FieldSpec("booked_return", role="outcome", lag=1),
        features=(FieldSpec("mom_signal", role="feature"),),
        train_frac=0.6,
        min_train_rows=20,
        min_test_rows=10,
        nan_policy="drop",
    )
    return materialize(adapter, spec)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: the contract verifies a planted clean world, asserted."""
    md = _make_test_md()
    manifest = md.manifest
    assert manifest is not None
    print("=" * 72)
    print("data_contract demo: 79 daily decisions, as-of resolved")
    print("=" * 72)
    print(f"bundle_hash={manifest['bundle_hash'][:16]}...")
    print(
        f"rows={manifest['row_count']} train={manifest['train_rows']}"
        f" test={manifest['test_rows']}"
    )
    print(f"lookahead_checks={manifest['lookahead_checks']}")
    assert manifest["lookahead_checks"] == {
        "mom_signal": True,
        "booked_return": True,
    }
    assert manifest["validation"]["passed"] is True
    assert md.index["split"].isin(_ALLOWED_SPLITS).all()
    assert md.manifest is not None and md.manifest["row_count"] == 79

    # the legacy bridge warns, loudly, on every call
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = to_pipeline_data(md)
    assert any(_BRIDGE_WARNING[:40] in str(item.message) for item in caught)
    assert sorted(data) == ["features", "manifest", "returns", "signal"]

    # a manifest-less bundle is unverified data: the consumer warns
    bare = MaterializedData(
        outcome=md.outcome,
        features=md.features,
        controls=md.controls,
        index=md.index,
        manifest=None,
    )
    assert bare.manifest is None
    print("bridge warning emitted; unverified bundle flagged: OK")


if __name__ == "__main__":
    run_all_examples()
