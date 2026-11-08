"""Data-contract pins: look-ahead is caught, not assumed.

Every world is deterministic (seed 42-45) synthetic tables through
``_TableAdapter``; each test names the mutation it must catch (the
mutation plan; every mutation is applied -> fail -> restored before
commit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.data_contract import (
    DataManifest,
    FieldMeta,
    FieldSpec,
    MaterializedData,
    MaterializeSpec,
    ValidationReport,
    _make_test_md,
    _TableAdapter,
    materialize,
    to_pipeline_data,
)
from atropos.feature_scan import feature_analysis

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
_VALIDATION_KEYS = frozenset({"passed", "fatal", "warnings"})


def _daily(
    name: str, periods: int, value: float, start: str = "2024-01-01"
) -> pd.Series:
    """One constant daily UTC series (deterministic worlds)."""
    return pd.Series(
        np.full(periods, value),
        index=pd.date_range(start, periods=periods, freq="D", tz="UTC"),
        name=name,
    )


def _adapter(
    series_map: dict[str, pd.Series],
    offsets: dict[str, pd.Timedelta] | None = None,
) -> _TableAdapter:
    """Adapter over constant series with optional declared offsets."""
    meta_map = {
        name: FieldMeta(
            name,
            t_known_offset=(offsets or {}).get(name, pd.Timedelta(0)),
        )
        for name in series_map
    }
    return _TableAdapter(meta_map, series_map)


def _spec(**overrides: object) -> MaterializeSpec:
    """Standard spec: one feature, one outcome, generous floors."""
    params: dict[str, object] = {
        "outcome": FieldSpec("ret", role="outcome"),
        "features": (FieldSpec("mom", role="feature"),),
        "min_train_rows": 10,
        "min_test_rows": 5,
    }
    params.update(overrides)
    return MaterializeSpec(**params)  # type: ignore[arg-type]


def test_lookahead_offset_shifts_and_verifies() -> None:
    """A declared offset moves the as-of cut and stays verified.

    World: feature known one day BEFORE its label (offset -1d).  At
    each decision ts the as-of value must be the NEXT label's value
    relative to the offset-0 world.  The independent post-check keeps
    the bundle verified.  Mutation fact: dropping t_known_offset from
    the PRODUCTION cutoff trips the independent verification and
    materialize raises -- this test pins the offset-0 world being
    refused-free and the shifted world's exact values.
    """
    raw_mom = _daily("mom", 30, 1.0)
    raw_mom.iloc[10] = 7.0  # one distinctive bar (label Jan 11)
    raw_ret = _daily("ret", 30, 0.5)
    base = materialize(
        _adapter({"mom": raw_mom, "ret": raw_ret}),
        _spec(),
    )
    shifted = materialize(
        _adapter(
            {"mom": raw_mom, "ret": raw_ret},
            offsets={"mom": pd.Timedelta(days=-1)},
        ),
        _spec(),
    )
    assert base.manifest is not None and shifted.manifest is not None
    assert base.manifest["lookahead_checks"] == {"mom": True, "ret": True}
    # as-of takes the LAST label <= cutoff: offset 0 -> decision Jan 11
    # sees label Jan 11; offset -1d -> decision Jan 10 already sees it
    boundary = pd.Timestamp("2024-01-11", tz="UTC")
    earlier = pd.Timestamp("2024-01-10", tz="UTC")
    assert float(base.features.loc[boundary, "mom"]) == 7.0
    assert float(base.features.loc[earlier, "mom"]) == 1.0
    assert float(shifted.features.loc[earlier, "mom"]) == 7.0
    assert float(shifted.features.loc[boundary, "mom"]) == 1.0


def test_lag_applied_after_asof() -> None:
    """lag=1 shifts the column by exactly one bar (test the arithmetic).

    Pins: outcome lag=0 mirrors raw labels; lag=1 delivers raw[t-1d].
    Mutation fact: dropping the lag term from the cutoff makes lag=1
    identical to lag=0 and the shifted assertion fails.
    """
    raw_ret = _daily("ret", 40, 0.5)
    raw_ret.iloc[20] = 3.0
    lagged = materialize(
        _adapter({"ret": raw_ret}),
        MaterializeSpec(
            outcome=FieldSpec("ret", role="outcome", lag=1),
            min_train_rows=10,
            min_test_rows=5,
            nan_policy="drop",
        ),
    )
    boundary = pd.Timestamp("2024-01-21", tz="UTC")  # grid day 20
    assert float(lagged.outcome.loc[boundary]) == 0.5  # raw[19], not 3.0
    unlagged = materialize(
        _adapter({"ret": raw_ret}),
        MaterializeSpec(
            outcome=FieldSpec("ret", role="outcome"),
            min_train_rows=10,
            min_test_rows=5,
        ),
    )
    assert float(unlagged.outcome.loc[boundary]) == 3.0


def test_nan_policy_strict_fatal() -> None:
    """A strict policy rejects NaN loudly, naming the field.

    World: the feature starts one day after the outcome, so the first
    decision row has no as-of value.  Mutation fact: deleting the
    strict-branch raise makes the NaN pass silently and the test
    fails on the expected exception.
    """
    raw_ret = _daily("ret", 30, 0.5)
    raw_mom = _daily("mom", 29, 1.0, start="2024-01-02")
    with pytest.raises(ValueError, match="strict"):
        materialize(
            _adapter({"ret": raw_ret, "mom": raw_mom}),
            _spec(nan_policy="strict"),
        )


def test_nan_policy_drop_and_ffill() -> None:
    """Drop removes incomplete rows; ffill patches interior gaps.

    Pins: drop shrinks row_count by exactly the incomplete rows and
    records ``nan_drop``; ffill fills an interior NaN and records
    ``nan_ffill``.  Mutation fact: a drop branch that forgets to
    propagate the shrink to the index frame changes row_count
    inconsistently and the counts assert fails.
    """
    raw_ret = _daily("ret", 30, 0.5)
    raw_mom = _daily("mom", 29, 1.0, start="2024-01-02")
    dropped = materialize(
        _adapter({"ret": raw_ret, "mom": raw_mom}),
        _spec(nan_policy="drop"),
    )
    assert dropped.manifest is not None
    assert dropped.manifest["row_count"] == 29
    assert dropped.manifest["dropped"]["nan_drop"] == 1

    raw_holey = _daily("mom", 30, 1.0)
    raw_holey.iloc[10] = np.nan
    filled = materialize(
        _adapter({"ret": raw_ret, "mom": raw_holey}),
        _spec(nan_policy="ffill"),
    )
    assert filled.manifest is not None
    assert filled.manifest["row_count"] == 30
    assert filled.manifest["dropped"]["nan_ffill"] >= 0
    assert float(filled.features["mom"].iloc[10]) == 1.0  # raw[9] value


def test_split_timestamp_based_and_embargo() -> None:
    """The split cuts the unique-ts grid; embargo rows land nowhere.

    World: 30 daily decisions, train_frac 0.6, embargo 2 days.  Pins:
    train rows end at the expected timestamp, test rows start after
    the embargo, ``dropped_embargo_gap`` counts the gap rows exactly.
    Mutation fact: ignoring the embargo assigns gap rows to train and
    the gap-count/boundary asserts fail.
    """
    raw = _daily("ret", 30, 0.5)
    raw_mom = _daily("mom", 30, 1.0)
    md = materialize(
        _adapter({"ret": raw, "mom": raw_mom}),
        _spec(embargo=pd.Timedelta(days=2)),
    )
    assert md.manifest is not None
    assert md.manifest["dropped_embargo_gap"] == 2
    splits = md.index["split"]
    train_ts = md.index.loc[splits == "train", "ts"]
    test_ts = md.index.loc[splits == "test", "ts"]
    expected_cut = int(0.6 * 30)
    assert train_ts.max() == pd.Timestamp(
        "2024-01-01", tz="UTC"
    ) + pd.Timedelta(days=expected_cut - 1)
    assert test_ts.min() == train_ts.max() + pd.Timedelta(days=3)
    assert md.manifest["train_rows"] + md.manifest["test_rows"] + 2 == 30
    # explicit train_end wins over train_frac
    explicit = materialize(
        _adapter({"ret": raw, "mom": raw_mom}),
        _spec(
            embargo=pd.Timedelta(days=0),
            train_end=pd.Timestamp("2024-01-10", tz="UTC"),
        ),
    )
    assert explicit.manifest is not None
    assert explicit.manifest["train_rows"] == 10


def test_bundle_hash_reacts_to_data() -> None:
    """Changing the DATA changes bundle_hash (prereg anchor works).

    Mutation fact: computing bundle_hash without feature_hashes makes
    the two worlds hash identically and the assert fails.
    """
    raw_a = _daily("ret", 30, 0.5)
    raw_mom_a = _daily("mom", 30, 1.0)
    raw_mom_b = _daily("mom", 30, 1.0)
    raw_mom_b.iloc[5] = 9.0
    md_a = materialize(_adapter({"ret": raw_a, "mom": raw_mom_a}), _spec())
    md_b = materialize(_adapter({"ret": raw_a, "mom": raw_mom_b}), _spec())
    assert md_a.manifest is not None and md_b.manifest is not None
    assert md_a.manifest["bundle_hash"] != md_b.manifest["bundle_hash"]
    assert len(md_a.manifest["bundle_hash"]) == 64


def test_manifest_schema_frozen() -> None:
    """The manifest and validation key sets are frozen (schema guard).

    Guards accidental key drift in either direction at any level.
    """
    raw_ret = _daily("ret", 30, 0.5)
    raw_mom = _daily("mom", 30, 1.0)
    md = materialize(_adapter({"ret": raw_ret, "mom": raw_mom}), _spec())
    assert md.manifest is not None
    assert frozenset(md.manifest) == _MANIFEST_KEYS
    assert frozenset(md.manifest["validation"]) == _VALIDATION_KEYS
    assert md.manifest["validation"]["passed"] is True
    assert md.manifest["validation"]["fatal"] == []


def test_to_pipeline_data_warns() -> None:
    """The legacy bridge warns on EVERY call (caught in the bridge).

    Mutation fact: removing the UserWarning emission (or its text)
    fails the pytest.warns match.
    """
    md = _make_test_md()
    with pytest.warns(UserWarning, match="LEGACY bridge"):
        data = to_pipeline_data(md)
    assert sorted(data) == ["features", "manifest", "returns", "signal"]


def test_feature_analysis_md_manifest_enforcement() -> None:
    """The md path VALIDATES the manifest; unwrap alone is not enough.

    Pins, in order of importance: (a) a bundle whose lookahead check
    was flipped to False is FATAL; (b) a failed validation battery is
    FATAL; (c) a manifest-less bundle analyzes but carries the loud
    unverified-data note; (d) features/controls alongside a bundle
    are rejected; (e) the equality pin: a clean bundle analyzes
    identically to the direct Series call.  Mutation fact: deleting
    the lookahead enforcement in feature_analysis makes (a) analyze
    silently and the pytest.raises fail.
    """
    md = _make_test_md()
    outcome = md.outcome
    features = {name: md.features[name] for name in md.features.columns}

    # (e) equality with the direct call
    direct = feature_analysis(
        outcome,
        features,
        {},
        n_perm_orth=10,
        n_perm_bucket=10,
        min_bucket_n=0,
        label="md-equality",
    )
    via_md = feature_analysis(
        md,
        n_perm_orth=10,
        n_perm_bucket=10,
        min_bucket_n=0,
        label="md-equality",
    )
    assert via_md["classification"] == direct["classification"]

    # (d) extras alongside a bundle are rejected
    with pytest.raises(ValueError, match="must be None"):
        feature_analysis(md, features=features)

    # (c) missing manifest -> loud note, analysis proceeds
    bare = MaterializedData(
        outcome=md.outcome,
        features=md.features,
        controls=md.controls,
        index=md.index,
        manifest=None,
    )
    report_bare = feature_analysis(
        bare,
        n_perm_orth=10,
        n_perm_bucket=10,
        min_bucket_n=0,
        label="md-bare",
    )
    assert any("unverified data" in note for note in report_bare["notes"])

    # (a) failed lookahead check -> fatal
    assert md.manifest is not None
    bad_lookahead: DataManifest = {
        **md.manifest,
        "lookahead_checks": {
            **md.manifest["lookahead_checks"],
            "mom_signal": False,
        },
    }
    md_bad = MaterializedData(
        outcome=md.outcome,
        features=md.features,
        controls=md.controls,
        index=md.index,
        manifest=bad_lookahead,
    )
    with pytest.raises(ValueError, match="lookahead check FAILED"):
        feature_analysis(md_bad)

    # (b) failed validation battery -> fatal
    bad_validation: DataManifest = {
        **md.manifest,
        "validation": ValidationReport(
            passed=False,
            fatal=["train coverage 3 < 100"],
            warnings=[],
        ),
    }
    md_invalid = MaterializedData(
        outcome=md.outcome,
        features=md.features,
        controls=md.controls,
        index=md.index,
        manifest=bad_validation,
    )
    with pytest.raises(ValueError, match="validation failed"):
        feature_analysis(md_invalid)


def test_materialize_unknown_field_and_guards() -> None:
    """Contract guards fail loudly (guards for the C1 battery).

    Pins: an unknown spec field, duplicate spec names, a role
    mismatch, a bad train_frac, a naive adapter index and a
    multi-entity field all raise ValueError.  Mutation fact: relaxing
    any guard makes the corresponding block silently accept garbage.
    """
    raw_ret = _daily("ret", 30, 0.5)
    raw_mom = _daily("mom", 30, 1.0)
    adapter = _adapter({"ret": raw_ret, "mom": raw_mom})
    with pytest.raises(ValueError, match="does not declare"):
        materialize(
            adapter,
            _spec(features=(FieldSpec("ghost", role="feature"),)),
        )
    with pytest.raises(ValueError, match="unique"):
        materialize(
            adapter,
            MaterializeSpec(
                outcome=FieldSpec("ret", role="outcome"),
                features=(
                    FieldSpec("mom", role="feature"),
                    FieldSpec("mom", role="feature"),
                ),
                min_train_rows=10,
                min_test_rows=5,
            ),
        )
    with pytest.raises(ValueError, match="role"):
        materialize(
            adapter,
            _spec(features=(FieldSpec("mom", role="control"),)),
        )
    with pytest.raises(ValueError, match="train_frac"):
        materialize(
            adapter,
            _spec(train_frac=1.5),
        )
    naive = pd.Series(
        np.ones(30), index=pd.date_range("2024-01-01", periods=30)
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        materialize(
            _TableAdapter({"ret": FieldMeta("ret")}, {"ret": naive}),
            MaterializeSpec(
                outcome=FieldSpec("ret", role="outcome"),
                min_train_rows=10,
                min_test_rows=5,
            ),
        )
    entity_meta = FieldMeta("mom", entity_axis="symbol")
    with pytest.raises(ValueError, match="deferred"):
        materialize(
            _TableAdapter(
                {"ret": FieldMeta("ret"), "mom": entity_meta},
                {"ret": raw_ret, "mom": raw_mom},
            ),
            _spec(),
        )
    with pytest.raises(ValueError, match="zscore"):
        FieldSpec("mom", role="feature", transform="zscore")


def test_embargo_footgun_warning() -> None:
    """embargo=0 on a sub-daily grid warns; embargo>0 does not.

    Pins the footgun guard from the plan (review item 4): the warning
    lives in the manifest's validation warnings, never blocks.
    """
    raw_ts = pd.date_range(
        "2024-01-01 00:00", periods=120, freq="h", tz="UTC"
    )
    raw_ret = pd.Series(np.full(120, 0.5), index=raw_ts, name="ret")
    raw_mom = pd.Series(np.full(120, 1.0), index=raw_ts, name="mom")
    md_zero = materialize(
        _adapter({"ret": raw_ret, "mom": raw_mom}),
        MaterializeSpec(
            outcome=FieldSpec("ret", role="outcome"),
            features=(FieldSpec("mom", role="feature"),),
            min_train_rows=20,
            min_test_rows=10,
        ),
    )
    assert md_zero.manifest is not None
    assert any(
        "embargo=0" in warning
        for warning in md_zero.manifest["validation"]["warnings"]
    )
    md_guarded = materialize(
        _adapter({"ret": raw_ret, "mom": raw_mom}),
        MaterializeSpec(
            outcome=FieldSpec("ret", role="outcome"),
            features=(FieldSpec("mom", role="feature"),),
            embargo=pd.Timedelta(hours=4),
            min_train_rows=20,
            min_test_rows=10,
        ),
    )
    assert md_guarded.manifest is not None
    assert not any(
        "embargo=0" in warning
        for warning in md_guarded.manifest["validation"]["warnings"]
    )
