"""Error-path ladder pins: legs + feature_scan validation rungs (W15-2).

Coverage-driven: every case targets an UNCOVERED raise line from the
V-0 coverage report.  One param/section per rung, match on a
distinctive fragment.  Mutation fact: deleting the pinned raise
fails the case.
"""

from __future__ import annotations

import pandas as pd
import pytest
from atropos.feature_scan import (
    bucket_scan,
    correlation_cluster,
    missingness_scan,
    rule_fit,
)
from atropos.legs import (
    And,
    Compare,
    EntrySpec,
    ExecutionSpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    Not,
    Or,
    SizingSpec,
    StopSpec,
    TargetSpec,
    validate_features_declared,
)


def test_compare_ladder() -> None:
    """Compare rungs: empty field, bad op, NaN value."""
    with pytest.raises(ValueError, match="field must be a non-empty str"):
        Compare("", ">", 1.0)
    dynamic_op = "==="  # deliberately invalid, bypasses the Literal
    with pytest.raises(ValueError, match="op must be one of"):
        Compare("mom", dynamic_op, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a finite number"):
        Compare("mom", ">", float("nan"))


def test_and_or_operand_guards() -> None:
    """And/Or need >= 2 operands; Not rejects empty children."""
    single = Compare("mom", ">", 1.0)
    with pytest.raises(ValueError, match="at least two operands"):
        And(single)
    with pytest.raises(ValueError, match="at least two operands"):
        Or(single)
    with pytest.raises(ValueError, match="Predicate"):
        Not(42)  # type: ignore[arg-type]


def test_stop_target_spec_ladder() -> None:
    """Stop/Target rungs: kind, multiple, empty field."""
    for spec_cls in (StopSpec, TargetSpec):
        bad_kind = "trailing"  # deliberately invalid
        with pytest.raises(ValueError, match="kind"):
            spec_cls(kind=bad_kind, multiple=1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="multiple"):
            spec_cls(kind="atr", multiple=0.0)
        with pytest.raises(ValueError, match="field"):
            spec_cls(kind="atr", multiple=1.0, field="")
        with pytest.raises(ValueError, match="multiple"):
            spec_cls(kind="atr", multiple=0.0)
        with pytest.raises(ValueError, match="field"):
            spec_cls(kind="atr", multiple=1.0, field="")


def test_entry_exit_spec_guards() -> None:
    """Entry cooldown sign, max_concurrent floor, exit needs a mechanism."""
    with pytest.raises(ValueError, match="cooldown"):
        EntrySpec(Compare("mom", ">", 1.0), cooldown=pd.Timedelta(days=-1))
    with pytest.raises(ValueError, match="max_concurrent"):
        EntrySpec(Compare("mom", ">", 1.0), max_concurrent=0)
    with pytest.raises(ValueError, match="at least one"):
        ExitSpec()


def test_instrument_ref_guards() -> None:
    """InstrumentRef: empty kind, deferred kinds, bad spec shape."""
    with pytest.raises(ValueError, match="kind"):
        InstrumentRef(kind="", spec={"field": "close"})
    for deferred_kind in ("composite", "option_leg", "spread"):
        with pytest.raises(ValueError, match="section 14"):
            InstrumentRef(deferred_kind, {})
    with pytest.raises(ValueError, match="field"):
        InstrumentRef("field", {})
    with pytest.raises(ValueError, match="field"):
        InstrumentRef("field", {"field": 42})


def test_execution_sizing_guards() -> None:
    """Execution: unknown order type/tif. Sizing: mode/param rungs."""
    with pytest.raises(ValueError, match="order_type"):
        ExecutionSpec(order_type="sniper")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time_in_force"):
        ExecutionSpec(time_in_force="GTCX")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        SizingSpec(mode="martingale", param=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="param"):
        SizingSpec(mode="fixed", param=-1.0)


def test_legspec_guards() -> None:
    """LegSpec: empty name, features_used mismatch, empty rationale."""
    good_entry = EntrySpec(Compare("mom", ">", 1.0))
    good_exit = ExitSpec(stop=StopSpec("atr", 2.0))
    with pytest.raises(ValueError, match="name"):
        LegSpec(
            name="",
            direction="long",
            entry=good_entry,
            exit=good_exit,
            instrument=InstrumentRef("field", {"field": "close"}),
            features_used=("mom", "close"),
            rationale="x",
        )
    with pytest.raises(ValueError, match="direction"):
        bad_direction = "sideways"  # deliberately invalid
        LegSpec(
            name="x",
            direction=bad_direction,  # type: ignore[arg-type]
            entry=good_entry,
            exit=good_exit,
            instrument=InstrumentRef("field", {"field": "close"}),
            features_used=("mom", "close"),
            rationale="x",
        )
    with pytest.raises(ValueError, match="features_used"):
        LegSpec(
            name="x",
            direction="long",
            entry=good_entry,
            exit=good_exit,
            instrument=InstrumentRef("field", {"field": "close"}),
            features_used=(),
            rationale="x",
        )
    with pytest.raises(ValueError, match="rationale"):
        LegSpec(
            name="x",
            direction="long",
            entry=good_entry,
            exit=good_exit,
            instrument=InstrumentRef("field", {"field": "close"}),
            features_used=("mom", "close"),
            rationale="",
        )
    with pytest.raises(ValueError, match="must declare every referenced"):
        validate_features_declared(
            LegSpec(
                name="x",
                direction="long",
                entry=EntrySpec(Compare("vx", ">", 1.0)),
                exit=good_exit,
                instrument=InstrumentRef("field", {"field": "close"}),
                features_used=("mom", "close"),
                rationale="x",
            )
        )


def test_missingness_scan_guards() -> None:
    """missingness_scan: bad threshold, name collisions, non-column."""
    rng = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match="must be a float in \\(0, 1\\)"):
        missingness_scan(rng, {"f": rng}, {}, nan_threshold=1.5)
    with pytest.raises(ValueError, match="collide"):
        missingness_scan(rng, {"f": rng}, {"f": rng})
    with pytest.raises(ValueError, match="single column"):
        missingness_scan(
            rng,
            {"f": pd.DataFrame({"a": [1.0], "b": [2.0]})},
            {},
        )


def test_correlation_cluster_guards() -> None:
    """correlation_cluster: empty features, no aligned rows."""
    series = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match="must not be empty"):
        correlation_cluster({}, series)
    misaligned = pd.Series([1.0], index=pd.RangeIndex(100, 101))
    with pytest.raises(ValueError, match="no rows survive alignment"):
        correlation_cluster(
            {"f": series, "o": misaligned},
            misaligned,
        )


def test_rule_fit_guards() -> None:
    """rule_fit: empty rules, empty/overlapping splits."""
    out = pd.Series([1.0] * 4)
    mask = [True, False, True, False]
    with pytest.raises(ValueError, match="must not be empty"):
        rule_fit(out, {}, train_idx=[0, 1], test_idx=[2, 3])
    with pytest.raises(ValueError, match="non-empty"):
        rule_fit(out, {"r": mask}, train_idx=[], test_idx=[2, 3])
    with pytest.raises(ValueError, match="overlap"):
        rule_fit(
            out,
            {"r": mask},
            train_idx=[0, 1, 2],
            test_idx=[1, 2, 3],
        )
    with pytest.raises(ValueError, match="out of range"):
        rule_fit(
            out,
            {"r": mask},
            train_idx=[0, 1, 999],
            test_idx=[2, 3],
        )


def test_bucket_scan_guards() -> None:
    """bucket_scan: n_buckets/perm/seed rungs."""
    out = pd.Series([1.0] * 8)
    feat = pd.Series([1.0] * 8)
    with pytest.raises(ValueError, match="n_buckets"):
        bucket_scan(out, {"f": feat}, n_buckets=1)
    with pytest.raises(ValueError, match="n_perm"):
        bucket_scan(out, {"f": feat}, n_perm=0)
    with pytest.raises(ValueError, match="seed"):
        bucket_scan(out, {"f": feat}, seed=-1)
