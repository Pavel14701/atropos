"""Leg-layer pins: predicates, declarations, stateless engine.

Each test names the mutation it must catch (mutation plan; applied ->
fail -> restored before commit).
"""

from __future__ import annotations

import pandas as pd
import pytest
from atropos.data_contract import MaterializedData
from atropos.legs import (
    And,
    Compare,
    EntrySpec,
    ExecutionSpec,
    ExitSpec,
    InstrumentRef,
    LegEngine,
    LegSpec,
    Not,
    Or,
    SizingSpec,
    StopSpec,
    TargetSpec,
    from_predicate_dict,
    validate_features_declared,
)

_BARS = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")


def _md() -> MaterializedData:
    """A 5-day bundle with distinctive mom/atr columns."""
    return MaterializedData(
        outcome=pd.Series(0.0, index=_BARS),
        features=pd.DataFrame(
            {"mom": [1.0, 3.0, -0.5, 4.0, 2.5], "atr": 0.5}, index=_BARS
        ),
        controls=pd.DataFrame(index=_BARS),
        index=pd.DataFrame(
            {"ts": _BARS, "entity": "", "split": "train"}, index=_BARS
        ),
    )


def _leg(**overrides: object) -> LegSpec:
    """The doctest world's leg (mom-momentum long, ATR stops)."""
    params: dict[str, object] = {
        "name": "momo_long",
        "direction": "long",
        "entry": EntrySpec(Compare("mom", ">", 2.0)),
        "exit": ExitSpec(
            stop=StopSpec("atr", 1.0),
            target=TargetSpec("atr", 2.0),
            signal=Compare("mom", "<", 0.0),
        ),
        "instrument": InstrumentRef("field", {"field": "mom"}),
        "features_used": ("mom", "atr"),
        "rationale": "momentum persistence at the daily grid",
    }
    params.update(overrides)
    return LegSpec(**params)  # type: ignore[arg-type]


def test_predicate_truth_table_and_roundtrip() -> None:
    """AST semantics + JSON roundtrip, node by node.

    Mutation fact: swapping And's all() for any() (an And->Or
    mutation) flips the conjunction row; a from_dict that drops the
    Not node (or mislabels kinds) breaks the roundtrip asserts.
    """
    assert Compare("mom", ">", 2.0).evaluate({"mom": 3.0}) is True
    assert Compare("mom", ">", 2.0).evaluate({"mom": 1.0}) is False
    # NaN is never a signal
    assert Compare("mom", ">", 2.0).evaluate({"mom": float("nan")}) is False
    conjunct = And(Compare("mom", ">", 2.0), Compare("atr", "<", 1.0))
    assert conjunct.evaluate({"mom": 3.0, "atr": 0.5}) is True
    assert conjunct.evaluate({"mom": 3.0, "atr": 1.5}) is False
    disjunct = Or(Compare("mom", ">", 2.0), Compare("atr", "<", 1.0))
    assert disjunct.evaluate({"mom": 1.0, "atr": 0.5}) is True
    assert disjunct.evaluate({"mom": 1.0, "atr": 1.5}) is False
    negated = Not(Compare("mom", ">", 2.0))
    assert negated.evaluate({"mom": 1.0}) is True
    # roundtrip through JSON-safe dicts is exact
    for predicate in (conjunct, disjunct, negated, Compare("mom", "<=", 1.0)):
        assert from_predicate_dict(predicate.to_dict()) == predicate
    # fields() collects distinct references left-to-right
    assert conjunct.fields() == ("mom", "atr")


def test_features_declared_both_directions() -> None:
    """The declaration contract bites in BOTH directions.

    Mutation fact: checking only used->declared (dropping the
    unused-declared branch) lets a silent declared column through
    and the second pytest.raises fails.
    """
    good = _leg()
    validate_features_declared(good)  # consistent: no raise
    with pytest.raises(ValueError, match="undeclared"):
        validate_features_declared(
            _leg(
                entry=EntrySpec(Compare("vx", ">", 2.0)),
                features_used=("mom", "atr"),
            )
        )
    with pytest.raises(ValueError, match="unused"):
        validate_features_declared(_leg(features_used=("mom", "atr", "vx")))


def test_engine_stateless_and_math() -> None:
    """Pure engine: same call twice -> identical decision.

    Pins the intent arithmetic (stop/target/risk/hold) and the
    statelessness law.  Mutation fact: any hidden engine state (a
    module-level cache or an instance counter) changes the second
    evaluation and the equality assert fails; wrong short-side
    mirroring flips the stop side and the short assert fails.
    """
    md = _md()
    leg = _leg()
    engine = LegEngine()
    first = engine.evaluate(leg, md, _BARS[1])
    second = engine.evaluate(leg, md, _BARS[1])
    assert first.entry_intent == second.entry_intent
    assert first.diagnostics == second.diagnostics
    intent = first.entry_intent
    assert intent is not None
    # decision at _BARS[1] fills at the NEXT bar's open (idx 2)
    assert intent.entry_idx == 2
    assert intent.side == 1
    # entry_ref = mom[1] = 3.0; atr = 0.5 -> stop 2.5, target 4.0
    assert (intent.stop_price, intent.target_price) == (2.5, 4.0)
    assert intent.risk_ref == 0.5
    assert intent.hold_bars >= 1_000_000  # no time exit declared
    # short side mirrors the stop below/above convention
    short_leg = _leg(direction="short")
    short_intent = engine.evaluate(short_leg, md, _BARS[1]).entry_intent
    assert short_intent is not None and short_intent.side == -1
    assert short_intent.stop_price == pytest.approx(3.5)
    assert short_intent.target_price == pytest.approx(2.0)
    # a time exit converts the wall-clock into bars
    timed = _leg(
        exit=ExitSpec(stop=StopSpec("atr", 1.0), time=pd.Timedelta(days=3))
    )
    timed_intent = engine.evaluate(timed, md, _BARS[1]).entry_intent
    assert timed_intent is not None
    assert timed_intent.hold_bars == 3


def test_exit_signal_and_forced_clamp() -> None:
    """Signal exits come from the engine; the last bar clamps entry.

    Mutation fact: wiring exit_requested to the ENTRY predicate (or
    ignoring the exit signal) flips the bars[2] result; removing the
    last-bar clamp breaks the forced assert.
    """
    md = _md()
    leg = _leg()
    engine = LegEngine()
    assert engine.evaluate(leg, md, _BARS[2]).exit_requested is True
    assert engine.evaluate_exit(leg, md, _BARS[2]) is True
    assert engine.evaluate_exit(leg, md, _BARS[0]) is False
    forced = engine.evaluate(leg, md, _BARS[4])
    assert forced.diagnostics["forced_last_bar"] is True
    if forced.entry_intent is not None:
        assert forced.entry_intent.entry_idx == 4


def test_spec_validation_ladder() -> None:
    """Exit needs a mechanism; deferred kinds are loud rejections.

    Mutation fact: dropping the all-None exit guard (or any deferred
    check) makes the corresponding constructor accept garbage and
    the pytest.raises fail.
    """
    with pytest.raises(ValueError, match="at least one"):
        ExitSpec(stop=None, target=None, time=None, signal=None)
    with pytest.raises(ValueError, match="deferred"):
        InstrumentRef("option_leg", {"right": "put"})
    with pytest.raises(ValueError, match="deferred"):
        ExecutionSpec(order_type="limit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deferred"):
        SizingSpec(mode="vol_target", param=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rationale"):
        _leg(rationale="")
    with pytest.raises(ValueError, match="features_used"):
        _leg(features_used=())
    # undeclared instrument field is caught by the declaration check
    with pytest.raises(ValueError, match="must declare every"):
        _leg(instrument=InstrumentRef("field", {"field": "close"}))
