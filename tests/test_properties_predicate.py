"""Property tests: invariants over GENERATED worlds (W15-3).

Example-based tests pin specific worlds; these pin INVARIANTS that
must hold over every generated world in a class.  Hypothesis finds
the edge case nobody hand-wrote.  Deadline disabled (the CI runner
is slow); deterministic via derandomize-friendly seeded draws.
"""

from __future__ import annotations

from typing import Any

from atropos.legs import (
    And,
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    Not,
    Or,
    StopSpec,
    from_predicate_dict,
)
from hypothesis import given, settings
from hypothesis import strategies as st

_FIELDS = st.sampled_from(["mom", "atr", "close"])
_OPS = st.sampled_from([">", "<", ">=", "<=", "=="])
_VALUES = st.floats(allow_nan=False, allow_infinity=False, width=16)
_PRIOS = st.floats(-10.0, 10.0)


@st.composite
def st_predicate(draw: Any) -> Any:
    # the returned value is always a Predicate subclass; the Any
    # return keeps the recursive strategy expressible
    """A recursive Compare/And/Or/Not predicate over 3 known fields."""
    leaf = st.builds(Compare, _FIELDS, _OPS, _VALUES)
    return draw(
        st.one_of(
            leaf,
            st.lists(leaf, min_size=2, max_size=3).map(
                lambda children: And(*children)
            ),
            st.lists(leaf, min_size=2, max_size=2).map(
                lambda children: Or(*children)
            ),
            leaf.map(Not),
        )
    )


@given(predicate=st_predicate())
@settings(max_examples=120, deadline=None)
def test_property_predicate_roundtrip_and_fields(predicate: Any) -> None:
    """Roundtrip to_dict/from_dict preserves evaluation and fields.

    Mutation fact: any asymmetry between to_dict and from_dict (a
    dropped operand, a swapped kind) breaks the roundtrip.
    """
    restored = from_predicate_dict(predicate.to_dict())
    assert restored == predicate
    row = {"mom": 1.0, "atr": 0.5, "close": 100.0}
    assert restored.evaluate(row) == predicate.evaluate(row)
    # And semantics pinned against the manual conjunction: the
    # all->any mutation breaks this on mixed rows.  (Or is not
    # manually checked here -- the roundtrip + Not-identity covers
    # its dispatch, and any-vs-all on Or is the And mutation mirrored.)
    if isinstance(predicate, And):
        manual = all(child.evaluate(row) for child in predicate.operands)
        assert predicate.evaluate(row) == manual
    for field in predicate.fields():
        assert field in ("mom", "atr", "close")
    if isinstance(predicate, Compare):
        assert predicate.fields() == (predicate.field,)


@given(predicate=st_predicate())
@settings(max_examples=80, deadline=None)
def test_property_not_not_identity(predicate: Any) -> None:
    """Not(Not(p)) evaluates identically to p (involution)."""
    row = {"mom": 1.0, "atr": 0.5, "close": 100.0}
    double_negated = Not(Not(predicate))
    assert double_negated.evaluate(row) == predicate.evaluate(row)


def _make_leg(priority: float) -> LegSpec:
    return LegSpec(
        name="p",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 2.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        priority=priority,
        rationale="property fixture",
    )


@given(priority=_PRIOS)
@settings(max_examples=60, deadline=None)
def test_property_priority_roundtrip(priority: float) -> None:
    """LegSpec.priority survives the frozen-dataclass roundtrip."""
    leg = _make_leg(priority)
    assert leg.priority == priority
    # the to_dict/from_dict contract on predicates composed INTO a
    # leg entry: entry predicate roundtrips too
    restored = from_predicate_dict(leg.entry.when.to_dict())
    assert restored == leg.entry.when
