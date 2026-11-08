"""Declarative LEG layer: one position lifecycle in four pure layers.

A :class:`LegSpec` declares ONE position lifecycle as four
independent mutation/diagnostic points: **signal** (:class:`EntrySpec`
plus the optional exit signal -- predicate ASTs over declared feature
columns; a weak/overfit predicate dies here alone: OVERFIT_DEATH),
**instrument** (:class:`InstrumentRef`, which feature column IS the
traded instrument; composites/spreads deferred), **execution**
(:class:`ExecutionSpec`, order type + time in force only -- the Book
owns the CostModel, so a cost change (COST_DEATH) never mutates a
leg) and **sizing** (:class:`SizingSpec`, fixed R per trade or a
fraction of equity risked per trade; the Book converts that to R
against its own equity).

Predicates are ASTs, not lambdas: frozen, hashable, JSON-serializable
trees that roundtrip exactly through ``to_dict``/``from_dict``,
dedupe in sets, survive preregistration hashes and mutate node-wise.

The engine is STATELESS and PURE: :class:`LegEngine` maps
``(leg, md, ts) -> LegDecision`` with no memory.  Positions,
cooldowns and max-concurrency are the Book's job (separate module),
never stored here, so walk-forward restarts and parallel per-regime
runs are just more calls.  Stop/target/time exits are delegated to
the simulator; the engine evaluates only the entry gate and the
signal exit.

The decision row is built ONLY from ``leg.features_used``: NaN
anywhere makes every comparison False (a NaN is not a signal), a
missing decision timestamp raises KeyError, predicates on control
columns are NOT allowed in the MVP, and entries fill at the NEXT
bar's open (``entry_idx`` = decision position + 1).  Deferred
features raise ``ValueError`` naming KNOWN-ISSUES section 14.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from atropos._utils import validate_int, validate_number
from atropos.backtesting import TradeIntent
from atropos.data_contract import MaterializedData

__all__ = [
    "And",
    "Compare",
    "EntrySpec",
    "ExecutionSpec",
    "ExitSpec",
    "InstrumentRef",
    "LegDecision",
    "LegEngine",
    "LegSpec",
    "Not",
    "Or",
    "Predicate",
    "SizingSpec",
    "StopSpec",
    "TargetSpec",
    "evaluate_predicate",
    "from_predicate_dict",
    "validate_features_declared",
]

_DEFERRED_NOTE = "deferred (KNOWN-ISSUES section 14)"
"""House phrase every deferred-feature ValueError must contain."""

_ZERO = pd.Timedelta(0)
"""Shared immutable zero offset (dataclass default)."""

_ALLOWED_COMPARE_OPS = (">", "<", ">=", "<=", "==")
_ALLOWED_TIF = ("IOC", "GTC", "day")
_ALLOWED_SIZING_MODES = ("fixed", "risk_per_trade")
_DEFERRED_SIZING_MODES = ("fraction_of_equity", "vol_target")
_DEFERRED_INSTRUMENT_KINDS = ("composite", "option_leg", "spread")

_FAR_PRICE_SENTINEL = 1e9
"""Absolute price distance of the no-stop sentinel stop."""

_FAR_TARGET_MULTIPLE = 1e9
"""Target sentinel factor: entry +/- factor * risk_ref."""

_NO_TIME_EXIT_BARS = 10_000_000
"""hold_bars sentinel when ExitSpec.time is None (no time exit)."""


# ---------------------------------------------------------------------------
# predicate AST (frozen, hashable, JSON-serializable, evaluable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Predicate:
    """Base of the predicate AST (abstract-ish node).

    ``evaluate`` / ``to_dict`` / ``fields`` raise NotImplementedError
    on the bare base; use a concrete node.

    Raises:
        NotImplementedError: when the bare base node is evaluated,
            serialized or queried directly.
        ValueError: from :meth:`from_dict` on an unknown ``kind`` tag
            or a malformed ``and``/``or`` payload.

    Example::

        >>> payload = {
        ...     "kind": "or",
        ...     "operands": [
        ...         {"kind": "compare", "field": "mom", "op": ">",
        ...          "value": 0.0},
        ...         {"kind": "compare", "field": "vol", "op": "<",
        ...          "value": 2.0},
        ...     ],
        ... }
        >>> tree = Predicate.from_dict(payload)
        >>> tree.evaluate({"mom": -1.0, "vol": 1.0})
        True
        >>> Predicate.from_dict(tree.to_dict()) == tree
        True
        >>> Predicate.from_dict({"kind": "xor"})
        Traceback (most recent call last):
            ...
        ValueError: unknown predicate kind 'xor'
    """

    def evaluate(self, row: Mapping[str, float]) -> bool:
        """Verdict of this subtree against one float row."""
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate is not implemented on the"
            " base predicate node"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict form (node tag under ``"kind"``)."""
        raise NotImplementedError(
            f"{type(self).__name__}.to_dict is not implemented on the"
            " base predicate node"
        )

    def fields(self) -> tuple[str, ...]:
        """Distinct referenced field names, left-to-right."""
        raise NotImplementedError(
            f"{type(self).__name__}.fields is not implemented on the"
            " base predicate node"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Predicate:
        """Reconstruct the exact tree from its ``to_dict`` form.

        Dispatches on the ``"kind"`` tag; the roundtrip
        ``from_dict(node.to_dict()) == node`` is exact (pinned by
        tests).  Missing payload keys raise KeyError; an unknown
        ``kind`` tag or an ``and``/``or`` payload with fewer than two
        ``operands`` raises ValueError.

        Args:
            payload: the JSON-safe dict produced by ``to_dict``.

        Returns:
            The reconstructed predicate tree.
        """
        kind = str(payload.get("kind"))
        if kind == "compare":
            return Compare(
                field=str(payload["field"]),
                op=payload["op"],
                value=payload["value"],
            )
        if kind == "not":
            return Not(Predicate.from_dict(payload["operand"]))
        if kind in ("and", "or"):
            raw_children = payload.get("operands")
            if not isinstance(raw_children, list) or len(raw_children) < 2:
                raise ValueError(
                    f"predicate kind {kind!r} needs an 'operands' list"
                    " of at least two children"
                )
            children = tuple(
                Predicate.from_dict(child) for child in raw_children
            )
            if kind == "and":
                return And(*children)
            return Or(*children)
        raise ValueError(f"unknown predicate kind {kind!r}")


@dataclass(frozen=True)
class Compare(Predicate):
    """One field-vs-constant comparison (the leaf of every tree).

    ``field``/``op``/``value``: evaluation reads ``row[field]``
    (missing field -> KeyError from the row access); NaN/inf observed
    values return False -- a NaN comparison is not a signal.  ``op``
    is one of ``">"``, ``"<"``, ``">="``, ``"<="``, ``"=="``;
    ``value`` is a finite float or a numeric str (kept as-is for JSON
    fidelity; numerics normalize to float).  An unknown op, a
    non-numeric value or an empty field name raises ValueError.

    Example::

        >>> node = Compare("mom", ">", 0.0)
        >>> (node.evaluate({"mom": 0.5}),
        ...  node.evaluate({"mom": float("nan")}))
        (True, False)
        >>> node.to_dict()
        {'kind': 'compare', 'field': 'mom', 'op': '>', 'value': 0.0}
        >>> from_predicate_dict(node.to_dict()) == node
        True
        >>> node.fields()
        ('mom',)
    """

    field: str
    op: Literal[">", "<", ">=", "<=", "=="]
    value: float | str

    def __post_init__(self) -> None:
        """Validate the leaf loudly at construction."""
        if not self.field or not isinstance(self.field, str):
            raise ValueError("Compare.field must be a non-empty str")
        if self.op not in _ALLOWED_COMPARE_OPS:
            raise ValueError(
                f"Compare: op must be one of"
                f" {list(_ALLOWED_COMPARE_OPS)}, got {self.op!r}"
            )
        if isinstance(self.value, str):
            # numeric str: validated, kept as-is for JSON fidelity
            validate_number("Compare.value", self.value)
        elif isinstance(self.value, (int, float)):
            object.__setattr__(
                self, "value", validate_number("Compare.value", self.value)
            )
        else:
            raise ValueError(
                "Compare.value must be a finite number or a numeric"
                f" str, got {type(self.value).__name__}"
            )

    def evaluate(self, row: Mapping[str, float]) -> bool:
        """True when the row's value satisfies the comparison."""
        observed = float(row[self.field])
        if not math.isfinite(observed):
            return False
        threshold = float(self.value)
        if self.op == ">":
            return observed > threshold
        if self.op == "<":
            return observed < threshold
        if self.op == ">=":
            return observed >= threshold
        if self.op == "<=":
            return observed <= threshold
        return observed == threshold

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe leaf form."""
        return {
            "kind": "compare",
            "field": self.field,
            "op": self.op,
            "value": self.value,
        }

    def fields(self) -> tuple[str, ...]:
        """The single referenced field."""
        return (self.field,)


@dataclass(frozen=True, init=False)
class And(Predicate):
    """Conjunction of two or more children (all must hold).

    Children live in a TUPLE (``operands``) so the node stays
    hashable and preregistrations can pin exact shapes; fewer than
    two operands raise ValueError.

    Example::

        >>> node = And(Compare("mom", ">", 0.0), Compare("vol", "<", 2.0))
        >>> node.evaluate({"mom": 0.5, "vol": 1.0})
        True
        >>> node.fields()
        ('mom', 'vol')
        >>> And(Compare("mom", ">", 0.0))
        Traceback (most recent call last):
            ...
        ValueError: And needs at least two operands, got 1
    """

    operands: tuple[Predicate, ...]

    def __init__(self, *operands: Predicate) -> None:
        """Collect the children and validate the arity floor."""
        object.__setattr__(self, "operands", operands)
        _validate_variadic("And", operands)

    def evaluate(self, row: Mapping[str, float]) -> bool:
        """True when every child holds on the row."""
        return all(child.evaluate(row) for child in self.operands)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe conjunction form."""
        return {
            "kind": "and",
            "operands": [child.to_dict() for child in self.operands],
        }

    def fields(self) -> tuple[str, ...]:
        """Distinct child fields, left-to-right."""
        return _distinct_fields(self.operands)


@dataclass(frozen=True, init=False)
class Or(Predicate):
    """Disjunction of two or more children (any may hold).

    Children live in a TUPLE (``operands``) so the node stays
    hashable; fewer than two operands raise ValueError.

    Example::

        >>> node = Or(Compare("mom", ">", 0.0), Compare("vol", "<", 2.0))
        >>> node.evaluate({"mom": -1.0, "vol": 1.0})
        True
        >>> node.fields()
        ('mom', 'vol')
    """

    operands: tuple[Predicate, ...]

    def __init__(self, *operands: Predicate) -> None:
        """Collect the children and validate the arity floor."""
        object.__setattr__(self, "operands", operands)
        _validate_variadic("Or", operands)

    def evaluate(self, row: Mapping[str, float]) -> bool:
        """True when any child holds on the row."""
        return any(child.evaluate(row) for child in self.operands)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe disjunction form."""
        return {
            "kind": "or",
            "operands": [child.to_dict() for child in self.operands],
        }

    def fields(self) -> tuple[str, ...]:
        """Distinct child fields, left-to-right."""
        return _distinct_fields(self.operands)


@dataclass(frozen=True)
class Not(Predicate):
    """Negation of exactly one child predicate.

    A non-Predicate ``operand`` raises ValueError.

    Example::

        >>> node = Not(Compare("mom", ">", 0.0))
        >>> (node.evaluate({"mom": -1.0}), node.evaluate({"mom": 1.0}))
        (True, False)
        >>> node.fields()
        ('mom',)
    """

    operand: Predicate

    def __post_init__(self) -> None:
        """Reject non-predicate children."""
        if not isinstance(self.operand, Predicate):
            raise ValueError(
                "Not.operand must be a Predicate node, got"
                f" {type(self.operand).__name__}"
            )

    def evaluate(self, row: Mapping[str, float]) -> bool:
        """True when the child fails on the row."""
        return not self.operand.evaluate(row)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe negation form."""
        return {"kind": "not", "operand": self.operand.to_dict()}

    def fields(self) -> tuple[str, ...]:
        """The child's distinct fields."""
        return self.operand.fields()


def _validate_variadic(tag: str, operands: tuple[Predicate, ...]) -> None:
    """Enforce the >= 2 arity floor shared by And and Or."""
    if len(operands) < 2:
        raise ValueError(
            f"{tag} needs at least two operands, got {len(operands)}"
        )


def _distinct_fields(children: tuple[Predicate, ...]) -> tuple[str, ...]:
    """Distinct child field names in left-to-right order."""
    collected: list[str] = []
    for child in children:
        collected.extend(child.fields())
    return tuple(dict.fromkeys(collected))


def evaluate_predicate(
    predicate: Predicate, row: Mapping[str, float]
) -> bool:
    """Evaluate one predicate AST against one float row.

    Args:
        predicate: the root predicate node (Compare/And/Or/Not).
        row: the feature row; missing fields raise KeyError, NaN
            values make every comparison False.

    Returns:
        The predicate verdict.

    Example::

        >>> evaluate_predicate(Not(Compare("mom", ">", 0.0)), {"mom": -1.0})
        True
    """
    return predicate.evaluate(row)


def from_predicate_dict(payload: Mapping[str, Any]) -> Predicate:
    """Reconstruct a predicate tree from its JSON-safe dict form.

    Module-level convenience wrapping :meth:`Predicate.from_dict`.

    Args:
        payload: the JSON-safe dict produced by ``to_dict``.

    Returns:
        The reconstructed predicate tree.

    Example::

        >>> payload = {"kind": "compare", "field": "mom", "op": ">=",
        ...            "value": 1.0}
        >>> from_predicate_dict(payload).evaluate({"mom": 1.0})
        True
    """
    return Predicate.from_dict(payload)


# ---------------------------------------------------------------------------
# spec layers (signal / instrument / execution / sizing)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StopSpec:
    """Stop-loss declaration: ATR-multiple or absolute price.

    ``kind="atr"``: stop at ``entry -/+ multiple * row[field]`` (ATR
    read at the decision bar); ``kind="price"``: the stop IS
    ``multiple`` (absolute price).  Shorts mirror longs
    (:class:`LegEngine` mirrors).  ``multiple`` must be finite > 0;
    an unknown kind or empty ``field`` raises ValueError.

    Example::

        >>> stop = StopSpec("atr", 1.5)
        >>> (stop.kind, stop.multiple, stop.field)
        ('atr', 1.5, 'atr')
        >>> StopSpec("atr", 0.0)
        Traceback (most recent call last):
            ...
        ValueError: StopSpec.multiple must be finite > 0, got 0.0
    """

    kind: Literal["atr", "price"]
    multiple: float
    field: str = "atr"

    def __post_init__(self) -> None:
        """Validate kind, multiple and field name at construction."""
        if self.kind not in ("atr", "price"):
            raise ValueError(
                f"StopSpec: kind must be one of ['atr', 'price'],"
                f" got {self.kind!r}"
            )
        number = validate_number("StopSpec.multiple", self.multiple)
        if number <= 0.0:
            raise ValueError(
                f"StopSpec.multiple must be finite > 0, got {number}"
            )
        if not self.field or not isinstance(self.field, str):
            raise ValueError("StopSpec.field must be a non-empty str")


@dataclass(frozen=True)
class TargetSpec:
    """Profit-target declaration: ATR-multiple or absolute price.

    ``kind="atr"``: target at ``entry +/- multiple * row[field]``
    (mirrored for shorts); ``kind="price"``: the target IS
    ``multiple`` (absolute price).  ``multiple`` must be finite > 0;
    an unknown kind or empty ``field`` raises ValueError.

    Example::

        >>> target = TargetSpec("atr", 2.0, field="atr")
        >>> (target.kind, target.multiple, target.field)
        ('atr', 2.0, 'atr')
        >>> TargetSpec("atr", float("nan"))
        Traceback (most recent call last):
            ...
        ValueError: TargetSpec.multiple must be a finite number, got nan
    """

    kind: Literal["atr", "price"]
    multiple: float
    field: str = "atr"

    def __post_init__(self) -> None:
        """Validate kind, multiple and field name at construction."""
        if self.kind not in ("atr", "price"):
            raise ValueError(
                f"TargetSpec: kind must be one of ['atr', 'price'],"
                f" got {self.kind!r}"
            )
        number = validate_number("TargetSpec.multiple", self.multiple)
        if number <= 0.0:
            raise ValueError(
                f"TargetSpec.multiple must be finite > 0, got {number}"
            )
        if not self.field or not isinstance(self.field, str):
            raise ValueError("TargetSpec.field must be a non-empty str")


@dataclass(frozen=True)
class EntrySpec:
    """Entry declaration: predicate gate plus Book-enforced throttles.

    ``cooldown`` (non-negative Timedelta, default zero) and
    ``max_concurrent`` (>= 1) are DECLARATIONS: the stateless engine
    stores no position state -- the Book layer enforces both when
    admitting the intent.  A non-Predicate ``when``, a negative
    cooldown or a max_concurrent below 1 raise ValueError.

    Example::

        >>> entry = EntrySpec(Compare("mom", ">", 0.0))
        >>> (entry.cooldown, entry.max_concurrent)
        (Timedelta('0 days 00:00:00'), 1)
    """

    when: Predicate
    cooldown: pd.Timedelta = _ZERO
    max_concurrent: int = 1

    def __post_init__(self) -> None:
        """Validate the gate and the throttles at construction."""
        if not isinstance(self.when, Predicate):
            raise ValueError(
                f"EntrySpec.when must be a Predicate node, got"
                f" {type(self.when).__name__}"
            )
        if not isinstance(self.cooldown, pd.Timedelta) or (
            self.cooldown < _ZERO
        ):
            raise ValueError(
                "EntrySpec.cooldown must be a non-negative"
                " pd.Timedelta when given"
            )
        validate_int("EntrySpec.max_concurrent", self.max_concurrent)


@dataclass(frozen=True)
class ExitSpec:
    """Exit declaration: stop, target, time and/or signal mechanisms.

    AT LEAST ONE mechanism must be set (else ValueError).  ``stop``/
    ``target``/``time`` are delegated to the simulator; only
    ``signal`` is evaluated by the engine.  ``time`` must be a
    positive Timedelta and becomes the TradeIntent's ``hold_bars``
    as ``max(1, time / median outcome spacing)``.

    Example::

        >>> ExitSpec(stop=None, target=None, time=None, signal=None)
        Traceback (most recent call last):
            ...
        ValueError: ExitSpec needs at least one of stop/target/time/signal
    """

    stop: StopSpec | None = None
    target: TargetSpec | None = None
    time: pd.Timedelta | None = None
    signal: Predicate | None = None

    def __post_init__(self) -> None:
        """Validate that at least one exit mechanism is set."""
        if all(
            item is None
            for item in (self.stop, self.target, self.time, self.signal)
        ):
            raise ValueError(
                "ExitSpec needs at least one of stop/target/time/signal"
            )
        if self.stop is not None and not isinstance(self.stop, StopSpec):
            raise ValueError(
                f"ExitSpec.stop must be a StopSpec or None, got"
                f" {type(self.stop).__name__}"
            )
        if self.target is not None and not isinstance(
            self.target, TargetSpec
        ):
            raise ValueError(
                f"ExitSpec.target must be a TargetSpec or None, got"
                f" {type(self.target).__name__}"
            )
        if self.time is not None and (
            not isinstance(self.time, pd.Timedelta) or self.time <= _ZERO
        ):
            raise ValueError(
                "ExitSpec.time must be a positive pd.Timedelta when given"
            )
        if self.signal is not None and not isinstance(self.signal, Predicate):
            raise ValueError(
                f"ExitSpec.signal must be a Predicate or None, got"
                f" {type(self.signal).__name__}"
            )


@dataclass(frozen=True)
class InstrumentRef:
    """What the leg trades: MVP = one feature column IS the instrument.

    The column named by ``spec["field"]`` (exactly that one key, a
    non-empty str, stored as a private copy) supplies the entry
    reference price; stops, targets and the risk anchor are expressed
    against it.  ``kind`` must be ``"field"``: the ``composite`` /
    ``option_leg`` / ``spread`` families are DEFERRED and their
    ValueErrors name KNOWN-ISSUES section 14.

    Example::

        >>> ref = InstrumentRef("field", {"field": "mom"})
        >>> ref.spec
        {'field': 'mom'}
        >>> try:
        ...     InstrumentRef("composite", {"field": "mom"})
        ... except ValueError as error:
        ...     "deferred (KNOWN-ISSUES section 14)" in str(error)
        True
    """

    kind: str
    spec: dict[str, Any]

    def __post_init__(self) -> None:
        """Pin the MVP instrument shape loudly."""
        if self.kind != "field":
            if self.kind in _DEFERRED_INSTRUMENT_KINDS:
                raise ValueError(
                    f"InstrumentRef: kind {self.kind!r} is"
                    f" {_DEFERRED_NOTE} (MVP resolves a single field)"
                )
            raise ValueError(
                "InstrumentRef: kind must be one of ['field'] (or a"
                f" deferred kind), got {self.kind!r}"
            )
        if (
            set(self.spec) != {"field"}
            or not isinstance(self.spec.get("field"), str)
            or not self.spec["field"]
        ):
            raise ValueError(
                "InstrumentRef: spec must be exactly"
                " {'field': <non-empty str>}, got"
                f" {sorted(self.spec)!r}"
            )
        object.__setattr__(self, "spec", dict(self.spec))


@dataclass(frozen=True)
class ExecutionSpec:
    """Execution declaration: order type and time in force ONLY.

    Slippage, commissions and gaps are deliberately NOT here -- the
    Book layer owns the CostModel, so a cost change (COST_DEATH)
    never mutates a leg.  ``time_in_force`` is one of ``"IOC"``,
    ``"GTC"``, ``"day"``; a non-market ``order_type`` is DEFERRED and
    its ValueError names KNOWN-ISSUES section 14.

    Example::

        >>> execution = ExecutionSpec()
        >>> (execution.order_type, execution.time_in_force)
        ('market', 'IOC')
        >>> try:
        ...     ExecutionSpec(order_type="limit")
        ... except ValueError as error:
        ...     "deferred (KNOWN-ISSUES section 14)" in str(error)
        True
    """

    order_type: Literal["market"] = "market"
    time_in_force: Literal["IOC", "GTC", "day"] = "IOC"

    def __post_init__(self) -> None:
        """Pin the MVP execution shape loudly."""
        if self.order_type != "market":
            raise ValueError(
                f"ExecutionSpec: order_type {self.order_type!r} is"
                f" {_DEFERRED_NOTE} (MVP is taker-market only)"
            )
        if self.time_in_force not in _ALLOWED_TIF:
            raise ValueError(
                f"ExecutionSpec: time_in_force must be one of"
                f" {list(_ALLOWED_TIF)}, got {self.time_in_force!r}"
            )


@dataclass(frozen=True)
class SizingSpec:
    """Sizing declaration: fixed R per trade, or equity fraction at risk.

    ``mode="fixed"``: ``param`` is the trade size in R units.
    ``mode="risk_per_trade"``: ``param`` is the fraction of equity
    risked per trade -- the BOOK converts that to R against its own
    equity, never this module.  ``param`` must be finite > 0; the
    ``fraction_of_equity`` / ``vol_target`` modes are DEFERRED and
    their ValueErrors name KNOWN-ISSUES section 14.

    Example::

        >>> sizing = SizingSpec("risk_per_trade", 0.005)
        >>> (sizing.mode, sizing.param)
        ('risk_per_trade', 0.005)
        >>> try:
        ...     SizingSpec("vol_target", 0.1)
        ... except ValueError as error:
        ...     "deferred (KNOWN-ISSUES section 14)" in str(error)
        True
    """

    mode: Literal["fixed", "risk_per_trade"]
    param: float

    def __post_init__(self) -> None:
        """Validate mode and param at construction."""
        if self.mode in _DEFERRED_SIZING_MODES:
            raise ValueError(
                f"SizingSpec: mode {self.mode!r} is {_DEFERRED_NOTE}"
            )
        if self.mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                f"SizingSpec: mode must be one of"
                f" {list(_ALLOWED_SIZING_MODES)}, got {self.mode!r}"
            )
        number = validate_number("SizingSpec.param", self.param)
        if number <= 0.0:
            raise ValueError(
                f"SizingSpec.param must be finite > 0, got {number}"
            )


@dataclass(frozen=True)
class LegSpec:
    """One declarative leg: name, direction, and the four layers.

    Layers: ``entry`` (signal gate + Book throttles), ``exit`` (stop/
    target/time/signal; at least one set), ``instrument`` (MVP: one
    feature field), ``execution`` (costs live in the Book's
    CostModel) and ``sizing`` (fixed R or equity fraction at risk).
    ``unit`` accepts ``"risk"`` only (anything else is DEFERRED,
    naming KNOWN-ISSUES section 14).  ``features_used`` must be
    non-empty, distinct, and declare the instrument field plus every
    ATR-kind stop/target field; :func:`validate_features_declared`
    adds the BOTH-directions audit.  ``rationale`` is a CONTRACT
    field, not documentation: an empty rationale raises
    "LegSpec.rationale is a contract field, not documentation: state
    why the leg might have edge".  ``priority`` is purely metadata in
    the MVP: an ordering weight for the shared Book walk (higher runs
    first on the same bar; negative = deliberately demoted; the
    tiebreaker after equal priorities is the declaration order in
    ``StrategySpec.legs`` -- a tuple, so deterministic); the Book's
    combination modes read it (checkpoint W3-2).

    Example::

        >>> leg = LegSpec(
        ...     name="momo_long", direction="long",
        ...     entry=EntrySpec(Compare("mom", ">", 0.0)),
        ...     exit=ExitSpec(stop=None, target=None, time=None,
        ...                   signal=Compare("mom", "<", 0.0)),
        ...     instrument=InstrumentRef("field", {"field": "mom"}),
        ...     features_used=("mom",), rationale="momentum persistence",
        ... )
        >>> (leg.direction, leg.unit, leg.execution.order_type)
        ('long', 'risk', 'market')
    """

    name: str
    direction: Literal["long", "short"]
    entry: EntrySpec
    exit: ExitSpec
    instrument: InstrumentRef
    unit: Literal["risk"] = "risk"
    execution: ExecutionSpec = ExecutionSpec()
    sizing: SizingSpec = SizingSpec(mode="fixed", param=1.0)
    features_used: tuple[str, ...] = ()
    rationale: str = ""
    priority: float = 0.0

    def __post_init__(self) -> None:
        """Validate identity, layers and the feature contract."""
        _validate_leg_fields(self)
        _validate_leg_contract(self)


def _validate_leg_fields(spec: LegSpec) -> None:
    """Identity, direction and layer-type checks for a LegSpec."""
    if not spec.name or not isinstance(spec.name, str):
        raise ValueError(
            f"LegSpec.name must be a non-empty str, got {spec.name!r}"
        )
    if spec.direction not in ("long", "short"):
        raise ValueError(
            f"LegSpec {spec.name!r}: direction must be one of"
            f" ['long', 'short'], got {spec.direction!r}"
        )
    if not isinstance(spec.entry, EntrySpec):
        raise ValueError(
            f"LegSpec {spec.name!r}: entry must be an EntrySpec, got"
            f" {type(spec.entry).__name__}"
        )
    if not isinstance(spec.exit, ExitSpec):
        raise ValueError(
            f"LegSpec {spec.name!r}: exit must be an ExitSpec, got"
            f" {type(spec.exit).__name__}"
        )
    if not isinstance(spec.instrument, InstrumentRef):
        raise ValueError(
            f"LegSpec {spec.name!r}: instrument must be an"
            f" InstrumentRef, got {type(spec.instrument).__name__}"
        )
    if spec.unit != "risk":
        raise ValueError(
            f"LegSpec {spec.name!r}: unit {spec.unit!r} is"
            f" {_DEFERRED_NOTE} (MVP sizes in R units only)"
        )


def _validate_leg_contract(spec: LegSpec) -> None:
    """Rationale, priority and features_used shape for a LegSpec."""
    if not spec.rationale:
        raise ValueError(
            "LegSpec.rationale is a contract field, not documentation:"
            " state why the leg might have edge"
        )
    validate_number("LegSpec.priority", spec.priority)
    features = tuple(spec.features_used)
    if not features:
        raise ValueError(
            f"LegSpec {spec.name!r}: features_used must be a"
            " non-empty tuple of feature column names"
        )
    for name in features:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"LegSpec {spec.name!r}: features_used entries must be"
                f" non-empty strs, got {name!r}"
            )
    if len(set(features)) != len(features):
        raise ValueError(
            f"LegSpec {spec.name!r}: features_used must be distinct,"
            f" got {list(features)}"
        )
    object.__setattr__(spec, "features_used", features)
    _validate_leg_references(spec, features)


def _validate_leg_references(
    spec: LegSpec, features: tuple[str, ...]
) -> None:
    """Instrument and ATR fields must be declared in features_used."""
    referenced = [str(spec.instrument.spec["field"])]
    if spec.exit.stop is not None and spec.exit.stop.kind == "atr":
        referenced.append(spec.exit.stop.field)
    if spec.exit.target is not None and spec.exit.target.kind == "atr":
        referenced.append(spec.exit.target.field)
    missing = sorted({name for name in referenced if name not in features})
    if missing:
        raise ValueError(
            f"LegSpec {spec.name!r}: features_used must declare every"
            f" referenced field, missing: {missing}"
        )


def validate_features_declared(leg: LegSpec) -> None:
    """Enforce the BOTH-directions feature contract for one leg.

    Every field referenced by the entry predicate, the exit signal
    predicate, the instrument spec and the ATR-kind stop/target specs
    must appear in ``features_used``, AND every name in
    ``features_used`` must be referenced by one of those.  Predicates
    on control columns are NOT allowed in the MVP: the decision row
    is built exclusively from the declared feature columns.

    Args:
        leg: the leg to check.

    Returns:
        None when the contract holds in both directions.

    Raises:
        ValueError: listing each violated direction separately
            (``"declared but unused: [...]"`` and ``"used but
            undeclared: [...]"``).

    Example::

        >>> from dataclasses import replace
        >>> leg = LegSpec(
        ...     name="momo_long", direction="long",
        ...     entry=EntrySpec(Compare("mom", ">", 0.0)),
        ...     exit=ExitSpec(stop=None, target=None, time=None,
        ...                   signal=Compare("rsi", "<", 0.0)),
        ...     instrument=InstrumentRef("field", {"field": "mom"}),
        ...     features_used=("mom", "rsi"),
        ...     rationale="momentum persistence at the daily grid",
        ... )
        >>> validate_features_declared(leg)
        >>> bad = replace(leg, features_used=("mom", "extra"))
        >>> try:
        ...     validate_features_declared(bad)
        ... except ValueError as error:
        ...     message = str(error)
        ...     ("declared but unused: ['extra']" in message
        ...      and "used but undeclared: ['rsi']" in message)
        True
    """
    used = list(leg.entry.when.fields())
    if leg.exit.signal is not None:
        used.extend(leg.exit.signal.fields())
    used.append(str(leg.instrument.spec["field"]))
    if leg.exit.stop is not None and leg.exit.stop.kind == "atr":
        used.append(leg.exit.stop.field)
    if leg.exit.target is not None and leg.exit.target.kind == "atr":
        used.append(leg.exit.target.field)
    used_set = set(used)
    declared = set(leg.features_used)
    unused = sorted(declared - used_set)
    undeclared = sorted(used_set - declared)
    if not unused and not undeclared:
        return
    problems: list[str] = []
    if unused:
        problems.append(f"declared but unused: {unused}")
    if undeclared:
        problems.append(f"used but undeclared: {undeclared}")
    raise ValueError(
        f"LegSpec {leg.name!r}: feature contract violated ("
        + "; ".join(problems)
        + ")"
    )


# ---------------------------------------------------------------------------
# decision + engine (stateless, pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegDecision:
    """Outcome of one engine evaluation (pure, diagnostics included).

    ``entry_intent`` is None unless the entry predicate fired;
    ``exit_requested`` reflects ONLY the signal exit (stop, target
    and time exits are delegated to the simulator).
    ``diagnostics`` always carries ``predicate_row`` (the float row
    evaluated), ``entry_predicate``, ``exit_predicate`` (None when
    no signal is declared) and ``forced_last_bar``.

    Example::

        >>> decision = LegDecision(
        ...     entry_intent=None, exit_requested=False,
        ...     diagnostics={"entry_predicate": False},
        ... )
        >>> (decision.entry_intent is None, decision.exit_requested)
        (True, False)
    """

    entry_intent: TradeIntent | None
    exit_requested: bool
    diagnostics: dict[str, Any]


def _feature_row(
    md: MaterializedData, names: tuple[str, ...], ts: pd.Timestamp
) -> dict[str, float]:
    """The decision row: one float per declared feature at ``ts``.

    A missing ``ts`` raises KeyError from the ``.loc`` access -- a
    caller bug, raised loudly.
    """
    columns = md.features[[*names]]
    row_values = columns.loc[columns.index == ts].to_numpy(dtype=float)
    if row_values.size == 0:
        raise KeyError(ts)
    return {
        name: float(value)
        for name, value in zip(names, row_values[0], strict=True)
    }


def _decision_position(index_frame: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Positional index of ``ts`` in the index frame's ``ts`` column.

    Raises KeyError when the timestamp is not a decision timestamp.
    """
    ts_series = index_frame["ts"]
    position = int(ts_series.searchsorted(ts, side="left"))
    if position >= ts_series.size or ts_series.iloc[position] != ts:
        raise KeyError(ts)
    return position


def _finite_row_value(row: Mapping[str, float], name: str, tag: str) -> float:
    """One finite float from the decision row (NaN/inf is fatal)."""
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(
            f"{tag} {name!r} is not finite at the decision bar;"
            " a NaN/inf value cannot anchor a trade"
        )
    return value


def _resolve_stop(
    stop: StopSpec | None,
    row: Mapping[str, float],
    entry_ref: float,
    side: int,
    leg_name: str,
) -> tuple[float, float]:
    """Resolve ``(stop_price, atr_value)`` for the decision bar.

    A missing stop uses the far price sentinel (entry shifted 1e9
    against the direction) so only target/time/signal can fire; a
    degenerate stop (at or beyond entry) raises ValueError.
    """
    if stop is None:
        return entry_ref - _FAR_PRICE_SENTINEL * side, 0.0
    if stop.kind == "price":
        stop_price = stop.multiple
        atr_value = 0.0
    else:
        atr_value = _finite_row_value(row, stop.field, "stop ATR field")
        stop_price = entry_ref - side * stop.multiple * atr_value
    if side == 1 and stop_price >= entry_ref:
        raise ValueError(
            f"LegSpec {leg_name!r}: stop above entry for long leg"
        )
    if side == -1 and stop_price <= entry_ref:
        raise ValueError(
            f"LegSpec {leg_name!r}: stop below entry for short leg"
        )
    return stop_price, atr_value


def _resolve_target(
    target: TargetSpec | None,
    row: Mapping[str, float],
    entry_ref: float,
    side: int,
    risk_ref: float,
) -> float:
    """Resolve the target price for the decision bar.

    A missing target uses a DIRECTIONAL far sentinel -- a price the
    trade can never reach, so only stop/time can fire:

    * long: ``entry + 1e9 * risk_ref`` (arbitrarily high; the scan
      hits a long target when ``high >= target``);
    * short: ``entry / 1e9`` (arbitrarily low but POSITIVE --
      TradeIntent rejects non-positive prices; the scan hits a
      short target when ``low <= target``).

    The v13 single ``entry + 1e9 * risk`` sentinel was instantly
    "hit" for shorts (their hit condition is ``low <= target``, and
    every low sits below ``entry + 1e9``), so no-target shorts
    exited on their entry bar and could never hold -- the carry
    wave's borrow accrual needs holding shorts, hence the fix.
    """
    if target is None:
        if side == 1:
            return entry_ref + _FAR_TARGET_MULTIPLE * risk_ref
        return entry_ref / _FAR_TARGET_MULTIPLE
    if target.kind == "price":
        return target.multiple
    distance = target.multiple * _finite_row_value(
        row, target.field, "target ATR field"
    )
    return entry_ref + side * distance


def _hold_bars(
    time_exit: pd.Timedelta | None,
    md: MaterializedData,
    leg_name: str,
) -> int:
    """Bars to hold: ``max(1, time / median spacing)`` or the sentinel.

    Raises ValueError when the outcome grid is too small to estimate
    the bar spacing.
    """
    if time_exit is None:
        return _NO_TIME_EXIT_BARS
    outcome_index = pd.DatetimeIndex(md.outcome.index)
    if outcome_index.size < 2:
        raise ValueError(
            f"LegSpec {leg_name!r}: converting ExitSpec.time into"
            f" hold_bars needs 2+ outcome rows, got {outcome_index.size}"
        )
    bar_delta = pd.Timedelta(np.median(np.diff(outcome_index.to_numpy())))
    return max(1, int(time_exit / bar_delta))


class LegEngine:
    """Stateless, pure engine: ``(leg, md, ts) -> LegDecision``.

    No ``__init__`` state, no position memory: cooldown and
    max-concurrency are the BOOK's job (declared on
    :class:`EntrySpec`, enforced in the Book layer), so the same
    (leg, bundle, timestamp) triple always maps to the same decision.
    Requires the leg to have passed
    :func:`validate_features_declared`.

    Example::

        >>> import pandas as pd
        >>> from atropos.data_contract import MaterializedData
        >>> from atropos.legs import (
        ...     Compare, EntrySpec, ExitSpec, InstrumentRef, LegEngine,
        ...     LegSpec, StopSpec, TargetSpec,
        ... )
        >>> bars = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        >>> md = MaterializedData(
        ...     outcome=pd.Series(0.0, index=bars),
        ...     features=pd.DataFrame(
        ...         {"mom": [1.0, 3.0, -0.5, 4.0], "atr": 0.5}, index=bars
        ...     ),
        ...     controls=pd.DataFrame(index=bars),
        ...     index=pd.DataFrame(
        ...         {"ts": bars, "entity": "", "split": "train"}, index=bars
        ...     ),
        ... )
        >>> leg = LegSpec(
        ...     name="momo_long", direction="long",
        ...     entry=EntrySpec(Compare("mom", ">", 2.0)),
        ...     exit=ExitSpec(
        ...         stop=StopSpec("atr", 1.0), target=TargetSpec("atr", 2.0),
        ...         time=None, signal=Compare("mom", "<", 0.0),
        ...     ),
        ...     instrument=InstrumentRef("field", {"field": "mom"}),
        ...     features_used=("mom", "atr"),
        ...     rationale="momentum persistence at the daily grid",
        ... )
        >>> engine = LegEngine()
        >>> no_signal = engine.evaluate(leg, md, bars[0])
        >>> (no_signal.entry_intent is None, no_signal.exit_requested)
        (True, False)
        >>> no_signal.diagnostics["predicate_row"]
        {'mom': 1.0, 'atr': 0.5}
        >>> intent = engine.evaluate(leg, md, bars[1]).entry_intent
        >>> (intent.entry_idx, intent.side)
        (2, 1)
        >>> (intent.stop_price, intent.target_price, intent.hold_bars)
        (2.5, 4.0, 10000000)
        >>> (intent.atr_value, intent.risk_ref)
        (0.5, 0.5)
        >>> engine.evaluate(leg, md, bars[2]).exit_requested
        True
        >>> engine.evaluate_exit(leg, md, bars[2])
        True
        >>> engine.evaluate_exit(leg, md, bars[0])
        False
        >>> forced = engine.evaluate(leg, md, bars[3])
        >>> forced.diagnostics["forced_last_bar"]
        True
        >>> forced.entry_intent.entry_idx  # clamped to the last bar
        3
    """

    __slots__ = ()

    def evaluate(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts: pd.Timestamp,
    ) -> LegDecision:
        """Map one leg, one bundle and one decision ts to a decision.

        The row is read from ``md.features`` at ``ts`` (only
        ``leg.features_used`` columns).  On a firing entry the
        TradeIntent follows the house convention
        ``entry_idx = decision position + 1`` (the simulator fills at
        the NEXT bar's open), clamped to the last bar with a
        ``forced_last_bar`` diagnostic.  ``risk_ref`` is
        ``abs(entry - stop)``; a missing target uses the far sentinel
        ``entry +/- 1e9 * risk_ref``; ``hold_bars`` is
        ``max(1, time / median outcome spacing)``, or the 10_000_000
        sentinel when ``ExitSpec.time`` is None.

        Args:
            leg: the leg spec (assumed to have passed
                :func:`validate_features_declared`).
            md: the materialized bundle (features on the decision
                grid; index frame with the ``ts`` column).
            ts: the decision timestamp; must be present in
                ``md.index["ts"]``.

        Returns:
            A :class:`LegDecision`: the (possibly None) entry
            TradeIntent, the signal-exit flag and the diagnostics.

        Raises:
            KeyError: when ``ts`` is not a decision timestamp of
                ``md``.
            ValueError: on a degenerate stop (at or beyond entry),
                a non-finite instrument/ATR value at ``ts``, or an
                outcome grid too small to convert ``ExitSpec.time``
                into ``hold_bars``.
        """
        row = _feature_row(md, leg.features_used, ts)
        entry_flag = leg.entry.when.evaluate(row)
        exit_flag: bool | None = (
            None if leg.exit.signal is None else leg.exit.signal.evaluate(row)
        )
        diagnostics: dict[str, Any] = {
            "predicate_row": dict(row),
            "entry_predicate": entry_flag,
            "exit_predicate": exit_flag,
            "forced_last_bar": False,
        }
        intent: TradeIntent | None = None
        if entry_flag:
            intent = self._build_intent(leg, md, ts, row, diagnostics)
        return LegDecision(
            entry_intent=intent,
            exit_requested=bool(exit_flag),
            diagnostics=diagnostics,
        )

    def evaluate_exit(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts: pd.Timestamp,
    ) -> bool:
        """Convenience: only the signal-exit part of :meth:`evaluate`.

        Used by the Book's second pass; time, stop and target exits
        stay delegated to the simulator.

        Args:
            leg: the leg spec.
            md: the materialized bundle.
            ts: the decision timestamp; must be present in
                ``md.index["ts"]``.

        Returns:
            True only when a signal exit is declared and fires on
            the row; False when no signal exit is declared.

        Raises:
            KeyError: when ``ts`` is not a decision timestamp of
                ``md``.
        """
        if leg.exit.signal is None:
            return False
        row = _feature_row(md, leg.features_used, ts)
        return leg.exit.signal.evaluate(row)

    def _build_intent(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts: pd.Timestamp,
        row: Mapping[str, float],
        diagnostics: dict[str, Any],
    ) -> TradeIntent:
        """Build the TradeIntent (house convention: fill at next open)."""
        position = _decision_position(md.index, ts)
        n_bars = int(md.index.shape[0])
        entry_idx = position + 1
        if entry_idx >= n_bars:
            entry_idx = n_bars - 1
            diagnostics["forced_last_bar"] = True
        side = 1 if leg.direction == "long" else -1
        instrument_field = str(leg.instrument.spec["field"])
        entry_ref = _finite_row_value(
            row, instrument_field, "instrument field"
        )
        stop_price, atr_value = _resolve_stop(
            leg.exit.stop, row, entry_ref, side, leg.name
        )
        risk_ref = abs(entry_ref - stop_price)
        target_price = _resolve_target(
            leg.exit.target, row, entry_ref, side, risk_ref
        )
        hold_bars = _hold_bars(leg.exit.time, md, leg.name)
        return TradeIntent(
            entry_idx=entry_idx,
            side=side,
            stop_price=stop_price,
            target_price=target_price,
            hold_bars=hold_bars,
            atr_value=atr_value,
            risk_ref=risk_ref,
        )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: predicates, declarations and the engine, asserted.

    Every printed verdict is asserted, so a regression crashes the
    demo instead of printing a wrong number quietly.
    """
    bars = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    md = MaterializedData(
        outcome=pd.Series(0.0, index=bars),
        features=pd.DataFrame(
            {"mom": [1.0, 3.0, -0.5, 4.0, 2.5], "atr": 0.5}, index=bars
        ),
        controls=pd.DataFrame(index=bars),
        index=pd.DataFrame(
            {"ts": bars, "entity": "", "split": "train"}, index=bars
        ),
    )
    leg = LegSpec(
        name="momo_long",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 2.0)),
        exit=ExitSpec(
            stop=StopSpec("atr", 1.0),
            target=TargetSpec("atr", 2.0),
            signal=Compare("mom", "<", 0.0),
        ),
        instrument=InstrumentRef("field", {"field": "mom"}),
        features_used=("mom", "atr"),
        rationale="demo: momentum persistence at the daily grid",
    )
    print("=" * 72)
    print("legs demo: predicate gate + stateless engine decisions")
    print("=" * 72)
    engine = LegEngine()
    no_signal = engine.evaluate(leg, md, bars[0])
    intent = engine.evaluate(leg, md, bars[1]).entry_intent
    assert intent is not None
    exit_flip = engine.evaluate_exit(leg, md, bars[2])
    print(
        f"bar0 entry=None: {no_signal.entry_intent is None};"
        f" bar1 intent: idx={intent.entry_idx} stop={intent.stop_price}"
        f" target={intent.target_price}; bar2 signal exit={exit_flip}"
    )
    assert no_signal.entry_intent is None
    assert intent.entry_idx == 2 and intent.side == 1
    assert (intent.stop_price, intent.target_price) == (2.5, 4.0)
    assert intent.risk_ref == 0.5
    assert exit_flip is True
    # the same triple maps to the same decision (statelessness)
    assert engine.evaluate(leg, md, bars[1]) == engine.evaluate(
        leg, md, bars[1]
    )
    validate_features_declared(leg)
    try:
        InstrumentRef("option_leg", {"right": "put"})
        raised = False
    except ValueError as error:
        raised = "section 14" in str(error)
    print(f"deferred instrument kind rejected loudly: {raised}")
    assert raised
    print(f"leg.priority default: {leg.priority}")
    assert leg.priority == 0.0


if __name__ == "__main__":
    run_all_examples()
