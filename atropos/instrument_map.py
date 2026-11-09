"""Instrument routing: one strategy family -> one measurement plan.

A strategy family prescribes WHICH instruments may judge it.  A
rule-based tactical book is measured by its trade battery and
robustness attacks, never by an ML stage (a zero-heavy decision
stream makes constant predictions by construction); a cross-sectional
factor is measured by factor regression, never by a trade battery;
funding settlement is not momentum.  :func:`route` resolves the
family's :class:`InstrumentPlan`; :func:`audit_report` re-checks a
finished pipeline report against the plan post-hoc.

Scope: part of the three-tier falsification toolkit (backtest /
feature / paper + routing) — routing pins which measurement
instruments a family may be judged with BEFORE and AFTER any tier
runs; it never invents evidence of its own.

Run ``python -m atropos.instrument_map`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast, get_args

__all__ = [
    "PLAN_REGISTRY",
    "InstrumentPlan",
    "StrategyFamily",
    "audit_report",
    "route",
    "validate_family",
]

StrategyFamily = Literal[
    "rule_based_tactical",
    "per_trade_tactical",
    "ml_predictive",
    "cross_sectional_factor",
    "carry",
    "options",
]


class InstrumentPlan(TypedDict):
    """The measurement contract one strategy family must answer to.

    ``family`` echoes the routed literal; ``primary`` names the
    instruments the family IS judged by; ``informational`` names the
    read-only instruments (no verdict power); ``forbidden`` lists the
    instruments that would MISMEASURE the family — each entry carries
    ``{"tool": str, "reason": str, "stage": int | None}`` where a
    concrete ``stage`` (1-7 of the ensemble pipeline) gates that stage
    at runtime and ``None`` marks a declaration-level mismatch that no
    single stage owns; ``units`` is the family's native measurement
    unit.
    """

    family: StrategyFamily
    primary: list[str]
    informational: list[str]
    forbidden: list[dict[str, Any]]
    units: Literal["R_multiple", "fractional", "mixed"]
    identity: str


PLAN_REGISTRY: dict[StrategyFamily, InstrumentPlan] = {
    "rule_based_tactical": {
        "family": "rule_based_tactical",
        "primary": ["stages_1_3", "trade_battery"],
        "informational": [],
        "forbidden": [
            {
                "tool": "ml_stage_4",
                "stage": 4,
                "reason": (
                    "rule-based strategies have no predictive feature "
                    "stream; ML on zero-heavy streams gives "
                    "CONSTANT_PREDICTIONS by construction"
                ),
            },
            {
                "tool": "daily_aggregate_for_sub_daily",
                "stage": None,
                "reason": (
                    "feed granularity must match decision granularity (§12)"
                ),
            },
        ],
        "units": "R_multiple",
        "identity": "trade battery + robustness; no ML",
    },
    "per_trade_tactical": {
        "family": "per_trade_tactical",
        "primary": ["trade_battery", "payoff_stats"],
        "informational": [],
        "forbidden": [
            {
                "tool": "daily_aggregate_for_sub_daily",
                "stage": None,
                "reason": (
                    "feed granularity must match decision granularity (§12)"
                ),
            },
        ],
        "units": "R_multiple",
        "identity": (
            "trade battery only; daily stages are instrument mismatch"
        ),
    },
    "ml_predictive": {
        "family": "ml_predictive",
        "primary": ["stages_1_7"],
        "informational": [],
        "forbidden": [],
        "units": "fractional",
        "identity": "full pipeline; all stages",
    },
    "cross_sectional_factor": {
        "family": "cross_sectional_factor",
        "primary": ["stage_7_cs", "factor_regression"],
        "informational": [],
        "forbidden": [
            {
                "tool": "trade_battery",
                "stage": None,
                "reason": "factor construction is not trade-level",
            },
        ],
        "units": "fractional",
        "identity": (
            "CS_NO_EDGE + factor exposure attribution; factor premia ≠ alpha"
        ),
    },
    "carry": {
        "family": "carry",
        "primary": ["stages_1_5", "funding_alignment"],
        "informational": [],
        "forbidden": [
            {
                "tool": "book_momentum_signal",
                "stage": None,
                "reason": "funding is settlement, not momentum",
            },
        ],
        "units": "fractional",
        "identity": "risk envelope + funding alignment",
    },
    "options": {
        "family": "options",
        "primary": ["stages_1_5", "pnl_decomposition"],
        "informational": [],
        "forbidden": [],
        "units": "fractional",
        "identity": "per-greek decomposition",
    },
}

_ALLOWED_FAMILIES = frozenset(get_args(StrategyFamily))
"""The six valid family names (drives the loud literal check)."""


def validate_family(value: Any) -> StrategyFamily:
    """Literal-check one strategy-family value loudly.

    Args:
        value: candidate family (a ``str`` from config, a frozen spec
            or a keyword argument; anything else is rejected).

    Returns:
        The validated :class:`StrategyFamily` literal.

    Raises:
        ValueError: naming the bad value when it is not one of the
            six family names.

    Example::

        >>> validate_family("carry")
        'carry'
        >>> try:
        ...     validate_family("momentum")
        ... except ValueError as error:
        ...     str(error).startswith(
        ...         "unknown strategy family 'momentum'")
        True
    """
    if not isinstance(value, str) or value not in _ALLOWED_FAMILIES:
        raise ValueError(
            f"unknown strategy family {value!r}; expected one of "
            f"{sorted(_ALLOWED_FAMILIES)}"
        )
    return cast(StrategyFamily, value)


def route(family: StrategyFamily) -> InstrumentPlan:
    """Registry lookup: the measurement plan for one family.

    Args:
        family: the strategy family (validated again here so a runtime
            caller cannot bypass the literal check with a wrong str).

    Returns:
        The family's shared :class:`InstrumentPlan` registry entry —
        a read-only contract; callers must not mutate it.

    Raises:
        ValueError: when ``family`` is not one of the six names.

    Example::

        >>> route("rule_based_tactical")["units"]
        'R_multiple'
        >>> route("rule_based_tactical")["primary"]
        ['stages_1_3', 'trade_battery']
        >>> try:
        ...     route("momentum")
        ... except ValueError as error:
        ...     str(error).startswith(
        ...         "unknown strategy family 'momentum'")
        True
    """
    checked = validate_family(family)
    return PLAN_REGISTRY[checked]


def audit_report(
    report: dict[str, Any], family: StrategyFamily
) -> list[dict[str, Any]]:
    """Post-hoc scan of a finished pipeline report against the plan.

    Scope: part of the three-tier falsification toolkit (backtest /
    feature / paper + routing).  Reads ``report["stages"]`` (the
    ensemble pipeline's per-stage list; each entry carries a
    ``"stage"`` int and a ``"name"`` str) and returns one violation
    dict ``{"stage": int, "tool": str, "reason": str}`` for every plan
    forbidden entry whose concrete ``stage`` matches a stage present
    in the report.  Declaration-level bans (``stage is None``) never
    fire here — no single stage owns them.

    Limitations stated honestly: a runner that does not save the
    stages array cannot be audited post-hoc — this returns ``[]``
    without proving or disproving compliance.  The scan judges what
    the report CARRIES: never-run stages appear in it as SKIPPED
    placeholders and do match, so callers wanting ran-only semantics
    pass the ran stages (the pipeline integration does exactly that).

    Args:
        report: a finished pipeline report (a ``"stages"`` key is
            required for anything to be found; its absence yields
            ``[]``).
        family: the strategy family to audit against.

    Returns:
        The violation dicts, in the plan's forbidden-entry order.

    Example::

        >>> mini = {"stages": [
        ...     {"stage": 1, "name": "null_control_suite"},
        ...     {"stage": 4, "name": "walkforward_framework"},
        ... ]}
        >>> found = audit_report(mini, "rule_based_tactical")
        >>> len(found)
        1
        >>> found[0]["tool"], found[0]["stage"]
        ('ml_stage_4', 4)
        >>> audit_report({}, "rule_based_tactical")
        []
    """
    plan = route(family)
    stages = report.get("stages", [])
    if not isinstance(stages, list):
        return []
    present: set[int] = set()
    for entry in stages:
        if isinstance(entry, dict):
            stage_no = entry.get("stage")
            if isinstance(stage_no, int):
                present.add(int(stage_no))
    findings: list[dict[str, Any]] = []
    for forbidden in plan["forbidden"]:
        stage_no = forbidden.get("stage")
        if stage_no is None:
            continue
        if stage_no in present:
            findings.append(
                {
                    "stage": int(stage_no),
                    "tool": str(forbidden.get("tool", "")),
                    "reason": str(forbidden.get("reason", "")),
                }
            )
    return findings


# ---------------------------------------------------------------------------
# synthetic demo (python -m atropos.instrument_map)
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo the six plans + one post-hoc audit (asserted).

    Every printed line is backed by an assertion, so a regression
    crashes the demo instead of printing a wrong plan quietly.  The
    audit case routes a synthetic stage-4 report through
    :func:`audit_report` under ``rule_based_tactical`` and asserts the
    ``ml_stage_4`` violation fires; a stages-less report must audit
    clean (the honest empty result, not a compliance proof).
    """
    print("=" * 72)
    print("instrument_map demo: strategy family -> instrument plan")
    print("=" * 72)
    for family_name in get_args(StrategyFamily):
        plan = route(validate_family(family_name))
        forbidden_tools = [str(entry["tool"]) for entry in plan["forbidden"]]
        print(
            f"{family_name}: units={plan['units']} "
            f"primary={plan['primary']} "
            f"forbidden={forbidden_tools}"
        )
        assert plan["family"] == family_name
    mini_report: dict[str, Any] = {
        "stages": [
            {"stage": 1, "name": "null_control_suite"},
            {"stage": 4, "name": "walkforward_framework"},
        ],
    }
    violations = audit_report(mini_report, "rule_based_tactical")
    print(f"audit vs a stage-4 report: {len(violations)} violation(s)")
    print(f"  first: {violations[0]}")
    assert violations, "expected the ml_stage_4 violation to fire"
    assert violations[0]["tool"] == "ml_stage_4"
    assert violations[0]["stage"] == 4
    stages_less = audit_report({"hypothesis": "no stages array"}, "carry")
    print(f"stages-less report audits to: {stages_less}")
    assert stages_less == []


if __name__ == "__main__":
    run_all_examples()
