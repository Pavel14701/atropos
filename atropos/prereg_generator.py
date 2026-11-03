"""Preregistration generator for the hypothesis-killing pipeline.

Turns a parameter dict into a frozen markdown prereg document.  Validates that
all required fields are present and that every kill criterion carries a
numeric threshold, so a hypothesis cannot be preregistered with a soft,
movable kill switch.

Required input keys::

    {
      "name":          str,
      "hypothesis":    str,
      "construction":  str | list[str],    # how the signal is built
      "nulls":         str | list[str],    # null hypotheses to be rejected
      "kill_criteria": list[dict | str],   # dicts need a numeric
                                           # "threshold"; plain strings
                                           # must contain a number
      "power_calc":    str | list | dict,  # power analysis summary
      "hash":          str,                # content hash of the frozen spec
    }

Usage::

    from atropos.prereg_generator import generate_prereg, validate_prereg

    problems = validate_prereg(params)   # [] when valid
    doc = generate_prereg(params)        # raises ValueError when invalid

Run ``python -m atropos.prereg_generator`` for a synthetic demo.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any

__all__ = ["REQUIRED_FIELDS", "generate_prereg", "validate_prereg"]

REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "hypothesis",
    "construction",
    "nulls",
    "kill_criteria",
    "power_calc",
    "hash",
)

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# The frozen threshold is the number a comparison operator points at,
# not merely the first number in the sentence: "IC < 0.05 on 200 days"
# must freeze 0.05, not 200.  "~" is excluded - it is the fuzzy default,
# not a comparison.
_OP_NUMBER_RE = re.compile(
    r"(?:<=|>=|==|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
_ALLOWED_COMPARISONS = {"<", "<=", ">", ">=", "==", "~"}


def _is_number(value: Any) -> bool:
    """True for a finite int/float (bools excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_list(value: Any) -> list[str]:
    """Coerce a scalar or list/tuple into a list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _nonempty_str(value: Any) -> bool:
    """True for a str that contains at least one non-whitespace char."""
    return isinstance(value, str) and bool(value.strip())


def _scalar_field_problems(params: dict[str, Any]) -> list[str]:
    """Problems with the plain-string fields name/hypothesis/hash."""
    return [
        f"field '{field}' must be a non-empty string"
        for field in ("name", "hypothesis", "hash")
        if not _nonempty_str(params[field])
    ]


def _content_field_problems(params: dict[str, Any]) -> list[str]:
    """Problems with the content fields construction/nulls/power_calc."""
    problems: list[str] = []
    for field in ("construction", "nulls", "power_calc"):
        value = params[field]
        if isinstance(value, dict):
            if not value:
                problems.append(f"field '{field}' must not be an empty dict")
        elif not any(item.strip() for item in _as_list(value)):
            problems.append(
                f"field '{field}' must be a non-empty string or list"
            )
    return problems


def _dict_criterion_problems(label: str, crit: dict[str, Any]) -> list[str]:
    """Problems with one dict-form kill criterion."""
    problems: list[str] = []
    if not _nonempty_str(crit.get("criterion", "")):
        problems.append(f"{label}: missing non-empty 'criterion' text")
    if "threshold" not in crit:
        problems.append(f"{label}: missing numeric 'threshold'")
    elif not _is_number(crit["threshold"]):
        problems.append(
            f"{label}: 'threshold' must be a finite number, "
            f"got {crit['threshold']!r}"
        )
    comparison = crit.get("comparison", "~")
    if comparison not in _ALLOWED_COMPARISONS:
        problems.append(
            f"{label}: comparison {comparison!r} not in "
            f"{sorted(_ALLOWED_COMPARISONS)}"
        )
    return problems


def _string_criterion_problems(label: str, crit: str) -> list[str]:
    """Problems with one string-form kill criterion."""
    if not crit.strip():
        return [f"{label}: empty string"]
    if _NUMBER_RE.search(crit) is None:
        return [f"{label}: no numeric threshold found in {crit!r}"]
    return []


def _kill_criteria_problems(kill_criteria: Any) -> list[str]:
    """Problems with the kill_criteria field as a whole."""
    problems: list[str] = []
    if isinstance(kill_criteria, str):
        kill_criteria = [kill_criteria]
    if (
        not isinstance(kill_criteria, (list, tuple))
        or len(kill_criteria) == 0
    ):
        problems.append(
            "kill_criteria must be a non-empty list (or a single string)"
        )
        return problems

    for idx, crit in enumerate(kill_criteria):
        label = f"kill_criteria[{idx}]"
        if isinstance(crit, dict):
            problems.extend(_dict_criterion_problems(label, crit))
        elif isinstance(crit, str):
            problems.extend(_string_criterion_problems(label, crit))
        else:
            problems.append(
                f"{label}: must be a dict or a string, "
                f"got {type(crit).__name__}"
            )
    return problems


def validate_prereg(params: dict[str, Any]) -> list[str]:
    """Return the list of validation problems; empty list means valid.

    Checks:
      * all REQUIRED_FIELDS present,
      * name / hypothesis / hash are non-empty strings,
      * construction / nulls / power_calc non-empty,
      * every kill criterion carries a numeric threshold.  Dict form:
        ``{"criterion": str, "threshold": number, "comparison": one of
        < <= > >= == ~}``; string form: must contain at least one number.

    Example::

        >>> bad = {"name": "x"}                    # doctest: +SKIP
        >>> validate_prereg(bad)                   # doctest: +SKIP
        ['missing required field: hypothesis', ...]
    """
    if not isinstance(params, dict):
        # Defensive runtime guard: callers may pass non-dict payloads.
        return [  # type: ignore[unreachable]
            f"params must be a dict, got {type(params).__name__}"
        ]

    problems = [
        f"missing required field: {field}"
        for field in REQUIRED_FIELDS
        if field not in params
    ]
    if problems:
        return problems

    problems.extend(_scalar_field_problems(params))
    problems.extend(_content_field_problems(params))
    problems.extend(_kill_criteria_problems(params["kill_criteria"]))
    return problems


def _criteria_rows(criteria: Any) -> list[tuple[str, str, str]]:
    """Extract ``(criterion, threshold, comparison)`` display rows."""
    items = [criteria] if isinstance(criteria, str) else list(criteria)
    rows: list[tuple[str, str, str]] = []
    for crit in items:
        if isinstance(crit, dict):
            rows.append(
                (
                    str(crit.get("criterion", "")).strip(),
                    str(crit["threshold"]),
                    str(crit.get("comparison", "~")),
                )
            )
        else:
            op_match = _OP_NUMBER_RE.search(crit)
            any_match = (
                op_match
                if op_match is not None
                else (_NUMBER_RE.search(crit))
            )
            # validate_prereg guarantees a number in string criteria.
            assert any_match is not None
            value = (
                op_match.group(1)
                if op_match is not None
                else any_match.group(0)
            )
            rows.append((crit.strip(), value, "~"))
    return rows


def _escape_table_cell(text: str) -> str:
    """Escape pipes and collapse newlines to keep one table row."""
    return text.replace("|", "\\|").replace("\n", " ")


def _render_construction_section(construction: Any) -> list[str]:
    """Render the ``## Construction`` markdown bullet list."""
    items = [
        f"- {item.strip()}" for item in _as_list(construction) if item.strip()
    ]
    return ["## Construction", "", *items, ""]


def _render_nulls_section(nulls: Any) -> list[str]:
    """Render the ``## Null hypotheses`` markdown section."""
    items = [
        f"**H{idx}:** {item.strip()}"
        for idx, item in enumerate(_as_list(nulls), start=1)
        if item.strip()
    ]
    return ["## Null hypotheses", "", *items, ""]


def _render_kill_criteria_section(criteria: Any) -> list[str]:
    """Render the ``## Kill criteria`` markdown table section."""
    rows = _criteria_rows(criteria)
    lines = [
        "## Kill criteria",
        "",
        "The hypothesis is KILLED if any of the following holds, "
        "measured on the",
        "frozen construction with frozen thresholds:",
        "",
        "| # | Criterion | Rule | Threshold |",
        "|---|-----------|------|-----------|",
    ]
    lines.extend(
        f"| {idx} | {_escape_table_cell(text)} | {comp} | {thr} |"
        for idx, (text, thr, comp) in enumerate(rows, start=1)
    )
    lines.append("")
    return lines


def _render_power_section(power_calc: Any) -> list[str]:
    """Render the ``## Power analysis`` markdown section."""
    lines = ["## Power analysis", ""]
    if isinstance(power_calc, dict):
        lines.extend(f"- **{key}:** {val}" for key, val in power_calc.items())
    else:
        lines.extend(
            f"- {item.strip()}"
            for item in _as_list(power_calc)
            if item.strip()
        )
    lines.append("")
    return lines


def _render_attestation_section() -> list[str]:
    """Render the closing ``## Attestation`` markdown section."""
    return [
        "## Attestation",
        "",
        "This specification (construction, nulls, kill criteria, power) "
        "was frozen",
        "before any outcome data was evaluated against it. Post-hoc "
        "changes invalidate",
        "the preregistration; any deviation must be logged as a separate "
        "amendment",
        "with a new hash.",
    ]


def generate_prereg(params: dict[str, Any], date: str | None = None) -> str:
    """Render the preregistration as a markdown string.

    Raises ``ValueError`` listing all validation problems if ``params``
    fails :func:`validate_prereg`.

    Example::

        >>> doc = generate_prereg({                     # doctest: +SKIP
        ...     "name": "edge_x", "hypothesis": "X predicts Y",
        ...     "construction": "zscore(X)",
        ...     "nulls": ["no incremental predictive power"],
        ...     "kill_criteria": [{"criterion": "|t| after controls",
        ...                        "comparison": "<", "threshold": 2.0}],
        ...     "power_calc": {"N": 2000}, "hash": "abc123",
        ... })
        >>> "# Preregistration: edge_x" in doc          # doctest: +SKIP
        True
    """
    problems = validate_prereg(params)
    if problems:
        raise ValueError(
            "invalid prereg params:\n  - " + "\n  - ".join(problems)
        )

    stamp = date or _dt.date.today().isoformat()

    lines: list[str] = [
        f"# Preregistration: {params['name'].strip()}",
        "",
        f"- **Frozen on:** {stamp}",
        f"- **Spec hash:** `{str(params['hash']).strip()}`",
        "",
        "## Hypothesis",
        "",
        params["hypothesis"].strip(),
        "",
    ]
    lines.extend(_render_construction_section(params["construction"]))
    lines.extend(_render_nulls_section(params["nulls"]))
    lines.extend(_render_kill_criteria_section(params["kill_criteria"]))
    lines.extend(_render_power_section(params["power_calc"]))
    lines.extend(_render_attestation_section())
    return "\n".join(lines)


def run_all_examples() -> None:
    """Demo: render a valid prereg and show a failing validation."""
    params = {
        "name": "demo_overnight_imbalance_edge",
        "hypothesis": (
            "Overnight order-flow imbalance predicts the next-day return "
            "incrementally to 7d and 30d momentum."
        ),
        "construction": [
            "signal = zscore(overnight_order_flow_imbalance)",
            "enter at next open, exit at next close",
            "long if signal > +0.5, short if signal < -0.5",
        ],
        "nulls": [
            "the edge is repackaged 7d/30d momentum",
            "the edge is concentrated in a single calendar year",
            "the edge is a clustering artifact of overlapping event days",
        ],
        "kill_criteria": [
            {
                "criterion": (
                    "|t| of new signal after momentum7 + momentum30 controls"
                ),
                "comparison": "<",
                "threshold": 2.0,
            },
            {
                "criterion": (
                    "share of total PnL contributed by the best subperiod"
                ),
                "comparison": ">",
                "threshold": 0.6,
            },
            (
                "event-day mean return not extreme vs 200 placebo draws "
                "(p >= 0.05)"
            ),
        ],
        "power_calc": {
            "partial_R2_target": 0.003,
            "N_required": 2600,
            "alpha": 0.05,
            "power": 0.80,
        },
        "hash": "9f2c1ab4",
    }
    print("=" * 72)
    print("prereg_generator demo: valid prereg")
    print("=" * 72)
    print(generate_prereg(params))

    bad = dict(params)
    bad["kill_criteria"] = [
        {
            "criterion": "t-stat after controls",
            "comparison": "<",
            "threshold": "small",
        }
    ]
    print()
    print("=" * 72)
    print(
        "prereg_generator demo: invalid params (non-numeric kill threshold)"
    )
    print("=" * 72)
    try:
        generate_prereg(bad)
    except ValueError as exc:
        print("ValueError raised as expected:")
        print(exc)


if __name__ == "__main__":
    run_all_examples()
