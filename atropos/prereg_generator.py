"""Preregistration generator for the falsification-toolkit pipeline.

Turns a parameter dict into a frozen markdown prereg document.  Validates
that all required fields are present and that every kill criterion is a
hard directional rule - a comparison operator bound to a numeric
threshold - so a hypothesis cannot be preregistered with a soft, movable
kill switch.

One deliberate exception to "hard directional": ``~`` remains an
allowed ``comparison`` value (removing it would be a breaking change),
but it is allowed only as an explicit diagnostic/approximate marker -
a kill criterion written with ``~`` must be understood as approximate;
prefer the directional operators (``<``, ``<=``, ``>``, ``>=``) for
hard kills.

Required input keys::

    {
      "name":          str,
      "hypothesis":    str,
      "construction":  str | list[str],    # how the signal is built
      "nulls":         str | list[str],    # null hypotheses to be rejected
      "kill_criteria": list[dict | str],   # dicts need "criterion", a
                                           # numeric "threshold" AND a
                                           # required "comparison" in
                                           # < <= > >= == ~; plain
                                           # strings must bind a
                                           # comparison operator to a
                                           # number ("IC < 0.05")
      "power_calc":    str | list | dict,  # power analysis summary
    }

The spec hash is NOT an input: :func:`generate_prereg` computes it
automatically (see :func:`spec_hash`), so the rendered document always
carries the hash of the spec it actually renders - the caller cannot
forget it, mistype it, or freeze a hash that does not match the frozen
fields.  A stale ``hash`` entry left in ``params`` is tolerated by
validation and ignored at render time.

Usage::

    from atropos.prereg_generator import generate_prereg, validate_prereg

    problems = validate_prereg(params)   # [] when valid
    doc = generate_prereg(params)        # raises ValueError when invalid

Run ``python -m atropos.prereg_generator`` for a synthetic demo.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from typing import Any

__all__ = [
    "REQUIRED_FIELDS",
    "generate_prereg",
    "spec_hash",
    "validate_prereg",
]

REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "hypothesis",
    "construction",
    "nulls",
    "kill_criteria",
    "power_calc",
)

# The frozen threshold is the number a comparison operator points at,
# not merely any number in the sentence: "IC < 0.05 on 200 days" must
# freeze 0.05, not 200, and "wait 200 days" freezes nothing at all.
# "~" is excluded - it is the fuzzy marker, not a comparison.
_OP_NUMBER_RE = re.compile(
    r"(<=|>=|==|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
# "~" is allowed only as an explicit diagnostic/approximate marker; a
# kill criterion written with "~" must be understood as approximate -
# prefer directional operators for hard kills.  Kept allowed so existing
# preregs do not break; see the module docstring.
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
    """Problems with the plain-string fields name/hypothesis."""
    return [
        f"field '{field}' must be a non-empty string"
        for field in ("name", "hypothesis")
        if not _nonempty_str(params[field])
    ]


def _content_field_problems(params: dict[str, Any]) -> list[str]:
    """Problems with the content fields construction/nulls/power_calc.

    Contract: ``construction`` and ``nulls`` must be a non-empty string
    or a non-empty list of strings; ``power_calc`` may additionally be
    a non-empty dict.  Anything else (None, int, dict in a string
    field, non-string list items) is reported instead of being
    silently stringified.
    """
    problems: list[str] = []
    for field in ("construction", "nulls", "power_calc"):
        value = params[field]
        expected = (
            "a non-empty string, list of strings, or non-empty dict"
            if field == "power_calc"
            else "a non-empty string or list of strings"
        )
        if isinstance(value, dict):
            if field != "power_calc":
                problems.append(
                    f"field '{field}' must be {expected}, got dict"
                )
            elif not value:
                problems.append(f"field '{field}' must not be an empty dict")
        elif isinstance(value, (list, tuple)):
            bad_items = [item for item in value if not isinstance(item, str)]
            if bad_items:
                problems.append(
                    f"field '{field}' list items must be strings, "
                    f"got {type(bad_items[0]).__name__}"
                )
            elif not any(item.strip() for item in value):
                problems.append(
                    f"field '{field}' must be {expected}; the list "
                    f"has no non-empty item"
                )
        elif not _nonempty_str(value):
            problems.append(
                f"field '{field}' must be {expected}, "
                f"got {type(value).__name__}"
            )
    return problems


def _dict_criterion_problems(label: str, crit: dict[str, Any]) -> list[str]:
    """Problems with one dict-form kill criterion.

    ``comparison`` is required, not defaulted: a dict criterion without
    an explicit direction would be a soft, movable kill switch.  The
    accepted set includes ``~``, which is allowed only as an explicit
    diagnostic/approximate marker - a kill criterion written with
    ``~`` must be understood as approximate; prefer directional
    operators for hard kills.
    """
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
    if "comparison" not in crit:
        problems.append(
            f"{label}: missing required 'comparison' "
            f"(one of {sorted(_ALLOWED_COMPARISONS)})"
        )
    elif crit["comparison"] not in _ALLOWED_COMPARISONS:
        problems.append(
            f"{label}: comparison {crit['comparison']!r} not in "
            f"{sorted(_ALLOWED_COMPARISONS)}"
        )
    return problems


def _string_criterion_problems(label: str, crit: str) -> list[str]:
    """Problems with one string-form kill criterion.

    A bare number is not enough: the string must bind a comparison
    operator to a number ("IC < 0.05").  Otherwise soft phrases like
    "wait 200 days" would slip through as kill criteria.
    """
    if not crit.strip():
        return [f"{label}: empty string"]
    if _OP_NUMBER_RE.search(crit) is None:
        return [
            f"{label}: no comparison operator bound to a numeric "
            f"threshold found in {crit!r} (e.g. 'IC < 0.05')"
        ]
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


def _canonical_json(params: dict[str, Any]) -> str:
    """Canonical JSON form hashed by :func:`spec_hash`.

    ``json.dumps`` with sorted keys and ASCII-only output: two dicts
    that differ only in insertion order hash identically, and the
    result is stable across platforms, locales and Python versions.
    The ``hash`` entry (if any) is excluded, so a hash can never hash
    itself.
    """
    frozen = {key: value for key, value in params.items() if key != "hash"}
    return json.dumps(frozen, sort_keys=True, ensure_ascii=True)


def spec_hash(params: dict[str, Any]) -> str:
    """Content hash of the frozen spec (first 8 sha256 hex chars).

    The hash covers the canonical JSON of ``params`` WITHOUT the
    ``hash`` key itself, so editing any frozen field (construction,
    nulls, kill criteria, power, ...) provably changes the hash: the
    rendered preregistration is tamper-evident by construction.  This
    is the value rendered as ``Spec hash`` in the document and the one
    to anchor a pipeline run to (e.g. ``ml_training``'s
    ``config["prereg_hash"]``).

    Example::

        >>> spec_hash({"name": "edge"}) == spec_hash({"name": "edge"})
        True
        >>> spec_hash({"name": "edge"}) == spec_hash({"name": "other"})
        False
        >>> a_stale_hash = spec_hash({"name": "edge"})
        >>> edited = {"name": "edge", "hash": a_stale_hash, "power_calc": "p"}
        >>> spec_hash(edited) == a_stale_hash
        False
    """
    return hashlib.sha256(
        _canonical_json(params).encode("utf-8")
    ).hexdigest()[:8]


def validate_prereg(params: Any) -> list[str]:
    """Return the list of validation problems; empty list means valid.

    Checks:
      * ``params`` is a dict (anything else is reported as a problem),
      * all REQUIRED_FIELDS present,
      * name / hypothesis are non-empty strings,
      * construction / nulls are a non-empty string or list of
        strings; power_calc may also be a non-empty dict,
      * every kill criterion is a hard directional rule.  Dict form:
        ``{"criterion": str, "threshold": number, "comparison": one of
        < <= > >= == ~}`` - ``comparison`` is REQUIRED, no default;
        string form: a comparison operator bound to a number
        (``"IC < 0.05"``), so bare numbers in prose ("wait 200 days")
        are rejected.

    ``hash`` is NOT a required field and is not validated: the spec
    hash is computed by :func:`spec_hash` at render time.  A stale
    ``hash`` entry left in ``params`` is tolerated (and ignored by
    :func:`generate_prereg`).

    ``~`` is allowed only as an explicit diagnostic/approximate marker;
    a kill criterion written with ``~`` must be understood as
    approximate - prefer directional operators for hard kills.

    Example::

        >>> bad = {"name": "x"}                    # doctest: +SKIP
        >>> validate_prereg(bad)                   # doctest: +SKIP
        ['missing required field: hypothesis', ...]
    """
    if not isinstance(params, dict):
        # Defensive runtime guard: callers may pass non-dict payloads.
        return [f"params must be a dict, got {type(params).__name__}"]

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
    """Extract ``(criterion, threshold, comparison)`` display rows.

    For string-form criteria the Rule column shows the operator that
    was actually matched in the string (e.g. ``>=`` in "p >= 0.05"),
    not a placeholder.
    """
    items = [criteria] if isinstance(criteria, str) else list(criteria)
    rows: list[tuple[str, str, str]] = []
    for crit in items:
        if isinstance(crit, dict):
            rows.append(
                (
                    str(crit.get("criterion", "")).strip(),
                    str(crit["threshold"]),
                    str(crit["comparison"]),
                )
            )
        else:
            op_match = _OP_NUMBER_RE.search(crit)
            if op_match is None:
                # validate_prereg guarantees operator+number in string
                # criteria; fail loudly instead of rendering a bogus
                # row if that contract is ever broken.
                raise ValueError(
                    f"string criterion lacks a comparison operator "
                    f"bound to a number: {crit!r}"
                )
            rows.append((crit.strip(), op_match.group(2), op_match.group(1)))
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


def _inline(text: Any) -> str:
    """Stringify ``text`` with whitespace collapsed onto one line."""
    return " ".join(str(text).split())


def _render_power_section(power_calc: Any) -> list[str]:
    """Render the ``## Power analysis`` markdown section.

    Values are stringified inline: a stray newline in a value must not
    break out of the bullet list.
    """
    lines = ["## Power analysis", ""]
    if isinstance(power_calc, dict):
        lines.extend(
            f"- **{_inline(key)}:** {_inline(val)}"
            for key, val in power_calc.items()
        )
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
        "was frozen before any outcome data was evaluated against it. "
        "Post-hoc changes invalidate the preregistration; any deviation "
        "must be logged as a separate amendment with a new hash.",
    ]


def generate_prereg(params: dict[str, Any], date: str | None = None) -> str:
    """Render the preregistration as a markdown string.

    The ``Spec hash`` line is computed automatically from the frozen
    fields (:func:`spec_hash`); callers never pass a hash.  A stale
    ``hash`` entry in ``params`` is ignored.

    Raises ``ValueError`` listing all validation problems if ``params``
    fails :func:`validate_prereg`, or if ``date`` is given and is not an
    ISO-formatted date (``YYYY-MM-DD``); ``date=None`` stamps today.

    Example::

        >>> doc = generate_prereg({                     # doctest: +SKIP
        ...     "name": "edge_x", "hypothesis": "X predicts Y",
        ...     "construction": "zscore(X)",
        ...     "nulls": ["no incremental predictive power"],
        ...     "kill_criteria": [{"criterion": "|t| after controls",
        ...                        "comparison": "<", "threshold": 2.0}],
        ...     "power_calc": {"N": 2000},
        ... })
        >>> "# Preregistration: edge_x" in doc          # doctest: +SKIP
        True
    """
    problems = validate_prereg(params)
    if problems:
        raise ValueError(
            "invalid prereg params:\n  - " + "\n  - ".join(problems)
        )

    if date is None:
        stamp = _dt.date.today().isoformat()
    else:
        # fromisoformat() on Python >= 3.11 also accepts extended ISO
        # forms ("20260115", week dates); the frozen stamp must be
        # unambiguously YYYY-MM-DD, so gate on the exact shape first.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError(
                f"date must be an ISO date (YYYY-MM-DD), got {date!r}"
            )
        try:
            stamp = _dt.date.fromisoformat(date).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"date must be an ISO date (YYYY-MM-DD), got {date!r}"
            ) from exc

    lines: list[str] = [
        f"# Preregistration: {params['name'].strip()}",
        "",
        f"- **Frozen on:** {stamp}",
        f"- **Spec hash:** `{spec_hash(params)}`",
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
    """Demo: render a valid prereg and show a failing validation.

    The demo doubles as the layer-0 verification test: the valid spec
    must validate and render (with the kill-criteria table showing the
    operators actually used, and the Spec hash computed from the frozen
    fields - not caller-supplied), and the soft kill switches
    (non-numeric threshold, "wait 200 days" prose, dict without a
    comparison) must all be rejected - a regression crashes the demo,
    not a hypothesis.
    """
    params: dict[str, Any] = {
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
    }
    print("=" * 72)
    print("prereg_generator demo: valid prereg")
    print("=" * 72)
    # layer-0 verification: the valid spec validates and renders with
    # the kill-criteria table binding the operators actually used, and
    # the Spec hash is derived from the frozen fields automatically
    assert validate_prereg(params) == []
    doc = generate_prereg(params)
    print(doc)
    assert "# Preregistration: demo_overnight_imbalance_edge" in doc
    assert f"- **Spec hash:** `{spec_hash(params)}`" in doc
    assert "## Attestation" in doc
    assert (
        "| 1 | \\|t\\| of new signal after momentum7 + momentum30"
        " controls | < | 2.0 |" in doc
    )
    assert (
        "| 2 | share of total PnL contributed by the best subperiod"
        " | > | 0.6 |" in doc
    )
    assert (
        "| 3 | event-day mean return not extreme vs 200 placebo draws"
        " (p >= 0.05) | >= | 0.05 |" in doc
    )

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
    raised = False
    try:
        generate_prereg(bad)
    except ValueError as exc:
        raised = True
        print("ValueError raised as expected:")
        print(exc)
    assert raised, "a non-numeric kill threshold must raise ValueError"
    assert any(
        "must be a finite number" in problem
        for problem in validate_prereg(bad)
    )

    # more soft kill switches that must be rejected: a bare number in
    # prose freezes nothing, and a dict criterion without a comparison
    # has no direction
    soft = dict(params)
    soft["kill_criteria"] = ["wait 200 days before final check"]
    soft_problems = validate_prereg(soft)
    # at least the soft criterion itself is reported (every other
    # field is still the valid spec above); a precise count is brittle
    # against future validation additions
    assert len(soft_problems) >= 1, soft_problems
    assert any(
        "no comparison operator" in problem for problem in soft_problems
    )

    directionless = dict(params)
    directionless["kill_criteria"] = [
        {"criterion": "t-stat after controls", "threshold": 2.0}
    ]
    directionless_problems = validate_prereg(directionless)
    assert directionless_problems, (
        "dict criterion without 'comparison' must be rejected"
    )
    assert "missing required 'comparison'" in directionless_problems[0]

    # tamper evidence: the Spec hash is derived from the frozen fields,
    # so moving any threshold after the freeze changes the hash
    tampered = dict(params)
    tampered["kill_criteria"] = [dict(params["kill_criteria"][0])]
    tampered["kill_criteria"][0]["threshold"] = 1.5
    assert spec_hash(tampered) != spec_hash(params), (
        "editing a frozen kill threshold must change the spec hash"
    )


if __name__ == "__main__":
    run_all_examples()
