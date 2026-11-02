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
      "kill_criteria": list[dict | str],   # dicts need a numeric "threshold";
                                           # plain strings must contain a number
      "power_calc":    str | list | dict,  # power analysis summary
      "hash":          str,                # content hash of the frozen spec
    }

Usage::

    from prereg_generator import generate_prereg, validate_prereg

    problems = validate_prereg(params)   # [] when valid
    doc = generate_prereg(params)        # raises ValueError when invalid

Run ``python prereg_generator.py`` for a synthetic demo.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any

__all__ = ["REQUIRED_FIELDS", "validate_prereg", "generate_prereg"]

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
_ALLOWED_COMPARISONS = {"<", "<=", ">", ">=", "==", "~"}


def _is_number(x: Any) -> bool:
    """True for a finite int/float (bools excluded)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _as_list(x: Any) -> list[str]:
    """Coerce a scalar or list/tuple into a list of strings."""
    if isinstance(x, (list, tuple)):
        return [str(item) for item in x]
    return [str(x)]


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def validate_prereg(params: dict[str, Any]) -> list[str]:
    """Return the list of validation problems; empty list means valid.

    Checks:
      * all REQUIRED_FIELDS present,
      * name / hypothesis / hash are non-empty strings,
      * construction / nulls / power_calc non-empty,
      * every kill criterion carries a numeric threshold (dict form:
        ``{"criterion": str, "threshold": number, "comparison": "<"|"<="|">"|">="|"=="|"~"}``;
        string form: must contain at least one number).

    Example::

        >>> bad = {"name": "x"}                    # doctest: +SKIP
        >>> validate_prereg(bad)                   # doctest: +SKIP
        ['missing required field: hypothesis', ...]
    """
    problems: list[str] = []
    if not isinstance(params, dict):
        return ["params must be a dict, got %s" % type(params).__name__]

    for field in REQUIRED_FIELDS:
        if field not in params:
            problems.append(f"missing required field: {field}")
    if problems:
        return problems

    for field in ("name", "hypothesis", "hash"):
        if not _nonempty_str(params[field]):
            problems.append(f"field '{field}' must be a non-empty string")

    for field in ("construction", "nulls", "power_calc"):
        v = params[field]
        if isinstance(v, dict):
            if not v:
                problems.append(f"field '{field}' must not be an empty dict")
        elif not any(s.strip() for s in _as_list(v)):
            problems.append(f"field '{field}' must be a non-empty string or list")

    kc = params["kill_criteria"]
    if isinstance(kc, str):
        kc = [kc]
    if not isinstance(kc, (list, tuple)) or len(kc) == 0:
        problems.append("kill_criteria must be a non-empty list (or a single string)")
        return problems

    for i, crit in enumerate(kc):
        label = f"kill_criteria[{i}]"
        if isinstance(crit, dict):
            if not _nonempty_str(crit.get("criterion", "")):
                problems.append(f"{label}: missing non-empty 'criterion' text")
            if "threshold" not in crit:
                problems.append(f"{label}: missing numeric 'threshold'")
            elif not _is_number(crit["threshold"]):
                problems.append(
                    f"{label}: 'threshold' must be a finite number, got {crit['threshold']!r}"
                )
            comp = crit.get("comparison", "~")
            if comp not in _ALLOWED_COMPARISONS:
                problems.append(
                    f"{label}: comparison {comp!r} not in {sorted(_ALLOWED_COMPARISONS)}"
                )
        elif isinstance(crit, str):
            if not crit.strip():
                problems.append(f"{label}: empty string")
            elif _NUMBER_RE.search(crit) is None:
                problems.append(f"{label}: no numeric threshold found in {crit!r}")
        else:
            problems.append(
                f"{label}: must be a dict or a string, got {type(crit).__name__}"
            )
    return problems


def generate_prereg(params: dict[str, Any], date: str | None = None) -> str:
    """Render the preregistration as a markdown string.

    Raises ``ValueError`` listing all validation problems if ``params`` fails
    :func:`validate_prereg`.

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
        raise ValueError("invalid prereg params:\n  - " + "\n  - ".join(problems))

    stamp = date or _dt.date.today().isoformat()

    kc = params["kill_criteria"]
    kc_items = [kc] if isinstance(kc, str) else list(kc)
    rows: list[tuple[str, str, str]] = []
    for crit in kc_items:
        if isinstance(crit, dict):
            rows.append(
                (
                    str(crit.get("criterion", "")).strip(),
                    str(crit["threshold"]),
                    str(crit.get("comparison", "~")),
                )
            )
        else:
            match = _NUMBER_RE.search(crit)
            rows.append((crit.strip(), match.group(0), "~"))

    lines: list[str] = []
    lines.append(f"# Preregistration: {params['name'].strip()}")
    lines.append("")
    lines.append(f"- **Frozen on:** {stamp}")
    lines.append(f"- **Spec hash:** `{str(params['hash']).strip()}`")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(params["hypothesis"].strip())
    lines.append("")
    lines.append("## Construction")
    lines.append("")
    for item in _as_list(params["construction"]):
        if item.strip():
            lines.append(f"- {item.strip()}")
    lines.append("")
    lines.append("## Null hypotheses")
    lines.append("")
    for i, item in enumerate(_as_list(params["nulls"]), start=1):
        if item.strip():
            lines.append(f"**H{i}:** {item.strip()}")
    lines.append("")
    lines.append("## Kill criteria")
    lines.append("")
    lines.append("The hypothesis is KILLED if any of the following holds, measured on the")
    lines.append("frozen construction with frozen thresholds:")
    lines.append("")
    lines.append("| # | Criterion | Rule | Threshold |")
    lines.append("|---|-----------|------|-----------|")
    for i, (text, thr, comp) in enumerate(rows, start=1):
        safe_text = text.replace("|", "\\|").replace("\n", " ")  # keep table intact
        lines.append(f"| {i} | {safe_text} | {comp} | {thr} |")
    lines.append("")
    lines.append("## Power analysis")
    lines.append("")
    pc = params["power_calc"]
    if isinstance(pc, dict):
        for k, v in pc.items():
            lines.append(f"- **{k}:** {v}")
    else:
        for item in _as_list(pc):
            if item.strip():
                lines.append(f"- {item.strip()}")
    lines.append("")
    lines.append("## Attestation")
    lines.append("")
    lines.append("This specification (construction, nulls, kill criteria, power) was frozen")
    lines.append("before any outcome data was evaluated against it. Post-hoc changes invalidate")
    lines.append("the preregistration; any deviation must be logged as a separate amendment")
    lines.append("with a new hash.")
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
                "criterion": "|t| of new signal after momentum7 + momentum30 controls",
                "comparison": "<",
                "threshold": 2.0,
            },
            {
                "criterion": "share of total PnL contributed by the best subperiod",
                "comparison": ">",
                "threshold": 0.6,
            },
            "event-day mean return not extreme vs 200 placebo draws (p >= 0.05)",
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
        {"criterion": "t-stat after controls", "comparison": "<", "threshold": "small"}
    ]
    print()
    print("=" * 72)
    print("prereg_generator demo: invalid params (non-numeric kill threshold)")
    print("=" * 72)
    try:
        generate_prereg(bad)
    except ValueError as exc:
        print("ValueError raised as expected:")
        print(exc)


if __name__ == "__main__":
    run_all_examples()
