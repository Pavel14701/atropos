"""Tests for atropos.prereg_generator validation and rendering.

Covers every problem class that ``validate_prereg`` must catch (missing
field, non-numeric dict threshold, bad comparison operator, string
criterion without a number), acceptance of a valid spec, ``ValueError``
propagation from ``generate_prereg``, markdown rendering with an
explicit ``date=``, and roundtrip stability.
"""

from __future__ import annotations

from typing import Any

import pytest
from atropos.prereg_generator import (
    REQUIRED_FIELDS,
    generate_prereg,
    validate_prereg,
)


def make_valid_params() -> dict[str, Any]:
    return {
        "name": "edge_overnight",
        "hypothesis": "Overnight imbalance predicts next-day return.",
        "construction": ["signal = zscore(imbalance)", "enter at open"],
        "nulls": ["edge is repackaged 7d momentum"],
        "kill_criteria": [
            {
                "criterion": "|t| after controls",
                "comparison": "<",
                "threshold": 2.0,
            },
            {
                "criterion": "share of PnL in best year",
                "comparison": ">",
                "threshold": 0.6,
            },
            "placebo p >= 0.05",
        ],
        "power_calc": {"N": 2600, "power": 0.8},
        "hash": "9f2c1ab4",
    }


def test_validate_prereg_accepts_valid_params() -> None:
    assert validate_prereg(make_valid_params()) == []


def test_validate_prereg_reports_every_missing_field() -> None:
    problems = validate_prereg({})
    assert problems == [
        f"missing required field: {field}" for field in REQUIRED_FIELDS
    ]


def test_validate_prereg_single_missing_field_short_circuits() -> None:
    params = make_valid_params()
    del params["power_calc"]
    assert validate_prereg(params) == ["missing required field: power_calc"]


def test_validate_prereg_rejects_non_numeric_thresholds() -> None:
    bad_criteria: list[dict[str, Any]] = [
        {
            "criterion": "t after controls",
            "comparison": "<",
            "threshold": "small",
        },
        {
            "criterion": "t after controls",
            "comparison": "<",
            "threshold": True,
        },
        {"criterion": "t after controls", "comparison": "<"},
    ]
    for bad_crit in bad_criteria:
        params = make_valid_params()
        params["kill_criteria"] = [bad_crit]
        problems = validate_prereg(params)
        assert len(problems) == 1, problems
        assert "kill_criteria[0]" in problems[0]
        expected = (
            "missing numeric 'threshold'"
            if "threshold" not in bad_crit
            else "must be a finite number"
        )
        assert expected in problems[0]


def test_validate_prereg_rejects_bad_comparison_operator() -> None:
    params = make_valid_params()
    params["kill_criteria"] = [
        {
            "criterion": "t after controls",
            "comparison": "=<",
            "threshold": 2.0,
        }
    ]
    problems = validate_prereg(params)
    assert len(problems) == 1
    assert "kill_criteria[0]" in problems[0]
    assert "not in" in problems[0]


def test_validate_prereg_rejects_string_criterion_without_number() -> None:
    for bad_criterion in ("beat the benchmark", "   "):
        params = make_valid_params()
        params["kill_criteria"] = [bad_criterion]
        problems = validate_prereg(params)
        assert len(problems) == 1
        assert "kill_criteria[0]" in problems[0]


def test_generate_prereg_raises_listing_all_problems() -> None:
    params = make_valid_params()
    params["kill_criteria"] = [
        {
            "criterion": "t after controls",
            "comparison": "=<",
            "threshold": "small",
        }
    ]
    with pytest.raises(ValueError, match="invalid prereg params") as exc:
        generate_prereg(params)
    message = str(exc.value)
    assert "must be a finite number" in message
    assert "not in" in message


def test_generate_prereg_renders_sections_and_table_rows() -> None:
    doc = generate_prereg(make_valid_params(), date="2026-01-15")
    assert doc.splitlines()[0] == "# Preregistration: edge_overnight"
    assert "- **Frozen on:** 2026-01-15" in doc
    assert "- **Spec hash:** `9f2c1ab4`" in doc
    for section in (
        "## Hypothesis",
        "## Construction",
        "## Null hypotheses",
        "## Kill criteria",
        "## Power analysis",
        "## Attestation",
    ):
        assert section in doc
    assert "| # | Criterion | Rule | Threshold |" in doc
    # pipes inside criterion text are escaped for the markdown table
    assert "| 1 | \\|t\\| after controls | < | 2.0 |" in doc
    # string criteria keep their extracted numeric threshold
    assert "| 3 | placebo p >= 0.05 | ~ | 0.05 |" in doc
    assert "- **N:** 2600" in doc


def test_generate_prereg_roundtrip_is_deterministic() -> None:
    first = generate_prereg(make_valid_params(), date="2026-03-01")
    second = generate_prereg(make_valid_params(), date="2026-03-01")
    assert first == second
    other_date = generate_prereg(make_valid_params(), date="2026-04-01")
    assert other_date != first
    assert "- **Frozen on:** 2026-04-01" in other_date


def test_string_criterion_freezes_operator_adjacent_number() -> None:
    params = make_valid_params()
    params["kill_criteria"] = [
        "mean IC < 0.05 on 200 days of history",
        "placebo p >= 0.05 vs 200 placebo draws",
        "net sharpe > 1.5 after costs",
    ]
    doc = generate_prereg(params, date="2026-01-15")
    assert "| ~ | 0.05 |" in doc
    assert "| ~ | 1.5 |" in doc
    assert "| ~ | 200 |" not in doc


def test_string_criterion_without_operator_falls_back_to_first_number() -> (
    None
):
    params = make_valid_params()
    params["kill_criteria"] = ["benchmark window of 63 days"]
    doc = generate_prereg(params, date="2026-01-15")
    assert "| ~ | 63 |" in doc
