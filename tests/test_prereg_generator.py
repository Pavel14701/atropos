"""Tests for atropos.prereg_generator validation and rendering.

Covers every problem class that ``validate_prereg`` must catch (missing
field, non-numeric dict threshold, missing or bad comparison operator,
string criterion without an operator bound to a number, wrong
content-field types), acceptance of a valid spec, ``ValueError``
propagation from ``generate_prereg`` (including non-ISO dates), markdown
rendering with an explicit ``date=``, and roundtrip stability.
"""

from __future__ import annotations

from typing import Any

import pytest
from atropos.prereg_generator import (
    REQUIRED_FIELDS,
    _criteria_rows,
    generate_prereg,
    run_all_examples,
    spec_hash,
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


def test_validate_prereg_requires_comparison_in_dict_criteria() -> None:
    # no default "~": a dict criterion without a direction is a soft
    # kill switch and must be rejected
    params = make_valid_params()
    params["kill_criteria"] = [
        {"criterion": "t after controls", "threshold": 2.0}
    ]
    assert validate_prereg(params) == [
        "kill_criteria[0]: missing required 'comparison' "
        "(one of ['<', '<=', '==', '>', '>=', '~'])"
    ]


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


def test_tilde_comparison_is_valid_but_approximate() -> None:
    """'~' stays allowed (not a breaking removal) but is approximate.

    Documented contract: ``~`` is allowed only as an explicit
    diagnostic/approximate marker - a kill criterion written with
    ``~`` must be understood as approximate; prefer directional
    operators for hard kills.  The rendered table must show ``~`` in
    the Rule column so readers see the criterion is not directional.
    """
    params = make_valid_params()
    params["kill_criteria"] = [
        {
            "criterion": "mean daily IC hovers around zero",
            "comparison": "~",
            "threshold": 0.0,
        }
    ]
    assert validate_prereg(params) == []
    doc = generate_prereg(params, date="2026-01-15")
    assert "| 1 | mean daily IC hovers around zero | ~ | 0.0 |" in doc


def test_validate_prereg_rejects_string_criterion_without_operator() -> None:
    # a bare number in prose ("wait 200 days") is a soft, movable kill
    # switch; only an operator bound to a number freezes a threshold
    bad_with_expected = (
        (
            "beat the benchmark",
            "no comparison operator bound to a numeric threshold",
        ),
        ("   ", "empty string"),
        (
            "wait 200 days before final check",
            "no comparison operator bound to a numeric threshold",
        ),
        (
            "benchmark window of 63 days",
            "no comparison operator bound to a numeric threshold",
        ),
    )
    for bad_criterion, expected in bad_with_expected:
        params = make_valid_params()
        params["kill_criteria"] = [bad_criterion]
        problems = validate_prereg(params)
        assert len(problems) == 1, problems
        assert problems[0].startswith("kill_criteria[0]:")
        assert expected in problems[0]


def test_validate_prereg_reports_non_dict_params() -> None:
    assert validate_prereg(["not", "a", "dict"]) == [
        "params must be a dict, got list"
    ]


def test_validate_prereg_rejects_wrong_content_field_types() -> None:
    # construction/nulls take str | list[str]; only power_calc accepts
    # a dict; None/int are reported, never silently stringified
    for field in ("construction", "nulls"):
        params = make_valid_params()
        params[field] = None
        assert validate_prereg(params) == [
            f"field '{field}' must be a non-empty string or list of "
            f"strings, got NoneType"
        ]
        params[field] = {"steps": "zscore"}
        assert validate_prereg(params) == [
            f"field '{field}' must be a non-empty string or list of "
            f"strings, got dict"
        ]
        params[field] = ["signal = zscore(x)", 42]
        assert validate_prereg(params) == [
            f"field '{field}' list items must be strings, got int"
        ]
        params[field] = ["   "]
        assert validate_prereg(params) == [
            f"field '{field}' must be a non-empty string or list of "
            f"strings; the list has no non-empty item"
        ]
    params = make_valid_params()
    params["power_calc"] = 42
    assert validate_prereg(params) == [
        "field 'power_calc' must be a non-empty string, list of "
        "strings, or non-empty dict, got int"
    ]
    params = make_valid_params()
    params["power_calc"] = {}
    assert validate_prereg(params) == [
        "field 'power_calc' must not be an empty dict"
    ]


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
    # the spec hash is computed from the frozen fields automatically
    assert f"- **Spec hash:** `{spec_hash(make_valid_params())}`" in doc
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
    # string criteria show the operator actually used in the text
    assert "| 3 | placebo p >= 0.05 | >= | 0.05 |" in doc
    assert "- **N:** 2600" in doc


def test_spec_hash_is_deterministic_and_content_bound() -> None:
    """Same spec -> same hash; any frozen-field edit changes it."""
    params = make_valid_params()
    assert spec_hash(params) == spec_hash(dict(params))
    # insertion order is irrelevant (canonical JSON sorts keys)
    reordered = {
        "power_calc": params["power_calc"],
        "kill_criteria": params["kill_criteria"],
        "nulls": params["nulls"],
        "construction": params["construction"],
        "hypothesis": params["hypothesis"],
        "name": params["name"],
    }
    assert spec_hash(reordered) == spec_hash(params)
    # editing any frozen field moves the hash (tamper evidence)
    tampered = make_valid_params()
    tampered["kill_criteria"] = [dict(params["kill_criteria"][0])]
    tampered["kill_criteria"][0]["threshold"] = 1.5
    assert spec_hash(tampered) != spec_hash(params)


def test_user_supplied_hash_is_tolerated_but_ignored() -> None:
    """'hash' is no longer an input: stale entries never reach the doc.

    Contract change (ATR-07): the spec hash is computed at render time,
    so a caller-supplied (possibly stale) hash cannot mask an edited
    spec; validation neither requires nor rejects the field.
    """
    stale = make_valid_params()
    stale["hash"] = "deadbeef"
    assert validate_prereg(stale) == []
    doc = generate_prereg(stale, date="2026-01-15")
    assert "- **Spec hash:** `deadbeef`" not in doc
    assert f"- **Spec hash:** `{spec_hash(stale)}`" in doc
    # the stale entry does not participate in the hash itself
    assert spec_hash(stale) == spec_hash(make_valid_params())
    # and dropping it entirely is the same spec
    assert generate_prereg(stale, date="2026-01-15") == generate_prereg(
        make_valid_params(), date="2026-01-15"
    )


def test_hash_is_no_longer_a_required_field() -> None:
    """'hash' left REQUIRED_FIELDS: the generator computes it itself."""
    assert "hash" not in REQUIRED_FIELDS
    params = make_valid_params()
    assert validate_prereg(params) == []
    assert f"- **Spec hash:** `{spec_hash(params)}`" in generate_prereg(
        params, date="2026-01-15"
    )


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
    # the Rule column carries the operator actually used in the text
    assert "| < | 0.05 |" in doc
    assert "| >= | 0.05 |" in doc
    assert "| > | 1.5 |" in doc
    assert "| ~ |" not in doc
    assert "| < | 200 |" not in doc


def test_criteria_rows_raises_on_operator_free_string() -> None:
    # defensive guard: post-validation this input cannot reach
    # _criteria_rows through generate_prereg, but a broken contract
    # must fail loudly, not render a bogus "~" row
    with pytest.raises(ValueError, match="comparison operator"):
        _criteria_rows(["wait 200 days"])


def test_generate_prereg_rejects_non_iso_date() -> None:
    for bad_date in ("garbage", "2026-13-45", "15.01.2026"):
        with pytest.raises(ValueError, match="ISO date"):
            generate_prereg(make_valid_params(), date=bad_date)


def test_generate_prereg_collapses_newlines_in_power_values() -> None:
    params = make_valid_params()
    params["power_calc"] = {"N": 2600, "note": "alpha\n= 0.05\nseeded"}
    doc = generate_prereg(params, date="2026-01-15")
    assert "- **note:** alpha = 0.05 seeded" in doc
    assert "\n= 0.05" not in doc


def test_run_all_examples_demo_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Layer-0: valid prereg renders, soft kill switches are rejected."""
    run_all_examples()
    output = capsys.readouterr().out
    assert "# Preregistration: demo_overnight_imbalance_edge" in output
    assert "| >= | 0.05 |" in output  # real operator in the Rule column
    assert "ValueError raised as expected" in output
    assert "must be a finite number, got 'small'" in output
