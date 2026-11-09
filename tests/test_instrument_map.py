"""Instrument-routing pins: family -> plan, enforce vs audit mode.

The 2026-11 lesson set (A: ML-on-P&L CONSTANT_PREDICTIONS; C2: daily
threshold on a per-trade book; E: half-normal abs identity; BAB/BAC:
factor-vs-alpha) becomes machine-checkable: PLAN_REGISTRY forbids the
mismatched instruments per family.  Each test names its mutation
(applied -> fail -> restored before commit); the sweep plan is 4
mutations M1-M4 (see the sweep section of the wave notes).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from atropos.instrument_map import (
    PLAN_REGISTRY,
    InstrumentPlan,
    audit_report,
    route,
    validate_family,
)


def _pipeline_report(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal pipeline-report shape for audit_report post-hoc."""
    return {"stages": stages, "overall_verdict": "SURVIVES"}


# -- registry integrity ------------------------------------------------------


def test_registry_covers_all_families() -> None:
    """Every family has a plan; every forbidden entry is typed.

    Entries carry tool + reason + a well-typed stage.

    Mutation: a family dropped from the registry or a forbidden entry
    missing its reason -> caught.
    """
    expected = {
        "rule_based_tactical",
        "per_trade_tactical",
        "ml_predictive",
        "cross_sectional_factor",
        "carry",
        "options",
    }
    assert set(PLAN_REGISTRY) == expected
    for family, plan in PLAN_REGISTRY.items():
        assert plan["family"] == family
        assert plan["primary"], f"{family}: primary must not be empty"
        assert plan["units"] in ("R_multiple", "fractional", "mixed")
        assert plan["identity"]
        for entry in plan["forbidden"]:
            assert entry["tool"], family
            assert entry["reason"], family
            assert entry["stage"] is None or isinstance(entry["stage"], int)


def test_route_and_validate_family() -> None:
    """Route returns the registry plan; validate_family is the gate.

    Mutation: route returning a default plan for unknown families ->
    the ValueError assert fails -> caught.
    """
    plan: InstrumentPlan = route("rule_based_tactical")
    assert plan is PLAN_REGISTRY["rule_based_tactical"]
    bad = cast(Any, "no_such_family")
    with pytest.raises(ValueError, match="no_such_family"):
        route(bad)
    with pytest.raises(ValueError, match="no_such_family"):
        validate_family(bad)
    assert validate_family("carry") == "carry"


def test_rule_based_forbids_ml_stage_four() -> None:
    """The A-lesson, pinned: ml_stage_4 forbidden with stage=4.

    Mutation: the ml_stage_4 entry's stage set to None (informational
    only) -> enforce loses its match -> test_enforce_raises fails ->
    caught there; this pin keeps the entry itself honest.
    """
    plan = route("rule_based_tactical")
    ml_entries = [
        entry for entry in plan["forbidden"] if entry["tool"] == "ml_stage_4"
    ]
    assert len(ml_entries) == 1
    assert ml_entries[0]["stage"] == 4
    assert "CONSTANT_PREDICTIONS" in ml_entries[0]["reason"]


# -- audit_report post-hoc (U3: the dead-code pin) ---------------------------


def test_audit_report_on_fixture_with_ml_stage() -> None:
    """audit_report on a report WITH a stages array is not dead code.

    A synthetic report that ran stage 4 + family rule_based_tactical
    must produce the ml_stage_4 violation.  Mutation M4: audit_report
    returning [] always -> caught.
    """
    report = _pipeline_report(
        [
            {"stage": 1, "name": "null_controls", "status": "PASS"},
            {"stage": 4, "name": "walkforward", "status": "PASS"},
        ]
    )
    violations = audit_report(report, "rule_based_tactical")
    assert violations, "ml_stage_4 must be flagged post-hoc"
    assert violations[0]["stage"] == 4
    assert violations[0]["tool"] == "ml_stage_4"
    # a clean report for the same family stays clean
    ok = _pipeline_report([{"stage": 1, "name": "null_controls"}])
    assert audit_report(ok, "rule_based_tactical") == []


def test_audit_report_no_stages_array_returns_empty() -> None:
    """No stages array -> [] — the documented honest limit.

    Runners that do not save the stages array (the 2026-11 deployed
    verdicts) can be neither proved nor disproved: [] is the contract,
    not a dead-code artifact.
    """
    bare = {"overall_verdict": "SURVIVES"}
    assert audit_report(bare, "rule_based_tactical") == []


# -- units declaration -------------------------------------------------------


def test_units_contract_declared_per_family() -> None:
    """R-multiple families declare it; fractional families too.

    The units_mismatch detection itself is future work — this pins
    the DECLARATION the future check will read.
    """
    assert route("rule_based_tactical")["units"] == "R_multiple"
    assert route("per_trade_tactical")["units"] == "R_multiple"
    assert route("carry")["units"] == "fractional"
    assert route("ml_predictive")["units"] == "fractional"


# -- pipeline integration: enforce vs audit (needs the A1 wiring) ------------


def _mini_data() -> dict[str, Any]:
    """A tiny valid payload for run_full_pipeline (synthetic feed)."""
    import numpy as np  # noqa: PLC0415  (test-local)

    rng = np.random.default_rng(3)
    rows_total = 340  # the pipeline floor is 300 aligned observations
    returns = (0.001 + 0.01 * rng.standard_normal(rows_total)).tolist()
    signal = [1.0 if value > 0 else -1.0 for value in returns]
    mom7 = [0.0] * 7 + [
        sum(returns[index - 7 : index]) for index in range(7, rows_total)
    ]
    mom30 = [0.0] * 30 + [
        sum(returns[index - 30 : index]) for index in range(30, rows_total)
    ]
    return {
        "returns": returns,
        "signal": signal,
        "mom7": mom7,
        "mom30": mom30,
        "hypothesis": "routing-pin fixture",
    }


def test_pipeline_audit_mode_no_family() -> None:
    """No family anywhere -> audit mode, violations [], nothing raises.

    Backward-compat pin: the 48 existing call shapes behave exactly as
    before, plus the always-present instrument_audit block.
    """
    from atropos.ensemble_pipeline import (  # noqa: PLC0415
        run_full_pipeline,
    )

    report = run_full_pipeline(_mini_data())
    audit = report["instrument_audit"]
    assert audit == {
        "family": None,
        "mode": "audit",
        "violations": [],
        "units_mismatch": None,
    }


def test_pipeline_enforce_raises_on_forbidden_stage() -> None:
    """M1 pin: rule_based_tactical + stage 4 -> ValueError.

    Enforce mode blocks the forbidden stage BEFORE it runs; the
    reason rides the message.
    """
    from atropos.ensemble_pipeline import (  # noqa: PLC0415
        run_full_pipeline,
    )

    with pytest.raises(ValueError, match="forbidden for family"):
        run_full_pipeline(_mini_data(), strategy_family="rule_based_tactical")


def test_pipeline_family_from_config() -> None:
    """M2 pin: config family enforces without the kwarg.

    hypothesis_config["strategy_family"] is a first-class declaration.
    """
    from atropos.ensemble_pipeline import (  # noqa: PLC0415
        run_full_pipeline,
    )

    data = _mini_data()
    with pytest.raises(ValueError, match="forbidden for family"):
        run_full_pipeline(data, {"strategy_family": "rule_based_tactical"})


def test_pipeline_kwarg_config_conflict_value_error() -> None:
    """U7: kwarg overrides config, but a CONFLICT is a loud ValueError."""
    from atropos.ensemble_pipeline import (  # noqa: PLC0415
        run_full_pipeline,
    )

    with pytest.raises(ValueError, match="conflicts with config"):
        run_full_pipeline(
            _mini_data(),
            {"strategy_family": "carry"},
            strategy_family="rule_based_tactical",
        )


def test_pipeline_audit_flags_post_hoc_on_completed_run() -> None:
    """A completed run audited post-hoc as rule_based -> flagged.

    The stage-4 violation surfaces from the SAVED report
    (cross-session re-judgment, no rerun).
    """
    from atropos.ensemble_pipeline import (  # noqa: PLC0415
        run_full_pipeline,
    )
    from atropos.instrument_map import (  # noqa: PLC0415
        audit_report,
    )

    report = run_full_pipeline(_mini_data())
    violations = audit_report(report, "rule_based_tactical")
    assert violations, "stage 4 ran -> must be flagged post-hoc"
    assert violations[0]["tool"] == "ml_stage_4"
