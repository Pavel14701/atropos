"""Tests for atropos.ensemble_pipeline: the seven-stage kill chain."""

from typing import Any

import numpy as np
import pytest
from atropos.ensemble_pipeline import (
    DEFAULT_CONFIG,
    _make_panel,
    _make_synthetic,
    run_full_pipeline,
)
from atropos.predictive_model_eval import ols_model_fn

# CI-budget config: same knobs as DEFAULT_CONFIG, fewer draws/sims so the
# whole battery stays fast at n=300 (defaults: n_perm_draws=200,
# battery n_boot=1000, mc n_sim=2000, importance n_bootstrap=40).
REDUCED_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}


def test_genuine_edge_survives_net_of_default_costs() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4), REDUCED_CONFIG
    )
    assert result["overall_verdict"] in {"SURVIVES", "SURVIVES_WITH_WARNINGS"}
    assert result["killed_by_stage"] is None
    assert result["kill_reason"] is None
    assert result["n_obs"] == 300
    assert str(result["hypothesis"]).startswith("persistent order-flow")
    assert isinstance(result["date_range"], tuple)
    assert len(result["date_range"]) == 2
    assert result["runtime_seconds"] >= 0.0
    stages = result["stages"]
    assert len(stages) == 7
    assert [stage["stage"] for stage in stages] == [1, 2, 3, 4, 5, 6, 7]
    assert all(stage["status"] in {"PASS", "WARN"} for stage in stages[:6])
    assert stages[6]["name"] == "cross_sectional"
    assert stages[6]["status"] == "SKIPPED"  # no panel in the input
    # stage 5 runs net of the default 10bps one-way cost
    stage5_cost = stages[4]["results"]["cost"]
    assert stage5_cost["cost_bps"] == 10.0
    assert stages[4]["results"]["cost_breach"] is False
    assert stage5_cost["annual_turnover"] > 0.0


def test_momentum_echo_killed_as_repackaging_at_stage_one() -> None:
    result = run_full_pipeline(
        _make_synthetic("momentum_echo", n_obs=300, seed=4), REDUCED_CONFIG
    )
    assert result["overall_verdict"] == "REPACKAGING"
    assert result["killed_by_stage"] == 1
    assert "known momentum" in str(result["kill_reason"])
    stages = result["stages"]
    assert len(stages) == 7
    first_stage, *later_stages = stages
    assert first_stage["status"] == "FAIL"
    assert first_stage["name"] == "null_control_suite"
    assert first_stage["results"]["momentum_control"]["verdict"] == "FAIL"
    for skipped_stage in later_stages:
        assert skipped_stage["status"] == "SKIPPED"
        assert skipped_stage["summary"] == "not run: candidate killed earlier"
        assert skipped_stage["results"] == {}


def _verdict_skeleton(result: dict[str, Any]) -> list[Any]:
    """Verdict/summary fields of a pipeline result, floats excluded."""
    return [
        result["overall_verdict"],
        result["killed_by_stage"],
        result["kill_reason"],
        result["n_obs"],
        result["hypothesis"],
        result["date_range"],
        [
            (
                stage["stage"],
                stage["name"],
                stage["status"],
                stage["summary"],
            )
            for stage in result["stages"]
        ],
    ]


def test_pipeline_deterministic_for_identical_inputs() -> None:
    first_run = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4), REDUCED_CONFIG
    )
    second_run = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4), REDUCED_CONFIG
    )
    assert _verdict_skeleton(first_run) == _verdict_skeleton(second_run)
    assert first_run["config"] == second_run["config"]


def test_default_config_contract_battery_key_is_n_boot() -> None:
    assert "n_boot" in DEFAULT_CONFIG["battery"]
    for required_key in (
        "n_perm_draws",
        "placebo_top_quantile",
        "battery",
        "walkforward",
        "purged_cv",
        "wf_eval",
        "importance",
        "mc",
        "deflated",
        "seed",
    ):
        assert required_key in DEFAULT_CONFIG


def test_config_overrides_merge_over_defaults() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4), REDUCED_CONFIG
    )
    merged = result["config"]
    assert merged["n_perm_draws"] == 30
    assert merged["battery"]["n_boot"] == 100
    assert merged["mc"]["n_sim"] == 100
    assert merged["importance"]["n_bootstrap"] == 10
    # non-overridden nested keys keep their defaults (merge, not replace)
    assert merged["battery"]["block"] == DEFAULT_CONFIG["battery"]["block"]
    assert merged["mc"]["block"] == DEFAULT_CONFIG["mc"]["block"]
    assert merged["deflated"] == DEFAULT_CONFIG["deflated"]
    assert merged["seed"] == DEFAULT_CONFIG["seed"]


def test_pipeline_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="must contain 'returns'"):
        run_full_pipeline({"returns": [0.1] * 320}, REDUCED_CONFIG)
    with pytest.raises(ValueError, match="need >= 300 aligned observations"):
        run_full_pipeline(
            {"returns": [0.01] * 250, "signal": [0.02] * 250},
            REDUCED_CONFIG,
        )


# ----------------------------------------------------------------------------
# cost model integration (stage 5)
# ----------------------------------------------------------------------------
def _thin_edge(n_obs: int = 300, seed: int = 4) -> dict[str, Any]:
    """Genuine edge with realistic ~2bp daily return amplitude."""
    data = _make_synthetic("genuine_edge", n_obs=n_obs, seed=seed)
    data["returns"] = data["returns"] * 0.02
    return data


def test_prohibitive_costs_kill_genuine_edge_as_cost_death() -> None:
    result = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG | {"cost": {"cost_bps": 300.0}},
    )
    assert result["overall_verdict"] == "COST_DEATH"
    assert result["killed_by_stage"] == 5
    assert "transaction costs" in str(result["kill_reason"])
    stage5 = result["stages"][4]
    assert stage5["status"] == "FAIL"
    assert stage5["results"]["cost_breach"] is True
    assert stage5["results"]["cost"]["breakeven_cost_bps"] < 300.0
    # stages 1-4 are cost-blind and still passed (WARN allowed); 6-7 never ran
    assert all(
        stage["status"] in {"PASS", "WARN"} for stage in result["stages"][:4]
    )
    assert result["stages"][5]["status"] == "SKIPPED"


def test_zero_costs_keep_old_gross_behaviour() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4),
        REDUCED_CONFIG | {"cost": {"cost_bps": 0.0}},
    )
    stage5 = result["stages"][4]["results"]
    assert stage5["cost"]["cost_bps"] == 0.0
    assert stage5["cost"]["cost_drag_mean"] == 0.0
    np.testing.assert_allclose(
        stage5["strategy_returns"], stage5["gross_strategy_returns"]
    )
    assert result["overall_verdict"].startswith("SURVIVES")


def test_turnover_ceiling_produces_warning_not_kill() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4),
        REDUCED_CONFIG | {"cost": {"max_turnover_annual": 1.0}},
    )
    assert result["overall_verdict"] == "SURVIVES_WITH_WARNINGS"
    assert result["stages"][4]["status"] == "WARN"
    assert "turnover" in result["stages"][4]["summary"]


# ----------------------------------------------------------------------------
# pluggable model (stage 6)
# ----------------------------------------------------------------------------
def test_knn_model_runs_stage_six_and_survives() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4),
        REDUCED_CONFIG
        | {"model": "knn", "model_params": {"n_neighbors": 25}},
    )
    stage6 = result["stages"][5]
    assert stage6["name"] == "predictive_model_eval"
    assert "[model=knn]" in stage6["summary"]
    assert stage6["status"] in {"PASS", "WARN"}
    assert result["overall_verdict"].startswith("SURVIVES")


def test_callable_model_factory_is_accepted() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=300, seed=4),
        REDUCED_CONFIG | {"model": ols_model_fn},
    )
    assert "[model=ols_model_fn]" in result["stages"][5]["summary"]
    assert result["overall_verdict"].startswith("SURVIVES")


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_full_pipeline(
            _make_synthetic("genuine_edge", n_obs=300, seed=4),
            REDUCED_CONFIG | {"model": "gradient_boosting"},
        )


# ----------------------------------------------------------------------------
# cross-sectional stage 7 (panel mode)
# ----------------------------------------------------------------------------
def test_panel_only_mode_runs_stage_seven_alone() -> None:
    result = run_full_pipeline(
        _make_panel("genuine_edge", n_dates=200, n_assets=30, seed=13),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"].startswith("SURVIVES")
    assert result["n_obs"] == 200
    stages = result["stages"]
    assert [stage["stage"] for stage in stages] == list(range(1, 8))
    for skipped_stage in stages[:6]:
        assert skipped_stage["status"] == "SKIPPED"
        assert (
            skipped_stage["summary"]
            == "not run: panel-only mode (no series inputs)"
        )
    assert stages[6]["name"] == "cross_sectional"
    assert stages[6]["status"] == "PASS"


def test_noise_panel_dies_as_cs_no_edge() -> None:
    result = run_full_pipeline(
        _make_panel("cs_noise", n_dates=200, n_assets=30, seed=13),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "CS_NO_EDGE"
    assert result["killed_by_stage"] == 7
    assert result["stages"][6]["status"] == "FAIL"


def test_series_plus_panel_runs_all_seven_stages() -> None:
    series_data = _make_synthetic("genuine_edge", n_obs=300, seed=4)
    panel_data = _make_panel(
        "genuine_edge", n_dates=200, n_assets=30, seed=13
    )
    result = run_full_pipeline(
        series_data | panel_data,
        REDUCED_CONFIG,
    )
    stages = result["stages"]
    assert len(stages) == 7
    assert all(stage["status"] in {"PASS", "WARN"} for stage in stages)
    assert result["overall_verdict"].startswith("SURVIVES")
