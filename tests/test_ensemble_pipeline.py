"""Tests for atropos.ensemble_pipeline: the seven-stage kill chain."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos import ensemble_pipeline
from atropos.ensemble_pipeline import (
    DEFAULT_CONFIG,
    _json_safe_value,
    _make_panel,
    _make_synthetic,
    run_full_pipeline,
)
from atropos.predictive_model_eval import ols_model_fn
from atropos.robustness_battery import run_battery

# CI-budget config: same knobs as DEFAULT_CONFIG, fewer draws/sims so the
# whole battery stays fast at n=370 (defaults: n_perm_draws=200,
# battery n_boot=1000, mc n_sim=2000, importance n_bootstrap=40).
REDUCED_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}

# _make_synthetic no longer supplies mom7/mom30 factors, so _align_inputs
# falls back to trailing-return sums: 30 leading rows are dropped and a
# synthetic of length N aligns to N - 30 observations.
SERIES_N_OBS = 400
ALIGNED_N_OBS = 370


def test_genuine_edge_survives_net_of_default_costs() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] in {"SURVIVES", "SURVIVES_WITH_WARNINGS"}
    assert result["killed_by_stage"] is None
    assert result["kill_reason"] is None
    assert result["n_obs"] == ALIGNED_N_OBS
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


def test_series_only_survivor_stage_seven_says_no_panel() -> None:
    """Series-only survivor reports stage 7 as "no panel data".

    Regression: a SURVIVING series-only candidate must not report
    stage 7 as skipped because it was "killed earlier".
    """
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] in {"SURVIVES", "SURVIVES_WITH_WARNINGS"}
    stage_seven = result["stages"][6]
    assert stage_seven["status"] == "SKIPPED"
    assert stage_seven["summary"] == "not run: optional stage (no panel data)"
    assert stage_seven["results"] == {}


def test_momentum_echo_killed_as_repackaging_at_stage_one() -> None:
    result = run_full_pipeline(
        _make_synthetic("momentum_echo", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "REPACKAGING"
    assert result["killed_by_stage"] == 1
    assert "known momentum" in str(result["kill_reason"])
    stages = result["stages"]
    assert len(stages) == 7
    first_stage, *later_stages = stages
    assert first_stage["status"] == "FAIL"
    assert first_stage["name"] == "null_control_suite"
    # the echo is caught by the known-features control: the synthetic
    # feature mom_noisy is an honest noisy read of latent momentum, so
    # controlling for it absorbs the echo (momentum_control now uses the
    # weaker trailing-return fallback)
    assert (
        first_stage["results"]["known_features_control"]["verdict"] == "FAIL"
    )
    assert first_stage["repackaging"] is True
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
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    second_run = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
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


def test_stage3_window_fallback_matches_default_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A battery dict without 'window' falls back to DEFAULT_CONFIG's.

    Regression: the fallback used to be a hardcoded 20 while
    DEFAULT_CONFIG said 10 - two different batteries depending on
    whether the caller merged the default config first.
    """
    captured: dict[str, Any] = {}

    def spy_battery(
        signal: Any, outcome: Any, params: dict[str, Any]
    ) -> pd.DataFrame:
        captured.update(params)
        return run_battery(signal, outcome, params)

    monkeypatch.setattr(ensemble_pipeline, "run_battery", spy_battery)
    aligned, _ = ensemble_pipeline._align_inputs(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4)
    )
    ensemble_pipeline._stage_3_robustness_battery(
        aligned,
        {"battery": {"threshold": None, "n_boot": 50}, "seed": 0},
    )
    assert "window" in captured
    assert captured["window"] == DEFAULT_CONFIG["battery"]["window"]


def test_config_overrides_merge_over_defaults() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
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
def _thin_edge(n_obs: int = SERIES_N_OBS, seed: int = 4) -> dict[str, Any]:
    """Genuine edge with realistic ~2bp daily return amplitude."""
    data = _make_synthetic("genuine_edge", n_obs=n_obs, seed=seed)
    data["returns"] = data["returns"] * 0.02
    return data


def test_prohibitive_costs_kill_genuine_edge_as_cost_death() -> None:
    result = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG | {"cost": {"cost_bps": 140.0}},
    )
    assert result["overall_verdict"] == "COST_DEATH"
    assert result["killed_by_stage"] == 5
    assert "transaction costs" in str(result["kill_reason"])
    stage5 = result["stages"][4]
    assert stage5["status"] == "FAIL"
    assert stage5["results"]["cost_breach"] is True
    assert stage5["results"]["cost"]["breakeven_cost_bps"] < 140.0
    # entry day excluded from both: cost_drag_mean is the BODY mean of
    # (gross - net), not the full-series mean (regression pin for the
    # [1:] unification with the Sharpe / drawdown envelope)
    gross = stage5["results"]["gross_strategy_returns"]
    net = stage5["results"]["strategy_returns"]
    assert stage5["results"]["cost"]["cost_drag_mean"] == pytest.approx(
        float(np.mean((gross - net)[1:]))
    )
    assert stage5["results"]["cost"]["cost_drag_mean"] != pytest.approx(
        float(np.mean(gross - net))
    )
    # stages 1-4 are cost-blind and still passed (WARN allowed); 6-7 never ran
    assert all(
        stage["status"] in {"PASS", "WARN"} for stage in result["stages"][:4]
    )
    assert result["stages"][5]["status"] == "SKIPPED"


def test_mdd_breach_takes_precedence_over_cost_death() -> None:
    """Gross passes, net AND mdd fail -> RISK, never COST_DEATH.

    Verdict matrix: a drawdown breach always dominates cost attribution.
    """
    result = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG
        | {
            "cost": {"cost_bps": 140.0},
            "mc": {"n_sim": 100, "worst_allowed_mdd_p5": -1e-9},
        },
    )
    assert result["overall_verdict"] == "RISK"
    assert result["killed_by_stage"] == 5
    stage5 = result["stages"][4]
    assert stage5["status"] == "FAIL"
    assert stage5["results"]["cost_breach"] is False
    assert "mDD" in stage5["summary"]


def test_zero_costs_keep_old_gross_behaviour() -> None:
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
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
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
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
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
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
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG | {"model": ols_model_fn},
    )
    assert "[model=ols_model_fn]" in result["stages"][5]["summary"]
    assert result["overall_verdict"].startswith("SURVIVES")


def test_callable_model_config_is_json_serializable() -> None:
    """Callable model becomes "<callable>" in the output config.

    Regression: a leaked callable would make the config
    non-JSON-serializable.
    """
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG | {"model": ols_model_fn},
    )
    assert result["config"]["model"] == "<callable>"
    json.dumps(result["config"])  # must not raise


def test_json_safe_value_handles_tuples_and_sets() -> None:
    """Tuples become lists, sets become sorted lists; still json.dumps-able.

    Regression: a bare set cannot be JSON-serialized at all, and a
    tuple containing a callable used to pass through with the callable
    still hidden inside.
    """
    assert _json_safe_value((1, 2, 3)) == [1, 2, 3]
    assert _json_safe_value((1, (2, 3))) == [1, [2, 3]]
    assert _json_safe_value({3, 1, 2}) == [1, 2, 3]  # sorted
    assert _json_safe_value({"b", "a"}) == ["a", "b"]
    assert _json_safe_value((ols_model_fn, 2)) == ["<callable>", 2]
    safe = _json_safe_value(
        {
            "tuple": (1, 2),
            "set": {"b", "a"},
            "callables": {ols_model_fn},
            "plain": 1.5,
        }
    )
    assert safe == {
        "tuple": [1, 2],
        "set": ["a", "b"],
        "callables": ["<callable>"],
        "plain": 1.5,
    }
    json.dumps(safe)  # must not raise


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_full_pipeline(
            _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
            REDUCED_CONFIG | {"model": "gradient_boosting"},
        )


# ----------------------------------------------------------------------------
# cross-sectional stage 7 (panel mode)
# ----------------------------------------------------------------------------
def test_panel_only_mode_runs_stage_seven_alone() -> None:
    """Panel-only survivor reads SURVIVES_CS_ONLY, never plain SURVIVES.

    ATR-03: stages 1-6 never ran, so "every attack missed" is not
    established -- the CS-scoped verdict keeps the evidence scope
    visible, and the skipped stages say only the cross-sectional stage
    was tested.
    """
    result = run_full_pipeline(
        _make_panel("genuine_edge", n_dates=200, n_assets=30, seed=13),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "SURVIVES_CS_ONLY"
    assert result["overall_verdict"] != "SURVIVES"
    assert result["killed_by_stage"] is None
    assert result["kill_reason"] is None
    assert result["n_obs"] == 200
    stages = result["stages"]
    assert [stage["stage"] for stage in stages] == list(range(1, 8))
    for skipped_stage in stages[:6]:
        assert skipped_stage["status"] == "SKIPPED"
        assert skipped_stage["summary"] == (
            "not run: panel-only mode - only the cross-sectional stage"
            " was tested"
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
    series_data = _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4)
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


# ----------------------------------------------------------------------------
# honest synthetic features (no lookahead)
# ----------------------------------------------------------------------------
def test_make_synthetic_mom12_is_shifted_not_rolled() -> None:
    """mom12 carries a 12-row NaN head (a true shift, not np.roll).

    Guard: np.roll would wrap future momentum into the head - a
    lookahead; the NaN rows are dropped by alignment instead.
    """
    data = _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4)
    features = data["features"]
    assert list(features.columns) == ["mom_noisy", "mom12"]
    mom12 = features["mom12"]
    assert mom12.head(12).isna().all()
    assert mom12.iloc[12:].notna().all()
    # the misleading mom7/mom30 entries are gone: momentum factors now
    # come from the pipeline's trailing-return fallback
    assert "mom7" not in data
    assert "mom30" not in data


def test_thin_panel_stage_seven_inconclusive_warns_not_kills() -> None:
    """Cross-sectional INCONCLUSIVE is a warning, not a kill.

    A panel with fewer effective dates than ``min_dates`` cannot test the
    cross-sectional killers; killing the candidate for missing data would
    violate the FAIL/INCONCLUSIVE taxonomy, so stage 7 must surface WARN
    and the run must survive with warnings -- under the CS-only verdict,
    since the panel-only run never tested the series attacks (ATR-03).
    """
    result = run_full_pipeline(
        _make_panel("genuine_edge", n_dates=15, n_assets=30, seed=13),
        REDUCED_CONFIG,
    )
    stage_seven = result["stages"][6]
    assert stage_seven["status"] == "WARN"
    assert stage_seven["summary"].startswith("INCONCLUSIVE")
    assert result["killed_by_stage"] is None
    assert result["overall_verdict"] == "SURVIVES_CS_ONLY_WITH_WARNINGS"


def test_collinear_control_warns_and_repackaging_still_kills() -> None:
    """A rank-deficient stage-1 control is WARN, not silent PASS or FAIL.

    A signal exactly equal to a known feature cannot be tested by
    ``known_features_control`` (collinear design -> INCONCLUSIVE); stage 1
    must report it as an untested warning while the specification curve
    still kills the exact repackaging at stage 2.
    """
    data = _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4)
    data["features"] = pd.DataFrame({"f1": data["signal"].copy()})
    result = run_full_pipeline(data, REDUCED_CONFIG)
    stage_one = result["stages"][0]
    assert stage_one["status"] == "WARN"
    assert "inconclusive (collinear, untested)" in stage_one["summary"]
    assert "known_features_control" in stage_one["summary"]
    assert result["killed_by_stage"] == 2
    assert result["overall_verdict"] == "REPACKAGING"


def test_stages_after_kill_are_skipped_with_killed_summary() -> None:
    """After a kill every later stage is SKIPPED with the kill reason.

    A pure-noise signal dies at stage 1; stages 2-7 must be SKIPPED and
    say "killed earlier" (the context-dependent summary fixed in review).
    """
    rng = np.random.default_rng(99)
    n_obs = 400
    index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    noise_data = {
        "returns": pd.Series(rng.standard_normal(n_obs) * 0.01, index=index),
        "signal": pd.Series(rng.standard_normal(n_obs), index=index),
        "hypothesis": "pure noise must die",
    }
    result = run_full_pipeline(noise_data, REDUCED_CONFIG)
    assert result["killed_by_stage"] == 1
    assert result["overall_verdict"] == "REPACKAGING"
    later_stages = [stage for stage in result["stages"] if stage["stage"] > 1]
    assert [stage["stage"] for stage in later_stages] == list(range(2, 8))
    for stage in later_stages:
        assert stage["status"] == "SKIPPED"
        assert stage["summary"] == "not run: candidate killed earlier"


# ----------------------------------------------------------------------------
# verdict-matrix completion: slow class, ARTIFACT, DEAD, NO_OOS_EDGE debt
# ----------------------------------------------------------------------------
def test_slow_genuine_edge_survives_battery_and_pipeline() -> None:
    """Slow-class edge (smoothed driver) survives; the battery passes it.

    The positive side of the fast/slow review contract: the robustness
    battery must let slow-robust edges through (window smoothing
    attenuates but never inverts them), unlike the fast class whose
    stage-3 kills are pinned - never loosened - in
    tests/test_pipeline_sweep.py.
    """
    result = run_full_pipeline(
        _make_synthetic("slow_genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] in {"SURVIVES", "SURVIVES_WITH_WARNINGS"}
    assert result["killed_by_stage"] is None
    battery_stage = result["stages"][2]
    assert battery_stage["name"] == "robustness_battery"
    assert battery_stage["status"] in {"PASS", "WARN"}


def test_concentrated_edge_dies_as_artifact_at_stage_one() -> None:
    """First-third-only edge: stage-1 concentration kill, not repackaging.

    The edge is genuine where it exists (full-sample regressions stay
    significant, so the factor controls PASS), but the evidence lives
    in one subperiod: the subperiod-concentration control FAILs and the
    verdict is ARTIFACT - the concentration arm of the stage-1 kill
    tree, distinct from REPACKAGING.
    """
    result = run_full_pipeline(
        _make_synthetic("concentrated_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "ARTIFACT"
    assert result["killed_by_stage"] == 1
    first_stage = result["stages"][0]
    assert first_stage["status"] == "FAIL"
    assert first_stage["repackaging"] is False
    assert first_stage["results"]["subperiod_stability"]["verdict"] == "FAIL"
    assert first_stage["results"]["momentum_control"]["verdict"] == "PASS"
    assert (
        first_stage["results"]["known_features_control"]["verdict"] == "PASS"
    )
    for skipped_stage in result["stages"][1:]:
        assert skipped_stage["status"] == "SKIPPED"


def test_unreachable_threshold_kills_genuine_edge_as_dead() -> None:
    """Battery threshold above every signal draw: never trades -> DEAD.

    Honest DEAD by configuration, not by loosening anything: threshold
    10.0 lies above every draw of a unit-std signal, the long/flat rule
    never takes a position ("no active days"), and stage 3 reports the
    battery's DEAD verdict.  Stages 1-2 are rule-blind and still pass:
    the edge is real, the entry rule is untradeable.
    """
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG | {"battery": {"threshold": 10.0}},
    )
    assert result["overall_verdict"] == "DEAD"
    assert result["killed_by_stage"] == 3
    stage_three = result["stages"][2]
    assert stage_three["status"] == "FAIL"
    table = stage_three["results"]["battery_table"]
    overall_note = str(table.loc[table["test"] == "OVERALL", "note"].iloc[0])
    assert "no active days" in overall_note
    assert all(
        stage["status"] in {"PASS", "WARN"} for stage in result["stages"][:2]
    )


def test_fragile_battery_kills_at_stage_three() -> None:
    """FRAGILE (< 5/7 battery) is a stage-3 FAIL -> DEAD, never a WARN.

    ATR-04 doctrine pin (FRAMEWORK.md ROBUSTNESS-DEATH: fewer than 5
    of 7 tests passed -> stop).  The slow-class world on sweep seed 7
    reads FRAGILE (3/7) under REDUCED_CONFIG; the pre-doctrine mapping
    let it through stage 3 on a WARN and the run survived -- the code
    must not be softer than the doctrine.
    """
    result = run_full_pipeline(
        _make_synthetic("slow_genuine_edge", n_obs=SERIES_N_OBS, seed=7),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "DEAD"
    assert result["killed_by_stage"] == 3
    assert result["kill_reason"] is not None
    assert "ROBUSTNESS-DEATH" in str(result["kill_reason"])
    stage_three = result["stages"][2]
    assert stage_three["status"] == "FAIL"
    assert stage_three["summary"].startswith("FRAGILE")
    assert "3/7" in stage_three["summary"]
    later_stages = result["stages"][3:]
    for stage in later_stages:
        assert stage["status"] == "SKIPPED"
        assert stage["summary"] == "not run: candidate killed earlier"


def test_undersized_walkforward_configuration_is_inconclusive() -> None:
    """train_window beyond the data: INCONCLUSIVE, not NO_OOS_EDGE.

    ATR-06: a configuration that cannot fit even one walk-forward fold
    (370 aligned observations, train_window=1000) means "could not
    test" -- undecided, never evidence of a missing edge.
    """
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG | {"walkforward": {"train_window": 1000}},
    )
    assert result["overall_verdict"] == "INCONCLUSIVE"
    assert result["killed_by_stage"] is None
    assert result["kill_reason"] == (
        "insufficient data for walk-forward configuration"
    )
    stage_four = result["stages"][3]
    assert stage_four["status"] == "INCONCLUSIVE"
    assert "insufficient data" in stage_four["summary"]
    assert "train_window" in str(stage_four["results"]["error"])
    for stage in result["stages"][4:]:
        assert stage["status"] == "SKIPPED"
        assert stage["summary"] == (
            "not run: pipeline stopped inconclusive "
            "(insufficient walk-forward data)"
        )


def _first_half_edge_world(n_obs: int = 400, seed: int = 4) -> dict[str, Any]:
    """The review's suggested NO_OOS_EDGE world: edge only in row range.

    Same latent structure as ``_make_synthetic('genuine_edge')`` but the
    returns load 2x on the driver over the FIRST half of the raw rows
    and not at all after: in-sample statistics (stages 1-2) are carried
    by the first half while the walk-forward OOS region (aligned rows
    ~181-370 after the 30-row alignment drop) sits in the edge-free
    tail.
    """
    rng = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    mom_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        mom[pos] = 0.5 * mom[pos - 1] + mom_shocks[pos]
    mom /= float(np.std(mom))
    driver = np.zeros(n_obs)
    driver_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        driver[pos] = 0.5 * driver[pos - 1] + driver_shocks[pos]
    driver /= float(np.std(driver))
    signal = driver + 0.3 * mom
    mask = np.zeros(n_obs)
    mask[: n_obs // 2] = 2.0
    fwd = mask * driver + 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
    mom_noisy = mom + 0.10 * rng.standard_normal(n_obs)
    mom12 = np.concatenate([np.full(12, np.nan), mom[:-12]]) + 0.10 * (
        rng.standard_normal(n_obs)
    )
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(fwd, index=idx),
        "signal": pd.Series(signal, index=idx),
        "features": pd.DataFrame(
            {"mom_noisy": mom_noisy, "mom12": mom12}, index=idx
        ),
        "hypothesis": "an edge that only ever existed in the first half",
    }


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "KNOWN DEBT (2026-11-03): NO_OOS_EDGE is the one verdict with no "
        "honest series-world construction in the current tree. The "
        "review's suggested first-half edge (driver x2 on the first 200 "
        "rows, 0 after) does NOT reach the walk-forward: it dies at "
        "stage 1 as ARTIFACT on every seed tried (subperiod_stability "
        "FAIL, max_share ~0.6-0.77 vs the auto cap 1/3 + 0.15 = 0.483; "
        "verified seeds 0-5 at n_obs=400). Mechanism analysis: "
        "(a) any time-CONCENTRATED edge trips the stage-1 concentration "
        "control before stage 4 ever runs; (b) spreading the edge so all "
        "three subperiod thirds stay under the cap caps the early "
        "loading at ~0.3, which then dies at stage 3 instead (battery "
        "baseline never excludes zero -> DEAD, the wrong verdict); "
        "(c) regime-flip worlds (returns = +driver then -driver) die at "
        "stage 1 as REPACKAGING (no full-sample linear edge for the "
        "momentum control - verified seeds 0-2); (d) return-MEAN-shift "
        "worlds (constant +mu early, -mu late) corrupt the walk-forward "
        "intercept exactly when they drive the battery's own subperiod "
        "thirds negative - stage-3 DEAD again. The structural cause: "
        "the battery attacks the LAGGED long/flat rule (harder) while "
        "stage 4 tests the contemporaneous sign rule (easier), so any "
        "world that clears stage 3 still has contemporaneous OOS power. "
        "Kill-gate thresholds were NOT touched. If this ever XPASSes, "
        "someone built the world honestly - promote it into the demo "
        "matrix."
    ),
    strict=True,
)
def test_first_half_edge_dies_as_no_oos_edge() -> None:
    """First-half edge must die as NO_OOS_EDGE (currently unreachable).

    Pins the gap: the pipeline's verdict matrix defines NO_OOS_EDGE
    (stage 4 walk-forward FAIL / stage 6 OOS FAIL), but every honest
    construction attempt is intercepted earlier - see the xfail reason
    for the full analysis. The assertion stays strict so an honest
    construction surfacing later turns the XPASS into a signal to
    promote the case into the demo matrix.
    """
    result = run_full_pipeline(_first_half_edge_world(), REDUCED_CONFIG)
    assert result["overall_verdict"] == "NO_OOS_EDGE", (
        f"verdict={result['overall_verdict']} "
        f"killed_by_stage={result['killed_by_stage']}"
    )


# ----------------------------------------------------------------------------
# ATR-07 / ATR-10: prereg anchor + specification-curve trust gates
# ----------------------------------------------------------------------------
def test_prereg_hash_echoes_into_the_report() -> None:
    """The report anchors to the frozen prereg hash verbatim (ATR-07).

    ``prereg_hash`` is a pure echo: the verdict must stay attached to
    the preregistration it was preregistered under, so the hash passed
    in comes back on the report; the default (None) marks the run as
    unanchored instead of pretending an anchor exists.
    """
    anchored = run_full_pipeline(
        _make_synthetic("momentum_echo", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
        prereg_hash="a1b2c3d4",
    )
    assert anchored["prereg_hash"] == "a1b2c3d4"
    # the echo works on a kill too: the anchor is run-level, not
    # verdict-level
    assert anchored["killed_by_stage"] == 1
    unanchored = run_full_pipeline(
        _make_synthetic("momentum_echo", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert unanchored["prereg_hash"] is None


def test_stage_two_thin_curve_survives_call_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SURVIVES call backed by < 5 testable specs must not pass (ATR-10).

    Taking the max over 1-4 specs is a multiple search too small to
    trust: the stage downgrades it to INCONCLUSIVE (WARN), never a
    silent pass.  The canned thin curve is unreachable through the
    pipeline's fixed 5x2 lattice for a non-constant signal (the five
    uncontrolled specs always test), which is exactly why the gate is
    asserted directly at the stage level.
    """
    table = pd.DataFrame(
        {
            "t_new": [3.2, 0.5, -0.4],
            "verdict": [
                "SURVIVES",
                "REPACKAGING-DEATH",
                "REPACKAGING-DEATH",
            ],
        }
    )
    thin_summary = {
        "verdict": "SURVIVES",
        "n_specs": 3,
        "n_surviving": 1,
        "n_surviving_controlled": 1,
        "max_abs_t": 3.2,
        "min_abs_t": 0.4,
        "sign_consistent": False,
        "t_rule": 2.0,
        "bonferroni_alpha": 0.05 / 3,
    }
    monkeypatch.setattr(
        ensemble_pipeline,
        "specification_curve",
        lambda *_args, **_kwargs: table,
    )
    monkeypatch.setattr(
        ensemble_pipeline,
        "curve_summary",
        lambda *_args, **_kwargs: thin_summary,
    )
    stage = ensemble_pipeline._stage_2_specification_curve(
        {"returns": None, "features": None, "signal": None}, {}
    )
    assert stage["status"] == "WARN"
    assert stage["summary"].startswith("INCONCLUSIVE")
    assert "only 3 testable specs" in stage["summary"]


def test_stage_two_five_spec_curve_is_trustworthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thin-curve gate boundary: exactly 5 testable specs pass.

    The downgrade fires strictly below 5 specs; a 5-spec SURVIVES call
    with a controlled survivor is a genuine stage-2 PASS (ATR-10 adds a
    floor, it does not move the |t| >= 2 rule).
    """
    table = pd.DataFrame({"t_new": [2.4, 2.1, 0.5, 0.3, 0.1], "n": [500] * 5})
    floor_summary = {
        "verdict": "SURVIVES",
        "n_specs": 5,
        "n_surviving": 2,
        "n_surviving_controlled": 1,
        "max_abs_t": 2.4,
        "min_abs_t": 0.1,
        "sign_consistent": True,
        "t_rule": 2.0,
        "bonferroni_alpha": 0.01,
    }
    monkeypatch.setattr(
        ensemble_pipeline,
        "specification_curve",
        lambda *_args, **_kwargs: table,
    )
    monkeypatch.setattr(
        ensemble_pipeline,
        "curve_summary",
        lambda *_args, **_kwargs: floor_summary,
    )
    stage = ensemble_pipeline._stage_2_specification_curve(
        {"returns": None, "features": None, "signal": None}, {}
    )
    assert stage["status"] == "PASS"
    assert stage["summary"].startswith("SURVIVES")
