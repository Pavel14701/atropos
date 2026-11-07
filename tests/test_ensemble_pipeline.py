"""Tests for atropos.ensemble_pipeline: the seven-stage kill chain."""

import json
from types import SimpleNamespace
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
from atropos.walkforward_framework import walkforward as real_walkforward

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


def test_momentum_echo_killed_as_repackaging_at_stage_two() -> None:
    """The echo's single stage-1 FAIL is a WARN (ATR-12); stage 2 kills.

    The known-features control FAILs the echo, but under the ATR-12
    null-controls hierarchy a single null FAIL is a WARN, not a kill
    (six controls at alpha=0.05 give a 26% false-kill rate under the
    old one-FAIL-kills rule).  The repackaging kill therefore lands at
    the stage-2 specification curve: no spec keeps |t| >= 2 once the
    known features are controlled for.
    """
    result = run_full_pipeline(
        _make_synthetic("momentum_echo", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "REPACKAGING"
    assert result["killed_by_stage"] == 2
    assert "repackaging of known information" in str(result["kill_reason"])
    stages = result["stages"]
    assert len(stages) == 7
    first_stage, second_stage, *later_stages = stages
    assert first_stage["status"] == "WARN"
    assert first_stage["name"] == "null_control_suite"
    # the echo is caught by the known-features control: the synthetic
    # feature mom_noisy is an honest noisy read of latent momentum, so
    # controlling for it absorbs the echo (momentum_control now uses the
    # weaker trailing-return fallback)
    assert (
        first_stage["results"]["known_features_control"]["verdict"] == "FAIL"
    )
    assert first_stage["repackaging"] is True
    # the kill itself moved to stage 2: the curve finds no surviving
    # controlled spec
    assert second_stage["name"] == "incremental_regression"
    assert second_stage["status"] == "FAIL"
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


def test_strategy_position_returns_have_exactly_one_lag() -> None:
    """strat[0] == 0, strat[t] == pos[t-1] * ret[t]; never a two-shift.

    Old behaviour caught: a lookahead leak (``strat[t] = pos[t] *
    ret[t]``) or a dropped-bar double shift (``strat[2:] = pos[:-2] *
    ret[2:]``) in ``_strategy_position_returns`` would silently change
    every stage-5 statistic (the same lag is documented for the stage-3
    battery's ``_strat_returns``).  Mutation check: changing the
    implementation to the two-shift form makes BOTH the exact-form
    assertion and the double-shift detector fail (verified in this
    session; implementation restored afterwards).
    """
    aligned = {
        "signal": pd.Series([2.0, 1.0, 0.5, 3.0, 4.0]),
        "returns": pd.Series([0.01, -0.02, 0.03, 0.04, -0.05]),
    }
    positions, strat = ensemble_pipeline._strategy_position_returns(
        aligned, 1.5
    )
    expected_positions = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    ret = aligned["returns"].to_numpy(dtype=float)
    np.testing.assert_array_equal(positions, expected_positions)
    assert strat[0] == 0.0
    expected_strat = np.zeros(ret.size, dtype=float)
    expected_strat[1:] = expected_positions[:-1] * ret[1:]
    np.testing.assert_allclose(strat, expected_strat)
    # double-shift detector: on this input the two-shift series differs
    # from the actual one (indices 1, 2 and 4), so the inequality below
    # is a real pin, not a vacuous pass
    two_shift = np.zeros(ret.size, dtype=float)
    two_shift[2:] = expected_positions[:-2] * ret[2:]
    assert not np.allclose(strat, two_shift)


# ----------------------------------------------------------------------------
# stage 5 breakeven gate: the reported breakeven number is now a kill
# attribution, not just a reported figure (cost_bps >= breakeven, with
# a finite positive breakeven and an intact drawdown tail, attributes
# a net-envelope death to costs -> the existing COST_DEATH path)
# ----------------------------------------------------------------------------
def _thin_edge_breakeven() -> float:
    """Breakeven bps of the _thin_edge world (gross-of-cost quantity).

    The breakeven depends only on the gross strategy returns and the
    positions behind them - both cost-blind - so it can be measured
    once at any cheap cost and reused to set the boundary cost.
    """
    probe = run_full_pipeline(
        _thin_edge(), REDUCED_CONFIG | {"cost": {"cost_bps": 10.0}}
    )
    return float(probe["stages"][4]["results"]["cost"]["breakeven_cost_bps"])


def test_costs_equal_to_breakeven_kill_as_cost_death() -> None:
    """cost_bps == breakeven (inclusive boundary) -> COST_DEATH.

    Old behaviour: the breakeven was computed and REPORTED by stage 5
    but never gated on - at an elevated Sharpe floor (3.0) the gross
    bootstrap p5 (2.26 on this world) cannot clear it, so the
    gross-envelope attribution failed and the identical run read RISK
    even though the configured cost exactly ate the edge.  The gate
    turns the breakeven number into the COST_DEATH attribution.  The
    world is built so the gate is load-bearing: gross p5 < floor
    (gross_pass False), net p5 <= floor, drawdown tail intact - the
    gate is the ONLY cost_breach source.  Mutation check: removing the
    breakeven term from the cost_breach expression flips this run to
    RISK and the test fails (verified in this session).
    """
    breakeven = _thin_edge_breakeven()
    result = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG
        | {"cost": {"cost_bps": breakeven}, "mc": {"min_sr_p5": 3.0}},
    )
    assert result["overall_verdict"] == "COST_DEATH"
    assert result["killed_by_stage"] == 5
    assert "transaction costs" in str(result["kill_reason"])
    stage5 = result["stages"][4]
    assert stage5["status"] == "FAIL"
    assert stage5["results"]["cost_breach"] is True
    # the gate (not the gross-envelope comparison) drove the attribution
    assert stage5["results"]["cost"]["breakeven_breach"] is True
    # boundary recorded as exactly inclusive in-world
    assert stage5["results"]["cost"]["cost_bps"] == pytest.approx(breakeven)
    assert stage5["results"]["cost"]["breakeven_cost_bps"] == pytest.approx(
        breakeven
    )


def test_breakeven_gate_stays_silent_below_breakeven() -> None:
    """Strictly below breakeven the gate never fires (boundary sharp).

    Old behaviour: same as above - no gate existed.  Mutating the gate
    to fire on any finite breakeven (dropping the cost comparison)
    makes ``breakeven_breach`` True at 0.9x breakeven and fails this
    test (mutation check verified in this session).  At 0.9x the
    pipeline may still die as COST_DEATH through the PRE-EXISTING
    gross-envelope attribution (the net mean is then ~90% eaten, the
    net envelope fails, the gross envelope passes) - that path is not
    the gate, so the sharp assertion is ``breakeven_breach is False``.
    At 0.5x nothing at stage 5 fails at all (the run, if it dies, dies
    later - e.g. the stage-6 OOS evaluation on the degraded net
    stream).
    """
    breakeven = _thin_edge_breakeven()
    just_below = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG | {"cost": {"cost_bps": 0.9 * breakeven}},
    )
    just_below_cost = just_below["stages"][4]["results"]["cost"]
    assert just_below_cost["cost_bps"] < just_below_cost["breakeven_cost_bps"]
    assert just_below_cost["breakeven_breach"] is False
    well_below = run_full_pipeline(
        _thin_edge(),
        REDUCED_CONFIG | {"cost": {"cost_bps": 0.5 * breakeven}},
    )
    stage5 = well_below["stages"][4]
    assert stage5["status"] in {"PASS", "WARN"}
    assert stage5["results"]["cost_breach"] is False
    assert stage5["results"]["cost"]["breakeven_breach"] is False


def test_breakeven_gate_silent_on_degenerate_breakeven() -> None:
    """Inf / NaN breakeven (never-trading) can never fire the gate.

    ``breakeven_cost_bps`` returns inf for a never-trading strategy
    with a positive gross mean and NaN when there is neither edge nor
    trading (atropos.cost_model).  Unit-level on
    ``_stage_5_monte_carlo``: a full-pipeline vehicle is impossible
    (a median threshold always has non-taken rows, and a battery
    threshold override that makes the stage-5 rule never trade dies at
    the battery first), so the gate's degenerate guards are pinned
    directly.  Mutation check: replacing the gate's finiteness guard
    with a bare ``breakeven > 0`` (inf reads as breached) flips the
    first world's ``breakeven_breach`` to True and fails the test
    (verified in this session).
    """
    rng = np.random.default_rng(11)
    cfg = ensemble_pipeline._merge_config(REDUCED_CONFIG)
    # never trades (positions constant 1), positive gross mean -> inf
    al_inf = {
        "signal": pd.Series([5.0] * 240, dtype=float),
        "returns": pd.Series(0.001 + 0.01 * rng.standard_normal(240)),
    }
    res_inf = ensemble_pipeline._stage_5_monte_carlo(al_inf, cfg, 0.0)
    assert res_inf["status"] in {"PASS", "WARN"}
    assert res_inf["results"]["cost_breach"] is False
    cost_inf = res_inf["results"]["cost"]
    assert np.isinf(cost_inf["breakeven_cost_bps"])
    assert cost_inf["breakeven_breach"] is False
    # never trades, negative gross mean -> NaN breakeven, RISK envelope,
    # but the cost gate still cannot fire (nothing to eat)
    al_nan = {
        "signal": pd.Series([5.0] * 240, dtype=float),
        "returns": pd.Series(-0.001 + 0.01 * rng.standard_normal(240)),
    }
    res_nan = ensemble_pipeline._stage_5_monte_carlo(al_nan, cfg, 0.0)
    cost_nan = res_nan["results"]["cost"]
    assert np.isnan(cost_nan["breakeven_cost_bps"])
    assert cost_nan["breakeven_breach"] is False
    assert res_nan["results"]["cost_breach"] is False
    assert res_nan["status"] == "FAIL"  # envelope death, attributed RISK


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

    A pure-noise signal fails several stage-1 null controls at once, so
    under the ATR-12 hierarchy (single FAIL = WARN, >= 2 FAILs = kill)
    it is still killed at stage 1.  Stages 2-7 must be SKIPPED and say
    "killed earlier" (the context-dependent summary fixed in review).
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
    # the multi-FAIL arm of the ATR-12 hierarchy: pure noise trips far
    # more than one control, so stage 1 kills outright
    assert result["killed_by_stage"] == 1
    assert result["overall_verdict"] == "REPACKAGING"
    first_stage = result["stages"][0]
    assert first_stage["status"] == "FAIL"
    assert first_stage["name"] == "null_control_suite"
    assert "0/6 controls passed" in first_stage["summary"]
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


def test_concentrated_edge_warns_at_stage_one_dies_no_oos_edge() -> None:
    """First-third-only edge: stage-1 concentration WARN, stage-6 kill.

    The edge is genuine where it exists (full-sample regressions stay
    significant, so the factor controls PASS), but the evidence lives
    in one subperiod: the subperiod-concentration control FAILs.  Under
    the ATR-12 hierarchy a single null FAIL is a WARN, not a kill, so
    the candidate survives stage 1 and runs the chain until the stage-6
    out-of-sample evaluation finds no durable edge: NO_OOS_EDGE (the
    old one-FAIL rule killed it at stage 1 as ARTIFACT).
    """
    result = run_full_pipeline(
        _make_synthetic("concentrated_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["killed_by_stage"] == 6
    assert result["overall_verdict"] == "NO_OOS_EDGE"
    first_stage = result["stages"][0]
    assert first_stage["status"] == "WARN"
    assert first_stage["repackaging"] is False
    assert first_stage["results"]["subperiod_stability"]["verdict"] == "FAIL"
    assert first_stage["results"]["momentum_control"]["verdict"] == "PASS"
    assert (
        first_stage["results"]["known_features_control"]["verdict"] == "PASS"
    )
    # the WARN lets the chain run deeper: stages 2-5 pass, the kill
    # lands at stage 6, and only stage 7 is skipped
    assert all(
        stage["status"] in {"PASS", "WARN"} for stage in result["stages"][1:5]
    )
    stage_six = result["stages"][5]
    assert stage_six["name"] == "predictive_model_eval"
    assert stage_six["status"] == "FAIL"
    assert result["stages"][6]["status"] == "SKIPPED"
    assert result["stages"][6]["summary"] == (
        "not run: candidate killed earlier"
    )


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


def test_first_half_edge_dies_as_no_oos_edge() -> None:
    """First-half edge dies as NO_OOS_EDGE at stage 6.

    Resolves the KNOWN DEBT of 2026-11-03: before ATR-12 the first-half
    world died at stage 1 as ARTIFACT (single subperiod_stability FAIL,
    max_share ~0.6-0.77 vs the auto cap 1/3 + 0.15 = 0.483), so
    NO_OOS_EDGE had no honest series-world construction.  Under the
    ATR-12 hierarchy that single null FAIL is a WARN, the world runs
    the chain into the OOS evaluation, and the edge-free tail is caught
    there: stage-6 sign-consistency FAIL -> NO_OOS_EDGE.

    Stage-4 shrink-gate measurement (2026-11-07): this world's
    wf_shrink_ratio is 0.246 (wf_t 3.32 over is_t 13.52) - finite and
    within the 2.0x cap, so the OVERFIT_DEATH gate must NOT fire and
    the NO_OOS_EDGE meaning is preserved verbatim; the pin below keeps
    that world honest against a gate that fires on weak ratios or
    against a shrinkage-NaN bug reclassifying it INCONCLUSIVE.
    """
    result = run_full_pipeline(_first_half_edge_world(), REDUCED_CONFIG)
    assert result["overall_verdict"] == "NO_OOS_EDGE", (
        f"verdict={result['overall_verdict']} "
        f"killed_by_stage={result['killed_by_stage']}"
    )
    assert result["killed_by_stage"] == 6
    first_stage = result["stages"][0]
    assert first_stage["status"] == "WARN"
    assert first_stage["results"]["subperiod_stability"]["verdict"] == "FAIL"
    stage_four = result["stages"][3]
    shrink = float(stage_four["results"]["wf_shrink_ratio"])
    assert np.isfinite(shrink) and 0.0 < shrink <= 2.0
    assert stage_four["results"]["verdict_reason"] is None
    stage_six = result["stages"][5]
    assert stage_six["name"] == "predictive_model_eval"
    assert stage_six["status"] == "FAIL"


# ----------------------------------------------------------------------------
# stage-4 shrink gate: OVERFIT_DEATH -> DEAD; shrink NaN -> INCONCLUSIVE
# ----------------------------------------------------------------------------
def _canned_wf(**overrides: Any) -> dict[str, Any]:
    """Stage-4 result shaped like walkforward_framework's dict.

    The honest world-construction debt (KNOWN-ISSUES section 3: no
    honest series world reaches stage 4 with a dead walk-forward)
    forces the OVERFIT_DEATH and shrink-NaN cases to be exercised at
    the mapping level: the canned dict is exactly what the framework
    returns for those verdicts, and the monkeypatch swaps only the
    stage-4 walk-forward call - stages 1-3 run for real.
    """
    folds = pd.DataFrame(
        {
            "fold": [0, 1],
            "train_end": [180, 240],
            "test_start": [180, 240],
            "test_end": [210, 270],
            "n_train": [180, 240],
            "purge_gap": [0, 0],
            "hit_rate": [0.6, 0.6],
            "fold_pnl_mean": [0.01, 0.01],
        }
    )
    result: dict[str, Any] = {
        "wf_t": 3.0,
        "wf_direction_consistency": 0.6,
        "wf_shrink_ratio": 0.5,
        "verdict": "PASS",
        "verdict_reason": None,
        "n_folds": 2,
        "in_sample_t": 6.0,
        "folds": folds,
        "warnings": [],
        "mode": "expanding",
        "purge_gap": 0,
        "n_pnl_unique": 60,
        "overlap_factor": 1.0,
    }
    result.update(overrides)
    return result


def test_overfit_shrinkage_kills_as_dead_at_stage_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OVERFIT_DEATH at stage 4 is DEAD, not NO_OOS_EDGE.

    Old behaviour: every stage-4 walk-forward FAIL was classified
    NO_OOS_EDGE and the computed shrinkage was never consulted, so an
    overfit signature (finite ratio strictly above the 2.0x cap) died
    with the wrong verdict and a generic reason.
    """
    overfit = _canned_wf(
        wf_t=3.4,
        wf_shrink_ratio=3.0,
        in_sample_t=1.1,
        verdict="FAIL",
        verdict_reason="OVERFIT_DEATH: shrinkage 3.00x",
    )
    monkeypatch.setattr(
        ensemble_pipeline, "walkforward", lambda *_a, **_k: overfit
    )
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "DEAD"
    assert result["killed_by_stage"] == 4
    assert "OVERFIT_DEATH" in str(result["kill_reason"])
    stage_four = result["stages"][3]
    assert stage_four["status"] == "FAIL"
    assert "OVERFIT_DEATH" in stage_four["summary"]
    for stage in result["stages"][4:]:
        assert stage["status"] == "SKIPPED"
        assert stage["summary"] == "not run: candidate killed earlier"


def test_overfit_gate_is_independent_of_wf_t_finiteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrink > cap still kills DEAD when wf_t itself is NaN.

    The gate reads only the shrink ratio; a NaN wf_t must not reroute
    the verdict into the plain weak-edge path.  (A finite ratio above
    the cap with a NaN wf_t cannot arise from the framework's own
    arithmetic - ratio = wf_t / is_t - which is exactly why this is
    pinned with an inconsistent canned dict: it locks the gate's
    ordering, not the arithmetic.)
    """
    overfit_nan = _canned_wf(
        wf_t=float("nan"),
        wf_direction_consistency=0.5,
        wf_shrink_ratio=3.0,
        in_sample_t=1.0,
        verdict="FAIL",
        verdict_reason="OVERFIT_DEATH: shrinkage 3.00x",
    )
    monkeypatch.setattr(
        ensemble_pipeline, "walkforward", lambda *_a, **_k: overfit_nan
    )
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "DEAD"
    assert result["killed_by_stage"] == 4
    assert "OVERFIT_DEATH" in str(result["kill_reason"])


def test_shrinkage_nan_is_inconclusive_with_visible_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN-shrink INCONCLUSIVE carries its own reason, not the generic.

    Old behaviour: the framework had no INCONCLUSIVE verdict and the
    pipeline's only inconclusive reason was the generic ATR-06
    "insufficient data" string, so a degenerate-baseline stop would
    have been misreported as a configuration problem.
    """
    nan_shrink = _canned_wf(
        wf_t=float("nan"),
        wf_direction_consistency=0.0,
        wf_shrink_ratio=float("nan"),
        in_sample_t=float("nan"),
        verdict="INCONCLUSIVE",
        verdict_reason=(
            "OVERFIT check undefined: shrinkage NaN (baseline t degenerate)"
        ),
    )
    monkeypatch.setattr(
        ensemble_pipeline, "walkforward", lambda *_a, **_k: nan_shrink
    )
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "INCONCLUSIVE"
    assert result["killed_by_stage"] is None
    assert result["kill_reason"] == (
        "OVERFIT check undefined: shrinkage NaN (baseline t degenerate)"
    )
    stage_four = result["stages"][3]
    assert stage_four["status"] == "INCONCLUSIVE"
    for stage in result["stages"][4:]:
        assert stage["status"] == "SKIPPED"
        assert stage["summary"] == (
            "not run: pipeline stopped inconclusive "
            "(insufficient walk-forward data)"
        )


def test_weak_oos_with_bounded_shrink_stays_no_oos_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain weak-OOS FAIL keeps NO_OOS_EDGE and the generic reason.

    The OVERFIT_DEATH classification must not hijack the ordinary
    missing-edge path: a weak wf_t with the shrink ratio inside the
    cap (verdict_reason None) reads NO_OOS_EDGE with the stage-4
    generic kill reason, no OVERFIT_DEATH substring anywhere.
    """
    weak = _canned_wf(
        wf_t=0.4,
        wf_direction_consistency=0.52,
        wf_shrink_ratio=0.8,
        in_sample_t=0.5,
        verdict="FAIL",
        verdict_reason=None,
    )
    monkeypatch.setattr(
        ensemble_pipeline, "walkforward", lambda *_a, **_k: weak
    )
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    assert result["overall_verdict"] == "NO_OOS_EDGE"
    assert result["killed_by_stage"] == 4
    assert result["kill_reason"] == (
        "no significant out-of-sample edge (expanding walk-forward)"
    )
    assert "OVERFIT_DEATH" not in str(result["kill_reason"])


def test_shrinkage_equal_to_cap_does_not_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ratio exactly AT the 2.0x boundary never kills the pipeline.

    Runs the REAL framework through the pipeline with an adaptive cap:
    the probe call reports the world's ratio s, the patched call re-runs
    the real walk-forward with ``shrink_cap=s`` (deterministic, so the
    equality is exact).  The strict ``>`` must keep the gate silent,
    the genuine edge walks through, and no verdict_reason appears.
    Mutation target: loosening ``>`` to ``>=`` makes this run DEAD.
    """

    def _at_cap(df_wf: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        probe = real_walkforward(df_wf, **kwargs)
        kwargs["shrink_cap"] = float(probe["wf_shrink_ratio"])
        return real_walkforward(df_wf, **kwargs)

    monkeypatch.setattr(ensemble_pipeline, "walkforward", _at_cap)
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    stage_four = result["stages"][3]
    shrink = float(stage_four["results"]["wf_shrink_ratio"])
    assert np.isfinite(shrink) and shrink > 0.0
    assert stage_four["results"]["verdict_reason"] is None
    assert result["killed_by_stage"] is None
    assert result["overall_verdict"] in {
        "SURVIVES",
        "SURVIVES_WITH_WARNINGS",
    }


def test_default_config_keeps_purge_gap_and_label_horizon_distinct() -> None:
    """DEFAULT_CONFIG carries label_horizon as its own key (= 1).

    Old behaviour: the walkforward config had only purge_gap and its
    comment conflated the two ("rows purged between train and test
    (label horizon ...)").  purge_gap stays the leakage guard;
    label_horizon documents the wf_t non-overlap assumption.
    """
    wf_cfg = DEFAULT_CONFIG["walkforward"]
    assert "purge_gap" in wf_cfg
    assert "label_horizon" in wf_cfg
    assert wf_cfg["purge_gap"] == 1
    assert wf_cfg["label_horizon"] == 1


def test_label_horizon_config_error_propagates() -> None:
    """label_horizon > 1 from the config raises loudly at stage 4.

    The config key is threaded to the framework verbatim; the >1
    ValueError is a caller/config error, so it propagates (never
    classified as the data-capacity INCONCLUSIVE a typo'd
    configuration would otherwise hide behind).
    """
    with pytest.raises(ValueError, match="overlap correction"):
        run_full_pipeline(
            _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
            REDUCED_CONFIG | {"walkforward": {"label_horizon": 5}},
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
    # verdict-level (ATR-12 hierarchy: single null FAIL = WARN,
    # killed at stage 2 by spec curve instead)
    assert anchored["killed_by_stage"] == 2
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


# ----------------------------------------------------------------------------
# ATR fixes: walk-forward error classification, deflated-Sharpe kill,
# battery block default, demo ranking error honesty
# ----------------------------------------------------------------------------
def test_walkforward_config_error_propagates_not_inconclusive() -> None:
    """train_window below the framework floor: loud ValueError.

    ATR-06 refinement: only the DATA-capacity error ("too small") is
    INCONCLUSIVE.  A CALLER-CONFIG error (train_window=5, below the
    walk-forward floor of 10) is a contract breach and must surface:
    the old broad classifier (``"train_window" in message``) turned
    this exact call into an INCONCLUSIVE report -- a typo'd config
    read as "could not test".
    """
    with pytest.raises(ValueError, match="train_window"):
        run_full_pipeline(
            _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
            REDUCED_CONFIG | {"walkforward": {"train_window": 5}},
        )


def test_deflated_performance_fail_kills_not_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATR-08: a deflated-Sharpe FAIL is a stage-6 KILL (NO_OOS_EDGE).

    Regression: the stage-6 aggregator appended the DSR verdict to
    the WARN list, so a deflated FAIL surfaced as WARN status and the
    pipeline read SURVIVES_WITH_WARNINGS.  Old behaviour: with the
    same monkeypatched FAIL, stage 6 read status "WARN" and the
    overall verdict started with "SURVIVES".
    """
    fail_dp = {"verdict": "FAIL", "p_value": 0.01, "n_strategies": 50}
    monkeypatch.setattr(
        ensemble_pipeline,
        "deflated_performance",
        lambda *_args, **_kwargs: fail_dp,
    )
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=SERIES_N_OBS, seed=4),
        REDUCED_CONFIG,
    )
    stage_six = result["stages"][5]
    assert stage_six["status"] == "FAIL"
    assert "deflated Sharpe" in stage_six["summary"]
    assert result["overall_verdict"] == "NO_OOS_EDGE"
    assert result["killed_by_stage"] == 6


def test_stage_three_battery_block_defaults_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct battery call without a battery block uses DEFAULT (40).

    Regression: the block fallback was the literal ``20`` while
    ``DEFAULT_CONFIG["battery"]["block"]`` is 40 -- a direct
    ``_stage_3_robustness_battery`` call without battery config
    silently bootstrapped with half the committed block length.  Old
    behaviour: this exact capture read block=20.
    """
    captured: dict[str, Any] = {}

    def fake_run_battery(
        _signal: pd.Series,
        _returns: pd.Series,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        captured.update(config)
        return pd.DataFrame(
            {"test": ["t1", "OVERALL"], "verdict": ["PASS", "ROBUST"]}
        )

    monkeypatch.setattr(ensemble_pipeline, "run_battery", fake_run_battery)
    monkeypatch.setattr(
        ensemble_pipeline, "battery_overall", lambda _table: "ROBUST"
    )
    ensemble_pipeline._stage_3_robustness_battery(
        {
            "signal": pd.Series([0.1, -0.1, 0.2, -0.05]),
            "returns": pd.Series([0.1, -0.1, 0.2, -0.05]),
        },
        {"battery": {}, "seed": 7},
    )
    assert captured["block"] == DEFAULT_CONFIG["battery"]["block"]
    assert captured["block"] == 40


def test_demo_ranking_config_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown ranking model must RAISE, never read as a skip.

    Regression: the demo's ``except ValueError`` swallowed every
    ranking failure as the honest single-class skip (report None,
    pipeline_completed=True) -- an unknown model name, bad
    model_params or non-numeric features all read as "no candidate to
    learn from".  Old behaviour: this exact call completed the demo
    with ranking gate "skipped" instead of raising.
    """
    import atropos.ranking as ranking_module  # noqa: PLC0415
    from atropos import demo_pipeline  # noqa: PLC0415

    monkeypatch.setattr(ranking_module, "MODEL_REGISTRY", {})
    with pytest.raises(ValueError, match="unknown model"):
        demo_pipeline.run_demo_pipeline(seed=41, quick=True, verbose=False)


def test_demo_single_class_skip_still_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genuine single-class world still skips honestly.

    The re-raise path must only reject NON-single-class errors: the
    ranking module's exact single-class marker still yields report
    None, gate "skipped" and a completed pipeline.  Old behaviour:
    unchanged (this is the contract the re-raise must not break).
    """
    from atropos import demo_pipeline  # noqa: PLC0415

    def raise_single_class(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError(
            "no group could be scored: every past-only training "
            "window was single-class (check the target or shrink "
            "embargo_groups)"
        )

    monkeypatch.setattr(
        demo_pipeline, "rank_within_groups", raise_single_class
    )
    summary = demo_pipeline.run_demo_pipeline(
        seed=41, quick=True, verbose=False
    )
    assert summary["ranking_gate"] == "skipped"
    assert summary["pipeline_completed"] is True


def test_demo_all_thin_groups_skip_gate_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every scored group with < 2 candidates: gate "skipped", done.

    Regression: the metric loop ran over ZERO rankable groups,
    ``np.mean([])`` was NaN, ``NaN >= floor`` read False and the gate
    reported "fail" (pipeline_completed False) for a world that
    simply had nothing to score.  Old behaviour: this exact fake
    report produced pipeline_completed False with a NaN gate.
    """
    from atropos import demo_pipeline  # noqa: PLC0415

    thin_group = SimpleNamespace(candidates=["0"], labels=[0.0], scores=[0.4])
    monkeypatch.setattr(
        demo_pipeline,
        "rank_within_groups",
        lambda *_args, **_kwargs: {"n_scored": 1, "groups": [thin_group]},
    )
    summary = demo_pipeline.run_demo_pipeline(
        seed=41, quick=True, verbose=False
    )
    assert summary["ranking_gate"] == "skipped"
    assert summary["pipeline_completed"] is True


# ----------------------------------------------------------------------------
# demo stage1<->stage5 alignment pins: the stage-5 maker/market legs must
# consume EXACTLY the stage-1 signal row (entry_pos, side, atr, sl, tp) on
# the same world with the same cost/hold contracts - one lag, not two, and
# the banner 0.25-ATR maker offset
# ----------------------------------------------------------------------------
def _demo_stage5_replay_inputs(seed: int) -> dict[str, Any]:
    """World arrays plus the first stage-1 signal row the demo consumes.

    Rebuilds the demo world/signals from the same seed
    :func:`atropos.demo_pipeline.run_demo_pipeline` draws from, so the
    returned fields are the independent recomputation source for the
    stage-5 legs.
    """
    from atropos import demo_pipeline  # noqa: PLC0415

    world = demo_pipeline.build_ohlc_world(
        n_bars=demo_pipeline._N_BARS, seed=seed
    )
    signals = demo_pipeline.make_signals(world, seed=seed)
    first = signals.iloc[0]
    return {
        "opens": world["open"].to_numpy(),
        "highs": world["high"].to_numpy(),
        "lows": world["low"].to_numpy(),
        "closes": world["close"].to_numpy(),
        "entry": int(first["entry_pos"]),
        "side": str(first["side"]),
        "sl": float(first["sl"]),
        "tp": float(first["tp"]),
        "atr": float(first["atr"]),
    }


def test_demo_stage5_aligns_with_stage1_signal_and_one_lag() -> None:
    """Stage-5 market leg == stage-1 replay at entry_pos, not entry_pos+1.

    Two old behaviours are caught:

    * input desync - the demo re-derives its stage-5 inputs inline
      (``signals.iloc[0]``, the world arrays, the hold window, the cost
      model); any drift of one of those references away from the
      stage-1 row (a different signal row, a world rebuilt with a
      different seed, a changed hold window or cost) silently changed
      the reported ``market_r_pess``.  The independent recomputation
      here uses ONLY the stage-1 signal fields, so it diverges on every
      one of those drifts.
    * double-lag consumption - shifting the demo's ``entry`` to
      ``entry_pos + 1`` (consuming TWO lags: the generator rule already
      fills at decision+1) changed the reported R.  Mutation check:
      editing the demo stage-5 ``entry`` to ``entry_pos + 1`` fails
      this test AND the desync assertion above (verified in this
      session; demo file restored bit-for-bit afterwards).

    R2 world separation (non-vacuousness): on seed 41 the one-lag and
    two-lag pessimistic R differ (-1.7803... vs -1.7776...), asserted
    in-world below, so "demo != two-lag" cannot pass vacuously.
    """
    from atropos import demo_pipeline  # noqa: PLC0415
    from atropos.backtesting import CostModel, simulate_signal  # noqa: PLC0415

    seed = 41
    summary = demo_pipeline.run_demo_pipeline(
        seed=seed, quick=True, verbose=False
    )
    replay = _demo_stage5_replay_inputs(seed)
    market_kwargs: dict[str, Any] = {
        "hold_bars": demo_pipeline._HOLD_BARS,
        "atr_value": replay["atr"],
        "cost": CostModel(),
    }

    def market_r_pess(entry: int) -> float:
        return float(
            simulate_signal(
                replay["opens"],
                replay["highs"],
                replay["lows"],
                replay["closes"],
                entry,
                replay["side"],
                replay["sl"],
                replay["tp"],
                **market_kwargs,
            )["r_pess"]
        )

    one_lag = market_r_pess(replay["entry"])
    two_lag = market_r_pess(replay["entry"] + 1)
    # R2: the world must separate one-shift from two-shift
    assert one_lag != two_lag
    # (i) data consistency: the demo consumed exactly this signal row
    assert summary["market_r_pess"] == pytest.approx(one_lag)
    # (ii) exactly one lag: not the two-shift value
    assert summary["market_r_pess"] != pytest.approx(two_lag)


def test_demo_stage5_maker_leg_uses_banner_half_atr_offset() -> None:
    """Maker leg limit sits at fill - 0.25 ATR (the stage-5 banner).

    Regression pin for the stage-2 double-offset fix: the demo used
    to pass the pre-offset banner limit as ``maker_entry_sim``'s
    ``fill_price`` reference, which placed the limit another default
    ``delta_atr=0.25`` inside - an effective 0.5 ATR against the
    banner's 0.25 (on seed 41 the leg missed its fill entirely:
    ``maker_r_pess=+0.000R``).  The recomputation places a limit
    exactly at the banner price (``maker_entry_sim`` with
    ``fill_price=opens[entry]`` and ``delta_atr=0.25``) and asserts
    equality with the demo's ``maker_r_pess``; a ``delta_atr=0.5``
    call must differ, and the world separation is asserted in-world
    (banner -1.8723R fills vs the 0.5-ATR limit missing the fill
    entirely on seed 41), so the equality assertion cannot pass
    vacuously against the doubled offset.
    """
    from atropos import demo_pipeline  # noqa: PLC0415
    from atropos.backtesting import maker_entry_sim  # noqa: PLC0415

    seed = 41
    summary = demo_pipeline.run_demo_pipeline(
        seed=seed, quick=True, verbose=False
    )
    replay = _demo_stage5_replay_inputs(seed)
    maker_kwargs: dict[str, Any] = {
        "fill_price": float(replay["opens"][replay["entry"]]),
        "atr_value": replay["atr"],
    }
    at_banner = maker_entry_sim(
        replay["opens"],
        replay["highs"],
        replay["lows"],
        replay["closes"],
        replay["entry"],
        replay["side"],
        replay["sl"],
        replay["tp"],
        delta_atr=0.25,
        **maker_kwargs,
    )
    at_double = maker_entry_sim(
        replay["opens"],
        replay["highs"],
        replay["lows"],
        replay["closes"],
        replay["entry"],
        replay["side"],
        replay["sl"],
        replay["tp"],
        delta_atr=0.5,
        **maker_kwargs,
    )
    # world separation: the banner offset and the doubled offset differ
    assert at_banner != at_double
    # the demo's maker leg must be the banner 0.25-ATR offset (fails on
    # the live double-offset bug - see the xfail reason)
    assert summary["maker_r_pess"] == pytest.approx(at_banner)
    assert summary["maker_r_pess"] != pytest.approx(at_double)
