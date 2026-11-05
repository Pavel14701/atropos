"""The framework tests itself: kill-chain doctrine tests.

Atropos exists to kill hypotheses, so its own killers are held to the
doctrine they enforce.  On deterministic synthetic worlds:

* every killer PASSES a planted, real edge (the chain must let truth
  through);
* every killer FAILS / kills / stays INCONCLUSIVE on pure noise and
  NEVER passes it;
* exact collinearity is reported INCONCLUSIVE, never an arbitrary
  t-statistic laundering itself as evidence;
* the bundled demos are verifications: their internal asserts must
  hold when the demo entry points are invoked directly.

Budget: the whole file runs in a few seconds (reduced bootstrap /
simulation counts, fixed seeds everywhere).
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.cross_sectional import run_cross_sectional_tests
from atropos.ensemble_pipeline import (
    _make_synthetic,
    run_full_pipeline,
)
from atropos.ensemble_pipeline import (
    run_all_examples as ensemble_demo,
)
from atropos.evaluation_metrics import deflated_sharpe
from atropos.incremental_regression import incremental_test
from atropos.null_control_suite import (
    calendar_control,
    known_features_control,
    momentum_control,
    placebo_placebo,
    shuffled_permutation,
    subperiod_stability,
)
from atropos.robustness_battery import battery_overall, run_battery
from atropos.track_record import run_all_examples as track_record_demo
from atropos.walkforward_framework import check_no_peeking, walkforward

_N_OBS = 600
_REDUCED_PIPELINE_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}


# ---------------------------------------------------------------------------
# synthetic worlds (fixed seeds; every verdict below is frozen)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted_world() -> dict[str, Any]:
    """A world with a genuine incremental edge plus a momentum confound."""
    rng = np.random.default_rng(2024)
    mom = np.zeros(_N_OBS)
    shocks = rng.standard_normal(_N_OBS)
    for day in range(1, _N_OBS):
        mom[day] = 0.7 * mom[day - 1] + 0.3 * shocks[day]
    driver = rng.standard_normal(_N_OBS)  # hidden true factor
    signal = driver + 0.5 * mom  # candidate signal, contaminated
    fwd = 0.5 * driver + 0.25 * mom + 0.5 * rng.standard_normal(_N_OBS)
    mom12 = (
        pd.Series(mom).shift(12).to_numpy(dtype=float)
        + rng.standard_normal(_N_OBS) * 0.1
    )
    return {
        "mom": mom,
        "mom12": mom12,
        "driver": driver,
        "signal": signal,
        "fwd": fwd,
        "controls": pd.DataFrame({"mom": mom, "mom12": mom12}),
        "features": pd.DataFrame({"mom": mom, "mom12": mom12}),
        "years": pd.get_dummies(
            np.repeat(np.arange(3), _N_OBS // 3)
        ).to_numpy(dtype=float),
        "events": np.argsort(signal)[-30:],
    }


@pytest.fixture(scope="module")
def noise_world() -> dict[str, Any]:
    """Independent signal/returns draws, no planted structure."""
    rng = np.random.default_rng(777)
    return {
        "signal": rng.standard_normal(_N_OBS),
        "fwd": rng.standard_normal(_N_OBS),
        "events": np.arange(30),
    }


def _battery_world(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Persistent AR(1) signal with a strongly loaded outcome."""
    rng = np.random.default_rng(seed)
    n_days = 400
    signal = np.zeros(n_days)
    for day in range(1, n_days):
        signal[day] = 0.5 * signal[day - 1] + rng.standard_normal()
    signal = signal / float(np.std(signal))
    outcome = 0.8 * signal + rng.standard_normal(n_days)
    return signal, outcome


def _panel(seed: int, planted: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional panel: persistent per-asset factor, z-scored."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 240, 40
    factor = np.zeros((n_dates, n_assets))
    shocks = rng.standard_normal((n_dates, n_assets))
    for row in range(1, n_dates):
        factor[row] = 0.85 * factor[row - 1] + shocks[row]
    std = factor.std(axis=1, keepdims=True)
    zfactor = (factor - factor.mean(axis=1, keepdims=True)) / np.where(
        std > 0.0, std, 1.0
    )
    noise = rng.standard_normal((n_dates, n_assets))
    signal = zfactor + 0.3 * noise if planted else noise
    returns = (
        0.0005
        + 0.004 * zfactor
        + 0.01 * rng.standard_normal((n_dates, n_assets))
    )
    return pd.DataFrame(signal), pd.DataFrame(returns)


# ---------------------------------------------------------------------------
# doctrine 1: every killer lets a planted edge through
# ---------------------------------------------------------------------------
def test_null_controls_pass_on_planted_edge(
    planted_world: dict[str, Any],
) -> None:
    """All six controls must PASS an honest incremental edge."""
    world = planted_world
    verdicts = {
        "shuffled_permutation": shuffled_permutation(
            world["signal"], world["fwd"], n_draws=100, seed=1
        )["verdict"],
        "placebo_placebo": placebo_placebo(
            world["events"], world["fwd"], n_draws=100, seed=2
        )["verdict"],
        "momentum_control": momentum_control(
            world["fwd"], world["signal"], world["mom"], world["mom12"]
        )["verdict"],
        "known_features_control": known_features_control(
            world["fwd"], world["signal"], world["features"]
        )["verdict"],
        "subperiod_stability": subperiod_stability(
            world["signal"] * world["fwd"], periods=3, n_perm=100, seed=3
        )["verdict"],
        "calendar_control": calendar_control(
            world["fwd"], world["signal"], world["years"]
        )["verdict"],
    }
    failed = {
        name: verdict
        for name, verdict in verdicts.items()
        if verdict != "PASS"
    }
    assert not failed, failed


def test_incremental_test_survives_real_increment(
    planted_world: dict[str, Any],
) -> None:
    """A true driver must SURVIVE the strict controlled gate."""
    world = planted_world
    result = incremental_test(
        world["fwd"], world["controls"], world["driver"]
    )
    assert result["verdict"] == "SURVIVES"
    assert abs(float(result["t_new"])) >= 2.0


def test_battery_robust_on_stable_rule() -> None:
    """The 7-test battery must call a stable rule ROBUST."""
    signal, outcome = _battery_world(seed=11)
    table = run_battery(
        signal,
        outcome,
        {
            "threshold": 0.0,
            "window": 10,
            "n_boot": 100,
            "block": 10,
            "seed": 0,
        },
    )
    assert battery_overall(table) == "ROBUST"


def test_cross_sectional_passes_on_planted_panel() -> None:
    """A genuine cross-sectional edge passes all three panel killers."""
    signal_panel, returns_panel = _panel(seed=5, planted=True)
    report = run_cross_sectional_tests(signal_panel, returns_panel)
    assert report["ic"]["verdict"] == "PASS"
    assert report["spread"]["verdict"] == "PASS"
    assert report["monotonicity"]["verdict"] == "PASS"
    assert report["overall_verdict"] == "PASS"


def test_walkforward_passes_on_planted_edge(
    planted_world: dict[str, Any],
) -> None:
    """The expanding walk-forward must certify the planted edge."""
    world = planted_world
    frame = pd.DataFrame({"signal": world["signal"], "outcome": world["fwd"]})
    result = walkforward(
        frame, train_window=250, test_window=50, step=50, purge_gap=1
    )
    assert result["verdict"] == "PASS"
    assert float(result["wf_t"]) >= 2.0
    assert check_no_peeking(result) is True


def test_full_pipeline_survives_genuine_edge() -> None:
    """The 7-stage chain lets the canonical genuine edge through."""
    report = run_full_pipeline(
        _make_synthetic("genuine_edge", seed=4), _REDUCED_PIPELINE_CONFIG
    )
    assert report["overall_verdict"].startswith("SURVIVES")


# ---------------------------------------------------------------------------
# doctrine 2: no killer ever passes pure noise
# ---------------------------------------------------------------------------
def test_null_controls_fail_on_noise(
    noise_world: dict[str, Any], planted_world: dict[str, Any]
) -> None:
    """All six controls must FAIL independent data."""
    noise = noise_world
    world = planted_world
    verdicts = {
        "shuffled_permutation": shuffled_permutation(
            noise["signal"], noise["fwd"], n_draws=100, seed=1
        )["verdict"],
        "placebo_placebo": placebo_placebo(
            noise["events"], noise["fwd"], n_draws=100, seed=2
        )["verdict"],
        "momentum_control": momentum_control(
            noise["fwd"], noise["signal"], world["mom"], world["mom12"]
        )["verdict"],
        "known_features_control": known_features_control(
            noise["fwd"], noise["signal"], world["features"]
        )["verdict"],
        "subperiod_stability": subperiod_stability(
            noise["signal"] * noise["fwd"], periods=3, n_perm=100, seed=3
        )["verdict"],
        "calendar_control": calendar_control(
            noise["fwd"], noise["signal"], world["years"]
        )["verdict"],
    }
    passed = {
        name: verdict
        for name, verdict in verdicts.items()
        if verdict == "PASS"
    }
    assert not passed, passed


def test_incremental_test_dies_on_noise(
    noise_world: dict[str, Any],
    planted_world: dict[str, Any],
) -> None:
    """Noise is a death verdict with and without controls -- never PASS."""
    noise = noise_world
    with_controls = incremental_test(
        noise["fwd"], planted_world["controls"], noise["signal"]
    )
    without_controls = incremental_test(noise["fwd"], None, noise["signal"])
    assert with_controls["verdict"] == "REPACKAGING-DEATH"
    assert without_controls["verdict"] == "NULL-DEATH"


def test_battery_dead_on_noise() -> None:
    """The battery must kill a never-trading/noise rule as DEAD."""
    rng = np.random.default_rng(12)
    signal = rng.standard_normal(400)
    outcome = rng.standard_normal(400)
    table = run_battery(
        signal,
        outcome,
        {
            "threshold": 0.0,
            "window": 10,
            "n_boot": 100,
            "block": 10,
            "seed": 0,
        },
    )
    assert battery_overall(table) == "DEAD"


def test_cross_sectional_fails_on_noise_panel() -> None:
    """Every panel killer rejects a noise ranking."""
    signal_panel, returns_panel = _panel(seed=99, planted=False)
    report = run_cross_sectional_tests(signal_panel, returns_panel)
    assert report["ic"]["verdict"] == "FAIL"
    assert report["spread"]["verdict"] == "FAIL"
    assert report["monotonicity"]["verdict"] == "FAIL"
    assert report["overall_verdict"] == "FAIL"


def test_walkforward_fails_on_noise() -> None:
    """Noise walk-forward: small t, PASS only at the nominal 5% rate.

    A single noise run CAN luck into |t| >= 2 (seed 21 of this very
    size does) -- that is the false-positive rate of one t-threshold,
    and the reason the doctrine chains seven killers instead of one.
    The honest noise contract is the rate: at most 2 of 10 fixed seeds
    may PASS and the median |wf_t| must stay small.
    """
    passes = 0
    abs_t_values: list[float] = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n_obs = 500
        signal = rng.standard_normal(n_obs)
        frame = pd.DataFrame(
            {"signal": signal, "outcome": rng.standard_normal(n_obs)}
        )
        result = walkforward(frame, train_window=200, test_window=50, step=50)
        abs_t_values.append(abs(float(result["wf_t"])))
        if result["verdict"] == "PASS":
            passes += 1
    assert passes <= 2, f"{passes}/10 noise walk-forwards PASSed"
    assert float(np.median(abs_t_values)) < 1.5


def test_deflated_sharpe_not_significant_on_noise() -> None:
    """Noise Sharpe must not systematically beat the deflation hurdle."""
    significant = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        returns = rng.standard_normal(1500) * 0.01
        result = deflated_sharpe(returns, n_trials=10, seed=seed)
        if float(result["p_value"]) < 0.05:
            significant += 1
    assert significant <= 2, f"{significant}/10 noise runs looked significant"


def test_full_pipeline_kills_pure_noise() -> None:
    """A pure-noise candidate never leaves the chain alive."""
    rng = np.random.default_rng(31)
    n_obs = 900
    index = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    data = {
        "returns": pd.Series(0.5 * rng.standard_normal(n_obs), index=index),
        "signal": pd.Series(rng.standard_normal(n_obs), index=index),
    }
    report = run_full_pipeline(data, _REDUCED_PIPELINE_CONFIG)
    assert report["overall_verdict"] not in {
        "SURVIVES",
        "SURVIVES_WITH_WARNINGS",
    }
    assert report["killed_by_stage"] is not None


# ---------------------------------------------------------------------------
# doctrine 3: collinearity is INCONCLUSIVE, not an arbitrary number
# ---------------------------------------------------------------------------
def test_incremental_test_inconclusive_on_collinear_signal(
    planted_world: dict[str, Any],
) -> None:
    """An exact control clone yields INCONCLUSIVE with a NaN t."""
    world = planted_world
    for clone in (2.0 * world["mom"], world["mom"]):
        result = incremental_test(world["fwd"], world["controls"], clone)
        assert result["verdict"] == "INCONCLUSIVE"
        assert math.isnan(float(result["t_new"]))


def test_known_features_control_inconclusive_on_clone(
    planted_world: dict[str, Any],
) -> None:
    """Signal == known feature: INCONCLUSIVE with the collinearity note."""
    world = planted_world
    result = known_features_control(
        world["fwd"], world["mom"], world["features"]
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "collinear with controls"
    assert math.isnan(float(result["stat"]))


def test_momentum_control_inconclusive_on_clone(
    planted_world: dict[str, Any],
) -> None:
    """Signal == mom7: INCONCLUSIVE with the collinearity note."""
    world = planted_world
    result = momentum_control(
        world["fwd"], world["mom"], world["mom"], world["mom12"]
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "collinear with controls"
    assert math.isnan(float(result["stat"]))


# ---------------------------------------------------------------------------
# doctrine 4: the demos ARE verifications
# ---------------------------------------------------------------------------
def test_track_record_demo_self_verifies() -> None:
    """track_record's demo asserts all four live fates itself."""
    track_record_demo()


def test_ensemble_demo_self_verifies() -> None:
    """ensemble_pipeline's demo asserts the four canonical kill cases."""
    ensemble_demo()


@pytest.mark.parametrize(
    ("module_name", "demo_name"),
    [
        ("atropos.cost_model", "run_all_examples"),
        ("atropos.evaluation_metrics", "run_all_examples"),
        ("atropos.prereg_generator", "run_all_examples"),
        ("atropos.incremental_regression", "run_all_examples"),
    ],
)
def test_light_module_demos_run_clean(
    module_name: str, demo_name: str
) -> None:
    """Light module demos must run without raising."""
    demo = getattr(importlib.import_module(module_name), demo_name)
    demo()
