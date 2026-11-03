"""Tests for atropos.null_control_suite: adversarial null controls.

Every test plants an effect whose strength is known by construction and
asserts the documented contract: PASS for a genuine edge, FAIL for pure
noise, repackaged momentum, cloned features or concentrated evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.null_control_suite import (
    calendar_control,
    known_features_control,
    momentum_control,
    placebo_placebo,
    shuffled_permutation,
    subperiod_stability,
)


def _ar_momentum(n_obs: int, persistence: float, seed: int) -> np.ndarray:
    """AR(1) factor playing the role of the momentum confounder."""
    gen = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    shock = gen.standard_normal(n_obs)
    for day in range(1, n_obs):
        mom[day] = (
            persistence * mom[day - 1] + (1.0 - persistence) * shock[day]
        )
    return mom


def test_shuffled_permutation_rejects_planted_alignment() -> None:
    gen = np.random.default_rng(7)
    signal = gen.standard_normal(300)
    outcome = signal + 0.5 * gen.standard_normal(300)
    result = shuffled_permutation(signal, outcome, n_draws=200, seed=0)
    assert result["verdict"] == "PASS"
    assert result["p_value"] <= 0.05
    assert result["stat"] > 0.8
    assert result["percentile"] >= 99.0
    assert result["n"] == 300
    assert result["n_draws"] == 200


def test_shuffled_permutation_keeps_null_on_pure_noise() -> None:
    gen = np.random.default_rng(21)
    signal = gen.standard_normal(300)
    outcome = gen.standard_normal(300)
    result = shuffled_permutation(signal, outcome, n_draws=200, seed=5)
    assert result["verdict"] == "FAIL"
    assert result["p_value"] > 0.05


def test_placebo_placebo_detects_planted_event_days() -> None:
    gen = np.random.default_rng(13)
    day_index = pd.date_range("2022-01-03", periods=300, freq="B")
    outcome = pd.Series(gen.standard_normal(300), index=day_index)
    event_positions = [40, 120, 199]
    outcome.iloc[event_positions] += 5.0
    result = placebo_placebo(event_positions, outcome, n_draws=200, seed=0)
    assert result["verdict"] == "PASS"
    assert result["p_value"] <= 0.05
    assert result["n_events"] == 3
    assert result["stat"] > 4.0
    assert result["placebo_mean_abs"] < 1.0
    # non-integer day labels must resolve through the Series index
    by_label = placebo_placebo(
        day_index[event_positions], outcome, n_draws=200, seed=0
    )
    assert by_label["verdict"] == "PASS"
    assert by_label["n_events"] == 3


def test_placebo_placebo_fails_on_ordinary_days() -> None:
    gen = np.random.default_rng(14)
    day_index = pd.date_range("2022-01-03", periods=300, freq="B")
    outcome = pd.Series(gen.standard_normal(300), index=day_index)
    result = placebo_placebo([10, 150, 250], outcome, n_draws=200, seed=0)
    assert result["verdict"] == "FAIL"
    assert result["p_value"] > 0.05
    assert result["n_events"] == 3


def test_momentum_control_absorbs_repackaged_momentum() -> None:
    momentum = _ar_momentum(400, 0.7, 101)
    gen = np.random.default_rng(102)
    repackaged = momentum + gen.standard_normal(400) * 0.35
    fwd_ret = 0.35 * momentum + gen.standard_normal(400)
    mom7 = momentum + gen.standard_normal(400) * 0.1
    mom30 = np.roll(momentum, 12) + gen.standard_normal(400) * 0.1
    result = momentum_control(fwd_ret, repackaged, mom7, mom30, robust=True)
    assert result["verdict"] == "FAIL"
    assert abs(result["stat"]) < 2.0
    assert result["HC0"] is True


def test_momentum_control_keeps_independent_signal() -> None:
    momentum = _ar_momentum(400, 0.7, 103)
    gen = np.random.default_rng(104)
    driver = gen.standard_normal(400)
    fwd_ret = 0.20 * driver + 0.25 * momentum + gen.standard_normal(400)
    mom7 = momentum + gen.standard_normal(400) * 0.1
    mom30 = np.roll(momentum, 12) + gen.standard_normal(400) * 0.1
    result = momentum_control(fwd_ret, driver, mom7, mom30, robust=True)
    assert result["verdict"] == "PASS"
    assert abs(result["stat"]) >= 2.0
    classic = momentum_control(fwd_ret, driver, mom7, mom30, robust=False)
    assert classic["HC0"] is False
    assert classic["verdict"] == "PASS"


def test_known_features_control_separates_clone_from_new_info() -> None:
    gen = np.random.default_rng(15)
    feat_one = gen.standard_normal(400)
    features = pd.DataFrame(
        {"feat_one": feat_one, "feat_two": gen.standard_normal(400)}
    )
    fwd_ret = 0.5 * feat_one + gen.standard_normal(400)

    clone = 0.9 * feat_one + gen.standard_normal(400) * 0.5
    cloned = known_features_control(fwd_ret, clone, features)
    assert cloned["verdict"] == "FAIL"
    assert abs(cloned["stat"]) < 2.0
    assert cloned["n_features"] == 2

    driver = gen.standard_normal(400)
    fwd_new = 0.4 * driver + 0.5 * feat_one + gen.standard_normal(400)
    original = known_features_control(fwd_new, driver, features)
    assert original["verdict"] == "PASS"
    assert abs(original["stat"]) >= 2.0


def test_subperiod_stability_separates_spread_from_concentrated() -> None:
    concentrated_ev = np.zeros(300)
    concentrated_ev[:100] = 1.0
    concentrated = subperiod_stability(
        concentrated_ev, periods=3, n_perm=200, seed=0
    )
    assert concentrated["verdict"] == "FAIL"
    assert concentrated["stat"] == pytest.approx(1.0)
    assert concentrated["sign_consistent"] is True
    assert concentrated["period_effects"] == pytest.approx([100.0, 0.0, 0.0])

    gen = np.random.default_rng(17)
    spread_ev = 0.2 + gen.standard_normal(300) * 0.05
    spread = subperiod_stability(spread_ev, periods=3, n_perm=200, seed=0)
    assert spread["verdict"] == "PASS"
    assert spread["stat"] < 0.6
    assert spread["sign_consistent"] is True


def test_subperiod_stability_needs_minimum_data() -> None:
    gen = np.random.default_rng(18)
    daily_ev = 0.2 + gen.standard_normal(20) * 0.05
    result = subperiod_stability(daily_ev, periods=3, n_perm=200, seed=0)
    assert result["verdict"] == "FAIL"
    assert "insufficient" in str(result.get("note", ""))


def test_calendar_control_contract_on_planted_edge() -> None:
    gen = np.random.default_rng(19)
    day_index = pd.date_range("2022-01-03", periods=400, freq="B")
    signal = pd.Series(gen.standard_normal(400), index=day_index)
    fwd_ret = pd.Series(
        0.3 * signal.to_numpy() + gen.standard_normal(400), index=day_index
    )
    dummies = pd.get_dummies(day_index.year).to_numpy(dtype=float)
    result = calendar_control(fwd_ret, signal, dummies)
    assert set(result) >= {
        "stat",
        "p_value",
        "verdict",
        "t_plain",
        "p_plain",
        "n_year_dummies",
        "n",
    }
    assert result["verdict"] == "PASS"
    assert result["n_year_dummies"] == 2
    assert result["n"] == 400
