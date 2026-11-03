"""Tests for atropos.robustness_battery: the 7-test adversarial battery.

A persistent planted edge must come back ROBUST, pure noise must come
back FRAGILE-or-DEAD, and an edge that flips into a significant loss
in one subperiod must come back DEAD.  Regression locks: an oversized
derived window is clamped instead of crashing, a never-trading or
baseline-FAILing rule is DEAD (never ROBUST), ``window >= n_obs`` is
rejected up front, and a degenerate placebo CI is INCONCLUSIVE rather
than a silent PASS.  n_boot and block stay small so every test runs
in seconds while remaining deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos import robustness_battery
from atropos.robustness_battery import (
    _roll_mean,
    battery_overall,
    run_all_examples,
    run_battery,
)

EXPECTED_TEST_ORDER = [
    "baseline",
    "threshold_sensitivity",
    "window_sensitivity",
    "subperiod_thirds",
    "leave_one_out",
    "placebo_lag",
    "declustering",
    "sign_monotonicity",
    "OVERALL",
]

_FAST_PARAMS: dict[str, float] = {"n_boot": 200, "block": 10, "seed": 0}


def _robust_edge(
    n_obs: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """AR(1) signal with a strong persistent next-period edge (demo style)."""
    gen = np.random.default_rng(seed)
    signal = np.zeros(n_obs)
    for day in range(1, n_obs):
        signal[day] = 0.7 * signal[day - 1] + gen.standard_normal()
    signal = signal / float(np.std(signal))
    outcome = 1.2 * signal + gen.standard_normal(n_obs) * 0.6
    base_params: dict[str, float] = {
        "threshold": float(np.quantile(signal, 0.75)),
        "window": 20.0,
        **_FAST_PARAMS,
    }
    return signal, outcome, base_params


def test_run_battery_returns_the_expected_table() -> None:
    signal, outcome, base_params = _robust_edge(400, 21)
    table = run_battery(signal, outcome, base_params)
    assert list(table.columns) == [
        "test",
        "stat",
        "t_stat",
        "ci_low",
        "ci_high",
        "verdict",
        "catastrophic",
        "note",
    ]
    assert list(table["test"]) == EXPECTED_TEST_ORDER
    assert table["test"].iloc[-1] == "OVERALL"
    # OVERALL stat == number of PASS rows among the seven attacks
    attacks = table[table["test"] != "OVERALL"]
    attacks = attacks[attacks["test"] != "baseline"]
    n_pass = int((attacks["verdict"] == "PASS").sum())
    overall_stat = float(
        table.loc[table["test"] == "OVERALL", "stat"].iloc[0]
    )
    assert overall_stat == float(n_pass)
    assert battery_overall(table) == str(
        table.loc[table["test"] == "OVERALL", "verdict"].iloc[0]
    )


def test_run_battery_robust_edge_is_robust() -> None:
    signal, outcome, base_params = _robust_edge(400, 21)
    table = run_battery(signal, outcome, base_params)
    assert battery_overall(table) == "ROBUST"
    assert float(table.loc[table["test"] == "OVERALL", "stat"].iloc[0]) == 7.0
    baseline = table.loc[table["test"] == "baseline"].iloc[0]
    assert baseline["verdict"] == "PASS"
    assert float(baseline["t_stat"]) >= 2.0
    assert not bool(
        table.loc[table["test"] == "OVERALL", "catastrophic"].iloc[0]
    )


def test_run_battery_pure_noise_is_fragile() -> None:
    gen = np.random.default_rng(3)
    signal = gen.standard_normal(300)
    outcome = gen.standard_normal(300)
    base_params = {
        "threshold": 0.0,
        "window": 10,
        "n_boot": 200,
        "block": 10,
        "seed": 3,
    }
    table = run_battery(signal, outcome, base_params)
    overall = battery_overall(table)
    assert overall in {"FRAGILE", "DEAD"}
    assert float(table.loc[table["test"] == "OVERALL", "stat"].iloc[0]) < 5.0
    assert not bool(
        table.loc[table["test"] == "OVERALL", "catastrophic"].iloc[0]
    )


def test_run_battery_edge_that_flips_is_dead() -> None:
    gen = np.random.default_rng(55)
    # always-active signal whose edge is positive early and strongly
    # negative in the last third: the battery must call it DEAD
    outcome = np.concatenate(
        [
            0.5 + gen.standard_normal(200) * 0.1,
            -0.9 + gen.standard_normal(100) * 0.1,
        ]
    )
    signal = np.ones(300)
    base_params = {"threshold": 0.5, "window": 10, **_FAST_PARAMS}
    table = run_battery(signal, outcome, base_params)
    assert battery_overall(table) == "DEAD"
    subperiod = table.loc[table["test"] == "subperiod_thirds"].iloc[0]
    assert bool(subperiod["catastrophic"])
    assert subperiod["verdict"] == "FAIL"
    assert bool(table.loc[table["test"] == "OVERALL", "catastrophic"].iloc[0])


def test_run_battery_validates_base_params() -> None:
    gen = np.random.default_rng(8)
    signal = pd.Series(gen.standard_normal(200))
    outcome = pd.Series(gen.standard_normal(200))
    with pytest.raises(ValueError, match="threshold"):
        run_battery(signal, outcome, {"window": 10})
    with pytest.raises(ValueError, match="window"):
        run_battery(signal, outcome, {"threshold": 0.0})
    with pytest.raises(ValueError, match="window must be"):
        run_battery(signal, outcome, {"threshold": 0.0, "window": 0})
    with pytest.raises(ValueError, match="finite"):
        run_battery(
            signal, outcome, {"threshold": float("nan"), "window": 10}
        )
    with pytest.raises(ValueError, match="at least 60"):
        run_battery(
            gen.standard_normal(50),
            gen.standard_normal(50),
            {"threshold": 0.0, "window": 10},
        )


def test_run_battery_is_deterministic_for_fixed_seed() -> None:
    signal, outcome, _ = _robust_edge(300, 77)
    base_params: dict[str, float] = {
        "threshold": float(np.quantile(signal, 0.75)),
        "window": 15.0,
        "n_boot": 100,
        "block": 10,
        "seed": 4,
    }
    first = run_battery(signal, outcome, base_params)
    second = run_battery(signal, outcome, base_params)
    assert first.equals(second)
    assert battery_overall(first) == battery_overall(second)


def test_battery_overall_requires_overall_row() -> None:
    bare = pd.DataFrame({"test": ["baseline"], "verdict": ["PASS"]})
    with pytest.raises(ValueError, match="OVERALL"):
        battery_overall(bare)


def test_roll_mean_clamps_oversized_windows() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # window > size and window == size degrade to the expanding mean
    # instead of the historical broadcast crash; window=3 checks the
    # trailing-window path and the expanding head
    assert np.allclose(_roll_mean(values, 8), [1.0, 1.5, 2.0, 2.5, 3.0])
    assert np.allclose(_roll_mean(values, 5), [1.0, 1.5, 2.0, 2.5, 3.0])
    assert np.allclose(_roll_mean(values, 3), [1.0, 1.5, 2.0, 3.0, 4.0])
    assert np.allclose(_roll_mean(values, 1), values)


def test_run_battery_window_sensitivity_clamps_oversized_window() -> None:
    # exact review-crash scenario: minimum sample n_obs=60, window=41,
    # so the derived 1.5x window is 62 > 60 and used to crash
    # _roll_mean with "operands could not be broadcast"
    signal, outcome, base_params = _robust_edge(60, 5)
    base_params["window"] = 41.0
    table = run_battery(signal, outcome, base_params)
    assert list(table["test"]) == EXPECTED_TEST_ORDER
    row = table.loc[table["test"] == "window_sensitivity"].iloc[0]
    assert "w=60 (clamped from 62)" in str(row["note"])
    assert battery_overall(table) in {"ROBUST", "FRAGILE", "DEAD"}


def test_run_battery_rejects_window_at_or_above_sample() -> None:
    gen = np.random.default_rng(9)
    signal = gen.standard_normal(200)
    outcome = gen.standard_normal(200)
    # window == n_obs would make the placebo sample all-NaN (silent
    # PASS); the rejection must happen up front, before any compute
    with pytest.raises(ValueError, match="placebo"):
        run_battery(signal, outcome, {"threshold": 0.0, "window": 200})
    with pytest.raises(ValueError, match="window must"):
        run_battery(signal, outcome, {"threshold": 0.0, "window": 250})


def test_run_battery_never_trading_rule_is_dead() -> None:
    # threshold above the signal maximum: no active days ever, the
    # attack tests have no edge to attack -- DEAD, never ROBUST
    signal, outcome, base_params = _robust_edge(300, 21)
    base_params["threshold"] = float(signal.max() + 1.0)
    table = run_battery(signal, outcome, base_params)
    assert list(table["test"]) == EXPECTED_TEST_ORDER
    baseline = table.loc[table["test"] == "baseline"].iloc[0]
    assert baseline["verdict"] == "FAIL"
    assert battery_overall(table) == "DEAD"
    note = str(table.loc[table["test"] == "OVERALL", "note"].iloc[0])
    assert "nothing to test" in note


def test_run_battery_dead_when_baseline_itself_fails() -> None:
    # pure noise: the baseline CI includes zero, so the baseline row
    # FAILs and the verdict must propagate to OVERALL as DEAD instead
    # of letting 7 hollow attack passes read as ROBUST
    gen = np.random.default_rng(3)
    signal = gen.standard_normal(300)
    outcome = gen.standard_normal(300)
    table = run_battery(
        signal, outcome, {"threshold": 0.0, "window": 10, **_FAST_PARAMS}
    )
    baseline = table.loc[table["test"] == "baseline"].iloc[0]
    assert baseline["verdict"] == "FAIL"
    assert battery_overall(table) == "DEAD"
    note = str(table.loc[table["test"] == "OVERALL", "note"].iloc[0])
    assert "baseline itself failed" in note


def test_placebo_degenerate_ci_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a degenerate placebo CI is untestable, not a pass: the row
    # verdict must be INCONCLUSIVE and must not count as a PASS
    signal, outcome, base_params = _robust_edge(300, 77)
    placebo_seed = int(base_params["seed"]) + 999
    real_effect = robustness_battery._effect

    def nan_ci_effect(
        sig_arr: np.ndarray, target: np.ndarray, **kwargs: Any
    ) -> dict[str, Any]:
        effect = real_effect(sig_arr, target, **kwargs)
        if int(kwargs["seed"]) == placebo_seed:
            effect = {**effect, "ci_low": float("nan")}
            effect = {**effect, "ci_high": float("nan")}
        return effect

    monkeypatch.setattr(robustness_battery, "_effect", nan_ci_effect)
    table = run_battery(signal, outcome, base_params)
    row = table.loc[table["test"] == "placebo_lag"].iloc[0]
    assert row["verdict"] == "INCONCLUSIVE"
    assert "degenerate" in str(row["note"])
    attacks = table.loc[
        ~table["test"].isin(["OVERALL", "baseline"])
        & (table["verdict"] == "PASS")
    ]
    overall_stat = float(
        table.loc[table["test"] == "OVERALL", "stat"].iloc[0]
    )
    assert overall_stat == float(len(attacks))
    assert battery_overall(table) in {"ROBUST", "FRAGILE", "DEAD"}


def test_run_all_examples_demo_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_all_examples()
    output = capsys.readouterr().out
    assert "OVERALL: ROBUST" in output  # planted-edge default config
    assert "OVERALL: DEAD" in output  # never-trading rule, asserted
    assert "rejected as expected" in output  # oversized window case
