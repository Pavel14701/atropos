"""Tests for atropos.incremental_regression: the repackaging kill rule.

A genuinely new predictor must keep |t| >= 2 after known controls and
across the specification curve; a repackaged one must die.  All data is
planted with known coefficients so the verdicts are structural.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.incremental_regression import (
    DEFAULT_TRANSFORMS,
    curve_summary,
    incremental_test,
    specification_curve,
)


def _genuine_edge(
    n_obs: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Outcome driven by the new signal and one control, plus noise."""
    gen = np.random.default_rng(seed)
    controls = pd.DataFrame(
        {
            "ctrl_a": gen.standard_normal(n_obs),
            "ctrl_b": gen.standard_normal(n_obs),
        }
    )
    signal = gen.standard_normal(n_obs)
    outcome = (
        0.4 * signal
        + 0.5 * controls["ctrl_a"].to_numpy()
        + gen.standard_normal(n_obs)
    )
    return outcome, controls, signal


def test_incremental_test_genuine_signal_survives() -> None:
    outcome, controls, signal = _genuine_edge(400, 95)
    result = incremental_test(outcome, controls, signal, robust=True)
    assert result["verdict"] == "SURVIVES"
    assert abs(result["t_new"]) >= 2.0
    assert result["delta_r2"] > 0.0
    assert result["r2_full"] >= result["r2_restricted"]
    assert result["n"] == 400
    assert result["robust"] is True


def test_incremental_test_repackaged_momentum_dies() -> None:
    gen = np.random.default_rng(96)
    n_obs = 400
    momentum = np.zeros(n_obs)
    shock = gen.standard_normal(n_obs)
    for day in range(1, n_obs):
        momentum[day] = 0.6 * momentum[day - 1] + 0.4 * shock[day]
    controls = pd.DataFrame(
        {
            "mom7": momentum + gen.standard_normal(n_obs) * 0.1,
            "mom30": np.roll(momentum, 12),
        }
    )
    outcome = 0.35 * momentum + gen.standard_normal(n_obs)
    signal = momentum + gen.standard_normal(n_obs) * 0.35
    result = incremental_test(outcome, controls, signal, robust=True)
    assert result["verdict"] == "REPACKAGING-DEATH"
    assert abs(result["t_new"]) < 2.0
    assert result["delta_r2"] >= 0.0


def test_incremental_test_accepts_none_controls() -> None:
    gen = np.random.default_rng(97)
    signal = gen.standard_normal(300)
    outcome = 0.4 * signal + gen.standard_normal(300)
    result = incremental_test(outcome, None, signal)
    assert result["verdict"] == "SURVIVES"
    assert abs(result["t_new"]) >= 2.0
    assert result["n"] == 300
    # a constant-only restricted model explains nothing
    assert result["r2_restricted"] == pytest.approx(0.0, abs=1e-12)


def test_specification_curve_default_grid_shape() -> None:
    outcome, controls, signal = _genuine_edge(300, 93)
    table = specification_curve(outcome, [controls], signal)
    assert len(table) == 10  # 5 transforms x (no controls + one set)
    assert set(table.columns) == {
        "spec",
        "transform",
        "controls",
        "n",
        "t_new",
        "p_new",
        "delta_r2",
        "verdict",
    }
    assert set(table["transform"]) == set(DEFAULT_TRANSFORMS)
    assert set(table["controls"]) == {"none", "set0"}
    assert list(table["spec"]) == list(range(10))


def test_specification_curve_custom_specs_are_keyword_only() -> None:
    outcome, controls, signal = _genuine_edge(300, 94)
    specs: list[dict[str, Any]] = [
        {"controls": 0, "transform": "rank"},
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.05},
    ]
    table = specification_curve(
        outcome, [controls], signal, specs_to_try=specs, robust=False
    )
    assert len(table) == 2
    assert list(table["transform"]) == ["rank", "winsorize"]
    assert list(table["controls"]) == ["set0", "none"]
    with pytest.raises(TypeError):
        specification_curve(  # type: ignore[call-arg]
            outcome, [controls], signal, specs
        )


def test_curve_summary_applies_the_kill_rule() -> None:
    surviving = curve_summary(pd.DataFrame({"t_new": [3.0, 0.5, -0.1]}))
    assert surviving["verdict"] == "SURVIVES"
    assert surviving["n_specs"] == 3
    assert surviving["n_surviving"] == 1
    assert surviving["max_abs_t"] == pytest.approx(3.0)
    assert surviving["min_abs_t"] == pytest.approx(0.1)
    assert surviving["sign_consistent"] is False

    dead = curve_summary(pd.DataFrame({"t_new": [1.5, -1.9, 0.2]}))
    assert dead["verdict"] == "REPACKAGING-DEATH"
    assert dead["n_surviving"] == 0
    assert dead["max_abs_t"] == pytest.approx(1.9)

    aligned = curve_summary(pd.DataFrame({"t_new": [2.5, 3.0]}))
    assert aligned["verdict"] == "SURVIVES"
    assert aligned["n_surviving"] == 2
    assert aligned["sign_consistent"] is True
    assert aligned["t_rule"] == 2.0


def test_curve_on_genuine_edge_survives_every_spec() -> None:
    outcome, controls, signal = _genuine_edge(300, 98)
    table = specification_curve(outcome, [controls], signal, robust=True)
    summary = curve_summary(table)
    assert summary["verdict"] == "SURVIVES"
    assert summary["n_specs"] == 10
    assert summary["n_surviving"] == 10
    assert summary["max_abs_t"] >= 2.0
    assert (table["verdict"] == "SURVIVES").all()


def test_robust_flag_variants_both_run_on_tiny_data() -> None:
    gen = np.random.default_rng(92)
    n_obs = 60
    controls = pd.DataFrame(
        {
            "ctrl_a": gen.standard_normal(n_obs),
            "ctrl_b": gen.standard_normal(n_obs),
        }
    )
    signal = gen.standard_normal(n_obs)
    outcome = (
        0.5 * signal
        + 0.3 * controls["ctrl_a"].to_numpy()
        + gen.standard_normal(n_obs)
    )
    result_robust = incremental_test(outcome, controls, signal, robust=True)
    result_classic = incremental_test(outcome, controls, signal, robust=False)
    assert result_robust["robust"] is True
    assert result_classic["robust"] is False
    for result in (result_robust, result_classic):
        assert np.isfinite(result["t_new"])
        assert result["verdict"] == "SURVIVES"
        assert result["n"] == n_obs


def test_winsorization_strength_changes_the_estimate() -> None:
    gen = np.random.default_rng(91)
    driver = gen.standard_normal(400)
    outcome = 0.4 * driver + gen.standard_normal(400)
    signal = driver + gen.standard_normal(400) * 0.3
    signal[0] = 60.0  # one extreme outlier dominates the raw scale
    specs = [
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.01},
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.45},
    ]
    table = specification_curve(outcome, [], signal, specs_to_try=specs)
    assert list(table["transform"]) == ["winsorize", "winsorize"]
    tight = float(table["t_new"].iloc[0])
    loose = float(table["t_new"].iloc[1])
    assert np.isfinite(tight)
    assert np.isfinite(loose)
    assert abs(tight - loose) > 0.5
