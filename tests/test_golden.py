"""Golden regression values on frozen seeds.

Recorded 2026-11-03, immediately after the 14-module fix wave (misalign
joint-mask, DSR sr_annual/n_trials/denom2 fixes, wf_t union fix,
n_dropped_paths fix). Any drift here means a behaviour change -- review
it, do not silently update the constant. Bootstrap-based numbers are
exact for a fixed seed because every draw comes from seeded
``default_rng`` streams.
"""

import numpy as np
import pandas as pd
import pytest
from atropos.bootstrap_ci import block_bootstrap_ci, stationary_bootstrap_ci
from atropos.cross_sectional import run_cross_sectional_tests
from atropos.ensemble_pipeline import _make_panel
from atropos.evaluation_metrics import deflated_sharpe, sharpe_ratio
from atropos.incremental_regression import incremental_test
from atropos.monte_carlo import max_drawdown_distribution, sharpe_distribution
from atropos.power_calculator import detectable_effect
from atropos.robustness_battery import battery_overall, run_battery
from atropos.walkforward_framework import walkforward


def _golden_returns(
    seed: int, n_obs: int, mu: float, sigma: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n_obs)


def test_golden_block_bootstrap_ci() -> None:
    """block_bootstrap_ci on seed-7 data, seed 0, 500 draws."""
    result = block_bootstrap_ci(
        _golden_returns(7, 300, 0.001, 0.01), n_boot=500, seed=0
    )
    assert result["point_estimate"] == pytest.approx(
        -0.0003186394375102834, rel=1e-12
    )
    assert result["ci_low"] == pytest.approx(
        -0.0014240217333472494, rel=1e-12
    )
    assert result["ci_high"] == pytest.approx(
        0.0006649882047749146, rel=1e-12
    )


def test_golden_stationary_bootstrap_ci() -> None:
    """stationary_bootstrap_ci, mean_block=10, seed 0, 500 draws."""
    result = stationary_bootstrap_ci(
        _golden_returns(7, 300, 0.001, 0.01),
        mean_block=10,
        n_boot=500,
        seed=0,
    )
    assert result["ci_low"] == pytest.approx(
        -0.0013394758383348652, rel=1e-12
    )
    assert result["ci_high"] == pytest.approx(
        0.0007067151666993498, rel=1e-12
    )


def test_golden_sharpe_ratio() -> None:
    """sharpe_ratio (block CI) on seed-3 data, seed 0."""
    result = sharpe_ratio(_golden_returns(3, 500, 0.001, 0.01), seed=0)
    assert result["sr"] == pytest.approx(2.4562512958653353, rel=1e-12)
    assert result["sr_ci"][0] == pytest.approx(1.062682721695907, rel=1e-12)
    assert result["sr_ci"][1] == pytest.approx(3.953982666134719, rel=1e-12)


def test_golden_deflated_sharpe() -> None:
    """DSR with n_trials=10: benchmark E[max of 10 normals] ~ 1.5746."""
    result = deflated_sharpe(
        _golden_returns(3, 500, 0.001, 0.01), n_trials=10, seed=0
    )
    assert result["dsr"] == pytest.approx(0.8900054738713381, rel=1e-12)
    assert result["p_value"] == pytest.approx(0.1099945261286619, rel=1e-12)
    assert result["sr_benchmark_annual"] == pytest.approx(
        1.5745983013457503, rel=1e-12
    )


def test_golden_max_drawdown_distribution() -> None:
    """Drawdown percentiles, 300 sims, seed 0, no dropped paths."""
    result = max_drawdown_distribution(
        _golden_returns(5, 400, 0.0005, 0.012), n_sim=300, seed=0
    )
    assert result["mdd_p5"] == pytest.approx(-0.4656287307717065, rel=1e-12)
    assert result["mdd_p50"] == pytest.approx(-0.2701000187079951, rel=1e-12)
    assert result["mdd_p95"] == pytest.approx(-0.14934991902974215, rel=1e-12)
    assert result["n_dropped_paths"] == 0


def test_golden_sharpe_distribution() -> None:
    """Annualized SR percentiles, 300 sims, seed 0."""
    result = sharpe_distribution(
        _golden_returns(5, 400, 0.0005, 0.012), periods=252, n_sim=300, seed=0
    )
    assert result["sr_p5"] == pytest.approx(-1.7406475278702207, rel=1e-12)
    assert result["sr_p50"] == pytest.approx(-0.17568627937985426, rel=1e-12)
    assert result["sr_p95"] == pytest.approx(1.1786965403902396, rel=1e-12)
    assert result["n_dropped_paths"] == 0


def test_golden_detectable_effect() -> None:
    """The docstring number: detectable_effect(2000, 25) ~ 0.00396.

    Regression: the ncp convention moved from ``sqrt(f2 * n_obs)`` to
    the exact residual-df form ``sqrt(f2 * (n_obs - n_predictors -
    1))`` (convention note 2026-11 in atropos.power_calculator), so
    the golden value moved 0.00391288 -> 0.00396421 (+1.3%).
    """
    effect = detectable_effect(n_obs=2000, n_predictors=25)
    assert effect == pytest.approx(0.00396421462980562, rel=1e-12)


def test_golden_incremental_test() -> None:
    """True incremental signal survives with the recorded t-stat."""
    rng = np.random.default_rng(21)
    n_obs = 500
    control = rng.standard_normal(n_obs)
    signal = 0.6 * control + rng.standard_normal(n_obs) * 0.8
    outcome = 0.3 * signal + 0.2 * control + rng.standard_normal(n_obs) * 0.5
    result = incremental_test(outcome, pd.DataFrame({"c0": control}), signal)
    assert result["verdict"] == "SURVIVES"
    assert result["t_new"] == pytest.approx(11.79271310835355, rel=1e-12)


def test_golden_walkforward() -> None:
    """wf_t over the union sample, step == test_window, seed-11 frame."""
    rng = np.random.default_rng(11)
    signal = np.concatenate(
        [
            rng.standard_normal(100),
            rng.standard_normal(100) + 1.0,
            rng.standard_normal(100) - 1.0,
            rng.standard_normal(60),
        ]
    )
    outcome = np.concatenate(
        [
            rng.standard_normal(100),
            rng.standard_normal(100) * 0.5 + 0.01,
            -rng.standard_normal(100) * 0.5 + 0.01,
            rng.standard_normal(60),
        ]
    )
    frame = pd.DataFrame({"signal": signal, "outcome": outcome})
    result = walkforward(frame, train_window=120, test_window=30, step=30)
    assert result["wf_t"] == pytest.approx(-0.5850783363121813, rel=1e-12)
    assert result["n_pnl_unique"] == 240


def test_golden_cross_sectional_and_battery() -> None:
    """Panel killer verdict + battery table shape on planted data."""
    panel = _make_panel("genuine_edge", n_dates=100, n_assets=25, seed=13)
    cs = run_cross_sectional_tests(
        panel["panel_signal"], panel["panel_returns"], cost_bps=10.0
    )
    assert cs["overall_verdict"] == "PASS"
    assert cs["ic"]["mean_ic"] == pytest.approx(
        0.30856153846153844, rel=1e-12
    )

    rng = np.random.default_rng(9)
    n_obs = 750
    signal = np.concatenate(
        [
            rng.standard_normal(n_obs // 2) + 0.5,
            rng.standard_normal(n_obs - n_obs // 2),
        ]
    )
    outcome = 0.02 * signal + rng.standard_normal(n_obs) * 0.05
    table = run_battery(signal, outcome, {"threshold": 0.0, "window": 20})
    assert battery_overall(table) == "FRAGILE"
    assert len(table) == 9  # baseline + 8 attacks
