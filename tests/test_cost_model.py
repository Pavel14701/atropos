"""Tests for atropos.cost_model turnover and transaction-cost tests.

Hand-computed net-return arrays, exact annual-turnover and breakeven
arithmetic, COST-DEATH verdicts on a seeded strong strategy, seed
determinism, input validation, and the inconclusive guards.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos.cost_model import (
    annual_turnover,
    apply_proportional_costs,
    breakeven_cost_bps,
    cost_death_test,
    turnover_series,
)


def _build_strategy(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Strong long/flat sign strategy on an AR(1) asset (750 days).

    The asset returns are positively autocorrelated, so being long
    whenever yesterday's return was positive captures a real edge
    while still trading often enough for costs to matter.
    """
    rng = np.random.default_rng(seed)
    n_obs = 750
    shocks = rng.standard_normal(n_obs)
    asset = np.zeros(n_obs)
    for step in range(1, n_obs):
        asset[step] = 0.5 * asset[step - 1] + 0.01 * shocks[step]
    positions = np.zeros(n_obs)
    positions[1:] = (asset[:-1] > 0).astype(float)
    return positions * asset, positions


def test_turnover_series_known_answer() -> None:
    tvr = turnover_series([0.0, 1.0, 1.0, 0.0, 0.0])
    assert tvr[0] == 0.0
    assert tvr == pytest.approx([0.0, 1.0, 0.0, 1.0, 0.0])
    assert turnover_series([2.0, 2.0, 2.0]) == pytest.approx([0.0, 0.0, 0.0])


def test_apply_proportional_costs_hand_computed() -> None:
    positions = [0.0, 1.0, 1.0, 0.0, 0.0]
    gross = [0.01, 0.02, -0.01, 0.0, 0.01]
    net = apply_proportional_costs(gross, positions, 10.0)
    # turnover = [0, 1, 0, 1, 0]; cost = 10 bps = 0.001 per unit traded
    assert net == pytest.approx([0.01, 0.019, -0.01, -0.001, 0.01])
    zero_cost = apply_proportional_costs(gross, positions, 0.0)
    assert zero_cost == pytest.approx(gross)


def test_annual_turnover_exact_small_case() -> None:
    out = annual_turnover([0.0, 1.0, 1.0, 0.0, 0.0], periods=10)
    # mean turnover = 2/5 = 0.4 per period -> 4.0 per 10 periods
    assert out["annual_turnover"] == pytest.approx(4.0)
    assert out["n_changes"] == 2
    assert out["n_obs"] == 5
    daily = annual_turnover([0.0, 1.0, 1.0, 0.0, 0.0])
    assert daily["annual_turnover"] == pytest.approx(0.4 * 252)


def test_breakeven_exact_and_zero_turnover() -> None:
    gross = [0.001, 0.001, 0.001, 0.001]
    out = breakeven_cost_bps(gross, [0.0, 1.0, 1.0, 1.0])
    # mean gross 0.001 / mean turnover 0.25 -> 1e4 * 0.004 = 40 bps
    assert out["breakeven_cost_bps"] == pytest.approx(40.0)
    flat = breakeven_cost_bps([0.001, 0.002], [1.0, 1.0])
    assert math.isinf(flat["breakeven_cost_bps"])


def test_cost_death_passes_at_10_bps() -> None:
    strategy, positions = _build_strategy(3)
    out = cost_death_test(
        strategy, positions, 10.0, n_boot=300, block=20, seed=5
    )
    assert out["verdict"] == "PASS"
    assert out["ci_low"] > 0.0
    assert out["cost_drag_mean"] > 0.0
    assert out["gross_sharpe"] > out["net_sharpe"]
    assert out["n_obs"] == 750
    assert out["n_changes"] > 0
    assert out["breakeven_cost_bps"] > 10.0
    assert math.isnan(out["p_value"])


def test_cost_death_fails_at_500_bps() -> None:
    strategy, positions = _build_strategy(3)
    out = cost_death_test(
        strategy, positions, 500.0, n_boot=300, block=20, seed=5
    )
    assert out["verdict"] == "FAIL"
    assert out["cost_drag_mean"] > 0.0
    assert out["stat"] < 0.0
    assert out["net_sharpe"] < 0.0


def test_cost_death_deterministic_with_fixed_seed() -> None:
    strategy, positions = _build_strategy(11)
    first = cost_death_test(strategy, positions, 25.0, seed=1)
    repeat = cost_death_test(strategy, positions, 25.0, seed=1)
    # p_value is a documented NaN placeholder: NaN != NaN, so compare
    # key by key with NaN treated as equal to itself
    assert first.keys() == repeat.keys()
    for key in first:
        if isinstance(first[key], float) and math.isnan(first[key]):
            assert math.isnan(float(repeat[key]))
        else:
            assert first[key] == repeat[key]


def test_apply_proportional_costs_validates_inputs() -> None:
    with pytest.raises(ValueError):
        apply_proportional_costs([0.01, 0.02], [0.0, 1.0], -1.0)
    with pytest.raises(ValueError):
        apply_proportional_costs([0.01, 0.02, 0.03], [0.0, 1.0], 10.0)


def test_breakeven_and_cost_death_validate_lengths() -> None:
    with pytest.raises(ValueError):
        breakeven_cost_bps([0.01, 0.02, 0.03], [0.0, 1.0])
    with pytest.raises(ValueError):
        cost_death_test([0.01, 0.02, 0.03], [0.0, 1.0], 10.0)
    with pytest.raises(ValueError):
        cost_death_test([0.01, 0.02], [0.0, 1.0], -5.0)


def test_cost_death_inconclusive_on_tiny_sample() -> None:
    out = cost_death_test(
        [0.01, 0.02, -0.01, 0.0, 0.01],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        10.0,
        seed=0,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert math.isnan(out["ci_low"])
    assert out["n_obs"] == 5


def test_cost_death_inconclusive_on_degenerate_series() -> None:
    # never trades -> constant net series -> zero std -> no verdict
    out = cost_death_test([0.001] * 20, [0.0] * 20, 10.0, seed=0)
    assert out["verdict"] == "INCONCLUSIVE"
    assert math.isnan(out["net_sharpe"])
    assert out["n_changes"] == 0
