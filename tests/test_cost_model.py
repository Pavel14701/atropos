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
    run_all_examples,
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
    assert out["p_value"] is None


def test_cost_death_fails_at_500_bps() -> None:
    strategy, positions = _build_strategy(3)
    out = cost_death_test(
        strategy, positions, 500.0, n_boot=300, block=20, seed=5
    )
    assert out["verdict"] == "FAIL"
    assert out["cost_drag_mean"] > 0.0
    assert out["net_mean"] < 0.0
    assert out["net_sharpe"] < 0.0


def test_cost_death_deterministic_with_fixed_seed() -> None:
    strategy, positions = _build_strategy(11)
    first = cost_death_test(strategy, positions, 25.0, seed=1)
    repeat = cost_death_test(strategy, positions, 25.0, seed=1)
    # some metrics are documented NaN placeholders: NaN != NaN, so
    # compare key by key with NaN treated as equal to itself
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


def test_clean_pair_is_jointly_aligned_regression() -> None:
    # Regression: NaNs used to be dropped from returns and positions
    # INDEPENDENTLY, so returns[t] could be charged against
    # positions[s] of a different period. The joint finite mask must
    # keep only t=0 and t=3, i.e. the pairs (0.01, 1.0) and (0.03, 0.0).
    returns = [0.01, float("nan"), 0.02, 0.03]
    positions = [1.0, 2.0, float("nan"), 0.0]
    # kept positions [1.0, 0.0] -> turnover [0.0, 1.0]; at 100 bps the
    # single traded unit costs exactly 0.01
    net = apply_proportional_costs(returns, positions, 100.0)
    assert net == pytest.approx([0.01, 0.03 - 0.01])
    # the same misaligned inputs through the full contract test: only
    # two aligned observations survive, so the bootstrap must refuse
    out = cost_death_test(returns, positions, 100.0, seed=0)
    assert out["n_obs"] == 2
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["n_changes"] == 1


def test_breakeven_dust_turnover_is_inf_not_astronomic() -> None:
    # Regression: mean turnover of ~1e-15 from re-derived weights is
    # numerical dust, not trading; breakeven must be inf instead of a
    # huge finite number like 1e13.
    gross = [0.001, 0.001, 0.001, 0.001]
    positions = [0.0, 4e-15, 0.0, 0.0]  # mean turnover = 1e-15
    dust = breakeven_cost_bps(gross, positions)
    assert math.isinf(dust["breakeven_cost_bps"])
    # genuinely small-but-real turnover still divides normally:
    # tvr = [0, 4e-4, 4e-4, 0] -> mean 2e-4 -> 1e4 * 0.001 / 2e-4
    real = breakeven_cost_bps(gross, [0.0, 4e-4, 0.0, 0.0])
    assert real["breakeven_cost_bps"] == pytest.approx(1e4 * 0.001 / 2e-4)


def test_cost_model_validates_parameters_before_data() -> None:
    # parameter validation runs before the data is touched: even a
    # non-array input must produce ValueError, not a coercion error
    garbage = object()
    with pytest.raises(ValueError):
        cost_death_test([0.01] * 10, [1.0] * 10, 10.0, periods=0)
    with pytest.raises(ValueError):
        cost_death_test([0.01] * 10, [1.0] * 10, 10.0, ci_level=1.0)
    with pytest.raises(ValueError):
        cost_death_test(garbage, garbage, 10.0, ci_level=0.0)
    with pytest.raises(ValueError):
        annual_turnover([0.0, 1.0], periods=0)


def test_cost_death_validates_own_parameters_upfront() -> None:
    """n_boot / block / min_net_sharpe are cost_death_test's own checks.

    Regression: ``n_boot=0`` used to surface only deep inside
    ``block_bootstrap_ci`` (after the data was cleaned); the error must
    name the parameter and fire before the data is touched. A negative
    ``min_net_sharpe`` is a floor that any Sharpe clears - forbidden in
    a kill framework.
    """
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        cost_death_test([0.01] * 10, [1.0] * 10, 10.0, n_boot=0)
    with pytest.raises(ValueError, match="block must be >= 1"):
        cost_death_test([0.01] * 10, [1.0] * 10, 10.0, block=0)
    with pytest.raises(ValueError, match="min_net_sharpe must be >= 0"):
        cost_death_test([0.01] * 10, [1.0] * 10, 10.0, min_net_sharpe=-0.5)
    # raised before the data is touched: garbage data still gets the
    # parameter error, not a coercion error
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        cost_death_test(object(), object(), 10.0, n_boot=0)
    # the boundary values themselves are legal
    okay = cost_death_test(
        [0.01] * 10, [1.0] * 10, 10.0, n_boot=1, block=1, min_net_sharpe=0.0
    )
    assert okay["verdict"] == "INCONCLUSIVE"  # 10 obs, single replicate


def test_integer_parameters_reject_fractional_values() -> None:
    """Non-integer counts raise ValueError, not silent truncation.

    Regression class: ``int(periods) >= 1`` guards used to truncate
    ``2.5`` to ``2`` silently.
    """
    with pytest.raises(ValueError, match="periods must be an integer"):
        annual_turnover(
            [0.0, 1.0],
            periods=2.5,  # type: ignore[arg-type]
        )
    garbage = object()  # the checks fire before the data is touched
    with pytest.raises(ValueError, match="periods must be an integer"):
        cost_death_test(
            garbage,
            garbage,
            10.0,
            periods=2.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="n_boot must be an integer"):
        cost_death_test(
            garbage,
            garbage,
            10.0,
            n_boot=2.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="block must be an integer"):
        cost_death_test(
            garbage,
            garbage,
            10.0,
            block=2.5,  # type: ignore[arg-type]
        )


def test_clean_pair_rejects_non_1d_inputs() -> None:
    """Inputs >1-D raise 'must be 1-D'; (n, 1) columns are rejected too.

    Regression: reshape(-1) used to flatten a 2-D ``(n, 1)`` column
    silently (and string wider arrays into bogus longer series).
    """
    column = np.zeros((4, 1))
    wide = np.zeros((2, 2))
    with pytest.raises(ValueError, match="strategy_returns must be 1-D"):
        apply_proportional_costs(column, np.zeros(4), 10.0)
    with pytest.raises(ValueError, match="positions must be 1-D"):
        apply_proportional_costs(np.zeros(4), column, 10.0)
    with pytest.raises(ValueError, match="strategy_returns must be 1-D"):
        breakeven_cost_bps(wide, np.zeros(4))
    with pytest.raises(ValueError, match="positions must be 1-D"):
        breakeven_cost_bps(np.zeros(4), wide)
    # 1-D lists and 0-d scalars keep working (a scalar is one period)
    assert apply_proportional_costs([0.01], [1.0], 10.0) == pytest.approx(
        [0.01]
    )
    assert apply_proportional_costs(0.01, 1.0, 10.0) == pytest.approx([0.01])


def test_demo_run_all_examples_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module demo runs clean and prints both COST-DEATH verdicts."""
    run_all_examples()
    out = capsys.readouterr().out
    assert "cost_model demo" in out
    assert "annual_turnover   :" in out
    assert "breakeven_cost_bps:" in out
    assert "cost_death_test( 10.0 bps): verdict=PASS" in out
    assert "cost_death_test(200.0 bps): verdict=FAIL" in out
