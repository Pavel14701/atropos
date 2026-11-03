"""Tests for atropos.monte_carlo: GBM paths, bootstrap, risk tails, MDE."""

from collections.abc import Callable

import numpy as np
import pytest
from atropos.monte_carlo import (
    bootstrap_resample,
    max_drawdown_distribution,
    minimum_detectable_effect,
    monte_carlo_paths,
    sharpe_distribution,
)
from atropos.power_calculator import detectable_effect


def test_monte_carlo_paths_shape_positivity_determinism() -> None:
    paths_first = monte_carlo_paths(
        100.0, 0.05, 0.20, n_paths=300, n_steps=50, seed=11
    )
    paths_repeat = monte_carlo_paths(
        100.0, 0.05, 0.20, n_paths=300, n_steps=50, seed=11
    )
    assert paths_first.shape == (300, 50)
    assert bool(np.all(paths_first > 0.0))
    assert np.array_equal(paths_first, paths_repeat)
    paths_other_seed = monte_carlo_paths(
        100.0, 0.05, 0.20, n_paths=300, n_steps=50, seed=12
    )
    assert not np.array_equal(paths_first, paths_other_seed)
    # dt is keyword-only and accepted alongside seed
    paths_daily = monte_carlo_paths(
        100.0, 0.05, 0.20, n_paths=10, n_steps=20, dt=1.0 / 252.0, seed=13
    )
    assert paths_daily.shape == (10, 20)


def test_monte_carlo_paths_zero_volatility_is_flat() -> None:
    flat = monte_carlo_paths(50.0, 0.0, 0.0, n_paths=6, n_steps=4, seed=1)
    assert flat.shape == (6, 4)
    # s0 is NOT a column: with mu=sigma=0 every step keeps S == s0 exactly
    assert bool(np.all(flat == 50.0))


def test_monte_carlo_paths_positive_drift_raises_terminal_price() -> None:
    deterministic = monte_carlo_paths(
        100.0, 0.50, 0.0, n_paths=64, n_steps=252, seed=2
    )
    assert bool(np.all(deterministic == deterministic[0]))
    assert float(deterministic[0, -1]) == pytest.approx(
        100.0 * np.exp(0.50), rel=1e-12
    )
    tiny_vol = monte_carlo_paths(
        100.0, 0.50, 0.01, n_paths=400, n_steps=252, seed=3
    )
    assert float(np.mean(tiny_vol[:, -1])) > 100.0


def test_monte_carlo_paths_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="s0 must be > 0"):
        monte_carlo_paths(-1.0, 0.1, 0.1, n_paths=2, n_steps=3)
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        monte_carlo_paths(1.0, 0.1, -0.1, n_paths=2, n_steps=3)
    with pytest.raises(ValueError, match="n_paths and n_steps must be >= 1"):
        monte_carlo_paths(1.0, 0.1, 0.1, n_paths=0, n_steps=3)


def test_monte_carlo_paths_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be > 0"):
        monte_carlo_paths(100.0, 0.05, 0.20, n_paths=2, n_steps=3, dt=0.0)
    with pytest.raises(ValueError, match="dt must be > 0"):
        monte_carlo_paths(100.0, 0.05, 0.20, n_paths=2, n_steps=3, dt=-0.1)


def test_bootstrap_of_constant_series_reproduces_constant() -> None:
    draws = bootstrap_resample(np.full(100, 3.5), np.mean, n_boot=200, seed=3)
    assert draws.shape == (200,)
    assert bool(np.all(draws == 3.5))


def test_bootstrap_drops_nan_and_fills_all_nan_input() -> None:
    gappy = np.arange(60, dtype=float)
    gappy[::7] = np.nan
    finite_mean = float(np.mean(gappy[np.isfinite(gappy)]))
    draws = bootstrap_resample(gappy, np.mean, n_boot=100, seed=4)
    assert draws.shape == (100,)
    assert bool(np.all(np.isfinite(draws)))
    assert abs(float(np.mean(draws)) - finite_mean) < 1.0
    all_nan = bootstrap_resample(
        np.full(20, np.nan), np.mean, n_boot=30, seed=5
    )
    assert bool(np.all(np.isnan(all_nan)))


def test_bootstrap_iid_and_block_modes_reproducible() -> None:
    sample = np.arange(100, dtype=float)
    iid_first = bootstrap_resample(sample, np.mean, n_boot=50, seed=5)
    iid_repeat = bootstrap_resample(sample, np.mean, n_boot=50, seed=5)
    assert np.array_equal(iid_first, iid_repeat)
    block_first = bootstrap_resample(
        sample, np.mean, n_boot=50, block=10, seed=5
    )
    block_repeat = bootstrap_resample(
        sample, np.mean, n_boot=50, block=10, seed=5
    )
    assert np.array_equal(block_first, block_repeat)
    assert not np.array_equal(iid_first, block_first)


def test_bootstrap_rejects_zero_replicates() -> None:
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        bootstrap_resample([1.0, 2.0, 3.0], np.mean, n_boot=0)


def test_bootstrap_rejects_non_callable_statistic() -> None:
    with pytest.raises(TypeError, match="statistic must be callable"):
        bootstrap_resample(
            [1.0, 2.0, 3.0],
            "not callable",  # type: ignore[arg-type]
        )


def test_max_drawdown_all_positive_returns_zero_drawdown() -> None:
    result = max_drawdown_distribution(
        np.full(200, 0.01), n_sim=200, block=20, seed=6
    )
    for key in ("mdd_p5", "mdd_p50", "mdd_p95", "n_valid_paths"):
        assert key in result
    assert result["mdd_p5"] == 0.0
    assert result["mdd_p50"] == 0.0
    assert result["mdd_p95"] == 0.0
    assert result["n_dropped_paths"] == 0
    assert result["n_valid_paths"] == 200
    assert result["n"] == 200
    assert result["block"] == 20


def test_max_drawdown_persistent_loss_is_total_compounded_loss() -> None:
    # constant -1% returns: every resample is the same monotonically
    # falling equity, so all percentile drawdowns equal the full loss.
    # Equity starts at the first compounded value 0.99, so the total
    # peak-to-trough drop is 0.99^199 - 1 (199 further steps).
    result = max_drawdown_distribution(
        np.full(200, -0.01), n_sim=200, block=20, seed=7
    )
    expected_mdd = 0.99**199 - 1.0
    assert result["mdd_p5"] == pytest.approx(expected_mdd, rel=1e-6)
    assert result["mdd_p95"] == pytest.approx(expected_mdd, rel=1e-6)
    assert result["mdd_p5"] <= result["mdd_p50"] <= result["mdd_p95"]


def test_max_drawdown_requires_twenty_returns() -> None:
    with pytest.raises(ValueError, match="at least 20 valid returns"):
        max_drawdown_distribution(np.full(10, 0.01), n_sim=50)


def test_max_drawdown_drops_paths_whose_equity_goes_nonpositive() -> None:
    # Regression: a resampled return <= -1 used to survive the "finite
    # drawdown" filter and could push mdd_p5 below -100%.  Such paths
    # must be dropped and counted instead.
    generator = np.random.default_rng(21)
    returns = 0.001 + 0.005 * generator.standard_normal(200)
    returns[40] = -1.5  # catastrophic day: any resampled path that draws
    # this observation compounds to non-positive equity
    result = max_drawdown_distribution(returns, n_sim=500, block=10, seed=22)
    assert result["n_dropped_paths"] >= 1
    assert (
        result["n_dropped_paths"] + result["n_valid_paths"] == result["n_sim"]
    )
    # surviving paths never hit zero equity: drawdown stays above -100%
    assert result["mdd_p5"] >= -1.0
    assert bool(np.isfinite(result["mdd_p5"]))


def test_max_drawdown_flags_insufficient_paths_for_percentiles() -> None:
    generator = np.random.default_rng(23)
    returns = generator.normal(0.001, 0.01, 100)
    result = max_drawdown_distribution(returns, n_sim=15, block=5, seed=24)
    assert result["n_valid_paths"] < 20
    assert result["note"].endswith("insufficient paths for percentiles")
    assert bool(np.isnan(result["mdd_p5"]))
    assert bool(np.isnan(result["mdd_p50"]))
    assert bool(np.isnan(result["mdd_p95"]))


def test_sharpe_distribution_keys_ordered_and_reproducible() -> None:
    generator = np.random.default_rng(8)
    returns = generator.normal(0.0008, 0.01, 300)
    first = sharpe_distribution(returns, n_sim=200, block=20, seed=9)
    repeat = sharpe_distribution(returns, n_sim=200, block=20, seed=9)
    for key in ("sr_p5", "sr_p50", "sr_p95", "n_valid_paths", "periods"):
        assert key in first
    assert first["sr_p5"] <= first["sr_p50"] <= first["sr_p95"]
    assert first["n_valid_paths"] == 200
    assert first["periods"] == 252
    assert first == repeat


def test_sharpe_distribution_counts_dropped_paths() -> None:
    generator = np.random.default_rng(25)
    returns = generator.normal(0.0008, 0.01, 300)
    result = sharpe_distribution(returns, n_sim=200, block=20, seed=26)
    # well-behaved input: nothing dropped, no warning note attached
    assert result["n_dropped_paths"] == 0
    assert result["n_dropped_paths"] + result["n_valid_paths"] == 200
    assert "note" not in result


def test_sharpe_drops_zero_variance_paths() -> None:
    # every resample of a constant series is constant: sd = 0, so the
    # Sharpe is undefined and each replicate must be dropped and counted
    # (0.5 is exact in binary, std is exactly 0.0, no rounding residue)
    constant = np.full(100, 0.5)
    result = sharpe_distribution(constant, n_sim=50, block=10, seed=27)
    assert result["n_dropped_paths"] == 50
    assert result["n_valid_paths"] == 0
    assert bool(np.isnan(result["sr_p50"]))
    assert result["note"] == "insufficient paths for percentiles"


def test_sharpe_flags_insufficient_paths_and_validates_periods() -> None:
    generator = np.random.default_rng(28)
    returns = generator.normal(0.0008, 0.01, 100)
    short = sharpe_distribution(returns, n_sim=12, block=5, seed=29)
    assert short["note"] == "insufficient paths for percentiles"
    assert bool(np.isnan(short["sr_p5"]))
    assert bool(np.isnan(short["sr_p50"]))
    assert bool(np.isnan(short["sr_p95"]))
    with pytest.raises(ValueError, match="periods must be >= 1"):
        sharpe_distribution(returns, n_sim=50, block=5, periods=0, seed=30)
    with pytest.raises(ValueError, match="periods must be >= 1"):
        sharpe_distribution(returns, n_sim=50, block=5, periods=-2, seed=31)


def test_minimum_detectable_effect_shrinks_with_sample() -> None:
    mde_500 = minimum_detectable_effect(500, n_predictors=5)
    mde_2000 = minimum_detectable_effect(2000, n_predictors=5)
    mde_5000 = minimum_detectable_effect(5000, n_predictors=5)
    assert mde_500 > mde_2000 > mde_5000
    assert all(value > 0.0 for value in (mde_500, mde_2000, mde_5000))


def test_minimum_detectable_effect_delegates_to_power_calculator() -> None:
    delegated = minimum_detectable_effect(
        800, power=0.90, alpha=0.01, n_predictors=3
    )
    direct = detectable_effect(
        n_obs=800, n_predictors=3, alpha=0.01, power=0.90
    )
    assert delegated == pytest.approx(direct)


def test_minimum_detectable_effect_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError, match="n must be >= 2"):
        minimum_detectable_effect(1)


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (
            lambda: monte_carlo_paths(
                1.0,
                0.1,
                0.1,
                2.5,  # type: ignore[arg-type]
                3,
            ),
            "n_paths",
        ),
        (
            lambda: monte_carlo_paths(
                1.0,
                0.1,
                0.1,
                2,
                2.5,  # type: ignore[arg-type]
            ),
            "n_steps",
        ),
        (
            lambda: bootstrap_resample(
                [1.0, 2.0],
                np.mean,
                n_boot=2.5,  # type: ignore[arg-type]
            ),
            "n_boot",
        ),
        (
            lambda: bootstrap_resample(
                [1.0, 2.0],
                np.mean,
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
        (
            lambda: max_drawdown_distribution(
                np.full(30, 0.01),
                n_sim=2.5,  # type: ignore[arg-type]
            ),
            "n_sim",
        ),
        (
            lambda: max_drawdown_distribution(
                np.full(30, 0.01),
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
        (
            lambda: sharpe_distribution(
                np.full(30, 0.01),
                n_sim=2.5,  # type: ignore[arg-type]
            ),
            "n_sim",
        ),
        (
            lambda: sharpe_distribution(
                np.full(30, 0.01),
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
        (
            lambda: sharpe_distribution(
                np.full(30, 0.01),
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: minimum_detectable_effect(
                2.5  # type: ignore[arg-type]
            ),
            "n",
        ),
        (
            lambda: minimum_detectable_effect(
                2000,
                n_predictors=2.5,  # type: ignore[arg-type]
            ),
            "n_predictors",
        ),
    ],
)
def test_integer_parameters_reject_fractional_values(
    call: Callable[[], object], name: str
) -> None:
    """Non-integer counts raise ValueError, not silent truncation.

    Regression class: ``int(value) >= 1`` guards used to truncate
    ``2.5`` silently, and an unvalidated ``block=2.5`` was clamped by
    ``block_indices`` without any error at all.
    """
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        call()
