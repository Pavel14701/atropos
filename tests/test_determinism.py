"""Determinism contracts: fixed seed -> bit-identical results.

Project law 3 (reproducibility): a fixed seed, a frozen snapshot and
pinned dependencies must reproduce every published number.  These tests
pin the three layers the law talks about:

* two calls with identical inputs and seeds produce identical results
  (scalars, dicts, DataFrames and nested pipeline payloads alike);
* seeded functions must draw from their own ``default_rng`` and never
  disturb the legacy global ``np.random`` state;
* the shared cleaner ``_utils.clean_1d`` is idempotent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from atropos._utils import clean_1d
from atropos.bootstrap_ci import block_bootstrap_ci
from atropos.cost_model import apply_proportional_costs
from atropos.ensemble_pipeline import run_full_pipeline
from atropos.evaluation_metrics import sharpe_ratio
from atropos.monte_carlo import max_drawdown_distribution
from atropos.robustness_battery import run_battery
from atropos.walkforward_framework import walkforward

_REDUCED_PIPELINE_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}


def _equivalent(left: Any, right: Any) -> bool:
    """Deep equality that understands DataFrames and ndarrays."""
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        return pd.DataFrame(left).equals(pd.DataFrame(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equivalent(item, other)
            for item, other in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


# ---------------------------------------------------------------------------
# bitwise reproducibility of the randomized public functions
# ---------------------------------------------------------------------------
def test_block_bootstrap_ci_is_bitwise_reproducible() -> None:
    """Same seed -> identical CI dict, float for float."""
    rng = np.random.default_rng(0)
    series = 0.001 + 0.01 * rng.standard_normal(300)
    first = block_bootstrap_ci(series, n_boot=100, block=10, seed=7)
    second = block_bootstrap_ci(series, n_boot=100, block=10, seed=7)
    assert first == second


def test_max_drawdown_distribution_is_bitwise_reproducible() -> None:
    """Same seed -> identical percentile envelope."""
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    first = max_drawdown_distribution(returns, n_sim=100, block=10, seed=7)
    second = max_drawdown_distribution(returns, n_sim=100, block=10, seed=7)
    assert first == second


def test_walkforward_is_bitwise_reproducible() -> None:
    """Same data -> identical t-stats, verdicts and fold table."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(400)
    frame = pd.DataFrame(
        {"signal": signal, "outcome": 0.4 * signal + rng.standard_normal(400)}
    )
    first = walkforward(frame, train_window=200, test_window=50, step=50)
    second = walkforward(frame, train_window=200, test_window=50, step=50)
    assert first.keys() == second.keys()
    for key in first:
        assert _equivalent(first[key], second[key]), key


def test_run_battery_is_bitwise_reproducible() -> None:
    """Same seed -> identical battery table, row for row."""
    rng = np.random.default_rng(11)
    n_days = 400
    signal = np.zeros(n_days)
    for day in range(1, n_days):
        signal[day] = 0.5 * signal[day - 1] + rng.standard_normal()
    signal = signal / float(np.std(signal))
    outcome = 0.8 * signal + rng.standard_normal(n_days)
    params: dict[str, Any] = {
        "threshold": 0.0,
        "window": 10,
        "n_boot": 100,
        "block": 10,
        "seed": 0,
    }
    first = run_battery(signal, outcome, params)
    second = run_battery(signal, outcome, params)
    assert first.equals(second)


def test_run_full_pipeline_is_bitwise_reproducible() -> None:
    """Same inputs + reduced config -> identical payload, stage by stage."""
    rng = np.random.default_rng(4)
    n_obs = 500
    driver = np.zeros(n_obs)
    driver_shocks = rng.standard_normal(n_obs)
    for step in range(1, n_obs):
        driver[step] = 0.5 * driver[step - 1] + driver_shocks[step]
    index = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    data = {
        "returns": pd.Series(
            driver + 0.5 * rng.standard_normal(n_obs), index=index
        ),
        "signal": pd.Series(driver, index=index),
    }
    first = run_full_pipeline(data, _REDUCED_PIPELINE_CONFIG)
    second = run_full_pipeline(data, _REDUCED_PIPELINE_CONFIG)
    assert first.keys() == second.keys()
    for key in first:
        if key == "runtime_seconds":
            continue  # wall-clock jitter is not part of the contract
        assert _equivalent(first[key], second[key]), key


# ---------------------------------------------------------------------------
# no leakage into the legacy global RNG
# ---------------------------------------------------------------------------
def test_seeded_functions_leave_global_rng_untouched() -> None:
    """default_rng users must not consume or reseed np.random state."""
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    gross = 0.001 + 0.01 * rng.standard_normal(200)
    positions = (gross > 0.0).astype(float)
    signal = np.zeros(300)
    shocks = rng.standard_normal(300)
    for day in range(1, 300):
        signal[day] = 0.5 * signal[day - 1] + shocks[day]

    np.random.seed(42)
    before = repr(np.random.get_state())
    sharpe_ratio(returns, seed=0)
    block_bootstrap_ci(returns, n_boot=50, block=10, seed=1)
    max_drawdown_distribution(returns, n_sim=50, block=10, seed=3)
    apply_proportional_costs(gross, positions, 10.0)
    run_battery(
        signal,
        returns,
        {
            "threshold": 0.0,
            "window": 10,
            "n_boot": 50,
            "block": 10,
            "seed": 0,
        },
    )
    after = repr(np.random.get_state())
    assert before == after


def test_global_rng_stream_is_usable_across_calls() -> None:
    """The legacy stream yields the same draws regardless of atropos."""
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)

    np.random.seed(123)
    expected = np.random.standard_normal(5).copy()

    np.random.seed(123)
    block_bootstrap_ci(returns, n_boot=50, block=10, seed=9)
    actual = np.random.standard_normal(5)

    assert np.array_equal(expected, actual)


# ---------------------------------------------------------------------------
# idempotence of the shared cleaner
# ---------------------------------------------------------------------------
def test_clean_1d_is_idempotent_on_dirty_input() -> None:
    """Cleaning a cleaned array returns the identical array."""
    dirty = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0])
    once = clean_1d(dirty)
    twice = clean_1d(once)
    assert np.array_equal(once, twice)
    assert np.array_equal(once, np.array([1.0, 3.0, 5.0]))


def test_clean_1d_is_idempotent_on_clean_input() -> None:
    """A already-clean random series is a fixed point of clean_1d."""
    rng = np.random.default_rng(5)
    values = rng.standard_normal(200)
    once = clean_1d(values)
    twice = clean_1d(once)
    assert np.array_equal(values, once)
    assert np.array_equal(once, twice)
