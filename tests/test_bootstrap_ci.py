"""Tests for atropos.bootstrap_ci percentile and block bootstraps.

Known-answer checks on constant and rotated series, seed determinism,
mean-block sensitivity, the median statistic, and the small-sample
insufficient-data guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos.bootstrap_ci import (
    block_bootstrap_ci,
    percentile_ci,
    stationary_bootstrap_ci,
)


def test_percentile_ci_on_constant_grid_and_empty() -> None:
    assert percentile_ci(np.full(100, 3.5)) == (3.5, 3.5)
    ci_low, ci_high = percentile_ci(np.arange(1000) / 1000.0)
    assert ci_low == pytest.approx(0.025, abs=1e-3)
    assert ci_high == pytest.approx(0.975, abs=1e-3)
    empty_low, empty_high = percentile_ci([])
    assert math.isnan(empty_low)
    assert math.isnan(empty_high)


def test_block_bootstrap_ci_constant_series_collapses() -> None:
    out = block_bootstrap_ci(np.full(120, 7.0), n_boot=100, block=15, seed=1)
    assert out["point_estimate"] == pytest.approx(7.0)
    assert out["ci_low"] == pytest.approx(7.0)
    assert out["ci_high"] == pytest.approx(7.0)
    assert out["se"] == pytest.approx(0.0)
    assert out["method"] == "moving_block"
    assert out["B"] == 100
    assert out["block"] == 15
    assert out["n"] == 120


def test_block_bootstrap_ci_reproducible_with_fixed_seed() -> None:
    data = np.random.default_rng(42).standard_normal(200)
    first = block_bootstrap_ci(data, n_boot=200, block=20, seed=1)
    repeat = block_bootstrap_ci(data, n_boot=200, block=20, seed=1)
    other_seed = block_bootstrap_ci(data, n_boot=200, block=20, seed=2)
    assert first["ci_low"] == repeat["ci_low"]
    assert first["ci_high"] == repeat["ci_high"]
    assert first["se"] == repeat["se"]
    assert first["ci_low"] != other_seed["ci_low"]


def test_block_bootstrap_ci_median_stat_and_block_clamp() -> None:
    # block >= n wraps the whole series into one circular block, so
    # every replicate is a rotation and the median CI collapses exactly
    out = block_bootstrap_ci(np.arange(100.0), n_boot=100, block=500, seed=0)
    assert out["block"] == 100
    assert out["point_estimate"] == pytest.approx(49.5)
    assert out["ci_low"] == pytest.approx(49.5)
    assert out["ci_high"] == pytest.approx(49.5)


def test_block_bootstrap_ci_insufficient_data_guard() -> None:
    out = block_bootstrap_ci([1.0, 2.0, float("nan")], n_boot=100, seed=3)
    assert out["method"] == "insufficient_data"
    assert out["n"] == 2
    assert out["B"] == 0
    assert out["point_estimate"] == pytest.approx(1.5)
    assert math.isnan(out["ci_low"])
    assert math.isnan(out["ci_high"])


def test_stationary_bootstrap_ci_constant_series_and_guard() -> None:
    out = stationary_bootstrap_ci(
        np.full(100, 5.0), n_boot=100, mean_block=10.0, seed=2
    )
    assert out["point_estimate"] == pytest.approx(5.0)
    assert out["ci_low"] == pytest.approx(5.0)
    assert out["ci_high"] == pytest.approx(5.0)
    assert out["method"] == "stationary"
    assert out["mean_block"] == pytest.approx(10.0)
    assert out["B"] == 100
    tiny = stationary_bootstrap_ci([1.0, 2.0], n_boot=50, seed=0)
    assert tiny["method"] == "insufficient_data"
    assert tiny["point_estimate"] == pytest.approx(1.5)


def test_stationary_bootstrap_ci_seed_and_mean_block_effects() -> None:
    data = np.random.default_rng(21).standard_normal(200)
    first = stationary_bootstrap_ci(data, n_boot=200, mean_block=5.0, seed=1)
    repeat = stationary_bootstrap_ci(data, n_boot=200, mean_block=5.0, seed=1)
    assert first["ci_low"] == repeat["ci_low"]
    assert first["ci_high"] == repeat["ci_high"]
    longer_blocks = stationary_bootstrap_ci(
        data, n_boot=200, mean_block=50.0, seed=1
    )
    assert longer_blocks["ci_low"] != first["ci_low"]


def test_stationary_bootstrap_ci_median_stat() -> None:
    data = np.random.default_rng(8).standard_normal(150)
    out = stationary_bootstrap_ci(
        data, n_boot=100, mean_block=10.0, stat=np.median, seed=4
    )
    assert out["point_estimate"] == pytest.approx(float(np.median(data)))
    assert out["n"] == 150
    assert out["B"] == 100
    assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]
