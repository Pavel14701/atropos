"""Tests for atropos.bootstrap_ci percentile and block bootstraps.

Known-answer checks on constant and rotated series, seed determinism,
mean-block sensitivity, the median statistic, and the small-sample
insufficient-data guard.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from atropos.bootstrap_ci import (
    block_bootstrap_ci,
    percentile_ci,
    run_all_examples,
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


def test_percentile_ci_validates_ci_level() -> None:
    for bad_level in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            percentile_ci(np.arange(100.0), ci_level=bad_level)


def test_bootstrap_inputs_validated_before_data() -> None:
    # validation runs before the data is touched: a non-array input
    # must produce the ValueError of the parameter check, not a
    # coercion error from numpy
    garbage = object()
    with pytest.raises(ValueError):
        block_bootstrap_ci(np.arange(50.0), ci_level=0.0)
    with pytest.raises(ValueError):
        block_bootstrap_ci(garbage, n_boot=0)
    with pytest.raises(ValueError):
        block_bootstrap_ci(garbage, block=0)
    with pytest.raises(ValueError):
        stationary_bootstrap_ci(garbage, n_boot=0)
    with pytest.raises(ValueError):
        stationary_bootstrap_ci(np.arange(50.0), ci_level=1.0)


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (
            lambda: block_bootstrap_ci(
                np.arange(50.0),
                n_boot=2.5,  # type: ignore[arg-type]
            ),
            "n_boot",
        ),
        (
            lambda: block_bootstrap_ci(
                np.arange(50.0),
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
        (
            lambda: stationary_bootstrap_ci(
                np.arange(50.0),
                n_boot=2.5,  # type: ignore[arg-type]
            ),
            "n_boot",
        ),
    ],
)
def test_integer_parameters_reject_fractional_values(
    call: Callable[[], object], name: str
) -> None:
    """Non-integer counts / block lengths raise ValueError.

    Regression class: the ``int(n_boot) >= 1`` guards used to truncate
    ``n_boot=2.5`` to ``2`` silently.
    """
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        call()


def test_numeric_parameter_object_raises_value_error() -> None:
    """``ci_level=object()`` raises ValueError naming the parameter.

    The docstrings promise ``Raises: ValueError`` before the data is
    touched; the old bare ``0.0 < float(ci_level)`` comparison escaped
    as ``TypeError`` instead.
    """
    for call in (
        lambda: percentile_ci(
            np.arange(50.0),
            ci_level=object(),  # type: ignore[arg-type]
        ),
        lambda: block_bootstrap_ci(
            np.arange(50.0),
            ci_level=object(),  # type: ignore[arg-type]
        ),
        lambda: stationary_bootstrap_ci(
            np.arange(50.0),
            ci_level=object(),  # type: ignore[arg-type]
        ),
    ):
        with pytest.raises(
            ValueError, match="ci_level must be a finite number"
        ):
            call()


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
    assert out["method"] == "moving_block"
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


def _demo_ar1_series() -> np.ndarray:
    """AR(1) series of bootstrap_ci demo 1 (rng seed 7, phi=0.6)."""
    rng = np.random.default_rng(7)
    shocks = rng.standard_normal(500)
    ar = np.zeros(500)
    for step in range(1, 500):
        ar[step] = 0.6 * ar[step - 1] + shocks[step]
    return ar


def test_seeded_regression_moving_block_demo_values() -> None:
    """Exact demo-1 numbers at its fixed data and seed.

    The demo itself only asserts structural facts and input-independent
    properties; the seeded point/CI/se values live here so any RNG or
    algorithm drift breaks loudly.
    """
    ar = _demo_ar1_series()
    ci_mb = block_bootstrap_ci(
        ar, n_boot=1000, block=20, stat=np.mean, seed=1
    )
    assert ci_mb["point_estimate"] == pytest.approx(
        -0.31767150717896936, abs=1e-12
    )
    assert ci_mb["ci_low"] == pytest.approx(-0.507772108612692, abs=1e-12)
    assert ci_mb["ci_high"] == pytest.approx(-0.11718517776506407, abs=1e-12)
    assert ci_mb["se"] == pytest.approx(0.09829426050787746, abs=1e-12)
    # the seeded containment the demo asserts as a fixed-seed fact
    assert ci_mb["ci_low"] < ci_mb["point_estimate"] < ci_mb["ci_high"]


def test_seeded_regression_stationary_demo_values() -> None:
    """Exact demo-2 numbers at its fixed data and seed.

    The heavy-tailed sample depends on the rng state left by demo 1,
    so the standard_normal draw is replayed in the same order.
    """
    rng = np.random.default_rng(7)
    rng.standard_normal(500)  # demo 1 consumed these shocks first
    heavy = rng.standard_t(3, size=400) * 1.5
    ci_st = stationary_bootstrap_ci(
        heavy, n_boot=300, mean_block=12, stat=np.median, seed=2
    )
    assert ci_st["point_estimate"] == pytest.approx(
        -0.07091350405825854, abs=1e-12
    )
    assert ci_st["ci_low"] == pytest.approx(-0.2656238892732639, abs=1e-12)
    assert ci_st["ci_high"] == pytest.approx(0.0955563306173082, abs=1e-12)
    assert ci_st["se"] == pytest.approx(0.07774504473919959, abs=1e-12)
    assert ci_st["ci_low"] < ci_st["point_estimate"] < ci_st["ci_high"]


def test_demo_run_all_examples_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module demo runs clean and prints all three demo blocks."""
    run_all_examples()
    out = capsys.readouterr().out
    assert "bootstrap_ci demo 1" in out
    assert "bootstrap_ci demo 2" in out
    assert "bootstrap_ci demo 3" in out
    assert "method=moving_block" in out
    assert "method=stationary" in out
    assert "method=insufficient_data" in out
    assert "ci_low=nan" in out
