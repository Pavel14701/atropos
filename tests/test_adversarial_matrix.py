"""Adversarial edge-case matrices: every review finding pinned by name.

Each parametrize row is one edge case from the 14-module code review.
Unlike property tests, these guarantee the exact scenario is covered.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.bootstrap_ci import block_bootstrap_ci
from atropos.evaluation_metrics import sharpe_ratio
from atropos.prereg_generator import validate_prereg
from atropos.robustness_battery import _roll_mean, run_battery
from atropos.walkforward_framework import walkforward


def make_walkforward_frame(seed: int = 11) -> pd.DataFrame:
    """Deterministic regime-switching frame (signal leads outcome)."""
    rng = np.random.default_rng(seed)
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
    return pd.DataFrame({"signal": signal, "outcome": outcome})


# ---------------------------------------------------------------------
# robustness_battery._roll_mean: window > n_obs used to crash with a
# broadcast error (review: n_obs=60, window=41 -> derived 62 > 61)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("window", [1, 5, 20, 40, 41, 59, 60, 61, 100, 1000])
def test_roll_mean_preserves_length_for_any_window(window: int) -> None:
    """Output length == input length for any window, never a crash."""
    values = np.arange(60, dtype=float)
    smoothed = _roll_mean(values, window)
    assert smoothed.shape == values.shape
    assert np.all(np.isfinite(smoothed))


@pytest.mark.parametrize("window", [60, 61, 100])
def test_run_battery_rejects_window_at_or_above_n_obs(window: int) -> None:
    """A window covering the whole sample is rejected up front."""
    rng = np.random.default_rng(2)
    n_obs = 60
    signal = rng.standard_normal(n_obs)
    outcome = 0.02 * signal + rng.standard_normal(n_obs) * 0.05
    with pytest.raises(ValueError, match="window"):
        run_battery(signal, outcome, {"threshold": 0.0, "window": window})


# ---------------------------------------------------------------------
# walkforward: step < test_window used to inflate wf_t by up to
# sqrt(test_window/step); after the union fix the statistics are
# computed over each OOS row exactly once
# ---------------------------------------------------------------------
@pytest.mark.parametrize("step", [5, 10, 15, 30])
def test_wf_t_not_inflated_by_overlapping_steps(step: int) -> None:
    """Same union sample -> comparable wf_t regardless of step."""
    frame = make_walkforward_frame()
    result = walkforward(frame, train_window=120, test_window=30, step=step)
    baseline = walkforward(
        make_walkforward_frame(), train_window=120, test_window=30, step=30
    )
    if step <= 30:
        # every step <= test_window walks the same 240 unique rows
        assert result["n_pnl_unique"] == baseline["n_pnl_unique"] == 240
        assert abs(result["wf_t"] - baseline["wf_t"]) < 1.2
        assert result["overlap_factor"] == pytest.approx(30 / step)
    else:
        assert result["n_pnl_unique"] < baseline["n_pnl_unique"]


# ---------------------------------------------------------------------
# prereg_generator: soft kill criteria must be rejected
# ---------------------------------------------------------------------
def make_valid_prereg() -> dict[str, Any]:
    """Canonical valid prereg params (mirrors test_prereg_generator)."""
    return {
        "name": "edge_overnight",
        "hypothesis": "Overnight imbalance predicts next-day return.",
        "construction": ["signal = zscore(imbalance)", "enter at open"],
        "nulls": ["edge is repackaged 7d momentum"],
        "kill_criteria": [
            {
                "criterion": "|t| after controls",
                "comparison": "<",
                "threshold": 2.0,
            },
            "placebo p >= 0.05",
        ],
        "power_calc": {"N": 2600, "power": 0.8},
        "hash": "9f2c1ab4",
        "snapshot": "2026-11-03",
    }


@pytest.mark.parametrize(
    "criterion",
    [
        "wait 200 days before final check",
        "benchmark window of 63 days",
        "hold at least 5 positions",
        "edge decays over time",
    ],
)
def test_soft_kill_string_criteria_rejected(criterion: str) -> None:
    """A bare number (a duration, a count) is not a kill threshold."""
    params = make_valid_prereg()
    params["kill_criteria"] = [criterion]
    problems = validate_prereg(params)
    assert problems, f"soft kill criterion accepted: {criterion!r}"
    assert any("comparison" in problem for problem in problems)


@pytest.mark.parametrize(
    "criterion", ["IC < 0.05 on 200 days", "p >= 0.05", "t <= 2.0"]
)
def test_hard_kill_string_criteria_accepted(criterion: str) -> None:
    """Comparison operator bound to a number stays valid."""
    params = make_valid_prereg()
    params["kill_criteria"] = [criterion]
    assert validate_prereg(params) == []


def test_dict_criterion_without_comparison_rejected() -> None:
    """Dict form requires an explicit comparison (no '~' default)."""
    params = make_valid_prereg()
    params["kill_criteria"] = [{"criterion": "t-stat", "threshold": 2.0}]
    problems = validate_prereg(params)
    assert any("comparison" in problem for problem in problems)


# ---------------------------------------------------------------------
# empty inputs: NaN + insufficient_data, never a crash or a fake 0
# ---------------------------------------------------------------------
def test_sharpe_ratio_empty_input_is_insufficient() -> None:
    """Empty returns -> NaN stats labelled insufficient_data."""
    result = sharpe_ratio(np.array([], dtype=float), seed=0)
    assert result["ci_method"] == "insufficient_data"
    assert np.isnan(result["sr"])


def test_block_bootstrap_empty_input_is_insufficient() -> None:
    """Empty sample -> the insufficient-data scaffold, not an error."""
    result = block_bootstrap_ci(np.array([], dtype=float), seed=0)
    assert result["method"] == "insufficient_data"
    assert np.isnan(result["point_estimate"])
