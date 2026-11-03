"""Tests for atropos.walkforward_framework: honest out-of-sample t-stats.

A planted linear edge must walk-forward to PASS with the expected fold
count and no peeking, pure noise must never PASS, the calendar guard
must fire, and tampered folds must be caught by check_no_peeking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.walkforward_framework import (
    check_no_peeking,
    frozen_threshold,
    walkforward,
)


def _edge_frame(n_obs: int, seed: int, coef: float) -> pd.DataFrame:
    """DataFrame with a signal column and a noisy linear outcome."""
    gen = np.random.default_rng(seed)
    signal = gen.standard_normal(n_obs)
    return pd.DataFrame(
        {
            "signal": signal,
            "outcome": coef * signal + gen.standard_normal(n_obs),
        }
    )


def _fit_slope(train_df: pd.DataFrame) -> dict[str, float]:
    signal = train_df["sig"].to_numpy(dtype=float)
    outcome = train_df["fwd_ret"].to_numpy(dtype=float)
    return {
        "slope": float(np.mean(signal * outcome) / np.mean(signal * signal))
    }


def _predict_slope(
    model: dict[str, float], test_df: pd.DataFrame
) -> np.ndarray:
    return model["slope"] * test_df["sig"].to_numpy(dtype=float)


def test_walkforward_planted_edge_passes() -> None:
    data = _edge_frame(360, 85, 0.5)
    result = walkforward(data, train_window=180, test_window=60, step=60)
    assert result["verdict"] == "PASS"
    assert result["wf_t"] >= 2.0
    assert result["wf_direction_consistency"] >= 0.5
    assert result["n_folds"] == 3
    assert result["mode"] == "expanding"
    assert result["warnings"] == []
    assert check_no_peeking(result) is True
    folds = result["folds"]
    assert set(folds.columns) >= {
        "fold",
        "train_end",
        "test_start",
        "test_end",
        "n_train",
        "hit_rate",
        "fold_pnl_mean",
    }
    # expanding mode: every fold trains on all history before its test
    assert list(folds["n_train"]) == [180, 240, 300]
    assert (folds["train_end"] <= folds["test_start"]).all()
    assert np.isfinite(result["in_sample_t"])
    assert 0.0 < result["wf_shrink_ratio"] <= 2.0


def test_walkforward_pure_noise_does_not_pass() -> None:
    gen = np.random.default_rng(86)
    data = pd.DataFrame(
        {
            "signal": gen.standard_normal(360),
            "outcome": gen.standard_normal(360),
        }
    )
    result = walkforward(data, train_window=180, test_window=60, step=60)
    assert result["wf_t"] < 2.0
    assert result["verdict"] in {"FAIL", "DEGRADED"}


def test_walkforward_fold_count_and_step_warning() -> None:
    data = _edge_frame(350, 87, 0.4)
    result = walkforward(data, train_window=150, test_window=40, step=50)
    assert result["n_folds"] == 4
    assert result["n_folds"] == (350 - 150 - 40) // 50 + 1
    assert result["warnings"] == [
        "step > test_window: folds skip periods between tests"
    ]


def test_check_no_peeking_rejects_tampered_folds() -> None:
    data = _edge_frame(240, 88, 0.4)
    result = walkforward(data, train_window=120, test_window=40, step=40)
    assert check_no_peeking(result) is True
    tampered = dict(result)
    tampered_folds = result["folds"].copy()
    tampered_folds["train_end"] = tampered_folds["train_end"] + 1
    tampered["folds"] = tampered_folds
    assert check_no_peeking(tampered) is False
    assert check_no_peeking({}) is False
    assert check_no_peeking({"folds": pd.DataFrame()}) is False


def test_calendar_columns_are_forbidden_unless_disabled() -> None:
    data = _edge_frame(360, 89, 0.5)
    poisoned = data.copy()
    poisoned["year"] = np.arange(360) // 180
    with pytest.raises(ValueError, match="calendar terms forbidden"):
        walkforward(poisoned, train_window=180, test_window=60, step=60)
    relaxed = walkforward(
        poisoned,
        train_window=180,
        test_window=60,
        step=60,
        forbid_calendar=False,
    )
    assert relaxed["n_folds"] == 3
    assert relaxed["verdict"] == "PASS"


def test_walkforward_input_validation() -> None:
    gen = np.random.default_rng(90)
    signal = gen.standard_normal(200)
    data = pd.DataFrame({"signal": signal, "fwd_ret": signal * 0.1})
    with pytest.raises(TypeError, match="DataFrame"):
        walkforward(
            signal,  # type: ignore[arg-type]
            train_window=60,
            test_window=30,
            step=30,
        )
    with pytest.raises(ValueError, match="outcome_col"):
        walkforward(data, train_window=60, test_window=30, step=30)
    renamed = data.rename(columns={"fwd_ret": "outcome"})
    with pytest.raises(ValueError, match="train_window"):
        walkforward(renamed, train_window=5, test_window=30, step=30)
    with pytest.raises(ValueError, match="too small"):
        walkforward(renamed, train_window=180, test_window=30, step=30)


def test_walkforward_custom_fns_and_keyword_only_api() -> None:
    gen = np.random.default_rng(81)
    signal = gen.standard_normal(300)
    data = pd.DataFrame(
        {
            "sig": signal,
            "fwd_ret": 0.6 * signal + gen.standard_normal(300),
        }
    )
    result = walkforward(
        data,
        train_window=120,
        test_window=60,
        step=60,
        fit_fn=_fit_slope,
        predict_fn=_predict_slope,
        outcome_col="fwd_ret",
    )
    assert result["verdict"] == "PASS"
    assert result["wf_t"] >= 2.0
    assert result["n_folds"] == 3
    assert check_no_peeking(result) is True
    # predict_fn, outcome_col and forbid_calendar are keyword-only
    with pytest.raises(TypeError):
        walkforward(  # type: ignore[call-arg]
            data, 120, 60, 60, _fit_slope, _predict_slope
        )


def test_frozen_threshold_selects_best_grid_point() -> None:
    gen = np.random.default_rng(71)
    signal = np.zeros(200)
    for day in range(1, 200):
        signal[day] = 0.6 * signal[day - 1] + gen.standard_normal()
    outcome = 0.5 * signal + gen.standard_normal(200)
    train_df = pd.DataFrame({"signal": signal, "outcome": outcome})
    grid = [
        float(np.quantile(signal, 0.3)),
        float(np.quantile(signal, 0.7)),
    ]
    chosen = frozen_threshold(train_df, "signal", "outcome", grid=grid)
    low = frozen_threshold(train_df, "signal", "outcome", grid=[grid[0]])
    high = frozen_threshold(train_df, "signal", "outcome", grid=[grid[1]])
    assert set(chosen) == {"threshold", "train_t"}
    assert chosen["threshold"] in grid
    assert chosen["train_t"] > 0.0
    assert chosen["train_t"] == pytest.approx(
        max(low["train_t"], high["train_t"])
    )
    default = frozen_threshold(train_df, "signal", "outcome")
    assert grid[0] <= default["threshold"] <= grid[1]
    assert default["train_t"] > 0.0


def test_walkforward_is_deterministic_for_identical_inputs() -> None:
    def build_data() -> pd.DataFrame:
        return _edge_frame(300, 66, 0.5)

    first = walkforward(
        build_data(), train_window=150, test_window=50, step=50
    )
    second = walkforward(
        build_data(), train_window=150, test_window=50, step=50
    )
    assert first["wf_t"] == second["wf_t"]
    assert first["folds"].equals(second["folds"])


def test_purge_gap_separates_train_from_test() -> None:
    data = _edge_frame(360, 85, 0.5)
    result = walkforward(
        data, train_window=150, test_window=50, step=50, purge_gap=7
    )
    folds = result["folds"]
    assert "purge_gap" in folds.columns
    assert (folds["purge_gap"] == 7).all()
    assert (
        (folds["train_end"] + folds["purge_gap"]) == folds["test_start"]
    ).all()
    assert (folds["n_train"] == folds["train_end"]).all()
    assert result["purge_gap"] == 7
    assert check_no_peeking(result) is True


def test_purge_gap_zero_keeps_touching_boundaries() -> None:
    data = _edge_frame(360, 85, 0.5)
    result = walkforward(data, train_window=150, test_window=50, step=50)
    folds = result["folds"]
    assert (folds["purge_gap"] == 0).all()
    assert (folds["train_end"] == folds["test_start"]).all()
    assert check_no_peeking(result) is True


def test_negative_purge_gap_is_rejected() -> None:
    data = _edge_frame(360, 85, 0.5)
    with pytest.raises(ValueError, match="purge_gap"):
        walkforward(
            data, train_window=150, test_window=50, step=50, purge_gap=-1
        )


def test_overlapping_test_slices_produce_warning() -> None:
    data = _edge_frame(360, 85, 0.5)
    result = walkforward(data, train_window=150, test_window=50, step=10)
    assert any("overlap" in warning for warning in result["warnings"])
