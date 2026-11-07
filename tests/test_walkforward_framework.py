"""Tests for atropos.walkforward_framework: honest out-of-sample t-stats.

A planted linear edge must walk-forward to PASS with the expected fold
count and no peeking, pure noise must never PASS, the calendar guard
must fire, and tampered folds must be caught by check_no_peeking.
Overlapping folds (step < test_window) are scored on the UNION of test
rows - each row exactly once - so refit frequency cannot inflate wf_t.
"""

from __future__ import annotations

import math
from typing import Any

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


def _fit_const(_train_df: pd.DataFrame) -> dict[str, float]:
    """Fit that ignores the train slice (deterministic union checks)."""
    return {"slope": 1.0}


def _predict_identity(
    _model: dict[str, float], test_df: pd.DataFrame
) -> np.ndarray:
    """Prediction equal to the signal column (no outcome needed)."""
    return test_df["signal"].to_numpy(dtype=float)


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


def test_calendar_guard_matches_whole_tokens_not_substrings() -> None:
    data = _edge_frame(360, 91, 0.5)
    # "window" contains "dow", "update" contains "date" - benign names
    for benign in ("window", "update"):
        framed = data.rename(columns={"signal": benign})
        result = walkforward(
            framed, train_window=180, test_window=60, step=60
        )
        assert result["n_folds"] == 3
    # genuine calendar dummies (incl. plurals) are still forbidden
    for banned in ("year", "month", "dow", "doy", "days", "dates"):
        framed = data.rename(columns={"signal": banned})
        with pytest.raises(ValueError, match="calendar terms forbidden"):
            walkforward(framed, train_window=180, test_window=60, step=60)


def test_calendar_guard_allows_day_suffix_names() -> None:
    data = _edge_frame(360, 97, 0.5)
    # suffix / digit-glued "day" names are naming decoration, not
    # calendar dummies
    for benign in ("5day", "trading_day", "trading_days"):
        framed = data.rename(columns={"signal": benign})
        result = walkforward(
            framed, train_window=180, test_window=60, step=60
        )
        assert result["n_folds"] == 3
    # ... while the day_of_week-class is still forbidden
    for banned in ("day", "days", "day_of_week", "day_of_month"):
        framed = data.rename(columns={"signal": banned})
        with pytest.raises(ValueError, match="calendar terms forbidden"):
            walkforward(framed, train_window=180, test_window=60, step=60)


def test_overlapping_folds_score_union_of_test_rows_once() -> None:
    data = _edge_frame(360, 92, 0.5)
    result = walkforward(
        data,
        train_window=150,
        test_window=30,
        step=1,
        fit_fn=_fit_const,
        predict_fn=_predict_identity,
    )
    assert result["n_folds"] == 181  # t0 = 150 .. 330
    assert result["n_pnl_unique"] == 210  # rows 150..359, each ONCE
    assert result["overlap_factor"] == pytest.approx(30.0)
    assert any("overlap" in warning for warning in result["warnings"])
    assert set(result) >= {
        "wf_t",
        "warnings",
        "n_pnl_unique",
        "overlap_factor",
        "purge_gap",
    }
    # honest t-stat: every OOS row exactly once, independently computed
    preds = data["signal"].to_numpy(dtype=float)[150:]
    realized = data["outcome"].to_numpy(dtype=float)[150:]
    pnl = np.sign(preds) * realized
    sd = float(np.std(pnl, ddof=1))
    honest_t = float(np.mean(pnl) / sd * np.sqrt(pnl.size))
    assert result["wf_t"] == pytest.approx(honest_t)
    direction = float(np.mean(np.sign(preds) == np.sign(realized)))
    assert result["wf_direction_consistency"] == pytest.approx(direction)
    # the naive overlapping stack counts each row up to 30 times
    # (5430 stacked rows) and inflates the t-stat by up to
    # sqrt(test_window / step) = sqrt(30); the fold-coverage ramps at
    # both ends of the sample pull it slightly below that bound
    signal_all = data["signal"].to_numpy(dtype=float)
    outcome_all = data["outcome"].to_numpy(dtype=float)
    naive_pnl = np.concatenate(
        [
            np.sign(signal_all[t0 : t0 + 30]) * outcome_all[t0 : t0 + 30]
            for t0 in range(150, 331)
        ]
    )
    assert naive_pnl.size == 181 * 30
    naive_sd = float(np.std(naive_pnl, ddof=1))
    naive_t = float(np.mean(naive_pnl) / naive_sd * np.sqrt(naive_pnl.size))
    assert naive_t == pytest.approx(np.sqrt(30.0) * result["wf_t"], rel=0.15)


def test_disjoint_folds_report_union_size_and_unit_overlap_factor() -> None:
    data = _edge_frame(360, 85, 0.5)
    result = walkforward(data, train_window=150, test_window=50, step=50)
    assert result["n_folds"] == 4
    assert result["n_pnl_unique"] == 200  # 4 disjoint folds x 50 rows
    assert result["overlap_factor"] == pytest.approx(1.0)
    assert result["warnings"] == []


def test_predict_fn_gets_test_frame_without_outcome() -> None:
    gen = np.random.default_rng(93)
    signal = gen.standard_normal(300)
    data = pd.DataFrame(
        {
            "sig": signal,
            "fwd_ret": 0.6 * signal + gen.standard_normal(300),
        }
    )

    def _peeking_predict(
        _model: dict[str, float], test_df: pd.DataFrame
    ) -> np.ndarray:
        return test_df["fwd_ret"].to_numpy(dtype=float)

    with pytest.raises(KeyError):
        walkforward(
            data,
            train_window=120,
            test_window=60,
            step=60,
            fit_fn=_fit_slope,
            predict_fn=_peeking_predict,
            outcome_col="fwd_ret",
        )


def test_frozen_threshold_raises_on_fully_degenerate_grid() -> None:
    train_df = pd.DataFrame({"signal": np.ones(50), "outcome": np.zeros(50)})
    with pytest.raises(ValueError, match="non-degenerate"):
        frozen_threshold(train_df, "signal", "outcome", grid=[0.5, 1.5])
    with pytest.raises(ValueError, match="non-degenerate"):
        frozen_threshold(train_df, "signal", "outcome", grid=[])
    # default grid on a constant signal is degenerate too
    with pytest.raises(ValueError, match="non-degenerate"):
        frozen_threshold(train_df, "signal", "outcome")


def test_frozen_threshold_skips_degenerate_grid_points() -> None:
    gen = np.random.default_rng(94)
    signal = gen.standard_normal(200)
    train_df = pd.DataFrame(
        {
            "signal": signal,
            "outcome": 0.5 * signal + gen.standard_normal(200),
        }
    )
    huge = float(np.max(signal) + 10.0)  # no trades -> degenerate
    chosen = frozen_threshold(train_df, "signal", "outcome", grid=[huge, 0.0])
    assert chosen["threshold"] == 0.0
    assert np.isfinite(chosen["train_t"])


def test_frozen_threshold_tstat_uses_ddof_one() -> None:
    gen = np.random.default_rng(95)
    signal = gen.standard_normal(120)
    outcome = 0.4 * signal + gen.standard_normal(120)
    train_df = pd.DataFrame({"signal": signal, "outcome": outcome})
    threshold = float(np.quantile(signal, 0.4))
    chosen = frozen_threshold(train_df, "signal", "outcome", grid=[threshold])
    pos = (signal > threshold).astype(float)
    pnl = np.zeros_like(outcome)
    pnl[1:] = pos[:-1] * outcome[1:]
    expected = float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(pnl.size))
    assert chosen["train_t"] == pytest.approx(expected)


def test_default_predict_constant_predictions_raise() -> None:
    gen = np.random.default_rng(96)
    data = pd.DataFrame(
        {"flat": np.zeros(300), "outcome": gen.standard_normal(300)}
    )
    with pytest.raises(ValueError, match="constant predictions"):
        walkforward(data, train_window=150, test_window=50, step=50)


# ----------------------------------------------------------------------------
# shrink gate (FRAMEWORK.md stage-6 kill criterion) + label_horizon
# ----------------------------------------------------------------------------
def _predict_zero(
    _model: dict[str, float], test_df: pd.DataFrame
) -> np.ndarray:
    """All-zero forecasts: sign 0 -> fully degenerate PnL statistics."""
    return np.zeros(len(test_df))


def _shrink_world(
    seed: int,
    clean_coef: float,
    clean_noise: float,
    n_obs: int = 420,
    split: int = 300,
) -> pd.DataFrame:
    """Edge-free early sample, strong late edge with clean noise.

    The full-sample in-sample reference t is diluted by the first 300
    rows (weak loading, wide noise) while the walk-forward union
    (rows 300..420 under train_window=300) sees only the late clean
    edge, so ``wf_shrink_ratio = wf_t / in_sample_t`` lands far above
    the 2.0x default cap.
    """
    gen = np.random.default_rng(seed)
    driver = gen.standard_normal(n_obs)
    outcome = np.empty(n_obs)
    outcome[:split] = 0.4 * driver[:split] + (
        2.0 * gen.standard_normal(split)
    )
    outcome[split:] = clean_coef * driver[split:] + (
        clean_noise * gen.standard_normal(n_obs - split)
    )
    return pd.DataFrame({"signal": driver, "outcome": outcome})


def _shrink_run(data: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Deterministic sign-rule walk-forward used by the shrink tests."""
    params: dict[str, Any] = {
        "train_window": 300,
        "test_window": 40,
        "step": 40,
        "fit_fn": _fit_const,
        "predict_fn": _predict_identity,
    }
    params.update(kwargs)
    return walkforward(data, **params)


def test_shrinkage_over_cap_is_overfit_death() -> None:
    """Ratio far above the cap: FAIL with an OVERFIT_DEATH marker.

    Old behaviour (pre-gate): the shrink ratio was computed but never
    consulted, so this world walked forward to PASS (wf_t ~ 15,
    direction ~ 0.99) no matter how far it departed from the in-sample
    reference.  The FRAMEWORK.md stage-6 kill criterion
    ("shrinkage > 2x -> OVERFIT-DEATH") now flips it to a FAIL whose
    verdict_reason carries the OVERFIT_DEATH marker.
    """
    result = _shrink_run(
        _shrink_world(seed=42, clean_coef=1.5, clean_noise=0.05)
    )
    shrink = float(result["wf_shrink_ratio"])
    assert np.isfinite(shrink) and shrink > 2.0
    assert result["verdict"] == "FAIL"
    reason = result["verdict_reason"]
    assert reason is not None
    assert "OVERFIT_DEATH" in reason
    assert "shrinkage" in reason


def test_shrinkage_gate_boundary_is_strict() -> None:
    """A ratio exactly AT the cap must not fire; a hair below does.

    The cap is adaptive: the probe run reports the world's ratio s,
    the second run sets ``shrink_cap=s`` (exact float equality - the
    strict ``>`` keeps the gate silent and the wf_t tier decides,
    PASS here), the third sets the adjacent-below double so
    ``s > cap`` fires.  Mutation target: loosening ``>`` to ``>=``
    turns the equality run into an OVERFIT_DEATH FAIL.
    """
    data = _shrink_world(seed=42, clean_coef=1.5, clean_noise=0.05)
    probe = _shrink_run(data)
    ratio = float(probe["wf_shrink_ratio"])
    assert np.isfinite(ratio) and ratio > 0.0
    assert probe["wf_t"] >= 2.0
    assert probe["wf_direction_consistency"] >= 0.5
    at_cap = _shrink_run(data, shrink_cap=ratio)
    assert at_cap["verdict"] == "PASS"
    assert at_cap["verdict_reason"] is None
    just_below = _shrink_run(
        data, shrink_cap=math.nextafter(ratio, -math.inf)
    )
    assert just_below["verdict"] == "FAIL"
    assert just_below["verdict_reason"] is not None
    assert "OVERFIT_DEATH" in str(just_below["verdict_reason"])


def test_shrinkage_at_cap_weak_wf_t_takes_the_tier() -> None:
    """At cap == ratio a weak wf_t takes its DEGRADED tier untouched.

    The gate applies ONLY above the strict cap; the tier semantics
    (wf_t >= 2 PASS / >= 1 DEGRADED / else FAIL) survive unchanged
    underneath it.
    """
    data = _shrink_world(seed=42, clean_coef=0.02, clean_noise=0.1)
    probe = _shrink_run(data)
    ratio = float(probe["wf_shrink_ratio"])
    assert np.isfinite(ratio) and ratio > 0.0
    assert 1.0 <= probe["wf_t"] < 2.0
    at_cap = _shrink_run(data, shrink_cap=ratio)
    assert at_cap["verdict"] == "DEGRADED"
    assert at_cap["verdict_reason"] is None


def test_shrinkage_nan_is_inconclusive_not_silent_pass() -> None:
    """Zero forecasts -> degenerate t's -> NaN ratio: INCONCLUSIVE.

    Old behaviour: a NaN wf_t slid into the bottom tier and read FAIL
    while the undefined shrinkage was silently ignored.  "The overfit
    check could not run" is distinct from "no edge": the verdict is
    INCONCLUSIVE with the explicit cause in verdict_reason.
    """
    result = walkforward(
        _edge_frame(360, 85, 0.5),
        train_window=150,
        test_window=50,
        step=50,
        fit_fn=_fit_const,
        predict_fn=_predict_zero,
    )
    assert np.isnan(result["wf_shrink_ratio"])
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["verdict_reason"] == (
        "OVERFIT check undefined: shrinkage NaN (baseline t degenerate)"
    )


def test_shrink_cap_must_be_finite_and_positive() -> None:
    """The cap is a validated parameter (finite, strictly positive)."""
    data = _edge_frame(360, 85, 0.5)
    for bad_cap in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="shrink_cap"):
            walkforward(
                data,
                train_window=150,
                test_window=50,
                step=50,
                shrink_cap=bad_cap,
            )


def test_label_horizon_one_keeps_purge_gap_bit_for_bit() -> None:
    """purge_gap=5 + label_horizon=1 is bit-for-bit purge_gap=5.

    Pins the no-conflation contract: label_horizon is never folded
    into the purge.  The absolute fold geometry (n_train/train_end
    with purge_gap=5) is the conflation-immune oracle - an equality
    check between two runs of the same code would pass even if the
    internals conflated the two parameters; the absolute counts
    (train_window=150 -> n_train 150/200/250/300, train_end exactly
    5 rows before test_start) fail under either conflation direction
    (purge := label_horizon, or purge := purge_gap + label_horizon).
    """
    data = _edge_frame(360, 85, 0.5)
    explicit = walkforward(
        data,
        train_window=150,
        test_window=50,
        step=50,
        purge_gap=5,
        label_horizon=1,
    )
    baseline = walkforward(
        data,
        train_window=150,
        test_window=50,
        step=50,
        purge_gap=5,
    )
    assert explicit["wf_t"] == baseline["wf_t"]
    assert explicit["verdict"] == baseline["verdict"]
    assert explicit["verdict_reason"] == baseline["verdict_reason"]
    assert explicit["folds"].equals(baseline["folds"])
    folds = explicit["folds"]
    assert list(folds["n_train"]) == [150, 200, 250, 300]
    assert list(folds["train_end"]) == [150, 200, 250, 300]
    assert ((folds["train_end"] + 5) == folds["test_start"]).all()


def test_label_horizon_above_one_is_rejected() -> None:
    """label_horizon > 1 needs an overlap correction that is unbuilt.

    Old behaviour: the parameter did not exist (an unexpected keyword
    raised TypeError, the guard nothing at all).  Overlapping labels
    inflate wf_t anti-conservatively, so the case must be a loud
    config ValueError, not a silently uncalibrated statistic; 0 /
    negative are invalid outright.
    """
    data = _edge_frame(360, 85, 0.5)
    with pytest.raises(ValueError, match="overlap correction"):
        walkforward(
            data,
            train_window=150,
            test_window=50,
            step=50,
            label_horizon=5,
        )
    with pytest.raises(ValueError, match="label_horizon"):
        walkforward(
            data,
            train_window=150,
            test_window=50,
            step=50,
            label_horizon=0,
        )
