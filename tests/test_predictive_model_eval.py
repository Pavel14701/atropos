"""Tests for atropos.predictive_model_eval: purged CV and OOS verdicts."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from atropos.predictive_model_eval import (
    ModelFn,
    Predictor,
    deflated_performance,
    feature_importance_stability,
    kernel_model_fn,
    knn_model_fn,
    ols_model_fn,
    purged_kfold_cv,
    sign_consistency,
    walkforward_model_eval,
)
from scipy import stats


def make_linear_problem(
    n_rows: int, noise_scale: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded 2-feature problem whose first feature drives the target."""
    generator = np.random.default_rng(seed)
    features = generator.standard_normal((n_rows, 2))
    noise = noise_scale * generator.standard_normal(n_rows)
    target = 0.8 * features[:, 0] + noise
    return features, target


def test_purged_kfold_blocks_partition_timeline_with_purge_gaps() -> None:
    total_rows = 300
    features, target = make_linear_problem(total_rows, 0.5, 11)
    purge_gap = 2
    result = purged_kfold_cv(
        features,
        target,
        n_splits=5,
        purge_gap=purge_gap,
        model_fn=ols_model_fn,
    )
    assert result["n"] == total_rows
    assert result["n_splits"] == 5
    assert result["purge_gap"] == purge_gap
    assert result["model_used"] is True
    folds = result["folds"]
    test_lo = folds["test_lo"].to_numpy(dtype=int)
    test_hi = folds["test_hi"].to_numpy(dtype=int)
    # contiguous, non-overlapping test blocks covering [0, n)
    assert int(test_lo[0]) == 0
    assert int(test_hi[-1]) == total_rows - 1
    assert bool(np.all(test_lo[1:] == test_hi[:-1] + 1))
    assert int(folds["n_test"].sum()) == total_rows
    # train sets exclude the purged band around every test block
    all_indices = np.arange(total_rows)
    n_train = folds["n_train"].to_numpy(dtype=int)
    for fold_pos in range(5):
        lo = int(test_lo[fold_pos])
        hi = int(test_hi[fold_pos])
        kept = (all_indices < lo - purge_gap) | (all_indices > hi + purge_gap)
        assert n_train[fold_pos] == int(np.sum(kept))
    # strong signal: high out-of-fold IC and hit rate, full oof coverage
    assert result["mean_ic"] > 0.5
    assert result["oos_hit_rate"] > 0.6
    # two-sided p-value of the per-fold IC t-stat
    assert 0.0 <= result["ic_p_value"] <= 1.0
    assert result["ic_p_value"] < 0.05
    # nothing skipped: every fold was scored
    assert bool(folds["skipped_reason"].isna().all())
    oof_predictions = result["oof_predictions"]
    assert oof_predictions is not None
    assert oof_predictions.shape == (total_rows,)
    assert int(np.isfinite(oof_predictions).sum()) == total_rows


def test_purged_kfold_purge_gap_shrinks_training_sets() -> None:
    features, target = make_linear_problem(240, 0.5, 12)
    no_gap = purged_kfold_cv(features, target, n_splits=4, purge_gap=0)
    gapped = purged_kfold_cv(features, target, n_splits=4, purge_gap=3)
    assert int(gapped["folds"]["n_train"].sum()) < int(
        no_gap["folds"]["n_train"].sum()
    )


def test_purged_kfold_skips_folds_with_too_few_train_rows() -> None:
    generator = np.random.default_rng(45)
    # 20 rows x 20 features: every fold trains on < n_features + 2 rows
    features = generator.standard_normal((20, 20))
    target = generator.standard_normal(20)
    result = purged_kfold_cv(
        features, target, n_splits=5, purge_gap=0, model_fn=ols_model_fn
    )
    folds = result["folds"]
    # the skip is now LOUD: each fold row says why it was not scored
    assert bool(folds["skipped_reason"].notna().all())
    reason = str(folds["skipped_reason"].iloc[0])
    assert "n_train=16" in reason
    assert "n_features+2=22" in reason
    assert bool(folds["fold_ic"].isna().all())
    assert bool(np.isnan(result["mean_ic"]))
    assert bool(np.isnan(result["ic_t"]))
    assert bool(np.isnan(result["ic_p_value"]))
    oof_predictions = result["oof_predictions"]
    assert oof_predictions is not None
    assert bool(np.isnan(oof_predictions).all())


def test_purged_kfold_p_value_dof_uses_scored_folds() -> None:
    """Mixed skip/scored runs must not inflate the t-test dof.

    Regression: ``ic_p_value`` used ``n_splits - 1`` degrees of freedom
    even when folds were skipped, while ``ic_t`` was computed from the
    scored folds only -- skipped folds silently made the p-value too
    small (anti-conservative).  With 4 scored folds of 5 splits the
    correct dof is 3, not 4, and the honest p-value is larger.
    """
    generator = np.random.default_rng(46)
    features = generator.standard_normal((21, 15))
    target = 0.9 * features[:, 0] + generator.standard_normal(21)
    result = purged_kfold_cv(
        features, target, n_splits=5, purge_gap=0, model_fn=ols_model_fn
    )
    folds = result["folds"]
    # exactly one skipped fold, four scored -> dof = 4 - 1 = 3
    assert int(folds["skipped_reason"].notna().sum()) == 1
    scored = int(np.isfinite(folds["fold_ic"]).sum())
    assert scored == 4
    assert bool(np.isfinite(result["ic_t"]))
    expected = float(2.0 * stats.t.sf(abs(result["ic_t"]), scored - 1))
    assert result["ic_p_value"] == pytest.approx(expected)
    # the old dof (n_splits - 1 = 4) understated this p-value
    stale = float(2.0 * stats.t.sf(abs(result["ic_t"]), 5 - 1))
    assert result["ic_p_value"] > stale


def test_purged_kfold_p_value_unchanged_when_nothing_skipped() -> None:
    """No-skip runs keep the historical dof (n_splits - 1)."""
    features, target = make_linear_problem(300, 0.5, 11)
    result = purged_kfold_cv(
        features, target, n_splits=5, purge_gap=2, model_fn=ols_model_fn
    )
    assert int(np.isfinite(result["folds"]["fold_ic"]).sum()) == 5
    expected = float(2.0 * stats.t.sf(abs(result["ic_t"]), 4))
    assert result["ic_p_value"] == pytest.approx(expected)


def test_purged_kfold_rejects_bad_arguments() -> None:
    features, target = make_linear_problem(60, 0.5, 13)
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        purged_kfold_cv(features, target, n_splits=1)
    with pytest.raises(ValueError, match="too small"):
        purged_kfold_cv(features, target, n_splits=20)  # needs 4*n_splits
    with pytest.raises(ValueError, match="rows but target has"):
        purged_kfold_cv(features, target[:-3])


def test_integer_parameters_reject_fractional_values() -> None:
    """Every integer parameter rejects ``2.5`` instead of truncating it.

    Regression class: ``int(n_splits)``-style coercion used to treat
    ``n_splits=2.5`` as 2 folds without a word.
    """
    features, target = make_linear_problem(120, 0.5, 77)
    returns = np.random.default_rng(78).normal(0.001, 0.01, 200)
    cases: list[tuple[Callable[[], object], str]] = [
        (
            lambda: purged_kfold_cv(
                features,
                target,
                n_splits=2.5,  # type: ignore[arg-type]
            ),
            "n_splits",
        ),
        (
            lambda: purged_kfold_cv(
                features,
                target,
                purge_gap=2.5,  # type: ignore[arg-type]
            ),
            "purge_gap",
        ),
        (
            lambda: walkforward_model_eval(
                features,
                target,
                ols_model_fn,
                train_window=2.5,  # type: ignore[arg-type]
            ),
            "train_window",
        ),
        (
            lambda: walkforward_model_eval(
                features,
                target,
                ols_model_fn,
                test_window=2.5,  # type: ignore[arg-type]
            ),
            "test_window",
        ),
        (
            lambda: walkforward_model_eval(
                features,
                target,
                ols_model_fn,
                step=2.5,  # type: ignore[arg-type]
            ),
            "step",
        ),
        (
            lambda: walkforward_model_eval(
                features,
                target,
                ols_model_fn,
                purge_gap=2.5,  # type: ignore[arg-type]
            ),
            "purge_gap",
        ),
        (
            lambda: feature_importance_stability(
                features,
                target,
                ols_model_fn,
                n_bootstrap=2.5,  # type: ignore[arg-type]
            ),
            "n_bootstrap",
        ),
        (
            lambda: feature_importance_stability(
                features,
                target,
                ols_model_fn,
                top_k=2.5,  # type: ignore[arg-type]
            ),
            "top_k",
        ),
        (
            lambda: deflated_performance(
                returns,
                2.5,  # type: ignore[arg-type]
                mdd=-0.1,
            ),
            "n_strategies",
        ),
        (
            lambda: deflated_performance(
                returns,
                5,
                mdd=-0.1,
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: knn_model_fn(
                features,
                target,
                n_neighbors=2.5,  # type: ignore[arg-type]
            ),
            "n_neighbors",
        ),
    ]
    for call, name in cases:
        with pytest.raises(ValueError, match=f"{name} must be an integer"):
            call()


def test_walkforward_real_edge_passes_noise_target_fails() -> None:
    features, target = make_linear_problem(300, 0.5, 14)
    strong = walkforward_model_eval(
        features, target, ols_model_fn, train_window=120, test_window=30
    )
    assert strong["n_folds"] >= 3
    assert strong["verdict"] == "PASS"
    assert strong["pnl_t"] >= 2.0
    assert strong["oos_ic"] > 0.5
    assert strong["oos_hit_rate"] > 0.6
    assert strong["predictions"].shape == strong["realized"].shape
    assert strong["predictions"].size == 180  # 6 folds x 30 rows
    noise_target = np.random.default_rng(15).standard_normal(300)
    weak = walkforward_model_eval(
        features, noise_target, ols_model_fn, train_window=120, test_window=30
    )
    assert weak["verdict"] == "FAIL"
    assert abs(weak["oos_ic"]) < 0.2


def test_walkforward_rejects_impossible_windows() -> None:
    features, target = make_linear_problem(120, 0.5, 16)
    with pytest.raises(ValueError, match="too small"):
        walkforward_model_eval(
            features, target, ols_model_fn, train_window=100, test_window=50
        )
    with pytest.raises(ValueError, match="train_window >= 10"):
        walkforward_model_eval(
            features, target, ols_model_fn, train_window=5, test_window=10
        )


def test_feature_importance_stability_ranks_true_driver_top() -> None:
    generator = np.random.default_rng(21)
    driver = generator.standard_normal(400)
    noise_feature = generator.standard_normal(400)
    features = pd.DataFrame({"driver": driver, "noise_feat": noise_feature})
    target = 0.8 * driver + 0.3 * generator.standard_normal(400)
    result = feature_importance_stability(
        features, target, ols_model_fn, n_bootstrap=40, top_k=1, seed=5
    )
    assert result["verdict"] == "STABLE"
    assert result["mean_pairwise_spearman"] >= 0.6
    assert result["n_valid"] == 40
    assert result["top_k"] == 1
    table = result["per_feature"]
    assert list(table["feature"]) == ["driver", "noise_feat"]
    by_name = table.set_index("feature")
    assert float(by_name.loc["driver", "top1_freq"]) > 0.9
    assert float(by_name.loc["noise_feat", "top1_freq"]) < 0.1
    assert float(by_name.loc["driver", "mean_rank"]) < float(
        by_name.loc["noise_feat", "mean_rank"]
    )
    mean_importance = table["mean_importance"].to_numpy(dtype=float)
    assert float(np.sum(mean_importance)) == pytest.approx(1.0)


def test_feature_importance_stability_top_k_column_contract() -> None:
    generator = np.random.default_rng(22)
    features = pd.DataFrame(
        {
            "driver": generator.standard_normal(300),
            "noise_feat": generator.standard_normal(300),
        }
    )
    target = 0.9 * features[
        "driver"
    ].to_numpy() + 0.2 * generator.standard_normal(300)
    result = feature_importance_stability(
        features, target, ols_model_fn, n_bootstrap=20, top_k=2, seed=6
    )
    assert "top2_freq" in result["per_feature"].columns
    top_two_freq = result["per_feature"]["top2_freq"].to_numpy(dtype=float)
    # with 2 features and top_k=2 every draw ranks both features top
    assert float(np.sum(top_two_freq)) == pytest.approx(1.0)


def test_feature_importance_stability_rejects_bad_arguments() -> None:
    generator = np.random.default_rng(23)
    features = generator.standard_normal((100, 3))
    target = generator.standard_normal(100)
    with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
        feature_importance_stability(
            features, target, ols_model_fn, n_bootstrap=1
        )
    with pytest.raises(ValueError, match="top_k must be in"):
        feature_importance_stability(features, target, ols_model_fn, top_k=0)
    with pytest.raises(ValueError, match="top_k must be in"):
        feature_importance_stability(features, target, ols_model_fn, top_k=4)


def test_sign_consistency_perfect_predictor_is_exact() -> None:
    predictions = np.random.default_rng(31).standard_normal(300)
    result = sign_consistency(predictions, predictions)
    assert result["directional_accuracy"] == 1.0
    assert result["verdict"] == "PASS"
    assert result["stable"] is True
    assert result["n"] == 300
    regimes = result["sign_stability_by_regime"]
    assert {"regime_1", "vol_tercile_low", "vol_tercile_high"} <= set(regimes)
    assert all(value == 1.0 for value in regimes.values())


def test_sign_consistency_noise_fails_near_one_half() -> None:
    generator = np.random.default_rng(32)
    predictions = generator.standard_normal(300)
    outcomes = generator.standard_normal(300)
    result = sign_consistency(predictions, outcomes)
    assert 0.35 < result["directional_accuracy"] < 0.65
    assert result["verdict"] == "FAIL"
    assert result["stable"] is False


def test_sign_consistency_validates_n_regimes() -> None:
    """``n_regimes`` is validated, not silently coerced.

    Regression: ``max(2, int(n_regimes))`` treated ``n_regimes=1`` as 2
    regimes and ``n_regimes=2.5`` as 2 without a word -- unlike every
    other integer parameter of this module, which raises ValueError.
    """
    predictions = np.random.default_rng(33).standard_normal(300)
    with pytest.raises(ValueError, match="n_regimes must be >= 2"):
        sign_consistency(predictions, predictions, n_regimes=1)
    with pytest.raises(ValueError, match="n_regimes must be an integer"):
        sign_consistency(predictions, predictions, n_regimes=2.5)


def test_sign_consistency_regime_keys_follow_n_regimes() -> None:
    """Time-block keys are neutral ``regime_{i}`` for any ``n_regimes``.

    Regression: the keys were hardcoded ``time_third_{i}`` -- wrong for
    ``n_regimes != 3`` (a 5-regime run reported five "thirds") and
    ambiguous about which split produced them.
    """
    predictions = np.random.default_rng(34).standard_normal(600)
    result = sign_consistency(predictions, predictions, n_regimes=5)
    regimes = result["sign_stability_by_regime"]
    assert {"regime_1", "regime_5"} <= set(regimes)
    assert not any(key.startswith("time_third_") for key in regimes)


def test_sign_consistency_edges() -> None:
    with pytest.raises(ValueError, match="values"):
        sign_consistency(np.ones(5), np.ones(6))
    tiny = sign_consistency(
        np.array([1.0, -1.0, 1.0]), np.array([1.0, -1.0, -1.0])
    )
    assert tiny["verdict"] == "FAIL"
    assert tiny["n"] == 3
    assert tiny["note"] == "insufficient data"
    assert bool(np.isnan(tiny["directional_accuracy"]))


def test_sign_consistency_constant_predictions_fail() -> None:
    # REGRESSION (blocker): "always +1" on an outcome with ~60% positives
    # used to PASS every accuracy/regime gate — but a constant predictor
    # is a beta bet on the outcome mean, not a model.  It must FAIL.
    generator = np.random.default_rng(33)
    outcomes = generator.choice(np.array([-1.0, 1.0]), size=300, p=[0.4, 0.6])
    result = sign_consistency(np.ones(300), outcomes)
    assert result["verdict"] == "FAIL"
    assert result["stable"] is False
    assert result["reason"] == "degenerate (constant) predictions"
    assert result["sign_stability_by_regime"] == {}
    assert result["n"] == 300
    assert 0.5 < result["directional_accuracy"] < 0.7


def test_sign_consistency_constant_outcome_inconclusive() -> None:
    predictions = np.random.default_rng(34).standard_normal(200)
    result = sign_consistency(predictions, np.ones(200))
    # no sign variation in the outcome: nothing to explain -> not FAIL,
    # the test itself is unidentifiable here
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "degenerate (constant) outcomes"
    assert result["sign_stability_by_regime"] == {}
    assert result["stable"] is False
    assert result["n"] == 200


def test_sign_consistency_excludes_exact_zeros() -> None:
    # sign(0) == 0 carries no direction: rows with an exact 0 on EITHER
    # side are excluded from every accuracy denominator instead of
    # silently counting as mismatches.
    n_rows = 30
    row_idx = np.arange(n_rows)
    zero_rows = row_idx % 5 == 4  # 6 rows with outcome exactly 0
    signs = np.where(row_idx % 2 == 0, 1.0, -1.0)
    magnitudes = np.where(row_idx % 10 < 4, 2.0, 1.0)
    predictions = np.where(zero_rows, 1.0, signs)
    outcomes = np.where(zero_rows, 0.0, signs * magnitudes)
    result = sign_consistency(predictions, outcomes)
    assert result["n"] == 30
    assert result["n_usable"] == 24
    assert result["directional_accuracy"] == 1.0  # not the naive 24/30
    # the low-|outcome| tercile contains only zero-outcome rows: with no
    # usable rows left it is omitted rather than scored
    assert "vol_tercile_low" not in result["sign_stability_by_regime"]
    assert result["verdict"] == "PASS"
    assert result["stable"] is True


def test_deflated_performance_many_strategies_raise_the_bar() -> None:
    returns = np.random.default_rng(41).normal(0.002, 0.01, 400)
    few = deflated_performance(returns, n_strategies=2, mdd=-0.15)
    many = deflated_performance(returns, n_strategies=5000, mdd=-0.15)
    assert few["verdict"] == "PASS"
    assert many["verdict"] == "FAIL"
    assert many["p_value"] > few["p_value"]
    assert many["sr_benchmark_annual"] > few["sr_benchmark_annual"]
    assert many["excess_sr"] < few["excess_sr"]
    assert few["excess_sr"] == pytest.approx(
        few["sr_annual"] - few["sr_benchmark_annual"]
    )
    assert few["n_strategies"] == 2


def test_deflated_performance_zero_mean_returns_fail() -> None:
    flat = np.random.default_rng(42).normal(0.0, 0.01, 400)
    result = deflated_performance(flat, n_strategies=2, mdd=-0.10)
    assert result["verdict"] == "FAIL"
    assert result["p_value"] > 0.05
    with pytest.raises(ValueError, match="n_strategies must be >= 1"):
        deflated_performance(flat, n_strategies=0, mdd=-0.10)


def test_deflated_performance_single_strategy_disables_penalty() -> None:
    # n_strategies=1 means "no selection among trials": the multiple-
    # testing penalty must vanish (DSR == PSR).  The DIRECTION below is
    # stable regardless of evaluation_metrics' own n_trials=1 handling:
    # one trial can never be punished harder than ten.
    returns = np.random.default_rng(43).normal(0.0005, 0.01, 500)
    single = deflated_performance(returns, n_strategies=1, mdd=-0.10)
    ten = deflated_performance(returns, n_strategies=10, mdd=-0.10)
    assert single["sr_annual"] == ten["sr_annual"]  # same returns
    assert single["dsr"] > ten["dsr"]
    assert single["p_value"] < ten["p_value"]
    assert single["sr_benchmark_annual"] <= ten["sr_benchmark_annual"]


def test_deflated_performance_rejects_positive_mdd() -> None:
    returns = np.random.default_rng(44).normal(0.001, 0.01, 300)
    # a positive "drawdown" is a sign error: it must not silently yield
    # deflated_calmar = inf
    with pytest.raises(ValueError, match="mdd must be <= 0"):
        deflated_performance(returns, n_strategies=5, mdd=0.15)
    # mdd == 0 ("no drawdown observed") stays allowed
    no_drawdown = deflated_performance(returns, n_strategies=5, mdd=0.0)
    assert no_drawdown["mdd"] == 0.0
    assert bool(np.isinf(no_drawdown["deflated_calmar"]))


def test_ols_model_fn_recovers_known_beta_exactly() -> None:
    generator = np.random.default_rng(51)
    features = generator.standard_normal((200, 2))
    true_beta = np.array([1.5, 0.7, -0.4])
    target = true_beta[0] + features @ true_beta[1:]
    model = ols_model_fn(features, target)
    coefficients = np.asarray(cast(Any, model).coef_, dtype=float)
    assert np.allclose(coefficients, true_beta)
    forecasts = np.asarray(model(features[:50]), dtype=float)
    assert forecasts.shape == (50,)  # (n, k) features -> (n,) forecasts
    assert np.allclose(forecasts, target[:50])


def test_ols_model_fn_predictor_and_importance_contract() -> None:
    generator = np.random.default_rng(52)
    features = generator.standard_normal((150, 2))
    target = 0.5 * features[:, 0] + generator.standard_normal(150)
    model = ols_model_fn(features, target)
    forecasts = np.asarray(model(features), dtype=float)
    assert forecasts.shape == (150,)
    single_feature_model = ols_model_fn(features[:, :1], target)
    two_dim_input = np.asarray(
        single_feature_model(generator.standard_normal((10, 1))), dtype=float
    )
    assert two_dim_input.shape == (10,)
    one_dim_input = np.asarray(
        single_feature_model(generator.standard_normal(10)), dtype=float
    )
    assert one_dim_input.shape == (10,)
    importances = np.asarray(
        cast(Any, model).feature_importances_, dtype=float
    )
    assert importances.shape == (2,)
    assert bool(np.all(importances >= 0.0))
    assert float(np.sum(importances)) == pytest.approx(1.0)


def test_ols_model_fn_rejects_rank_deficient_design() -> None:
    # fail-closed: a duplicated feature column makes the OLS solution
    # non-unique; the fit must raise instead of silently returning a
    # minimum-norm coefficient pick
    generator = np.random.default_rng(53)
    column = generator.standard_normal(120)
    duplicated = np.column_stack([column, column])
    target = 0.5 * column + generator.standard_normal(120)
    with pytest.raises(ValueError, match="rank-deficient"):
        ols_model_fn(duplicated, target)
    # a constant (zero-variance) feature column is collinear with the
    # intercept and is rejected by the same rank check
    with pytest.raises(ValueError, match="rank-deficient"):
        ols_model_fn(np.column_stack([column, np.full(120, 3.0)]), target)
    # the same data with an independent second column still fits
    model = ols_model_fn(
        np.column_stack([column, generator.standard_normal(120)]), target
    )
    assert np.asarray(model(duplicated[:10]), dtype=float).shape == (10,)


def _demo_linear_chain_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Replicate the ``run_all_examples`` stage-1 data exactly (seed 21).

    Draw order matters: f1, f2, f3, then the target noise from the same
    generator, matching ``_demo_linear_chain`` in the module.
    """
    generator = np.random.default_rng(21)
    f1 = generator.standard_normal(800)
    f2 = generator.standard_normal(800)
    f3 = generator.standard_normal(800)
    target = 0.6 * f1 + 0.3 * f2 + generator.standard_normal(800)
    features = pd.DataFrame({"alpha_1": f1, "alpha_2": f2, "noise_3": f3})
    return features, target


def test_seeded_regression_walkforward_demo_config() -> None:
    # exact pin of the demo's seed-specific walk-forward values
    # (run_all_examples asserts only the soft n_folds >= 10 floor)
    features, target = _demo_linear_chain_data()
    wfe = walkforward_model_eval(
        features, target, ols_model_fn, train_window=200, test_window=50
    )
    assert wfe["n_folds"] == 12
    assert wfe["verdict"] == "PASS"
    assert wfe["pnl_t"] > 5.0


def test_seeded_regression_importance_demo_config() -> None:
    # exact pin of the demo's seed-specific importance-stability values
    # (run_all_examples asserts only the soft n_valid >= 30 floor)
    features, target = _demo_linear_chain_data()
    stab = feature_importance_stability(
        features, target, ols_model_fn, n_bootstrap=40, top_k=1, seed=1
    )
    assert stab["n_valid"] == 40
    assert stab["verdict"] == "STABLE"
    assert list(stab["per_feature"]["feature"]) == [
        "alpha_1",
        "alpha_2",
        "noise_3",
    ]


# ---------------------------------------------------------------------------
# native non-linear models: knn_model_fn / kernel_model_fn
# ---------------------------------------------------------------------------
def _mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error between two 1-D float arrays."""
    values = np.asarray(predicted, dtype=float).reshape(-1)
    return float(np.mean((actual - values) ** 2))


def test_native_models_beat_ols_on_noiseless_quadratic() -> None:
    grid = np.linspace(-2.0, 2.0, 401)
    features = grid.reshape(-1, 1)
    target = grid**2
    train_rows = np.arange(0, grid.size, 2)
    test_rows = np.arange(1, grid.size, 2)
    held_out_features = features[test_rows]
    held_out_target = target[test_rows]
    ols_model = ols_model_fn(features[train_rows], target[train_rows])
    knn_model = knn_model_fn(
        features[train_rows], target[train_rows], n_neighbors=3
    )
    kernel_model = kernel_model_fn(features[train_rows], target[train_rows])
    ols_mse = _mean_squared_error(
        held_out_target, ols_model(held_out_features)
    )
    knn_mse = _mean_squared_error(
        held_out_target, knn_model(held_out_features)
    )
    kernel_mse = _mean_squared_error(
        held_out_target, kernel_model(held_out_features)
    )
    assert knn_mse < ols_mse
    assert kernel_mse < ols_mse


def test_native_model_closures_contract_and_determinism() -> None:
    generator = np.random.default_rng(60)
    features = generator.standard_normal((120, 3))
    target = 0.5 * features[:, 0] + generator.standard_normal(120)
    query = generator.standard_normal((25, 3))
    knn_first = knn_model_fn(features, target, n_neighbors=4)
    knn_second = knn_model_fn(features, target, n_neighbors=4)
    kernel_first = kernel_model_fn(features, target)
    kernel_second = kernel_model_fn(features, target)
    for first_model, second_model in (
        (knn_first, knn_second),
        (kernel_first, kernel_second),
    ):
        first_preds = np.asarray(first_model(query), dtype=float)
        second_preds = np.asarray(second_model(query), dtype=float)
        assert first_preds.shape == (25,)
        assert bool(np.array_equal(first_preds, second_preds))
    flat_preds = np.asarray(
        knn_first(generator.standard_normal(7)), dtype=float
    )
    assert flat_preds.shape == (7,)
    assert cast(Any, knn_first).n_neighbors_ == 4
    assert float(cast(Any, kernel_first).bandwidth_) > 0.0


def test_native_model_importance_flags_true_driver() -> None:
    generator = np.random.default_rng(61)
    driver = generator.standard_normal(300)
    noise_feature = generator.standard_normal(300)
    features = np.column_stack([driver, noise_feature])
    target = 2.0 * driver + 0.1 * generator.standard_normal(300)
    for model in (
        knn_model_fn(features, target, n_neighbors=5),
        kernel_model_fn(features, target),
    ):
        importances = np.asarray(
            cast(Any, model).feature_importances_, dtype=float
        )
        assert importances.shape == (2,)
        assert bool(np.all(importances >= 0.0))
        assert float(np.sum(importances)) == pytest.approx(1.0)
        assert importances[0] > importances[1]


def test_native_models_reject_bad_arguments() -> None:
    generator = np.random.default_rng(62)
    features = generator.standard_normal((40, 2))
    target = generator.standard_normal(40)
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        knn_model_fn(features, target, n_neighbors=0)
    with pytest.raises(ValueError, match="exceeds n_train"):
        knn_model_fn(features, target, n_neighbors=41)
    with pytest.raises(ValueError, match="bandwidth must be > 0"):
        kernel_model_fn(features, target, bandwidth=-0.5)
    with pytest.raises(ValueError, match="bandwidth must be > 0"):
        kernel_model_fn(features, target, bandwidth=0.0)


def test_kernel_model_tiny_weight_sums_fall_back_to_nearest() -> None:
    train_features = np.array([[0.0], [1.0]])
    train_target = np.array([5.0, -5.0])
    model = kernel_model_fn(train_features, train_target, bandwidth=1.0)
    # the query sits 7.75 standardized units from the nearer training
    # row: kernel weights sum to ~9e-14 — strictly positive but pure
    # underflow noise, so the nearest-target fallback must kick in
    borderline = np.asarray(model(np.array([[4.875]])), dtype=float)
    assert borderline.shape == (1,)
    assert float(borderline[0]) == pytest.approx(-5.0, abs=1e-12)
    # full underflow (query astronomically far) takes the same fallback
    far = np.asarray(model(np.array([[1e9]])), dtype=float)
    assert float(far[0]) == pytest.approx(-5.0, abs=1e-12)


def _static_model_factory(model_value: Any) -> ModelFn:
    """model_fn that ignores the data and returns a fixed fitted model."""

    def factory(
        features_train: np.ndarray, target_train: np.ndarray
    ) -> Predictor:
        _ = (features_train, target_train)
        return cast(Predictor, model_value)

    return factory


def test_extract_importances_dict_and_object_share_keys() -> None:
    generator = np.random.default_rng(65)
    features = generator.standard_normal((80, 2))
    target = generator.standard_normal(80)
    coef_dict = {"coef": np.array([-3.0, 1.0])}
    coef_object = SimpleNamespace(coef=np.array([-3.0, 1.0]))
    importance_dict = {"feature_importances": np.array([0.2, 0.8])}
    importance_object = SimpleNamespace(
        feature_importances=np.array([0.2, 0.8])
    )
    models = {
        "coef_dict": coef_dict,
        "coef_object": coef_object,
        "importance_dict": importance_dict,
        "importance_object": importance_object,
    }
    mean_importances: dict[str, np.ndarray] = {}
    for label, fitted_model in models.items():
        result = feature_importance_stability(
            features,
            target,
            _static_model_factory(fitted_model),
            n_bootstrap=3,
            seed=2,
        )
        assert result["n_valid"] == 3
        # identical importances on every draw -> perfectly rank-stable
        assert result["mean_pairwise_spearman"] == pytest.approx(1.0)
        mean_importances[label] = result["per_feature"][
            "mean_importance"
        ].to_numpy(dtype=float)
    # coef-style keys: |value| -> [3, 1] normalized = [0.75, 0.25]
    assert np.allclose(mean_importances["coef_dict"], [0.75, 0.25])
    assert np.allclose(mean_importances["coef_object"], [0.75, 0.25])
    # importance-style keys are returned as-is (no abs distortion)
    assert np.allclose(mean_importances["importance_dict"], [0.2, 0.8])
    assert np.allclose(mean_importances["importance_object"], [0.2, 0.8])
    # dict and object models now behave IDENTICALLY for every key
    assert np.array_equal(
        mean_importances["coef_dict"], mean_importances["coef_object"]
    )
    assert np.array_equal(
        mean_importances["importance_dict"],
        mean_importances["importance_object"],
    )


def test_knn_integrates_with_purged_kfold_cv() -> None:
    features, target = make_linear_problem(400, 0.5, 63)
    result = purged_kfold_cv(
        features,
        target,
        n_splits=5,
        purge_gap=1,
        model_fn=lambda feats, targ: knn_model_fn(feats, targ, n_neighbors=5),
    )
    assert result["model_used"] is True
    assert bool(np.isfinite(result["mean_ic"]))
    assert result["mean_ic"] > 0.3


def test_knn_runs_feature_importance_stability() -> None:
    features, target = make_linear_problem(250, 0.5, 64)
    result = feature_importance_stability(
        features,
        target,
        model_fn=lambda feats, targ: knn_model_fn(feats, targ, n_neighbors=5),
        n_bootstrap=10,
        seed=3,
    )
    assert isinstance(result, dict)
    assert result["n_bootstrap"] == 10
    assert result["n_valid"] == 10
    assert "verdict" in result


def test_walkforward_model_eval_purge_separates_train_and_test() -> None:
    features, target = make_linear_problem(320, 0.5, 71)
    result = walkforward_model_eval(
        features,
        target,
        ols_model_fn,
        train_window=150,
        test_window=50,
        purge_gap=5,
    )
    folds = result["folds"]
    assert (folds["purge_gap"] == 5).all()
    assert ((folds["train_hi"] + 5) < folds["test_lo"]).all()
    assert (folds["n_train"] == 150).all()
    assert result["purge_gap"] == 5
