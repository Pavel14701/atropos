"""Tests for atropos.predictive_model_eval: purged CV and OOS verdicts."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from atropos.predictive_model_eval import (
    deflated_performance,
    feature_importance_stability,
    kernel_model_fn,
    knn_model_fn,
    ols_model_fn,
    purged_kfold_cv,
    sign_consistency,
    walkforward_model_eval,
)


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


def test_purged_kfold_rejects_bad_arguments() -> None:
    features, target = make_linear_problem(60, 0.5, 13)
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        purged_kfold_cv(features, target, n_splits=1)
    with pytest.raises(ValueError, match="too small"):
        purged_kfold_cv(features, target, n_splits=20)  # needs 4*n_splits
    with pytest.raises(ValueError, match="rows but target has"):
        purged_kfold_cv(features, target[:-3])


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
    assert {"time_third_1", "vol_tercile_low", "vol_tercile_high"} <= set(
        regimes
    )
    assert all(value == 1.0 for value in regimes.values())


def test_sign_consistency_noise_fails_near_one_half() -> None:
    generator = np.random.default_rng(32)
    predictions = generator.standard_normal(300)
    outcomes = generator.standard_normal(300)
    result = sign_consistency(predictions, outcomes)
    assert 0.35 < result["directional_accuracy"] < 0.65
    assert result["verdict"] == "FAIL"
    assert result["stable"] is False


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
