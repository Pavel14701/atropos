"""API contract tests for the atropos package.

Locks three public contracts of every ``atropos/*.py`` module:

* ``__all__`` purity: the demo entry point ``run_all_examples`` is not
  part of the public export list, and every exported name resolves via
  ``getattr``.
* Result-dict stability: the key sets of the representative public
  functions are snapshotted as constants; adding or removing a key
  without updating this file breaks the build.
* Input immutability and documented degenerate-input behaviour.
"""

from __future__ import annotations

import importlib
import math
import pkgutil
from collections.abc import Callable, Mapping
from typing import Any

import atropos
import numpy as np
import pandas as pd
import pytest
from atropos.bootstrap_ci import block_bootstrap_ci, percentile_ci
from atropos.cost_model import apply_proportional_costs, cost_death_test
from atropos.cross_sectional import run_cross_sectional_tests
from atropos.evaluation_metrics import (
    deflated_sharpe,
    max_drawdown,
    sharpe_ratio,
)
from atropos.incremental_regression import incremental_test
from atropos.monte_carlo import max_drawdown_distribution
from atropos.walkforward_framework import walkforward

MODULE_NAMES: tuple[str, ...] = tuple(
    sorted(name for _, name, _ in pkgutil.iter_modules(atropos.__path__))
)
"""Every importable ``atropos.*`` submodule (private ones included)."""


# ---------------------------------------------------------------------------
# 1. __all__ purity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module_name",
    MODULE_NAMES,
)
def test_run_all_examples_is_not_exported(module_name: str) -> None:
    """The module demo must stay out of the public API surface."""
    module = importlib.import_module(f"atropos.{module_name}")
    exported = list(getattr(module, "__all__", []))
    assert "run_all_examples" not in exported, exported


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_exported_name_exists(module_name: str) -> None:
    """Each ``__all__`` entry must resolve to a module attribute."""
    module = importlib.import_module(f"atropos.{module_name}")
    exported = list(getattr(module, "__all__", []))
    assert exported, f"atropos.{module_name} has an empty __all__"
    for name in exported:
        assert hasattr(module, name), (module_name, name)


# ---------------------------------------------------------------------------
# 2. result-dict key snapshots
# ---------------------------------------------------------------------------
def _sharpe_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    return sharpe_ratio(returns, seed=0)


def _block_bootstrap_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    series = 0.001 + 0.01 * rng.standard_normal(300)
    return block_bootstrap_ci(series, n_boot=50, block=10, seed=1)


def _cost_death_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    gross = 0.001 + 0.01 * rng.standard_normal(200)
    positions = (gross > 0.0).astype(float)
    return cost_death_test(gross, positions, 5.0, n_boot=50, seed=2)


def _mdd_distribution_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    return max_drawdown_distribution(returns, n_sim=50, block=10, seed=3)


def _walkforward_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(400)
    frame = pd.DataFrame(
        {"signal": signal, "outcome": 0.4 * signal + rng.standard_normal(400)}
    )
    return walkforward(frame, train_window=200, test_window=50, step=50)


def _cross_sectional_report() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((60, 12))
    signal_panel = pd.DataFrame(latent)
    returns_panel = pd.DataFrame(
        0.004 * latent + 0.01 * rng.standard_normal((60, 12))
    )
    report = run_cross_sectional_tests(
        signal_panel, returns_panel, min_dates=10
    )
    return dict(report)


def _cs_ic_result() -> Mapping[str, Any]:
    result: Mapping[str, Any] = _cross_sectional_report()["ic"]
    return result


def _cs_spread_result() -> Mapping[str, Any]:
    result: Mapping[str, Any] = _cross_sectional_report()["spread"]
    return result


def _cs_mono_result() -> Mapping[str, Any]:
    result: Mapping[str, Any] = _cross_sectional_report()["monotonicity"]
    return result


def _incremental_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    controls = rng.standard_normal((300, 2))
    new_signal = rng.standard_normal(300)
    target = (
        0.3 * new_signal + 0.5 * controls[:, 0] + rng.standard_normal(300)
    )
    return incremental_test(
        target, pd.DataFrame(controls, columns=["c0", "c1"]), new_signal
    )


def _deflated_sharpe_result() -> Mapping[str, Any]:
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    return deflated_sharpe(returns, n_trials=10, seed=4)


RESULT_BUILDERS: dict[str, Callable[[], Mapping[str, Any]]] = {
    "sharpe_ratio": _sharpe_result,
    "block_bootstrap_ci": _block_bootstrap_result,
    "cost_death_test": _cost_death_result,
    "max_drawdown_distribution": _mdd_distribution_result,
    "walkforward": _walkforward_result,
    "run_cross_sectional_tests": _cross_sectional_report,
    "run_cross_sectional_tests.ic": _cs_ic_result,
    "run_cross_sectional_tests.spread": _cs_spread_result,
    "run_cross_sectional_tests.mono": _cs_mono_result,
    "incremental_test": _incremental_result,
    "deflated_sharpe": _deflated_sharpe_result,
}

EXPECTED_KEY_SETS: dict[str, frozenset[str]] = {
    "sharpe_ratio": frozenset(
        {"sr", "sr_ci", "sr_per_period", "ci_method", "n", "rf"}
    ),
    "block_bootstrap_ci": frozenset(
        {
            "point_estimate",
            "ci_low",
            "ci_high",
            "se",
            "n",
            "B",
            "block",
            "method",
        }
    ),
    "cost_death_test": frozenset(
        {
            "net_mean",
            "p_value",
            "verdict",
            "net_sharpe",
            "gross_sharpe",
            "cost_drag_mean",
            "ci_low",
            "ci_high",
            "n_obs",
            "breakeven_cost_bps",
            "annual_turnover",
            "n_changes",
        }
    ),
    "max_drawdown_distribution": frozenset(
        {
            "mdd_p5",
            "mdd_p50",
            "mdd_p95",
            "n_valid_paths",
            "n_dropped_paths",
            "n_sim",
            "n",
            "block",
            "note",
        }
    ),
    "walkforward": frozenset(
        {
            "wf_t",
            "wf_direction_consistency",
            "wf_shrink_ratio",
            "verdict",
            "n_folds",
            "in_sample_t",
            "folds",
            "warnings",
            "mode",
            "purge_gap",
            "n_pnl_unique",
            "overlap_factor",
        }
    ),
    "run_cross_sectional_tests": frozenset(
        {
            "ic",
            "spread",
            "monotonicity",
            "overall_verdict",
            "n_dates",
            "n_assets_mean",
            "cost_bps",
        }
    ),
    "run_cross_sectional_tests.ic": frozenset(
        {
            "stat",
            "p_value",
            "verdict",
            "mean_ic",
            "ic_std",
            "ic_tstat",
            "ic_ir",
            "positive_share",
            "n_dates",
            "method",
        }
    ),
    "run_cross_sectional_tests.spread": frozenset(
        {
            "stat",
            "p_value",
            "verdict",
            "gross_mean",
            "net_mean",
            "cost_drag_mean",
            "gross_sharpe",
            "net_sharpe",
            "net_tstat",
            "annual_turnover",
            "n_dates",
            "n_assets_mean",
            "n_quantiles",
            "cost_bps",
        }
    ),
    "run_cross_sectional_tests.mono": frozenset(
        {
            "stat",
            "p_value",
            "verdict",
            "quantile_means",
            "spearman",
            "n_dates",
            "n_quantiles",
        }
    ),
    "incremental_test": frozenset(
        {
            "t_new",
            "delta_r2",
            "verdict",
            "coef_new",
            "p_new",
            "r2_restricted",
            "r2_full",
            "n",
            "robust",
        }
    ),
    "deflated_sharpe": frozenset(
        {
            "dsr",
            "p_value",
            "sr_annual",
            "sr_benchmark_annual",
            "sr_var_annual",
            "sr_var_source",
            "skew",
            "kurtosis",
            "n_trials",
            "n",
        }
    ),
}


def test_snapshot_registry_is_complete() -> None:
    """Every builder has a snapshot and every snapshot has a builder."""
    assert set(RESULT_BUILDERS) == set(EXPECTED_KEY_SETS)


@pytest.mark.parametrize("label", sorted(RESULT_BUILDERS))
def test_result_dict_keys_are_frozen(label: str) -> None:
    """Adding or removing a result key must be a deliberate act."""
    result = RESULT_BUILDERS[label]()
    expected = EXPECTED_KEY_SETS[label]
    assert frozenset(result) == expected, (
        f"{label}: added={sorted(frozenset(result) - expected)} "
        f"removed={sorted(expected - frozenset(result))}"
    )


# ---------------------------------------------------------------------------
# 3. input immutability
# ---------------------------------------------------------------------------
def test_sharpe_ratio_does_not_mutate_input() -> None:
    """sharpe_ratio must treat the return series as read-only."""
    rng = np.random.default_rng(3)
    returns = 0.0005 + 0.01 * rng.standard_normal(300)
    original = returns.copy()
    sharpe_ratio(returns, seed=0)
    assert np.array_equal(returns, original)


def test_block_bootstrap_ci_does_not_mutate_input() -> None:
    """block_bootstrap_ci must not reorder or edit the series."""
    rng = np.random.default_rng(3)
    series = 0.001 + 0.01 * rng.standard_normal(300)
    original = series.copy()
    block_bootstrap_ci(series, n_boot=50, block=10, seed=1)
    assert np.array_equal(series, original)


def test_apply_proportional_costs_does_not_mutate_inputs() -> None:
    """Cost application must not touch gross returns or positions."""
    rng = np.random.default_rng(3)
    gross = 0.001 + 0.01 * rng.standard_normal(200)
    positions = (gross > 0.0).astype(float)
    gross_copy = gross.copy()
    positions_copy = positions.copy()
    apply_proportional_costs(gross, positions, 10.0)
    assert np.array_equal(gross, gross_copy)
    assert np.array_equal(positions, positions_copy)


def test_incremental_test_does_not_mutate_inputs() -> None:
    """Alignment must not rewrite target, controls, or signal."""
    rng = np.random.default_rng(3)
    controls = rng.standard_normal((300, 2))
    new_signal = rng.standard_normal(300)
    target = (
        0.3 * new_signal + 0.5 * controls[:, 0] + rng.standard_normal(300)
    )
    controls_df = pd.DataFrame(controls, columns=["c0", "c1"])
    controls_before = controls_df.copy()
    signal_copy = new_signal.copy()
    target_copy = target.copy()
    incremental_test(target, controls_df, new_signal)
    assert np.array_equal(new_signal, signal_copy)
    assert np.array_equal(target, target_copy)
    assert controls_df.equals(controls_before)


# ---------------------------------------------------------------------------
# 4. degenerate inputs: assert exactly the documented behaviour
# ---------------------------------------------------------------------------
def test_block_bootstrap_ci_small_sample_scaffold() -> None:
    """< 8 finite observations: NaN bounds, method 'insufficient_data'."""
    result = block_bootstrap_ci([1.0, 2.0, float("nan")], seed=0)
    assert result["method"] == "insufficient_data"
    assert result["n"] == 2
    assert result["B"] == 0
    assert math.isnan(result["point_estimate"]) is False
    assert result["point_estimate"] == 1.5
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])
    assert math.isnan(result["se"])


def test_block_bootstrap_ci_empty_input() -> None:
    """No finite data at all: NaN point estimate, honest scaffold."""
    result = block_bootstrap_ci([], seed=0)
    assert result["method"] == "insufficient_data"
    assert result["n"] == 0
    assert math.isnan(result["point_estimate"])


def test_percentile_ci_empty_input() -> None:
    """Nothing finite remains: the documented (nan, nan) tuple."""
    low, high = percentile_ci([])
    assert math.isnan(low)
    assert math.isnan(high)


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"n_boot": 0}, "n_boot < 1"),
        ({"block": 0}, "block < 1"),
        ({"ci_level": 0.0}, "ci_level == 0"),
        ({"ci_level": 1.0}, "ci_level == 1"),
    ],
)
def test_block_bootstrap_ci_rejects_bad_constants(
    kwargs: dict[str, Any], label: str
) -> None:
    """Constant validation raises ValueError before touching data."""
    rng = np.random.default_rng(0)
    series = rng.standard_normal(50)
    with pytest.raises(ValueError, match=label.split(" ", maxsplit=1)[0]):
        block_bootstrap_ci(series, **kwargs)


def _valid_walkforward_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(60)
    return pd.DataFrame(
        {"signal": signal, "outcome": 0.4 * signal + rng.standard_normal(60)}
    )


def test_walkforward_rejects_non_dataframe() -> None:
    """Walkforward is DataFrame-only by contract (TypeError)."""
    with pytest.raises(TypeError, match="DataFrame"):
        walkforward(np.zeros((60, 2)))  # type: ignore[arg-type]


def test_walkforward_rejects_missing_outcome_column() -> None:
    """A missing outcome_col is a ValueError, not a KeyError."""
    frame = _valid_walkforward_frame().drop(columns=["outcome"])
    with pytest.raises(ValueError, match="outcome_col"):
        walkforward(frame, train_window=30, test_window=10, step=10)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"train_window": 9}, "train_window"),
        ({"test_window": 0}, "test_window"),
        ({"step": 0}, "step"),
        ({"purge_gap": -1}, "purge_gap"),
        ({"train_window": 55, "test_window": 10}, "too small"),
    ],
)
def test_walkforward_rejects_bad_constants(
    overrides: dict[str, Any], match: str
) -> None:
    """Window/step/purge constants are validated with ValueError."""
    defaults: dict[str, Any] = {
        "train_window": 30,
        "test_window": 10,
        "step": 10,
        "purge_gap": 0,
    }
    with pytest.raises(ValueError, match=match):
        walkforward(_valid_walkforward_frame(), **{**defaults, **overrides})


def test_walkforward_rejects_calendar_columns() -> None:
    """Calendar-looking feature names raise before any fitting."""
    frame = _valid_walkforward_frame()
    frame["year"] = np.arange(len(frame), dtype=float) // 250
    with pytest.raises(ValueError, match="calendar"):
        walkforward(frame, train_window=30, test_window=10, step=10)


def test_sharpe_ratio_small_sample_has_no_ci() -> None:
    """N < 8: no bootstrap CI, honest 'insufficient_data' method."""
    result = sharpe_ratio([0.01, 0.02, -0.005, 0.011, 0.004, -0.002], seed=0)
    assert result["ci_method"] == "insufficient_data"
    assert result["n"] == 6
    low, high = result["sr_ci"]
    assert math.isnan(low)
    assert math.isnan(high)


def test_max_drawdown_empty_curve_scaffold() -> None:
    """Empty equity curve: NaN drawdown, duration 0, nothing underwater."""
    result = max_drawdown([])
    assert math.isnan(result["mdd"])
    assert result["mdd_duration"] == 0
    assert result["peak_index"] == -1
    assert result["trough_index"] == -1
    assert result["still_underwater"] is False
    assert result["n"] == 0


def test_max_drawdown_rejects_multidimensional_input() -> None:
    """2-D input must raise instead of being silently flattened."""
    with pytest.raises(ValueError, match="1-D"):
        max_drawdown(np.ones((3, 4)))


def test_max_drawdown_rejects_non_positive_equity() -> None:
    """Multiplicative drawdown needs a strictly positive curve."""
    with pytest.raises(ValueError, match="strictly positive"):
        max_drawdown([1.0, 1.1, 0.0, 1.2])
