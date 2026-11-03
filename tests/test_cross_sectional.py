"""Tests for atropos.cross_sectional: panel-level hypothesis killers.

Every test plants an effect whose strength is known by construction:
a perfect forecast must reach IC = 1, an anti-signal must reach -1,
the documented leg scaling must reproduce gross_mean = 0.02 exactly,
turnover must follow the entry-cost + weight-change convention, and
pure noise must die in the aggregate runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.cross_sectional import (
    cross_sectional_ic,
    cs_monotonicity,
    quantile_spread,
    run_cross_sectional_tests,
)


def _panel(values: np.ndarray, start: str = "2024-01-02") -> pd.DataFrame:
    """Wrap a 2-D array into a dated panel (rows = dates, cols = assets)."""
    dates = pd.date_range(start, periods=values.shape[0], freq="B")
    tickers = [f"TK{asset_pos:02d}" for asset_pos in range(values.shape[1])]
    return pd.DataFrame(values, index=dates, columns=tickers)


def _planted_panel(seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded 240x40 panel with a persistent planted cross-sectional edge.

    Mirrors the demo in ``atropos.cross_sectional.run_all_examples``:
    latent AR(0.85) factor z-scored per date, returns
    ``0.0005 + 0.004*z + noise``, signal ``z + 0.3*noise``.
    """
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 240, 40
    latent = np.zeros((n_dates, n_assets))
    shocks = rng.standard_normal((n_dates, n_assets))
    for date_pos in range(1, n_dates):
        latent[date_pos] = 0.85 * latent[date_pos - 1] + shocks[date_pos]
    cross_std = latent.std(axis=1, keepdims=True)
    zsignal = (latent - latent.mean(axis=1, keepdims=True)) / np.where(
        cross_std > 0.0, cross_std, 1.0
    )
    returns = (
        0.0005
        + 0.004 * zsignal
        + 0.01 * rng.standard_normal((n_dates, n_assets))
    )
    signal = zsignal + 0.3 * rng.standard_normal((n_dates, n_assets))
    return _panel(signal), _panel(returns)


def _rank_stable_pair(
    n_dates: int = 6, n_assets: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Signals keep one ordering every date; top assets earn +1%.

    Returns:
        ``(signals, returns)`` arrays where the bottom-2 assets earn
        -1% per date, the top-2 earn +1% and the middle is flat, so
        the documented leg scaling gives a gross spread of exactly
        0.02 per date.
    """
    signals = np.tile(np.arange(1.0, n_assets + 1.0), (n_dates, 1))
    asset_rank = np.arange(n_assets)
    ret_row = np.where(
        asset_rank < 2,
        -0.01,
        np.where(asset_rank >= n_assets - 2, 0.01, 0.0),
    )
    return signals, np.tile(ret_row, (n_dates, 1))


def test_ic_is_one_when_signal_equals_returns() -> None:
    gen = np.random.default_rng(3)
    values = gen.standard_normal((6, 6))
    result = cross_sectional_ic(_panel(values), _panel(values.copy()))
    assert result["mean_ic"] == pytest.approx(1.0)
    assert result["stat"] == pytest.approx(1.0)
    assert result["verdict"] == "PASS"
    assert result["n_dates"] == 6
    assert result["positive_share"] == pytest.approx(1.0)
    assert result["ic_tstat"] >= 2.0
    assert result["p_value"] == pytest.approx(0.0, abs=1e-12)
    pearson = cross_sectional_ic(
        _panel(values), _panel(values.copy()), method="pearson"
    )
    assert pearson["mean_ic"] == pytest.approx(1.0)
    assert pearson["verdict"] == "PASS"


def test_ic_is_minus_one_for_anticorrelated_panels() -> None:
    gen = np.random.default_rng(4)
    returns = gen.standard_normal((6, 6))
    result = cross_sectional_ic(_panel(-returns), _panel(returns))
    assert result["mean_ic"] == pytest.approx(-1.0)
    assert result["ic_tstat"] <= -2.0
    assert result["positive_share"] == pytest.approx(0.0)
    assert result["verdict"] == "FAIL"


def test_quantile_spread_gross_mean_and_cost_death() -> None:
    signals, returns = _rank_stable_pair()
    result = quantile_spread(_panel(signals), _panel(returns))
    # documented leg scaling: mean(top) - mean(bottom) = +0.01 - (-0.01)
    assert result["gross_mean"] == pytest.approx(0.02)
    assert result["n_dates"] == 6
    assert result["n_assets_mean"] == pytest.approx(10.0)
    # entry cost on the first date only (weights then unchanged)
    assert result["annual_turnover"] == pytest.approx(84.0)
    assert result["cost_drag_mean"] == pytest.approx(
        result["gross_mean"] - result["net_mean"]
    )
    assert result["verdict"] == "PASS"
    # huge costs flip the same edge to a cost-death FAIL
    expensive = quantile_spread(
        _panel(signals), _panel(returns), cost_bps=20000.0
    )
    assert expensive["net_mean"] < 0.0
    assert expensive["gross_mean"] == pytest.approx(0.02)
    assert expensive["verdict"] == "FAIL"


def test_quantile_spread_turnover_exact_on_rank_reversal() -> None:
    signals = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    returns = np.array(
        [[-0.01, 0.0, 0.0, 0.0, 0.01], [0.01, 0.0, 0.0, 0.0, -0.01]]
    )
    result = quantile_spread(_panel(signals), _panel(returns))
    # entry: sum(|w0|) = 2; reversal: both legs flip -> sum(|dw|) = 4
    assert result["annual_turnover"] == pytest.approx(756.0)
    assert result["gross_mean"] == pytest.approx(0.02)
    assert result["n_dates"] == 2
    assert result["n_assets_mean"] == pytest.approx(5.0)


def test_monotonicity_passes_on_linear_and_fails_on_flat() -> None:
    signals = np.tile(np.arange(1.0, 11.0), (6, 1))
    linear_returns = np.tile(np.linspace(-0.01, 0.01, 10), (6, 1))
    good = cs_monotonicity(_panel(signals), _panel(linear_returns))
    assert good["spearman"] == pytest.approx(1.0)
    assert good["stat"] == pytest.approx(1.0)
    assert good["verdict"] == "PASS"
    assert good["quantile_means"] == pytest.approx(
        [-0.0088888888, -0.0044444444, 0.0, 0.0044444444, 0.0088888888]
    )
    flat = cs_monotonicity(_panel(signals), _panel(np.full((6, 10), 0.005)))
    assert not np.isfinite(flat["spearman"])
    assert flat["verdict"] == "FAIL"
    assert flat["quantile_means"] == pytest.approx([0.005] * 5)


def test_min_assets_drops_thin_dates() -> None:
    gen = np.random.default_rng(5)
    signal = gen.standard_normal((6, 6))
    returns = gen.standard_normal((6, 6))
    signal[2, :2] = np.nan  # date 2 has only 4 valid assets left
    strict = cross_sectional_ic(_panel(signal), _panel(returns))
    assert strict["n_dates"] == 5
    relaxed = cross_sectional_ic(
        _panel(signal), _panel(returns), min_assets=4
    )
    assert relaxed["n_dates"] == 6


def test_run_cross_sectional_tests_planted_passes_noise_fails() -> None:
    signal, returns = _planted_panel(seed=11)
    report = run_cross_sectional_tests(signal, returns)
    assert report["ic"]["verdict"] == "PASS"
    assert report["spread"]["verdict"] == "PASS"
    assert report["monotonicity"]["verdict"] == "PASS"
    assert report["overall_verdict"] == "PASS"
    assert report["n_dates"] == 240
    assert report["n_assets_mean"] == pytest.approx(40.0)
    assert report["cost_bps"] == pytest.approx(10.0)
    # determinism: the analysis holds no hidden randomness
    assert run_cross_sectional_tests(signal, returns) == report

    gen = np.random.default_rng(99)
    noise_signal = _panel(gen.standard_normal((240, 40)))
    noise_returns = _panel(0.0005 + 0.01 * gen.standard_normal((240, 40)))
    noise_report = run_cross_sectional_tests(noise_signal, noise_returns)
    assert noise_report["overall_verdict"] == "FAIL"
    assert noise_report["ic"]["verdict"] == "FAIL"
    assert noise_report["spread"]["verdict"] == "FAIL"


def test_degenerate_panels_fail_gracefully() -> None:
    empty_signal = pd.DataFrame()
    empty_returns = pd.DataFrame()
    for report in (
        cross_sectional_ic(empty_signal, empty_returns),
        quantile_spread(empty_signal, empty_returns),
        cs_monotonicity(empty_signal, empty_returns),
    ):
        assert report["verdict"] == "FAIL"
        assert report["n_dates"] == 0
    combined = run_cross_sectional_tests(empty_signal, empty_returns)
    assert combined["overall_verdict"] == "FAIL"


def test_invalid_parameters_raise_value_error() -> None:
    gen = np.random.default_rng(6)
    signal = _panel(gen.standard_normal((6, 6)))
    returns = _panel(gen.standard_normal((6, 6)))
    with pytest.raises(ValueError):
        cross_sectional_ic(signal, returns, method="kendall")
    with pytest.raises(ValueError):
        cross_sectional_ic(signal, returns, min_assets=0)
    with pytest.raises(ValueError):
        quantile_spread(signal, returns, n_quantiles=1)
    with pytest.raises(ValueError):
        quantile_spread(signal, returns, cost_bps=-5.0)
    with pytest.raises(ValueError):
        cs_monotonicity(signal, returns, n_quantiles=1)
    with pytest.raises(ValueError):
        cross_sectional_ic(signal.to_numpy(), returns)  # type: ignore[arg-type]
