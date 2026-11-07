"""Tests for atropos.cross_sectional: panel-level hypothesis killers.

Every test plants an effect whose strength is known by construction:
a perfect forecast must reach IC = 1, an anti-signal must reach -1,
the documented leg scaling must reproduce gross_mean = 0.02 exactly,
turnover must follow the entry-cost + weight-change convention,
skipped dates must HOLD the book (P&L accrues, turnover does not),
thin samples must die as INCONCLUSIVE rather than a spurious PASS,
and pure noise must die in the aggregate runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.cross_sectional import (
    _align_panels,
    _date_quantiles,
    cross_sectional_ic,
    cs_monotonicity,
    quantile_spread,
    run_all_examples,
    run_cross_sectional_tests,
)
from scipy import stats


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
    result = cross_sectional_ic(
        _panel(values), _panel(values.copy()), min_dates=2
    )
    assert result["mean_ic"] == pytest.approx(1.0)
    assert result["stat"] == pytest.approx(1.0)
    assert result["n_dates"] == 6
    assert result["positive_share"] == pytest.approx(1.0)
    # IC = 1 on EVERY date -> zero dispersion -> t = +inf; an
    # infinite t-stat must not crown a PASS.
    assert np.isinf(result["ic_tstat"])
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["p_value"] == pytest.approx(0.0, abs=1e-12)
    pearson = cross_sectional_ic(
        _panel(values), _panel(values.copy()), method="pearson", min_dates=2
    )
    assert pearson["mean_ic"] == pytest.approx(1.0)
    # raw Pearson carries fp noise (IC = 1 - 1e-16-ish), so its t is
    # huge but FINITE: the edge direction is what matters here
    assert pearson["ic_tstat"] >= 2.0


def test_ic_is_minus_one_for_anticorrelated_panels() -> None:
    gen = np.random.default_rng(4)
    returns = gen.standard_normal((6, 6))
    result = cross_sectional_ic(
        _panel(-returns), _panel(returns), min_dates=2
    )
    assert result["mean_ic"] == pytest.approx(-1.0)
    # constant IC = -1 -> t = -inf (non-finite) -> INCONCLUSIVE, but
    # the direction of the edge is still reported.
    assert result["ic_tstat"] == float("-inf")
    assert result["positive_share"] == pytest.approx(0.0)
    assert result["verdict"] == "INCONCLUSIVE"


def test_quantile_spread_gross_mean_and_cost_death() -> None:
    signals, returns = _rank_stable_pair()
    result = quantile_spread(_panel(signals), _panel(returns), min_dates=2)
    # documented leg scaling: mean(top) - mean(bottom) = 0.01 - (-0.01)
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
        _panel(signals), _panel(returns), cost_bps=20000.0, min_dates=2
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


def test_quantile_spread_holds_book_through_skipped_dates() -> None:
    """Skipped dates HOLD: P&L accrues on the old book, turnover stays 0.

    Panel layout (6 assets, Q = 2 -> long assets 4-6, short 1-3):

    - date 0: only 2 valid assets (below min_assets) but the book is
      still flat -> no return observation at all;
    - date 1: traded, gross = mean(top) - mean(bottom) = 0.02;
    - date 2: fully tied signal (degenerate cross-section, an extreme
      bucket is empty) -> HOLD: gross = sum(w * r) with the NaN leg
      return counted as 0.0, turnover 0;
    - date 3: thin date below min_assets -> HOLD again;
    - date 4: traded, same ranking as date 1 -> weights unchanged,
      turnover 0.
    """
    signals = np.array(
        [
            [np.nan, 1.0, 2.0, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    returns = np.array(
        [
            [np.nan, -0.01, -0.01, np.nan, np.nan, np.nan],
            [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
            [-0.02, -0.02, -0.02, 0.02, 0.02, np.nan],
            [-0.02, -0.02, np.nan, np.nan, np.nan, np.nan],
            [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
        ]
    )
    result = quantile_spread(
        _panel(signals), _panel(returns), n_quantiles=2, min_dates=2
    )
    # the flat lead date contributes NO observation; the two skipped
    # dates in the middle DO accrue the held book's return
    assert result["n_dates"] == 4
    held_tied = (1 / 3) * (0.02 + 0.02 + 0.0) + 3 * (-1 / 3) * (-0.02)
    held_thin = (-1 / 3) * (-0.02) + (-1 / 3) * (-0.02)
    assert result["gross_mean"] == pytest.approx(
        float(np.mean([0.02, held_tied, held_thin, 0.02]))
    )
    # turnover: entry 2.0 on date 1 only; holds cost nothing and the
    # final trade re-uses the same weights -> 0
    assert result["annual_turnover"] == pytest.approx(0.5 * 252.0)
    assert result["cost_drag_mean"] == pytest.approx(0.002 / 4.0)
    assert result["net_mean"] == pytest.approx(
        result["gross_mean"] - 0.002 / 4.0
    )
    # n_assets_mean averages TRADED dates only (both have 6 valid)
    assert result["n_assets_mean"] == pytest.approx(6.0)
    assert result["verdict"] == "PASS"


def test_monotonicity_passes_on_linear_and_fails_on_flat() -> None:
    signals = np.tile(np.arange(1.0, 11.0), (6, 1))
    linear_returns = np.tile(np.linspace(-0.01, 0.01, 10), (6, 1))
    good = cs_monotonicity(
        _panel(signals), _panel(linear_returns), min_dates=2
    )
    assert good["spearman"] == pytest.approx(1.0)
    assert good["stat"] == pytest.approx(1.0)
    assert good["verdict"] == "PASS"
    assert good["quantile_means"] == pytest.approx(
        [-0.0088888888, -0.0044444444, 0.0, 0.0044444444, 0.0088888888]
    )
    flat = cs_monotonicity(
        _panel(signals), _panel(np.full((6, 10), 0.005)), min_dates=2
    )
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


def test_min_dates_gates_thin_planted_edge_to_inconclusive() -> None:
    """Two effective dates with a planted edge must not PASS/FAIL.

    With the default ``min_dates = 20`` every killer reports
    INCONCLUSIVE on a 2-date panel (finite t-statistics, so the gate
    is the sample size itself), and the aggregator propagates it.
    """
    planted_signal, planted_returns = _planted_panel(seed=11)
    short_signal = planted_signal.iloc[:2]
    short_returns = planted_returns.iloc[:2]
    for report in (
        cross_sectional_ic(short_signal, short_returns),
        quantile_spread(short_signal, short_returns),
        cs_monotonicity(short_signal, short_returns),
    ):
        assert report["n_dates"] == 2
        assert report["verdict"] == "INCONCLUSIVE"
    combined = run_cross_sectional_tests(short_signal, short_returns)
    assert combined["overall_verdict"] == "INCONCLUSIVE"


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


def test_p_value_uses_student_t_not_normal() -> None:
    """``p_value`` must be the t-tail with df = n - 1, not normal."""
    signal, returns = _planted_panel(seed=11)
    ic = cross_sectional_ic(signal, returns)
    expected_t = 2.0 * stats.t.sf(abs(ic["ic_tstat"]), df=ic["n_dates"] - 1)
    assert ic["p_value"] == pytest.approx(expected_t)
    spread = quantile_spread(signal, returns)
    assert spread["p_value"] == pytest.approx(
        2.0 * stats.t.sf(abs(spread["net_tstat"]), df=spread["n_dates"] - 1)
    )
    # a 2-observation spread has df = 1: the Student-t tail at
    # t = 17 is ~0.037 (Cauchy-like), while the old normal tail would
    # be ~1e-64 - the difference is not a rounding artefact
    reversal_signals = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    reversal_returns = np.array(
        [[-0.01, 0.0, 0.0, 0.0, 0.01], [0.01, 0.0, 0.0, 0.0, -0.01]]
    )
    reversal = quantile_spread(
        _panel(reversal_signals), _panel(reversal_returns)
    )
    assert reversal["net_tstat"] == pytest.approx(17.0)
    assert reversal["p_value"] == pytest.approx(
        2.0 * stats.t.sf(abs(reversal["net_tstat"]), df=1)
    )
    assert reversal["p_value"] > 0.01


def test_ic_return_series_flag() -> None:
    signal, returns = _planted_panel(seed=11)
    default = cross_sectional_ic(signal, returns)
    assert "ic_series" not in default
    with_series = cross_sectional_ic(signal, returns, return_series=True)
    series = with_series["ic_series"]
    assert len(series) == with_series["n_dates"] == default["n_dates"]
    assert float(np.mean(series)) == pytest.approx(default["mean_ic"])


def test_date_quantiles_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        _date_quantiles(np.array([], dtype=float), 3)


def test_run_cross_sectional_tests_validates_cost_bps() -> None:
    signal, returns = _planted_panel(seed=11)
    for bad_cost in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            run_cross_sectional_tests(signal, returns, cost_bps=bad_cost)


def test_custom_thresholds_change_verdicts() -> None:
    """Hardcoded cutoffs are parameters: raising them flips verdicts."""
    signal, returns = _planted_panel(seed=11)
    baseline = run_cross_sectional_tests(signal, returns)
    assert baseline["overall_verdict"] == "PASS"
    strict_t = run_cross_sectional_tests(signal, returns, tstat_cutoff=1000.0)
    assert strict_t["ic"]["verdict"] == "FAIL"
    assert strict_t["spread"]["verdict"] == "FAIL"
    assert strict_t["overall_verdict"] == "FAIL"
    strict_share = run_cross_sectional_tests(
        signal, returns, positive_share_min=0.99
    )
    assert strict_share["ic"]["verdict"] == "FAIL"


def test_mono_threshold_parameter() -> None:
    """A 0.9 Spearman panel passes 0.5 but dies against 0.95."""
    signals = np.tile(np.arange(1.0, 11.0), (6, 1))
    # bucket means rise except one inversion -> Spearman(bucket, mean)
    # = 0.9 by construction
    row = np.array(
        [-0.008, -0.008, -0.004, -0.004, 0.004, 0.004, 0.0, 0.0, 0.008, 0.008]
    )
    returns = np.tile(row, (6, 1))
    default = cs_monotonicity(_panel(signals), _panel(returns), min_dates=2)
    assert default["spearman"] == pytest.approx(0.9)
    assert default["verdict"] == "PASS"
    strict = cs_monotonicity(
        _panel(signals), _panel(returns), min_dates=2, mono_spearman_min=0.95
    )
    assert strict["verdict"] == "FAIL"


def test_invalid_parameters_raise_value_error() -> None:
    gen = np.random.default_rng(6)
    signal = _panel(gen.standard_normal((6, 6)))
    returns = _panel(gen.standard_normal((6, 6)))
    with pytest.raises(ValueError):
        cross_sectional_ic(signal, returns, method="kendall")
    with pytest.raises(ValueError):
        cross_sectional_ic(signal, returns, min_assets=0)
    with pytest.raises(ValueError):
        cross_sectional_ic(signal, returns, min_dates=0)
    with pytest.raises(ValueError):
        quantile_spread(signal, returns, n_quantiles=1)
    with pytest.raises(ValueError):
        quantile_spread(signal, returns, cost_bps=-5.0)
    with pytest.raises(ValueError):
        quantile_spread(signal, returns, min_dates=0)
    with pytest.raises(ValueError):
        cs_monotonicity(signal, returns, n_quantiles=1)
    with pytest.raises(ValueError):
        cs_monotonicity(signal, returns, min_dates=0)
    with pytest.raises(ValueError):
        run_cross_sectional_tests(signal, returns, min_dates=0)
    with pytest.raises(ValueError):
        cross_sectional_ic(signal.to_numpy(), returns)  # type: ignore[arg-type]


def test_verdict_thresholds_are_validated() -> None:
    """tstat_cutoff / positive_share_min / mono_spearman_min ranges.

    A zero/negative t cutoff, a share outside (0, 1) or a Spearman
    floor outside [-1, 1] are meaningless thresholds and must raise
    before any panel is aligned.
    """
    gen = np.random.default_rng(7)
    signal = _panel(gen.standard_normal((6, 6)))
    returns = _panel(gen.standard_normal((6, 6)))
    with pytest.raises(ValueError, match="tstat_cutoff must be > 0"):
        cross_sectional_ic(signal, returns, tstat_cutoff=0.0)
    with pytest.raises(ValueError, match="tstat_cutoff must be > 0"):
        cross_sectional_ic(signal, returns, tstat_cutoff=float("nan"))
    with pytest.raises(ValueError, match="tstat_cutoff must be > 0"):
        quantile_spread(signal, returns, tstat_cutoff=-2.0)
    for bad_share in (0.0, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="positive_share_min"):
            cross_sectional_ic(signal, returns, positive_share_min=bad_share)
    for bad_mono in (1.01, -1.01, float("nan")):
        with pytest.raises(
            ValueError, match=r"mono_spearman_min must be in \[-1, 1\]"
        ):
            cs_monotonicity(signal, returns, mono_spearman_min=bad_mono)
    # the orchestrator validates all three families up front
    with pytest.raises(ValueError, match="tstat_cutoff must be > 0"):
        run_cross_sectional_tests(signal, returns, tstat_cutoff=0.0)
    with pytest.raises(ValueError, match="positive_share_min"):
        run_cross_sectional_tests(signal, returns, positive_share_min=0.0)
    with pytest.raises(ValueError, match=r"mono_spearman_min"):
        run_cross_sectional_tests(signal, returns, mono_spearman_min=2.0)
    # boundary values are legal thresholds
    boundary = cs_monotonicity(
        signal, returns, min_dates=1, mono_spearman_min=1.0
    )
    assert -1.0 <= boundary["spearman"] <= 1.0 or np.isnan(
        boundary["spearman"]
    )


def test_align_panels_rejects_scrambled_plain_index() -> None:
    """A scrambled non-DatetimeIndex raises instead of reordering.

    Regression: the aligner sorted only DatetimeIndex inputs and
    reindexed plain indexes in the order GIVEN, so a scrambled file
    silently produced order-dependent turnover (420 vs 504 on the
    committed demo panel, depending on file row order).  Old
    behaviour: the exact call below ran without error and returned
    misaligned frames.
    """
    values = np.arange(24, dtype=float).reshape(6, 4)
    order = [3, 0, 4, 1, 5, 2]
    signal = pd.DataFrame(values, index=pd.Index(order))
    returns = pd.DataFrame(values + 1.0, index=pd.Index(order))
    with pytest.raises(ValueError, match="sorted ascending"):
        _align_panels(signal, returns, 2)
    # a monotonic plain index still aligns (no dates needed)
    signal_ok = pd.DataFrame(values, index=pd.Index(sorted(order)))
    returns_ok = pd.DataFrame(values + 1.0, index=pd.Index(sorted(order)))
    signal_frame, returns_frame, valid_positions, _ = _align_panels(
        signal_ok, returns_ok, 2
    )
    assert len(signal_frame) == len(returns_frame) == 6
    assert valid_positions == [0, 1, 2, 3, 4, 5]


def test_demo_run_all_examples_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module demo runs clean: planted edge PASSes, noise FAILs."""
    run_all_examples()
    out = capsys.readouterr().out
    assert "cross_sectional demo" in out
    assert "[planted edge] overall=PASS" in out
    assert "[pure noise] overall=FAIL" in out
    assert "ic           :" in out
    assert "quantile_spread:" in out
    assert "monotonicity :" in out
