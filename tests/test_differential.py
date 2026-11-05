"""Differential tests: two independent paths must give one answer.

Each test computes the same quantity twice -- once through a public
Atropos API and once through a manual reimplementation (or through a
sibling module that shares the underlying contract, e.g. the single
``block_indices`` index domain) -- and asserts the answers agree.  A
divergence means the modules drifted apart, not that one specific
input broke: this catches classes of bugs, not single bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos._utils import apply_stat_rowwise, block_indices
from atropos.bootstrap_ci import block_bootstrap_ci, percentile_ci
from atropos.cost_model import apply_proportional_costs, cost_death_test
from atropos.evaluation_metrics import max_drawdown, sharpe_ratio
from atropos.monte_carlo import (
    bootstrap_resample,
    max_drawdown_distribution,
    sharpe_distribution,
)
from atropos.null_control_suite import _pearson_r
from atropos.predictive_model_eval import _pearson
from atropos.track_record import _max_drawdown
from atropos.walkforward_framework import check_no_peeking, walkforward


def test_block_indices_matches_independent_formula() -> None:
    """block_indices equals a hand-written circular moving-block formula.

    The manual path re-derives the clamp, the block count, the wrap
    around and the truncation from scratch, with its own identically
    seeded generator, so any change in the RNG protocol breaks here.
    """
    for n_obs, n_sims, block in ((37, 9, 8), (37, 5, 100), (64, 6, 1)):
        drawn = block_indices(
            n_obs, n_sims, block, np.random.default_rng(101)
        )
        effective_block = min(max(block, 1), n_obs)
        n_blocks = int(np.ceil(n_obs / effective_block))
        starts = np.random.default_rng(101).integers(
            0, n_obs, size=(n_sims, n_blocks)
        )
        manual = (
            (starts[:, :, None] + np.arange(effective_block)) % n_obs
        ).reshape(n_sims, n_blocks * effective_block)[:, :n_obs]
        assert drawn.shape == (n_sims, n_obs)
        assert np.issubdtype(drawn.dtype, np.integer)
        assert np.array_equal(drawn, manual)


def test_block_bootstrap_ci_equivalent_to_bootstrap_resample() -> None:
    """Same seed, same data: bootstrap_ci and monte_carlo agree bitwise.

    ``bootstrap_ci.block_bootstrap_ci`` and
    ``monte_carlo.bootstrap_resample`` are two public doors into the
    same ``block_indices`` machinery; both clean non-finite entries the
    same way, so their bootstrap distributions must coincide exactly.
    """
    data = np.random.default_rng(11).normal(0.001, 0.01, 300)
    data[[5, 77, 201]] = np.nan
    draws = bootstrap_resample(data, np.mean, n_boot=250, block=13, seed=123)
    interval = block_bootstrap_ci(
        data, n_boot=250, block=13, stat=np.mean, seed=123
    )
    expected_bounds = percentile_ci(draws, 0.95)
    assert interval["ci_low"] == expected_bounds[0]
    assert interval["ci_high"] == expected_bounds[1]
    assert interval["se"] == float(np.std(draws, ddof=1))
    assert interval["point_estimate"] == float(
        np.mean(data[np.isfinite(data)])
    )


def test_sharpe_ratio_ci_equivalent_to_block_indices_recompute() -> None:
    """sharpe_ratio's CI is exactly the block_indices SR bootstrap.

    Recomputes the whole CI from ``block_indices`` + numpy by mirroring
    the documented arithmetic (excess returns, ddof=1, sqrt(periods)
    rescale, finite filter, tail=(1-ci_level)/2) on the same seed.
    """
    returns = np.random.default_rng(3).normal(0.0004, 0.011, 200)
    n_boot, block_len, periods, rf, seed = 400, 17, 252, 0.03, 5
    result = sharpe_ratio(
        returns,
        rf=rf,
        periods=periods,
        n_boot=n_boot,
        block=block_len,
        seed=seed,
    )
    excess = returns - float(rf) / float(periods)
    idx = block_indices(200, n_boot, block_len, np.random.default_rng(seed))
    sds = excess[idx].std(axis=1, ddof=1)
    mus = excess[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(sds > 0, mus / sds, np.nan) * float(np.sqrt(periods))
    boot = boot[np.isfinite(boot)]
    tail = (1.0 - 0.95) / 2.0
    expected_ci = (
        float(np.quantile(boot, tail)),
        float(np.quantile(boot, 1.0 - tail)),
    )
    stdev = float(np.std(excess, ddof=1))
    expected_pp = float(np.mean(excess) / stdev)
    assert result["ci_method"] == "moving_block"
    assert result["sr_ci"] == expected_ci
    assert result["sr_per_period"] == expected_pp
    assert result["sr"] == expected_pp * float(np.sqrt(periods))


def test_max_drawdown_matches_manual_equity_formula() -> None:
    """max_drawdown equals the manual equity/running-peak minus one."""
    curve = np.asarray([1.0, 3.0, 2.0, 4.0])
    assert max_drawdown(curve)["mdd"] == pytest.approx(2.0 / 3.0 - 1.0)
    returns = np.random.default_rng(25).normal(0.0003, 0.012, 300)
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    manual = float(np.min(equity / peak - 1.0))
    assert max_drawdown(equity)["mdd"] == manual


def test_max_drawdown_distribution_matches_per_path_metric() -> None:
    """monte_carlo's mdd percentiles equal per-path evaluation_metrics.

    Rebuilds every resampled path with ``block_indices`` on the same
    seed and scores each compounded equity curve through
    ``evaluation_metrics.max_drawdown``; the p5/p50/p95 of those
    per-path values must equal ``max_drawdown_distribution`` exactly.
    """
    returns = np.random.default_rng(21).normal(0.0003, 0.01, 260)
    n_sim, block_len, seed = 120, 13, 77
    result = max_drawdown_distribution(
        returns, n_sim=n_sim, block=block_len, seed=seed
    )
    idx_rows = block_indices(
        260, n_sim, block_len, np.random.default_rng(seed)
    )
    per_path = np.asarray(
        [
            max_drawdown(np.cumprod(1.0 + returns[row]))["mdd"]
            for row in idx_rows
        ]
    )
    expected = np.quantile(per_path, [0.05, 0.50, 0.95])
    assert result["n_dropped_paths"] == 0
    assert result["mdd_p5"] == expected[0]
    assert result["mdd_p50"] == expected[1]
    assert result["mdd_p95"] == expected[2]


def test_sharpe_distribution_matches_manual_percentiles() -> None:
    """sharpe_distribution's percentiles equal a manual recompute."""
    returns = np.random.default_rng(21).normal(0.0003, 0.01, 260)
    n_sim, block_len, periods, seed = 120, 13, 252, 78
    result = sharpe_distribution(
        returns,
        n_sim=n_sim,
        block=block_len,
        periods=periods,
        seed=seed,
    )
    idx_rows = block_indices(
        260, n_sim, block_len, np.random.default_rng(seed)
    )
    sd_rows = returns[idx_rows].std(axis=1, ddof=1)
    mu_rows = returns[idx_rows].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr_rows = np.where(sd_rows > 0, mu_rows / sd_rows, np.nan) * float(
            np.sqrt(periods)
        )
    sr_rows = sr_rows[np.isfinite(sr_rows)]
    expected = np.quantile(sr_rows, [0.05, 0.50, 0.95])
    assert result["sr_p5"] == expected[0]
    assert result["sr_p50"] == expected[1]
    assert result["sr_p95"] == expected[2]


def test_apply_proportional_costs_matches_manual_shifted_diff() -> None:
    """Net returns equal gross minus cost_bps/1e4 * |diff| shifted by 1.

    The first element carries no turnover (no previous position), so it
    must come through untouched; every later element is charged against
    ``|pos[t] - pos[t-1]|``.
    """
    rng = np.random.default_rng(31)
    n_obs = 300
    gross = rng.normal(0.0006, 0.01, n_obs)
    positions = np.zeros(n_obs)
    positions[1:] = np.where(gross[:-1] > 0, 1.0, -1.0)
    cost_bps = 12.5
    net = apply_proportional_costs(gross, positions, cost_bps)
    manual = gross.copy()
    manual[1:] = gross[1:] - cost_bps / 1e4 * np.abs(np.diff(positions))
    assert np.array_equal(net, manual)

    report = cost_death_test(
        gross, positions, cost_bps, n_boot=300, block=21, seed=13
    )
    assert report["net_mean"] == float(np.mean(net))
    assert report["cost_drag_mean"] == float(np.mean(gross - net))


def _fit_noop(_train_df: pd.DataFrame) -> dict[str, int]:
    """Stand-in model that carries no state (identity forecast rule)."""
    return {}


def _predict_signal(
    _model: dict[str, int], test_df: pd.DataFrame
) -> np.ndarray:
    """Forecast equal to the raw signal column of the test slice."""
    return test_df["signal"].to_numpy(dtype=float)


def test_walkforward_wf_t_matches_manual_oos_recompute() -> None:
    """wf_t (step == test_window) equals a manual OOS t-stat by folds.

    With disjoint test slices the union of out-of-sample rows is the
    concatenation of the fold slices in order, so ``wf_t`` must equal
    the t-stat recomputed by hand from ``folds`` test_start/test_end.
    """
    rng = np.random.default_rng(17)
    n_rows = 400
    signal = rng.standard_normal(n_rows)
    outcome = 0.3 * signal + rng.standard_normal(n_rows) * 0.5
    frame = pd.DataFrame({"signal": signal, "outcome": outcome})
    result = walkforward(
        frame,
        train_window=150,
        test_window=25,
        step=25,
        fit_fn=_fit_noop,
        predict_fn=_predict_signal,
    )
    folds = result["folds"]
    stacked = np.concatenate(
        [
            np.sign(signal[int(fold.test_start) : int(fold.test_end)])
            * outcome[int(fold.test_start) : int(fold.test_end)]
            for fold in folds.itertuples()
        ]
    )
    manual_t = float(
        np.mean(stacked) / np.std(stacked, ddof=1) * np.sqrt(stacked.size)
    )
    assert result["n_folds"] == 10
    assert result["overlap_factor"] == 1.0
    assert result["n_pnl_unique"] == stacked.size
    assert result["wf_t"] == manual_t
    assert check_no_peeking(result) is True


def test_track_record_mdd_matches_evaluation_metrics_and_manual() -> None:
    """track_record._max_drawdown == evaluation_metrics.max_drawdown.

    Both are compared against a hand-written min(equity / running-peak
    - 1) over 50 random series, with the equity built IDENTICALLY on
    every side: the ``[1.0]`` initial-capital prefix followed by
    cumprod(1 + rets) -- ``_max_drawdown`` compounds that path
    internally, ``max_drawdown`` receives it pre-compounded, so any
    prefix or peak-off-by-one divergence between the modules breaks
    here.  Returns are drawn in (-0.95, 0.95) so the curve stays
    strictly positive (``max_drawdown`` rejects non-positive equity)
    and the wipe-out branch (``equity[-1] <= 0`` -> -1.0) never fires.
    """
    rng = np.random.default_rng(51)
    for _trial in range(50):
        rets = rng.uniform(-0.95, 0.95, 120)
        equity = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
        peak = np.maximum.accumulate(equity)
        manual = float(np.min(equity / peak - 1.0))
        from_track = _max_drawdown(rets)
        from_eval = max_drawdown(equity)["mdd"]
        assert from_track == pytest.approx(manual, abs=1e-12)
        assert from_eval == pytest.approx(manual, abs=1e-12)


def test_pearson_helpers_agree_between_null_control_and_model_eval() -> None:
    """null_control_suite._pearson_r == predictive_model_eval._pearson.

    The two modules each keep a private Pearson helper with an
    identical contract (size < 3 or zero sd -> NaN, else
    ``np.corrcoef``); 100 random pairs (correlated and independent)
    must agree to 1e-12 or the correlation contract has drifted
    between the null-control and model-eval sides of the library.
    Both helpers see only clean finite inputs here -- their NaN
    handling is exercised by one constant-side probe each, which must
    also agree (both NaN).
    """
    rng = np.random.default_rng(77)
    for _trial in range(100):
        left = rng.standard_normal(80)
        right = (
            0.2 * left + rng.standard_normal(80)
            if _trial % 2 == 0
            else rng.standard_normal(80)
        )
        assert _pearson_r(left, right) == pytest.approx(
            _pearson(left, right), abs=1e-12
        )
    constant = np.full(80, 3.0)
    fresh = rng.standard_normal(80)
    assert np.isnan(_pearson_r(constant, fresh))
    assert np.isnan(_pearson(constant, fresh))


def test_bootstrap_resample_equals_block_indices_recompute() -> None:
    """monte_carlo.bootstrap_resample == manual block_indices pipeline.

    The manual path re-derives each replicate matrix from
    ``_utils.block_indices`` with an identically seeded generator and
    applies the statistic through ``_utils.apply_stat_rowwise`` -- the
    exact decomposition ``bootstrap_resample`` documents -- across
    block/seed combinations including ``block == n_obs`` (a single
    wrapped block per replicate).  The iid branch (``block=None``) is
    checked against its own documented protocol, a plain
    ``rng.integers`` draw.
    """
    rng = np.random.default_rng(29)
    n_obs, n_boot = 150, 200
    data = rng.standard_normal(n_obs) * 0.01
    for block, seed in ((20, 5), (7, 11), (150, 42)):
        drawn = bootstrap_resample(
            data, np.mean, n_boot=n_boot, block=block, seed=seed
        )
        idx = block_indices(n_obs, n_boot, block, np.random.default_rng(seed))
        manual = apply_stat_rowwise(data[idx], np.mean)
        assert np.array_equal(drawn, manual)

    drawn_iid = bootstrap_resample(data, np.mean, n_boot=n_boot, seed=3)
    idx_iid = np.random.default_rng(3).integers(
        0, n_obs, size=(n_boot, n_obs)
    )
    manual_iid = apply_stat_rowwise(data[idx_iid], np.mean)
    assert np.array_equal(drawn_iid, manual_iid)
