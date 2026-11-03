"""Tests for atropos.evaluation_metrics backtest metrics.

Known-answer checks for equity compounding, drawdown, trade metrics and
turnover; sign, key and determinism checks for the Sharpe family,
including the deflated Sharpe penalty for multiple trials.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos.evaluation_metrics import (
    deflated_sharpe,
    equity_from_returns,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    turnover,
    win_rate,
)


def test_equity_from_returns_compounds_exactly() -> None:
    curve = equity_from_returns([0.1, -0.05])
    assert curve == pytest.approx([1.1, 1.045])
    doubled = equity_from_returns([0.1, -0.05], initial=2.0)
    assert doubled == pytest.approx([2.2, 2.09])
    nan_dropped = equity_from_returns([0.1, float("nan"), 0.0])
    assert nan_dropped == pytest.approx([1.1, 1.1])


def test_max_drawdown_on_crafted_equity_curves() -> None:
    out = max_drawdown([1.0, 1.2, 0.9, 1.1])
    assert out["mdd"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert out["mdd"] == pytest.approx(-0.25)
    assert out["peak_index"] == 1
    assert out["trough_index"] == 2
    assert out["mdd_duration"] == 2
    assert out["still_underwater"]
    assert out["n"] == 4
    recovered = max_drawdown([1.0, 1.2, 0.9, 1.3])
    assert recovered["mdd"] == pytest.approx(-0.25)
    assert not recovered["still_underwater"]


def test_max_drawdown_guards_and_empty_input() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        max_drawdown([1.0, 0.5, -0.2])
    empty = max_drawdown([])
    assert math.isnan(empty["mdd"])
    assert empty["n"] == 0
    assert empty["mdd_duration"] == 0


def test_sharpe_ratio_signs_keys_and_ci_bounds() -> None:
    noise = np.random.default_rng(7).standard_normal(200)
    # drift dominates the seed-7 noise draw so the signs are unambiguous
    positive = sharpe_ratio(0.003 + 0.01 * noise, n_boot=200, seed=3)
    negative = sharpe_ratio(-0.003 + 0.01 * noise, n_boot=200, seed=3)
    assert positive["sr"] > 0
    assert negative["sr"] < 0
    assert set(positive) == {
        "sr",
        "sr_ci",
        "sr_per_period",
        "ci_method",
        "n",
        "rf",
    }
    assert positive["ci_method"] == "moving_block"
    assert positive["n"] == 200
    assert positive["sr"] == pytest.approx(
        positive["sr_per_period"] * float(np.sqrt(252))
    )
    ci_low, ci_high = positive["sr_ci"]
    assert ci_low < ci_high


def test_sharpe_ratio_seed_determinism_and_flat_returns() -> None:
    returns = 0.0005 + 0.01 * np.random.default_rng(4).standard_normal(120)
    first = sharpe_ratio(returns, n_boot=200, seed=9)
    repeat = sharpe_ratio(returns, n_boot=200, seed=9)
    other_seed = sharpe_ratio(returns, n_boot=200, seed=10)
    assert first["sr_ci"] == repeat["sr_ci"]
    assert first["sr_ci"] != other_seed["sr_ci"]
    # zero variance returns carry no Sharpe information at all
    flat = sharpe_ratio([0.01] * 50, n_boot=200, seed=1)
    assert math.isnan(flat["sr"])
    assert flat["ci_method"] == "insufficient_data"
    assert math.isnan(flat["sr_ci"][0])


def test_deflated_sharpe_penalizes_more_trials() -> None:
    returns = 0.0004 + 0.01 * np.random.default_rng(3).standard_normal(300)
    few = deflated_sharpe(returns, n_trials=2, seed=0)
    many = deflated_sharpe(returns, n_trials=100, seed=0)
    assert many["sr_benchmark_annual"] > few["sr_benchmark_annual"]
    assert many["dsr"] < few["dsr"]
    assert many["p_value"] > few["p_value"]
    assert few["p_value"] == pytest.approx(1.0 - few["dsr"])
    assert few["sr_var_source"] == "given"
    assert few["n_trials"] == 2
    boot = deflated_sharpe(
        returns,
        n_trials=10,
        sr_var_annual="bootstrap",
        n_boot=100,
        seed=0,
    )
    assert boot["sr_var_source"] == "bootstrap"
    assert boot["sr_var_annual"] > 0


def test_profit_factor_and_win_rate_on_known_trades() -> None:
    out = profit_factor([100.0, -40.0, 60.0, -10.0])
    assert out["pf"] == pytest.approx(160.0 / 50.0)
    assert out["gross_win"] == pytest.approx(160.0)
    assert out["gross_loss"] == pytest.approx(50.0)
    assert out["n_winners"] == 2
    assert out["n_losers"] == 2
    all_win = profit_factor([10.0, 20.0])
    assert all_win["pf"] == math.inf
    assert all_win["gross_loss"] == 0.0
    empty = profit_factor([])
    assert math.isnan(empty["pf"])
    assert empty["n_trades"] == 0

    wins = win_rate([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    assert wins["wr"] == pytest.approx(0.75)
    assert wins["ci_low"] <= 0.75 <= wins["ci_high"]
    assert wins["n"] == 8
    perfect = win_rate([1.0, 1.0, 1.0, 1.0])
    assert perfect["wr"] == 1.0
    assert perfect["ci_high"] == 1.0
    empty_wr = win_rate([])
    assert math.isnan(empty_wr["wr"])
    assert empty_wr["n"] == 0


def test_turnover_on_known_position_series() -> None:
    out = turnover([0.0, 1.0, 1.0, -1.0, 0.0])
    # diffs are |1| + |0| + |2| + |1| = 4 over 4 rebalances
    assert out["turnover_per_period"] == pytest.approx(1.0)
    assert out["rt_per_year"] == pytest.approx(252.0)
    assert out["n_rebalances"] == 4
    constant = turnover([1.0] * 10)
    assert constant["turnover_per_period"] == 0.0
    assert constant["rt_per_year"] == 0.0
    short = turnover([1.0])
    assert math.isnan(short["turnover_per_period"])
