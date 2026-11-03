"""Tests for atropos.evaluation_metrics backtest metrics.

Known-answer checks for equity compounding, drawdown, trade metrics and
turnover; sign, key and determinism checks for the Sharpe family,
including the deflated Sharpe penalty for multiple trials, the
``sr_annual``/``rf`` inputs of DSR and its degenerate-denominator NaN.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from atropos.evaluation_metrics import (
    calmar_ratio,
    deflated_sharpe,
    equity_from_returns,
    max_drawdown,
    profit_factor,
    run_all_examples,
    sharpe_ratio,
    turnover,
    win_rate,
)
from scipy import stats


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


def test_max_drawdown_rejects_2d_input() -> None:
    grid = np.array([[1.0, 1.2, 0.9], [1.1, 1.3, 1.0]])
    with pytest.raises(ValueError, match="1-D"):
        max_drawdown(grid)
    # the same data flattened explicitly is fine
    flat = max_drawdown(grid.reshape(-1))
    assert flat["n"] == 6


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


def test_block_bootstrap_numbers_unchanged_after_refactor() -> None:
    """Refactor guard: shared _utils helpers reproduce historical values.

    The move of the inline block bootstrap to ``_utils.block_indices``
    and of ``_clean`` to ``_utils.clean_1d`` must be bit-compatible
    with the pre-refactor RNG order at a fixed seed.
    """
    generator = np.random.default_rng(9)
    n_obs = 756
    shocks = generator.standard_normal(n_obs)
    ar = np.zeros(n_obs)
    for step in range(1, n_obs):
        ar[step] = 0.3 * ar[step - 1] + 0.95 * shocks[step]
    returns = 0.0006 + 0.01 * ar

    sr = sharpe_ratio(returns, rf=0.02, seed=0)
    assert sr["sr_ci"] == pytest.approx(
        (-0.4167637458935403, 2.556882296222872), abs=1e-12
    )
    # exact demo values (the demo keeps only the seeded containment
    # sr_low < sr < sr_high, which is not a theorem, only a fixed-seed
    # fact)
    assert sr["sr"] == pytest.approx(1.0888354644333431, abs=1e-12)
    assert sr["sr_ci"][0] < sr["sr"] < sr["sr_ci"][1]
    dsr = deflated_sharpe(returns, n_trials=2, seed=0)
    assert dsr["dsr"] == pytest.approx(0.8856476225494652, abs=1e-12)
    boot = deflated_sharpe(
        returns,
        n_trials=10,
        sr_var_annual="bootstrap",
        n_boot=200,
        seed=0,
    )
    assert boot["sr_var_annual"] == pytest.approx(0.53875895088458, abs=1e-12)
    assert boot["dsr"] == pytest.approx(0.5418688773414638, abs=1e-12)


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


def test_deflated_sharpe_single_trial_equals_psr() -> None:
    returns = 0.0004 + 0.01 * np.random.default_rng(3).standard_normal(300)
    single = deflated_sharpe(returns, n_trials=1, seed=0)
    # one trial = no selection among strategies: the benchmark is zero
    # and the multiple-testing penalty must vanish entirely
    assert single["sr_benchmark_annual"] == 0.0
    sr_pp = float(returns.mean() / returns.std(ddof=1))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    denom = float(
        np.sqrt(1.0 - skew * sr_pp + (kurt - 1.0) / 4.0 * sr_pp * sr_pp)
    )
    psr = float(stats.norm.cdf(sr_pp * np.sqrt(returns.size - 1) / denom))
    assert single["dsr"] == pytest.approx(psr)
    # monotone penalty: 1 -> 10 -> 100 trials deflates DSR stepwise
    ten = deflated_sharpe(returns, n_trials=10, seed=0)
    hundred = deflated_sharpe(returns, n_trials=100, seed=0)
    assert single["dsr"] > ten["dsr"] > hundred["dsr"]


def test_deflated_sharpe_sr_annual_drives_dsr() -> None:
    returns = 0.0004 + 0.01 * np.random.default_rng(3).standard_normal(300)
    sample = deflated_sharpe(returns, n_trials=5, seed=0)
    higher = deflated_sharpe(returns, n_trials=5, sr_annual=2.0, seed=0)
    assert higher["sr_annual"] == 2.0
    assert higher["dsr"] > sample["dsr"]
    # known answer: the DSR math uses sr_annual / sqrt(periods)
    sr_pp_used = 2.0 / float(np.sqrt(252))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    denom = float(
        np.sqrt(
            1.0
            - skew * sr_pp_used
            + (kurt - 1.0) / 4.0 * sr_pp_used * sr_pp_used
        )
    )
    sr0_pp = sample["sr_benchmark_annual"] / float(np.sqrt(252))
    expected = float(
        stats.norm.cdf(
            (sr_pp_used - sr0_pp) * np.sqrt(returns.size - 1) / denom
        )
    )
    assert higher["dsr"] == pytest.approx(expected)
    # roundtrip: feeding back the sample's own annualized SR is a no-op
    roundtrip = deflated_sharpe(
        returns, n_trials=5, sr_annual=sample["sr_annual"], seed=0
    )
    assert roundtrip["dsr"] == pytest.approx(sample["dsr"])


def test_deflated_sharpe_degenerate_denominator_is_nan() -> None:
    rets = np.concatenate([np.full(39, 0.01), np.full(54, -0.05)])
    degenerate_sr_annual = 97.13331045527151
    skew = float(stats.skew(rets))
    kurt = float(stats.kurtosis(rets, fisher=False))
    sr_pp_used = degenerate_sr_annual / float(np.sqrt(252))
    denom2 = (
        1.0 - skew * sr_pp_used + (kurt - 1.0) / 4.0 * sr_pp_used * sr_pp_used
    )
    # a two-point sample probed at the vertex of the quadratic: float
    # rounding pushes the denominator just below zero, so the DSR
    # approximation is out of its domain and must return NaN, not a
    # fabricated 0/1 from clamping
    assert denom2 <= 0.0
    out = deflated_sharpe(rets, n_trials=5, sr_annual=degenerate_sr_annual)
    assert math.isnan(out["dsr"])
    assert math.isnan(out["p_value"])
    sane = deflated_sharpe(rets, n_trials=5)
    assert math.isfinite(sane["dsr"])


def test_deflated_sharpe_rf_subtracts_from_returns() -> None:
    returns = 0.0004 + 0.01 * np.random.default_rng(3).standard_normal(300)
    gross = deflated_sharpe(returns, n_trials=5, seed=0)
    net = deflated_sharpe(returns, n_trials=5, rf=0.02, seed=0)
    expected_pp = (returns.mean() - 0.02 / 252.0) / returns.std(ddof=1)
    assert net["sr_annual"] == pytest.approx(
        expected_pp * float(np.sqrt(252))
    )
    assert net["sr_annual"] < gross["sr_annual"]


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
    # diffs are |1| + |0| + |2| + |1| = 4 across 3 actual changes
    assert out["turnover_per_period"] == pytest.approx(1.0)
    assert out["rt_per_year"] == pytest.approx(252.0)
    assert out["n_rebalances"] == 3
    constant = turnover([1.0] * 10)
    assert constant["turnover_per_period"] == 0.0
    assert constant["rt_per_year"] == 0.0
    assert constant["n_rebalances"] == 0
    short = turnover([1.0])
    assert math.isnan(short["turnover_per_period"])


def test_public_parameters_validated_before_data() -> None:
    """Upfront ValueError naming the PUBLIC parameter, before the data.

    ``garbage = object()`` proves the checks fire before any input
    coercion: the parameter ValueError must win over a numpy
    coercion error.
    """
    garbage = object()
    with pytest.raises(ValueError, match="ci_level"):
        sharpe_ratio(garbage, ci_level=1.5)
    with pytest.raises(ValueError, match="periods"):
        sharpe_ratio(garbage, periods=0)
    with pytest.raises(ValueError, match="n_boot"):
        sharpe_ratio(garbage, n_boot=0)
    with pytest.raises(ValueError, match="block"):
        sharpe_ratio(garbage, block=0)
    with pytest.raises(ValueError, match="rf"):
        sharpe_ratio(garbage, rf=float("nan"))
    with pytest.raises(ValueError, match="ci_level"):
        win_rate(garbage, ci_level=0.0)
    with pytest.raises(ValueError, match="periods"):
        turnover(garbage, periods=0)
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(garbage, n_trials=0)
    with pytest.raises(ValueError, match="n_trials"):
        # fractional trials must be rejected, not truncated to 2
        deflated_sharpe(garbage, n_trials=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rf"):
        deflated_sharpe(garbage, n_trials=5, rf=float("nan"))
    with pytest.raises(ValueError, match="sr_annual"):
        deflated_sharpe(garbage, n_trials=5, sr_annual=float("inf"))
    # None sr_annual stays legal (it means "estimate from returns")
    sample = deflated_sharpe(
        np.linspace(-0.02, 0.02, 30), n_trials=1, sr_annual=None
    )
    assert sample["n"] == 30


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (
            lambda: sharpe_ratio(
                object(),
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: sharpe_ratio(
                object(),
                n_boot=2.5,  # type: ignore[arg-type]
            ),
            "n_boot",
        ),
        (
            lambda: sharpe_ratio(
                object(),
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
        (
            lambda: calmar_ratio(
                object(),
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: turnover(
                object(),
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: deflated_sharpe(
                object(),
                n_trials=2.5,  # type: ignore[arg-type]
            ),
            "n_trials",
        ),
        (
            lambda: deflated_sharpe(
                object(),
                n_trials=2,
                periods=2.5,  # type: ignore[arg-type]
            ),
            "periods",
        ),
        (
            lambda: deflated_sharpe(
                object(),
                n_trials=2,
                n_boot=2.5,  # type: ignore[arg-type]
            ),
            "n_boot",
        ),
        (
            lambda: deflated_sharpe(
                object(),
                n_trials=2,
                block=2.5,  # type: ignore[arg-type]
            ),
            "block",
        ),
    ],
)
def test_integer_parameters_reject_fractional_values(
    call: Callable[[], object], name: str
) -> None:
    """Non-integer counts raise ValueError before the data is touched.

    Regression class: ``int(value) >= 1`` guards used to truncate
    ``n_boot=2.5`` / ``periods=2.5`` silently (and ``calmar_ratio``
    did not check ``periods`` at all).
    """
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        call()


def test_numeric_parameters_reject_object_with_value_error() -> None:
    """``object()`` numeric params raise ValueError naming the parameter.

    The docstrings promise ``Raises: ValueError`` before the data is
    touched; the old bare ``0 < x < 1`` / ``math.isfinite`` checks
    escaped as ``TypeError`` instead.
    """
    garbage = object()
    with pytest.raises(ValueError, match="ci_level must be a finite"):
        sharpe_ratio(
            garbage,
            ci_level=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="rf must be a finite"):
        sharpe_ratio(garbage, rf=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_boot must be an integer"):
        sharpe_ratio(garbage, n_boot=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sr_annual must be a finite"):
        deflated_sharpe(
            garbage,
            n_trials=5,
            sr_annual=object(),  # type: ignore[arg-type]
        )


def test_demo_run_all_examples_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module demo runs clean and prints every metric block."""
    run_all_examples()
    out = capsys.readouterr().out
    assert "evaluation_metrics demo" in out
    assert "sharpe_ratio : " in out
    assert "moving_block" in out
    assert "max_drawdown : " in out
    assert "calmar_ratio : " in out
    assert "deflated_sharpe(trials=  1)" in out
    assert "deflated_sharpe(trials=100)" in out
    assert "profit_factor: " in out
    assert "win_rate     : " in out
    assert "turnover     : " in out
