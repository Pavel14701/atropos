"""Tests for atropos.track_record production track-record verdicts.

Hand-computed Lo (2002) standard errors and divergence z-statistics,
exact drawdown-barrier boundary cases (a breach at equality), the
LIVE_OK / LIVE_KILLED / INCONCLUSIVE contract on deterministic
[flat, gain] tile streams (sample Sharpe is exact, no seed luck),
barrier precedence over the decision point, and prereg validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos.track_record import (
    drawdown_floor_check,
    live_verdict,
    prereg_divergence,
    sharpe_standard_error,
)


def _tile_sr(gain: float, n_pairs: int) -> float:
    """Annualized Sharpe of a [0.0, gain] tile (mean = std = gain/2)."""
    stream = np.tile([0.0, gain], n_pairs)
    stdev = float(np.std(stream, ddof=1))
    return float(np.mean(stream) / stdev * np.sqrt(252.0))


def test_sharpe_standard_error_known_answer() -> None:
    # periods=1: se = sqrt((1 + 0.5*1)/100) = sqrt(0.015)
    assert sharpe_standard_error(1.0, 100, periods=1) == pytest.approx(
        math.sqrt(0.015), abs=1e-12
    )
    # annualized case: SR_pp = 1.5/sqrt(252); se rescaled by sqrt(252)
    assert sharpe_standard_error(1.5, 4032) == pytest.approx(
        0.250557, abs=1e-4
    )
    # negative Sharpe: squared inside the formula, still positive se
    assert sharpe_standard_error(-1.0, 100, periods=1) == pytest.approx(
        math.sqrt(0.015), abs=1e-12
    )


def test_sharpe_standard_error_validates() -> None:
    with pytest.raises(ValueError):
        sharpe_standard_error(float("nan"), 100)
    with pytest.raises(ValueError):
        sharpe_standard_error(float("inf"), 100)
    with pytest.raises(ValueError):
        sharpe_standard_error(1.0, 0)
    with pytest.raises(ValueError):
        sharpe_standard_error(1.0, 100, periods=0)


def test_prereg_divergence_hand_computed() -> None:
    out = prereg_divergence(0.5, 504, 1.5, 1008)
    assert out["se_live"] == pytest.approx(0.707101, abs=1e-4)
    assert out["se_expected"] == pytest.approx(0.501114, abs=1e-4)
    assert out["z"] == pytest.approx(-1.153847, abs=1e-4)
    assert out["kill_z"] == 1.64
    # z=-1.154: killed at kill_z=1.0, alive at the 1.64 default
    loose = prereg_divergence(0.5, 504, 1.5, 1008, kill_z=1.0)
    assert loose["diverged"]
    assert not out["diverged"]
    # outperformance is never evidence: z far above 0 -> no kill flag
    strong = prereg_divergence(3.0, 504, 1.5, 1008)
    assert strong["z"] > 0.0
    assert not strong["diverged"]


def test_prereg_divergence_validates() -> None:
    with pytest.raises(ValueError):
        prereg_divergence(float("nan"), 504, 1.5, 1008)
    with pytest.raises(ValueError):
        prereg_divergence(0.5, 504, float("inf"), 1008)
    with pytest.raises(ValueError):
        prereg_divergence(0.5, 0, 1.5, 1008)
    with pytest.raises(ValueError):
        prereg_divergence(0.5, 504, 1.5, 0)
    with pytest.raises(ValueError):
        prereg_divergence(0.5, 504, 1.5, 1008, kill_z=0.0)
    with pytest.raises(ValueError):
        prereg_divergence(0.5, 504, 1.5, 1008, kill_z=5.0)


def test_drawdown_floor_boundary_and_breach() -> None:
    wiped = drawdown_floor_check([-0.30], dd_floor=-0.25)
    assert wiped["mdd"] == pytest.approx(-0.30)
    assert wiped["breached"]
    assert wiped["n_obs"] == 1
    # mdd exactly AT the floor counts as a breach (<=)
    edge = drawdown_floor_check([0.10, -0.05], dd_floor=-0.05)
    assert edge["mdd"] == pytest.approx(-0.05)
    assert edge["breached"]
    # above the floor: no breach
    fine = drawdown_floor_check([0.10, -0.05], dd_floor=-0.25)
    assert fine["mdd"] == pytest.approx(-0.05)
    assert not fine["breached"]
    # total wipeout -> -1.0
    ruin = drawdown_floor_check([-1.0, 5.0], dd_floor=-0.25)
    assert ruin["mdd"] == -1.0
    assert ruin["breached"]
    # empty track: nothing to breach
    empty = drawdown_floor_check([], dd_floor=-0.25)
    assert empty["mdd"] == 0.0
    assert not empty["breached"]
    assert empty["n_obs"] == 0


def test_drawdown_floor_validates() -> None:
    with pytest.raises(ValueError):
        drawdown_floor_check([0.01, 0.02], dd_floor=0.0)
    with pytest.raises(ValueError):
        drawdown_floor_check([0.01, 0.02], dd_floor=0.5)


def test_live_verdict_ok_on_consistent_track() -> None:
    live = np.tile([0.0, 0.016], 500)  # n=1000, SR ~ 15.87
    expected = _tile_sr(0.016, 1000)  # same law, n=2000
    out = live_verdict(
        live,
        expected,
        n_backtest=2000,
        min_observations=500,
        dd_floor=-0.40,
    )
    assert out["verdict"] == "LIVE_OK"
    assert out["reason"] == "consistent_with_prereg"
    assert out["live_sharpe"] == pytest.approx(15.866577, abs=1e-3)
    assert out["z"] == pytest.approx(0.0, abs=0.1)
    assert out["se_live"] == pytest.approx(0.614630, abs=1e-3)
    assert not out["dd_breached"]
    assert out["n_obs"] == 1000
    assert out["expected_sharpe"] == pytest.approx(15.870542, abs=1e-3)


def test_live_verdict_killed_below_prereg() -> None:
    live = np.tile([0.0, 0.016], 500)  # SR ~ 15.87
    out = live_verdict(
        live,
        30.0,  # prereg promised double the delivered Sharpe
        n_backtest=2000,
        min_observations=500,
        dd_floor=-0.40,
    )
    assert out["verdict"] == "LIVE_KILLED"
    assert out["reason"] == "live_below_prereg"
    assert out["z"] < -1.64


def test_live_verdict_killed_by_drawdown_barrier() -> None:
    crashed = np.concatenate(
        [np.tile([0.0, 0.016], 60), np.full(40, -0.01)]
    )  # n=160, mdd ~ -33%
    at_decision_point = live_verdict(
        crashed,
        15.87,
        n_backtest=2000,
        min_observations=150,
        dd_floor=-0.25,
    )
    assert at_decision_point["verdict"] == "LIVE_KILLED"
    assert at_decision_point["reason"] == "drawdown_floor_breached"
    assert at_decision_point["mdd"] < -0.25
    # the barrier fires even before the decision point is reached
    before_decision_point = live_verdict(
        crashed,
        15.87,
        n_backtest=2000,
        min_observations=999,
        dd_floor=-0.25,
    )
    assert before_decision_point["verdict"] == "LIVE_KILLED"
    assert before_decision_point["reason"] == "drawdown_floor_breached"


def test_live_verdict_inconclusive_on_short_track() -> None:
    out = live_verdict(
        np.tile([0.0, 0.016], 40),  # n=80 < 500
        15.87,
        n_backtest=2000,
        min_observations=500,
        dd_floor=-0.40,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reason"] == "track_too_short"
    assert math.isnan(out["live_sharpe"])
    assert math.isnan(out["z"])
    assert out["mdd"] == 0.0
    assert not out["dd_breached"]


def test_live_verdict_inconclusive_on_degenerate_track() -> None:
    out = live_verdict(
        [0.001] * 200,  # constant returns: Sharpe undefined
        15.87,
        n_backtest=2000,
        min_observations=100,
        dd_floor=-0.50,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reason"] == "degenerate_returns"
    assert math.isnan(out["live_sharpe"])


def test_live_verdict_inconclusive_on_empty_track() -> None:
    out = live_verdict([], 15.87, n_backtest=2000)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reason"] == "empty_track_record"
    assert out["n_obs"] == 0


def test_live_verdict_is_pure_function() -> None:
    live = np.tile([0.0, 0.016], 300)
    first = live_verdict(live, 15.87, n_backtest=2000)
    repeat = live_verdict(live, 15.87, n_backtest=2000)
    assert first.keys() == repeat.keys()
    for key in first:
        assert first[key] == repeat[key]


def test_live_verdict_validates_prereg_parameters() -> None:
    live = [0.001, 0.002, -0.001] * 100
    with pytest.raises(ValueError):
        live_verdict(live, 0.0, n_backtest=2000)  # dead prereg
    with pytest.raises(ValueError):
        live_verdict(live, -1.0, n_backtest=2000)
    with pytest.raises(ValueError):
        live_verdict(live, float("nan"), n_backtest=2000)
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=0)
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=2000, periods=0)
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=2000, min_observations=2)
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=2000, dd_floor=0.0)
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=2000, kill_z=0.0)
