"""Tests for atropos.track_record production track-record verdicts.

Hand-computed Lo (2002) standard errors and divergence z-statistics,
exact drawdown-barrier boundary cases (a breach at equality), the
LIVE_OK / LIVE_KILLED / INCONCLUSIVE contract on deterministic
[flat, gain] tile streams (sample Sharpe is exact, no seed luck),
barrier precedence over the decision point, and prereg validation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from atropos import track_record
from atropos.track_record import (
    DrawdownNotComputable,
    drawdown_floor_check,
    live_verdict,
    prereg_divergence,
    run_all_examples,
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
    assert out["se_live"] == pytest.approx(0.707282, abs=1e-4)
    assert out["se_expected"] == pytest.approx(0.501114, abs=1e-4)
    assert out["z"] == pytest.approx(-1.153584, abs=1e-4)
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


def test_drawdown_mid_path_negative_equity_reads_total_loss() -> None:
    # regression: only the TERMINAL equity used to trigger the -1.0
    # sentinel, so a levered path that went negative mid-way and then
    # recovered reported mdd ~ -1.33 -- a "drawdown" deeper than
    # -100%, inexpressible as a fraction of the peak and inconsistent
    # with monte_carlo.max_drawdown_distribution, which drops paths
    # whose equity goes non-positive at ANY step.  Any non-positive
    # equity step now collapses to the documented -1.0 sentinel.
    recovering = [0.5, -4.0 / 3.0, -1.8]  # equity [1.0, 1.5, -0.5, 0.4]
    check = drawdown_floor_check(recovering, dd_floor=-0.25)
    assert check["mdd"] == -1.0  # old code: -1.3333...
    assert check["breached"] is True
    assert check["n_obs"] == 3
    verdict = live_verdict(
        recovering, 1.5, n_backtest=2000, min_observations=8
    )
    assert verdict["verdict"] == "LIVE_KILLED"
    assert verdict["reason"] == "drawdown_floor_breached"
    assert verdict["mdd"] == -1.0


def test_drawdown_floor_validates() -> None:
    with pytest.raises(ValueError):
        drawdown_floor_check([0.01, 0.02], dd_floor=0.0)
    with pytest.raises(ValueError):
        drawdown_floor_check([0.01, 0.02], dd_floor=0.5)


def test_drawdown_overflow_fails_closed() -> None:
    # 1 + 1e308 == 1e308 in float64, so cumprod overflows to inf,
    # inf/inf makes mdd NaN and `nan <= floor` is False.  The barrier
    # must fail closed: never a silent LIVE_OK on uncomputable input.
    overflow = [1e308, 1e308, 1e308]
    with pytest.raises(ValueError, match="non-finite"):
        drawdown_floor_check(overflow, dd_floor=-0.25)
    out = live_verdict(overflow, 1.5, n_backtest=2000)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reason"] == "drawdown_not_computable"
    assert math.isnan(out["mdd"])
    assert not out["dd_breached"]
    # fails closed even past the decision point: not-computable takes
    # precedence over the too-short gate
    at_decision_point = live_verdict(
        overflow, 1.5, n_backtest=2000, min_observations=8
    )
    assert at_decision_point["reason"] == "drawdown_not_computable"


def test_drawdown_not_computable_is_narrow_value_error() -> None:
    # the overflow sentinel is a dedicated class, and it IS still a
    # ValueError (backward compatibility for external handlers)
    assert issubclass(DrawdownNotComputable, ValueError)
    with pytest.raises(DrawdownNotComputable, match="non-finite"):
        drawdown_floor_check([1e308, 1e308, 1e308], dd_floor=-0.25)
    # legacy `except ValueError` callers intercept the very same raise
    with pytest.raises(ValueError, match="non-finite"):
        drawdown_floor_check([1e308, 1e308, 1e308], dd_floor=-0.25)


def test_live_verdict_does_not_mask_foreign_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the INCONCLUSIVE degradation is reserved for DrawdownNotComputable
    # alone: an ordinary ValueError raised anywhere else must PROPAGATE
    # (a validation bug must stay visible, not become a soft verdict)
    def _boom(_rets: np.ndarray, _floor: float) -> dict[str, Any]:
        raise ValueError("unexpected bug inside the drawdown leg")

    monkeypatch.setattr(track_record, "_drawdown_floor_from_clean", _boom)
    with pytest.raises(ValueError, match="unexpected bug"):
        track_record.live_verdict([0.01, -0.02], 1.5, n_backtest=100)
    # invalid dd_floor is upfront VALIDATION: it raises, it never
    # degrades to INCONCLUSIVE
    with pytest.raises(ValueError, match="dd_floor must be < 0"):
        track_record.live_verdict(
            [0.01, -0.02], 1.5, n_backtest=100, dd_floor=0.0
        )


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
        live_verdict(live, 1.5, n_backtest=2000, dd_floor=float("nan"))
    with pytest.raises(ValueError):
        live_verdict(live, 1.5, n_backtest=2000, kill_z=0.0)


def test_run_all_examples_asserts_all_four_fates() -> None:
    # the demo is itself a regression test: its internal asserts on
    # the four canonical verdicts (LIVE_OK, divergence kill, DD kill,
    # too short) must hold
    run_all_examples()


# -- informational metrics wave: live_sortino / live_cagr --------------------


def test_live_verdict_informational_keys() -> None:
    """live_sortino/live_cagr ride the verdict as read-outs.

    They never touch the verdict token: on the existing kill fixture
    the verdict stays LIVE_KILLED while live_sortino is None (n=3
    < 30, the sortino gate) and on a 40-observation OK-track both are
    numbers.  Mutation: verdict logic reading the new keys -> the
    kill-path asserts fail -> caught.
    """
    recovering = [0.5, -4.0 / 3.0, -1.8]
    verdict = live_verdict(
        recovering, 1.5, n_backtest=2000, min_observations=8
    )
    assert verdict["verdict"] == "LIVE_KILLED"  # unchanged by the wave
    assert verdict["live_sortino"] is None  # n=3 < 30
    assert verdict["live_cagr"] is not None  # active-convention cagr

    rng = np.random.default_rng(11)
    steady = (0.004 + 0.01 * rng.standard_normal(40)).tolist()
    ok = live_verdict(steady, 0.5, n_backtest=2000, min_observations=8)
    assert ok["verdict"] in {"LIVE_OK", "LIVE_KILLED", "INCONCLUSIVE"}
    assert ok["live_sortino"] is not None  # n=40 >= 30
    assert ok["live_cagr"] is not None
