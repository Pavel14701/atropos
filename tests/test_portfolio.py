"""Tests for the atropos.portfolio multi-slot layer.

Hand-checked worlds pin the replay arithmetic bit-by-bit: slot cap
and release, cooldown, REPLACE-low/high credit bounds with the
reconstructed partial-hold P&L, correlation-aware admission (clone
reject, independent admit, inconclusive admit, same-instrument
stacking), event-based equity and per-bar exposure, the three
portfolio gates (including the exact delegation to
``deflated_sharpe_from_moments``), determinism and the chaos guards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.backtesting import DEFAULT_COST, simulate_signal
from atropos.evaluation_metrics import deflated_sharpe_from_moments
from atropos.portfolio import (
    _policy_credit,
    filter_by_correlation,
    portfolio_gates,
    run_portfolio,
    run_portfolio_replay,
)
from atropos.portfolio_protocol import hash_kill_prereg
from scipy.stats import kurtosis, skew, spearmanr

_DRIFT = np.repeat([0.10, 0.05], 5).astype(np.float64)
"""Healthy 10-point EV stream: finite DSR, no drawdown."""


# ---------------------------------------------------------------------------
# world builders
# ---------------------------------------------------------------------------
def _panel(
    instrument: str,
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
) -> pd.DataFrame:
    """One-instrument OHLC panel on the required MultiIndex.

    Columns form the (field, instrument) product.
    """
    columns = pd.MultiIndex.from_product(
        [["open", "high", "low", "close"], [instrument]]
    )
    data = np.column_stack(
        [
            np.asarray(opens, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(closes, dtype=np.float64),
        ]
    )
    return pd.DataFrame(data, columns=columns)


def _world() -> pd.DataFrame:
    """Flat 30-bar world with programmed bumps.

    A target bump at bar 10 and a stop dip at bar 20; time exits
    resolve in 4 bars under ``hold_bars=4``.
    """
    opens = np.full(30, 100.0)
    highs = np.full(30, 100.5)
    lows = np.full(30, 99.5)
    highs[10] = 104.0
    lows[20] = 98.0
    return _panel("solo", opens, highs, lows, opens)


def _multi_panel() -> pd.DataFrame:
    """Two flat instruments (AAA, BBB) on one shared 30-bar grid."""
    columns = pd.MultiIndex.from_product(
        [["open", "high", "low", "close"], ["AAA", "BBB"]]
    )
    flat = np.full(30, 100.0)
    highs = flat + 0.5
    lows = flat - 0.5
    data = np.column_stack([flat, flat, highs, highs, lows, lows, flat, flat])
    return pd.DataFrame(data, columns=columns)


def _signals(
    decisions: list[int],
    sides: list[str],
    sls: list[float],
    tps: list[float],
    scores: list[float],
    *,
    instruments: list[str] | None = None,
    priorities: list[float] | None = None,
    proxies: list[np.ndarray] | None = None,
    atr_entry: float | None = None,
) -> pd.DataFrame:
    """Signal frame with the required portfolio columns."""
    count = len(decisions)
    frame = pd.DataFrame(
        {
            "decision_idx": decisions,
            "side": sides,
            "instrument": instruments
            if instruments is not None
            else ["solo"] * count,
            "sl": sls,
            "tp": tps,
            "score": scores,
        }
    )
    if priorities is not None:
        frame["priority"] = priorities
    if proxies is not None:
        # a 1-D object column of per-row proxy series (pandas-stubs
        # safe even when every proxy shares one length)
        proxy_column = np.empty(count, dtype=object)
        for slot, proxy in enumerate(proxies):
            proxy_column[slot] = proxy
        frame["r_proxy"] = proxy_column
    if atr_entry is not None:
        frame["atr_entry"] = atr_entry
    return frame


def _fingerprint(result: Mapping[str, Any]) -> tuple[Any, ...]:
    """Array-free comparable fingerprint of a replay result."""
    return (
        result["equity_curve"].tolist(),
        result["exposure"]["gross"].tolist(),
        result["exposure"]["net"].tolist(),
        [
            (row["decision_idx"], row["slot"], row["credit_r"])
            for row in result["accepted"]
        ],
        [row["reason"] for row in result["rejected"]],
        list(result["summary"].items()),
    )


# ---------------------------------------------------------------------------
# 1. multi-slot mechanics
# ---------------------------------------------------------------------------
def test_two_slots_hold_two_concurrent() -> None:
    """Third overlapping signal under FCFS is a capacity reject."""
    frame = _signals(
        [0, 1, 2],
        ["long"] * 3,
        [99.0] * 3,
        [200.0] * 3,
        [0.5, 0.6, 0.9],
    )
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)
    assert [row["decision_idx"] for row in result["accepted"]] == [0, 1]
    assert [row["reason"] for row in result["rejected"]] == ["capacity"]
    assert result["summary"]["concurrent_max"] == 2
    assert int(np.max(result["exposure"]["gross"])) == 2


def test_exit_frees_slot() -> None:
    """A decision bar past an exit re-uses the lowest freed slot."""
    frame = _signals(
        [0, 1, 11],
        ["long"] * 3,
        [99.0, 99.0, 99.0],
        [104.0, 200.0, 200.0],
        [0.5, 0.6, 0.7],
    )
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)
    accepted_decisions = [row["decision_idx"] for row in result["accepted"]]
    assert accepted_decisions == [0, 1, 11]
    assert [row["slot"] for row in result["accepted"]] == [0, 1, 0]
    assert result["rejected"] == []


def test_per_slot_attribution_sums_to_equity() -> None:
    """per_slot trades count and ev_r cover the whole book."""
    frame = _signals(
        [0, 1, 11],
        ["long"] * 3,
        [99.0, 99.0, 99.0],
        [104.0, 200.0, 200.0],
        [0.5, 0.6, 0.7],
    )
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)
    assert len(result["per_slot"]) == 2
    assert result["per_slot"][0]["trades"] == 2
    assert result["per_slot"][1]["trades"] == 1
    total_ev = math.fsum(entry["ev_r"] for entry in result["per_slot"])
    assert math.isclose(total_ev, result["final_equity_r"], rel_tol=1e-12)
    assert result["per_slot"][0]["slot"] == 0
    assert result["per_slot"][1]["slot"] == 1


def test_priority_wins_same_bar() -> None:
    """Same-bar candidates compete on priority (one slot)."""
    frame = _signals(
        [0, 0],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.1, 0.9],
        priorities=[0.1, 0.9],
    )
    result = run_portfolio_replay(frame, _world(), slots=1, hold_bars=4)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["priority"] == 0.9
    assert result["rejected"][0]["reason"] == "capacity"


def test_cooldown_blocks_then_frees() -> None:
    """cooldown=2 blocks decisions exit+1, frees at exit+2."""
    frame = _signals(
        [0, 5, 6],
        ["long"] * 3,
        [99.0] * 3,
        [200.0] * 3,
        [0.5, 0.6, 0.7],
    )
    result = run_portfolio_replay(
        frame, _world(), slots=1, hold_bars=4, cooldown=2
    )
    accepted_decisions = [row["decision_idx"] for row in result["accepted"]]
    assert accepted_decisions == [0, 6]
    assert [row["reason"] for row in result["rejected"]] == ["cooldown"]


def test_open_ended_trade_holds_slot_to_data_end() -> None:
    """No exit inside the data: slot held to the end, credit zero."""
    opens = np.full(8, 100.0)
    panel = _panel("solo", opens, opens + 0.5, opens - 0.5, opens)
    frame = _signals([6], ["long"], [99.0], [200.0], [0.5])
    result = run_portfolio_replay(frame, panel, slots=2, hold_bars=4)
    assert result["summary"]["n_unresolved"] == 1
    row = result["accepted"][0]
    assert row["exit_idx"] == -1
    assert math.isnan(row["r_pess"])
    assert row["credit_r"] == 0.0
    assert row["credit_bar"] == 7  # the last bar
    assert result["equity_curve"].tolist() == [0.0]
    assert result["exposure"]["gross"].tolist()[7] == 1.0


def test_unsorted_input_replays_identically() -> None:
    """The deterministic sort makes frame order irrelevant."""
    ordered = _signals(
        [0, 5, 11],
        ["long"] * 3,
        [99.0] * 3,
        [200.0] * 3,
        [0.5, 0.6, 0.7],
    )
    shuffled = ordered.iloc[::-1].reset_index(drop=True)
    first = run_portfolio_replay(ordered, _world(), slots=2, hold_bars=4)
    second = run_portfolio_replay(shuffled, _world(), slots=2, hold_bars=4)
    assert _fingerprint(first) == _fingerprint(second)


def test_fill_past_data_rejected() -> None:
    """A signal whose fill bar would be past the data is a ValueError."""
    frame = _signals([29], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="past the data"):
        run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)


# ---------------------------------------------------------------------------
# 2. REPLACE-low vs REPLACE-high
# ---------------------------------------------------------------------------
def _replace_frame() -> pd.DataFrame:
    """Three longs built for the REPLACE scenarios.

    Two fill the slots; the third (score 0.9) strictly beats the worst
    open holder (score 0.5) at decision bar 2.
    """
    return _signals(
        [0, 1, 2],
        ["long"] * 3,
        [99.0] * 3,
        [200.0] * 3,
        [0.5, 0.6, 0.9],
    )


def test_replace_low_displaces_worst_open() -> None:
    """replace_low closes the lowest-score holder and takes its slot."""
    result = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    assert result["summary"]["n_replaced"] == 1
    assert [row["score"] for row in result["replaced"]] == [0.5]
    displaced = result["accepted"][0]
    assert displaced["replaced"] is True
    assert displaced["force_exit_idx"] == 2.0
    assert displaced["slot"] == result["accepted"][2]["slot"]


def test_replace_credits_zero_low_full_high() -> None:
    """Displaced credit: exactly 0 under low, full r_pess under high."""
    low = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    high = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_high",
    )
    displaced_low = low["accepted"][0]
    displaced_high = high["accepted"][0]
    assert displaced_low["credit_r"] == 0.0
    assert displaced_high["credit_r"] == displaced_high["r_pess"]
    assert math.isfinite(displaced_low["force_close_r"])


def test_policy_credit_unresolved_displaced_is_zero() -> None:
    """An unresolved displaced row credits 0 under BOTH policies.

    Unit regression: the ``replaced`` branch used to precede the
    unresolved check, so a displaced trade with ``exit_idx == -1``
    (its isolated sim never resolved, ``r_pess`` stale NaN) credited
    NaN under ``replace_high`` and poisoned the whole equity stream.
    Its realized partial P&L sits on ``force_close_r`` -- the credit
    itself is exactly 0.
    """
    row = {
        "replaced": True,
        "exit_idx": -1,
        "r_pess": float("nan"),
        "force_close_r": 0.0,
    }
    assert _policy_credit(row, "replace_high") == 0.0
    assert _policy_credit(row, "replace_low") == 0.0


def test_same_bar_displacement_unresolved_credits_zero() -> None:
    """Same-bar displacement of an unresolved trade: finite equity.

    End-to-end regression: two same-bar signals, one slot, replace_high
    -- the score-0.9 candidate displaces the score-0.5 fill the very
    bar it fills, so the displaced row keeps ``exit_idx == -1`` with a
    stale NaN ``r_pess``.  The old replaced-branch-first credit made
    the equity curve and drawdown NaN (run_portfolio_replay silently;
    run_portfolio raises on the NaN summary).
    """
    opens = np.full(8, 100.0)
    panel = _panel("solo", opens, opens + 0.5, opens - 0.5, opens)
    frame = _signals(
        [0, 0],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
    )
    result = run_portfolio_replay(
        # hold_bars=8 on the 8-bar world: the fill at bar 1 cannot
        # resolve (time exit past the data) -> exit_idx == -1, NaN r_pess
        frame,
        panel,
        slots=1,
        policy="replace_high",
        hold_bars=8,
    )
    assert result["summary"]["n_replaced"] == 1
    displaced = result["accepted"][0]
    assert displaced["exit_idx"] == -1
    assert displaced["replaced"] is True
    assert math.isnan(displaced["r_pess"])
    assert displaced["credit_r"] == 0.0
    assert bool(np.all(np.isfinite(result["equity_curve"])))
    assert math.isfinite(result["max_drawdown_r"])
    assert result["final_equity_r"] == 0.0


def test_replace_ev_bounds_formula() -> None:
    """EV(high) - EV(low) equals the displaced trades' full r_pess."""
    low = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    high = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_high",
    )
    replaced_r = math.fsum(float(row["r_pess"]) for row in low["replaced"])
    assert math.isclose(
        high["final_equity_r"] - low["final_equity_r"],
        replaced_r,
        rel_tol=1e-12,
    )


def test_force_close_r_is_reconstructed_partial_hold() -> None:
    """force_close_r equals simulate_signal truncated to the force bar.

    The bit-exact recompute pins the reconstruction contract: the
    replaced row carries the P&L to the replacement bar, never the
    stale isolated exit.
    """
    frame = _replace_frame()
    frame["atr_entry"] = 1.0
    panel = _world()
    result = run_portfolio_replay(
        frame, panel, slots=2, hold_bars=4, policy="replace_low"
    )
    displaced = result["accepted"][0]
    force_bar = int(displaced["force_exit_idx"])
    fill_idx = int(displaced["fill_idx"])
    opens = panel[("open", "solo")].to_numpy(dtype=np.float64)
    highs = panel[("high", "solo")].to_numpy(dtype=np.float64)
    lows = panel[("low", "solo")].to_numpy(dtype=np.float64)
    closes = panel[("close", "solo")].to_numpy(dtype=np.float64)
    expected = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        fill_idx,
        str(displaced["side"]),
        float(displaced["sl"]),
        float(displaced["tp"]),
        force_bar - fill_idx + 1,
        1.0,
        cost=DEFAULT_COST,
    )["r_pess"]
    assert displaced["force_close_r"] == expected
    assert displaced["exit_idx"] != force_bar  # the isolated exit stays


def test_score_tie_never_displaces() -> None:
    """Equal score is not a strictly better candidate."""
    frame = _signals(
        [0, 1, 2],
        ["long"] * 3,
        [99.0] * 3,
        [200.0] * 3,
        [0.5, 0.6, 0.5],
    )
    result = run_portfolio_replay(
        frame, _world(), slots=2, hold_bars=4, policy="replace_low"
    )
    assert result["summary"]["n_replaced"] == 0
    assert [row["reason"] for row in result["rejected"]] == ["inferior_score"]


def test_replaced_credit_bar_stays_original_exit() -> None:
    """Donor parity: the displaced trade credits at its own exit bar."""
    result = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    displaced = result["accepted"][0]
    assert displaced["credit_bar"] == displaced["exit_idx"]
    assert displaced["exit_idx"] == 4


def test_replaced_list_rows_carry_reason_and_force_fields() -> None:
    """The replaced ledger mirrors the engine's force-close contract."""
    result = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_high",
    )
    row = result["replaced"][0]
    assert row["reason"] == "replaced"
    assert row["force_exit_idx"] == 2.0
    assert math.isfinite(row["force_close_r"])
    assert row["credit_r"] == row["r_pess"]


# ---------------------------------------------------------------------------
# 3. correlation-aware admission
# ---------------------------------------------------------------------------
def test_replay_rejects_correlated_clone() -> None:
    """A same-instrument candidate cloning an open position is rejected."""
    proxy = np.linspace(-1.0, 1.0, 25)
    frame = _signals(
        [0, 2],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        instruments=["AAA", "AAA"],
        proxies=[proxy, proxy * 1.1],
    )
    result = run_portfolio_replay(
        frame, _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
    )
    assert [row["reason"] for row in result["rejected"]] == [
        "correlation_cap"
    ]
    assert result["summary"]["n_correlation_rejects"] == 1


def test_replay_admits_cross_instrument_candidate() -> None:
    """A candidate on a DIFFERENT instrument is never proxy-compared.

    Regression pin for ATR-02: the replay used to pair the candidate's
    proxy with every open holder's snapshot regardless of instrument,
    so an independent instrument carrying a clone-like proxy was
    rejected with |rho| ~ 1.  The filter consults same-instrument
    holders only; a cross-instrument candidate takes the free slot.
    """
    proxy = np.linspace(-1.0, 1.0, 25)
    frame = _signals(
        [0, 2],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        instruments=["AAA", "BBB"],
        proxies=[proxy, proxy * 1.1],
    )
    result = run_portfolio_replay(
        frame, _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
    )
    assert result["rejected"] == []
    assert result["summary"]["n_correlation_rejects"] == 0
    assert [row["instrument"] for row in result["accepted"]] == [
        "AAA",
        "BBB",
    ]


def test_replay_admits_stale_cross_instrument_clone() -> None:
    """A 100-bar clone candidate on ANOTHER instrument admits (ATR-02).

    The proxies are IDENTICAL and the snapshot is stale (10 bars vs
    100), so under the old instrument-blind extension this pair
    produced |rho| = 1 and a false reject; the snapshot-extension
    alignment itself stays same-instrument only.
    """
    snapshot = np.linspace(-1.0, 1.0, 10)  # ends at its decision bar
    longer = np.tile(snapshot, 10)  # 100 bars, same shape -> |rho| = 1
    frame = _signals(
        [9, 21],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        instruments=["AAA", "BBB"],
        proxies=[snapshot, longer],
    )
    result = run_portfolio_replay(
        frame, _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
    )
    assert result["rejected"] == []
    assert [row["instrument"] for row in result["accepted"]] == [
        "AAA",
        "BBB",
    ]


def test_replay_admits_independent_candidate() -> None:
    """An uncorrelated same-instrument-free candidate takes a slot."""
    zigzag = np.tile([1.0, -1.0], 13)[:25]
    frame = _signals(
        [0, 2],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        instruments=["AAA", "BBB"],
        proxies=[np.linspace(-1.0, 1.0, 25), zigzag],
    )
    result = run_portfolio_replay(
        frame, _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
    )
    assert result["rejected"] == []
    assert [row["instrument"] for row in result["accepted"]] == [
        "AAA",
        "BBB",
    ]


def test_replay_extends_stale_position_snapshots() -> None:
    """Rejection needs the snapshot EXTENSION, not plain tail alignment.

    The candidate cell is the open position's pattern ``head`` plus six
    fresh bars (``tail``) that DECORRELATE the shifted tail window:
    bar-for-bar extension reconstructs the candidate cell exactly
    (|rho| = 1 -> reject), while tail-alignment-without-extension pairs
    ``head`` against the shifted ``head[6:] + tail`` window
    (|rho| ~ 0.27 < cap 0.7 -> admit).  The verdict therefore flips on
    the extension branch alone -- a plain tail-aligned filter would
    admit the clone.
    """
    gen = np.random.default_rng(11)  # seed pinned: the split below holds
    head = gen.standard_normal(20)  # snapshot, ends at its decision bar
    tail = gen.standard_normal(6)
    longer = np.concatenate([head, tail])  # candidate cell, +6 bars
    no_extension_rho = float(spearmanr(head, longer[-20:]).statistic)
    assert abs(no_extension_rho) < 0.7  # tail alignment alone admits
    assert (
        float(spearmanr(longer, longer).statistic) > 0.7
    )  # extension reads |rho| = 1
    frame = _signals(
        [19, 25],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        instruments=["AAA", "AAA"],
        proxies=[head, longer],
    )
    result = run_portfolio_replay(
        frame, _multi_panel(), slots=2, hold_bars=100, corr_cap=0.7
    )
    assert [row["reason"] for row in result["rejected"]] == [
        "correlation_cap"
    ]


def test_same_instrument_stacking_is_correlated() -> None:
    """A second signal on an open instrument carries |rho| = 1."""
    proxy = np.linspace(-1.0, 1.0, 25)
    frame = _signals(
        [0, 2],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
        proxies=[proxy, proxy],
    )
    result = run_portfolio_replay(
        frame, _world(), slots=2, hold_bars=4, corr_cap=0.99
    )
    assert [row["reason"] for row in result["rejected"]] == [
        "correlation_cap"
    ]


def test_filter_rejects_clone_and_admits_independent() -> None:
    """Unit contract: True = admit, False = correlated above cap."""
    proxy = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02])
    other = np.array([0.02, 0.01, -0.01, 0.03, -0.02, 0.01, -0.03, 0.01])
    frame = pd.DataFrame(
        {
            "instrument": ["btc", "eth", "gold"],
            "r_proxy": [proxy, proxy * 1.1, other],
        }
    )
    streams = {"open_1": proxy}
    assert filter_by_correlation(frame, 1, streams, min_overlap=4) is False
    assert filter_by_correlation(frame, 2, streams, min_overlap=4) is True


def test_filter_insufficient_overlap_admits() -> None:
    """Too little shared history is inconclusive -> admit."""
    proxy = np.linspace(-1.0, 1.0, 8)
    frame = pd.DataFrame(
        {"instrument": ["btc", "eth"], "r_proxy": [proxy, proxy * 2.0]}
    )
    result = filter_by_correlation(
        frame, 1, {"open_1": proxy}, min_overlap=50
    )
    assert result is True


def test_filter_constant_stream_admits() -> None:
    """A degenerate (constant) proxy cannot correlate -> admit."""
    proxy = np.linspace(-1.0, 1.0, 30)
    flat = np.full(30, 7.5)
    frame = pd.DataFrame(
        {"instrument": ["btc", "eth"], "r_proxy": [proxy, flat]}
    )
    result = filter_by_correlation(frame, 1, {"open_1": flat}, min_overlap=20)
    assert result is True


def test_filter_no_open_positions_admits() -> None:
    """Empty stream mapping: nothing to correlate against."""
    proxy = np.linspace(-1.0, 1.0, 30)
    frame = pd.DataFrame({"instrument": ["btc"], "r_proxy": [proxy]})
    assert filter_by_correlation(frame, 0, {}, min_overlap=20) is True


def test_filter_validates_candidate_before_empty_early_return() -> None:
    """A NaN candidate proxy fails loudly even with no open positions.

    Regression: the candidate proxy was validated only AFTER the
    empty-streams early return, so a NaN candidate plus an empty
    mapping silently admitted (the NaN never reached a comparison).
    """
    proxy = np.linspace(-1.0, 1.0, 8)
    proxy[2] = np.nan
    frame = pd.DataFrame({"instrument": ["btc"], "r_proxy": [proxy]})
    with pytest.raises(ValueError, match="finite"):
        filter_by_correlation(frame, 0, {}, min_overlap=4)


def test_filter_cap_is_strict_inequality() -> None:
    """|rho| == cap admits; only a STRICTLY larger magnitude rejects."""
    proxy = np.linspace(-1.0, 1.0, 40)
    frame = pd.DataFrame(
        {"instrument": ["btc", "eth"], "r_proxy": [proxy, proxy]}
    )
    streams = {"open_1": proxy}
    assert filter_by_correlation(frame, 1, streams, corr_cap=1.0) is True
    result = filter_by_correlation(frame, 1, streams, corr_cap=0.999)
    assert result is False


def test_filter_requires_columns_and_valid_index() -> None:
    """Missing columns / out-of-range candidate_idx raise ValueError."""
    proxy = np.linspace(-1.0, 1.0, 30)
    frame = pd.DataFrame({"instrument": ["btc"], "r_proxy": [proxy]})
    with pytest.raises(ValueError, match="r_proxy"):
        filter_by_correlation(
            pd.DataFrame({"instrument": ["btc"]}), 0, {"open": proxy}
        )
    with pytest.raises(ValueError, match="past the frame"):
        filter_by_correlation(frame, 5, {"open": proxy})
    with pytest.raises(ValueError, match="instrument"):
        filter_by_correlation(
            pd.DataFrame({"r_proxy": [proxy]}), 0, {"open": proxy}
        )


def test_filter_validates_constants() -> None:
    """corr_cap / min_overlap / window validation happens up front."""
    proxy = np.linspace(-1.0, 1.0, 30)
    frame = pd.DataFrame(
        {"instrument": ["btc", "eth"], "r_proxy": [proxy, proxy]}
    )
    with pytest.raises(ValueError, match="corr_cap"):
        filter_by_correlation(frame, 1, {"open": proxy}, corr_cap=1.5)
    with pytest.raises(ValueError, match="corr_cap"):
        filter_by_correlation(frame, 1, {"open": proxy}, corr_cap=-0.1)
    with pytest.raises(ValueError, match="min_overlap"):
        filter_by_correlation(frame, 1, {"open": proxy}, min_overlap=1)
    with pytest.raises(ValueError, match="window"):
        filter_by_correlation(frame, 1, {"open": proxy}, window=0)


def test_corr_requires_proxy_column() -> None:
    """Enabling corr_cap without an r_proxy column is a ValueError."""
    frame = _signals(
        [0, 2],
        ["long", "long"],
        [99.0, 99.0],
        [200.0, 200.0],
        [0.5, 0.9],
    )
    with pytest.raises(ValueError, match="r_proxy"):
        run_portfolio_replay(
            frame, _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
        )


# ---------------------------------------------------------------------------
# 4. event equity and exposure
# ---------------------------------------------------------------------------
def test_equity_is_bitwise_cumsum_of_credits() -> None:
    """equity_curve == cumsum(credits ordered by (credit_bar, seq))."""
    frame = _signals(
        [0, 1, 2, 11],
        ["long"] * 4,
        [99.0] * 4,
        [104.0, 200.0, 200.0, 200.0],
        [0.5, 0.6, 0.9, 0.4],
    )
    result = run_portfolio_replay(
        frame, _world(), slots=2, hold_bars=4, policy="replace_low"
    )
    ordered = sorted(result["accepted"], key=lambda row: row["credit_bar"])
    credited = np.asarray(
        [row["credit_r"] for row in ordered], dtype=np.float64
    )
    expected = np.cumsum(credited)
    assert result["equity_curve"].tolist() == expected.tolist()
    assert len(result["equity_curve"]) == len(result["accepted"])


def test_max_drawdown_matches_equity_recompute() -> None:
    """The reported drawdown is the leading-zero-peak max shortfall."""
    frame = _signals(
        [0, 1, 11, 12],
        ["long"] * 4,
        [99.0] * 4,
        [104.0, 200.0, 104.0, 200.0],
        [0.5, 0.6, 0.7, 0.8],
    )
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)
    equity = result["equity_curve"]
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    assert result["max_drawdown_r"] == float(np.max(peaks[1:] - equity))
    assert result["final_equity_r"] == float(equity[-1])


def test_empty_signals_empty_scaffold() -> None:
    """No signals: an honest all-zero scaffold, no gates."""
    frame = _signals([], [], [], [], [])
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)
    assert result["accepted"] == []
    assert result["rejected"] == []
    assert result["replaced"] == []
    assert result["equity_curve"].size == 0
    assert result["max_drawdown_r"] == 0.0
    assert result["final_equity_r"] == 0.0
    assert result["per_slot"] == [
        {"slot": 0, "trades": 0, "ev_r": 0.0},
        {"slot": 1, "trades": 0, "ev_r": 0.0},
    ]
    assert result["exposure"]["gross"].tolist() == [0.0] * 30
    assert result["exposure"]["net"].tolist() == [0.0] * 30
    assert result["summary"]["n_accepted"] == 0
    assert result["summary"]["concurrent_max"] == 0
    assert result["gates"] == []
    assert result["portfolio_verdict"] == "not_evaluated"


def test_exposure_gross_and_net_series_exact() -> None:
    """Gross counts held positions; net is the signed sum."""
    frame = _signals(
        [0, 3],
        ["long", "short"],
        [90.0, 110.0],
        [110.0, 90.0],
        [0.5, 0.6],
    )
    result = run_portfolio_replay(frame, _world(), slots=2, hold_bars=10)
    gross = result["exposure"]["gross"].tolist()
    net = result["exposure"]["net"].tolist()
    # long holds bars 1..10, short bars 4..13 (both time exits): the
    # legs overlap bars 4..10
    expected_gross = [0.0, 1.0, 1.0, 1.0] + [2.0] * 7 + [1.0] * 3
    expected_gross = expected_gross + [0.0] * 16
    expected_net = [0.0, 1.0, 1.0, 1.0] + [0.0] * 7 + [-1.0] * 3
    expected_net = expected_net + [0.0] * 16
    assert gross == expected_gross
    assert net == expected_net
    assert (
        result["exposure"]["concurrent"].tolist()
        == result["exposure"]["gross"].tolist()
    )


def test_exposure_stays_bounded_by_slots() -> None:
    """gross/net never exceed the slot cap, even with replacements."""
    result = run_portfolio_replay(
        _replace_frame(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    gross = result["exposure"]["gross"]
    net = result["exposure"]["net"]
    assert bool(np.all(gross <= 2.0))
    assert bool(np.all(np.abs(net) <= 2.0))
    # the displaced trade ends at its force bar (2); the successor
    # fills on bar 3, so the hand count around the replacement holds
    assert gross.tolist()[0:7] == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0]


# ---------------------------------------------------------------------------
# 5. portfolio gates
# ---------------------------------------------------------------------------
def test_gates_report_structure() -> None:
    """Fixed gate order and the frozen report keys."""
    report = portfolio_gates(
        np.cumsum(_DRIFT),
        _DRIFT,
        min_trades=10,
        n_trials=1,
        concurrent=np.full(10, 2.0),
        net_exposure=np.tile([1.0, -1.0], 5),
    )
    assert frozenset(report) == {
        "gates",
        "portfolio_verdict",
        "n_trades",
        "min_trades",
        "prereg_hash",
    }
    assert report["prereg_hash"] is None  # legacy kwargs path: unanchored
    assert [gate["gate"] for gate in report["gates"]] == [
        "deflated_sharpe",
        "drawdown_floor",
        "side_concentration",
    ]
    for gate in report["gates"]:
        assert frozenset(gate) == {
            "gate",
            "triggered",
            "value",
            "floor",
            "reason",
        }
    assert report["n_trades"] == 10
    assert report["min_trades"] == 10


def test_dsr_gate_delegates_exactly() -> None:
    """The DSR value is deflated_sharpe_from_moments of the moments.

    The trial variance fed to the formula is the gate's own single-
    stream estimate ``denom2 / (n_obs - 1)`` in per-period SR units
    (ATR-01), recomputed here independently.
    """
    ev_stream = np.arange(10, dtype=np.float64) * 0.05
    spread = float(np.std(ev_stream, ddof=1))
    sr = float(np.mean(ev_stream) / spread)
    skew_val = float(skew(ev_stream))
    kurt_val = float(kurtosis(ev_stream, fisher=False))
    denom2 = 1.0 - skew_val * sr + ((kurt_val - 1.0) / 4.0) * sr * sr
    expected = deflated_sharpe_from_moments(
        sr, 10, skew_val, kurt_val, denom2 / 9.0, n_trials=3
    )
    report = portfolio_gates(
        np.cumsum(ev_stream),
        ev_stream,
        dsr_min=0.90,
        min_trades=10,
        n_trials=3,
    )
    dsr_gate = report["gates"][0]
    assert dsr_gate["value"] == expected
    assert dsr_gate["floor"] == 0.90
    assert dsr_gate["triggered"] is False


def test_dsr_gate_healthy_book_passes_with_multiple_trials() -> None:
    """Annual Sharpe ~3 over 500 periods clears the 0.95 floor at 5 trials.

    Regression pin for ATR-01: the gate used to hard-wire a UNIT trial
    variance, i.e. sibling dispersion of one per-period SR unit -- an
    annual Sharpe of ~18-20 on a daily grid, which deflated even a
    real book below any floor.  The variance now comes from the stream
    itself, so a healthy stream passes with the multiple-testing
    penalty actually applied (n_trials=5, not the gate-bypassing 1).
    """
    daily_sr = 3.0 / math.sqrt(252.0)
    half_step = 0.01
    mean_step = daily_sr * half_step
    ev_stream = np.tile(
        [mean_step - half_step, mean_step + half_step], 250
    ).astype(np.float64)
    assert ev_stream.size == 500
    spread = float(np.std(ev_stream, ddof=1))
    sr = float(np.mean(ev_stream) / spread)
    assert sr == pytest.approx(daily_sr, rel=1e-2)
    skew_val = float(skew(ev_stream))
    kurt_val = float(kurtosis(ev_stream, fisher=False))
    denom2 = 1.0 - skew_val * sr + ((kurt_val - 1.0) / 4.0) * sr * sr
    expected = deflated_sharpe_from_moments(
        sr, 500, skew_val, kurt_val, denom2 / 499.0, n_trials=5
    )
    report = portfolio_gates(np.cumsum(ev_stream), ev_stream, min_trades=500)
    dsr_gate = report["gates"][0]
    assert dsr_gate["value"] == expected
    assert expected >= 0.95
    assert dsr_gate["triggered"] is False
    assert report["portfolio_verdict"] == "SURVIVES"


def test_dsr_gate_noise_stream_triggers() -> None:
    """A zero-SR stream cannot clear the 0.95 floor: fail CLOSED."""
    noise = np.tile([1.0, -1.0], 5).astype(np.float64)
    report = portfolio_gates(
        np.cumsum(noise), noise, min_trades=10, n_trials=1
    )
    dsr_gate = report["gates"][0]
    assert dsr_gate["triggered"] is True
    assert report["portfolio_verdict"] == "DEAD(deflated_sharpe)"


def test_dsr_gate_nan_is_fail_closed() -> None:
    """Too few observations: NaN DSR triggers, honestly."""
    short = np.array([0.1, -0.2, 0.3, 0.1, -0.1])
    report = portfolio_gates(
        np.cumsum(short), short, min_trades=0, n_trials=1
    )
    dsr_gate = report["gates"][0]
    assert math.isnan(dsr_gate["value"])
    assert dsr_gate["triggered"] is True
    assert "fail-closed" in dsr_gate["reason"]


def test_dsr_gate_boundary_equal_dsr_triggers() -> None:
    """DSR exactly AT the floor triggers (the prereg's '>' is exclusive).

    Boundary pin: the prereg comparison reads "dsr passes when above
    dsr_min", so equality must fail.  The engine used to trigger on a
    strict ``dsr < dsr_min``, letting an exactly-at-floor DSR pass.
    """
    ev_stream = np.arange(10, dtype=np.float64) * 0.05
    spread = float(np.std(ev_stream, ddof=1))
    sr = float(np.mean(ev_stream) / spread)
    skew_val = float(skew(ev_stream))
    kurt_val = float(kurtosis(ev_stream, fisher=False))
    denom2 = 1.0 - skew_val * sr + ((kurt_val - 1.0) / 4.0) * sr * sr
    expected = deflated_sharpe_from_moments(
        sr, 10, skew_val, kurt_val, denom2 / 9.0, n_trials=3
    )
    assert math.isfinite(expected)
    at_floor = portfolio_gates(
        np.cumsum(ev_stream),
        ev_stream,
        dsr_min=expected,
        min_trades=10,
        n_trials=3,
    )
    dsr_gate = at_floor["gates"][0]
    assert dsr_gate["value"] == expected
    assert dsr_gate["triggered"] is True
    # one ulp BELOW the observed DSR: strictly above -> passes
    above = portfolio_gates(
        np.cumsum(ev_stream),
        ev_stream,
        dsr_min=float(np.nextafter(expected, 0.0)),
        min_trades=10,
        n_trials=3,
    )
    assert above["gates"][0]["triggered"] is False


def test_drawdown_gate_boundary_equal_dd_triggers() -> None:
    """A drawdown exactly AT the floor breaches (consistency pin).

    The drawdown gate has always triggered on ``value <= floor``; this
    pins the exact-equality boundary (all values binary-exact: 1.0 -
    0.75 = 0.25) alongside the DSR and concentration pins.
    """
    credited = np.array([1.0, -0.25])
    report = portfolio_gates(
        np.cumsum(credited),
        credited,
        dsr_min=0.0,
        dd_floor=-0.25,
        min_trades=2,
        n_trials=1,
    )
    dd_gate = report["gates"][1]
    assert dd_gate["value"] == pytest.approx(-0.25)
    assert dd_gate["floor"] == pytest.approx(-0.25)
    assert dd_gate["triggered"] is True


def test_drawdown_floor_breach_triggers() -> None:
    """Reaching the floor breaches; the verdict names the gate.

    The EV stream stays healthy (finite DSR clears the zero floor) so
    the DRAWDOWN gate is the first killer and names the verdict.
    """
    credited = np.array(
        [3.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    report = portfolio_gates(
        np.cumsum(credited), credited, dsr_min=0.0, min_trades=10, n_trials=1
    )
    dd_gate = report["gates"][1]
    assert dd_gate["value"] == pytest.approx(-4.0)
    assert dd_gate["floor"] == -0.20
    assert dd_gate["triggered"] is True
    assert report["portfolio_verdict"] == "DEAD(drawdown_floor)"


def test_drawdown_floor_respected() -> None:
    """A shallow drawdown stays above the floor."""
    equity = np.array([1.0, 0.9, 1.2])
    report = portfolio_gates(
        equity,
        np.array([1.0, -0.1, 0.3]),
        min_trades=3,
        n_trials=1,
        dsr_min=0.0,
    )
    dd_gate = report["gates"][1]
    assert dd_gate["triggered"] is False
    assert dd_gate["value"] == pytest.approx(-0.1)


def test_concentration_gate_triggered() -> None:
    """All-one-side books trip the same-side share cap."""
    report = portfolio_gates(
        np.cumsum(_DRIFT),
        _DRIFT,
        min_trades=10,
        n_trials=1,
        dsr_min=0.0,
        concurrent=np.full(10, 2.0),
        net_exposure=np.full(10, 2.0),
    )
    conc_gate = report["gates"][2]
    assert conc_gate["value"] == pytest.approx(1.0)
    assert conc_gate["floor"] == pytest.approx(0.80)
    assert conc_gate["triggered"] is True
    assert report["portfolio_verdict"] == "DEAD(side_concentration)"


def test_concentration_gate_balanced_passes() -> None:
    """A 3:1 split (0.75) under the 0.80 cap does not trigger."""
    report = portfolio_gates(
        np.cumsum(_DRIFT),
        _DRIFT,
        min_trades=10,
        n_trials=1,
        dsr_min=0.0,
        concurrent=np.full(10, 4.0),
        net_exposure=np.full(10, 2.0),
    )
    conc_gate = report["gates"][2]
    assert conc_gate["value"] == pytest.approx(0.75)
    assert conc_gate["triggered"] is False


def test_concentration_gate_boundary_equal_share_triggers() -> None:
    """Share exactly AT the cap triggers (the prereg's '<' is exclusive).

    10 concurrent with net 6: share = 0.5 + 6 / 20 = 0.80 against cap
    0.80.  The prereg comparison reads "concentration passes when below
    the cap", so equality must fail.  The engine used to trigger on a
    strict ``value > share_cap``, letting an exactly-at-cap book slip
    through.
    """
    report = portfolio_gates(
        np.cumsum(_DRIFT),
        _DRIFT,
        min_trades=10,
        n_trials=1,
        dsr_min=0.0,
        concurrent=np.full(10, 10.0),
        net_exposure=np.full(10, 6.0),
    )
    conc_gate = report["gates"][2]
    assert conc_gate["value"] == pytest.approx(0.80)
    assert conc_gate["floor"] == pytest.approx(0.80)
    assert conc_gate["triggered"] is True


def test_power_floor_counts_only_nonzero_credits() -> None:
    """A zero-credit row does not count toward the min_trades floor.

    Regression: the power floor counted ``equity.size`` rows, so a
    trade crediting exactly 0 (an unresolved trade, a
    ``replace_low``-displaced trade, a breakeven exit) padded the
    floor.  Here two rows carry ONE realized credit: the honest count
    is 1 < min_trades=2 -> UNDERPOWERED, while the old row count read
    2 and let the gates decide (verdict DEAD(deflated_sharpe) on this
    too-short EV stream).
    """
    report = portfolio_gates(
        np.array([2.0, 2.0]),
        np.array([2.0, 0.0]),
        min_trades=2,
        n_trials=1,
    )
    assert report["n_trades"] == 1
    assert report["portfolio_verdict"] == "UNDERPOWERED"


def test_concentration_gate_empty_book_is_zero() -> None:
    """Bars with no positions contribute nothing; share reads 0.0."""
    report = portfolio_gates(
        np.array([]),
        np.array([]),
        min_trades=0,
        n_trials=1,
        concurrent=np.zeros(4),
        net_exposure=np.zeros(4),
    )
    conc_gate = report["gates"][2]
    assert conc_gate["value"] == 0.0
    assert conc_gate["triggered"] is False
    assert "no positions" in conc_gate["reason"]


def test_concentration_gate_skipped_without_exposure() -> None:
    """Without exposure series the gate is skipped honestly (NaN)."""
    report = portfolio_gates(
        np.cumsum(_DRIFT), _DRIFT, min_trades=10, n_trials=1
    )
    conc_gate = report["gates"][2]
    assert math.isnan(conc_gate["value"])
    assert conc_gate["triggered"] is False
    assert "skipped" in conc_gate["reason"]


def test_verdict_underpowered() -> None:
    """Below min_trades the verdict is UNDERPOWERED, not SURVIVES."""
    report = portfolio_gates(
        np.cumsum(_DRIFT), _DRIFT, min_trades=11, n_trials=1
    )
    assert report["portfolio_verdict"] == "UNDERPOWERED"
    assert report["n_trades"] == 10
    assert report["min_trades"] == 11


def test_verdict_survives_healthy_world() -> None:
    """A drifted stream with balanced sides survives all three gates."""
    report = portfolio_gates(
        np.cumsum(_DRIFT),
        _DRIFT,
        min_trades=10,
        n_trials=1,
        concurrent=np.full(10, 2.0),
        net_exposure=np.tile([1.0, -1.0], 5),
    )
    assert report["portfolio_verdict"] == "SURVIVES"
    assert not any(gate["triggered"] for gate in report["gates"])


def test_gates_validate_constants_and_shapes() -> None:
    """Constant and shape validation raises before touching data."""
    equity = np.cumsum(_DRIFT)
    with pytest.raises(ValueError, match="dsr_min"):
        portfolio_gates(equity, _DRIFT, dsr_min=1.5)
    with pytest.raises(ValueError, match="dd_floor"):
        portfolio_gates(equity, _DRIFT, dd_floor=0.0)
    with pytest.raises(ValueError, match="max_same_side_share"):
        portfolio_gates(equity, _DRIFT, max_same_side_share=0.0)
    with pytest.raises(ValueError, match="n_trials"):
        portfolio_gates(equity, _DRIFT, n_trials=0)
    with pytest.raises(ValueError, match="1-D"):
        portfolio_gates(np.ones((3, 4)), _DRIFT)
    with pytest.raises(ValueError, match="finite"):
        portfolio_gates(equity, np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="equal length"):
        portfolio_gates(
            equity,
            _DRIFT,
            concurrent=np.ones(10),
            net_exposure=np.ones(9),
        )


# ---------------------------------------------------------------------------
# 6. the run_portfolio pipeline
# ---------------------------------------------------------------------------
def _proxy_frame() -> pd.DataFrame:
    """Signal frame with an uncorrelated proxy pair for gated runs."""
    growing = np.linspace(0.0, 1.0, 30)
    zigzag = np.tile([1.0, -1.0], 15)
    return _signals(
        [0, 3],
        ["long", "long"],
        [99.0, 99.0],
        [104.0, 200.0],
        [0.5, 0.6],
        instruments=["AAA", "BBB"],
        proxies=[growing, zigzag],
    )


def test_run_portfolio_attaches_gates_and_verdict() -> None:
    """run_portfolio = replay + gates; only the gate keys differ."""
    panel = _multi_panel()
    frame = _proxy_frame()
    pipelined = run_portfolio(
        frame, panel, slots=2, hold_bars=4, min_trades=1
    )
    plain = run_portfolio_replay(
        frame, panel, slots=2, hold_bars=4, corr_cap=0.7
    )
    assert pipelined["portfolio_verdict"] != "not_evaluated"
    assert len(pipelined["gates"]) == 3
    pipelined_equity = pipelined["equity_curve"].tolist()
    assert pipelined_equity == plain["equity_curve"].tolist()
    assert pipelined["summary"] == plain["summary"]
    assert (
        pipelined["exposure"]["gross"].tolist()
        == plain["exposure"]["gross"].tolist()
    )
    assert (
        pipelined["exposure"]["net"].tolist()
        == plain["exposure"]["net"].tolist()
    )
    # UNDERPOWERED is impossible with min_trades=1: the verdict comes
    # from the gates on the two credited trades
    assert pipelined["portfolio_verdict"].startswith(("SURVIVES", "DEAD"))


def test_replay_gates_stay_not_evaluated() -> None:
    """The raw replay carries no gates by contract."""
    result = run_portfolio_replay(
        _proxy_frame(), _multi_panel(), slots=2, hold_bars=4, corr_cap=0.7
    )
    assert result["gates"] == []
    assert result["portfolio_verdict"] == "not_evaluated"


def test_run_portfolio_on_empty_book_is_underpowered() -> None:
    """An empty book underpowers honestly (the DSR gate fails closed)."""
    empty = _signals([], [], [], [], [])
    report = run_portfolio(
        empty, _world(), slots=2, hold_bars=4, corr_cap=None
    )
    assert report["portfolio_verdict"] == "UNDERPOWERED"
    assert report["summary"]["n_accepted"] == 0
    dsr_gate = report["gates"][0]
    assert dsr_gate["triggered"] is True


# ---------------------------------------------------------------------------
# 6b. the frozen preregistration path
# ---------------------------------------------------------------------------
def _kill_prereg() -> dict[str, Any]:
    """A valid frozen prereg with NON-legacy gate thresholds.

    The thresholds (0.90 / -0.10 / 0.90) deliberately differ from the
    legacy kwargs defaults (0.95 / -0.20 / 0.80): the echoed gate
    floors prove the executed gates derive from the prereg.
    """
    return {
        "hypothesis": "Sector momentum portfolio (top-3 / bottom-3).",
        "gates": {
            "dsr_min": {
                "enabled": True,
                "threshold": 0.90,
                "comparison": ">",
            },
            "dd_floor": {
                "enabled": True,
                "threshold": -0.10,
                "comparison": ">",
            },
            "side_concentration": {
                "enabled": True,
                "threshold": 0.90,
                "comparison": "<",
            },
        },
        "slots": 3,
        "policy": "replace_low",
        "corr_cap": 0.70,
        "alpha": 0.05,
        "data_window": {"start": "2005-01-01", "end": "2024-12-31"},
        "snapshot_sha256": "a" * 64,
        "instruments": ["AAA", "BBB"],
        "frozen_date": "2026-11-06",
    }


def test_run_portfolio_prereg_binds_hash_and_thresholds() -> None:
    """The prereg path derives thresholds and embeds the verified hash.

    The gate floors echo the PRERG thresholds (0.90 / -0.10 / 0.90),
    not the legacy kwargs defaults (0.95 / -0.20 / 0.80) -- thresholds
    derive only from the hashed prereg -- and the verified hash lands
    in ``summary["prereg_hash"]``.
    """
    prereg = _kill_prereg()
    committed = hash_kill_prereg(prereg)
    result = run_portfolio(
        _proxy_frame(),
        _multi_panel(),
        slots=2,
        hold_bars=4,
        min_trades=1,
        prereg=prereg,
        prereg_hash=committed,
    )
    assert result["summary"]["prereg_hash"] == committed
    floors = [gate["floor"] for gate in result["gates"]]
    assert floors == [0.90, -0.10, 0.90]
    assert result["portfolio_verdict"].startswith(
        ("SURVIVES", "DEAD", "UNDERPOWERED")
    )


def test_run_portfolio_prereg_hash_mismatch_raises() -> None:
    """Any drift between prereg and committed hash fails loudly.

    Regression: the prereg hash was never bound to gate execution, so
    thresholds could silently differ from the hashed preregistration.
    """
    prereg = _kill_prereg()
    with pytest.raises(ValueError, match="hash mismatch"):
        run_portfolio(
            _proxy_frame(),
            _multi_panel(),
            slots=2,
            hold_bars=4,
            prereg=prereg,
            prereg_hash="0" * 64,
        )


def test_run_portfolio_prereg_requires_hash() -> None:
    """A prereg without its committed hash is refused (fail-closed)."""
    with pytest.raises(ValueError, match="prereg_hash is required"):
        run_portfolio(
            _proxy_frame(),
            _multi_panel(),
            slots=2,
            hold_bars=4,
            prereg=_kill_prereg(),
        )


def test_run_portfolio_prereg_disabled_gate_raises() -> None:
    """A disabled prereg gate must not silently skip a check."""
    prereg = _kill_prereg()
    prereg["gates"]["dd_floor"]["enabled"] = False
    with pytest.raises(ValueError, match="disabled"):
        run_portfolio(
            _proxy_frame(),
            _multi_panel(),
            slots=2,
            hold_bars=4,
            prereg=prereg,
            prereg_hash=hash_kill_prereg(prereg),
        )


def test_run_portfolio_prereg_invalid_dict_raises() -> None:
    """A malformed prereg is validated before anything runs."""
    prereg = _kill_prereg()
    prereg["alpha"] = 0.9  # outside the (0, 0.2] prereg domain
    with pytest.raises(ValueError, match="invalid prereg"):
        run_portfolio(
            _proxy_frame(),
            _multi_panel(),
            slots=2,
            hold_bars=4,
            prereg=prereg,
            prereg_hash=hash_kill_prereg(prereg),
        )


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------
def test_replay_deterministic_bit_for_bit() -> None:
    """Two identical runs agree bit-for-bit (same seeded inputs)."""
    frame = _signals(
        [0, 1, 2, 11, 12],
        ["long"] * 5,
        [99.0] * 5,
        [104.0, 200.0, 200.0, 104.0, 200.0],
        [0.5, 0.6, 0.9, 0.4, 0.55],
    )
    first = run_portfolio_replay(
        frame, _world(), slots=2, hold_bars=4, policy="replace_low"
    )
    second = run_portfolio_replay(
        frame.copy(),
        _world(),
        slots=2,
        hold_bars=4,
        policy="replace_low",
    )
    assert _fingerprint(first) == _fingerprint(second)
    assert (
        first["accepted"][0]["force_close_r"]
        == second["accepted"][0]["force_close_r"]
    )


def test_gated_pipeline_deterministic_bit_for_bit() -> None:
    """The full pipeline (corr filter + gates) reproduces exactly."""
    first = run_portfolio(
        _proxy_frame(),
        _multi_panel(),
        slots=2,
        hold_bars=4,
        min_trades=1,
    )
    second = run_portfolio(
        _proxy_frame(),
        _multi_panel(),
        slots=2,
        hold_bars=4,
        min_trades=1,
    )
    assert _fingerprint(first) == _fingerprint(second)
    assert first["portfolio_verdict"] == second["portfolio_verdict"]
    assert first["gates"] == second["gates"]


# ---------------------------------------------------------------------------
# 8. chaos guards
# ---------------------------------------------------------------------------
def test_nan_prices_rejected() -> None:
    """A non-finite price cell fails the panel validation loudly."""
    opens = np.full(30, 100.0)
    highs = np.full(30, 100.5)
    highs[5] = np.nan
    panel = _panel("solo", opens, highs, opens - 0.5, opens)
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="finite"):
        run_portfolio_replay(frame, panel, slots=2, hold_bars=4)


def test_flat_price_columns_rejected() -> None:
    """Without the (field, instrument) MultiIndex the panel is invalid."""
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    flat = pd.DataFrame(
        np.full((30, 4), 100.0),
        columns=["open", "high", "low", "close"],
    )
    with pytest.raises(ValueError, match="MultiIndex"):
        run_portfolio_replay(frame, flat, slots=2, hold_bars=4)


def test_nan_padded_panel_rejected() -> None:
    """A NaN-padded (mis-aligned) instrument grid fails loudly."""
    columns = pd.MultiIndex.from_product(
        [["open", "high", "low", "close"], ["AAA", "BBB"]]
    )
    flat_a = np.full(30, 100.0)
    flat_b = np.full(30, 100.0)
    flat_b[25:] = np.nan  # pandas concat pads shorter grids this way
    data = np.column_stack(
        [flat_a, flat_b, flat_a, flat_b, flat_a, flat_b, flat_a, flat_b]
    )
    panel = pd.DataFrame(data, columns=columns)
    frame = _signals(
        [0],
        ["long"],
        [99.0],
        [200.0],
        [0.5],
        instruments=["AAA"],
    )
    with pytest.raises(ValueError, match="finite"):
        run_portfolio_replay(frame, panel, slots=2, hold_bars=4)


def test_missing_price_field_rejected() -> None:
    """An instrument without its full OHLC quadruple is invalid."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("open", "AAA"),
            ("high", "AAA"),
            ("close", "AAA"),
        ]
    )
    panel = pd.DataFrame(np.full((30, 3), 100.0), columns=columns)
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="missing price fields"):
        run_portfolio_replay(frame, panel, slots=2, hold_bars=4)


def test_unknown_instrument_rejected() -> None:
    """Signals may only reference instruments the panel carries."""
    frame = _signals(
        [0],
        ["long"],
        [99.0],
        [200.0],
        [0.5],
        instruments=["ZZZ"],
    )
    with pytest.raises(ValueError, match="price panel"):
        run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)


def test_nonstring_instrument_label_rejected() -> None:
    """A non-string level-1 label is a loud error, not a collision.

    Regression: the panel is keyed by the raw column level-1 value but
    signals are matched on ``str(instrument)``, so a non-string label
    (e.g. the int ``7``) used to surface only as a confusing "not in
    the price panel" -- or collide silently with its ``str()`` form --
    instead of being rejected as a type error.
    """
    columns = pd.MultiIndex.from_tuples(
        [("open", 7), ("high", 7), ("low", 7), ("close", 7)]
    )
    panel = pd.DataFrame(np.full((30, 4), 100.0), columns=columns)
    frame = _signals(
        [0],
        ["long"],
        [99.0],
        [200.0],
        [0.5],
        instruments=["7"],
    )
    with pytest.raises(ValueError, match="must be a string"):
        run_portfolio_replay(frame, panel, slots=2, hold_bars=4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slots": 0}, "slots"),
        ({"slots": 2.5}, "slots"),
        ({"cooldown": -1}, "cooldown"),
        ({"hold_bars": 0}, "hold_bars"),
        ({"policy": "replace"}, "policy"),
        ({"corr_cap": 1.5}, "corr_cap"),
    ],
)
def test_replay_rejects_bad_constants(
    kwargs: dict[str, Any], match: str
) -> None:
    """Invalid replay constants raise ValueError before the replay."""
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match=match):
        run_portfolio_replay(frame, _world(), **kwargs)


def test_missing_required_column_rejected() -> None:
    """Each required signal column is enforced."""
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="required columns"):
        run_portfolio_replay(
            frame.drop(columns=["score"]),
            _world(),
            slots=2,
            hold_bars=4,
        )


def test_bad_side_rejected() -> None:
    """Side is a strict long/short literal."""
    frame = _signals([0], ["up"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="side"):
        run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)


def test_nonfinite_score_rejected() -> None:
    """Scores must be finite (they drive the REPLACE contests)."""
    frame = _signals([0], ["long"], [99.0], [200.0], [float("nan")])
    with pytest.raises(ValueError, match="finite"):
        run_portfolio_replay(frame, _world(), slots=2, hold_bars=4)


def test_cost_must_be_cost_model() -> None:
    """A non-CostModel cost argument is a loud ValueError."""
    frame = _signals([0], ["long"], [99.0], [200.0], [0.5])
    with pytest.raises(ValueError, match="CostModel"):
        run_portfolio_replay(
            frame,
            _world(),
            slots=2,
            hold_bars=4,
            cost=0.001,  # type: ignore[arg-type]
        )


def test_nonfinite_proxy_rejected() -> None:
    """A NaN r_proxy cell fails the frame validation."""
    proxy = np.linspace(-1.0, 1.0, 25)
    proxy[3] = np.nan
    frame = _signals(
        [0],
        ["long"],
        [99.0],
        [200.0],
        [0.5],
        proxies=[proxy],
    )
    with pytest.raises(ValueError, match="finite"):
        run_portfolio_replay(
            frame, _world(), slots=2, hold_bars=4, corr_cap=0.7
        )
