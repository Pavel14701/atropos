"""Paper-protocol pins: every named issue is provably reachable.

The protocol has 11 named issues: 10 parity + 1 statistical
(dd_divergence).  Five explicit mutations are caught by the mutation
tests below (applied -> fail -> restored before commit); the remaining
tests are lawful pins without mutations.  Fixtures are deterministic
synthetic trade frames built by the helpers at the bottom of this
file; no network, no randomness beyond the protocol's own seed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.paper_protocol import (
    PaperVerdict,
    _paired_bootstrap_p,
    run_paper_protocol,
)
from atropos.ranking import paired_bootstrap_diff

_BAR_MIN = 15
_BASE = "2026-11-02 00:00"

_NORMALIZATION: dict[str, list[str]] = {
    "time_exit": ["mtm", "mtm_48", "expiry", "time_exit"],
    "stop_exit": ["stop", "sl"],
    "tp_exit": ["tp", "take_profit"],
    "funding_payment": ["funding_payment", "funding_settle"],
    "rebalance": ["rebalance", "periodic_close"],
    "early_close": ["early_close", "manual"],
    "exercise": ["exercise", "assigned"],
}


def _spec(leg: str = "c_zone", **overrides: Any) -> dict[str, dict[str, Any]]:
    """One-leg frozen spec per the protocol schema, window = data."""
    spec: dict[str, Any] = {
        "exit_rules": {"mtm_bars": 48, "tp_r": 3.0, "stop_buffer_atr": 0.25},
        "exit_type_normalization": {
            key: list(values) for key, values in _NORMALIZATION.items()
        },
        "fee": {"taker_bps_per_leg": 5.0, "round_trip": "single"},
        "universe": ["BTC", "ETH", "SOL"],
        "sizing_rule": "composite_buckets",
        "bar_delta_min": _BAR_MIN,
        "paper_window": {},
    }
    spec.update(overrides)
    return {leg: spec}


def _window_for(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Exact bounds of the given rows (frozen mirrors paper entries)."""
    entries = [row["entry_ts"] for row in rows]
    exits = [row["exit_ts"] for row in rows]
    return {"start": min(entries), "end": max(exits)}


_KIND_BY_INDEX = ("tp", "stop", "mtm")
_PNL_BY_KIND = {"tp": 2.8, "stop": -1.05, "mtm": 0.55}
_EXIT_PRICE_BY_KIND = {"tp": 103.0, "stop": 99.0, "mtm": 100.5}
_EXIT_BARS_BY_KIND = {"tp": 8, "stop": 6, "mtm": 48}


def _rows(
    count: int, coins: tuple[str, ...] = ("BTC", "ETH", "SOL")
) -> list[dict[str, Any]]:
    """Deterministic synthetic trades: tp/stop/mtm cycle, spread out."""
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp(_BASE, tz="UTC")
    for index in range(count):
        kind = _KIND_BY_INDEX[index % 3]
        entry = base + pd.Timedelta(minutes=30 * index)
        bars = _EXIT_BARS_BY_KIND[kind]
        rows.append(
            {
                "coin": coins[index % len(coins)],
                "entry_ts": entry,
                "exit_ts": entry + pd.Timedelta(minutes=_BAR_MIN * bars),
                "exit_type": kind,
                "pnl_r": _PNL_BY_KIND[kind],
                "fee_r": 0.09,
                "size": 1.0,
                "side": "long",
                "entry_price": 100.0,
                "exit_price": _EXIT_PRICE_BY_KIND[kind],
                "signal_ts": entry - pd.Timedelta(seconds=10),
                "funding_rate": 0.0001,
            }
        )
    return rows


def _paper_frame(
    rows: list[dict[str, Any]],
    leg: str = "c_zone",
    sizing_rule: str = "composite_buckets",
    bar_delta_min: int = _BAR_MIN,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["leg"] = leg
    frame["sizing_rule"] = sizing_rule
    frame.attrs["bar_delta_min"] = bar_delta_min
    return frame


def _frozen_dict(
    rows_by_leg: dict[str, list[dict[str, Any]]],
    spec: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for leg, rows in rows_by_leg.items():
        spec[leg]["paper_window"] = _window_for(rows)
        out[leg] = pd.DataFrame(rows)
    return out


def _run(
    paper: pd.DataFrame,
    frozen: dict[str, pd.DataFrame],
    spec: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> PaperVerdict:
    return run_paper_protocol(paper, frozen, spec, **kwargs)


def _issue_names(verdict: PaperVerdict) -> set[str]:
    report = verdict["parity_report"]
    return {issue["name"] for issue in report["issues"]}


# -- explicit mutations (applied -> fail -> restored) -----------------------


def test_parity_exit_mismatch() -> None:
    """Paper exits stop where frozen exited mtm -> PARITY_FAIL.

    Type-only divergence: exit_ts and pnl_r stay identical, so the
    type comparison is the ONLY path that can fire the bucket.

    Mutation: drop the exit_type comparison -> parity comes back
    clean -> the verdict flips to INSUFFICIENT_N (n<min) -> caught.
    """
    rows = _rows(6)
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    frozen_rows[1]["exit_type"] = "mtm"
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert verdict["killed_by_stage"] == 3
    assert "exit_mismatch" in _issue_names(verdict)


def test_parity_fee_mismatch() -> None:
    """Paper fees at 2x frozen (the DTE double-charge shape).

    Mutation: drop the fee comparison -> no fee_mismatch -> caught.
    """
    rows = _rows(6)
    paper = _paper_frame(rows)
    paper["fee_r"] = 0.18
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "fee_mismatch" in _issue_names(verdict)
    issue = next(
        item
        for item in verdict["parity_report"]["issues"]
        if item["name"] == "fee_mismatch"
    )
    assert issue["n_affected"] == 6


def test_insufficient_n_gate() -> None:
    """Clean 5-trade run with min_n_trades=20 -> INSUFFICIENT_N.

    Mutation: remove the n-gate -> statistical stage runs on tiny n
    and the verdict flips to SURVIVES_TO_LIVE -> caught.
    """
    rows = _rows(5)
    paper = _paper_frame(rows)
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "INSUFFICIENT_N"
    assert verdict["verdict_class"] == "INCONCLUSIVE"
    assert verdict["killed_by_stage"] == 5
    assert verdict["n_frozen_matched"] == 5


def test_statistical_fail_significant_drift() -> None:
    """Systematic -0.005R per trade, inside parity tolerance.

    Each matched pair differs by 0.005R (< parity_tolerance_r=0.01, so
    parity passes) but 40 trades accumulate -0.2R and the paired
    bootstrap separates it: p_le0 ~ 0 < 0.05 -> STATISTICAL_FAIL.

    Mutation: invert/ignore the p_le0 comparison -> verdict stays
    SURVIVES_TO_LIVE -> caught.
    """
    frozen_rows = _rows(40)
    paper_rows = [dict(row) for row in frozen_rows]
    for row in paper_rows:
        row["pnl_r"] = row["pnl_r"] - 0.005
    paper = _paper_frame(paper_rows)
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "STATISTICAL_FAIL"
    assert verdict["verdict_class"] == "NEEDS_ANALYSIS"
    assert verdict["killed_by_stage"] == 6
    assert verdict["statistical_report"]["mean_drift_p_le0"] < 0.05
    # the drift path is the p-value path: no named statistical issue
    assert verdict["statistical_report"]["issues"] == []


def test_survives_to_live_happy_path() -> None:
    """Clean 40-trade run with execution evidence -> SURVIVES_TO_LIVE.

    Mutation: break the final verdict assembly (wrong token, swapped
    class) -> caught by the token/class asserts.
    """
    rows = _rows(40)
    paper = _paper_frame(rows)
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "SURVIVES_TO_LIVE"
    assert verdict["verdict_class"] == "PASS"
    assert verdict["portfolio_verdict"] == "SURVIVES_TO_LIVE"
    assert verdict["killed_by_stage"] is None
    assert verdict["per_leg"] is None  # single-leg run
    assert verdict["facts"]["notes"] == (
        "11 parity + 1 statistical = 12 named issues"
    )
    assert verdict["facts"]["n_parity_issue_types"] == 11
    assert verdict["facts"]["n_statistical_issue_types"] == 1
    dd = verdict["dd_comparison"]
    assert set(dd) == {"paper_dd_r", "frozen_dd_r", "ratio", "breach"}
    assert dd["breach"] is False


# -- lawful pins (no mutations) ---------------------------------------------


def test_frozen_window_gap_hard_fail() -> None:
    """Paper entry outside the declared window -> frozen_window_gap.

    The window covers the five in-window trades only; the sixth paper
    trade sits one bar BEFORE the window start -> stage-2 kill.
    """
    rows = _rows(6)
    early = dict(rows[0])
    early["entry_ts"] = pd.Timestamp(_BASE, tz="UTC") - pd.Timedelta(
        minutes=_BAR_MIN
    )
    early["exit_ts"] = early["entry_ts"] + pd.Timedelta(minutes=8 * _BAR_MIN)
    in_window = rows[1:]
    paper = _paper_frame([early, *in_window])
    spec = _spec()
    spec["c_zone"]["paper_window"] = _window_for(in_window)
    frozen = _frozen_dict({"c_zone": [dict(row) for row in in_window]}, spec)
    verdict = _run(paper, frozen, spec)
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert verdict["killed_by_stage"] == 2
    assert "frozen_window_gap" in _issue_names(verdict)


def test_frozen_stale_span_gap() -> None:
    """Frozen outcomes from another era -> no overlap -> stage-2 kill.

    The anti-STALE trap: frozen outcomes must OVERLAP the declared
    window (coverage is deliberately NOT required -- a sparse frozen
    population near the edges is parity's business, not a gate).
    """
    rows = _rows(5)
    paper = _paper_frame(rows)
    stale_rows = [dict(row) for row in rows]
    for row in stale_rows:
        row["entry_ts"] = row["entry_ts"] - pd.Timedelta(days=14)
        row["exit_ts"] = row["exit_ts"] - pd.Timedelta(days=14)
    frozen = _frozen_dict({"c_zone": stale_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert verdict["killed_by_stage"] == 2
    assert "frozen_window_gap" in _issue_names(verdict)
    assert verdict["n_frozen_matched"] == 0


def test_feed_granularity_mismatch() -> None:
    """Paper feed 1m vs frozen spec 15m -> hard fail, never a tolerance."""
    rows = _rows(5)
    paper = _paper_frame(rows, bar_delta_min=1)
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert verdict["killed_by_stage"] == 1
    assert "feed_granularity_mismatch" in _issue_names(verdict)
    assert verdict["feed_granularity"] == {
        "paper": 1,
        "frozen": 15,
        "match": False,
    }


def test_sizing_rule_mismatch() -> None:
    """neutral_1.0 paper vs composite_buckets spec -> rule-level fail.

    The DTE trap: per-trade size would match (both 1.0) while the
    sizing RULE diverges; the rule-level check catches it regardless.
    """
    rows = _rows(5)
    paper = _paper_frame(rows, sizing_rule="neutral_1.0")
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "sizing_rule_mismatch" in _issue_names(verdict)
    assert verdict["sizing_rule"] == {
        "paper": "neutral_1.0",
        "frozen": "composite_buckets",
        "match": False,
    }


def test_dd_divergence_flagged() -> None:
    """Same trades, different frame ORDER -> paper DD 3x frozen DD.

    Parity matches per-trade (order-independent) while the cumulative
    paths diverge: paper books every loss first, frozen alternates.
    DD is computed on the frames' row order per v2 sect. 5.
    """
    frozen_rows = _rows(40)
    losses = [row for row in frozen_rows if row["pnl_r"] < 0]
    gains = [row for row in frozen_rows if row["pnl_r"] >= 0]
    paper_rows = losses + gains  # deepest drawdown first
    paper = _paper_frame(paper_rows)
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "STATISTICAL_FAIL"
    assert verdict["killed_by_stage"] == 6
    assert verdict["statistical_report"]["issues"] == ["dd_divergence"]
    dd = verdict["dd_comparison"]
    assert dd["breach"] is True
    assert dd["ratio"] > 1.5
    assert dd["paper_dd_r"] > dd["frozen_dd_r"]


def test_dd_comparison_populated_on_parity_fail() -> None:
    """B12: dd_comparison carries numbers even when parity kills.

    The post-mortem value of the protocol is the dd evidence on a
    PARITY_FAIL; a refactor dropping the computation must fail here.
    """
    rows = _rows(5)
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    frozen_rows[0]["exit_type"] = "mtm"
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    dd = verdict["dd_comparison"]
    assert set(dd) == {"paper_dd_r", "frozen_dd_r", "ratio", "breach"}
    assert dd["breach"] is False  # breach only rules inside stage 6
    assert dd["paper_dd_r"] > 0.0


def test_funding_rate_mismatch() -> None:
    """Carry leg: settlement rate off by 0.7bps at 0.5bps tolerance."""
    rows = _rows(5)
    for row in rows:
        row["exit_type"] = "funding_payment"
        row["pnl_r"] = 0.2
        row["exit_ts"] = row["entry_ts"] + pd.Timedelta(minutes=8 * _BAR_MIN)
        row["exit_price"] = 100.0
        row["funding_rate"] = 0.0001
    spec = _spec()
    spec["c_zone"]["funding"] = {
        "settlement_interval_hours": 8,
        "rate_tolerance_bps": 0.5,
    }
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    frozen_rows[2]["funding_rate"] = 0.00003  # 0.7bps away, tolerance 0.5
    frozen = _frozen_dict({"c_zone": frozen_rows}, spec)
    verdict = _run(paper, frozen, spec)
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "funding_rate_mismatch" in _issue_names(verdict)
    alignment = verdict["funding_alignment"]
    assert alignment is not None and alignment["n_matched"] == 5


def test_pnl_component_mismatch() -> None:
    """Options leg: total pnl equal, opposite delta/vega errors."""
    rows = _rows(5)
    for row in rows:
        row["pnl_r"] = 0.0
        for component in ("delta", "gamma", "theta", "vega", "fee"):
            row[f"pnl_{component}"] = 0.0
    spec = _spec()
    spec["c_zone"]["pnl_decomposition"] = {
        "enabled": True,
        "components": ["delta", "gamma", "theta", "vega", "fee"],
        "tolerance_per_component_r": 0.02,
    }
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    frozen_rows[1]["pnl_delta"] = 0.1
    frozen_rows[1]["pnl_vega"] = -0.1
    frozen = _frozen_dict({"c_zone": frozen_rows}, spec)
    verdict = _run(paper, frozen, spec)
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "pnl_component_mismatch" in _issue_names(verdict)
    decomposition = verdict["pnl_decomposition"]
    assert decomposition is not None and decomposition["n_matched"] == 5


def test_exit_type_normalization() -> None:
    """Aliases resolve to one canonical token; unknown token is loud."""
    rows = _rows(5)
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    # row 2 is paper-"mtm" -> frozen "expiry": both -> time_exit -> match
    frozen_rows[2]["exit_type"] = "expiry"
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert "exit_mismatch" not in _issue_names(verdict)
    # a token outside the normalization map refuses silently passing
    alien = _paper_frame(_rows(3))
    alien["exit_type"] = alien["exit_type"].astype(object)
    alien.loc[0, "exit_type"] = "alien_token"
    with pytest.raises(ValueError, match=r"alien_token|exit_type"):
        _run(alien, _frozen_dict({"c_zone": _rows(3)}, _spec()), _spec())


def test_unmatched_entry_issue() -> None:
    """Paper entry with no frozen counterpart -> unmatched_entry."""
    rows = _rows(5)
    extra = dict(rows[0])
    extra["entry_ts"] = rows[-1]["entry_ts"] + pd.Timedelta(minutes=30)
    extra["exit_ts"] = extra["entry_ts"] + pd.Timedelta(minutes=8 * _BAR_MIN)
    paper = _paper_frame([*rows, extra])
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "unmatched_entry" in _issue_names(verdict)
    assert verdict["n_paper_trades"] == 6
    assert verdict["n_frozen_matched"] == 5


def test_universe_violation_issue() -> None:
    """Paper trades a coin outside the frozen universe."""
    rows = _rows(5)
    extra = dict(rows[0])
    extra["coin"] = "NEAR"
    extra["entry_ts"] = rows[-1]["entry_ts"] + pd.Timedelta(minutes=30)
    extra["exit_ts"] = extra["entry_ts"] + pd.Timedelta(minutes=8 * _BAR_MIN)
    paper = _paper_frame([*rows, extra])
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, _spec())
    verdict = _run(paper, frozen, _spec())
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "universe_violation" in _issue_names(verdict)
    assert "unmatched_entry" not in _issue_names(verdict)


def test_parity_pass_parity_only_attestation() -> None:
    """PARITY_PASS = full pass with ALL operational checks skipped.

    Without entry/exit prices and signal_ts there is no execution-
    quality evidence: parity is attested (PASS class) but the verdict
    stays below SURVIVES_TO_LIVE, which demands execution evidence.
    """
    rows = _rows(40)
    slim = [
        {
            key: row[key]
            for key in ("coin", "entry_ts", "exit_ts", "exit_type", "pnl_r")
        }
        for row in rows
    ]
    paper = _paper_frame(slim)
    frozen_full = _frozen_dict(
        {"c_zone": [dict(row) for row in rows]}, _spec()
    )
    frozen_slim = frozen_full["c_zone"][
        ["coin", "entry_ts", "exit_ts", "exit_type", "pnl_r"]
    ]
    verdict = _run(paper, {"c_zone": frozen_slim}, _spec())
    assert verdict["overall_verdict"] == "PARITY_PASS"
    assert verdict["verdict_class"] == "PASS"
    report = verdict["operational_report"]
    assert report["checks_skipped"], "expected skipped operational checks"


def test_multi_leg_portfolio_aggregation() -> None:
    """Per-leg verdicts are not shadowed; portfolio uses the cascade."""
    good_rows = _rows(40)
    frozen_bad = _rows(5, coins=("BTC",))
    # defect the PAPER side only (frozen copies the rows first);
    # row 2 is paper-"mtm" by the cycle -> book it as a "stop" instead
    paper_bad = [dict(row) for row in frozen_bad]
    paper_bad[2]["exit_type"] = "stop"
    paper_bad[2]["exit_price"] = 99.0  # direction-consistent stop fill
    paper = pd.concat(
        [
            _paper_frame(good_rows, leg="c_zone"),
            _paper_frame(paper_bad, leg="a_trail"),
        ],
        ignore_index=True,
    )
    spec = _spec("c_zone")
    spec.update(_spec("a_trail"))
    frozen = _frozen_dict(
        {"c_zone": [dict(row) for row in good_rows], "a_trail": frozen_bad},
        spec,
    )
    verdict = _run(paper, frozen, spec)
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert verdict["portfolio_verdict"] == "PARITY_FAIL"
    assert verdict["per_leg"] is not None
    assert set(verdict["per_leg"]) == {"c_zone", "a_trail"}
    assert verdict["per_leg"]["c_zone"]["overall_verdict"] == (
        "SURVIVES_TO_LIVE"
    )
    assert verdict["per_leg"]["a_trail"]["overall_verdict"] == "PARITY_FAIL"
    assert "PARITY_FAIL" in verdict["kill_reason"]
    assert "a_trail" in verdict["kill_reason"]
    # per_leg entries are LegVerdicts: no per_leg key inside
    assert "per_leg" not in verdict["per_leg"]["c_zone"]


def test_local_bootstrap_matches_ranking() -> None:
    """The local bootstrap is behavior-pinned to ranking's helper.

    Same inputs + seed must give the same p-value (< 1e-9 apart); if
    ranking.paired_bootstrap_diff ever changes, this pin forces the
    local copy to be re-derived.
    """
    rng = np.random.default_rng(7)
    arm_a = rng.normal(0.3, 1.0, size=60)
    arm_b = arm_a - rng.normal(0.05, 0.5, size=60)
    arm_b[3] = np.nan  # non-finite pairs must be dropped identically
    local = _paired_bootstrap_p(arm_a, arm_b, n_boot=500, seed=7)
    reference = paired_bootstrap_diff(arm_a, arm_b, n_boot=500, seed=7)[
        "p_le0"
    ]
    assert abs(local - reference) < 1e-9


def test_paper_report_carries_verdict_metadata() -> None:
    """Interpretation carries the concrete numbers (M1 run-facts).

    Mutation: templates that return generic strings without the
    counts -> the number asserts fail -> caught.
    """
    rows = _rows(6)
    paper = _paper_frame(rows)
    frozen_rows = [dict(row) for row in rows]
    # row 2 is paper-"mtm" (cycle tp/stop/mtm) -> freeze it as "stop"
    frozen_rows[2]["exit_type"] = "stop"
    frozen = _frozen_dict({"c_zone": frozen_rows}, _spec())
    verdict = _run(paper, frozen, _spec(), label="dte_c_zone")
    assert str(verdict["n_paper_trades"]) in verdict["interpretation"]
    assert str(verdict["n_frozen_matched"]) in verdict["interpretation"]
    assert "exit_mismatch" in verdict["interpretation"]
    assert "exit_mismatch" in verdict["what_would_change"]
    assert verdict["next_step"]
    assert verdict["informative"] is True
    assert verdict["kill_reason"]
    assert verdict["label"] in verdict["interpretation"]


# -- input validation (ERROR-PATH LAW: every rung pinned) --------------------


def _expect_value_error(case: Callable[[], Any]) -> None:
    with pytest.raises(ValueError):
        case()


def test_sizing_activation_stale_fires() -> None:
    """Ledger large enough but mode still fallback -> runtime defect.

    The contract token matches by construction, yet the composite
    never activated despite n_ob_rows >= composite_activation_n —
    exactly the 'fix shipped but never switched on' scenario.
    """
    rows = _rows(5)
    paper = _paper_frame(rows)
    paper.attrs["n_ob_rows"] = 250
    paper.attrs["sizing_mode_active"] = "neutral_fallback"
    spec = _spec()
    spec["c_zone"]["composite_activation_n"] = 200
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, spec)
    verdict = _run(paper, frozen, spec)
    assert verdict["overall_verdict"] == "PARITY_FAIL"
    assert "sizing_activation_stale" in _issue_names(verdict)


def test_sizing_activation_not_stale_below_threshold() -> None:
    """n_ob_rows below activation n -> no issue (healthy fallback)."""
    rows = _rows(5)
    paper = _paper_frame(rows)
    paper.attrs["n_ob_rows"] = 12
    paper.attrs["sizing_mode_active"] = "neutral_fallback"
    spec = _spec()
    spec["c_zone"]["composite_activation_n"] = 200
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, spec)
    verdict = _run(paper, frozen, spec)
    assert "sizing_activation_stale" not in _issue_names(verdict)


def test_sizing_activation_skipped_without_attrs() -> None:
    """Missing attrs -> check goes to checks_skipped, never fires."""
    rows = _rows(5)
    paper = _paper_frame(rows)  # no n_ob_rows / sizing_mode_active attrs
    spec = _spec()
    spec["c_zone"]["composite_activation_n"] = 200
    frozen = _frozen_dict({"c_zone": [dict(row) for row in rows]}, spec)
    verdict = _run(paper, frozen, spec)
    skipped = " ".join(verdict["parity_report"].get("checks_skipped", []))
    assert "sizing_activation" in skipped
    assert "sizing_activation_stale" not in _issue_names(verdict)


def test_issue_taxonomy_boundary() -> None:
    """B11, structurally: 11 parity + 1 statistical, never mixed.

    The counts come from the module's own taxonomy constants, so a
    new issue type updates the facts note and this pin together.
    """
    from atropos.paper_protocol import (  # noqa: PLC0415  (test-local)
        _PARITY_ISSUE_ORDER,
    )

    assert len(_PARITY_ISSUE_ORDER) == 11
    assert "sizing_activation_stale" in _PARITY_ISSUE_ORDER
    assert "dd_divergence" not in _PARITY_ISSUE_ORDER


def test_error_paths() -> None:
    """Every validation rung raises ValueError (never silent)."""
    rows = _rows(3)
    base_paper = _paper_frame(rows)
    base_frozen = _frozen_dict(
        {"c_zone": [dict(row) for row in rows]}, _spec()
    )
    base_spec = _spec()

    # missing required paper column
    no_leg = base_paper.drop(columns=["leg"])
    _expect_value_error(lambda: _run(no_leg, base_frozen, base_spec))
    # missing sizing_rule column
    no_rule = base_paper.drop(columns=["sizing_rule"])
    _expect_value_error(lambda: _run(no_rule, base_frozen, base_spec))
    # missing attrs bar_delta_min
    no_attrs = base_paper.copy()
    no_attrs.attrs.pop("bar_delta_min")
    _expect_value_error(lambda: _run(no_attrs, base_frozen, base_spec))
    # missing spec key
    bad_spec = {key: dict(value) for key, value in base_spec.items()}
    bad_spec["c_zone"].pop("universe")
    _expect_value_error(lambda: _run(base_paper, base_frozen, bad_spec))
    # requested leg missing from frozen_outcomes
    _expect_value_error(
        lambda: _run(
            base_paper, base_frozen, base_spec, legs=["c_zone", "a_trail"]
        )
    )
    # reserved portfolio leg name: leg EXISTS in frozen but is reserved
    reserved_spec = _spec("portfolio")
    reserved_paper = _paper_frame(rows, leg="portfolio")
    reserved_paper.attrs["bar_delta_min"] = _BAR_MIN
    reserved_frozen = _frozen_dict(
        {"portfolio": [dict(row) for row in rows]}, reserved_spec
    )
    _expect_value_error(
        lambda: _run(
            reserved_paper,
            reserved_frozen,
            reserved_spec,
            legs=["portfolio"],
        )
    )
    # unknown exit_type token (not in normalization map)
    alien = _paper_frame(_rows(3))
    alien["exit_type"] = alien["exit_type"].astype(object)
    alien.loc[1, "exit_type"] = "mystery_exit"
    _expect_value_error(lambda: _run(alien, base_frozen, base_spec))
    # funding section without funding_rate column
    funding_spec = _spec()
    funding_spec["c_zone"]["funding"] = {
        "settlement_interval_hours": 8,
        "rate_tolerance_bps": 0.5,
    }
    no_rate = base_paper.drop(columns=["funding_rate"])
    _expect_value_error(lambda: _run(no_rate, base_frozen, funding_spec))
    # decomposition enabled without pnl_* columns
    decomp_spec = _spec()
    decomp_spec["c_zone"]["pnl_decomposition"] = {
        "enabled": True,
        "components": ["delta", "gamma", "theta", "vega", "fee"],
        "tolerance_per_component_r": 0.02,
    }
    _expect_value_error(lambda: _run(base_paper, base_frozen, decomp_spec))
    # invalid numeric parameters
    _expect_value_error(
        lambda: _run(base_paper, base_frozen, base_spec, n_bootstrap=0)
    )
    _expect_value_error(
        lambda: _run(base_paper, base_frozen, base_spec, dd_breach_ratio=0.0)
    )
    _expect_value_error(
        lambda: _run(base_paper, base_frozen, base_spec, min_n_trades=0)
    )
