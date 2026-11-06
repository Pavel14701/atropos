"""Portfolio-level controls on the 10-sector SPDR panel (preregistered).

Public portfolio controls for the multi-slot portfolio layer
(:mod:`atropos.portfolio`): does the gate machinery separate a real
cross-sectional book from noise wearing the same clothes?  Three
phases, each with a frozen preregistration printed BEFORE its run
(the ``run_example.py`` discipline, applied at the portfolio level):

PHASE A -- null (FPR calibration): books built from the momentum
portfolio's exact bars, cadence, cohort sizes and stop geometry but a
RANDOM cross-sectional selection (3 random longs / 3 random shorts per
month, scores ``U(0, 1)``), replayed through ``run_portfolio_replay``
+ ``portfolio_gates`` over many seeded worlds.  The preregistered
expectation: NO world is certified (verdict ``SURVIVES``); the share
that is, is the machinery's false-certification rate (FPR) -- reported
honestly, never pinned (it is a random quantity by nature).

PHASE B -- momentum portfolio (positive control): the real
cross-sectional 12-1 book (long top-3 / short bottom-3 sector ETFs by
trailing momentum) through ``run_portfolio`` -- full machinery
including correlation admission.  The preregistered question is
whether the gates certify it; the honest verdict (``SURVIVES`` or
``DEAD``) is recorded either way -- both calibrate the machinery,
provided the gates fire for the right reasons.

PHASE C -- random portfolio (negative control): one seeded noise book
through the SAME full machinery as Phase B.  The preregistered
expectation: NOT ``SURVIVES``.  A ``SURVIVES`` here certifies pure
noise and is the portfolio layer's false-certification finding.

Usage::

    uv run python examples/real_data/run_portfolio_controls.py          # full
    uv run python examples/real_data/run_portfolio_controls.py --quick  # CI

``--quick`` reduces ONLY the Phase A world count (20 vs 100 seeded
nulls); every other constant is identical.  The full run is the
contract.  Data: the committed sector snapshots (see
strategies/sector_momentum.py and README.md for provenance; the panel
starts at the XLRE inception 2015-10-08).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from atropos import generate_prereg  # noqa: E402
from atropos.portfolio import (  # noqa: E402
    GateReport,
    GateVerdict,
    PortfolioResult,
    portfolio_gates,
    run_portfolio,
    run_portfolio_replay,
)
from atropos.prereg_generator import validate_prereg  # noqa: E402
from strategies.sector_momentum import (  # noqa: E402
    build_null_signals,
    build_portfolio_dataset,
    cross_sectional_momentum_12_1,
    load_prices,
)

FREEZE_DATE = "2026-11-06"
"""Date the three preregs were frozen; explicit so reruns render
identical text instead of silently moving the stamp to today."""

# --- frozen portfolio configuration (shared by all three phases) ---
SLOTS = 6
"""One slot per booked position: 3 longs + 3 shorts per monthly cohort."""
POLICY = "fcfs"
"""Capacity rejection -- replacement politics would confound the controls."""
CORR_CAP = 0.90
"""Correlation admission cap (Blocks same-instrument stacking at
|rho|=1 while admitting structural cross-sector correlation)."""
HOLD_BARS = 21
"""Maximum hold (~one month, the cohort cadence)."""
DSR_MIN = 0.95
"""Deflated-Sharpe floor of the per-bar EV stream."""
DD_FLOOR = -15.0
"""Drawdown floor in R units: ~2.5 fully stopped-out cohorts from the
peak -- a disaster bar, not the edge discriminator (the DSR gate does
that)."""
MAX_SAME_SIDE = 0.80
"""Same-side concentration cap."""
N_TRIALS = 1
"""One preregistered construction -- no multiple-testing credit."""
MIN_TRADES = 100
"""Credited-trade floor for a verdict other than UNDERPOWERED."""
EV_PERIOD_BARS = 1
"""Per-bar EV buckets feeding the DSR gate."""

COST_BPS = 2
"""Flat round-trip friction (baked into the datasets' cost model)."""

NULL_BASE_SEED = 20261106
"""World w of Phase A uses seed ``NULL_BASE_SEED + w``."""
PHASE_C_SEED = 20261107
"""Fixed seed of the Phase C single noise world."""
FULL_NULL_WORLDS = 100
"""Phase A worlds in the full (contract) run."""
QUICK_NULL_WORLDS = 20
"""Phase A worlds in the --quick CI preview."""

FROZEN_CONFIG: dict[str, Any] = {
    "slots": SLOTS,
    "policy": POLICY,
    "corr_cap": CORR_CAP,
    "hold_bars": HOLD_BARS,
    "dsr_min": DSR_MIN,
    "dd_floor_r": DD_FLOOR,
    "max_same_side_share": MAX_SAME_SIDE,
    "n_trials": N_TRIALS,
    "min_trades": MIN_TRADES,
    "ev_period_bars": EV_PERIOD_BARS,
    "cost_bps": COST_BPS,
}

_DATA_NOTE = (
    "data: committed Select Sector SPDR snapshots data/<TICKER>.csv "
    "(XLF XLE XLK XLI XLV XLY XLP XLRE XLU XLB), adjusted OHLC, "
    "sha256-pinned; request window 2005-01-01..2025-01-01 but the "
    "real-estate fund (XLRE) incepted 2015-10-08, so the shared "
    "finite bar grid starts 2015-10-08 (2323 bars)"
)
_AUDIT_NOTE = (
    "construction audit (pre-run, diagnostic only): the IDEALIZED "
    "no-stop monthly long-short on the same scores measures ~0 edge "
    "on this window (annualized mean -0.7%, Sharpe about -0.1) -- "
    "the 2005-2015 momentum golden decade is unavailable below the "
    "XLRE inception, so a DEAD verdict on the engine book would be "
    "the machinery refusing a signal with no measured sample edge"
)


def _spec_hash(params: dict[str, Any]) -> str:
    """First 8 hex chars of sha256 over the frozen params (sans hash)."""
    frozen = {key: value for key, value in params.items() if key != "hash"}
    canonical = json.dumps(frozen, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _prereg_text(params: dict[str, Any]) -> str:
    """Validate one frozen phase spec and render its prereg document.

    Raises:
        ValueError: if the frozen params ever fail validation (the
            failure must be loud, never a softened spec).
    """
    complete = {**params, "hash": _spec_hash(params)}
    problems = validate_prereg(complete)
    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(f"frozen prereg params are invalid:\n  - {detail}")
    return generate_prereg(complete, date=FREEZE_DATE)


PHASE_A_PARAMS: dict[str, Any] = {
    "name": "portfolio_null_admission_fpr",
    "hypothesis": (
        "NEGATIVE CONTROL (FPR calibration of the portfolio gates). "
        "Books built from the momentum portfolio's exact bars, monthly "
        "cadence, cohort sizes and 2xATR stop geometry, but with a "
        "RANDOM cross-sectional selection (3 random longs and 3 "
        "random shorts per cohort, admission scores U(0,1)), carry no "
        "edge.  Replayed through run_portfolio_replay (slots=6, fcfs, "
        "correlation filter off) and judged by portfolio_gates "
        "(dsr_min 0.95 with n_trials=1, dd_floor -15 R, same-side cap "
        "0.80, min_trades 100), they must NOT be certified.  The "
        "preregistered expectation is zero worlds at verdict SURVIVES "
        "over the seeded world set; any certified world is the "
        "portfolio layer's false-certification (FPR) finding, "
        "reported honestly rather than hidden.  The FPR number itself "
        "is a random quantity by nature and is never pinned -- the "
        "STRUCTURE (well-formed 3-gate reports, a resolvable verdict "
        "per world) is the contract."
    ),
    "construction": [
        _DATA_NOTE,
        "null books: identical decision bars, cohort sizes (3 long / "
        "3 short), 2xATR(14) stop geometry around the decision close, "
        "21-bar hold cap and 13-month formation warm-up as the "
        "momentum book; instruments per cohort drawn uniformly "
        "without replacement; scores U(0,1); see "
        "strategies.sector_momentum.build_null_signals",
        "engine path (the raw seam): run_portfolio_replay(signals, "
        "panel, slots=6, policy='fcfs', corr_cap=None, hold_bars=21, "
        "cost=flat 2 bps RT) then portfolio_gates(equity_curve, "
        "per-bar EV stream, dsr_min=0.95, dd_floor=-15.0, "
        "max_same_side_share=0.80, n_trials=1, min_trades=100, "
        "concurrent and net exposure attached)",
        "correlation filter disabled for this phase: the MC needs no "
        "r_proxy columns (the full machinery WITH the filter is "
        "Phase C's control)",
        f"worlds: {FULL_NULL_WORLDS} seeded nulls (seed = "
        f"{NULL_BASE_SEED} + w); --quick reduces ONLY this count "
        f"(to {QUICK_NULL_WORLDS}); world 0's full gate anatomy is "
        "printed as the representative",
    ],
    "nulls": [
        "the random books carry a tradeable edge (refuted when the "
        "gates trigger on them)",
        "the gate machinery cannot separate noise from an edge at "
        "these thresholds (measured by the certified share of worlds)",
    ],
    "kill_criteria": [
        {
            "criterion": (
                "share of null worlds certified (portfolio verdict "
                "SURVIVES) -- the false-certification rate"
            ),
            "comparison": ">",
            "threshold": 0.0,
        },
        {
            "criterion": (
                "share of null worlds UNDERPOWERED (fewer than 100 "
                "credited trades -- the gates would not resolve)"
            ),
            "comparison": ">",
            "threshold": 0.0,
        },
        {
            "criterion": (
                "share of worlds with a malformed gate report (fewer "
                "than 3 gates or an unknown verdict token)"
            ),
            "comparison": ">",
            "threshold": 0.0,
        },
    ],
    "power_calc": {
        "N_required": 100,
        "alpha": 0.05,
        "power": 0.80,
        "note": (
            "100 null worlds bound a 0% certified share with 95% "
            "confidence at roughly 3% (rule of three): the run can "
            "honestly report 'no world certified' without proving "
            "the FPR is exactly zero"
        ),
    },
    "hash": "",
}


PHASE_B_PARAMS: dict[str, Any] = {
    "name": "sector_xs_momentum_portfolio",
    "hypothesis": (
        "POSITIVE CONTROL.  Cross-sectional 12-1 momentum on the ten "
        "Select Sector SPDR ETFs -- long the top-3 / short the "
        "bottom-3 by trailing 12-1 total-return momentum, monthly "
        "cohorts, 2xATR(14) stop/target around the decision close, "
        "21-bar hold, within-cohort rank-normalized admission scores "
        "-- replayed through the FULL portfolio machinery "
        "(run_portfolio: slots=6, fcfs, correlation admission at "
        "|Spearman| <= 0.90 on growing return proxies, flat 2 bps "
        "round trip) leaves an edge the portfolio gates certify: "
        "deflated Sharpe of the per-bar EV stream >= 0.95 "
        "(n_trials=1), event drawdown above -15 R, same-side share "
        "<= 0.80, at least 100 credited trades.  Honest expectation "
        "recorded BEFORE the run: the XLRE-constrained panel covers "
        "2016-12..2024-11, a window whose idealized no-stop "
        "long-short measures ~0 edge (see the construction audit), "
        "and the engine book additionally pays the overnight gap "
        "into momentum entries (levels are placed from the decision "
        "close; winners gap away from them).  A DEAD verdict from "
        "these gates would be the machinery correctly refusing a "
        "book without a sample edge; a SURVIVES verdict would "
        "certify a real anomaly.  Both verdicts are informative; a "
        "malformed or UNDERPOWERED run is not."
    ),
    "construction": [
        _DATA_NOTE,
        "signal: monthly month-end adjusted closes; mom_12_1 = "
        "M[m-1]/M[m-13] - 1 (12-month return skipping the newest "
        "month); z-scored across the ten instruments per month-end",
        "cohorts: decision at the month-end's last trading bar (all "
        "inputs use data up to and including that close), fill at "
        "decision_idx + 1 (the engine's generator rule: next bar's "
        "open); long top-3 / short bottom-3; 6 rows per cohort; "
        "cohorts whose 21-bar hold window would pass the last bar "
        "are dropped",
        "stops: sl/tp = 2xATR(14, Wilder) around the DECISION close "
        "(levels known at decision time); the engine resolves fills "
        "and intrabar touches pessimistically (SL first)",
        "scores: within-cohort rank normalization via "
        "atropos.ranking.rank_normalize_group (the engine's "
        "within-group admission contract); priority = conviction",
        "r_proxy: per-instrument daily adjusted returns from one "
        "common anchor through the decision bar (the correlation "
        "filter's growing series)",
        "engine: run_portfolio(signals, panel, slots=6, "
        "policy='fcfs', corr_cap=0.90, hold_bars=21, cost=flat 2 bps "
        "RT, dsr_min=0.95, dd_floor=-15.0, max_same_side_share=0.80, "
        "n_trials=1, min_trades=100, ev_period_bars=1)",
        _AUDIT_NOTE,
    ],
    "nulls": [
        "the engine book's outcome is driven by the stop/gap "
        "geometry rather than the momentum signal (the idealized "
        "no-stop long-short holds ~0 edge on this window)",
        "any apparent edge is a cost artifact of the flat 2 bps "
        "round-trip model",
        "the book cannot survive admission at all (correlation "
        "rejects and capacity rejections thin the cohorts below the "
        "power floor)",
    ],
    "kill_criteria": [
        {
            "criterion": (
                "deflated Sharpe of the per-bar portfolio EV stream "
                "(n_trials=1)"
            ),
            "comparison": "<",
            "threshold": 0.95,
        },
        {
            "criterion": "event-basis max drawdown of the equity, in R",
            "comparison": "<",
            "threshold": -15.0,
        },
        {
            "criterion": "max same-side share of concurrent positions",
            "comparison": ">",
            "threshold": 0.8,
        },
        {
            "criterion": (
                "credited trades (the verdict is UNDERPOWERED below this)"
            ),
            "comparison": "<",
            "threshold": 100,
        },
    ],
    "power_calc": {
        "N_required": 100,
        "alpha": 0.05,
        "power": 0.80,
        "note": (
            "monthly-cohort book: ~97 cohorts x 6 candidates across "
            "20 years of requested history; the 100-credited-trade "
            "floor is well below the expected filled book, so the "
            "gates resolve on edge quality, not book size"
        ),
    },
    "hash": "",
}


PHASE_C_PARAMS: dict[str, Any] = {
    "name": "portfolio_noise_full_machinery",
    "hypothesis": (
        "NEGATIVE CONTROL.  One seeded random book (exactly the "
        "Phase A null construction, different fixed seed) through the "
        "SAME full machinery as the momentum control (run_portfolio "
        "with correlation admission at |Spearman| <= 0.90 on growing "
        "return proxies): the machinery must refuse it -- the "
        "portfolio verdict is NOT SURVIVES.  A SURVIVES verdict "
        "certifies pure noise and is the portfolio layer's "
        "false-certification finding, reported honestly.  The four "
        "kill criteria below are the individual certification "
        "conditions; the noise book is certified -- and this "
        "control's claim is killed -- only when ALL of them hold "
        "together."
    ),
    "construction": [
        _DATA_NOTE,
        "noise book: strategies.sector_momentum.build_null_signals "
        f"with fixed seed {PHASE_C_SEED}, r_proxy columns attached "
        "(same construction as Phase A's nulls: random 3 long / 3 "
        "short per monthly cohort, scores U(0,1), 2xATR stops, "
        "21-bar hold, 13-month warm-up -- identical bars and "
        "structure to the momentum book)",
        "engine: run_portfolio(signals, panel, slots=6, "
        "policy='fcfs', corr_cap=0.90, hold_bars=21, cost=flat 2 bps "
        "RT, dsr_min=0.95, dd_floor=-15.0, max_same_side_share=0.80, "
        "n_trials=1, min_trades=100, ev_period_bars=1) -- the FULL "
        "machinery including the correlation filter, which must "
        "reject same-instrument re-entries (|rho| = 1)",
        f"single fixed-seed world (seed {PHASE_C_SEED}): the "
        "FPR-over-worlds measurement is Phase A's job; this phase "
        "demonstrates the full pipeline's behavior on one honest "
        "noise draw",
    ],
    "nulls": [
        "the full machinery certifies pure noise at these thresholds "
        "(all four certification conditions hold -- the "
        "false-certification finding)",
        "the correlation admission fails to block same-instrument "
        "re-entries in the noise book",
    ],
    "kill_criteria": [
        {
            "criterion": (
                "noise-book deflated Sharpe at or above the 0.95 "
                "certification floor (n_trials=1)"
            ),
            "comparison": ">=",
            "threshold": 0.95,
        },
        {
            "criterion": (
                "noise-book event drawdown above the -15 R disaster "
                "floor (no blowup observed)"
            ),
            "comparison": ">",
            "threshold": -15.0,
        },
        {
            "criterion": (
                "noise-book max same-side share within the 0.80 cap"
            ),
            "comparison": "<=",
            "threshold": 0.8,
        },
        {
            "criterion": (
                "noise-book credited trades above the 100 floor (the "
                "verdict resolves instead of UNDERPOWERED)"
            ),
            "comparison": ">",
            "threshold": 100,
        },
    ],
    "power_calc": {
        "N_required": 100,
        "alpha": 0.05,
        "power": 0.80,
        "note": (
            "single-world control: the DSR gate alone mis-certifies "
            "roughly 5% of noise worlds at n_trials=1 (a z-statistic "
            "at the 1.645 bar), which is why the world-level FPR "
            "measurement lives in Phase A; this phase pins the FULL "
            "pipeline's anatomy on one seeded draw"
        ),
    },
    "hash": "",
}


def _daily_ev(result: PortfolioResult, period: int) -> np.ndarray:
    """Per-bucket EV stream (the same bucketing run_portfolio uses)."""
    n_bars = int(result["exposure"]["gross"].size)
    n_buckets = max(1, -(-n_bars // period))
    stream = np.zeros(n_buckets, dtype=np.float64)
    for row in result["accepted"]:
        bucket = int(row["credit_bar"]) // period
        stream[bucket] += float(row["credit_r"])
    return stream


def _gates_for(result: PortfolioResult) -> GateReport:
    """Run the frozen gates over a replay result (the manual seam)."""
    return portfolio_gates(
        result["equity_curve"],
        _daily_ev(result, EV_PERIOD_BARS),
        dsr_min=DSR_MIN,
        dd_floor=DD_FLOOR,
        max_same_side_share=MAX_SAME_SIDE,
        n_trials=N_TRIALS,
        min_trades=MIN_TRADES,
        concurrent=result["exposure"]["concurrent"],
        net_exposure=result["exposure"]["net"],
    )


def _check_gate_report(report: GateReport, context: str) -> None:
    """Mechanical structure check -- the only hard failure mode.

    A malformed gate report is an engineering failure and raises; a
    triggered gate or an unwelcome verdict never does (verdicts are
    recorded, not enforced).
    """
    expected = ("deflated_sharpe", "drawdown_floor", "side_concentration")
    gates = report["gates"]
    if tuple(gate["gate"] for gate in gates) != expected:
        raise RuntimeError(
            f"{context}: malformed gate report, expected gates "
            f"{expected}, got {[gate['gate'] for gate in gates]}"
        )
    verdict = report["portfolio_verdict"]
    allowed = ("SURVIVES", "UNDERPOWERED")
    if not verdict.startswith("DEAD(") and verdict not in allowed:
        raise RuntimeError(
            f"{context}: unknown portfolio verdict {verdict!r}"
        )


def _print_gates(gates: list[GateVerdict]) -> None:
    """Tombstone-style rendering of the three gate verdicts."""
    for gate in gates:
        value = float(gate["value"])
        rendered = f"{value:.4f}" if np.isfinite(value) else "nan"
        state = "TRIGGERED" if gate["triggered"] else "pass"
        print(f"    {gate['gate']:<22} {state:<10} value={rendered}")
        print(f"      {gate['reason']}")


def _print_replay_summary(result: PortfolioResult) -> None:
    """One-line replay economics (R units, event basis)."""
    summary = result["summary"]
    print(
        f"    accepted={summary['n_accepted']}  rejected="
        f"{summary['n_rejected']} (corr "
        f"{summary['n_correlation_rejects']}, capacity "
        f"{summary['n_rejected'] - summary['n_correlation_rejects']}"
        f")  replaced={summary['n_replaced']}  unresolved="
        f"{summary['n_unresolved']}"
    )
    print(
        f"    final_equity={result['final_equity_r']:+.2f}R  "
        f"max_drawdown={result['max_drawdown_r']:.2f}R  "
        f"concurrent_max={summary['concurrent_max']}"
    )


def _audit_idealized(scores: pd.DataFrame, panel: pd.DataFrame) -> str:
    """Pre-run construction audit: idealized no-stop long-short.

    Diagnostic ONLY (it consumes the month AFTER the signal, which is
    what makes it an audit rather than a signal): the equal-weight
    close-to-close long-short the engine book is supposed to express.
    """
    month_end = panel["close"].resample("ME").last()
    forward = month_end.shift(-1) / month_end - 1.0
    monthly: list[float] = []
    for position in range(min(len(scores), len(forward))):
        valid = scores.iloc[position].dropna()
        if len(valid) < 6:
            continue
        nxt = forward.iloc[position]
        if bool(nxt.isna().any()):
            continue
        ranked = valid.sort_values(ascending=False)
        value = 0.5 * float(nxt[ranked.index[:3]].mean()) - 0.5 * float(
            nxt[ranked.index[-3:]].mean()
        )
        monthly.append(value)
    series = pd.Series(monthly, dtype="float64")
    ann_mean = float(series.mean() * 12.0)
    ann_sd = float(series.std(ddof=1) * np.sqrt(12.0))
    sharpe = ann_mean / ann_sd if ann_sd > 0.0 else float("nan")
    return (
        f"idealized no-stop long-short audit: months={len(series)}  "
        f"ann_mean={ann_mean * 100:+.2f}%  ann_sd={ann_sd * 100:.2f}%  "
        f"ann_sharpe={sharpe:+.2f}"
    )


def run_phase_a(worlds: int) -> dict[str, Any]:
    """Phase A: MC over seeded null worlds through the raw seam."""
    print("=" * 78)
    print("PHASE A. NULL (random admission) -- FPR calibration")
    print("=" * 78)
    print(_prereg_text(PHASE_A_PARAMS))
    print()
    panel = load_prices()
    cost_model = build_null_signals(
        panel, NULL_BASE_SEED, include_r_proxy=False
    )["cost_model"]
    verdicts: Counter[str] = Counter()
    gate_fires: Counter[str] = Counter()
    representative: dict[str, Any] | None = None
    underpowered = 0
    for world in range(worlds):
        null = build_null_signals(
            panel, NULL_BASE_SEED + world, include_r_proxy=False
        )
        replay = run_portfolio_replay(
            null["signals"],
            panel,
            slots=SLOTS,
            policy=POLICY,
            corr_cap=None,
            hold_bars=HOLD_BARS,
            cost=cost_model,
        )
        report = _gates_for(replay)
        _check_gate_report(report, f"phase A world {world}")
        verdicts[report["portfolio_verdict"]] += 1
        if report["portfolio_verdict"] == "UNDERPOWERED":
            underpowered += 1
        for gate in report["gates"]:
            if gate["triggered"]:
                gate_fires[gate["gate"]] += 1
        if world == 0:
            representative = {
                "verdict": report["portfolio_verdict"],
                "gates": report["gates"],
                "replay": replay,
            }
    assert representative is not None
    certified = verdicts.get("SURVIVES", 0)
    fpr = certified / worlds
    print(f"worlds evaluated: {worlds}")
    print(
        "verdict counts    : "
        + ", ".join(
            f"{verdict} x {count}"
            for verdict, count in sorted(verdicts.items())
        )
    )
    print(
        f"FALSE-CERTIFICATION RATE (SURVIVES share): {fpr:.3f} "
        f"({certified}/{worlds} worlds)"
    )
    print(
        "gate trigger rates: "
        + ", ".join(
            f"{name} {count}/{worlds}"
            for name, count in sorted(gate_fires.items())
        )
    )
    if certified > 0:
        print(
            f">>> FPR FINDING: the gates certified {certified} pure-"
            "noise world(s) at the frozen thresholds -- recorded as "
            "the portfolio layer's false-certification rate, not "
            "hidden."
        )
    else:
        print(
            "no world certified: the gates refused every noise book "
            "(the FPR-control expectation holds)."
        )
    print()
    print(f"representative world (seed {NULL_BASE_SEED}) anatomy:")
    rep_replay = representative["replay"]
    _print_replay_summary(rep_replay)
    print(f"    verdict: {representative['verdict']}")
    _print_gates(representative["gates"])
    if certified == 0:
        fpr_verdict = f"HELD (0/{worlds} certified)"
    else:
        fpr_verdict = f"FPR FINDING ({certified}/{worlds} certified)"
    return {
        "verdict": fpr_verdict,
        "label": "A null (random admission, MC)",
        "fpr": fpr,
        "certified": certified,
        "worlds": worlds,
        "underpowered": underpowered,
        "gate_fires": dict(gate_fires),
    }


def run_phase_b() -> dict[str, Any]:
    """Phase B: the real momentum book through the full machinery."""
    print("=" * 78)
    print("PHASE B. MOMENTUM PORTFOLIO (positive control)")
    print("=" * 78)
    print(_prereg_text(PHASE_B_PARAMS))
    print()
    panel = load_prices()
    scores = cross_sectional_momentum_12_1(panel)
    dataset = build_portfolio_dataset(panel, scores, cost_bps=COST_BPS)
    signals = dataset["signals"]
    first = int(signals["decision_idx"].min())
    last = int(signals["decision_idx"].max())
    sides = signals["side"].value_counts()
    print(
        f"dataset: {len(signals)} entry rows, {dataset['n_cohorts']} "
        f"monthly cohorts ({sides.get('long', 0)} long / "
        f"{sides.get('short', 0)} short), decisions at bars "
        f"{first}..{last} ({panel.index[first].date()}.."
        f"{panel.index[last].date()})"
    )
    print(f"    {_audit_idealized(scores, panel)}")
    result = run_portfolio(
        signals,
        panel,
        slots=SLOTS,
        policy=POLICY,
        corr_cap=CORR_CAP,
        hold_bars=HOLD_BARS,
        cost=dataset["cost_model"],
        dsr_min=DSR_MIN,
        dd_floor=DD_FLOOR,
        max_same_side_share=MAX_SAME_SIDE,
        n_trials=N_TRIALS,
        min_trades=MIN_TRADES,
        ev_period_bars=EV_PERIOD_BARS,
    )
    _check_gate_report(
        {
            "gates": result["gates"],
            "portfolio_verdict": result["portfolio_verdict"],
            "n_trades": len(result["equity_curve"]),
            "min_trades": MIN_TRADES,
        },
        "phase B",
    )
    print()
    _print_replay_summary(result)
    print(f"portfolio verdict: {result['portfolio_verdict']}")
    _print_gates(result["gates"])
    return {
        "verdict": result["portfolio_verdict"],
        "label": "B momentum (positive control)",
        "triggered": [
            gate["gate"] for gate in result["gates"] if gate["triggered"]
        ],
    }


def run_phase_c() -> dict[str, Any]:
    """Phase C: one seeded noise world through the full machinery."""
    print("=" * 78)
    print("PHASE C. RANDOM PORTFOLIO (negative control, full machinery)")
    print("=" * 78)
    print(_prereg_text(PHASE_C_PARAMS))
    print()
    panel = load_prices()
    dataset = build_null_signals(panel, PHASE_C_SEED, include_r_proxy=True)
    signals = dataset["signals"]
    sides = signals["side"].value_counts()
    print(
        f"dataset: {len(signals)} entry rows, {dataset['n_cohorts']} "
        f"monthly cohorts ({sides.get('long', 0)} long / "
        f"{sides.get('short', 0)} short, random selection, scores "
        "U(0,1))"
    )
    result = run_portfolio(
        signals,
        panel,
        slots=SLOTS,
        policy=POLICY,
        corr_cap=CORR_CAP,
        hold_bars=HOLD_BARS,
        cost=dataset["cost_model"],
        dsr_min=DSR_MIN,
        dd_floor=DD_FLOOR,
        max_same_side_share=MAX_SAME_SIDE,
        n_trials=N_TRIALS,
        min_trades=MIN_TRADES,
        ev_period_bars=EV_PERIOD_BARS,
    )
    _check_gate_report(
        {
            "gates": result["gates"],
            "portfolio_verdict": result["portfolio_verdict"],
            "n_trades": len(result["equity_curve"]),
            "min_trades": MIN_TRADES,
        },
        "phase C",
    )
    print()
    _print_replay_summary(result)
    print(f"portfolio verdict: {result['portfolio_verdict']}")
    _print_gates(result["gates"])
    if result["portfolio_verdict"] == "SURVIVES":
        print(
            ">>> FPR FINDING: the full machinery CERTIFIED pure noise "
            "(all four certification conditions held) -- recorded, "
            "not hidden."
        )
    else:
        print(
            "the machinery refused the noise book: the negative-"
            "control expectation holds."
        )
    return {
        "verdict": result["portfolio_verdict"],
        "label": "C random (negative control)",
        "triggered": [
            gate["gate"] for gate in result["gates"] if gate["triggered"]
        ],
    }


def main() -> int:
    """Preregs -> three phases -> summary table; returns 0 always.

    An unwelcome verdict is the controls WORKING, not an error: the
    exit code only reflects mechanical failures.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CI preview: 20 Phase A null worlds instead of 100 "
        "(every other constant is identical)",
    )
    args = parser.parse_args()
    worlds = QUICK_NULL_WORLDS if args.quick else FULL_NULL_WORLDS
    mode = "quick preview" if args.quick else "full protocol"

    print("=" * 78)
    print(
        f"PORTFOLIO CONTROLS: MOMENTUM vs NOISE ON 10 SECTOR SPDRs ({mode})"
    )
    print("=" * 78)
    print()
    print("FROZEN CONFIGURATION (shared by all phases):")
    for key, value in FROZEN_CONFIG.items():
        print(f"    {key:<22} {value}")
    print()
    results = [
        run_phase_a(worlds),
        run_phase_b(),
        run_phase_c(),
    ]
    print()
    print("=" * 78)
    print("SUMMARY (verdicts are recorded, never enforced)")
    print("=" * 78)
    header = f"    {'phase':<36} {'verdict':<26} triggered"
    print(header)
    for result in results:
        triggered = result.get("triggered")
        if triggered is None:
            fires = result.get("gate_fires", {})
            triggered = sorted(fires)
        rendered = ",".join(triggered) if triggered else "(none -- certified)"
        print(f"    {result['label']:<36} {result['verdict']:<26} {rendered}")
    print()
    print("WHAT EACH PHASE CALIBRATES")
    print("-" * 78)
    print("A: the gates' false-certification rate on noise books with")
    print("   the momentum book's exact structure (FPR control; the")
    print("   number is a random quantity and is never pinned).")
    print("B: whether the machinery certifies a real cross-sectional")
    print("   book on an honest sample -- SURVIVES or DEAD are both")
    print("   informative; the STRUCTURE of the kill is the contract.")
    print("C: the full machinery (correlation admission included) on")
    print("   one seeded noise world -- must refuse what A measured.")
    print()
    print("Verdicts are evaluation output, not advice: adapt these")
    print("calls inside your own pipeline and decide what they mean")
    print("for you.  Gate semantics and defaults: atropos.portfolio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
