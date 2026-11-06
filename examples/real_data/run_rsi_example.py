"""Real-data example: SPY RSI(2) mean-reversion through the ML kill protocol.

The NEGATIVE CONTROL of the two real-data examples.  A runnable
example of CALLING the atropos framework on real market data -- adapt
these calls inside your own research pipeline.  The order is the
discipline: the frozen preregistration is printed FIRST, then the
dataset is built, then the engine runs, then the verdicts are printed
and YOU decide what to do with them (``run_ml_kill_protocol`` only
returns a report dict; nothing here "takes over").

RSI(2) mean-reversion on SPY is the canonical strategy that looks
tradable gross but carries a few-bp net edge against a ~170 bp
per-trade sd on a two-day turnover.  The preregistered expectation
below is ``DEAD`` -- the engine should kill it -- which is what makes
this the negative control twin of ``run_example.py`` (TSMOM).

Usage::

    uv run python examples/real_data/run_rsi_example.py           # full
    uv run python examples/real_data/run_rsi_example.py --quick   # CI

``--quick`` runs the SAME frozen dataset, gates and walk-forward with
one model (gbm) and fewer bootstrap draws (500 vs 2000); it finishes
in seconds for CI.  The full run evaluates the preregistered
three-model comparison.  Reduced bootstrap draws make t-statistics
noisy, so the quick preview MAY differ from the full protocol -- the
full run is the contract.

Data: the committed offline snapshot (see strategies/spy_rsi.py and
README.md for provenance and caveats).  Expected verdicts:
README.md / expected_rsi_output.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from atropos import generate_prereg  # noqa: E402
from atropos.ml_training import run_ml_kill_protocol  # noqa: E402
from atropos.prereg_generator import validate_prereg  # noqa: E402
from strategies.spy_rsi import COST_RT, build_dataset  # noqa: E402

FREEZE_DATE = "2026-11-06"
"""Date the RSI prereg was frozen; explicit so reruns render identical
text instead of silently moving the stamp to today."""

PREREG_PARAMS: dict[str, Any] = {
    "name": "spy_rsi2_meanrev_ml",
    "hypothesis": (
        "NEGATIVE CONTROL. Two-period RSI mean-reversion on SPY -- "
        "short the next bar's open after RSI(2) > 70, long after "
        "RSI(2) < 30, first exit of RSI(2) crossing 50, a 2xATR(14) "
        "stop or target, or 5 bars held -- leaves an executable edge "
        "net of 2 bps round-trip costs: classifiers on the RSI state "
        "and volatility conditioning take a book that clears the cost "
        "line out-of-sample, ranks features consistently across folds "
        "and holds a deflated Sharpe above 0.95.  The published "
        "reversal edge is thin (a few bp per trade on a two-day "
        "hold), so the PREREGISTERED EXPECTATION is the opposite of "
        "the hypothesis: this run dies, most plausibly at the cost "
        "line -- the framework must kill what cannot survive its own "
        "turnover net of costs."
    ),
    "construction": [
        "data: committed SPY daily snapshot data/SPY.csv, adjusted "
        "close, 2005-01-03..2024-12-31 (one adjustment vintage, "
        "sha256-pinned; intraday Open/High/Low mapped into adjusted "
        "space by the per-bar AdjClose/Close factor)",
        "indicators: RSI(2) and RSI(14) with Wilder smoothing, "
        "ATR(14) Wilder-smoothed true range, vol_20d trailing "
        "20-day log-return sd",
        "entries: close t with RSI(2) > 70 shorts the open of bar "
        "t+1; RSI(2) < 30 longs it (signal close strictly precedes "
        "the execution price)",
        "exits: first of intrabar 2xATR(14) stop/target touch "
        "(stops take priority; a stop gapped through at the open "
        "fills at the open), RSI(2) crossing 50 at a close, or the "
        "5-bar cap at that close; the book is non-overlapping",
        "features at the signal close: rsi_2, rsi_14, vol_20d; "
        "battery_controls=[] -- the raw signal is itself a model "
        "feature, so the incremental gate reports inconclusive and "
        "is NOT a kill criterion for this run",
        "exec = signed return from entry open to rule exit, net of "
        "2 bps RT cost; target = (exec > 0); long and short events "
        "share one frame signed by side",
        "engine: run_ml_kill_protocol, models gbm/rf/logit, "
        "battery_controls=[], k_blocks=6 expanding walk-forward, "
        "horizon 168h, purge 24h, embargo 168h, threshold grid "
        "0.30..0.70 step 0.05, bootstrap_b=2000",
        "gates: profit_bar=0.0 (EV of taken trades must be positive "
        "net of costs), importance_rho_min=0.5, dsr_min=0.95, "
        "n_min_events=100",
    ],
    "nulls": [
        "the gross reversal edge is a cost artifact: taken-trade EV "
        "does not clear zero net of 2 bps round-trip costs",
        "per-fold feature rankings are noise: importance stability at "
        "or below chance",
        "the trade stream is indistinguishable from noise at daily "
        "frequency: deflated Sharpe of the daily EV stream below 0.95",
    ],
    "kill_criteria": [
        "OOS EV of taken trades < 0.0000 net of 2 bps RT costs",
        "median pairwise Spearman of per-fold importances < 0.5",
        "deflated Sharpe of the daily EV stream < 0.95",
    ],
    "power_calc": {
        "N_required": 100,
        "alpha": 0.05,
        "power": 0.80,
        "note": (
            "daily-turnover book: 100 declustered taken events are "
            "the floor at which the gates resolve; at the observed "
            "~1.7% per-trade sd they detect per-trade edges of "
            "roughly 2 x 1.7% / sqrt(100) ~ 34 bp -- the expected "
            "few-bp RSI edge sits far below that bar, which is WHY "
            "the honest outcome here is a kill, not a confirmation"
        ),
    },
    "hash": "",
}


def spec_hash() -> str:
    """First 8 hex chars of sha256 over the frozen params (sans hash)."""
    frozen = {k: v for k, v in PREREG_PARAMS.items() if k != "hash"}
    canonical = json.dumps(frozen, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def build_prereg_text() -> str:
    """Validate the frozen params and render the prereg document.

    Returns:
        The markdown preregistration as produced by
        ``atropos.generate_prereg``, stamped with :data:`FREEZE_DATE`.

    Raises:
        ValueError: if the frozen params ever fail validation (the
            failure must be loud, never a softened spec).
    """
    params = {**PREREG_PARAMS, "hash": spec_hash()}
    problems = validate_prereg(params)
    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(f"frozen prereg params are invalid:\n  - {detail}")
    return generate_prereg(params, date=FREEZE_DATE)


FROZEN_CONFIG: dict[str, object] = {
    "horizon_h": 168,  # 7 calendar days >= the 5-trading-bar max hold
    "purge_h": 24,
    "embargo_h": 168,  # one full hold period before each test block
    "k_blocks": 6,
    "n_min_events": 100,  # daily-turnover book; see prereg power_calc
    "profit_bar": 0.0,  # taken-trade EV must clear zero NET of costs
    "prereg_hash": spec_hash(),  # anchor the verdict to the frozen spec
}
QUICK_CONFIG: dict[str, object] = {
    **FROZEN_CONFIG,
    "models": ["gbm"],
    "bootstrap_b": 500,
}
FULL_CONFIG: dict[str, object] = {
    **FROZEN_CONFIG,
    "models": ["gbm", "rf", "logit"],
}


def fmt_bp(value: float | None) -> str:
    """Format a per-trade EV in basis points (n/a when undefined)."""
    if value is None:
        return "n/a"
    return f"{value * 1e4:.1f}bp"


def print_dataset_summary(dataset: dict[str, Any]) -> None:
    """Print the frozen dataset's shape and marginal statistics."""
    events = dataset["events"]
    exec_returns = dataset["exec_returns"]
    sides = events["side"].value_counts()
    print("=" * 78)
    print("DATASET (frozen construction, committed snapshot)")
    print("=" * 78)
    print(
        f"bars        : {len(dataset['prices'])} daily rows "
        f"({dataset['prices'].index[0].date()}.."
        f"{dataset['prices'].index[-1].date()})"
    )
    print(
        f"events      : {len(events)} mean-reversion entries "
        f"({events['entry_date'].min().date()}.."
        f"{events['entry_date'].max().date()}; "
        f"{sides.get('long', 0)} long / {sides.get('short', 0)} short)"
    )
    print(f"features    : {', '.join(dataset['feature_frame'].columns)}")
    print(
        f"base rate   : P(exec > 0) = {(dataset['target'] == 1).mean():.3f}"
    )
    mean_bp = exec_returns.mean() * 1e4
    print(
        f"trade return: gross {mean_bp + COST_RT * 1e4:.1f}bp"
        f"  net {mean_bp:.1f}bp  sd {exec_returns.std() * 1e4:.1f}bp"
    )
    print(
        f"hold        : mean {events['hold_bars'].mean():.2f} bars "
        f"(cap {events['hold_bars'].max()}), net of 2bps RT"
    )


def print_report(report: dict[str, Any]) -> None:
    """Tombstone-style rendering of the report dict the framework returned."""
    print("=" * 78)
    print("ML KILL PROTOCOL -- WALK-FORWARD VERDICTS")
    print("=" * 78)
    print(f"run verdict : {report['verdict']}")
    print(
        f"book        : n_rows={report['n_rows']}  "
        f"n_events={report['n_events']}  "
        f"(floor {report['n_min_events']}, powered={report['powered']})"
    )
    print(
        f"linear book : battery EV={fmt_bp(report['ev_battery'])}  "
        f"baseline EV={fmt_bp(report['ev_baseline'])}"
    )
    for name, entry in report["models"].items():
        print(
            f"  model {name:<6} verdict={entry['verdict']:<28} "
            f"EV={fmt_bp(entry['ev'])} "
            f"DSR={entry['deflated_sharpe']} "
            f"rho={entry['importance_stability']} "
            f"events={entry['n_events']}"
        )
        for gate in entry["kills"]:
            state = (
                "off"
                if not gate["enabled"]
                else (
                    "INCONCLUSIVE"
                    if gate["inconclusive"]
                    else ("TRIGGERED" if gate["triggered"] else "pass")
                )
            )
            print(f"    {gate['check']:<20} {state}")
            if gate["reason"]:
                print(f"      {gate['reason']}")
    incremental = report["incremental"]
    for name, detail in incremental.items():
        print(
            f"  incremental {name}: t_vs_battery="
            f"{_fmt_stat(detail['t_vs_battery'])}  "
            f"t_vs_baseline={_fmt_stat(detail['t_vs_baseline'])}"
        )


def _fmt_stat(value: float | None) -> str:
    """Format a bootstrap t-statistic (n/a when undefined)."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main() -> int:
    """Prereg -> dataset -> engine run -> verdicts; returns 0 always.

    A kill verdict is the protocol WORKING, not an error: the exit
    code only reflects mechanical failures.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CI preview: gbm only, bootstrap_b=500 (default: full "
        "preregistered protocol, bootstrap_b=2000)",
    )
    args = parser.parse_args()

    config = QUICK_CONFIG if args.quick else FULL_CONFIG
    mode = "quick preview" if args.quick else "full protocol"

    print("=" * 78)
    print(f"SPY RSI(2) MEAN-REVERSION THROUGH THE ML KILL PROTOCOL ({mode})")
    print("=" * 78)
    print()
    print("STEP 1. PREREGISTRATION (printed BEFORE any result; you call")
    print("the framework, the framework validates the hypothesis YOU froze)")
    print("-" * 78)
    print(build_prereg_text())
    print()
    print("STEP 2. DATASET from the committed snapshot")
    print("-" * 78)
    dataset = build_dataset()
    print_dataset_summary(dataset)
    print()
    print("STEP 3. ENGINE RUN (run_ml_kill_protocol returns one dict)")
    print("-" * 78)
    report = run_ml_kill_protocol(
        dataset["feature_frame"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        config=config,
        battery_controls=[],
    )
    # wall-clock time is deliberately NOT printed: stdout must stay
    # byte-stable so expected_rsi_output.md and the tests stay honest
    print(
        f"engine returned (bootstrap_b={report['config']['bootstrap_b']}, "
        f"models={report['config']['models']}, "
        f"prereg_hash={report['config']['prereg_hash']})"
    )
    print()
    print_report(report)
    print()
    print("WHY THIS IS THE NEGATIVE CONTROL")
    print("-" * 78)
    print("RSI(2) mean-reversion looks tradable gross: a real, published")
    print("reversal edge whose taken book even clears a 2 bps cost line.")
    print("But the edge is too thin to survive walk-forward scrutiny:")
    print("fold-to-fold feature rankings do not repeat and the daily EV")
    print("stream misses the deflated-Sharpe multiple-testing bar.")
    print("Expected: DEAD -- and a kill here is the framework WORKING:")
    print("it correctly killed a strategy that looks good gross but is")
    print("too weak to survive honest evaluation.  The companion example")
    print("run_example.py (SPY 12-1 TSMOM) shows the same engine on a")
    print("book whose cost-line edge is an order of magnitude stronger.")
    print()
    print("Verdicts are evaluation output, not advice: adapt these calls")
    print("inside your own pipeline and decide what a kill means for you.")
    print("Gate semantics and defaults: DEFAULT_TRAINING_CONFIG and")
    print("docs/how-to/05-run-ml-kill-protocol.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
