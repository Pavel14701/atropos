"""Real-data example: SPY 12-1 momentum through the ML kill protocol.

A runnable example of CALLING the atropos library on real market data
-- adapt these calls inside your own research pipeline.  The order is
the discipline: the frozen preregistration is printed FIRST, then the
dataset is built, then the engine runs, then the verdicts are printed
and YOU decide what to do with them (``run_ml_kill_protocol`` only
returns a report dict; nothing here "takes over").

Usage::

    uv run python examples/real_data/run_example.py           # full protocol
    uv run python examples/real_data/run_example.py --quick   # CI preview

``--quick`` runs the SAME frozen dataset, gates and walk-forward with
one model (gbm) and fewer bootstrap draws (500 vs 2000); it finishes
in a few seconds for CI.  The full run evaluates the preregistered
three-model comparison.  Reduced bootstrap draws make the incremental
t-statistic noisy, so the quick preview's per-model verdict MAY differ
from the full protocol's -- the full run is the contract.

Data: the committed offline snapshot (see strategies/spy_tsmom.py and
README.md for provenance and caveats).  Expected verdicts:
README.md / expected_output.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from atropos.ml_training import run_ml_kill_protocol  # noqa: E402
from preregistration import build_prereg_text  # noqa: E402
from strategies.spy_tsmom import build_dataset  # noqa: E402

FROZEN_CONFIG: dict[str, object] = {
    "horizon_h": 744,  # 31 calendar days >= the 21-trading-day label
    "purge_h": 24,
    "embargo_h": 744,  # one month of embargo before each test block
    "k_blocks": 6,
    "n_min_events": 50,  # monthly book; see prereg power_calc
    "profit_bar": 0.0,  # taken-trade EV must clear zero NET of costs
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


def print_dataset_summary(dataset: dict) -> None:
    """Print the frozen dataset's shape and marginal statistics."""
    events = dataset["events"]
    exec_returns = dataset["exec_returns"]
    print("=" * 78)
    print("DATASET (frozen construction, committed snapshot)")
    print("=" * 78)
    print(
        f"bars        : {len(dataset['prices'])} daily rows "
        f"({dataset['prices'].index[0].date()}.."
        f"{dataset['prices'].index[-1].date()})"
    )
    print(
        f"events      : {len(events)} long monthly entries "
        f"({events['entry_date'].min().date()}.."
        f"{events['entry_date'].max().date()})"
    )
    print(f"features    : {', '.join(dataset['feature_frame'].columns)}")
    print(
        f"base rate   : P(exec > 0) = {(dataset['target'] == 1).mean():.3f}"
    )
    print(
        f"exec return : mean {exec_returns.mean() * 1e4:.1f}bp  "
        f"sd {exec_returns.std() * 1e4:.1f}bp  "
        f"(21d hold, net of 2bps RT)"
    )


def print_report(report: dict) -> None:
    """Tombstone-style rendering of the report dict the library returned."""
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
    print(f"SPY 12-1 TSMOM THROUGH THE ML KILL PROTOCOL ({mode})")
    print("=" * 78)
    print()
    print("STEP 1. PREREGISTRATION (printed BEFORE any result; you call")
    print("the library, the library validates the hypothesis YOU froze)")
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
        config=config,  # type: ignore[arg-type]
        battery_controls=["mom_12_1"],
    )
    # wall-clock time is deliberately NOT printed: stdout must stay
    # byte-stable so expected_output.md and the tests stay honest
    print(
        f"engine returned (bootstrap_b={report['config']['bootstrap_b']}, "
        f"models={report['config']['models']})"
    )
    print()
    print_report(report)
    print()
    print("Verdicts are evaluation output, not advice: adapt these calls")
    print("inside your own pipeline and decide what a kill means for you.")
    print("Gate semantics and defaults: DEFAULT_TRAINING_CONFIG and")
    print("docs/how-to/05-run-ml-kill-protocol.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
