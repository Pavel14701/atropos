"""The full backtest pipeline: simulate -> rank -> normalize -> admit.

Thin executable wrapper over :mod:`atropos.demo_pipeline` - the demo
logic moved INTO the package so the same pipeline ships one CLI
command:

    atropos demo --mode quick   # reduced model budget, seconds
    atropos demo --mode full    # this transcript (same run)

This file remains for the how-to walk-through and for a no-CLI check
of the full-budget transcript (deterministic, seed=41):

    uv run python docs/examples/backtest_pipeline.py

The five stages wire the three backtest layers together the way a
research pipeline would (isolated replay -> slot machine -> past-only
gbm ranking -> RANKING_METRICS gbm-vs-random -> maker-vs-market).
Data: SYNTHETIC (seed=41), a trend + panic OHLC world - same honest
marking as the VRP example: this demonstrates the PIPELINE, not a
market claim.
"""

from __future__ import annotations

from atropos.demo_pipeline import run_demo_pipeline


def main() -> int:
    """Run the packaged demo pipeline (full budget, seed=41)."""
    summary = run_demo_pipeline(seed=41, quick=False)
    return 0 if summary["pipeline_completed"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
