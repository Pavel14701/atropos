---
name: 20-paper-parity
description: Judge a deployed paper leg with atropos.paper_protocol.run_paper_protocol — parity against the frozen spec, verdict tokens, and the success criteria.
when_to_use: When a paper run has closed trades and the question is "did the engine execute the frozen spec?" — forward parity, not edge.
version: 0.1.0
---

# SKILL 20: PAPER PARITY — forward test of the implementation
> Code canonical. Single home: `atropos.paper_protocol.run_paper_protocol`.
> Law: KNOWN-ISSUES §15. Companion: skill 19 (declare the family first).

> Paper is a forward test of the IMPLEMENTATION, not the edge.
> The frozen sim already validated the strategy; paper answers a
> different question: "does the engine execute the same strategy
> the frozen spec describes?" The 2026-11 rev-1 run lost -8.12R
> because the answer was NO (missing MTM-48 exit, double fee) and
> nobody had a formal verdict for 7 days. This skill makes the
> verdict same-day.

## 1. Build the three inputs

- `paper_log`: the closed trades (leg, coin, entry_ts, exit_ts,
  exit_type, pnl_r; + sizing_rule column; attrs bar_delta_min;
  optional execution columns — fee_r, size, side, entry/exit_price,
  signal_ts, funding_rate).
- `frozen_outcomes`: {leg: DataFrame} — the frozen sim replayed on
  the SAME bars the paper engine saw (feed granularity must match —
  §12).
- `frozen_spec[leg]`: exit_rules, exit_type_normalization (REQUIRED),
  fee, universe, sizing_rule, bar_delta_min, paper_window, and the
  family declaration (`family` + `tools_used` from skill 19).

`tools_used` is a DECLARATION, not a measurement: enforcement
validates the declaration against the family, not against actual
execution (KNOWN-ISSUES §17).

## 2. Run it, read the verdict

`run_paper_protocol(...)` → one of six verdicts:
PARITY_PASS / PARITY_FAIL / OPERATIONAL_FAIL / STATISTICAL_FAIL /
INSUFFICIENT_N / SURVIVES_TO_LIVE. Issue types: 10 parity
(exit_mismatch, fee_mismatch, sizing_mismatch, sizing_rule_mismatch,
universe_violation, unmatched_entry, frozen_window_gap,
feed_granularity_mismatch, funding_rate_mismatch, pnl_component_mismatch)
+ 1 statistical (dd_divergence) + operational issues
(latency_breach, exit_direction_violation, instrument_violation).

## 3. The success criteria (frozen — do not reinterpret)

- Fix verification at n=5-10 closed trades: a **CLEAN parity report**
  (issues empty, matched == paper trades) with verdict
  `INSUFFICIENT_N` is the EXPECTED success signal.
- `PARITY_PASS` requires n_matched ≥ 20.
- `SURVIVES_TO_LIVE` additionally requires execution evidence
  (side + entry/exit prices + signal_ts actually verified).
- ANY parity issue on NEW trades = kill + post-mortem before
  restart. The rev-1 lesson: a parity failure is a kill, not a
  "bad week".

## NEVER

- Reinterpret INSUFFICIENT_N at n=5-10 as failure — it is the
  designed outcome of a clean small sample.
- Skip the frozen_window check because "the sim obviously covers
  it" — stale frozen sets are the trap the check exists for.
- Treat `unmatched_entry` as a defect without reading the feed
  construction (rolling-head live vs fixed-head replay is a
  population difference).
- Backout historical fees to make an old parity report clean —
  historical mismatches are evidence, not noise.
