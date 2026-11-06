# Portfolio kill-protocol calibration: public data first, private blind

> **SKELETON — phases pending calibration runs.** The gate-progression
> table below is the contract; the Status column is the placeholder.
> Fill each Status from the `run_portfolio_controls` output as the
> phase lands — nothing here claims a result yet.

**Purpose.** Protocol calibration for the portfolio kill gates —
public data only, verifiable by anyone. The gates that will one day
judge a private system (`SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED`,
how-to 09 step 4) are first judged on worlds whose right answer is
known in advance: noise must die, a known live edge must survive.
Every phase runs on public data (index/ETF history and synthetic
nulls), so anyone can re-run the progression and check it.

This is pre-registration by construction: the gates, their floors and
the seeds are committed to git — under a verifiable hash — BEFORE the
first look at private data. By the time a private system runs blind,
the criteria it will be judged by are already frozen in public.

## Why the gates need calibration

The portfolio verdict rests on three gates, in fixed order: deflated
Sharpe on the per-period EV stream (`dsr_min`), the drawdown floor on
the event equity curve (`dd_floor`), and the same-side concentration
cap (`max_same_side_share`). A gate that fires on everything is a
rubber stamp; a gate that never fires is decoration. Both failure
modes are public-data questions, answered before any private contact:

- **Type I** — how often does a NO-EDGE book pass? (null FPR)
- **Type II** — does a known live edge pass? (SPY, momentum)

## Gate progression

| Phase | Data | Expected | Status |
|---|---|---|---|
| 2. Null FPR | synthetic no-edge books, seeded, many draws | measured false-positive rate within the preregistered budget; noise books end `DEAD(<gate>)` | pending |
| 3. SPY positive | SPY daily history, long book | `SURVIVES` — the calibrated gates refuse to kill a live edge | pending |
| 4. RSI negative | RSI-based decoy on public daily data | `DEAD(<gate>)` — the known-worthless decoy dies | pending |
| 5. 10-name momentum portfolio | 10 liquid names, momentum book | `SURVIVES` — the documented effect passes all three gates | pending |
| 6. Random portfolio negative | random instruments, random sides | `DEAD(<gate>)` — same behavior as the random-walk world of how-to 09 | pending |

The table pins only the pass/fail direction of each phase; the exact
book construction is the calibration runner's (`run_portfolio_controls`)
job. Status vocabulary: `pending` -> `PASS` / `FAIL`, read straight
from the runner's output for that phase. A `FAIL` anywhere stops the
progression — the gates are not calibrated until all five phases are
`PASS`, and no blind private run may start before that.

The plan's step 1 (the contract: `PortfolioKillPrereg`, `hash_prereg`,
`save_prereg_snapshot`, `check_calibration_progression` in
`atropos.portfolio_protocol`) and step 7 (commit the hash) bracket
this table — freeze, run, commit. Step 8 is the blind run, below.

## The prereg hash

The gates are only meaningful if a reader can verify they were not
retuned after seeing the data. The mechanism:

1. Freeze the protocol as a `PortfolioKillPrereg`: slots, policy,
   `corr_cap`, `hold_bars`, the gate floors, `min_trades`, seeds.
2. Hash it with `hash_prereg` — a deterministic digest over the
   canonical serialization (`atropos.portfolio_protocol`).
3. Persist the snapshot with `save_prereg_snapshot` to `prereg.json`,
   next to this document — the file carries the hash and every frozen
   field.
4. **Commit before any private-system look.** `prereg.json` plus this
   document go into git, and the commit is pushed, BEFORE private
   data is touched. The git history IS the timestamp.

The one-sentence rule: **commit before any private-system look.** A
hash in a local commit is weak; a hash in a pushed public commit is
pre-registration by construction. `check_calibration_progression`
reports which phases have run — it cannot rewrite them.

## Blind-run protocol (phase 8)

Only after every phase above is `PASS` and the hash is committed may
a private system run blind. The run itself is an ordinary
`atropos.portfolio` run; what differs is publication:

- **Published** — the aggregate verdict only: the kill verdict, the
  portfolio verdict (`SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED`),
  and the `prereg.json` hash the run was judged under.
  `docs/own-system.md.template` is the fill-in for exactly that
  aggregate.
- **Not published** — everything else: signals, instruments,
  timestamps, per-trade ledgers, gate values, exposure series. The
  details stay private; the template's delta-vs-donor section is the
  private artifact.

Reading a blind aggregate honestly: a `DEAD(<gate>)` on a system that
never touched the gates is the calibrated protocol doing its job; a
`SURVIVES` means "not killed by gates calibrated in public" — never
"trading-ready" (the `POSITIONING.md` line holds for private verdicts
too). And `UNDERPOWERED` is undecided, not a pass — record it as such
(how-to 09, traps).

## Verifying someone else's calibration

Public calibration means the reader does not have to trust the
author. Without any private data, a verifier can:

1. Check out the commit that added `prereg.json`, re-hash the
   snapshot with `hash_prereg`, and compare against the committed
   hash.
2. Re-run `run_portfolio_controls` per phase on the same public data
   and seeds, and compare each result against the Status column
   above.
3. Confirm that any shared blind aggregate postdates the prereg
   commit (`git log --format=%aI -1 -- prereg.json`).

If any step disagrees, the calibration claim is void — publish the
disagreement rather than patching the table silently.

## Links

- Reading portfolio results (gates, verdicts, the correlation cap) —
  `docs/how-to/09-read-portfolio-results.md`.
- Portfolio replay and gates — `atropos.portfolio` (`run_portfolio`,
  `run_portfolio_replay`, `portfolio_gates`); runnable demo:
  `uv run python -m atropos.portfolio`.
- Prereg contract and hashing — `atropos.portfolio_protocol`
  (`PortfolioKillPrereg`, `hash_prereg`, `save_prereg_snapshot`,
  `check_calibration_progression`), landing alongside this skeleton.
- The reference pipeline the portfolio layer sits behind —
  `docs/examples/backtest_pipeline.py`.
- What Atropos does and does not do — `POSITIONING.md`.
