# Portfolio kill-protocol calibration: public data first, private blind

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
| 2. Null FPR | synthetic no-edge books, seeded, many draws | measured false-positive rate within the preregistered budget; noise books end `DEAD(<gate>)` | PASS — `HELD (0/100 certified)`: FPR 0.000, every noise book died (gate fires: deflated_sharpe 100/100, drawdown_floor 95/100, side_concentration 100/100) |
| 3. SPY positive | SPY daily history, long book | `SURVIVES` — the calibrated gates refuse to kill a live edge | FAIL — `DEAD(importance_stability)`: the edge passed incremental (t=7.8), cost_line and deflated_sharpe (DSR 0.981), but per-fold feature rankings flipped (median pairwise Spearman -0.50 < 0.50) |
| 4. RSI negative | RSI-based decoy on public daily data | `DEAD(<gate>)` — the known-worthless decoy dies | PASS — `DEAD(importance_stability)`; deflated_sharpe fired too (DSR 0.925 < 0.95); incremental INCONCLUSIVE (no battery controls supplied) |
| 5. 10-name momentum portfolio | 10 liquid names, momentum book | `SURVIVES` — the documented effect passes all three gates | FAIL — `DEAD(deflated_sharpe)`: all three gates fired on a book whose preregistered audit measured ~0 in-window edge (-0.74%/yr, Sharpe -0.11); the live-edge direction is untested on this panel |
| 6. Random portfolio negative | random instruments, random sides | `DEAD(<gate>)` — same behavior as the random-walk world of how-to 09 | PASS — `DEAD(side_concentration)`: the only gate that fired; DSR and drawdown passed on this draw, so the structural gate carried the kill |

The Expected column was frozen before the runs and is not
retrofitted; the exact book construction is the calibration runner's
(`run_portfolio_controls`) job. Status vocabulary: each cell records
the phase's actual verdict verbatim from its runner output, plus the
PASS / FAIL judgment against the pinned direction. A `FAIL` anywhere
stops the progression — the gates are not calibrated until all five
phases are `PASS`, and no blind private run may start before that.

### Recorded outcomes (full run, 2026-11-06)

Verbatim transcripts live in `examples/real_data/`
(`expected_portfolio_controls.md` for phases 2, 5 and 6;
`expected_output.md` and `expected_rsi_output.md` for phases 3 and
4). `tests/test_real_data_example.py` pins the verdicts, the
prereg-before-results order and the snapshot checksums — float
digits may shift across BLAS builds, verdicts may not. Each portfolio
phase printed its hash-stamped preregistration BEFORE its run (null
admission `5bf0ac47`, momentum `3372d2e2`, noise `a53c2b6f`).

What the recorded run established:

- **Noise dies (phases 2 and 6).** The null Monte Carlo certified 0
  of 100 seeded noise books — FPR 0.000, inside the budget (the rule
  of three bounds the true certified share at roughly 3% with 95%
  confidence; `check_calibration_progression`'s tolerance is
  2 x alpha). Gate anatomy on noise: `deflated_sharpe` fired in
  100/100 worlds and `side_concentration` in 100/100, but
  `drawdown_floor` in only 95/100 — five percent of noise worlds
  random-walked without ever falling 15 R below a peak and were
  still caught by the other two gates.
- **The decoy dies (phase 4).** RSI(2) mean-reversion looks tradable
  gross (its taken book even clears the 2 bps cost line) and died
  `DEAD(importance_stability)` with `deflated_sharpe` also triggered
  — a kill here is the framework working.
- **The positive direction is NOT yet demonstrated (phases 3 and 5,
  both FAIL).** Phase 3: the SPY 12-1 book cleared three of the four
  ML-kill-protocol gates — incremental (t = 7.8 over the linear
  battery), cost_line (positive OOS EV net of costs) and
  deflated_sharpe (0.981) — and still died on `importance_stability`:
  per-fold feature rankings flipped sign across folds. Phase 5: the
  honest outcome was preregistered before the run — the XLRE
  inception (2015-10-08) cuts the panel's shared history to
  2016-12..2024-11, whose idealized no-stop momentum long-short
  measures ~0 edge (-0.74%/yr, Sharpe -0.11), so the gates refused a
  book with no measured sample edge. That calibrates the refusal
  direction, not the pinned "live edge passes" claim.
- **Structural-gate observation.** `side_concentration` fired in
  every phase and world at these thresholds (value 1.0000): with 6
  slots and partial-cohort admission (positions riding their 21-bar
  cap while the next cohort's signals are decided), some bar always
  ends single-side. On this book shape it is a structural flag, not
  an edge discriminator — a threshold-calibration observation,
  recorded here rather than tuned away.

The progression gate agrees with the two FAILs:

```bash
uv run python -c "from atropos.portfolio_protocol import check_calibration_progression as c; print(c(0.0, 'DEAD(deflated_sharpe)', 'DEAD(side_concentration)')['next_step'])"
# -> recalibrate: positive control
```

Recalibration means a NEW preregistration with a new hash — a fresh
panel or window where the documented effect exists in-sample, or a
separately logged amendment of the gate set — never a silent retune
of frozen thresholds after seeing these verdicts. Until a positive
control records `SURVIVES`, the calibration is incomplete and the
blind run below stays closed.

## The prereg hash

The gates are only meaningful if a reader can verify they were not
retuned after seeing the data. The mechanism:

1. Freeze the protocol as the prereg dict (the `PortfolioKillPrereg`
   contract: hypothesis, slots, policy, `corr_cap`, `hold_bars`, the
   gate floors, `min_trades`, seeds, data window, snapshot
   checksums).
2. Validate it with `validate_kill_prereg` (every field checked; an
   empty problem list = valid), then hash it with `hash_kill_prereg`
   — sha256 over the canonical JSON (sans hash).
3. Persist the snapshot with `save_prereg_snapshot` to `prereg.json`,
   next to this document — the file carries the hash and every frozen
   field (one-shot: the function refuses to overwrite a frozen
   prereg).
4. **Commit before any private-system look.** `prereg.json` plus this
   document go into git, and the commit is pushed, BEFORE private
   data is touched. The git history IS the timestamp.

The one-sentence rule: **commit before any private-system look.** A
hash in a local commit is weak; a hash in a pushed public commit is
pre-registration by construction. `check_calibration_progression`
reports which phases have run — it cannot rewrite them.

Standing: the umbrella `prereg.json` has not been written yet. The
three phase preregs of the recorded run are the hash-stamped public
record of the calibration; the umbrella snapshot must be committed
before any private contact.

## Blind-run protocol (phase 8)

Only after every phase above is `PASS` and the hash is committed may
a private system run blind. The run itself is an ordinary
`atropos.portfolio` run; what differs is publication:

**Standing: closed.** The recorded progression holds FAIL on both
positive controls (above); no private system may be judged under
this protocol until recalibration records a live edge at `SURVIVES`.

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
   snapshot with `hash_kill_prereg`, and compare against the
   committed hash.
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
- Prereg contract, hashing and the progression gate —
  `atropos.portfolio_protocol` (`validate_kill_prereg`,
  `hash_kill_prereg`, `save_prereg_snapshot`,
  `check_calibration_progression`, `anchor_prereg`).
- The reference pipeline the portfolio layer sits behind —
  `docs/examples/backtest_pipeline.py`.
- What Atropos does and does not do — `POSITIONING.md`.
