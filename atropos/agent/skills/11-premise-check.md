---
name: 11-premise-check
description: Premise check before any strategy: graveyard, trigger, sign, data, instrument, cost line (stage 1/7).
when_to_use: Before any strategy line for a new crypto hypothesis; when porting an academic effect from equities.
version: 1.0.0
---

# SKILL 1: PREMISE CHECK — the premise check protocol (Stage 1 of 7)
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Self-contained protocol: the agent needs only this file + data.
> Death taxonomy: FRAMEWORK.md (repo root) §4; tombstones live in
> your workspace layout (`hypotheses/graveyard/`, registry —
> `hypotheses/INDEX.csv`): scaffold with `atropos init` (files are
> generated, never overwritten).

## PURPOSE

Answer: does the trigger premise exist at all in crypto — BEFORE any
strategy line. The cheapest stage of the pipeline: a hypothesis that
dies here cost 1-3 queries, not a prereg slot and a week. A cheap
early death = the pipeline working, not a loss.

## INPUTS

1. A hypothesis statement with an explicit trigger: "when X, then Y
   over horizon L".
2. The `data/` inventory of the repo: which series are free, history
   depth.
3. Instruments: Deribit = BTC+ETH options only; alt options delisted
   (SOL — 2022-12); index crypto options do not exist.
4. Graveyard: `docs/THEORY_GRAVEYARD.md` — check BEFORE anything
   else.

## PROCEDURE (order; each step = a measurable log output)

1. **GRAVEYARD CHECK FIRST.** Find the closest family in
   THEORY_GRAVEYARD.md. Differing only by a parameter/wrapper =
   reopening: run ONLY with a new dated prereg that discloses the
   closed history (3 of the reviewer's 49 directions had already been
   closed in-house).
2. **EXISTS (does the condition occur?).** One bounded query: trigger
   frequency, factor spread p10/p50/p90, share of days with the
   condition. The trigger never fires / fires less often than the
   unit of measurement → PREMISE-DEATH.
3. **SIGN CHECK.** For ports from equities — a raw sign test (zero
   parameters, one query). The risk-premium sign in crypto is
   negative: an inversion (INVERTED) is recorded permanently, the
   inverted version is never preregged.
4. **DATA AVAILABILITY.** Is there free multi-year history? A paid
   source = DATA-BLOCKED, with a list of what would open the
   question. A short window (< 1 year) — "SHORT WINDOW DISCLOSED
   LOUDLY" in the brief.
5. **INSTRUMENT (does it exist?).** Are the tradable instrument and
   the execution domain real? Delisting / HFT colocation / MEV blocks
   = INSTRUMENT-BLOCKED — "does not exist as a question".
6. **COST-LINE PRE-PROBE (amplitude vs costs).** House line ~10 bp
   taker RT / 20 bp conservative RT. Gross amplitude < RT/2 = the
   family is dead before strategy; a turnover forecast > 20 RT/year =
   construction closed.

## KILL CRITERIA (any single one closes the stage)

- **GRAVEYARD-HIT** — the family is closed, no new argument.
- **PREMISE-DEATH** — the trigger never occurs / occurs less often
  than the unit of measurement / is inverted.
- **DATA-BLOCKED** — no free multi-year history.
- **INSTRUMENT-BLOCKED** — the instrument/execution domain does not
  exist.
- **COST-LINE** — gross amplitude below costs (the arithmetic is
  shared across the family).
Record the cause of death + resurrection conditions: a new
argument/data/instrument; "none known" = honest.

## PASS CRITERIA

The trigger occurs at a measurable frequency; free history >= the
minimum window; the instrument trades; no grave, or a disclosed new
mechanism exists; the sign is not inverted; amplitude above the cost
line.
→ Next: 14-incremental-test (the incremental — the screen's start) +
12-null-control (nulls).

## REAL EXAMPLE (from the graveyard)

**RND strike-density audit (round-16, DATA-DEATH):** 105,713 rows,
2975 days. The strike-density gate PASSED (median 15 ≥ 8), but
density non-negativity failed — 79.8% of core cells < 90%; the
Black-Scholes smoke test showed a "pure saw" (27-30 sign flips):
quote noise destroys the second derivative of the smile. The family
closed before any shape/GLM tests. Nearby: the Alt-VRP wheel — the
share of SOL days with IV<30% = 0.000% (PREMISE-DEATH); dispersion
arbitrage — index crypto options do not exist (INSTRUMENT-BLOCKED);
IV momentum — AC(1) of ΔIV = −0.268, the premise is inverted.

## COMMON MISTAKES

- Building infrastructure/fetchers before the premise check (CASE 4:
  an 8.9 bp basis killed by one bounded probe); silently testing on a
  short window without disclosure.
- Skipping the graveyard: re-testing something buried = mining.
- Porting an academic sign from stocks without a sign premise check
  (flag 5; MAX: LS −47.2%/year, the bottom of the shuffled
  distribution).
- Reading "premise alive" as "strategy alive": TS-1 — the premise
  survived, the economics ≈ 0 (COST-DEATH at the next step).
