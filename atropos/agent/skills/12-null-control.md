---
name: 12-null-control
description: 6 null controls and the placebo design: shuffled, momentum-null, known-features, subperiod, LOO, calendar (stage 2/7).
when_to_use: At the start of the screen together with the incremental test (14-incremental-test); after the premise check passes.
version: 1.0.0
---


> Note: log/registrar excerpts below quote the closed donor corpus (not shipped). The workspace layout itself is yours: scaffold with `atropos init` (files are generated, never overwritten).
# SKILL 2: NULL CONTROL — the null control protocol (Stage 2 of 7)
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Self-contained protocol: the agent needs only this file + data.
> Taxonomy: FRAMEWORK.md (repo root) §4 (NULL/OVERFIT/INVERTED-DEATH);
> placebo test: TEMPLATES.md (repo root) template 3 [5].

## PURPOSE

Does the result differ from the random geometry of the same class —
row shuffles, random levels, feature permutations, random betas. A
signal indistinguishable from the placebo is noise or folklore
(NULL-DEATH); a signal at the BOTTOM of the distribution is an
anti-signal (INVERTED-DEATH). House standard: 200 draws, the seed
recorded in the log.

## INPUTS

1. A candidate that passed 11-premise-check (premise alive), with the
   raw metric from 13-signal-test.
2. Data with exact paths; the label/signal that gets shuffled.
3. The external reviewer's null verbatim, if claimed (block 5 of the
   brief).

## PROCEDURE — 6 mandatory nulls (the house set)

1. **SHUFFLED.** 200 permutations of the signal×outcome link (the
   alignment is destroyed, the outcome distributions remain); seed in
   the log.
2. **MOMENTUM NULL (the naive baseline).** Naive momentum (30d or the
   candidate's horizon) is the minimal competitor: not beating it = a
   rediscovery of the trend (flag 1).
3. **KNOWN-FEATURES PLACEBO.** 200 shuffled-feature runs of the full
   pipeline (as CTREND): the real one must beat the shuffled.
4. **SUBPERIOD.** The effect across thirds of the sample/years — the
   null "one regime carries everything" (the full battery:
   15-robustness-battery, test [3]).
5. **LEAVE-ONE-OUT.** Drop one symbol/day at a time: an effect living
   on a single carrier (Friday-13th = the day 2021-08-13) is not a
   market effect.
6. **CALENDAR.** Placebo geometry of the class (CME gap: revisit of
   any 3-day level); year dummies — 16-walkforward-calendar.

**PLACEBO DESIGN.** 200 draws for EVERY factor/arm; the report = the
real's percentile within its own placebo sample (k/200 draws
match-or-beat) + the direction convention declared BEFORE the run:
[HIGH = good] (r12: % of shuffles the real beats) or [LOW = good]
(r11: the share of draws with EV ≥ the real).

**ANTI-BIAS PROTOCOL.** Compute the metric on shuffled/GBM series
BEFORE any claim: shuffled-H ≥ real-H = the estimator is measuring
its own bias (Hurst: shuffled 0.627 > real 0.618). Elliott: 100% of
600 shuffled series contain "valid" structure; GBM has more than the
market (150 vs 128). The real inside the 95% bands = the metric is
uninformative, there is nothing to test.

## KILL CRITERIA

- **NULL-DEATH**: the real ≤ the 95th percentile of its own placebo
  sample (p > 0.05) — whatever the t-stats say.
- **OVERFIT-DEATH**: shuffled-H ≥ real-H — the estimator's bias
  dominates.
- **INVERTED-DEATH**: the prereg book sits at the bottom of the
  distribution (≤ 5th percentile: MAX — beats 5.0% of shuffles).
- **ANTI-ANOMALY**: the real is WORSE than random (CME gaps >1.5%:
  fill 54.8% vs null 71.1%) — folklore refuted by a single number.

## PASS CRITERIA

The real > the 95th percentile under its own convention in EVERY
applicable null; anti-bias clean (shuffled below the real); the
momentum baseline beaten. Passing the placebo is NECESSARY but NOT
sufficient: it checks the effect is not noise; only the controls
(14-incremental-test) check it is not SOMEONE ELSE'S.

## REAL EXAMPLE (from the graveyard)

**PASS:** POC-avoidance — placebo percentile **0.0** under BOTH nulls
(0/200 draws match-or-beat, p ≤ 0.005) — genuine, battery ROBUST 7/7
(`runs/poc_robustness.log`). **FAIL:** E6 z-score — PRIMARY at the
100th percentile (everything PASS), but holdout F3 — **41st
percentile**, the CI covers 0 (`runs/retest_e6.log`); CTREND — 200
shuffled-feature draws: p 0.74, the real BELOW the placebo mean
(`runs/round13_ctrend.log`). Caveat: GLM v2 — placebo percentile 100
(the raw ordering is real), but the death was incremental (t +0.73):
a placebo pass does not save you from repackaging.

## COMMON MISTAKES

- Fewer than 200 draws or an unpinned seed; the percentile convention
  not declared before the run (HIGH vs LOW = opposite verdicts on the
  same number).
- "Placebo ROBUST = the signal is alive": CVD passed with percentile
  100 and died on the incremental; nulls + the incremental go
  TOGETHER, as the screen's start.
- A placebo that destroys more than the alignment; skipping anti-bias
  — the Hurst class dies 10/10: the estimator must be audited by a
  synthetic null just like the signal.
