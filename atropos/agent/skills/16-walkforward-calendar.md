---
name: 16-walkforward-calendar
description: Walk-forward on frozen thresholds + calendar control: year dummies, shrinkage vs discovery, fresh days (stages 6-7/7).
when_to_use: Last, after the robustness battery passes.
version: 1.0.0
---

# SKILL 6: WALK-FORWARD + CALENDAR CONTROL (Stages 6-7 of 7, the final ones)
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Self-contained protocol: the agent needs only this file + data.
> Taxonomy: FRAMEWORK.md (repo root) §2 stages 6-7, §3 LAW 5;
> prohibitions: CASES_AND_FLAGS.md (repo root) flag 7 + CASE 5.

## PURPOSE

Does the effect transfer to data that took part neither in threshold
fitting (walk-forward) nor in year selection (calendar control). A
yearly regime in a feature/weight/rubric = calendar fitting: crypto
years are heterogeneous, and a "signal" with a yearly term is a
regime label, masked momentum (flag 7).

## INPUTS

1. A candidate that passed 15-robustness-battery, with a known
   discovery point (the raw effect value for the shrinkage
   comparison).
2. A panel with dates; the full list of the candidate's
   parameters/thresholds/weights.

## PROCEDURE

1. **EXPANDING TRAIN -> TEST (not rolling).** Each fold trains on ALL
   history before its test slice, applies thresholds frozen on that
   train (p25/p50/p75) to the next fixed test slice (say 30d), then
   the train EXPANDS; shift, repeat. Doctrine: FRAMEWORK.md (repo
   root) §2 stage 6, "Why expanding, not rolling" (FRAMEWORK.md:221-233)
   - for a falsification stage more accumulated history must mean more
   held-out evidence and a sharper verdict; a rolling window discards
   data exactly when confidence should grow and smuggles in a free,
   itself-tunable window-length hyperparameter. The reference
   implementation (`atropos.walkforward_framework`) offers no rolling
   mode at all; the earlier expanding ban is obsolete and superseded.
2. **THRESHOLD FROZEN ON TRAIN.** ZERO parameters are revisited after
   the data (flag 3); editing a threshold after the run = a new
   version, the old test is void.
3. **SHRINKAGE CHECK.** The OOS effect vs discovery: a multiple
   shrinkage (Ex-kurt: −0.024 → −0.00457, **~5x shrink**, t −0.57) =
   overfit.
4. **YEAR DUMMIES IN REGRESSION.** The spec with year fixed effects:
   the signal's t with year dummies dropping < 2 OR the beta flipping
   sign across cells = **calendar contamination**. ETF flows: every
   year-dummy spec flips the beta sign; the full-sample t 2.96 exists
   only when pooling years without FE (5/20 specs with |t|≥2 —
   exactly the cells without year dummies).
5. **FRESH-DAYS REPLICATION.** Replication on days seen by neither
   the score nor the author: the only test that catches a yearly
   prior.
6. **CALENDAR TERMS FORBIDDEN IN SCORING.** The scoring/rubric must
   contain no year/regime terms (explicit dummies, multi-year
   lookbacks, "author's knowledge"); GLM v2: calendar terms are
   forbidden outright. The calendar class (holidays, hours, sessions,
   Friday-13th) is buried — do not run it without a new mechanism.

## KILL CRITERIA

- **Shrinkage > 2× → OVERFIT-DEATH.** Code semantics
  (`atropos.walkforward_framework.walkforward`): the shrink ratio
  `wf_shrink_ratio = wf_t / in_sample_t` FINITE and STRICTLY above
  `shrink_cap` (default 2.0) → verdict FAIL with `verdict_reason`
  starting `OVERFIT_DEATH` (strict `>`: a ratio exactly at the cap
  does not fire). A NaN ratio (degenerate baseline t) → INCONCLUSIVE,
  never a silent pass. Source: FRAMEWORK.md (repo root) §2 stage 6
  kill criterion — "shrinkage > 2× (the test result is worse than
  half the train result) → OVERFIT-DEATH, stop".
- **WF t tiers (only when the shrink gate did not fire).** `wf_t ≥ 2`
  with direction consistency ≥ 0.5 → PASS; `wf_t ≥ 1` → DEGRADED;
  otherwise FAIL — the OOS effect not replicated.
- **The effect in a single subperiod** — a regime artifact (a rerun
  of battery test [3]).
- **Calendar-dependent:** t < 2 with year dummies or the beta sign
  flipping with year fixed effects = a calendar prior, not a signal.

## PASS CRITERIA

WF t ≥ 2 on frozen thresholds with the shrink ratio not over the 2×
cap; the CI excludes 0; the effect lives in several subperiods; the
incremental holds with year dummies; replication on fresh days
reproduces the pilot.
→ Red thread (FRAMEWORK.md (repo root) §5): one run, one verdict —
graveyard or a dated prereg; whatever survives everything = a
conditional survivor with gates, NOT a leg of a system.

## REAL EXAMPLE (GLM v1 — calendar contamination)

**The GLM v1 rubric (tombstone 118, strike S2):** the SPOT30 pilot
**+0.53 (t +5.50)**, dir hit positive in every year
(64/75/73/76/77/58%). Then 189 FRESH days: **+0.053 (t +0.73)**, dir
hit 52% against a base of 56%, the 2026-flag hit 42%. Diagnosis: the
pilot = a selection of 80 stratified days + the rubric term "yearly
regime" (2023-25 up / 2022+26 bear), acknowledged as a calendar prior
in §2 of its own prereg. Outcome: 3 strikes — permanent close,
calendar terms forbidden in v2. Nearby: Ex-kurt WF — the in-sample
spread (t −4.54) shrank ~5x in walk-forward (t −0.57) → GRAVEYARD.

## COMMON MISTAKES

- A rolling window instead of expanding — rolling discards
  accumulated held-out evidence and adds a tunable window-length knob
  (FRAMEWORK.md:221-233 supersedes the old rolling mandate);
  revisiting a threshold after the first WF block ("too harsh") =
  mining.
- Checking only WF and not the year dummies: the full-sample t 2.96
  exists only through pooling years (ETF flows) — WF without year FE
  will miss the calendar signal.
- Counting a pilot on stratified days as validation: only forward
  replication on days outside the pilot sample catches a yearly
  prior.
- Keeping a "yearly regime" in the rubric as "context" — it is a
  scoring term; the ban is absolute (GLM v2 §6).
