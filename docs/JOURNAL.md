# Verdict journal — SYSTEM-9 legs, split BY FEED

Verdicts are anchored to a FEED, not to a leg: the same leg on a
different feed representation is a different run with a different
verdict (FRAMEWORK.md §4). Read every entry below as
"<leg> on <feed>", never as "<leg> is dead/alive".

**Feed-granularity rule** (added after the C''-daily design error,
see the VOID entry below): the feed granularity must match the
strategy's decision granularity. Aggregating a sub-daily strategy to
daily booked R is not a time-frame test — it is signal loss: the
daily aggregate collapses the leg to "how much was earned", and the
why (zone entry, direction, zone size, retest delay) is gone.

Prereg frames: skill-driven runs frozen before data touch
(spec_hash 79f8c57f / e9998eb7 / c183527a, daily feed sha
b5078612e27735...); trade-level and predictor-form runs anchored to
their frozen feed files (prereg_hash in each entry). Structured
source of record: `Desktop/atropos_transfer/journal_verdicts.json`
(outside this repo) + `atropos_transfer/skillrun/` reports.

**Verdict summary (valid runs only):**

| Leg × feed | Verdict | Class |
|---|---|---|
| A-daily (booked R, 502 obs) | ROBUSTNESS_DEATH | NEEDS_ANALYSIS |
| **C''-trade-level (3398 trades)** | **REPACKAGING** | NEEDS_ANALYSIS |
| E-daily (booked R, 502 obs) | ARTIFACT | NEEDS_ANALYSIS |
| E-predictor (30946 asset-days) | RISK | NEEDS_ANALYSIS |
| A:agg (2407 obs, external) | NO_OOS_EDGE | NEEDS_ANALYSIS |

VOID — not a verdict: C''-daily (daily aggregate of a 15m
OB-retest leg; see the VOID entry below).

## Feed 1 — daily booked-R (502 obs: signal = net exposure sign,
## returns = daily booked R net of frozen fees)

### A-daily — verdict: ROBUSTNESS_DEATH (class NEEDS_ANALYSIS)

- baseline mean **+0.0156R/day** (positive) over 71 active days —
  but not significant, and 5/7 attacks failed:
  threshold_sensitivity, window_sensitivity, subperiod_thirds,
  leave_one_out, declustering
- interpretation: "battery OVERALL DEAD while the baseline mean is
  positive but not significant — the baseline does not survive the
  attacks; NOT proven no-edge"
- next_step: mechanism analysis (strategy-class property vs data
  property) — do not scale until explained
- prereg_hash 79f8c57f; killed_by_stage 3
- note: the pre-fix classifier read this feed DEAD/class DEAD
  ("proven no-edge") — that read was the v8 glue defect, corrected
  by the OVERALL-DEAD split

### C''-daily — VOID (design error, not a verdict)

Status: **VOID — the feed is invalid for this leg by construction.**
The daily aggregate is not a time-frame test of C'': collapsing a
15m OB-retest leg to one number per day ("how much was earned")
destroys the signal — the zone entry, direction, zone size and
retest delay are gone. The run measured "daily booked-R P&L of C''
read as a momentum signal", which is not a test of C''. The run's
mechanical output (battery OVERALL DEAD, baseline mean −0.0114R/day
over 266 active days, 5/7 attacks failed) is recorded here as a
design-error record, NOT as a verdict on the leg.

Cause: one dump format for all legs (`days / signal / returns`).
Valid for A (a trend leg lives for days); circular for E; and for
C'' a complete signal loss. Nobody stopped at "this feed is invalid
for C'' by construction".

The ONLY valid C'' verdict is trade-level: REPACKAGING / class
NEEDS_ANALYSIS (Feed 2 below).

Unblocks: nothing to fix in the pipeline — the lesson is the
feed-granularity rule (header above, skill 02). Any future C''
evaluation is trade-level only.

### E-daily — verdict: ARTIFACT (class NEEDS_ANALYSIS)

- stage-1 null controls failed on 502 obs (the effect is
  indistinguishable from alignment/placebo/concentration/calendar
  nulls); 306 active days
- next_step: check the feed for lookahead/alignment/calendar
  artifacts
- prereg_hash c183527a; killed_by_stage 1

## Feed 4 — A:agg (2407 obs, external harness run — NOT re-run on
## the new classifier)

### A:agg — verdict: NO_OOS_EDGE (class NEEDS_ANALYSIS) — UNCHANGED

- battery PASS 5/7 at stage 3, killed at stage 6 (NO_OOS_EDGE);
  baseline +0.052
- unchanged, not re-run on the new classifier: stage 3 PASSED, so
  the battery OVERALL-DEAD split does not apply to this run (the
  split only re-classifies battery-OVERALL-DEAD outcomes)
- next_step: the entry-level edge lacks an OOS explanation through
  the tested features — feature/mechanism question, separate work

## Feed 2 — C'' trade-level (3398 OB-retest trades, 12 symbols) —
## THE valid C'' feed

### C''-trade-level — verdict: REPACKAGING (class NEEDS_ANALYSIS)

- stage 1: 5/6 null controls failed (placebo PASSED) — the
  trade-level edge is explained by known momentum/feature factors;
  "not novel, but not dead"
- next_step: identify the parent factor; orthogonal-information
  check
- prereg 6ef9383289163d0e; n_obs 3368

## Feed 3 — E predictor-form (30946 asset-days, 30 perps,
## signal = funding z-score → returns = forward funding)

### E-predictor — verdict: RISK (class NEEDS_ANALYSIS)

- stages 1-4 ALL PASS (stage-2 max|t| 69.1; stage-4 wf_t 79.6,
  shrink 1.00); stage-5 risk envelope breached (SR p5 −5.02, mDD
  p5 −100% vs −60% limit) — the premium is real, the sizing/scale
  question is the finding
- prereg b1a9c4764f68bdff

## Reading these verdicts

DEAD = proven absence of edge ON THIS FEED (two cases only: battery
baseline catastrophic, or CS_NO_EDGE at an XS target).
NEEDS_ANALYSIS = the gate was informative but did not prove absence
— the entry names the next step. SURVIVES = could not be killed,
not proof of truth. Per-token meanings: GLOSSARY.md; the five-facts
rule: FRAMEWORK.md §4.

Baseline-numbers note (probe vs pipeline): the pre-fix discussion
quoted A +0.0083 and C'' −0.0058 — those were raw-532-row probes;
the pipeline runs the battery on the aligned 502-row series (30-row
mom30 warmup dropped), measuring A +0.0156 and C'' −0.0114.
Verified by direct run: same battery, same threshold, different
observation windows. Classification unchanged on both (A positive →
ROBUSTNESS_DEATH; C'' negative → proven-absence DEAD); the
classifier fix did not touch the baseline-mean computation (it only
reads the row).

Sweep note (seed-7, noise world): the v7 pre-decision ("battery
OVERALL DEAD → DEAD/class DEAD") assumed a no-active-days baseline;
the corrected classifier reads that world ROBUSTNESS_DEATH /
NEEDS_ANALYSIS (baseline positive-but-insignificant). The v7
pre-read was wrong, not the classifier.
