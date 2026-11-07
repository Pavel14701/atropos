# Verdict journal — SYSTEM-9 legs, split BY FEED

Verdicts are anchored to a FEED, not to a leg: the same leg on a
different feed representation is a different run with a different
verdict (FRAMEWORK.md §4). Read every entry below as
"<leg> on <feed>", never as "<leg> is dead/alive".

Prereg frames: skill-driven runs frozen before data touch
(spec_hash 79f8c57f / e9998eb7 / c183527a, daily feed sha
b5078612e27735...); trade-level and predictor-form runs anchored to
their frozen feed files (prereg_hash in each entry). Structured
source of record: `Desktop/atropos_transfer/journal_verdicts.json`
(outside this repo) + `atropos_transfer/skillrun/` reports.

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

### C''-daily — verdict: DEAD (class DEAD — proven no-edge)

- battery baseline mean **−0.0114R/day** over 266 active days —
  catastrophically unprofitable; 5/7 attacks failed
- interpretation: "battery baseline is catastrophically unprofitable
  ... proven no-edge on this feed"
- next_step: stop — rebuild the rule or fix the feed before
  re-testing
- prereg_hash e9998eb7; killed_by_stage 3

### E-daily — verdict: ARTIFACT (class NEEDS_ANALYSIS)

- stage-1 null controls failed on 502 obs (the effect is
  indistinguishable from alignment/placebo/concentration/calendar
  nulls); 306 active days
- next_step: check the feed for lookahead/alignment/calendar
  artifacts
- prereg_hash c183527a; killed_by_stage 1

## Feed 2 — C'' trade-level (3398 OB-retest trades, 12 symbols)

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

Sweep note (seed-7, noise world): the v7 pre-decision ("battery
OVERALL DEAD → DEAD/class DEAD") assumed a no-active-days baseline;
the corrected classifier reads that world ROBUSTNESS_DEATH /
NEEDS_ANALYSIS (baseline positive-but-insignificant). The v7
pre-read was wrong, not the classifier.
