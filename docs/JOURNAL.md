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
- **Orthogonal feature check (aeb9332+)**: full HC0 regression of
  net_r on direction / strength / zone_width_atr / retest_delay with
  momentum controls (n=3368, Bonferroni threshold |t| ≥ 2.5):
  **zone_width_atr t = −9.33** and **retest_delay t = −9.92** —
  HIGHLY significant orthogonal contributors (both p < 1e-20);
  strength t = −2.08 (p = 0.038, marginal); direction itself
  t = −1.72 (p = 0.085, NOT significant).  The OB zone-selection
  geometry carries real information that momentum controls do not
  capture — the C'' REPACKAGING stands for the direction claim, but
  the zone-geometry finding opens a new analysis thread: which zone
  properties carry the contribution (tight/fast vs wide/slow)?
- prereg 6ef9383289163d0e; n_obs 3368

### C''-trade feature analysis (v12) — exploration, NOT validation

Run: `atropos.feature_analysis` over the same 3398-trade cache
(outcome net_r; features direction / strength / zone_width_atr /
retest_delay; controls mom7/mom30; Bonferroni family = 4 clusters,
gate |t| ≥ 2.50; 60/40 frozen split). Full report verbatim:
`Desktop/atropos_transfer/feature_scan_report.json`. This is the
zone-geometry thread the orthogonal check opened, mapped through the
feature lab. **feature_analysis is exploration, not validation;
validation requires prereg + run_full_pipeline.**

- classification: **retest_delay STRONG_UNVERIFIED**,
  **zone_width_atr STRONG_UNVERIFIED** (orthogonal pass + monotone;
  the regime split below is data-derived, so the cell cannot become
  STRONG); **direction PROXY, strength PROXY** (orthogonal gate fail
  — consistent with the v11 HC0 ts −1.72 / −2.08); cluster count
  n_effective 4.
- missingness: zero NaN flags on all four features.
- bucket_scan (feed-quantile edges; see the mis-scale note below):
  zone_width_atr monotone, bucket means 0.837 → 0.261 across the
  width quintiles, Q2−Q1 net_t −6.43 / norm_t −6.43, hit-rate z
  only ≈ −2.0 — **the width effect is a magnitude effect, not a
  hit-rate effect**. retest_delay monotone, means 0.717 → 0.118,
  all three signals fire (t −7.50) — **delay kills hit rate AND
  magnitude**.
- interaction zone_width × delay: NOT present (F p = 0.18, no
  low-power cells) — the two effects are additive.
- rules (frozen 60/40): `narrow_fast` (width < 1.5 & delay ≤ 5)
  delta_test **+0.37R** (n 149) and `narrow_only` +0.37R (n 389) —
  both valid; `best_valid_rule = narrow_fast` (the deltas tie, the
  tool names the higher one).
- threshold sweep width ∈ [1.2, 2.4]: **plateau, no edge** — every
  point valid, delta_test 0.32–0.42 (optimal_range the whole sweep):
  a wide ridge, not a pick-one-number spike.
- regime scan (trailing-30-trade net_r std, calm/storm at the
  median — **data-derived, prereg_hash None → REGIME_UNVERIFIED**):
  retest_delay monotone in both regimes (stable); zone_width_atr
  flips — calm rho −0.000 (nothing), storm rho −0.088 (monotone):
  the width effect concentrates in high-volatility regimes. This is
  a hypothesis for a PREREGISTERED indicator, nothing more.
- rule generation (grid width < t1 & delay ≤ t2, 16 combos,
  train-only search, top-5 judged on test): best
  `width < 1.5 & delay ≤ 10` delta_test **+0.62R** (n 230), warning
  False; the train-top combos (width < 1.2) all starve on test
  (insufficient_n) — the n-guard, not the test split, killed them.
- feature_importance (linear, permutation, diagnostic only):
  retest_delay 0.056 [0.044, 0.069] > zone_width_atr 0.037
  [0.029, 0.045] >> strength 0.0025 > direction 0.0018 — the same
  ordering the v11 HC0 t's gave.
- mis-scale note (honest record): the plan's zone_width_atr numbers
  (edges 0.2–0.8, rules < 0.4) sit BELOW the feed's support (0.74–
  7.83, median 1.82); the first run returned all-insufficient /
  low-power cells — the tool's power guards did their job. The
  recorded run uses feed-quantile values, which are data-derived:
  everything above stays exploratory-grade regardless.
- strength is ~90% zeros in this cache (max 0.026) — its PROXY is a
  gate fail on a near-degenerate column, flagged here so nobody
  reads it as a tested negative.

### C''-trade data-contract smoke (v13) — plumbing, facts unchanged

The SAME cache (3398 trades) through the new data-contract layer
(`atropos_transfer/run_data_contract.py`): TidyAdapter → materialize
→ feature_analysis(md).  **Classification identical to the v12 entry
above** (direction/strength PROXY; zone_width_atr/retest_delay
STRONG_UNVERIFIED) — the contract changes HOW data enters, not WHAT
it says.  Manifest: bundle_hash 51a43f4d16924699…, lookahead_checks
all True (verified, not assumed), row_count 3398.  Decision clock is
a SEQUENCE CLOCK (trade date + within-day rank as minutes): it
preserves trade order; as-of windows degenerate to row order
(offsets are 0).  The embargo=0-on-sub-daily warning fires as
designed on this clock.  Momentum controls are NOT as-of-safable on
a sequence clock and were omitted from the materialized bundle —
the v12 entry above (with mom7/mom30) remains the analysis of
record.

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


## Coverage baseline (wave v15, V-0 — 2026-11-08)

First measured baseline (pytest-cov 7.1.0, branch mode, full suite):

* **line 87.78%** (10730 stmts, 1233 missed), **branch 85.4%**
  (2791/3268 covered, 389 partial).
* Top-10 missed: feature_scan 172, backtesting 137 (maker variants),
  portfolio 115 (replay internals), ranking 88, data_contract 81
  (validation ladders), predictive_model_eval 70, legs 69
  (validation ladders), monte_carlo 69, ml_ensemble 61,
  walkforward_framework 56.

**Conditional target** (the honest version — baseline is BELOW the
95/90 aspiration, so the goal is improvement, not the aspiration
itself): 100% of raise/warn/fatal lines covered (wave v15 V-2,
coverage-driven), then line >= 92%, branch >= 89%; anything beyond
is best-effort.  The misses are concentrated in validation ladders
and the maker/backtest variants -- exactly the error paths V-2
targets -- NOT in the verdict-critical chain (cli.py is 99%,
cross_sectional 96%, ml_training 98%).

## Distribution rename decision (wave v15 — 2026-11-08)

Decision 0 (Option A): the DISTRIBUTION is renamed to
`atropos-quant` (the name `atropos` on PyPI belongs to an unrelated
bioinformatics tool), the IMPORT stays `atropos` -- precedents
(scikit-learn/sklearn, pillow/PIL, beautifulsoup4/bs4) all split
distribution from import; the import name is the daily UX, the
distribution name is a one-time `pip install`.  The CLI command
moves to `atropos-quant` (the plain `atropos` command would
collide with the bioinformatics package's CLI).  A NEGATIVE PIN
(`import atropos_quant` -> ImportError) guarantees no silent
import rename later: adding an `atropos_quant` package requires a
dedicated wave with a shim and a deprecation cycle.  Recorded as
an explicit decision, not an accident.
