# Verdict journal — SYSTEM-9 legs through the pipeline

Real SYSTEM-9 legs run through `atropos.ensemble_pipeline.run_full_pipeline`
(verdict-taxonomy build, post-21a1ccb), sequential runs, threads pinned
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`). Config = pipeline defaults
with `target="ts"` explicit. Feeds were built from the frozen trade/funding
caches of the trading repo by the transfer-layer builders; `prereg_hash`
anchors each verdict to its frozen feed file. Structured source of record:
`Desktop/atropos_transfer/journal_verdicts.json` (outside this repo).

## SYSTEM-9 leg C'' — verdict: REPACKAGING

- verdict: **REPACKAGING** (verdict_class NEEDS_ANALYSIS, informative true),
  killed_by_stage 1, failed_test None
- n_obs: 3368 | active_days: 3368 | window: 2024-03-18 .. 2026-09-18
- interpretation: the signal's edge matches a known factor (stage-2 spec
  curve, n_obs=3368); not novel, but not dead
- next_step: identify the parent factor; orthogonal-information check
- what_would_change: evidence the signal carries information orthogonal to
  the parent factor
- stage 1 (null_control_suite): FAIL — 1/6 controls passed; failed:
  shuffled_permutation, momentum_control, known_features_control,
  subperiod_stability, calendar_control

Run facts: prereg_hash `6ef9383289163d0e:C''` (sha256 of the frozen feed
file `c2_trade_feed.json`); feed = 3398 cached OB-retest trades
(`runs/ob_oos_trades_cache`, 12 symbols, tp3.0 okx21), returns = booked
net_r, signal = direction signed by the frozen walk-forward composite
bucket (GRID_CONSERVATIVE, pooled expanding-month thresholds, min_bucket
50, warmup 6 months → 457 neutral trades); builder `build_trade_feed.py`
in the transfer layer; no budget reductions.

## SYSTEM-9 leg E — verdict: RISK

- verdict: **RISK** (verdict_class NEEDS_ANALYSIS, informative true),
  killed_by_stage 5, failed_test None
- n_obs: 30916 | active_days: 30525 | window: 2023-12-29 .. 2026-11-07
- interpretation: drawdown/risk envelope breached while statistical gates
  pass — the edge is real, the sizing is wrong
- next_step: reduce exposure, diversify, accept lower scale
- what_would_change: sizing within risk limits
- stage 1 (null_control_suite): WARN — 5/6 controls passed; failed:
  subperiod_stability
- stage 5 (monte_carlo): FAIL — SR p5=-5.02 <= floor 0.00; mDD
  p5=-100.0% < limit -60.0%

Run facts: prereg_hash `b1a9c4764f68bdff:E` (sha256 of the frozen feed
file `funding_feed_e.json`); feed = PREDICTOR form, 30946 asset-days
(frozen v1 UNIVERSE, 30 Binance perps, `data/funding_binance`, window
2023-09-29..2026-11-07): returns = next-day funding accrual (sum of the
day's 8h rates, frozen `to_daily` agg), signal = trailing-7d mean funding
z-scored by trailing-90d std, both ending d-1 (point-in-time; 2700 warmup
rows dropped by the aligner); NO booked R, NO position state, NO costs;
builder `build_e_predictor_feed.py` in the transfer layer.

---

## Reading these verdicts

Per the kill-verdict semantics (FRAMEWORK.md §4): neither entry is a
statement that the strategy is dead. C'' is REPACKAGING — its trade-level
edge is explained by known momentum/feature factors; the orthogonal-
information check is the next step. E is RISK — its predictor form
(funding z-score → forward funding) passes stages 1-4 outright (wf_t 79.6,
shrink 1.00) and dies only on the risk envelope: the premium is real, the
sizing/scale question is the finding. Both are NEEDS_ANALYSIS — the next
step is named in each entry, and "what would change the verdict" is stated
per token in GLOSSARY.md.
