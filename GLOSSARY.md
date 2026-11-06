# Glossary

The vocabulary of the kill chain. Each term: what it means here, and a
mini-example of its use. For the full death taxonomy see `FRAMEWORK.md` §4.

## Lifecycle

**Kill-protocol** — the framework's core loop: a preregistration is
frozen, the adversarial pipeline runs (rule chain: 7 stages,
`atropos.ensemble_pipeline.run_full_pipeline`; ML chain:
`run_ml_kill_protocol`, four gates over purged walk-forward), and the
hypothesis leaves with exactly one verdict. Example: prereg committed
Monday → 7 stages Tuesday → `REPACKAGING` at stage 1 → tombstone.

**Preregistration (prereg)** — a hypothesis document frozen *before* the
data is touched: prediction, construction, nulls, numeric kill criteria,
power calc, spec hash. Generated and validated from a params dict by
`atropos.prereg_generator.generate_prereg`. Example: a prereg that says
"|t| after momentum controls < 2.0 => KILLED" is committed on Monday, so
Tuesday's regression cannot move the goalposts.

**Kill criterion** — a directional numeric rule written in the prereg
(`metric comparison threshold`, e.g. `IC < 0.05`), never edited after the
first data touch. Example: "net-of-cost Sharpe floor 0.0 breached in the
Monte Carlo p5 tail" is a criterion; "looks weak" is not.

**COD (cause of death)** — the recorded reason a hypothesis died: which
stage killed it and why. Example: `killed_by_stage=1, COD: repackaging —
adds nothing beyond known momentum`. The pipeline's `kill_reason` field.

**Tombstone** — the append-only record of a finished hypothesis: verdict,
COD, evidence, resurrection conditions. Never edited or deleted. Example:
`graveyard/H-0007.md` recording "DEAD — battery catastrophic failure on
window sensitivity, w=15 inverts the spread".

**Graveyard** — the collection of all tombstones; the first thing you
check before proposing anything (a structurally identical KILLED
hypothesis is not re-run without a new mechanism). Example: a graveyard
check stops you from re-testing "overnight gap reversal" that died in
2024-03 with the same prediction and kill criterion.

**Resurrection conditions** — the part of a tombstone that states under
what new facts a dead hypothesis may be reopened. Example: "reopened only
if a new venue cuts costs below 2 bps" for a COST_DEATH tombstone.

## Verdicts

**SURVIVED / KILLED / INCONCLUSIVE** — the three abstract outcomes. The
pipeline emits concrete names: `SURVIVES`, `SURVIVES_WITH_WARNINGS`
(the survived family), seven kill verdicts (`REPACKAGING`, `ARTIFACT`,
`DEAD`, `NO_OOS_EDGE` x2 stages, `COST_DEATH`, `RISK`, `CS_NO_EDGE`),
and `INCONCLUSIVE` when the data cannot decide. Example: SURVIVES means
"earned the right to a live track record", not "is true".

**Null control** — a test that destroys the hypothesis if the effect
survives only in the un-manipulated world: permutation, placebo,
sub-periods, calendar, known-features. Example: shuffling event dates
kills an "event-driven" edge whose whole Sharpe came from three days.

**Repackaging** — the "new" signal is a known factor (e.g. momentum) in
disguise; stage 1/2 verdict when known-features controls absorb it.
Example: signal = 7-day momentum + noise regresses onto observed momentum
with R^2 ~ 1 => REPACKAGING.

**Cost death** — the gross edge is real but net-of-cost returns do not
clear the floor; transaction costs ate the edge (stage 5, `COST_DEATH`).
Example: 40 bps round-trip on a 6 bps mean spread => gross SR 1.2,
net SR < 0.

**CS_NO_EDGE** — the cross-sectional death: per-date IC, net quantile
spread, or quantile monotonicity fails on a panel (stage 7). Example:
top-minus-bottom quintile spread is +12 bps gross but -3 bps net of
costs and non-monotone across quintiles.

## Methods

**Walk-forward + purge gap** — out-of-sample evaluation that rolls a
train/test window through time, with rows purged at the seam so the
label horizon cannot leak from train into test. Example: train 180 days,
purge 1 day (one-period forward return), test 30 days, step 30.

**DSR / PSR** — deflated and probabilistic Sharpe ratios (Bailey & Lopez
de Prado): shrink a Sharpe ratio for the number of trials tried (DSR) or
relative to a benchmark SR (PSR; DSR with one trial). Example: SR 2.0
looks great until DSR deflates it for the 100 backtests that produced it.

**Block bootstrap** — resampling *contiguous blocks* of a return series
(preserving autocorrelation) instead of iid rows, to build honest
confidence intervals. Example: 40-day blocks resampled 1000 times give
the Sharpe CI that an iid bootstrap would understate.

## ML engine

**Model registry** — the open `MODEL_REGISTRY` in `atropos.ml_training`:
name -> factory; a factory takes keyword arguments from
`DEFAULT_TRAINING_CONFIG["model_params"][name]` plus a `seed` and
returns an unfitted sklearn-style estimator. Eight bundled entries
(gbm, logit, rf, extratrees, hist_gbm, mlp, knn, qda) plus the
`ensemble` composite after `import atropos.ml_ensemble`. Example:
`MODEL_REGISTRY["xgb"] = make_xgb` immediately makes
`config={"model": "xgb"}` work — no engine changes.

**Walk-forward (ML)** — the engine's expanding-block CV over
monotone int64-microsecond timestamps: every fold fits fresh on
strictly earlier rows and scores the next block; models share one CV
pass and are compared, never selected (the engine compares, the
caller decides). Example: `k_blocks=6` folds over hourly stamps;
per-model verdicts land in `report["models"][name]`.

**Purge / embargo** — the two leakage guards at the train/test seam:
purge drops train rows whose label window overlaps the test fold,
embargo additionally drops rows just before the test start so stale
information cannot leak backwards. Example:
`purge_embargo_mask(..., horizon_h=1, purge_h=2, embargo_h=24)`
keeps a train row only if label window and 24 embargo hours both end
before the test stamp.

**OOF (out-of-fold)** — scores produced on rows the fold's model
never trained on; the only honest currency for combining models.
Example: `oof_score_matrix` builds member scores per past-only time
tile, so the stacking meta-learner never reads in-sample scores.

**Stacking / nesting** — stacking trains a meta-learner over member
scores; it is honest only inside two nestings ("no stacking without
nesting"): the outer walk-forward folds and the inner past-only OOF
tiles. Example: a meta-learner fitted on in-sample member scores
reads overfit scores and learns garbage weights — the exact failure
the nesting rule exists to prevent.

**UNDERPOWERED** — the ML run verdict when fewer than `n_min_events`
declustered events are taken: undecided, NOT a kill (gate flags stay
informational). Example: 200 independent events under a 500 floor
come back `UNDERPOWERED`; a fuller snapshot may still decide the
hypothesis, so the unified chain maps it to INCONCLUSIVE.

**Cost-aware EV (executable return)** — the engine scores thresholds
on EV over executable returns the caller has already netted of costs
(`exec = fwd - cost`), so the cost-line gate kills trades that cannot
pay their way. Example: a gross 8 bp edge at a 20 bp round-trip cost
has negative executable EV -> `DEAD(cost_line)`.

**Unified verdict (kill_chain)** — the aggregate of the rule and ML
chains: KILLED if either chain kills, SURVIVES only if every executed
chain survives, INCONCLUSIVE when a chain cannot decide. Example:
rule SURVIVES + ML `UNDERPOWERED` -> `INCONCLUSIVE [ml]`; rule
REPACKAGING + ML DEAD -> `KILLED [rule+ml]`.

## Backtest pipeline

The reference check that runs after a verdict (`POSITIONING.md`):
simulate → rank → admission → metrics. Its vocabulary:

**CandidateGroup** — the interchange object of the ranking layer
(`atropos.ranking`): one decision point's candidates with RAW model
scores and realized labels (`NaN` label = untradeable). The admission
layer consumes it only after `rank_normalize_group`. Example:
`CandidateGroup(group_id="2024-01-02", candidates=["c_long",
"c_short"], scores=[0.9, 0.7], labels=[0.012, -0.004])`.

**RANKING_METRICS** — the open per-group ranking-quality registry in
`atropos.ranking`: `ndcg@k`, `precision@k`, `mrr`, `spearman`;
registering a metric (`RANKING_METRICS["ap@k"] = my_fn`) makes it
available to comparison sweeps. Metrics read RAW scores of scored
groups only — within a group raw and normalized scores induce the same
order. Example: the golden path's mrr is 0.759 (gbm) vs 0.556 (random).

**rank_normalize_group** — within-group rank normalization to `[0, 1]`
with average ties (`atropos.ranking`; `atropos.backtesting` re-exports
it): best candidate 1.0, worst 0.0, ties share the average rank
position; a single-candidate group maps to exactly `0.5` (neutral).
Admission reads ONLY normalized scores — raw cross-model probabilities
(a logit 0.9, a kNN 0.7) are not comparable. Example: `[5.0, 5.0, 1.0]`
→ `[0.75, 0.75, 0.0]`.

**Admission policy** — the slot-capacity replay over a candidate
stream (`atropos.backtesting.run_admission_policy`): `fcfs` (slots
full → reject), `replace_low` / `replace_high` (a newcomer STRICTLY
beating the worst open score closes that trade, which contributes 0 /
keeps its full isolated R). Example: cap=2, the best trade arrives
while both slots are busy — FCFS rejects it; REPLACE-low swaps out the
worst holder.

**RANKER_SUITABILITY** — honest per-model suitability marking in
`atropos.ranking` — information, not a gate: `gbm`/`logit`/`hist_gbm`
high, `rf`/`extratrees`/`mlp` medium, `knn`/`qda` low (noisy
probabilities on small candidate groups; under ~10 candidates per
group prefer the high marks). Example: a 6-candidate group world —
mrr@2 gbm 0.404, logit 0.397, knn 0.385.

**Pointwise vs lambdarank** — the two ranking oracles in
`atropos.ranking`: `rank_within_groups` fits any registry classifier
pointwise on `P(win)` (the default, no extra dependencies), while
`rank_with_lambdarank` ports the donor `LGBMRanker` fit (lightgbm is
an optional dependency, imported lazily inside the call). They rank by
different semantics; compare their picks through `RANKING_METRICS`,
never assert them equal.

**ev_threshold** — the EV-calibrated probability threshold
(`atropos.ranking`): scans the grid with INCLUSIVE `>=` plus the
all-trades floor (`-inf`) and returns `(threshold, ev, n_taken)` —
deliberately different from `atropos.ml_training.pick_threshold`
(strict `>`, bare grid, bare threshold float). Example: on the same
`proba`/`exec_vals`, `pick_threshold` returns `0.5` (a loss-making
cut) while `ev_threshold` returns `(-inf, 0.03, 3)` — the floor wins.

**force_close_r** — the RECONSTRUCTED P&L of a force-closed trade
(`run_position_state_machine`, `allow_reverse=True`): an opposite-side
signal closes the open trade at its decision bar and the row's isolated
`r_pess`/`exit_idx` stay STALE — read `force_close_r` instead. Example:
isolated `exit_idx` 5 but force-closed at bar 3 → `force_close_r`
`0.697` next to the stale `exit_idx` 5.

**same_bar_reentry** — the `run_position_state_machine` flag deciding
whether a signal decided on the exit bar may take the freed slot:
default skips it (`same_bar_after_exit` — SL-first pessimism means the
position is still held during that bar); with the flag the signal
takes the slot (cooldown still applies on the exit bar).

**Gap-through-stop** — the `simulate_signal` contract for an entry bar
that already OPENS beyond the stop: the trade is a scratch at costs
(exit on the entry bar), never a +R windfall in gap-distance units.
Example: a long planned with a stop at 102 fills at the 101 open →
`r_opt -0.253` / `r_pess -0.404`, `exit_idx` = the entry bar.

## Doctrine

**Defense-in-depth** — no single test decides; seven stages of
independent attacks must all miss before anything SURVIVES. Example: a
signal can pass null controls and still die at stage 5 on the drawdown
tail.

**INCONCLUSIVE vs FAIL** — FAIL is evidence against the hypothesis
(a stage attacked and won); INCONCLUSIVE means the data cannot decide
(too short, snapshot does not cover it) and reopening on a new snapshot
is allowed. Example: a 90-day live track is INCONCLUSIVE at
`live_verdict` (shorter than the default 252-observation decision
point), not a pass and not a kill.
