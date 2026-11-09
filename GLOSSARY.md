# Glossary

The vocabulary of the kill chain. Each term: what it means here, and a
mini-example of its use. For the full death taxonomy see `FRAMEWORK.md` §4.

## Lifecycle

**Kill-protocol** — the framework's core loop: a preregistration is
frozen, the adversarial pipeline runs (rule chain: 7 stages,
`atropos.ensemble_pipeline.run_full_pipeline`; ML chain:
`run_ml_kill_protocol`, four gates over purged walk-forward), and the
hypothesis leaves with exactly one verdict. Example: prereg committed
Monday → 7 stages Tuesday → `REPACKAGING` at stage 1 → routing to
the packaging-vs-direct comparison (§18); the tombstone records the
borrowed edge.

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
pipeline emits concrete tokens, and the authoritative token list lives
in exactly ONE place — the module docstring of
`atropos.ensemble_pipeline`; docs must not re-enumerate it (hardcoded
counts drifted 14/15/16/18 across README, GLOSSARY and how-tos — the
reference kills the drift). Shape of the set as of this writing: the
survived family (`SURVIVES`, `SURVIVES_WITH_WARNINGS`, plus the
panel-only forms `SURVIVES_CS_ONLY` /
`SURVIVES_CS_ONLY_WITH_WARNINGS`), the kill tokens (`DEAD`,
`ROBUSTNESS_DEATH`, `PLACEBO_VIOLATION`, `OVERFIT_DEATH`,
`NO_OOS_EDGE`, `COST_DEATH`, `DEGENERATE_FEED`,
`CONSTANT_PREDICTIONS`, `CS_NO_EDGE`, `RISK`, `ARTIFACT`,
`REPACKAGING`), and `INCONCLUSIVE` when the data cannot decide. The
generic `DEAD` is retired as a catch-all: it survives only
as the proven-absence token (battery baseline catastrophically dead, or
`CS_NO_EDGE` with config target `"xs"`). Every report also carries
`verdict_class` (DEAD / NEEDS_ANALYSIS / INCONCLUSIVE / SURVIVES),
`informative`, `interpretation`, `next_step`, `what_would_change`,
`failed_test`, `active_days`. Example: SURVIVES means "earned the right
to a live track record", not "is true".

### The kill tokens, one by one

Each kill token, exactly four fields — Means / Why fired / Next step /
What would change:

**DEAD** (code token; doctrine name: n/a — this is the proven-absence
class)
- **Means:** absence of edge proven on this feed.
- **Why fired:** battery baseline catastrophically dead (mean ≤ 0 with
  power / no active days), or CS tests found no XS structure at
  required power.
- **Next step:** stop pursuing this signal on this feed; if you believe
  the edge exists, the feed is the problem — new feed, new run, new
  verdict.
- **What would change:** nothing on this feed. DEAD is not transferable
  across feeds.

**ROBUSTNESS_DEATH** (doctrine: ROBUSTNESS-DEATH, FRAMEWORK.md:404)
- **Means:** the signal survived some robustness tests but failed a
  specific structural one — about the signal's shape, not its P&L.
- **Why fired:** a specific battery test returned FAIL while the
  baseline remained positive — check the battery table for WHICH test
  (smoothing inversion is characteristic of mean-reverting books;
  thirds sign-flip means regime-bound; LOO non-persistence means no
  single day holds it).
- **Next step:** name the failed test and the mechanism it implies; if
  the mechanism is a strategy-class property, the class is the issue;
  if a data property, the OOS period is.
- **What would change:** a different signal representation where the
  failed test is not structurally hostile — a different test, NOT a
  softened test.

**PLACEBO_VIOLATION**
- **Means:** the apparent edge survives when the signal is lagged into
  the future — causality violation.
- **Why fired:** placebo test with lag-k produced non-zero edge — the
  most serious informative kill: the strategy as measured is not a
  strategy.
- **Next step:** audit the feed for lookahead (alignment, point-in-time
  semantics, indicator warmup); do not trade until explained.
- **What would change:** a strictly point-in-time feed: edge
  disappears → it was a leak; survives → the verdict changes.

**OVERFIT_DEATH** (doctrine: OVERFIT-DEATH, FRAMEWORK.md:407)
- **Means:** in-sample edge exists, OOS is worse by more than the
  shrink cap — does not generalize as measured.
- **Why fired:** shrink = OOS/IS > 2.
- **Next step:** check whether the IS period overlaps a regime the OOS
  does not.
- **What would change:** an OOS period overlapping the IS regime
  distribution.

**NO_OOS_EDGE**
- **Means:** robustness passed, but forward-looking predictive power is
  not significant in purged evaluation — the tested features do not
  capture the mechanism.
- **Why fired:** OOS predictive metric below threshold while battery
  passed.
- **Next step:** a FEATURE question: test execution quality,
  microstructure, regime timing, carry — do not force the same
  features through a stricter threshold.
- **What would change:** a feature set capturing the actual mechanism.

**COST_DEATH**
- **Means:** gross edge exists, net of modelled costs it does not.
- **Why fired:** breakeven cost ≥ modelled cost at required
  significance.
- **Next step:** reduce turnover, improve execution, or accept lower
  scale — execution problem.
- **What would change:** cheaper execution or a lower-turnover variant.

**DEGENERATE_FEED**
- **Means:** the feed cannot be tested (constant signal, constant
  returns, zero variance, all-NaN).
- **Why fired:** `np.ptp == 0` or all-NaN on signal or returns
  (pre-flight).
- **Next step:** fix the feed — common causes: paused leg not excluded,
  key-mapping error, degenerate threshold.
- **What would change:** a valid feed — any prior verdict on this feed
  is voided in the journal.

**CONSTANT_PREDICTIONS**
- **Means:** the walk-forward model produced constant predictions — the
  model degenerated and the signal was not tested.
- **Why fired:** `preds_std` degenerate inside the walk-forward.
- **Next step:** check the model spec (too simple, window too short,
  single-class fold) — model problem, not signal problem.
- **What would change:** a model spec producing non-constant
  predictions.

**CS_NO_EDGE** (DEAD class only with config target `"xs"`; the default
`"ts"` target reads NEEDS_ANALYSIS)
- **Means:** cross-sectional structure absent at required power.
- **Why fired:** XS tests non-significant at power.
- **Next step:** XS by design → this is DEAD; time-series strategy →
  ignore this verdict for this signal.
- **What would change:** nothing on this feed if XS was the target; a
  different target is a different test.

**RISK**
- **Means:** passes statistical gates, fails risk gates (concentration,
  drawdown, exposure).
- **Why fired:** risk threshold breached.
- **Next step:** reduce exposure, diversify, accept lower scale — the
  edge is real, the sizing is wrong.
- **What would change:** sizing within risk limits.

**ARTIFACT**
- **Means:** the effect is indistinguishable from
  alignment/placebo/concentration/calendar nulls.
- **Why fired:** the null families reject the candidate.
- **Next step:** check the feed for lookahead/alignment/calendar
  artifacts; if clean, the signal is not distinguishable from noise on
  this feed.
- **What would change:** a different signal or feed where at least one
  null is rejected with power.

**REPACKAGING**
- **Means:** the signal's edge matches a known factor — a ROUTING
  verdict, not a death: the candidate is a view of a known premium.
- **Why fired:** stage 1/2 known-features controls absorb the signal —
  it adds nothing beyond the known factor.  Momentum and trend are the
  most robust premia of the last century — a repackaging is a view of
  that premium.
- **Next step:** REPACKAGING routes to the
  packaging-vs-direct-replication comparison: run the DIRECT factor
  implementation (e.g. plain time-series momentum) on the same
  universe and bars through the same cost model.  The packaging must
  beat the direct implementation net of costs, or offer capacity /
  diversification / risk-profile edge.
- **What would change:** a repackaging that cannot is COST_DEATH;
  orthogonal information beyond the factor upgrades it to a genuine
  re-discovery.

**The survives family, one by one** (PASS-class; the distinctions are
evidence scope, not correctness):

**SURVIVES**
- **Means:** every executed attack missed on this feed.
- **Why fired:** all series stages PASS, no warnings.
- **Next step:** the preregistered forward test (`live_verdict`) —
  never immediate capital.
- **What would change:** a new snapshot, a new feed, a new run —
  survival is not transferable.

**SURVIVES_WITH_WARNINGS**
- **Means:** survived with caveats — undecided checks rode along.
- **Why fired:** all stages PASS with WARNs present (undecided
  controls, turnover, deflated Sharpe).
- **Next step:** write the WARNs into the tombstone; decide whether
  the caveats are compatible with the decision.
- **What would change:** a run where the warned checks decide.

**SURVIVES_CS_ONLY** / **SURVIVES_CS_ONLY_WITH_WARNINGS**
- **Means:** only the cross-sectional stage ran (panel-only mode) and
  passed — the evidence scope stays in the verdict instead of
  borrowing the stronger all-stages label.
- **Why fired:** stage 7 PASS (with/without warnings) on a panel
  alone; stages 1-6 SKIPPED for lack of series data.
- **Next step:** a series feed is needed before any stronger survival
  claim; do not read CS-only as all-stages `SURVIVES`.
- **What would change:** series data supplied — the full pipeline
  runs and a new verdict is earned.

**The five-facts rule.** Every kill reports (1) which gate fired, (2)
on what feed (anchor, n_obs, active_days), (3) which specific test
failed, (4) whether the failure is informative, (5) what would change
the verdict. A verdict without all five facts is not a verdict.

Naming: code tokens use the underscore form (`ROBUSTNESS_DEATH`); the
FRAMEWORK doctrine uses the hyphen form (ROBUSTNESS-DEATH,
FRAMEWORK.md §4).

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
`r_pess`/`exit_idx` stay STALE — read `force_close_r` instead,
recomputed by `simulate_signal` from the ORIGINAL entry with the hold
truncated to the closing bar. Example:
isolated `exit_idx` 5 but force-closed at bar 3 → `force_close_r`
`0.697` next to the stale `exit_idx` 5. The portfolio replay reuses
the field for displaced trades (`atropos.portfolio.run_portfolio_replay`):
the replaced position's partial-hold P&L is recomputed to the
replacement bar, while its equity credit still follows the policy — 0
under `replace_low`, isolated `r_pess` under `replace_high`.

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

## Portfolio layer

The concurrent-positions generalization of the admission replay
(`atropos.portfolio`; how-to 09). Its vocabulary:

**Portfolio replay** — the multi-slot replay
(`atropos.portfolio.run_portfolio_replay`): candidate signals are
processed in deterministic order over N concurrent slots, a signal
decided at bar `d` fills at `d + 1`, and every outcome lands in a
ledger row — `accepted` / `rejected` / `replaced` with reasons
(`capacity`, `inferior_score`, `correlation_cap`, `cooldown`). Example:
slots=2, both occupied, FCFS → the newcomer joins `rejected` with
`capacity`.

**Slot admission** — which slot a portfolio candidate takes
(`atropos.portfolio`): the correlation filter consults the OCCUPIED
slots first, then a free slot wins (lowest index) or the admission
policy decides (`atropos.backtesting.run_admission_policy` semantics:
`fcfs` first-come, `replace_low` / `replace_high` evict the worst open
score); a trade holds its slot through its exit bar plus the
per-slot `cooldown`, and an unresolved trade holds it to the end of
the data.

**Correlation cap** — the `|Spearman|` admission threshold
(`atropos.portfolio.filter_by_correlation`): a candidate whose recent
R proxy correlates strictly above the cap with ANY open position's
proxy is rejected (reason `correlation_cap`); insufficient overlap is
inconclusive and admits — not-enough-data is not correlation. Example:
a second signal on an instrument with an OPEN position carries
|rho| = 1 against it and dies under any cap below 1.

**Exposure (gross/net)** — the per-bar position series of a replay
(`atropos.portfolio.ExposureSeries`): `gross` counts held positions,
`net` is the signed sum (longs minus shorts), both bounded by the slot
count; `net` is what the same-side concentration gate reads. Example:
two longs and one short open → gross 3, net 1.

**Portfolio DSR** — the deflated Sharpe computed on the portfolio's
per-period EV stream (`atropos.portfolio.portfolio_gates`, delegating
to `atropos.evaluation_metrics.deflated_sharpe_from_moments`): R
credited per aggregation period is the stream, and a non-finite DSR
fails closed. Example: a 120-bar book with 3 credited trades →
DSR ≈ 0 → `DEAD(deflated_sharpe)` once `min_trades` is met.

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


## Data contract + legs (v13)

**MaterializedData** — the verified bundle every framework function
should trust: outcome/features/controls aligned as-of onto a decision
grid, plus a decision index (ts/entity/split/regime) and a manifest.
Example: `md = materialize(adapter, spec)`; `feature_analysis(md)`
validates the manifest instead of just unwrapping it.

**t_known_offset** — (moment a value is known) − (its timestamp
label). Bar closes carry 0; the OPEN of a close-labelled bar carries
`-bar_delta`. The resolver cuts at `t - t_known_offset - lag*bar_delta`
and then re-verifies the produced column against an independently
recomputed cutoff — look-ahead is caught, not assumed. Example: a
feature declared with offset 0 delivers `raw[t]`; with lag 1 it
delivers `raw[t-1]` (the forward-label convention).

**DataManifest** — provenance + verification record of one
materialization: full sha256 `bundle_hash` (the anchor a prereg
cites), per-field `lookahead_checks`, `feature_hashes`, validation
battery outcome. Example: two runs over changed data produce
different bundle hashes — a prereg anchored to the old hash is
invalid by construction.

**Adapter** — the dumb source normalizer (`fields/extent/fetch/
as_of/validate_self`): declares FieldMeta, serves raw fields, and
does NO lag/alignment/splits (the resolver owns those). Example:
`OHLCVAdapter(bars, "close", 1h, symbol="TEST")` serves flat close
series; the open of a close-labelled bar declares offset `-bar_delta`.

**LegSpec** — a position-lifecycle declaration in four separate
layers (signal / instrument / execution / sizing), each an
independent mutation and diagnostic point. Predicates are ASTs
(Compare/And/Or/Not) — serializable, hashable, node-mutable; lambdas
are forbidden. Example: `LegSpec(name=..., entry=EntrySpec(
Compare("mom", ">", 2.0)), exit=ExitSpec(stop=StopSpec("atr", 2.0)),
features_used=("mom", "atr"), rationale="...")`.

**Book** — the imperative shell: owns positions/cooldown/equity,
delegates stop/target/time resolution to the PUBLIC
`backtesting.simulate_intents` (SL-first + gap-through-stop by
construction), handles signal/forced exits itself. Example: a Book
trade equals the simulate_intents result on the same intent — pinned
equality against the public API.

**TradeRecord** — one closed trade as the SINGLE analysis input:
exit_reason ("stop"/"target"/"time"/"signal"/"forced" — 90% stops
means stop-hunting, not signal) plus features_at_entry/features_at_exit
snapshots (as-of at the entry decision bar / the exit bar). Example:
`records_to_features(run_strategy(...)["records"]["leg"])` feeds
bucket_scan / feature_importance directly.

**StrategySpec** — `tuple[LegSpec, ...] + PortfolioPolicy`: one
object shape for momentum, pairs and multi-leg options. MVP
combination is "independent" — legs' positions COEXIST (NOT netting);
each leg runs in a fresh Book (isolation by construction). Example:
an iron condor is four option_legs + caps; today it is four field
legs — the homogeneous shape is already the contract.


## Composition + carry (v14)

**walk_portfolio** — the multi-leg shared walk: one decision grid,
per-leg cooldowns, a portfolio slot cap, and the ORDERING PIN (exits
→ slot-release → netting → entries; a slot freed on bar b admits on
bar b).  Example: two legs, cap 1, both fire bar 0 → priority mode
admits the higher priority; sequential admits the first declared.

**Netting** — an opposite-direction entry on the same instrument
closes ALL positions on it (exit_reason "netted") and is consumed;
`netted_remainder = max(|new| − Σold, 0)` is RECORDED (clamped: a
reduction is not a loss; > 0 warns — signal loss, honestly
reported).  Partial netting is deferred.  Example: long 1R + short
2R → long closes "netted", remainder 1.0R flagged and warned.

**Priority** — an ordering weight for the shared walk: same-bar
entries compete by (−priority, declaration index); a freed slot
admits by the same rule; displacement (closing a lower-priority
holder) does NOT exist.  Example: priorities 9 and 1 on cap 1 →
the 9-leg enters; the 1-leg waits; neither closes the other.

**Borrow/funding (carry)** — flat annual bps, REQUIRED parameters
(zero warns — no silent short-cost understatement), accrued per
CLOSED hold bar (k = exit_bar − entry_bar; same-bar → 0) on a
CALENDAR-derived periods (XNYS daily = 252, not 365).  Notional is
pinned: size_r × risk_ref × fill/(fill − stop) long, fill/(stop −
fill) short.  Funding is sign-carried: positive → longs pay.
Example: a holding short with borrow 2000/funding −100 pays BOTH,
recorded separately in diagnostics (borrow_r/funding_r/taker_r).
