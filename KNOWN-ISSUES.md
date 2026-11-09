# Known issues

Honest inventory of everything the test suite does not prove, and why.
None of these are silent: each is pinned in the suite (skips are documented
in code, xfails are `strict=True`, so an unexpected change breaks the build)
and each names what would resolve it.

Structure: **behavioral facts** (§4) + **by design, not closeable**
(§1a-d, §2, §7) + **backlog with triggers** (§3, §5a, §5b, §6, §8,
§10) + **migration note, permanent** (§9).
Known issues in the strict sense — an open code bug — none; everything
here is a documented property, a deliberate choice, or a tracked gap
with its release condition. Last verified: 2812 passed /
18 skipped / 2 xfailed.

## 1. Eighteen skipped tests — four documented groups

### 1a. Four skips — chaos-matrix deduplication (rule side)

Status: skipped (documented, by design — not closeable).

Cause: the chaos matrices in `tests/test_chaos.py` skip 4 corruption
combos (`overflow_1e308` for `block_bootstrap_ci`,
`stationary_bootstrap_ci`, `apply_proportional_costs`,
`cost_death_test`) because each is already covered by a dedicated
finding test with a stricter, pinned assertion. Running both would
duplicate coverage, not add it.

Unblocks: nothing — this is deliberate deduplication. If a combo's
dedicated test is ever removed, its matrix skip must go with it.

### 1b. Eleven skips — chaos-matrix deduplication (ML side)

Status: skipped (documented, by design — not closeable).

Cause: the same deduplication pattern on the walk-forward primitives.
`tests/test_chaos.py` (`_ML_SKIP`) skips 11 corrupted-input combos of
`rolling_z` (subnormal 1e-320, overflow 1e308), `decluster_count`
(all-inf, alternating-inf, overflow stamps), `fold_edges`
(alternating-inf stamps), `purge_embargo_mask` (all-inf, all-NaN,
alternating-inf, overflow stamps) and `stream_stats` (overflow 1e308).
Each combo is covered by a dedicated `test_finding_*` test that pins
the primitive's exact current behaviour on that corruption (the
ml-chaos-1..7 findings: fabricated ±inf / -0.0 z-scores under
subnormal or overflow input, OverflowError on infinite stamps,
refusal vs garbage masks on non-finite stamps, std=inf leakage,
opaque IndexError on size-mismatched `ev_of_taken` /
`pick_threshold` inputs).

Unblocks: nothing — deliberate deduplication, same contract as 1a.

### 1c. One skip — knn cannot beat the linear comparator structurally

Status: skipped (documented, with the structural reason in code —
by design, not closeable).

Cause: `tests/test_ml_training.py`
(`test_registry_model_beats_logit_on_planted_nonlinearity[knn]`).
Distance-weighted kNN probabilities are compressed toward the base
rate, so its EV-argmax threshold saturates at the grid floor and its
broad book only ties the logit tail book on the mixed
linear+interaction world — measured within ±1 bp of the comparator
across seeds, row counts and neighbourhood sizes 25/50/75. Beating a
linear cherry-picker from a compressed score is a structural limit,
not a defect to fix or a check to weaken.

Unblocks: nothing known — this is the model class's honest shape.
The other three contract families (determinism, probability
validity, full engine path) still cover knn end to end.

### 1d. Two skips — lambdarank oracle without the optional extra

Status: skipped (environment-dependent, by design — not closeable;
closes itself when the optional extra is installed).

Cause: `tests/test_ranking.py` gates the two lightgbm-backed
lambdarank tests (`rank_with_lambdarank` determinism/validity and
`compare_rankers_on_intersection` score-key discovery) behind
`pytest.importorskip("lightgbm")`. `lightgbm` is the OPTIONAL
`[ranking]` extra — `atropos.ranking` imports it lazily and every
other path runs without it — so on an environment without the extra
the two tests skip instead of failing. In CI (the ranking-extra job)
lightgbm is installed and both run.

Unblocks: `uv sync --extra ranking` (or any environment with
`lightgbm>=4.7` installed) — then the group disappears and the suite
is back to sixteen skips.

## 2. xfail — fast-only synthetic edge dies on 2/10 sweep seeds

Status: `xfail(strict=True)`, `tests/test_pipeline_sweep.py`
(`test_genuine_edge_survives_on_every_sweep_seed`, `@slow`) — by
design, not closeable: the xfail IS the honest record.

Cause: the FAST edge class (raw AR(0.5) driver) is killed by the
robustness battery on sweep seeds 2 and 7. The battery is RIGHT: the
driver's memory (~2 periods) is shorter than the battery's smoothing
windows (5/10/15), so window-smoothing monotonically inverts the
signal-to-return alignment (a smoothed high reads as "the state was
high over the whole window" — a mean-reversion anti-signal). The class
is genuinely not smoothing-robust; softening the window-sensitivity rule
was rejected as p-hacking the killer.

Unblocks: already compensated — the SLOW edge class
(`kind="slow_genuine_edge"`), whose information window overlaps the
smoothing instead of washing past it, passes the battery on all 10 swept
seeds (`test_edge_class_contract_fast_vs_slow`). The xfail stays as the
honest record that the fast class is battery-mortal.

## 3. Tracked gap — no honest world where the walk-forward itself kills (stage 4)

Status: tracked limitation, **no xfail** — there is no test vehicle to
mark. The stage-6 `NO_OOS_EDGE` path IS pinned by a live, unmarked
test: `tests/test_ensemble_pipeline.py`
(`test_first_half_edge_dies_as_no_oos_edge`) builds a first-half-edge
world that clears stages 1-4 (walk-forward PASS, `wf_shrink_ratio`
0.246, `verdict_reason is None` — measured and asserted in-test) and
dies at stage 6 with `NO_OOS_EDGE`. (A prior revision of this entry
claimed that test was `xfail(strict=True)`; the marker no longer
exists — the claim was stale, masked by the xfail total of 2 coming
from §2 and §5a. Rewritten 2026-11-07.)

Cause: the remaining gap is narrower than the old wording — no honest
synthetic world has been constructed where the WALK-FORWARD stage
itself kills (a stage-4 `FAIL`/`OVERFIT_DEATH` while stages 1-3 pass).
Structural reason: stage 3 (the robustness battery, which attacks the
lagged long/flat rule — harder) is stricter than stage 4
(walk-forward, which tests the contemporaneous sign rule — easier),
so worlds that clear stage 3 still show walk-forward power; worlds
that fail stage 3 never reach stage 4. The shrink gate's OVERFIT
mapping is pinned at stage 4 via the canned-monkeypatch pattern
(see `tests/test_ensemble_pipeline.py::test_overfit_shrinkage_kills_as_dead_at_stage_four`).

Trigger to close: a genuinely honest series world whose walk-forward
dies while everything before it passes — then pin it with a regular
(not xfail) test and promote it into the demo matrix. The
cross-sectional `CS_NO_EDGE` path already covers the "no OOS edge"
family on panels.

## 4. ML engine behavioral facts (not issues)

Status: factual properties, each pinned in the factory or config it
belongs to — documented behavior, not defects.

- **knn under the beats-logit gate** — see 1c: compressed scores
  saturate the EV-argmax threshold; the engine still reports knn's
  own gates honestly in `report["models"]["knn"]`.
- **mlp `max_iter=500` budget** — the default converges without
  sklearn's `ConvergenceWarning` on engine-scale datasets (a few
  hundred rows or more); on tiny or XOR-like test samples adam may
  stop at the budget. That outcome is deterministic, observable via
  `loss_curve_`, and only noisy, not wrong (the factory docstring
  pins this; the default was raised from sklearn's 200 precisely to
  stop the warning at engine scale).
- **hist_gbm `early_stopping=False`, explicitly** — sklearn would
  auto-enable a seeded validation split beyond 10k rows; the explicit
  off keeps the iteration schedule identical at every dataset size
  (auto mode would need a dedicated validation split and a
  size-dependent schedule).
- **Default `dsr_min=0.95`** — the deflated-Sharpe gate clears at the
  0.95 quantile, the multiple-testing-safe equivalent of one-sided
  p < 0.05. Raise it in `config["checks"]` if you want a stricter
  bar; it is a starting point, not a frozen constant.

## 5. Follow-ups (not blocking, tracked)

Status: planned.

### 5a. ML integration gaps (review-tier P3, deliberately deferred)

The ML engine runs four default evaluation gates (incremental,
cost_line, importance_stability, deflated_sharpe) plus an opt-in fifth
(`checks.robustness`, default off: threshold/window sensitivity over
OOS probabilities). These house-wide integrations are scoped and
scheduled, NOT forgotten:

- **ML robustness gate is opt-in, not default** — `checks.robustness`
  (implemented: threshold-shift sensitivity over OOS probabilities,
  `robustness_shift` ±0.20 by default, INCONCLUSIVE below the event
  floor) stays off by default so the four-gate contract is unchanged;
  an ML edge fragile to its own threshold passes unless the caller
  enables the check. Trigger to close: a calibration pass on real
  data flips the default to `True`.
- **ML forward gate** — a `SURVIVES` verdict is still in-sample + OOF;
  no 52-week live-track check exists for ML PnL. Trigger to close:
  ML strategy returns routed through `track_record.live_verdict`
  (~2 days, scoped).
- **ML prereg enforcement (opt-in shipped)** — CLOSED at the opt-in
  level: `strict_prereg_hash=True` + absent hash raises `ValueError`
  before any model fits; the default stays `False` (RuntimeWarning +
  completed run) for backward compat — pinned by
  `tests/test_ml_training.py::test_strict_prereg_hash_config_gate`.
  Trigger to close (fully): the default flips to `True` once the
  workflow tooling always emits `spec_hash`.
- **ML calibration @slow** — DSR p-values under pure null are not
  uniform for the ML stream: the KS contract test exists and is pinned
  as a strict xfail (`tests/test_calibration.py::
  test_ml_engine_deflated_sharpe_pvalues_ks_uniform_under_null`, 98
  finite null-world p-values, KS p ~= 0.001, rejection rate ~0.08 —
  under the 10% bound but skewed low). Trigger to close: fix the DSR
  standardization on short zero-padded daily streams; the strict xfail
  then XPASSes and forces itself out of the suite.

- **Mutation testing** — waiting for external users before the suite
  grows another layer. Trigger to close: first external adoption.
- **Two-level performance budgets** — per-module and whole-suite
  budgets instead of the current single level. Trigger to close:
  suite growth makes the single-level budgets flaky.
- **Real-data golden tests** — the current golden tests are
  synthetic-only. Trigger to close: a frozen CSV snapshot +
  version-pinned pandas/numpy (or an explicit tolerance
  specification) + golden metrics pinned.
- **Cross-version matrix CI** — conflicts with the single pinned
  `uv.lock`. Trigger to close: a separate lockfile resolution
  strategy.

### 5b. Backtest-layer watch-items (tracked)

Status: tracked, not blocking; the safe usage rule is already
documented where the hazard lives.

- **`n_skipped` cross-ranker comparability** — RESOLVED:
  `atropos.ranking.compare_rankers_on_intersection` computes both
  rankers' metrics on the INTERSECTION of `scored_group_ids` and runs
  the paired bootstrap on it, so unequal skip sets can no longer fake
  a metric difference; the raw-report idiom stays documented in
  `docs/how-to/07-choose-a-ranker.md` for callers who need the full
  lists.
- **`embargo_until` enforcement in `paired_bootstrap_diff`** — CLOSED
  (louder than the old watch wording asked for): `enforce_embargo=True`
  with incomplete metadata (`current_fold` / `group_ids` /
  `embargo_until_by_group`) raises a loud `ValueError`
  (`atropos/ranking.py`, "needs current_fold, group_ids and
  embargo_until_by_group") — pinned at
  `tests/test_ranking.py::test_paired_bootstrap_*` (the
  `match="needs current_fold"` raise pin). The default-off path keeps
  the caller-owned contract unchanged (metadata without the flag is a
  dead configuration and also raises). No warning layer was added on
  top of the existing raise — a warning would have been a downgrade.
- **`subperiod_stability` gates on the concentration cap; the
  permutation p-value is a reported diagnostic, not a gate** — the
  verdict requires sign consistency across chunks AND `max_share`
  strictly below the (auto-scaled) cap; the permutation probability
  that a random shuffle shows at least this concentration is computed
  and returned but does not FAIL the control. Deliberate statistic
  choice, not an oversight — but a reviewer may want it reversed.
  Trigger to close: a preregistered decision to gate on the
  permutation p (with the same strict-α boundary pins as the other
  permutation controls) or an explicit docstring rationale that the
  cap is the preregistered gate and p is diagnostic-only.

## 6. Walk-forward label overlap is an accepted (uncalibrated) risk

Status: accepted risk behind a hard guard; pinned by tests
(`tests/test_walkforward_framework.py::
test_label_horizon_above_one_is_rejected`,
`tests/test_ensemble_pipeline.py::test_label_horizon_config_error_propagates`).

Cause: the `wf_t` statistic of `atropos.walkforward_framework`
assumes NON-overlapping forward-return labels inside the test union
(`label_horizon=1`). When the true label horizon is > 1, adjacent
labels share return rows, the effective sample size is smaller than
`n_pnl_unique`, and `wf_t` (and the shrink ratio with it) is
anti-conservative — the bias is uncalibrated. The fold-overlap
deduplication (union counting: one prediction per row) does NOT
address this — it dedups rows across overlapping folds, not label
overlap within a row's forward window. There is NO compensation
anywhere else in the chain: the block-bootstrap checks of stages 5
and 7 are independent additional attack surfaces, not a replacement
for overlap correction. The hard guard: `walkforward(label_horizon > 1)`
raises `ValueError` (a config error that propagates through the
pipeline — never classified INCONCLUSIVE). Separately, a NaN shrink
ratio (degenerate baseline t) reads INCONCLUSIVE with an explicit
reason, never a silent pass; if it occurs on valid data more often
than expected — investigate as a bug, not an accepted risk.

Trigger to close: a P2 wave implementing the Newey-West correction
(lags = horizon − 1) of `wf_t`, calibrated on the null at horizon in
{1, 5, 20}, with horizon=1 bit-for-bit identical to the current
statistic — then the guard is lifted.

**Also in the walk-forward layer — `check_no_peeking` is a tamper
detector, not a lookahead proof.** For a result the module produced,
the check re-verifies bookkeeping the loop itself wrote (it reduces
to `t0 <= t0`); it fires only when a caller edits the fold frame
(that direction IS pinned: `test_check_no_peeking_rejects_tampered_folds`).
Structural anti-peeking is enforced by construction — `fit_fn` never
receives test rows (`test_predict_fn_gets_test_frame_without_outcome`)
and the no-lookahead suite pins fold-0 invariance; the docstring now
states this scope honestly. Trigger to close: an independent
lookahead test that re-derives the train/test boundaries from the
raw data instead of the module's own folds frame.

## 7. `cost_death_test` is a standalone API, not a pipeline stage

Status: deliberate design, not closeable — not wired into
`atropos.ensemble_pipeline.run_full_pipeline` (pinned by tests in
`tests/test_cost_model.py`, exported from `atropos.cost_model`, demoed
via `run_all_examples`).

Cause: `cost_death_test` is a self-contained contract test (block
bootstrap of net P&L against a cost hurdle) for callers who want the
cost-death question answered standalone. The pipeline does NOT call
it: stage 5 carries its own net/gross bootstrap attribution with the
breakeven gate (`cost_bps >= breakeven_cost_bps` attributes a
net-envelope death to costs as COST_DEATH), and the ML chain has its
own cost gate (`_gate_cost_line` in `atropos.ml_training`). Three
cost-death surfaces, three implementations — none is a wrapper of
another; only the constants differ (docs law 3: 20bp round trip,
engine taker default: 25bp round trip, stage-5 default: 10bp one-way).

Unblocks: nothing — the duplication is intentional (standalone API vs
pipeline attribution vs ML gate). If the three ever need to share a
hurdle constant, centralize it in `atropos.cost_model` and re-export;
do not wire `cost_death_test` into the pipeline as a stage.

## 8. Three 7-stage protocols exist (docs vs code drift)

Status: documented drift, ANNOTATED — the CODE is canonical; all 12
agent skills now carry the header "Code canonical. Map:
FRAMEWORK.md:69-81. Deviations: KNOWN-ISSUES §8"; a full docs sync is
a future P2 wave.

Cause: three seven-stage protocols are in circulation: (a) the
researcher protocol in FRAMEWORK.md §2; (b) the shipped code stages
of `atropos.ensemble_pipeline.run_full_pipeline`; (c) the agent
skills 11-16. Their stage numbering does not line up (e.g.
walk-forward is code stage 4, FRAMEWORK.md researcher stage 6, and
skills 16 describes it as "stages 6-7 of 7"). The CODE is canonical;
the map between the researcher protocol and the code is
FRAMEWORK.md:69-81 ("Implementation map"). FRAMEWORK.md:63
("skipping any stage = protocol violation") is binding for the
researcher protocol; the obsolete expanding-window ban inside the
skills is historical — current doctrine is the code plus
FRAMEWORK.md:221-233 ("Why expanding, not rolling").

Trigger to close: a P2 wave that syncs skill and FRAMEWORK.md stage
numbering/wording to the code, plus a contract test pinning the
FRAMEWORK.md:69-81 map.

## 9. Verdict semantics changed (2026-11-07) — reading pre-taxonomy tombstones

Status: documented migration note, permanent — not closeable (old
tombstones are append-only and are never rewritten).

Cause: runs before 21a1ccb used the generic `DEAD` verdict for two
different deaths — stage-3 robustness-battery kills and stage-4
walk-forward OVERFIT deaths. The verdict taxonomy (CHANGELOG 1.2.1)
replaced the generic `DEAD` with specific tokens: `ROBUSTNESS_DEATH` /
`PLACEBO_VIOLATION` at stage 3, `OVERFIT_DEATH` at stage 4; `DEAD` now
means only proven absence of edge. Disambiguate old tombstones by
`killed_by_stage`: 3 = robustness battery (today `ROBUSTNESS_DEATH` or
`PLACEBO_VIOLATION`), 4 + an OVERFIT marker in the kill reason =
`OVERFIT_DEATH`. The graveyard cod-tag "DEAD" stays as the umbrella
term — the workspace template is unchanged; only the pipeline's token
set is new. Additionally, `CS_NO_EDGE` maps to the DEAD verdict class
only when the config target is `"xs"`; the default target `"ts"`
treats the XS stage as a sanity check and reports NEEDS_ANALYSIS —
backward-incompatible for callers reading `verdict_class`, documented
in CHANGELOG 1.2.1. Sweep-history note: the pre-fix classifier read
the pure-noise seed-7 world as DEAD/class DEAD (a v7 pre-decision
that assumed a no-active-days baseline — incorrect: that world's
baseline mean is positive-but-insignificant), so the seed-7 noise pin
now reads ROBUSTNESS_DEATH / NEEDS_ANALYSIS per the corrected
classifier; the same split applies to any battery OVERALL DEAD with a
positive baseline.

Unblocks: nothing — this section is the standing reading guide for
pre-taxonomy tombstones, not a defect.

## 10. Thirds-window instability — unresolved

Status: unresolved, not investigated.

Cause: the external A:agg feed showed a thirds sign-flip between a
532-row window and a 2407-row window of the same strategy: the
subperiod-thirds battery test (sign + CI alive in all three thirds)
flipped its conclusion depending only on the window chosen for the
same underlying strategy. Potential fourth Atropos defect: the thirds
test is sensitive to window choice, so a single-window thirds verdict
may not be stable.

Trigger to close: an independent thirds-test study across window
sizes (does the sign-flip rate collapse to the null under window
resampling?); on results, either pin a window-selection rule or
document the sensitivity as an accepted property.

## 11. Orthogonal-check tool (analytic, not part of the 7-stage protocol)

Status: shipped (v11). Not a protocol stage.

Cause: `atropos/orthogonal.py` — `orthogonal_check(outcome, features,
controls)` + `signal_vs_controls(outcome, signal, controls)`. Used
ad-hoc to decompose whether a candidate feature carries information
beyond known controls. Not invoked by `run_full_pipeline`; callers
use it directly on a feed.

Used in: C''-trade (zone_width_atr t=−9.33, retest_delay t=−9.92
after mom7/mom30 controls; direction t=−1.72 not significant),
E-predictor (funding level z t=+4.89 beyond persistence),
A-daily (signal_vs_controls: t ≈ stage-1 value, consistency check).

Unblocks: nothing (analytic tool, not a gap).
Trigger to close: N/A — permanent tool. Add to §8 protocol sync only
if integrated as a stage (currently not planned).

## 12. Feed granularity must match strategy decision granularity

Status: rule documented (skill 02, FRAMEWORK §4); C''-daily VOID as
example.

Cause: aggregating a sub-daily strategy's decisions to a daily P&L
stream is signal loss, not a time-frame test. The feed must operate
at the granularity the strategy actually decides at. C''-daily
(daily aggregate of a 15m OB-retest leg) was run and returned DEAD;
that DEAD is a verdict on the aggregate, not on the leg, and
C''-daily is now marked VOID in docs/JOURNAL.md as a design error.

Unblocks: prevents repeating the error on future legs.
Trigger to close: N/A — permanent rule. The VOID example stays as
the canonical illustration.

## 13. Feature-analysis tool (analytic, not part of the 7-stage protocol)

Status: shipped (2026-11-08). Not a protocol stage.

Cause: atropos/feature_scan.py — feature_analysis orchestrator (11
functions: missingness_scan, correlation_cluster, orthogonal_check
(§11), bucket_scan, interaction_scan, rule_fit, regime_scan,
threshold_scan, auto_discovery, feature_importance,
auto_rule_generation). Returns the extended classification
(STRONG / REGIME_DEPENDENT / STRONG_UNVERIFIED / SHAPE_UNKNOWN /
PROXY / NOISE / EXPLORATORY); the report schema was frozen at the
first checkpoint commit and every key is present on every outcome.
Used in: C''-trade (results in docs/JOURNAL.md, feed 2).
Explicitly out of scope: ML black-box SHAP (permutation importance
only); auto-discovery never produces STRONG (always a hypothesis
list under the exploratory doctrine note); auto_rule_generation
searches values on train only and lets the untouched test split
judge (overfit_warning fires when every kept candidate fails).
Verification: 28 test items; 25-mutation sweep — every mutation
applied -> pinned test failed -> module restored clean.
Unblocks: nothing (analytic tool, not a gap).
Trigger to close: N/A — permanent tool. Add to §8 protocol sync only
if integrated as a stage (currently not planned). If a future wave
wants auto-discovery -> prereg path, that is a separate protocol
change, not a tool feature.

## 14. Data contract + legs layer (v13) + composition/carry (v14) — surfaces, roadmap, deferred families

Status: v13 shipped (2026-11-08), wave v14 shipped (2026-11-08).

### The four diagnostic surfaces (the §14 ranking, decided BEFORE wave 3)

A missing surface = a class of strategies Atropos cannot issue a
verdict for.  Ranked by breadth (who is affected), not visibility:

| Surface | Unlocks | Affected |
|---|---|---|
| composition (netting/sequential/priority) | portfolio-level DSR/DD/concentration gates | EVERY multi-leg strategy (pairs, calendars, condors) |
| short-cost honesty (borrow/funding) | honest COST_DEATH for shorts | EVERY short strategy |
| execution realism (limit/stop + participation) | COST_DEATH where fills fail | every strategy with execution constraints |
| cross-sectional/universe | bucket_scan by ENTITY, not time | everyone without OHLCV/Tidy data |

### Roadmap (decided, not re-litigated)

* Wave 3 (DONE, this section): netting/sequential/priority +
  borrow/funding.  Shipped: Book.walk_portfolio (shared walk, the
  ORDERING PIN: exits -> slot-release -> netting -> entries; the
  tuple declaration order is the tiebreak; priorities not applied
  in netting; NO displacement), netting = full collapse +
  netted_remainder (clamped: a reduction is not a loss; > 0 warns),
  exit_reason "netted", carry accrual (calendar-derived periods —
  XNYS 252/1638, 24/7 365/8760 — NOT year/bar_delta, which
  understated equity short-carry by ~31%), discrete bar semantics
  k = exit_bar − entry_bar, REQUIRED carry params with loud
  zero-warnings, funding sign-carried (negative legal),
  portfolio_gates over the walk's equity/exposure, cost_model
  apply_holding_costs + breakeven_holding_cost_bps (periods
  REQUIRED).
* Wave 4: options — OptionsChainAdapter + option_leg +
  surface scalars + cost_regime on InstrumentRef.  NOW
  portfolio-diagnosable because netting exists.
* Wave 5: execution realism — limit/stop orders + participation
  caps (COST_DEATH where strategies fail to fill).
* Wave 6: cross-sectional/universe + fraction_of_equity/vol_target
  sizing (a different analysis mode: bucket_scan by entity).

### Deferred in wave 3 (each a loud ValueError naming this section)

* Displacement by priority — needs a stateful intra-bar mutation +
  PnL reconstruction + a conflict with netted semantics.  MVP
  priority resolves ordering and freed-slot competition only.
* Partial netting (opening the remainder) — mutable size, three
  new states.  The remainder is RECORDED (netted_remainder +
  diagnostics flag + portfolio warning), not silently dropped.
* Rate SERIES from data — needs per-instrument series in Book and
  a FieldSpec for funding/borrow with t_known_offset (perps:
  funding changes every 8h; hard-to-borrow equities).  MVP: flat
  annual bps = a cost POLICY; series = DATA; not mixed.
* cost_regime on InstrumentRef (equity/perp/future) — wave 4.
* gross/net cap ENFORCEMENT — reported as pinned-format notes.
* OptionsChainAdapter / EventAdapter / FuturesAdapter /
  WidePanelAdapter — waves 4-6.
* zscore transform; multi-entity materialization; run_full_pipeline
  signature integration — unchanged from v13.

Shipped in v13 (unchanged): the data contract, adapters, legs,
Book, StrategySpec; the v13 deferred list above migrates here.

Verification: v14 sweeps 22/22 CAUGHT rc=1 (8 carry/periods +
9 walk/netting + 5 dispatch/gates/holding); full suite green;
§7 law respected (cost_death_test stays standalone — portfolio
honesty comes from carry INSIDE the equity, not a new gate).

Unblocks: wave 4 (options).  Waves 5-6 unblocked transitively.
Trigger to close: N/A — permanent layer; each deferred line graduates
by its own wave and removes itself here.

## 15. Paper parity protocol (forward test, not a protocol stage)

Status: shipped (2026-11-09).  Separate entry point
(``atropos/paper_protocol.py``); NOT invoked by ``run_full_pipeline``.

Cause: ``run_paper_protocol(paper_log, frozen_outcomes, frozen_spec,
*, legs, ...) -> PaperVerdict``.  Six verdicts (PARITY_PASS,
PARITY_FAIL, OPERATIONAL_FAIL, STATISTICAL_FAIL, INSUFFICIENT_N,
SURVIVES_TO_LIVE), four classes (PASS / FAIL / NEEDS_ANALYSIS /
INCONCLUSIVE).  Validates that paper execution matches the frozen
spec per leg and at portfolio level: 10 parity issue types +
1 statistical issue type (``dd_divergence``); dd numbers are reported
on every verdict (the breach gate only rules inside the statistical
stage).

Supports every strategy class via frozen_spec extension (class-
specific sections are opt-in via frozen_spec; absence = skipped,
not inferred):

* ``exit_type_normalization`` (required) — maps class-specific exit
  names onto normalized tokens (time_exit, stop_exit, tp_exit,
  funding_payment, rebalance, early_close, exercise); a token
  outside the map is a ValueError, never a silent mismatch.
* ``funding`` section (optional) — carry legs: settlement-interval
  context + funding-rate alignment at tolerance.
* ``pnl_decomposition`` section (optional) — options legs: per-greek
  comparison (delta/gamma/theta/vega/fee) with a per-component
  tolerance; equal totals with opposite greek errors are caught.
* ``sizing_rule`` + per-trade ``size`` — rule-level AND trade-level
  sizing checks.
* ``bar_delta_min`` in ``paper_log.attrs`` + frozen_spec — the §12
  feed-granularity rule as a hard gate (1m paper vs 15m frozen is a
  fail, never a tolerance).
* ``dd_breach_ratio`` — drawdown-divergence gate independent of mean
  pnl (frame-order cumsum; ordering differences are the signal).

Out of scope: SPEC CORRECTNESS — the protocol validates that paper
matches frozen, not that frozen is right (frozen correctness goes
through ``run_full_pipeline`` on the backtest feed); capital
authorization (``sizing_note`` in the report — policy, not code).

Evidence semantics (read before quoting numbers):

* ``dd_comparison`` is computed on the FULL leg frames (paper n vs
  frozen n), not on matched pairs — it is POPULATION-DIRECTIONAL
  evidence, never a statistically established ratio.
* Parity issue counts (exit/fee/sizing/universe/...) are
  DETERMINISTIC per-trade facts — they do not depend on n.
* ``unmatched_entry`` may be a legitimate POPULATION DIFFERENCE
  (e.g. a rolling-head live detector vs a fixed-head replay), not a
  defect; read it together with the run's feed construction.
* ``SURVIVES_TO_LIVE`` requires execution-quality evidence: the
  paper log must carry ``side`` + ``entry_price``/``exit_price``
  (+ ``signal_ts`` for latency).  A paper log without execution
  columns caps at ``PARITY_PASS`` (parity-only attestation).
* ``PARITY_PASS``/``SURVIVES_TO_LIVE`` need ``n_matched >=
  min_n_trades``; below that the verdict is ``INSUFFICIENT_N`` and
  the fix-verification signal is a CLEAN parity report (issues
  empty, matched == paper trades), not the token.

Real-data coverage: validated on a tactical/time-exit leg (C'',
2026-11 DTE post-mortem).  Carry (``funding``) and options
(``pnl_decomposition``) sections are implemented and unit-tested;
real-data validation pending — future wave when a deployed system
runs carry/options paper legs.

DTE fee-model ledger (2026-11 fix wave): per-leg fee conventions are
VERIFIED only where the leg's frozen sim was read — C'' =
``split_5bp_5bp`` (5bp entry + 5bp exit = 10bp RT, ob_zone_sizing),
E = ``maker_half_rt`` (funding_carry_v2, maker distinct by design).
A/BAB/BAC remain ``taker_rt_at_entry`` named-current with UNAUDITED
conventions: any fee change outside C''/E requires reading that
leg's sim load_trades first.  Sizing is a CONTRACT token
(``composite_primary_neutral_fallback_until_n200``, §4 amendment
2026-11-09): parity never compares runtime mode
(``sizing_mode_active``), the ``sizing_activation_stale`` issue
catches fallback still active at ``n_ob_rows >= composite_activation_n``.

Unblocks: formal forward-parity testing for the paper/live phase of
any deployed system (first application: the 2026-11 DTE post-mortem).

Trigger to close: N/A — permanent protocol.

## 16. Performance metrics are informational — they never gate

Status: shipped (2026-11-09).  `evaluation_metrics.sortino_ratio` /
`payoff_stats` + `calmar_ratio(period_years=...)`; read-out blocks in
`run_full_pipeline` (`strategy_metrics`), `paper_protocol`
(`statistical_report` per-trade sharpe/sortino) and `track_record`
(`live_sortino`/`live_cagr`).

The law: **a metric never gates, never overrides, never appears in an
issue.**  Sharpe 2.3 on 71 active days is not an edge — it is an
insignificant mean with outliers; the kill-protocol answers "can this
be killed", not "how good is the return" (the A-daily case that
motivated the verdict taxonomy).

Reading discipline (each pinned by tests):

* `hit_rate` is strategy-class-dependent (mean-reversion can show 80%
  hit rate with a negative Sharpe); it ships ONLY inside
  `payoff_stats` next to `avg_win_r`/`avg_loss_r`/`win_loss_ratio` —
  42% at +3R/−1R is a different strategy than 42% at +1R/−1R.
  Breakeven trades are excluded from the hit_rate denominator
  (differs from `win_rate`'s full-n convention on zero-PnL trades;
  both single-home).
* `sortino` values are None below 30 observations (downside dev is
  noise under ~30 points) — in `paper_protocol` that means
  `n_matched < 30`.
* `strategy_metrics["n_obs"]` counts ACTIVE periods (finite non-zero
  returns — the standing active convention `active_days` uses);
  `period_years` is the CALENDAR span of the full window; CAGR/Calmar
  use the calendar span (71 active days over 502 calendar days is
  1.38 years, not 0.28).  Zeros are compounding-neutral, so drawdowns
  on the filtered series match the full stream.
* `sharpe_ci` in `strategy_metrics` is omitted below 50 active
  observations and is a minimum-cost noisy directional interval above
  — never a significance test.
* The pipeline block is per-period only: hit_rate / n_trades /
  profit_factor are trade-stream metrics and live in trade-stream
  contexts (`payoff_stats` / `profit_factor`).  `paper_protocol`
  metrics are per-trade (R-multiple) — `per_trade_sharpe` never
  annualizes; the multi-leg merge POOLS matched arrays across legs
  (R-normalized).

Trigger to close: N/A — permanent law.

## 17. Instrument routing is a prereg decision

Status: shipped (2026-11-10).  Machine-checkable mapping in
`atropos/instrument_map.py` (`PLAN_REGISTRY`, six families); pipeline
enforcement is opt-in, audit fallback otherwise.

Cause: a strategy family dictates WHICH instruments — battery tools
and pipeline stages — may judge it.  The family is declared in the
prereg/config as `strategy_family` (explicitly, before data; the
`run_full_pipeline` kwarg overrides `hypothesis_config["strategy_family"]`,
and both set and differing is a loud ValueError).  Declared, the run
is ENFORCE mode: before each stage executes, a plan forbidden entry
whose concrete `stage` matches that stage's number (1-7) raises
ValueError, and `report["instrument_audit"]["violations"]` repeats the
post-hoc `atropos.instrument_map.audit_report` findings for the stages
that did run.  Not declared, the run is AUDIT mode: nothing is
enforced, `instrument_audit` records `mode: "audit"` with
`family: None`.

The PLAN_REGISTRY forbidden table (six families — rule_based_tactical,
per_trade_tactical, ml_predictive, cross_sectional_factor, carry,
options; four carry forbidden cells, `ml_predictive` and `options`
none; authoritative machine-checkable source: `PLAN_REGISTRY` in
`atropos/instrument_map.py` — the rows below are a frozen snapshot):

| Family | Forbidden tool (stage) | Reason |
|---|---|---|
| rule_based_tactical | `ml_stage_4` (stage 4) | rule-based strategies have no predictive feature stream; ML on zero-heavy streams gives CONSTANT_PREDICTIONS by construction |
| rule_based_tactical | `daily_aggregate_for_sub_daily` (declaration-level) | feed granularity must match decision granularity (§12) |
| per_trade_tactical | `daily_aggregate_for_sub_daily` (declaration-level) | feed granularity must match decision granularity (§12) |
| cross_sectional_factor | `trade_battery` (declaration-level) | factor construction is not trade-level |
| carry | `book_momentum_signal` (declaration-level) | funding is settlement, not momentum |

Paper-protocol side: the frozen spec may declare `family` and
`tools_used` (the DECLARED measurement tools).  Neither declared — no
instrument audit; exactly one declared — audit mode (the declaration
is recorded, not validated, nothing fails — the honest skip); both
declared — enforce: the declared tools intersecting the family's
forbidden set is an operational `instrument_violation` (the 3rd
operational issue type, alongside latency_breach and
exit_direction_violation; a FAMILY-CONTRACT violation class, never a
parity issue) and the leg dies `OPERATIONAL_FAIL` before any stage
runs (`killed_by_stage=4` semantics).

**U9.** `tools_used` is a declaration, not a measurement; enforcement
validates the declaration against the family, not against actual
execution.  Actual execution verification requires engine-level
instrumentation (future, if needed).

Positioning note: Atropos is a falsification toolkit, not a compiler —
routing constrains what may JUDGE a family, it does not produce
strategies.  The compiler returns when a real-data paper run reaches
SURVIVES_TO_LIVE — trigger, not vibe.

Trigger to close: N/A — permanent law.

## 18. REPACKAGING is a routing verdict — known premia are the raw material

Status: shipped (2026-11-10); semantic amendment of the verdict's
next-step/what-would-change (verdict_class unchanged: NEEDS_ANALYSIS).

Content: the momentum/trend premium is the most robust documented
premium of the last century and dominates the crypto cross-section;
deployed quant books are overwhelmingly repackagings of it; a
REPACKAGING verdict therefore routes to the
packaging-vs-direct-replication comparison (same universe, same
bars, same cost model: net EV, turnover cost per unit of premium,
capacity, residual correlation, drawdown profile) instead of implying
dead-end analysis.  The guard: a repackaging that cannot beat the
direct implementation net of costs is COST_DEATH.

Trigger to close: N/A — permanent interpretation.
