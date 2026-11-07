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
with its release condition. Last verified: 2781 passed /
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
in CHANGELOG 1.2.1.

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
