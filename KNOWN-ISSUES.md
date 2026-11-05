# Known issues

Honest inventory of everything the test suite does not prove, and why.
None of these are silent: each is pinned in the suite (skips are documented
in code, xfails are `strict=True`, so an unexpected change breaks the build)
and each names what would resolve it. Last verified: 2309 passed /
16 skipped / 2 xfailed.

## 1. Sixteen skipped tests — three documented groups

### 1a. Four skips — chaos-matrix deduplication (rule side)

Status: skipped (documented, by design).

Cause: the chaos matrices in `tests/test_chaos.py` skip 4 corruption
combos (`overflow_1e308` for `block_bootstrap_ci`,
`stationary_bootstrap_ci`, `apply_proportional_costs`,
`cost_death_test`) because each is already covered by a dedicated
finding test with a stricter, pinned assertion. Running both would
duplicate coverage, not add it.

Unblocks: nothing — this is deliberate deduplication. If a combo's
dedicated test is ever removed, its matrix skip must go with it.

### 1b. Eleven skips — chaos-matrix deduplication (ML side)

Status: skipped (documented, by design).

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

Status: skipped (documented, with the structural reason in code).

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

## 2. xfail — fast-only synthetic edge dies on 2/10 sweep seeds

Status: `xfail(strict=True)`, `tests/test_pipeline_sweep.py`
(`test_genuine_edge_survives_on_every_sweep_seed`, `@slow`).

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

## 3. xfail — no honest synthetic construction for NO_OOS_EDGE

Status: `xfail(strict=True)`, `tests/test_ensemble_pipeline.py`
(`test_first_half_edge_dies_as_no_oos_edge`, `@slow`).

Cause: `NO_OOS_EDGE` is the one verdict that cannot be built with honest
synthetic series data in the current tree. Structural reason: stage 3
(the robustness battery, which attacks the lagged long/flat rule —
harder) is stricter than stage 4 (walk-forward, which tests the
contemporaneous sign rule — easier), so any world that clears stage 3
still shows contemporaneous out-of-sample power; concentrated-edge
worlds die earlier as `ARTIFACT` or `DEAD` instead. Kill-gate thresholds
were deliberately not touched to force the verdict.

Unblocks: a genuinely honest series world whose walk-forward dies while
everything before it passes — then promote it into the demo matrix. The
cross-sectional `CS_NO_EDGE` path already covers the "no OOS edge"
family on panels.

## 4. ML engine known limitations (documented, not defects)

Status: factual properties of the ML stack, each documented in the
factory or config it belongs to.

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

The ML engine runs four evaluation gates (incremental, cost_line,
importance_stability, deflated_sharpe). These house-wide integrations
are scoped and scheduled, NOT forgotten:

- **ML robustness gate** — `robustness_battery` (threshold/window
  sensitivity) is not applied to model predictions yet; an ML edge
  fragile to its own threshold passes all four gates. Resolution:
  `_gate_robustness` over OOS probabilities in `_evaluate_models`
  (~1 day).
- **ML forward gate** — a `SURVIVES` verdict is still in-sample + OOF;
  no 52-week live-track check exists for ML PnL. Resolution: route ML
  strategy returns through `track_record.live_verdict` (~2 days).
- **ML prereg enforcement** — `ml_training` accepts a run without a
  prereg hash; the rule-based chain documents prereg-before-pipeline
  but ML does not. Resolution: `RuntimeWarning` when
  `config["prereg_hash"]` is absent (~2 hours).
- **ML calibration @slow** — DSR p-values under pure null are not
  KS-tested for the ML stream (rule-side permutation FPR is
  calibrated). Resolution: `test_ml_calibration.py`, 100 seeded noise
  worlds (~2 hours).

- **Mutation testing** — waiting for external users before the suite
  grows another layer.
- **Two-level performance budgets** — per-module and whole-suite
  budgets instead of the current single level.
- **Real-data golden tests** — need a frozen data snapshot; the current
  golden tests are synthetic-only.
- **Cross-version matrix CI** — conflicts with the single pinned
  `uv.lock`; needs a separate resolution strategy first.
