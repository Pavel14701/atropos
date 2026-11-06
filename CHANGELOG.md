# Changelog

## 1.1.0 — ML engine, backtest pipeline and portfolio waves

- Walk-forward ML training engine (`atropos.ml_training`):
  `run_ml_kill_protocol` evaluates a caller-built dataset (feature
  frame + 0/1 target + per-event executable returns + int64
  microsecond timestamps) through an expanding-block walk-forward
  with purge + embargo, fits any model from the open `MODEL_REGISTRY`
  (gbm, logit, rf, extratrees, hist_gbm, mlp, knn, qda) through the
  identical path, and gates it with configurable checks — incremental
  EV vs a linear battery, cost line, importance stability, deflated
  Sharpe. Verdicts: `SURVIVES`, `DEAD(<gate>)`, `UNDERPOWERED`
  (undecided — reported, not a kill).
- Incremental-gate FPR recalibration (`atropos.ml_training`): the
  gate's row-bootstrap tested H0 against fixed thresholds — the
  winner's curse of picking the argmax-EV threshold was not
  reproduced by row resampling, and seeded noise worlds died 43-46%
  of the time at a nominal 5%. The symmetric full-procedure
  bootstrap (both arms re-select thresholds per draw) centers the
  null at zero: measured FPR 0/100 at `bootstrap_b` of 100 and 500,
  pinned by a calibration test; the SPY TSMOM real-data verdict moves
  from `DEAD(importance_stability)` to `DEAD(incremental)`.
- Documented-kill demo case (`atropos.ml_training`): the module demo
  gains a fifth, asserted case — a statistics-calibrated reproduction
  of a documented tombstone (VRP-panic conditional short straddle,
  closed donor corpus): a synthetic world calibrated to the
  tombstone's marginal statistics (per-trade EV, negative conditional
  lift, panic-window VRP proxy) must come back `DEAD(...)` with both
  documented cause-of-death gates triggered (cost line +
  incremental); pinned by `test_vrp_panic_tombstone_reproduction`.
- Ensemble composite (`atropos.ml_ensemble`): `run_ensemble_report`
  combines registry members as `mean` / `weighted` / `rank_mean` /
  `stacking` over a past-only out-of-fold score matrix ("no stacking
  without nesting"); importing the module registers `"ensemble"` in
  `MODEL_REGISTRY` with no engine changes — nine entries
  (8 base + ensemble).
- Backtest pipeline wave (`atropos.backtesting`, `atropos.ranking`,
  `atropos.backtest_metrics`): the reference survivor-to-book replay —
  `simulate_signal` (net-of-taker-cost R with stop-first pessimism) →
  `run_position_state_machine` → `rank_within_groups` (past-only
  oracle, optional lightgbm lambdarank behind the `[ranking]` extra)
  → `run_admission_policy` (FCFS vs REPLACE under a slot cap) →
  per-trade metrics; `compare_rankers_on_intersection` makes the
  cross-ranker intersection comparison the default path and
  `CandidateGroup.embargo_until` carries the fold embargo.
- Portfolio wave (`atropos.portfolio`, `atropos.portfolio_protocol`):
  multi-slot replay with correlation-aware admission
  (`run_portfolio`) and portfolio-level kill gates
  (`portfolio_gates`: deflated Sharpe across trials, drawdown floor,
  same-side concentration) on the daily EV stream, plus the portfolio
  kill-protocol prereg and its calibration progression
  (`docs/calibration.md`).
- Unified kill chain (`atropos.kill_chain.run_full_kill_chain`): one
  call runs the rule pipeline and — with an `ml_dataset` — the ML
  engine, returning `{"rule", "ml", "unified"}`: KILLED if either
  chain kills, SURVIVES only if both survive, INCONCLUSIVE when the
  data cannot decide (e.g. ML `UNDERPOWERED`).
- RSI(2) negative control on real data (`examples/real_data/`): the
  committed SPY snapshot pushed through the ML engine kills the
  mean-reversion strategy (`DEAD(importance_stability)`) — proof the
  library kills known-weak strategies on real data; the prereg
  predicted `cost_line`, the verdict is right with a documented
  mechanism mismatch (2bps round-trip is too low to kill a 5-bar
  hold), recorded as "right verdict, wrong mechanism" without tuning.
- Runtime workspace generation: `atropos init` materialises
  `GRAVEYARD.md`, `hypotheses/INDEX.csv` and `hypotheses/README.md`
  idempotently — existing files are never overwritten.
- DSR and purge unification: one computational home for the
  Bailey & Lopez de Prado deflated Sharpe formula
  (`atropos.evaluation_metrics.deflated_sharpe_from_moments`; the
  returns-based and ML-based versions are wrappers) and one
  time-based purge/embargo home (`atropos._utils.purge_embargo_mask`,
  the row-based `purged_kfold_cv` keeps its own row-distance house).
- Preregistration bound to the run (ATR-07): `generate_prereg`
  computes the spec hash itself — sha256 over the canonical JSON of
  the frozen fields (first 8 hex chars, new `spec_hash` helper) — so
  the caller can no longer forget, mistype, or pre-freeze a hash that
  does not match the spec; `hash` is no longer a required field and a
  stale `hash` entry is ignored. `run_full_pipeline` accepts a
  keyword-only `prereg_hash` and echoes it as `report["prereg_hash"]`,
  anchoring a verdict to the prereg it was preregistered under.
- Specification-curve trust gates (ATR-10): stage 2 of the pipeline
  now requires at least 5 testable specs before it trusts the lenient
  `SURVIVES` call — fewer specs downgrade the stage to INCONCLUSIVE
  (WARN), never a silent pass; `curve_summary` reports
  `bonferroni_alpha` (`0.05 / n_specs`, watch-item diagnostic for the
  max-over-specs multiple search; the `|t| >= 2` verdict gate itself
  is unchanged).
- Walk-forward doctrine sync (ATR-09): FRAMEWORK.md stage 6 now
  prescribes EXPANDING windows (with the falsification rationale —
  more accumulated history means more held-out evidence and a sharper
  verdict; rolling would add a tunable window-length hyperparameter),
  superseding the obsolete "expanding is forbidden" wording; the
  `walkforward` docstrings state the same reason explicitly.
- Documentation translated to English: FRAMEWORK.md, the methodology
  trio, the contributor context, the how-to guides, notebooks and the
  agent context (AGENTS.md workflow + all ten skills) are fully
  translated with zero Cyrillic; API names, verdict tokens and
  §-anchors unchanged.
- Positioning (POSITIONING.md): the honest one-paragraph pitch —
  Atropos replaces the scripts researchers write around vectorbt, not
  vectorbt itself; the pipeline IS a backtester, a falsification-
  first one rather than a general-purpose engine, and the deliberate
  omissions are listed as scope; the three adoption conditions that
  would earn a stronger claim are recorded explicitly — none
  achieved by code.
- Docs drift (ATR-13): README reflects reality (Python 3.12,
  scikit-learn as a core dependency, lightgbm optional, nine-entry
  `MODEL_REGISTRY` = 8 base + `ensemble`, "most modules ship runnable
  demos"); FRAMEWORK.md gains an implementation map from the canonical
  7-stage protocol to the pipeline's actual 7 stages.
- Packaging (ATR-14): `license` (MIT file), `authors`, trove
  `classifiers` and `project.urls` added; `requires-python` raised to
  `>= 3.12` (CI and `.python-version` were already 3.12-only); the
  package ships an empty PEP 561 `py.typed` marker.
- Portability test: every shipped text artifact is scanned for
  machine-specific traces — personal user names, absolute Windows
  user paths, references to the private donor corpus the case-study
  numbers were taken from; any new occurrence fails the suite.
- Test suite: 2668 passed / 16 skipped / 3 xfailed — golden,
  adversarial, chaos, ML-contract, determinism, meta-AST and
  own-doctrine layers (see KNOWN-ISSUES.md for every skip and xfail).

## 1.0.0 — first public release

- 7-stage kill chain in one call: `run_full_pipeline` chains null
  controls, specification curve, robustness battery, walk-forward,
  Monte Carlo risk, predictive evaluation, and cross-sectional tests,
  and returns one of nine verdicts (SURVIVES … CS_NO_EDGE).
- Preregistration workflow: `generate_prereg` validates a params dict
  and freezes it into markdown before any data is touched;
  `atropos.track_record.live_verdict` judges live PnL against the
  prereg in production.
- Test suite: 890 passed / 4 skipped / 2 xfailed — golden, adversarial,
  chaos, calibration, determinism, meta-AST and own-doctrine layers
  (see KNOWN-ISSUES.md for every skip and xfail).
- Agent-context CLI: `atropos agent-setup` / `agent-context` /
  `agent-doctor` install and verify the AGENTS.md + skills bundle for
  AI coding agents.
- License: MIT.
