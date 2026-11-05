# Changelog

## Unreleased

- Walk-forward ML training engine (`atropos.ml_training`):
  `run_ml_kill_protocol` evaluates a caller-built dataset (feature
  frame + 0/1 target + per-event executable returns + int64
  microsecond timestamps) through an expanding-block walk-forward
  with purge + embargo, fits any model from the open eight-entry
  `MODEL_REGISTRY` (gbm, logit, rf, extratrees, hist_gbm, mlp, knn,
  qda) through the identical path, and gates it with configurable
  checks — incremental EV vs a linear battery, cost line, importance
  stability, deflated Sharpe. Verdicts: `SURVIVES`, `DEAD(<gate>)`,
  `UNDERPOWERED` (undecided — reported, not a kill).
- Ensemble composite (`atropos.ml_ensemble`): `run_ensemble_report`
  combines registry members as `mean` / `weighted` / `rank_mean` /
  `stacking` over a past-only out-of-fold score matrix ("no stacking
  without nesting"); importing the module registers `"ensemble"` in
  `MODEL_REGISTRY` with no engine changes.
- Unified kill chain (`atropos.kill_chain.run_full_kill_chain`): one
  call runs the rule pipeline and — with an `ml_dataset` — the ML
  engine, returning `{"rule", "ml", "unified"}`: KILLED if either
  chain kills, SURVIVES only if both survive, INCONCLUSIVE when the
  data cannot decide (e.g. ML `UNDERPOWERED`).
- Runtime workspace generation: `atropos init` materialises
  `GRAVEYARD.md`, `hypotheses/INDEX.csv` and `hypotheses/README.md`
  idempotently — existing files are never overwritten.
- DSR and purge unification: one computational home for the
  Bailey & Lopez de Prado deflated Sharpe formula
  (`atropos.evaluation_metrics.deflated_sharpe_from_moments`; the
  returns-based and ML-based versions are wrappers) and one
  time-based purge/embargo home (`atropos._utils.purge_embargo_mask`,
  the row-based `purged_kfold_cv` keeps its own row-distance house).
- Portability test: every shipped text artifact is scanned for
  machine-specific traces — personal user names, absolute Windows
  user paths, references to the private donor corpus the case-study
  numbers were taken from; any new occurrence fails the suite.
- Test suite: 2309 passed / 16 skipped / 2 xfailed — golden,
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
