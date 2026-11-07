# Changelog

## 1.2.0 — independent-audit fix waves (review + defect registry)

Every fix below is pinned by a test that fails on the pre-fix code
(mutation-verified); repro outputs are recorded in the respective
commit messages.

- Kill-chain semantics (ATR-18): a rule-chain `INCONCLUSIVE` no longer
  unifies as `KILLED` — all four rule-INCONCLUSIVE × ML cells now read
  `INCONCLUSIVE [rule]` per the decision matrix (the ATR-06 branch was
  dead code); undecided is not death.
- Portfolio policy credit (ATR-19): an unresolved displaced trade
  credits `_NO_CREDIT` under both policies (the `replace_high` branch
  no longer poisons the equity stream with NaN); the power floor
  counts only realized credits; correlation admission validates the
  candidate before the empty-streams early return; non-string
  instrument labels are rejected loudly.
- Prereg → gates bridge (ATR-07): `gates_kwargs_from_prereg` +
  `run_portfolio(prereg=..., prereg_hash=...)` — thresholds derive
  ONLY from the frozen prereg, the hash is verified loudly and embedded
  in the summary and gate reports; engine boundaries aligned to the
  prereg `comparison` semantics at equality (concentration `>=`,
  DSR `<=`); prereg validation gains threshold domains, strict
  `YYYY-MM-DD` dates, 64-char hash anchoring and merged problem lists.
- Walk-forward (ATR-09, partial — see KNOWN-ISSUES §6): the FRAMEWORK
  stage-6 kill criterion is now enforced — `shrink_ratio > shrink_cap`
  (config `walkforward.shrink_cap`, default 2.0) → `FAIL` with
  `verdict_reason="OVERFIT_DEATH: shrinkage Nx"`, which stage 4 maps
  to `DEAD`; a NaN shrink (degenerate baseline t) reads INCONCLUSIVE,
  never a silent pass; `label_horizon > 1` raises (no overlap
  correction exists — KNOWN-ISSUES §6 accepted risk; the parameter is
  separate from `purge_gap`); skill 16 synced to the expanding doctrine
  (FRAMEWORK.md §221-233); three-protocol note recorded (§8, code
  canonical).
- Cost line (ATR-11): stage 5 now gates the breakeven —
  `cost_bps >= breakeven` (finite, positive) attributes `COST_DEATH`
  (boundary inclusive, pinned; mdd dominance unchanged); the three
  cost baselines (docs 20bp law, engine 25bp taker, stage-5 10bp
  one-way) are stated in FRAMEWORK.md; METHODOLOGY.md law references
  de-conflated.
- Demo integrity (ATR-05): demo signals carry the decision-bar ATR
  (no more one-bar lookahead in sl/tp/ranker feature; transcript
  refreshed — ndcg@k 0.819, mrr 0.778); the stage-5 maker leg sits at
  the banner 0.25 ATR (the double-offset regression is pinned by a
  separation-asserted test); ranking failures are no longer masked as
  honest single-class skips (only the marker matches); caller-config
  errors fail loudly instead of reading as INCONCLUSIVE; demo stage 1
  ↔ stage 5 alignment is pinned (same signal, exactly one lag).
- Statistics core: power noncentrality uses residual df (MDE
  0.0039129 → 0.0039642, golden pinned, docstring documents the
  rejected Cohen f²N convention); track-record max drawdown sends any
  non-positive equity path to the −1.0 sentinel; `mean_block` is
  validated loudly; the known-features control keeps small-scale
  informative features (relative variance floor).
- Backtester: `maker_entry_sim` docs declare the fill-bar contract
  (semantics unchanged); the cost_model overflow handler no longer
  crashes on opposite-sign extreme positions; `_align_panels` rejects
  scrambled plain indices loudly; `sharpe_ratio` no longer warns below
  two observations; the spy_tsmom docstring states the real ~1-2 bar
  event overlap.
- Infrastructure: the portability ratchet scans `.yml/.yaml/.json/
  .cfg/.txt`; the conventions and meta-architecture scans walk
  subpackages recursively; `atropos init` writes atomically;
  `generate_prereg` accepts only strict `YYYY-MM-DD`; the purged-K-fold
  p-value uses scored-fold dof; regime keys are neutral
  (`regime_i`); the root docstring documents the sklearn import cost.
- CI/docs: the module-demo smoke is a rot-proof meta-test (discovers
  and runs every `__main__` module; `ranking` gains a self-checking
  demo); docs claims synced (12 skills, 16 `__all__` entries, honest
  "Most modules" demo claim); KNOWN-ISSUES §6/§7/§8 added; test
  counters match reality.
- Push-gate hardening: a stale `xfail(strict)` claim in KNOWN-ISSUES
  §3 (the marker no longer existed, masked by the xfail total) was
  caught by an external review and rewritten as a tracked gap — the
  whole inventory then went through a grep-based claim sweep
  (counters, xfail claims, absolute claims, how-to transcripts, all
  12 skills annotated with the canonical map) and a closure pass:
  every backlog item carries an explicit `Trigger to close:`; §5b
  closes via the already-louder `enforce_embargo` ValueError; §5a
  prereg closes at the opt-in level (`strict_prereg_hash`); §4
  reframed as behavioral facts; known issues in the strict sense: none.
- CI split: `pytest -m "not slow"` is the default path; the @slow
  bucket (calibration battery + the ~140s module-demo meta-test) runs
  as its own retryable step — coverage preserved, default path fast.
  The cli argparse exclusion gained a behavioral pin (bare
  `python -m atropos.cli` must exit 2).
- Test suite: 2675 → 2781 passed, 18 skipped, 2 xfailed — every
  registry defect fixed above is pinned; ATR-09 and ATR-13 are
  explicitly "partially closed + accepted risk" (KNOWN-ISSUES §6/§8),
  not silently claimed done.

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
  agent context (AGENTS.md workflow + all twelve skills) are fully
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
- Test suite: 2734 passed / 18 skipped / 2 xfailed — golden,
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
