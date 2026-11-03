# Changelog

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
