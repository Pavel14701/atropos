# Expected output: `run_example.py --quick` (actual transcript)

Verbatim stdout of

```bash
uv run python examples/real_data/run_example.py --quick
```

on the committed snapshot (see README.md for provenance).  **Quick
mode deviates from the frozen prereg by design**: `bootstrap_b=500`
(vs 2000) and `models=['gbm']` (vs gbm/rf/logit) so CI stays fast —
full mode matches the prereg.  **Verdicts
are the contract; floats may shift**: EV/DSR/rho digits can differ
across BLAS builds and library versions, but the printed verdicts, the
prereg-before-results order and the snapshot checksum are pinned by
`tests/test_real_data_example.py`.

```text
==============================================================================
SPY 12-1 TSMOM THROUGH THE ML KILL PROTOCOL (quick preview)
==============================================================================

STEP 1. PREREGISTRATION (printed BEFORE any result; you call
the library, the library validates the hypothesis YOU froze)
------------------------------------------------------------------------------
# Preregistration: spy_tsmom_12_1_ml

- **Frozen on:** 2026-11-05
- **Spec hash:** `af8161cc`

## Hypothesis

Twelve-minus-one momentum on SPY -- long the first trading day of every month whose prior month-end 12-1 momentum is positive, held 21 trading days -- leaves an executable edge net of 2 bps round-trip costs, and classifiers on volatility and moving-average conditioning take a book that clears the cost line out-of-sample, beats the linear momentum battery incrementally, ranks features consistently across folds and holds a deflated Sharpe above 0.95.

## Construction

- data: committed SPY daily snapshot data/SPY.csv, adjusted close, 2005-01-03..2024-12-31 (one adjustment vintage, sha256-pinned)
- signal: monthly resample (ME); mom_12_1 = P[m-1]/P[m-13] - 1 (12-month return skipping the newest month)
- entries: first trading day of month m+1 when mom_12_1 > 0 (long/flat); exit 21 trading days later at the close
- features at the entry close: vol_20d, ma_gap_50, ma_gap_200; raw mom_12_1 feeds the linear battery only
- exec = 21-day forward close-to-close return - 2 bps RT cost; target = (exec > 0)
- engine: run_ml_kill_protocol, models gbm/rf/logit, battery_controls=[mom_12_1], k_blocks=6 expanding walk-forward, horizon 744h, purge 24h, embargo 744h, threshold grid 0.30..0.70 step 0.05, bootstrap_b=2000
- gates: incremental_t=2.0, profit_bar=0.0 (EV of taken trades must be positive net of costs), importance_rho_min=0.5, dsr_min=0.95, n_min_events=50

## Null hypotheses

**H1:** the conditioning adds nothing: a linear battery on raw mom_12_1 trades an equal-or-better book than the classifiers
**H2:** the edge is repackaged equity drift: taken-trade EV does not exceed the unconditional drift of long months
**H3:** per-fold feature rankings are noise: importance stability at or below chance

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | paired bootstrap t of taken-trade EV vs the linear mom_12_1 battery | < | 2.0 |
| 2 | OOS EV of taken trades < 0.0000 net of 2 bps RT costs | < | 0.0000 |
| 3 | median pairwise Spearman of per-fold importances < 0.5 | < | 0.5 |
| 4 | deflated Sharpe of the daily EV stream < 0.95 | < | 0.95 |

## Power analysis

- **N_required:** 50
- **alpha:** 0.05
- **power:** 0.8
- **note:** monthly book: 50 taken monthly events are the floor at which the gates resolve; they detect monthly effects of roughly 2 x 4.2% / sqrt(50) ~ 1.2% -- smaller true edges are killed, not confirmed

## Attestation

This specification (construction, nulls, kill criteria, power) was frozen before any outcome data was evaluated against it. Post-hoc changes invalidate the preregistration; any deviation must be logged as a separate amendment with a new hash.

STEP 2. DATASET from the committed snapshot
------------------------------------------------------------------------------
==============================================================================
DATASET (frozen construction, committed snapshot)
==============================================================================
bars        : 5033 daily rows (2005-01-03..2024-12-31)
events      : 186 long monthly entries (2006-03-01..2024-11-01)
features    : vol_20d, ma_gap_50, ma_gap_200, mom_12_1
base rate   : P(exec > 0) = 0.661
exec return : mean 99.1bp  sd 416.4bp  (21d hold, net of 2bps RT)

STEP 3. ENGINE RUN (run_ml_kill_protocol returns one dict)
------------------------------------------------------------------------------
engine returned (bootstrap_b=500, models=['gbm'])

==============================================================================
ML KILL PROTOCOL -- WALK-FORWARD VERDICTS
==============================================================================
run verdict : DEAD(importance_stability)
book        : n_rows=186  n_events=113  (floor 50, powered=True)
linear book : battery EV=84.7bp  baseline EV=104.5bp
  model gbm    verdict=DEAD(importance_stability)   EV=97.0bp DSR=0.9814864879621145 rho=-0.5 events=113
    incremental          pass
    cost_line            pass
    importance_stability TRIGGERED
      median pairwise Spearman of per-fold importances -0.500 < 0.5: feature rankings unstable
    deflated_sharpe      pass
  incremental gbm: t_vs_battery=7.824  t_vs_baseline=-3.463

Verdicts are evaluation output, not advice: adapt these calls
inside your own pipeline and decide what a kill means for you.
Gate semantics and defaults: DEFAULT_TRAINING_CONFIG and
docs/how-to/05-run-ml-kill-protocol.md.
```
