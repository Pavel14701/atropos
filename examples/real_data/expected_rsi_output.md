# Expected output: `run_rsi_example.py --quick` (actual transcript)

Verbatim stdout of

```bash
uv run python examples/real_data/run_rsi_example.py --quick
```

on the committed snapshot (see README.md for provenance).  **Quick
mode deviates from the frozen prereg by design**: `bootstrap_b=500`
(vs 2000) and `models=['gbm']` (vs gbm/rf/logit) so CI stays fast --
full mode matches the prereg and returns the same run verdict,
`DEAD(importance_stability)`, with rf and logit also dying at
`deflated_sharpe` (0.874 and 0.902 < 0.95).  **Verdicts are the
contract; floats may shift**: EV/DSR/rho digits can differ across
BLAS builds and library versions, but the printed verdicts, the
prereg-before-results order and the snapshot checksum are pinned by
`tests/test_real_data_example.py`.

## Reading the negative control honestly

This run is the second real-data example, and its job is to show the
engine KILLING a weak book on real data.  Three things in the
transcript are worth reading closely:

1. **The verdict is `DEAD`, as preregistered, and anchored to the
   frozen spec** (`prereg_hash=5b8bdc1c` echoes the hash printed in
   STEP 1 -- the engine ran the frozen hypothesis, not an amended
   one).  Every model dies; the run echoes the first non-surviving
   model's gate.
2. **The preregistered mechanism guess was wrong, and that is
   recorded, not smoothed over.**  The frozen hypothesis expected
   the kill "most plausibly at the cost line".  Reality: the taken
   book CLEARS the 2 bps cost line (gbm +14.0bp, rf +20.9bp, logit
   +23.6bp EV net of costs) -- `cost_line` passes everywhere.  The
   deaths come from the harder gates: gbm's per-fold feature
   rankings do not repeat (`rho=-0.5`) and rf/logit miss the
   deflated-Sharpe multiple-testing bar.  Right verdict, wrong
   mechanism -- exactly what honest preregistration is for.
3. **The incremental gate is `INCONCLUSIVE`, by construction.**
   `battery_controls=[]` because the RSI signal is itself a model
   feature; the book must stand on the remaining gates.  This is
   also why this example complements the TSMOM one, where the
   incremental test fires.

```text
==============================================================================
SPY RSI(2) MEAN-REVERSION THROUGH THE ML KILL PROTOCOL (quick preview)
==============================================================================

STEP 1. PREREGISTRATION (printed BEFORE any result; you call
the framework, the framework validates the hypothesis YOU froze)
------------------------------------------------------------------------------
# Preregistration: spy_rsi2_meanrev_ml

- **Frozen on:** 2026-11-06
- **Spec hash:** `5b8bdc1c`

## Hypothesis

NEGATIVE CONTROL. Two-period RSI mean-reversion on SPY -- short the next bar's open after RSI(2) > 70, long after RSI(2) < 30, first exit of RSI(2) crossing 50, a 2xATR(14) stop or target, or 5 bars held -- leaves an executable edge net of 2 bps round-trip costs: classifiers on the RSI state and volatility conditioning take a book that clears the cost line out-of-sample, ranks features consistently across folds and holds a deflated Sharpe above 0.95.  The published reversal edge is thin (a few bp per trade on a two-day hold), so the PREREGISTERED EXPECTATION is the opposite of the hypothesis: this run dies, most plausibly at the cost line -- the framework must kill what cannot survive its own turnover net of costs.

## Construction

- data: committed SPY daily snapshot data/SPY.csv, adjusted close, 2005-01-03..2024-12-31 (one adjustment vintage, sha256-pinned; intraday Open/High/Low mapped into adjusted space by the per-bar AdjClose/Close factor)
- indicators: RSI(2) and RSI(14) with Wilder smoothing, ATR(14) Wilder-smoothed true range, vol_20d trailing 20-day log-return sd
- entries: close t with RSI(2) > 70 shorts the open of bar t+1; RSI(2) < 30 longs it (signal close strictly precedes the execution price)
- exits: first of intrabar 2xATR(14) stop/target touch (stops take priority; a stop gapped through at the open fills at the open), RSI(2) crossing 50 at a close, or the 5-bar cap at that close; the book is non-overlapping
- features at the signal close: rsi_2, rsi_14, vol_20d; battery_controls=[] -- the raw signal is itself a model feature, so the incremental gate reports inconclusive and is NOT a kill criterion for this run
- exec = signed return from entry open to rule exit, net of 2 bps RT cost; target = (exec > 0); long and short events share one frame signed by side
- engine: run_ml_kill_protocol, models gbm/rf/logit, battery_controls=[], k_blocks=6 expanding walk-forward, horizon 168h, purge 24h, embargo 168h, threshold grid 0.30..0.70 step 0.05, bootstrap_b=2000
- gates: profit_bar=0.0 (EV of taken trades must be positive net of costs), importance_rho_min=0.5, dsr_min=0.95, n_min_events=100

## Null hypotheses

**H1:** the gross reversal edge is a cost artifact: taken-trade EV does not clear zero net of 2 bps round-trip costs
**H2:** per-fold feature rankings are noise: importance stability at or below chance
**H3:** the trade stream is indistinguishable from noise at daily frequency: deflated Sharpe of the daily EV stream below 0.95

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | OOS EV of taken trades < 0.0000 net of 2 bps RT costs | < | 0.0000 |
| 2 | median pairwise Spearman of per-fold importances < 0.5 | < | 0.5 |
| 3 | deflated Sharpe of the daily EV stream < 0.95 | < | 0.95 |

## Power analysis

- **N_required:** 100
- **alpha:** 0.05
- **power:** 0.8
- **note:** daily-turnover book: 100 declustered taken events are the floor at which the gates resolve; at the observed ~1.7% per-trade sd they detect per-trade edges of roughly 2 x 1.7% / sqrt(100) ~ 34 bp -- the expected few-bp RSI edge sits far below that bar, which is WHY the honest outcome here is a kill, not a confirmation

## Attestation

This specification (construction, nulls, kill criteria, power) was frozen before any outcome data was evaluated against it. Post-hoc changes invalidate the preregistration; any deviation must be logged as a separate amendment with a new hash.

STEP 2. DATASET from the committed snapshot
------------------------------------------------------------------------------
==============================================================================
DATASET (frozen construction, committed snapshot)
==============================================================================
bars        : 5033 daily rows (2005-01-03..2024-12-31)
events      : 1184 mean-reversion entries (2005-02-02..2024-12-19; 516 long / 668 short)
features    : rsi_2, rsi_14, vol_20d
base rate   : P(exec > 0) = 0.633
trade return: gross 7.6bp  net 5.6bp  sd 169.5bp
hold        : mean 2.75 bars (cap 5), net of 2bps RT

STEP 3. ENGINE RUN (run_ml_kill_protocol returns one dict)
------------------------------------------------------------------------------
engine returned (bootstrap_b=500, models=['gbm'], prereg_hash=5b8bdc1c)

==============================================================================
ML KILL PROTOCOL -- WALK-FORWARD VERDICTS
==============================================================================
run verdict : DEAD(importance_stability)
book        : n_rows=1184  n_events=507  (floor 100, powered=True)
linear book : battery EV=n/a  baseline EV=23.6bp
  model gbm    verdict=DEAD(importance_stability)   EV=14.0bp DSR=0.9246289785813642 rho=-0.5 events=507
    incremental          INCONCLUSIVE
      battery controls not supplied: incremental edge over a linear battery is not decidable in this run
    cost_line            pass
    importance_stability TRIGGERED
      median pairwise Spearman of per-fold importances -0.500 < 0.5: feature rankings unstable
    deflated_sharpe      TRIGGERED
      deflated Sharpe of the daily EV stream 0.9246 < 0.95: edge does not survive multiple-testing discount
  incremental gbm: t_vs_battery=n/a  t_vs_baseline=-20.057

WHY THIS IS THE NEGATIVE CONTROL
------------------------------------------------------------------------------
RSI(2) mean-reversion looks tradable gross: a real, published
reversal edge whose taken book even clears a 2 bps cost line.
But the edge is too thin to survive walk-forward scrutiny:
fold-to-fold feature rankings do not repeat and the daily EV
stream misses the deflated-Sharpe multiple-testing bar.
Expected: DEAD -- and a kill here is the framework WORKING:
it correctly killed a strategy that looks good gross but is
too weak to survive honest evaluation.  The companion example
run_example.py (SPY 12-1 TSMOM) shows the same engine on a
book whose cost-line edge is an order of magnitude stronger.

Verdicts are evaluation output, not advice: adapt these calls
inside your own pipeline and decide what a kill means for you.
Gate semantics and defaults: DEFAULT_TRAINING_CONFIG and
docs/how-to/05-run-ml-kill-protocol.md.
```
