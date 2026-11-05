# Real-data example: SPY 12-1 time-series momentum through the ML kill protocol

A runnable example of **calling the atropos library** on real market
data.  Atropos is a library, not a framework: this directory is a
scripted demonstration of the calls — `generate_prereg` validates and
renders the hypothesis you froze, `run_ml_kill_protocol` returns one
report dict, and YOU decide what the verdicts mean for your pipeline.
Adapt these calls inside your own research code; nothing here takes
over control flow.

The strategy is the classic single-asset time-series momentum
(Moskowitz, Ooi & Pedersen 2012, adapted to SPY): each month, go long
the next month when the 12-month return **skipping the most recent
month** is positive, flat otherwise — and ask the ML kill protocol
whether classifiers on conditioning features produce a book that
survives the four gates out-of-sample.

## Quick start (offline — uses the committed snapshot)

```bash
uv sync
uv run python examples/real_data/run_example.py --quick   # CI preview, ~3s
uv run python examples/real_data/run_example.py           # full protocol, ~6s
```

`--quick` runs the SAME frozen dataset, gates and walk-forward with one
model (`gbm`) and fewer bootstrap draws (500 vs 2000).  The full run
evaluates the preregistered three-model comparison
(`models=["gbm", "rf", "logit"]`).  Expected output is committed in
[expected_output.md](expected_output.md) (actual `--quick` transcript).

Order matters and the runner prints it: **prereg first, results
second**.  `preregistration.py` holds the frozen params; the spec hash
changes if you edit any frozen field.  Generating the prereg before
looking at results is YOUR discipline — the library validates whatever
hypothesis text you froze — and `tests/test_real_data_example.py`
pins that order.

## Layout

```
examples/real_data/
├── README.md                   # this file
├── data/SPY.csv                # committed daily snapshot (5033 bars)
├── data/SPY.csv.sha256         # pins the snapshot bytes
├── download_data.py            # refresh utility (Stooq -> Yahoo fallback)
├── strategies/spy_tsmom.py     # load_spy / make_features / make_events
├── preregistration.py          # frozen params-dict for generate_prereg
├── run_example.py              # --quick / full: prereg -> run -> verdicts
└── expected_output.md          # actual transcript of the --quick run
```

## Data provenance and caveats

- **Snapshot**: `data/SPY.csv`, SPY (SPDR S&P 500 ETF) daily bars,
  2005-01-03 .. 2024-12-31, 5033 rows.  `AdjClose` is the split- AND
  dividend-adjusted close (the total-return series the momentum
  literature uses).
- **Source**: Yahoo Finance chart API, retrieved 2026-11-05.  The
  Stooq CSV endpoint (`https://stooq.com/q/d/l/?s=spy.us...`) named in
  the original plan now gates plain-HTTP downloads (JavaScript
  proof-of-work, then `Access denied` even with the challenge
  cleared); `download_data.py` tries it first and falls back to Yahoo,
  normalizing to the snapshot's column layout.
- **Vintage**: adjusted closes are vintage-dependent — every new SPY
  dividend re-bases the whole back-adjusted history, and the provider
  recomputes them per request (re-fetches differ in the last digits of
  `AdjClose` even within one day; raw OHLCV is stable).  The snapshot
  freezes ONE vintage; `SPY.csv.sha256` pins it.
  `download_data.py` compares hashes and RAISES with advice to keep
  the committed snapshot rather than silently mixing vintages.
- **Rights**: not for redistribution — source data belongs to the
  provider; the committed snapshot is for reproducing this example
  only.
- **Cost model**: a flat 2 bps round-trip per monthly trade.  Real
  books face spreads/impact/funding; raising the cost changes the
  economics, not the code.

## The frozen protocol (what the runner does)

1. **Dataset** (`strategies/spy_tsmom.py`): monthly `ME` resample of
   `AdjClose`; `mom_12_1 = P[m-1]/P[m-13] - 1` (12-month return
   skipping the newest month); entry = first trading day of the next
   month when `mom_12_1 > 0`, exit 21 trading days later at the close.
   Features at the entry close: `vol_20d`, `ma_gap_50`, `ma_gap_200`,
   plus the raw `mom_12_1`.  No lookahead: the skip-1-month signal
   lags the entry by >= 3 weeks by construction, and the 21-day
   forward return of every kept event ends inside the snapshot.
2. **Engine calls** (`run_example.py`):
   `run_ml_kill_protocol(features, target, exec_returns, times_us,
   config=..., battery_controls=["mom_12_1"])` with `k_blocks=6`
   expanding walk-forward, `horizon_h=744, purge_h=24, embargo_h=744`,
   threshold grid 0.30..0.70, `profit_bar=0.0` (taken-trade EV must be
   positive NET of the 2 bps cost), gates at their defaults
   (`incremental_t=2.0`, `importance_rho_min=0.5`, `dsr_min=0.95`) and
   `n_min_events=50`.
3. **Battery design**: `mom_12_1` is passed as a battery control, so
   the classifiers are judged on the conditioning features against a
   linear logit holding the classic momentum signal — the incremental
   gate is exactly the question "does ML add value over raw TSMOM?".

The dataset has 186 long monthly entries (2006-03 .. 2024-11, base
rate `P(exec > 0) = 0.66`, mean net exec +99 bp, sd 416 bp) — 30+ per
the engine's power floor (113 taken events out-of-sample, floor 50).
`n_min_events=50` is a deliberate, preregistered choice for a monthly
book: 50 taken monthly events resolve monthly effects of roughly
1.2% (2 x 4.2% / sqrt(50)); smaller true edges are killed, not
confirmed.  This is weaker power than the engine's 500-event default,
which assumes a much denser book.

## Expected verdicts (verdicts are the contract; floats may shift)

**Expected run verdict: `DEAD(importance_stability)` — and that is the
point, not a failure.** The engine killed a strategy that clears the
cost line and beats the linear battery by 7.8 sigmas, because its
per-fold feature-importance rankings are unstable. A first-run `DEAD`
on a "known good" strategy demonstrates that this library is not a
confirmation machine: a high incremental t-stat is necessary but not
sufficient — if you cannot explain WHY the model wins consistently
(the importance story flips fold to fold), the edge is not portable.

Gate-calibration note: `importance_stability` compares per-fold
importance rankings by pairwise Spearman. With only THREE features the
ranking space is tiny (rho quantizes to {-1, -0.5, 0, 0.5, 1}), so the
0.5 cutoff is conservative to the point of near-binariness — the gate
is calibrated for wider feature sets (10+). With 3 features treat an
importance kill as "untested portability", not as strong evidence of
instability.

Data finding (not a bug): the linear 12-1 battery (84.7 bp/month)
UNDERPERFORMS buy-and-hold drift (104.5 bp) on this 2005-2024 sample —
`t_vs_baseline` is negative. The classifier (97.0 bp) beats the
battery but not the baseline. This is consistent with the
well-documented underperformance of SPY time-series momentum versus
buy-and-hold over this period; it is a finding about the data, not a
sign error in the framework.

From the committed snapshot and seeds (both modes are
byte-deterministic; floats like EV/DSR may shift a digit across BLAS
builds — the verdicts are the contract):

| Mode | Run verdict | Per-model |
|---|---|---|
| `--quick` (gbm, b=500) | `DEAD(importance_stability)` | gbm `DEAD(importance_stability)` |
| full (gbm, rf, logit, b=2000) | `DEAD(importance_stability)` | gbm `DEAD(importance_stability)`; rf `DEAD(importance_stability)`; logit `SURVIVES` |

Reading: every model's taken book clears the cost line (EV 97-116 bp
net of 2 bps vs a battery EV of 85 bp for the classic linear book) and
clears the deflated-Sharpe gate, but per-fold permutation-importance
rankings are unstable for the tree models (median pairwise Spearman
-0.5, i.e. the "which feature matters" story flips from fold to
fold) — so the honest tombstone is `DEAD(importance_stability)`, while
the plain linear model on the same conditioning features survives all
four gates.  The batteries' own numbers are printed so you can judge
the linear TSMOM book itself (85 bp/month net).

`tests/test_real_data_example.py` pins: exit code 0, the prereg
printed before any verdict, the `--quick` verdict tokens, and the
snapshot checksum.

## Adapt this to your data

1. Replace the snapshot with your instrument's daily CSV (same
   columns) or rewrite `strategies/spy_tsmom.py` for your
   construction — keep the no-lookahead rules.
2. Edit `preregistration.py` BEFORE looking at any walk-forward
   output: hypothesis, construction, nulls, kill criteria, power.
   `validate_prereg` refuses soft criteria; `generate_prereg` renders
   the document.
3. Call `run_ml_kill_protocol` with your frame + `config` overrides
   (gates, thresholds, `models`) and read the returned dict.  Verdict
   semantics: `SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED`; per-model
   verdicts live in `report["models"]` — the engine never selects a
   winner, you do.
