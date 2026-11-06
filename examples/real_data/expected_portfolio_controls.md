# Expected output: `run_portfolio_controls.py` (actual transcript)

Verbatim stdout of

```bash
uv run python examples/real_data/run_portfolio_controls.py
```

(the FULL protocol) on the committed 10-sector SPDR snapshots, run
2026-11-06.  **`--quick` deviates from the full protocol by design and
only in Phase A**: the null-world count drops to 20 (vs 100 seeded
nulls) so CI stays fast; every other constant — datasets, gates,
thresholds, seeds of Phases B/C — is identical, and Phases B/C print
byte-identical output in both modes.  **Verdicts are the contract;
floats may shift**: EV/DSR/drawdown digits can differ across BLAS
builds and library versions, but the printed verdicts, the
prereg-before-results order and the snapshot checksums are pinned by
`tests/test_real_data_example.py`.  The Phase A FPR number is a random
quantity by nature (it counts certified noise worlds) and is reported,
never pinned.

Frozen prereg hashes (any edit to a phase's frozen params changes its
hash): null admission `5bf0ac47`, momentum control `3372d2e2`, noise
control `a53c2b6f`.

## Verdict table (the recorded outcomes)

| Phase | Verdict | Triggered gates |
|---|---|---|
| A null (random admission, MC, 100 worlds) | `HELD (0/100 certified)` | deflated_sharpe 100/100, drawdown_floor 95/100, side_concentration 100/100 |
| B momentum (positive control) | `DEAD(deflated_sharpe)` | all three |
| C random (negative control, full machinery) | `DEAD(side_concentration)` | side_concentration only |

```text
==============================================================================
PORTFOLIO CONTROLS: MOMENTUM vs NOISE ON 10 SECTOR SPDRs (full protocol)
==============================================================================

FROZEN CONFIGURATION (shared by all phases):
    slots                  6
    policy                 fcfs
    corr_cap               0.9
    hold_bars              21
    dsr_min                0.95
    dd_floor_r             -15.0
    max_same_side_share    0.8
    n_trials               1
    min_trades             100
    ev_period_bars         1
    cost_bps               2

==============================================================================
PHASE A. NULL (random admission) -- FPR calibration
==============================================================================
# Preregistration: portfolio_null_admission_fpr

- **Frozen on:** 2026-11-06
- **Spec hash:** `5bf0ac47`

## Hypothesis

NEGATIVE CONTROL (FPR calibration of the portfolio gates). Books built from the momentum portfolio's exact bars, monthly cadence, cohort sizes and 2xATR stop geometry, but with a RANDOM cross-sectional selection (3 random longs and 3 random shorts per cohort, admission scores U(0,1)), carry no edge.  Replayed through run_portfolio_replay (slots=6, fcfs, correlation filter off) and judged by portfolio_gates (dsr_min 0.95 with n_trials=1, dd_floor -15 R, same-side cap 0.80, min_trades 100), they must NOT be certified.  The preregistered expectation is zero worlds at verdict SURVIVES over the seeded world set; any certified world is the portfolio layer's false-certification (FPR) finding, reported honestly rather than hidden.  The FPR number itself is a random quantity by nature and is never pinned -- the STRUCTURE (well-formed 3-gate reports, a resolvable verdict per world) is the contract.

## Construction

- data: committed Select Sector SPDR snapshots data/<TICKER>.csv (XLF XLE XLK XLI XLV XLY XLP XLRE XLU XLB), adjusted OHLC, sha256-pinned; request window 2005-01-01..2025-01-01 but the real-estate fund (XLRE) incepted 2015-10-08, so the shared finite bar grid starts 2015-10-08 (2323 bars)
- null books: identical decision bars, cohort sizes (3 long / 3 short), 2xATR(14) stop geometry around the decision close, 21-bar hold cap and 13-month formation warm-up as the momentum book; instruments per cohort drawn uniformly without replacement; scores U(0,1); see strategies.sector_momentum.build_null_signals
- engine path (the raw seam): run_portfolio_replay(signals, panel, slots=6, policy='fcfs', corr_cap=None, hold_bars=21, cost=flat 2 bps RT) then portfolio_gates(equity_curve, per-bar EV stream, dsr_min=0.95, dd_floor=-15.0, max_same_side_share=0.80, n_trials=1, min_trades=100, concurrent and net exposure attached)
- correlation filter disabled for this phase: the MC needs no r_proxy columns (the full machinery WITH the filter is Phase C's control)
- worlds: 100 seeded nulls (seed = 20261106 + w); --quick reduces ONLY this count (to 20); world 0's full gate anatomy is printed as the representative

## Null hypotheses

**H1:** the random books carry a tradeable edge (refuted when the gates trigger on them)
**H2:** the gate machinery cannot separate noise from an edge at these thresholds (measured by the certified share of worlds)

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | share of null worlds certified (portfolio verdict SURVIVES) -- the false-certification rate | > | 0.0 |
| 2 | share of null worlds UNDERPOWERED (fewer than 100 credited trades -- the gates would not resolve) | > | 0.0 |
| 3 | share of worlds with a malformed gate report (fewer than 3 gates or an unknown verdict token) | > | 0.0 |

## Power analysis

- **N_required:** 100
- **alpha:** 0.05
- **power:** 0.8
- **note:** 100 null worlds bound a 0% certified share with 95% confidence at roughly 3% (rule of three): the run can honestly report 'no world certified' without proving the FPR is exactly zero

## Attestation

This specification (construction, nulls, kill criteria, power) was frozen before any outcome data was evaluated against it. Post-hoc changes invalidate the preregistration; any deviation must be logged as a separate amendment with a new hash.

worlds evaluated: 100
verdict counts    : DEAD(deflated_sharpe) x 100
FALSE-CERTIFICATION RATE (SURVIVES share): 0.000 (0/100 worlds)
gate trigger rates: deflated_sharpe 100/100, drawdown_floor 95/100, side_concentration 100/100
no world certified: the gates refused every noise book (the FPR-control expectation holds).

representative world (seed 20261106) anatomy:
    accepted=557  rejected=25 (corr 0, capacity 25)  replaced=0  unresolved=0
    final_equity=+2.41R  max_drawdown=18.31R  concurrent_max=6
    verdict: DEAD(deflated_sharpe)
    deflated_sharpe        TRIGGERED  value=0.5415
      dsr 0.542 below floor 0.950
    drawdown_floor         TRIGGERED  value=-18.3119
      drawdown -18.312 reached the floor -15.000
    side_concentration     TRIGGERED  value=1.0000
      same-side share 1.000 above cap 0.800
==============================================================================
PHASE B. MOMENTUM PORTFOLIO (positive control)
==============================================================================
# Preregistration: sector_xs_momentum_portfolio

- **Frozen on:** 2026-11-06
- **Spec hash:** `3372d2e2`

## Hypothesis

POSITIVE CONTROL.  Cross-sectional 12-1 momentum on the ten Select Sector SPDR ETFs -- long the top-3 / short the bottom-3 by trailing 12-1 total-return momentum, monthly cohorts, 2xATR(14) stop/target around the decision close, 21-bar hold, within-cohort rank-normalized admission scores -- replayed through the FULL portfolio machinery (run_portfolio: slots=6, fcfs, correlation admission at |Spearman| <= 0.90 on growing return proxies, flat 2 bps round trip) leaves an edge the portfolio gates certify: deflated Sharpe of the per-bar EV stream >= 0.95 (n_trials=1), event drawdown above -15 R, same-side share <= 0.80, at least 100 credited trades.  Honest expectation recorded BEFORE the run: the XLRE-constrained panel covers 2016-12..2024-11, a window whose idealized no-stop long-short measures ~0 edge (see the construction audit), and the engine book additionally pays the overnight gap into momentum entries (levels are placed from the decision close; winners gap away from them).  A DEAD verdict from these gates would be the machinery correctly refusing a book without a sample edge; a SURVIVES verdict would certify a real anomaly.  Both verdicts are informative; a malformed or UNDERPOWERED run is not.

## Construction

- data: committed Select Sector SPDR snapshots data/<TICKER>.csv (XLF XLE XLK XLI XLV XLY XLP XLRE XLU XLB), adjusted OHLC, sha256-pinned; request window 2005-01-01..2025-01-01 but the real-estate fund (XLRE) incepted 2015-10-08, so the shared finite bar grid starts 2015-10-08 (2323 bars)
- signal: monthly month-end adjusted closes; mom_12_1 = M[m-1]/M[m-13] - 1 (12-month return skipping the newest month); z-scored across the ten instruments per month-end
- cohorts: decision at the month-end's last trading bar (all inputs use data up to and including that close), fill at decision_idx + 1 (the engine's generator rule: next bar's open); long top-3 / short bottom-3; 6 rows per cohort; cohorts whose 21-bar hold window would pass the last bar are dropped
- stops: sl/tp = 2xATR(14, Wilder) around the DECISION close (levels known at decision time); the engine resolves fills and intrabar touches pessimistically (SL first)
- scores: within-cohort rank normalization via atropos.ranking.rank_normalize_group (the engine's within-group admission contract); priority = conviction
- r_proxy: per-instrument daily adjusted returns from one common anchor through the decision bar (the correlation filter's growing series)
- engine: run_portfolio(signals, panel, slots=6, policy='fcfs', corr_cap=0.90, hold_bars=21, cost=flat 2 bps RT, dsr_min=0.95, dd_floor=-15.0, max_same_side_share=0.80, n_trials=1, min_trades=100, ev_period_bars=1)
- construction audit (pre-run, diagnostic only): the IDEALIZED no-stop monthly long-short on the same scores measures ~0 edge on this window (annualized mean -0.7%, Sharpe about -0.1) -- the 2005-2015 momentum golden decade is unavailable below the XLRE inception, so a DEAD verdict on the engine book would be the machinery refusing a signal with no measured sample edge

## Null hypotheses

**H1:** the engine book's outcome is driven by the stop/gap geometry rather than the momentum signal (the idealized no-stop long-short holds ~0 edge on this window)
**H2:** any apparent edge is a cost artifact of the flat 2 bps round-trip model
**H3:** the book cannot survive admission at all (correlation rejects and capacity rejections thin the cohorts below the power floor)

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | deflated Sharpe of the per-bar portfolio EV stream (n_trials=1) | < | 0.95 |
| 2 | event-basis max drawdown of the equity, in R | < | -15.0 |
| 3 | max same-side share of concurrent positions | > | 0.8 |
| 4 | credited trades (the verdict is UNDERPOWERED below this) | < | 100 |

## Power analysis

- **N_required:** 100
- **alpha:** 0.05
- **power:** 0.8
- **note:** monthly-cohort book: ~97 cohorts x 6 candidates across 20 years of requested history; the 100-credited-trade floor is well below the expected filled book, so the gates resolve on edge quality, not book size

## Attestation

This specification (construction, nulls, kill criteria, power) was frozen before any outcome data was evaluated against it. Post-hoc changes invalidate the preregistration; any deviation must be logged as a separate amendment with a new hash.

dataset: 582 entry rows, 97 monthly cohorts (291 long / 291 short), decisions at bars 289..2301 (2016-11-30..2024-11-29)
    idealized no-stop long-short audit: months=97  ann_mean=-0.74%  ann_sd=7.04%  ann_sharpe=-0.11

    accepted=535  rejected=47 (corr 39, capacity 8)  replaced=0  unresolved=0
    final_equity=-29.33R  max_drawdown=31.82R  concurrent_max=6
portfolio verdict: DEAD(deflated_sharpe)
    deflated_sharpe        TRIGGERED  value=0.1066
      dsr 0.107 below floor 0.950
    drawdown_floor         TRIGGERED  value=-31.8197
      drawdown -31.820 reached the floor -15.000
    side_concentration     TRIGGERED  value=1.0000
      same-side share 1.000 above cap 0.800
==============================================================================
PHASE C. RANDOM PORTFOLIO (negative control, full machinery)
==============================================================================
# Preregistration: portfolio_noise_full_machinery

- **Frozen on:** 2026-11-06
- **Spec hash:** `a53c2b6f`

## Hypothesis

NEGATIVE CONTROL.  One seeded random book (exactly the Phase A null construction, different fixed seed) through the SAME full machinery as the momentum control (run_portfolio with correlation admission at |Spearman| <= 0.90 on growing return proxies): the machinery must refuse it -- the portfolio verdict is NOT SURVIVES.  A SURVIVES verdict certifies pure noise and is the portfolio layer's false-certification finding, reported honestly.  The four kill criteria below are the individual certification conditions; the noise book is certified -- and this control's claim is killed -- only when ALL of them hold together.

## Construction

- data: committed Select Sector SPDR snapshots data/<TICKER>.csv (XLF XLE XLK XLI XLV XLY XLP XLRE XLU XLB), adjusted OHLC, sha256-pinned; request window 2005-01-01..2025-01-01 but the real-estate fund (XLRE) incepted 2015-10-08, so the shared finite bar grid starts 2015-10-08 (2323 bars)
- noise book: strategies.sector_momentum.build_null_signals with fixed seed 20261107, r_proxy columns attached (same construction as Phase A's nulls: random 3 long / 3 short per monthly cohort, scores U(0,1), 2xATR stops, 21-bar hold, 13-month warm-up -- identical bars and structure to the momentum book)
- engine: run_portfolio(signals, panel, slots=6, policy='fcfs', corr_cap=0.90, hold_bars=21, cost=flat 2 bps RT, dsr_min=0.95, dd_floor=-15.0, max_same_side_share=0.80, n_trials=1, min_trades=100, ev_period_bars=1) -- the FULL machinery including the correlation filter, which must reject same-instrument re-entries (|rho| = 1)
- single fixed-seed world (seed 20261107): the FPR-over-worlds measurement is Phase A's job; this phase demonstrates the full pipeline's behavior on one honest noise draw

## Null hypotheses

**H1:** the full machinery certifies pure noise at these thresholds (all four certification conditions hold -- the false-certification finding)
**H2:** the correlation admission fails to block same-instrument re-entries in the noise book

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | noise-book deflated Sharpe at or above the 0.95 certification floor (n_trials=1) | >= | 0.95 |
| 2 | noise-book event drawdown above the -15 R disaster floor (no blowup observed) | > | -15.0 |
| 3 | noise-book max same-side share within the 0.80 cap | <= | 0.8 |
| 4 | noise-book credited trades above the 100 floor (the verdict resolves instead of UNDERPOWERED) | > | 100 |

## Power analysis

- **N_required:** 100
- **alpha:** 0.05
- **power:** 0.8
- **note:** single-world control: the DSR gate alone mis-certifies roughly 5% of noise worlds at n_trials=1 (a z-statistic at the 1.645 bar), which is why the world-level FPR measurement lives in Phase A; this phase pins the FULL pipeline's anatomy on one seeded draw

## Attestation

This specification (construction, nulls, kill criteria, power) was frozen before any outcome data was evaluated against it. Post-hoc changes invalidate the preregistration; any deviation must be logged as a separate amendment with a new hash.

dataset: 582 entry rows, 97 monthly cohorts (291 long / 291 short, random selection, scores U(0,1))

    accepted=458  rejected=124 (corr 122, capacity 2)  replaced=0  unresolved=0
    final_equity=+43.21R  max_drawdown=12.66R  concurrent_max=6
portfolio verdict: DEAD(side_concentration)
    deflated_sharpe        pass       value=0.9828
      ok
    drawdown_floor         pass       value=-12.6557
      ok
    side_concentration     TRIGGERED  value=1.0000
      same-side share 1.000 above cap 0.800
the machinery refused the noise book: the negative-control expectation holds.

==============================================================================
SUMMARY (verdicts are recorded, never enforced)
==============================================================================
    phase                                verdict                    triggered
    A null (random admission, MC)        HELD (0/100 certified)     deflated_sharpe,drawdown_floor,side_concentration
    B momentum (positive control)        DEAD(deflated_sharpe)      deflated_sharpe,drawdown_floor,side_concentration
    C random (negative control)          DEAD(side_concentration)   side_concentration

WHAT EACH PHASE CALIBRATES
------------------------------------------------------------------------------
A: the gates' false-certification rate on noise books with
   the momentum book's exact structure (FPR control; the
   number is a random quantity and is never pinned).
B: whether the machinery certifies a real cross-sectional
   book on an honest sample -- SURVIVES or DEAD are both
   informative; the STRUCTURE of the kill is the contract.
C: the full machinery (correlation admission included) on
   one seeded noise world -- must refuse what A measured.

Verdicts are evaluation output, not advice: adapt these
calls inside your own pipeline and decide what they mean
for you.  Gate semantics and defaults: atropos.portfolio.
```

## What each phase calibrates

**Data layer.**  Ten Select Sector SPDR snapshots (XLF XLE XLK XLI
XLV XLY XLP XLRE XLU XLB), each sha256-pinned, request window
2005-01-01..2025-01-01.  One data fact shapes everything downstream:
the real-estate Select Sector fund is `XLRE` (there is no `XRE`
ticker) and it INCEPTED 2015-10-08, so the shared finite bar grid
starts 2015-10-08 (2323 bars, not 5033).  The 10-name cross-section
only exists from that date; the momentum book therefore covers
2016-12..2024-11 (97 monthly cohorts, 582 entry rows, 291 long /
291 short).

**Phase A — null (FPR calibration).**  100 seeded noise books with
the momentum book's exact bars, cadence, cohort sizes and stop
geometry (random 3-long/3-short selection, `U(0,1)` scores) through
`run_portfolio_replay` + `portfolio_gates`: ZERO worlds certified
(FPR 0.000; the "rule of three" bounds the true certified share at
roughly 3% with 95% confidence — the run honestly reports "no world
certified" without claiming the FPR is exactly zero).  Gate anatomy
on noise: `deflated_sharpe` fired in 100/100 worlds,
`side_concentration` in 100/100, but `drawdown_floor` in only 95/100
— five percent of noise worlds random-walked without ever falling
15 R below a peak, and were still caught by the other two gates.  A
certified world would have been the portfolio layer's
false-certification finding, printed as `>>> FPR FINDING` and
recorded, not hidden.

**Phase B — momentum (positive control).**  The honest recorded
outcome is `DEAD(deflated_sharpe)`, and the preregistration said
before the run that this would be the machinery WORKING, not
failing.  The chain of evidence: (1) the construction audit
(printed before the verdict) measures the IDEALIZED no-stop
long-short on the same scores at -0.74%/yr, Sharpe about -0.11 —
the 2005-2015 momentum golden decade is unavailable below the XLRE
inception, so the engine book expresses a signal with no measured
sample edge; (2) the engine book additionally pays the overnight
gap into momentum entries (stop/target levels are placed from the
decision close; winners gap away from them) and 2 bps round-trip
friction, so its -29.33 R final equity is consistent with a ~zero
underlying edge, not with a bug — the dataset is balanced (291
long / 291 short), fully resolved (0 unresolved trades) and
admission behaves (39 correlation rejects are same-instrument
re-entries at |rho| = 1, 8 capacity rejects are riders holding
slots).  What the phase calibrates: the gates fire on the RIGHT
quantities for the right reasons — a book with ~zero per-bar EV
drift scores DSR 0.107 against a 0.95 floor and is refused before
its drawdown even matters.

**Phase C — noise through the full machinery (negative control).
The instructive one.**  The single seeded noise draw ran to
+43.21 R and PASSED both edge-sensitive gates: deflated Sharpe
0.9828 (floor 0.95) and drawdown -12.66 R (floor -15).  It was
refused ONLY by `side_concentration` (1.0000 > 0.80).  Three
calibration lessons, all recorded: (1) at `n_trials=1` the DSR bar
is passable by luck — by construction it certifies about 5% of
pure-noise EV streams (a z-statistic at the 1.645 bar), and this
world is that tail event; (2) the edge-independent structural gate
carried the kill — single-gate trust in DSR would have certified
noise here; (3) this is exactly why the FPR measurement is
Monte-Carlo over worlds (Phase A), not one draw.  Note the
asymmetry with Phase A: this book ran WITH the correlation
admission (122 correlation rejects — the filter blocking
same-instrument re-entries), which thinned and re-shaped the book
relative to Phase A's worlds; both constructions were refused.

**Cross-phase calibration observation.**  `side_concentration`
fired in EVERY phase and world at these thresholds (value 1.0000):
with 6 slots and partial-cohort admission (positions riding their
21-bar cap while the next cohort's signals are decided), some bar
always ends up single-side.  On this book shape the gate is a
structural flag, not an edge discriminator — a threshold-
calibration observation for anyone reusing this configuration,
recorded here rather than tuned away.

**Prereg discipline.**  Each phase's frozen preregistration is
printed (and hash-stamped) before that phase's run; the FPR number
is never pinned — `tests/test_real_data_example.py` pins the
STRUCTURE (checksums, clean exit through all three phases,
well-formed three-gate reports on the null).

## Reproduce

```bash
uv sync
uv run python examples/real_data/run_portfolio_controls.py --quick   # CI preview, ~8s
uv run python examples/real_data/run_portfolio_controls.py           # full contract, ~20s
```

Snapshots: `data/<TICKER>.csv` + `.sha256` sidecars, refreshable via
`uv run python examples/real_data/download_data.py` (Stooq ->
Yahoo fallback, strict per-ticker verification; see README.md for
provenance and rights).
