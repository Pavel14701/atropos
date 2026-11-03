# Glossary

The vocabulary of the kill chain. Each term: what it means here, and a
mini-example of its use. For the full death taxonomy see `FRAMEWORK.md` §4.

## Lifecycle

**Preregistration (prereg)** — a hypothesis document frozen *before* the
data is touched: prediction, construction, nulls, numeric kill criteria,
power calc, spec hash. Generated and validated from a params dict by
`atropos.prereg_generator.generate_prereg`. Example: a prereg that says
"|t| after momentum controls < 2.0 => KILLED" is committed on Monday, so
Tuesday's regression cannot move the goalposts.

**Kill criterion** — a directional numeric rule written in the prereg
(`metric comparison threshold`, e.g. `IC < 0.05`), never edited after the
first data touch. Example: "net-of-cost Sharpe floor 0.0 breached in the
Monte Carlo p5 tail" is a criterion; "looks weak" is not.

**COD (cause of death)** — the recorded reason a hypothesis died: which
stage killed it and why. Example: `killed_by_stage=1, COD: repackaging —
adds nothing beyond known momentum`. The pipeline's `kill_reason` field.

**Tombstone** — the append-only record of a finished hypothesis: verdict,
COD, evidence, resurrection conditions. Never edited or deleted. Example:
`graveyard/H-0007.md` recording "DEAD — battery catastrophic failure on
window sensitivity, w=15 inverts the spread".

**Graveyard** — the collection of all tombstones; the first thing you
check before proposing anything (a structurally identical KILLED
hypothesis is not re-run without a new mechanism). Example: a graveyard
check stops you from re-testing "overnight gap reversal" that died in
2024-03 with the same prediction and kill criterion.

**Resurrection conditions** — the part of a tombstone that states under
what new facts a dead hypothesis may be reopened. Example: "reopened only
if a new venue cuts costs below 2 bps" for a COST_DEATH tombstone.

## Verdicts

**SURVIVED / KILLED / INCONCLUSIVE** — the three abstract outcomes. The
pipeline emits concrete names: `SURVIVES`, `SURVIVES_WITH_WARNINGS`
(the survived family), seven kill verdicts (`REPACKAGING`, `ARTIFACT`,
`DEAD`, `NO_OOS_EDGE` x2 stages, `COST_DEATH`, `RISK`, `CS_NO_EDGE`),
and `INCONCLUSIVE` when the data cannot decide. Example: SURVIVES means
"earned the right to a live track record", not "is true".

**Null control** — a test that destroys the hypothesis if the effect
survives only in the un-manipulated world: permutation, placebo,
sub-periods, calendar, known-features. Example: shuffling event dates
kills an "event-driven" edge whose whole Sharpe came from three days.

**Repackaging** — the "new" signal is a known factor (e.g. momentum) in
disguise; stage 1/2 verdict when known-features controls absorb it.
Example: signal = 7-day momentum + noise regresses onto observed momentum
with R^2 ~ 1 => REPACKAGING.

**Cost death** — the gross edge is real but net-of-cost returns do not
clear the floor; transaction costs ate the edge (stage 5, `COST_DEATH`).
Example: 40 bps round-trip on a 6 bps mean spread => gross SR 1.2,
net SR < 0.

**CS_NO_EDGE** — the cross-sectional death: per-date IC, net quantile
spread, or quantile monotonicity fails on a panel (stage 7). Example:
top-minus-bottom quintile spread is +12 bps gross but -3 bps net of
costs and non-monotone across quintiles.

## Methods

**Walk-forward + purge gap** — out-of-sample evaluation that rolls a
train/test window through time, with rows purged at the seam so the
label horizon cannot leak from train into test. Example: train 180 days,
purge 1 day (one-period forward return), test 30 days, step 30.

**DSR / PSR** — deflated and probabilistic Sharpe ratios (Bailey & Lopez
de Prado): shrink a Sharpe ratio for the number of trials tried (DSR) or
relative to a benchmark SR (PSR; DSR with one trial). Example: SR 2.0
looks great until DSR deflates it for the 100 backtests that produced it.

**Block bootstrap** — resampling *contiguous blocks* of a return series
(preserving autocorrelation) instead of iid rows, to build honest
confidence intervals. Example: 40-day blocks resampled 1000 times give
the Sharpe CI that an iid bootstrap would understate.

## Doctrine

**Defense-in-depth** — no single test decides; seven stages of
independent attacks must all miss before anything SURVIVES. Example: a
signal can pass null controls and still die at stage 5 on the drawdown
tail.

**INCONCLUSIVE vs FAIL** — FAIL is evidence against the hypothesis
(a stage attacked and won); INCONCLUSIVE means the data cannot decide
(too short, snapshot does not cover it) and reopening on a new snapshot
is allowed. Example: a 90-day live track is INCONCLUSIVE at
`live_verdict` (shorter than the default 252-observation decision
point), not a pass and not a kill.
