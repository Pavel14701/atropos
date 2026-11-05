---
name: 15-robustness-battery
description: A battery of 7 robustness tests: threshold, window, thirds, LOO, placebo, de-clustering, sign ladder (stage 5/7).
when_to_use: After the incremental test passes; never before it.
version: 1.0.0
---


> Note: log/registrar excerpts below quote the closed donor corpus (not shipped). The workspace layout itself is yours: scaffold with `atropos init` (files are generated, never overwritten).
# SKILL 5: ROBUSTNESS BATTERY — the 7-test battery (Stage 5 of 7)

> Self-contained protocol: the agent needs only this file + data.
> Taxonomy: FRAMEWORK.md (repo root) §4 (ROBUSTNESS/FRAGILE-DEATH);
> template: TEMPLATES.md (repo root) template 3; thresholds:
> PREREG_OFI §7 + candidate_shelf_battery.

## PURPOSE

Does the effect survive independent perturbations: threshold, window,
subperiod, panel composition, placebo, clustering, the sign ladder.
An effect living at a single point of parameter space or in a single
period is an artifact. No gates: the verdict comes from measured
numbers, not from a narrative. Run AFTER the incremental
(14-incremental-test), not instead of it.

## INPUTS

1. A candidate that passed 14-incremental-test (incremental t ≥ 2),
   with the original specification (base threshold/window).
2. Anchor [0]: a bit-for-bit reproduction of the original split
   BEFORE the tests.

## PROCEDURE — 7 tests, thresholds fixed (PREREG_OFI §7)

1. **THRESHOLD SENSITIVITY:** threshold {0.5x, 1x, 1.5x, 2x} of the
   base. The effect changing > 2x under a threshold shift = DEAD (a
   one-parameter crack).
2. **WINDOW SENSITIVITY:** window {0.5x, 1x, 2x}. Alive only at the
   base window = window fitting (Amihud 60d: Sharpe 0.03).
3. **SUBPERIOD THIRDS:** thirds of the sample (or years). Alive (sign
   + CI) in ALL three = ROBUST; alive in one third = a regime
   artifact (Amihud: +83 / +55 / −38; BAC: 76% of EV from 2021-22).
4. **LEAVE-ONE-OUT:** drop one symbol/asset at a time. Dropping the
   "carrier" kills it = not a market effect (Friday-13th: min z +1.74
   — the entire effect = one day, 2021-08-13).
5. **PLACEBO NULL:** 200 runs, seed pinned; the signal×outcome
   alignment is destroyed. The real must land above the 95th
   percentile.
6. **DE-CLUSTERING:** drop the top 1% of observations + cluster-SE by
   dates/months next to the naive t (the naive one is inflated; GLM
   v2: t 3.09 → 2.70).
7. **SIGN STABILITY:** a monotone quantile ladder q1 < q2 < q3 < q4.
   A zigzag = only the extreme bucket carries the effect = noise
   (Ex-kurt: strictly decreasing blocks 2/27 — a tail artifact).

**VERDICT ASSEMBLY:** a table [1]-[7] of True/False from measured
numbers; FINAL: ROBUST | NOT ROBUST — an enumeration, no explanatory
phrases.

## KILL CRITERIA

- **≥ 2 broken tests = ROBUSTNESS-DEATH** (the default).
- **≥ 1 broken test AND a systematic pattern in the fails** (all fine
  except one period/symbol) = FRAGILE-DEATH / a regime artifact.
- Alive in exactly one source configuration = FRAGILE (ETF flows:
  walk-forward decay 0.46x, the "conditional YES" withdrawn).

## PASS CRITERIA

**≥ 5/7 tests pass AND no systematic pattern in the fails** (single
fails are explained by measured numbers, not a narrative). The
threshold gates of the live prereg (PREREG_OFI §7) — the sign stable
across terciles, invariance to the normalizer — are miniatures of
tests [1] and [7].

## REAL EXAMPLE (shelf battery — three verdicts from one battery)

`runs/candidate_shelf_battery.log` (anchor [0]: all three rebuilt
MATCH):
- **BAB — ROBUST:** thirds +107/+23/+19 (all +), 90d Sharpe 1.30,
  monthly 1.46 beats-all, LOO +35/+41/+39, placebo p0.5 [LOW=good] →
  survivor (PREREG_BAB_TIER2).
- **BAC — CONCENTRATED:** pctl 99, the mechanics healthy (LOO clean,
  monthly alive) — BUT 2021-22 carries 76% of EV, 2025-26 −11%/year →
  not a candidate, only a conditional with an acknowledged decay.
- **Amihud — FRAGILE:** base +33.8%/year (pctl 98!), thirds
  +83/+55/−38, 60d Sharpe 0.03, monthly −9.4 → FRAGILE-DEATH: the
  placebo percentile does not save you from non-monotone thirds.

## COMMON MISTAKES

- The battery BEFORE the incremental: CVD passed 6/6 and died on the
  controls — the order is fixed (the incremental first, the battery
  after).
- Rescuing a candidate with a narrative: 2 fails + "but that is
  explained by such-and-such" = NOT ROBUST; the explanation goes into
  the post-mortem (template 4), not into the verdict.
- No anchor [0]: a battery on a non-reproduced spec measures air.
  Counting a subset of tests: de-clustering [6] and the ladder [7]
  are the most frequent killers of tail artifacts (Ex-kurt, cascades).
- A subperiod fail "will survive walk-forward" — no: thirds and WF
  catch the same thing; a thirds fail = the sentence.
