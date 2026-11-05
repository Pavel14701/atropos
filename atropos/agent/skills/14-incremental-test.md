---
name: 14-incremental-test
description: The decisive incremental test: t of the signal's own coefficient on top of frozen controls (stage 4/7).
when_to_use: The mandatory screen start together with the nulls; after a raw t >= 2 from 13-signal-test.
version: 1.0.0
---


> Note: log/registrar excerpts below quote the closed donor corpus (not shipped). The workspace layout itself is yours: scaffold with `atropos init` (files are generated, never overwritten).
# SKILL 4: INCREMENTAL TEST — the decisive test (Stage 4 of 7, THE DECISIVE KILLER)

> Self-contained protocol: the agent needs only this file + data.
> Taxonomy: FRAMEWORK.md (repo root) §3 LAW 2, §4 (INCREMENTAL-DEATH);
> teardown: CASES_AND_FLAGS.md (repo root) CASE 1 (CVD).

## PURPOSE

Does the signal add t > 2 with its OWN coefficient on top of the
known controls. The raw effect proves nothing: known features absorb
the lift of the "new" signal — it turns out to be a noisy proxy of
someone else's feature. The pipeline's main killer; since 2026-11-01
the incremental is the MANDATORY start of the screen (the wave-3
lesson).

## INPUTS

1. A raw signal with t ≥ 2 from 13-signal-test (raw beauty is NOT a
   survival obligation).
2. A panel: forward_return + the signal + the full control columns.

## PROCEDURE

1. **FREEZE CONTROL LIST (frozen before the run).** House minimum of
   known controls:
   - momentum: mom7 / mom30 (or the candidate's horizon);
   - OB features (for microstructure candidates): strength /
     zone_width / retest_delay;
   - vol level: iv_pct / rv_pct;
   - |BTC-move| (the confounder of conditioning reads);
   - year dummies (calendar control — see 16-walkforward-calendar).
2. **REGRESSION.** `forward_return ~ known_controls + new_signal`,
   SE: HC0 / Newey-West (lag ≥ L) / clustering by dates-months.
   Pooled with instrument fixed effects OR per-instrument — one of
   the two, from the prereg.
3. **READ OUT.** t(new_signal), delta-R², the beta sign; next to it
   the t without controls (raw) to compare absorption.
4. **PER-SYMBOL SPLIT.** The coefficient per symbol: tearing in
   different directions (BTC −0.41 / ETH +0.78 / SOL +0.94 for CVD) =
   the "signal" is a noisy proxy.
5. **ABSORPTION REPORT.** Which controls absorbed the lift: for CVD
   these are zone_width_atr (t −9.6) and retest_delay (t −9.2).
   Caution: a raw regression t of ±10-14 = the |BTC-move| confounder,
   not the signal.

## KILL CRITERIA

- **t(new_signal) < 2 after controls = REPACKAGING-DEATH**
  (INCREMENTAL-DEATH) — the effect is real but it is a rediscovery of
  a known feature; regardless of the raw statistics and the battery
  result.
- ΔR² ≈ 0 with a formally significant t — "letter PASS / filling
  DEAD" (GLM v2: SPOT30 t +3.10 formally, incremental t +0.73).
- A family closes by strikes: 3 kill mechanics = permanent close
  (GLM: repackaging → calendar → conjunction).

**RULE (red thread):** "beautiful raw numbers mean nothing if
incremental kills them".

## PASS CRITERIA

t(new_signal) ≥ 2 after the FULL frozen control list AND ΔR² ≥ the
prereg threshold (reference PREREG_OFI §7: ΔR² ≥ 0.003).
→ Next: 15-robustness-battery (the robustness battery). A passer:
POC-avoidance — t −16.55 AFTER the OB controls, orthogonal to OB
(|ρ| ≤ 0.143) — the first candidate in 9 rounds to pass the
incremental test.

## REAL EXAMPLE (from the graveyard — a 3-strike family closure)

**The CVD/GLM family:** the raw premise **t +5.50** → the incremental
after OB controls **t −0.20** — the entire lift absorbed by
retest_delay and zone_width. Same mechanism: CVD raw +0.217R, battery
ROBUST 6/6, placebo percentile 100 — but slope_z t +1.05 after
controls; the BTC.D proxy t 3.52 → 0.16; COT cot_z t −0.99 after
mom30; SVI-slope ΔR² ≈ 0.001 on top of ATM IV. Each looked alive
until a single regression; the regression cost hours, not years
(`runs/cvd_ob_independence.log`, `runs/glm_v2_postmortem.log`).

## COMMON MISTAKES

- Skipping the incremental because "the battery is ROBUST": the
  battery checks the effect is not noise; only the controls check it
  is not SOMEONE ELSE'S (CASE 1).
- A control list picked after the data — the control set must be
  frozen.
- Claiming victory on a formal t > 2 in the raw spec while the
  incremental is < 2 (GLM v2: "letter PASS, filling DEAD"); forgetting
  ΔR² — a significant t with no gain in explained variance = a noisy
  duplicate.
- Reading REPACKAGING as "almost alive": the family closes;
  successors are admitted with prior ≤ 20% and one dated run
  (PREREG_GLM_SPOT §8).
