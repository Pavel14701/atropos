---
name: 13-signal-test
description: Raw signal test: EV, t on cluster-SE, direction vs prereg, per-year/per-asset tables, power (stage 3/7).
when_to_use: After the premise check; together with the nulls and the incremental test at the start of the screen.
version: 1.0.0
---


> Note: log/registrar excerpts below quote the closed donor corpus (not shipped). The workspace layout itself is yours: scaffold with `atropos init` (files are generated, never overwritten).
# SKILL 3: SIGNAL TEST — the raw signal test protocol (Stage 3 of 7)
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Self-contained protocol: the agent needs only this file + data.
> Full taxonomy: FRAMEWORK.md (repo root) §4 (NULL-DEATH,
> ANTI-SIGNAL-DEATH); screen template: TEMPLATES.md (repo root)
> template 2 (§4 — the signal test).

## PURPOSE

Is there a raw effect, in which direction (direction vs prereg), and
does it hold across years (per-year table). An honest raw picture
BEFORE any controls: the controls (14-incremental-test) answer "is
this effect someone else's"; the raw test answers "is there anything
at all to control" and "is there enough power".

## INPUTS

1. A candidate with a premise check from 11-premise-check; the frozen
   construction from the prereg (cadence, horizon L,
   overlapping/non-overlapping forward returns).
2. A data panel with exact paths; the book's convention (costs,
   timing without look-ahead) declared before the run.

## PROCEDURE (each step = a measurable log output)

1. **RAW EFFECT SIZE.** EV per trade/day (bp or R) + hit rate; n
   observations. No eyeball metrics — only numbers in the log.
2. **T-STAT (cluster-SE).** The t-statistic with clustered/Newey-West
   SE (lag ≥ L for overlapping windows) NEXT TO the naive i.i.d. SE
   for comparison. Naive t on clustered data is inflated.
3. **DIRECTION vs PREREG.** A sign against the frozen direction is a
   failure regardless of the t magnitude. The abs variant goes to the
   log only, never to the kill.
4. **PER-YEAR TABLE (mandatory).** Year: n, EV, t; sign score k/N
   years; EV concentration across years (flag 4: > ~70% in one year =
   an artifact prior).
5. **PER-ASSET TABLE.** Positive assets k/N; an effect on one symbol
   = not a market effect (see LOO, 15-robustness-battery).
6. **POWER CALCULATION (recorded in the prereg before the run).**
   N = (2·sigma/EV)^2 ~ observations for t ≈ 2 at the expected effect
   EV (z_0.975 + z_0.8 ≈ 2.8; the exact formula depends on the chosen
   test). N < N_min → status UNPOWERED/UNDECIDABLE — an honest
   outcome, not a spin.
7. **SANITY: "BOTH LEGS POSITIVE" = A BUG.** The long and short arms
   positive at once — look-ahead/alignment, not an edge (the lowcaps
   lesson: "+3.2.. +4.5%/day on both legs 23/23" = a caught bug, not
   a find).

## KILL CRITERIA

- **t < 2** — no statistical signal (unpowered or zero).
- **The sign is opposite to the prereg direction** — the book is
  killed, the inverted version is never preregged (INVERTED-DEATH at
  the sign stage).
- **Per-year sign flips > 50%** — the sign floats, the effect is
  regime-driven.
- **ANTI-SIGNAL**: both legs positive = a panel drift artifact
  (ANTI-SIGNAL-DEATH) — or an alignment bug; either way = stop.

## PASS CRITERIA

t ≥ 2 on the clustered SE; the sign matches the prereg; the sign is
stable in > 50% of years (better: all years); the effect does not sit
in a single year/asset; N ≥ N_min from the prereg.
→ Next: 14-incremental-test (the incremental — the decisive test).

## REAL EXAMPLE (from the graveyard)

**XS reversal alt/BTC (tombstone 60/105):** all 9 configs
net-negative, BOTH legs positive (legs +37..+123%/year) = a drift
artifact; pooled ev −0.44% (t −3.19), 7/8 years negative, 1/9 assets
> 0 (`runs/xs_reversion_screen.log`). The spreads TREND — the
reversal was the anti-signal, the flip = a closed family. **ETF flows
(round-16):** the inflow>2σ bucket gives +50.9 bp/day, but **t 1.32
at n=55** — unpowered: a pretty EV with power below N_min is not a
signal. Nearby: the Unlock Effect — p 0.19-0.40 across all windows,
the direction guessed wrong.

## COMMON MISTAKES

- Celebrating a large EV at n < N_min (ETF flows: +50.9 bp at t 1.32).
- Overlapping forward returns with a naive t — the SE is understated
  by multiples.
- Skipping the per-year split: 2021-22 concentration is the main
  decay/artifact marker (BAC: 76% of EV from 2021-22, 2025-26
  −11%/year).
- Reading "both legs +" as a double edge — it is a bug or a drift
  mirror.
- Hunting for rescue with another window/threshold after a kill — one
  run, one verdict; "but it is alive at another window" = mining.
