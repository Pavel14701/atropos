---
name: 22-carry-testing
description: Judge a funding-carry book with its NATIVE instruments (funding alignment + risk envelope + flip-cost economics), not the ranking battery.
when_to_use: When a funding-carry family is proposed, revived, or up for a verdict — and before quoting ANY number off a carry/flip accrual stream.
version: 0.1.0
---

# SKILL 22: CARRY TESTING — judge the rent, price the flips
> Code canonical. Single home: `atropos.carry_instruments` — this
> skill references it, never re-enumerates it.  Law: KNOWN-ISSUES §18
> (the packaging-vs-direct comparison a revival must win).  Routing:
> skill 19 (`carry` plan); identities: skill 21.

> The 2026-11 doctrine recap: carry was killed as a DEPLOYED LEG —
> not because funding fails to persist (it persists: panel IC +0.394,
> t=7.1), but because (a) the TRADES don't clear flip costs
> (per-flip t 0.86-1.46 on n 9-56 — 0.1R maker per flip ate half the
> NEAR carry), and (b) the booked stream is a half-normal rent:
> Sharpe 12.66 = √(2/π × 252), the abs() identity (skill 21), not an
> edge. Persistence was never the question; the flip-cost line was.

## 1. The native instruments

`atropos.carry_instruments` (the `carry` plan's primary set — skill
19; book-momentum is FORBIDDEN, funding is settlement, not momentum):

- `funding_alignment` — the settlement-calendar audit: holds vs the
  funding settlement calendar, accrued-vs-recomputed accrual drift,
  and `flip_cost_share` (flip costs as a share of gross carry
  collected). This is where the 2026 kill actually lived: alignment
  was fine, `flip_cost_share` was fatal.
- `carry_risk_envelope` — the shape of the rent: accrual drawdown,
  losing streaks, and vol-targeting feasibility (can the book be
  sized to a target without destroying the accrual in fees?).

## 2. What a revival requires

A carry revival is NOT a new funding-persistence panel. It is:

1. the native instruments PASSING — `funding_alignment` clean
   (calendar matched, drift bounded, `flip_cost_share` small), the
   risk envelope survivable; and
2. the §18 packaging-vs-direct comparison: the carry overlay must
   beat the plain hold-collect-funding book NET of flip costs, same
   universe, same bars, same cost model (net EV, turnover cost per
   unit of carry collected, capacity, drawdown profile). An overlay
   that cannot beat plain collection is COST_DEATH (§18 guard).

## 3. Reading discipline

- Persistence ≠ edge: funding IC t=7.1 says the rent exists, not
  that a strategy clears costs on top of it.
- The abs() identity (skill 21): a Sharpe ≈ 12.66 on the accrual
  stream is the signature of costless |noise| collection — a
  read-out, never evidence.
- Flip costs are first-order, not a footnote: 0.1R maker per flip
  is a third of a typical daily carry accrual; quote net-of-flip
  numbers or quote nothing.
- The only real question is alpha_t on the RESIDUAL after the rent:
  regress the book's returns on the plain carry collection; the
  leftover is the claim under test.

## NEVER

- Judge carry by annualized Sharpe on the accrual stream (the
  half-normal identity makes it a tautology).
- Ignore or defer flip costs to "execution detail" — they are the
  2026 kill mechanism.
- Treat funding persistence (panel IC) as signal quality.
- Revive carry on a persistence panel alone, without the §18
  packaging-vs-direct comparison.
