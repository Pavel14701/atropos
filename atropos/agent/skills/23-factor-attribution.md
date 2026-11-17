---
name: 23-factor-attribution
description: Attribute a strategy's P&L to factor proxies (MKT / BAB-proxy) with atropos.factor_attribution — FACTOR_EXPOSURE vs ALPHA_CANDIDATE routing.
when_to_use: When a factor-family book (beta/correlation construction) hits CS_NO_EDGE or any book's P&L needs decomposition into known premia before an alpha claim.
version: 0.1.0
---

# SKILL 23: FACTOR ATTRIBUTION — exposure is a verdict, not a failure
> Code canonical. Single home: `atropos.factor_attribution` — this
> skill references it, never re-enumerates it.  Law: KNOWN-ISSUES §18
> (factor premia are the raw material).  Routing: skill 19
> (`cross_sectional_factor` plan forbids trade_battery).

> The 2026-11 BAB/BAC pause context: the books were paused on
> CS_NO_EDGE read through RANKING instruments — an instrument
> MISMATCH for factor constructions (PLAN_REGISTRY:
> `cross_sectional_factor` primary = stage-7 CS + factor
> attribution; trade_battery is FORBIDDEN — a factor construction is
> not trade-level). The native instrument, `factor_regression`, is
> now implemented. Re-read the pause through it before writing the
> tombstone.

## 1. Build the proxies, run the regression

- `construct_factor_proxies(panel)` — MKT (equal-weight cross-section
  return) and `bab_proxy` (low-beta-minus-high-beta, trailing 60d
  beta sorts, rebalanced on the sort, not the P&L).
- `factor_regression(returns, factors)` — the book's return stream
  regressed on the proxies: betas, R², and the alpha t-stat
  (`alpha_t`) on the residual.

## 2. Route on the verdict

`factor_verdict` returns exactly one of:

| Verdict | Rule | Meaning / action |
|---|---|---|
| `FACTOR_EXPOSURE` | R² ≥ 0.7 and \|alpha_t\| < 2 | the book IS the factor. Deploy as a conscious exposure choice (capacity, cost, drawdown of the KNOWN premium) — never claim alpha. |
| `ALPHA_CANDIDATE` | \|alpha_t\| ≥ 2 | the real finding: returns the factor proxies do NOT explain. Re-test out-of-sample before any claim — this is a candidate, not a pass. |
| `RESIDUAL_WEAK` | neither | R² low AND alpha weak: noise, not a factor book and not alpha. Kill or redesign. |

## 3. The §18 link

Factor premia are the raw material (KNOWN-ISSUES §18): a
factor-exposure book earns existence by CONSCIOUS exposure choice —
cheaper/cleaner/capacity-richer than the direct implementation —
not by alpha claims. The honest §18 comparison for a FACTOR_EXPOSURE
book: net EV and turnover cost per unit of premium captured vs the
direct factor replication, same universe, same cost model.

## NEVER

- Read CS_NO_EDGE as death for a factor book without running factor
  attribution first (skill 19's NEVER, repeated because the 2026-11
  pause happened anyway).
- Claim alpha on an R²-heavy book — high R² with weak alpha_t IS
  the factor-exposure finding.
- Fabricate a size proxy without market-cap data — the proxy set is
  MKT and BAB; add a factor only when its raw input exists in the
  panel.
- Promote ALPHA_CANDIDATE on the in-sample regression alone — the
  OOS re-test is the verdict, not the flag.
