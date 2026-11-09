---
name: 21-units-and-identities
description: The units law (1R = 1% of equity), the half-normal identity for abs()-structures, additive-vs-multiplicative accounting, and metrics that never gate.
when_to_use: Before quoting ANY performance number — Sharpe, drawdown, CAGR — and before interpreting a metrics block on a report.
version: 0.1.0
---

# SKILL 21: UNITS AND IDENTITIES — read numbers without lying
> Law: KNOWN-ISSUES §16 (single source of the metrics-never-gate rule — this
> skill references it, never re-enumerates it).  Code canonical:
> `atropos.evaluation_metrics`.

## 1. The units law

SYSTEM-9 books P&L in R with **1R = 1% of equity** (`RISK_PCT = 0.01`).
R values are PERCENTS of equity, not fractions.  Consequences:

- Equity metrics (compounding, drawdown, CAGR, Calmar) MUST compound
  `R × 0.01`.  Feeding raw R into `1 + r` inflates every day ×100:
  a normal −1R stop day reads as −100% equity (C2's maxDD = None was
  the fail-honest guard), a carry stream reads as −99.9% ghosts.
- Sharpe and Sortino are SCALE-INVARIANT (ratios) — units do not
  change them.  Only multiplicative metrics are units-sensitive.

## 2. The half-normal identity (carry / abs()-structures)

A strategy that costlessly collects |noise| (carry state machines
that flip on sign change) books a HALF-NORMAL stream:

    E[|r|] / σ = √(2/π) ≈ 0.798  (daily, for ANY zero-mean noise)
    annualized: √(2/π × 252) ≈ 12.66

This is **E[|r|] annualized — NOT a Sharpe** (Sharpe uses the full σ;
the half-normal identity uses σ·√(2/π)).  Measured: E:BTC daily SR
0.7963 vs identity 0.7979 → annualized 12.64.  A Sharpe ≈ 12.66 on a
carry/flip book is the SIGNATURE of a costless abs() structure, not
an edge.  Documentation only — no code module computes it.

## 3. Additive vs multiplicative

- Additive ΣR is the honest wealth measure for books with
  concurrent positions and fixed-fraction risk: each trade risks 1%
  of CURRENT equity, and summing R across concurrent trades then
  compounding the daily aggregate double-counts reinvestment.
- Multiplicative compounding is valid only for a single sequential
  fractional-return stream.  Daily-aggregate R books: quote ΣR and
  the trade battery, not compounded equity.
- Zeros are compounding-neutral — filtering them changes means/stds,
  not compounded equity.

## 4. Metrics never gate (§16)

A metric never overrides a verdict.  Sharpe 12.6 with verdict
ARTIFACT is the system working: the kill-protocol answers "can this
be killed", not "how good is the return".  Read metrics blocks as
read-outs for the principal; the paired bootstrap, the battery, and
the parity report decide.

## NEVER

- Quote an annualized Sharpe on a carry/flip book without the
  half-normal check.
- Compound R values without the ×0.01 units contract.
- Report max_drawdown from a multiplicative equity on an R stream
  with concurrent positions.
- Let ANY metric flip a verdict token.
