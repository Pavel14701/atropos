---
name: 19-instrument-routing
description: Route a strategy family to its correct instruments via atropos.instrument_map.PLAN_REGISTRY before running anything.
when_to_use: Before any pipeline or battery run on a new leg/strategy — decide which instruments are primary, informational, or FORBIDDEN for the family.
version: 0.1.0
---

# SKILL 19: INSTRUMENT ROUTING — the right battery for the right family
> Code canonical. Single home: `atropos.instrument_map.PLAN_REGISTRY` — this skill
> references it, never re-enumerates it. Law: KNOWN-ISSUES §17.

> The 2026-11 lesson set: leg A (rule-based) crashed the ML stage with
> CONSTANT_PREDICTIONS; C'' (per-trade) was battery-killed on a daily
> threshold rule that does not exist in its spec; E (carry) showed
> Sharpe 12.66 — the abs() identity, not an edge; BAB/BAC (factor
> books) were CS_NO_EDGE — a factor construction, not a ranking
> signal. Every one of these was an instrument mismatch, not a
> strategy fact. Routing makes them mechanically unrepeatable.

## 1. Declare the family FIRST

`strategy_family` ∈ {`rule_based_tactical`, `per_trade_tactical`,
`ml_predictive`, `cross_sectional_factor`, `carry`, `options`}.

Declare it in the prereg / hypothesis config (`strategy_family` key)
or pass it explicitly to `run_full_pipeline`. Declared → **enforce
mode** (forbidden stages raise ValueError before running). Not
declared → **audit mode** (the report carries `instrument_audit`
with violations flagged; nothing blocks — use only for legacy
replays). Explicit kwarg overrides config; a conflict between the
two is a ValueError, not a silent override.

## 2. Read the plan, then run only the primary instruments

`atropos.instrument_map.route(family)` returns the plan: `primary`
(must run), `informational` (may run, never gates), `forbidden`
(tool + stage + reason — enforce blocks stage-matched entries).
Shapes:

- `rule_based_tactical` (price rules: crosses, trails) → stages 1-3 +
  trade battery. ML stage FORBIDDEN (no predictive feature stream —
  CONSTANT_PREDICTIONS by construction).
- `per_trade_tactical` (TP/stop/MTM per-trade books) → trade battery
  + payoff stats ONLY. Daily-aggregate instruments FORBIDDEN (§12).
- `carry` (funding/settlement books) → risk envelope + funding
  alignment. Book-momentum FORBIDDEN (funding is settlement, not
  momentum). Watch the half-normal identity — see skill 21.
- `cross_sectional_factor` (beta/correlation books) → stage-7 CS +
  factor attribution. Trade battery FORBIDDEN (factor construction
  is not trade-level). CS_NO_EDGE may be correct — factor premia are
  not alpha; attribute before believing either way.
- `ml_predictive` → the full pipeline; the only family that owns
  stage 4.
- `options` → per-greek decomposition (`pnl_decomposition`); total
  P&L parity hides opposite greek errors.

## 3. Audit before believing

`atropos.instrument_map.audit_report(report, family)` re-judges a
COMPLETED report post-hoc (matches the saved `stages` array against
forbidden stages) — no rerun needed, cross-session. Honest limit:
runners that do not save the stages array return [] (that neither
proves nor disproves compliance).

## NEVER

- Run stage 4 (ML) on a rule-based family "just to see" — the guard
  fires, and the ERROR hides the real stages 1-3 results.
- Judge a per-trade strategy on a daily threshold battery.
- Quote an annualized Sharpe on a carry book without the half-normal
  check (skill 21).
- Call a factor book's CS_NO_EDGE a death verdict without factor
  attribution.
