---
name: 18-read-portfolio-results
description: Read portfolio-level results from the backtest pipeline
when_to_use: after running run_portfolio or run_portfolio_replay
version: 0.1.0
---

# SKILL 18: READ PORTFOLIO RESULTS — the multi-slot book and its gates
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Portfolio-level readout of a tradable candidate. Public API:
> `atropos.portfolio`: `run_portfolio_replay(signals_df, prices, *,
> slots, policy, corr_cap, ...) -> PortfolioResult` (the N-slot
> replay: ledgers, exposure, event equity) and `run_portfolio(...)`
> — the same replay + `portfolio_gates` + verdict in one call (the
> function a user runs). The signals fed in are the ranked,
> rank-normalized candidates of skill `17`; this skill reads the
> BOOK they build.

## WHEN TO APPLY / NOT APPLY

APPLY after `run_portfolio` or `run_portfolio_replay` to read the
portfolio result: N concurrent slots over a real OHLC panel, the
admission ledgers, the event equity curve, the per-bar exposure and
the three portfolio gates.

NOT for:

- the single-instrument backtest (isolated replay, one-slot state
  machine, admission + per-trade metrics) — skill `17-backtest-pipeline`
  covers that readout;
- tuning `slots` / `policy` / gate floors after a look at the result
  — result shopping; every number is frozen in the prereg (skill `01`);
- a first verdict: the portfolio layer never substitutes for the kill
  protocol; no SURVIVES from skills `02`/`04` → no portfolio run.

## READING A PortfolioResult (one dict, fixed keys)

`run_portfolio_replay` and `run_portfolio` return the same
`PortfolioResult` TypedDict; the gated run only fills two keys.

- `accepted` / `rejected` / `replaced` — ledger rows in admission
  order; every rejected/replaced row carries a `reason`
  (`"capacity"`, `"inferior_score"`, `"correlation_cap"`,
  `"cooldown"`). Accepted rows add `slot`, `fill_idx`, `exit_idx`,
  `r_pess`, `credit_bar`, `credit_r`, `replaced`,
  `force_close_r` (displaced only).
- `equity_curve` — EVENT-based cumulative R: one point per accepted
  trade, credited at the trade's exit bar (`credit_bar`; an
  unresolved trade credits 0 at the last bar). This is NOT a daily
  mark-to-market curve.
- `exposure` — per-bar series `gross` (unit positions held), `net`
  (longs minus shorts) and `concurrent` (== gross: each position
  enters at exactly one risk unit, so both are bounded by `slots`).
- `summary` — `n_accepted` / `n_rejected` / `n_replaced` /
  `n_correlation_rejects` / `n_unresolved`, `ev` (final event
  equity), `dd` (event-basis max drawdown), `concurrent_max`.
- `per_slot` — attribution rows (`slot`, `trades`, `ev_r`): which
  slot carried the book.
- `gates` — EMPTY from the raw replay (then `portfolio_verdict` is
  `"not_evaluated"`); populated by `run_portfolio` /
  `portfolio_gates` in fixed order: `deflated_sharpe`,
  `drawdown_floor`, `side_concentration`. Each gate is a
  `GateVerdict`: `gate`, `triggered`, `value` (NaN when the gate had
  no data), `floor`, `reason` — read `value` vs `floor`, not just
  the boolean.

```python
from atropos.portfolio import run_portfolio

book = run_portfolio(signals_df, prices, slots=4, policy="fcfs")
print(book["portfolio_verdict"], book["summary"]["n_accepted"])
for gate in book["gates"]:
    print(gate["gate"], gate["value"], "vs floor", gate["floor"])
```

## PORTFOLIO VERDICTS (3) — WHAT EACH MEANS

| Verdict | Meaning | Agent action |
|---|---|---|
| `SURVIVES` | all three gates passed, >= `min_trades` credited trades | report the book as gate-conditional: a portfolio that passed TODAY's floors, not a guarantee |
| `DEAD(<gate>)` | the first triggered gate of the three | spell out the gate from the parentheses and its `value` vs `floor` from `gates`; do not re-run with softer floors |
| `UNDERPOWERED` | fewer than `min_trades` credited trades (the curve's length IS the trade count) | NOT a death: undecided. More data / a longer window; re-running on the same snapshot is forbidden |

The three gates, fixed order:

| Gate | Check |
|---|---|
| `deflated_sharpe` | per-period portfolio EV stream (R credited per `ev_period_bars` bucket) through the shared DSR formula; FAIL-CLOSED on a non-finite DSR (too few observations cannot count as verified) |
| `drawdown_floor` | event-basis max drawdown of `equity_curve` vs `dd_floor`; REACHING the floor breaches it |
| `side_concentration` | max same-side (long/short) share of concurrent positions vs the cap; skipped honestly (value NaN) without the exposure series |

## TRAPS

- The equity curve is EVENT-based (R credited at each trade's
  exit), not a daily mark-to-market series: path P&L between
  credits is invisible by construction, and the curve's length is
  the credited-trade count, not the bar count. Never annualize
  Sharpe from it as if it were a daily series.
- The drawdown runs against a LEADING-ZERO peak (the curve is
  anchored at 0.0 before the first credit): a first-trade loss IS a
  drawdown from the start. This is deliberately not flattered; do
  not "fix" it into a prettier number.
- The correlation cap ADMITS on inconclusive pairs: below
  `corr_min_overlap` proxy bars (or a degenerate proxy) the verdict
  is "insufficient data", not "uncorrelated" — the candidate
  enters. Small overlap is NOT evidence of low correlation; report
  `n_correlation_rejects` together with the overlap setting.
- REPLACE-high vs REPLACE-low credit a DISPLACED trade differently:
  `replace_low` credits 0 (a mid-flight close is assumed to give
  back), `replace_high` credits its full isolated `r_pess`. The
  honest partial-hold P&L the displaced trade actually realized
  sits in its `force_close_r` field and NEVER overwrites the policy
  credit — read it, do not add it into the equity a second time.
- `r_proxy` must be the instrument's GROWING return series from ONE
  common anchor through the row's decision bar, and `score` must be
  within-group rank-normalized (skill `17`) — raw cross-model
  probabilities do not order a book.

## OUTPUT CONTRACT

Every portfolio report must state:

- `portfolio_verdict` + each gate's `value` vs `floor` (or "gate
  skipped: no data" where the series was missing);
- `slots`, `policy`, and the ledger counts (`n_accepted`,
  `n_rejected`, `n_replaced`, `n_correlation_rejects`,
  `n_unresolved`);
- the final event equity `ev` and `dd` in R, marked as EVENT-basis;
- an exposure summary (`concurrent_max`, gross/net peaks) — the
  full per-bar series only on request;
- data provenance: real snapshot id, or SYNTHETIC + seed;
- and the line: **SURVIVES ≠ tradable.**
