# Methodology reference: every claim single-homes

The falsification toolkit's method map. Every statistical method the
docs lean on lives in exactly one function — this table is the map.
When a README, GLOSSARY or how-to makes a claim about a method, the
claim resolves here: the module.function in column 2 is the single
home, and docs must not duplicate its logic. The authoritative verdict
list (the other single-source rule) lives in the
`atropos.ensemble_pipeline` module docstring — docs must not
re-enumerate it either.

## The table

| Method | Single-home (module.function) | Used by |
|---|---|---|
| Deflated Sharpe (Bailey & López de Prado 2014) | `atropos.evaluation_metrics.deflated_sharpe_from_moments` | kill_chain, ml_training, robustness gates |
| Sharpe standard error (Lo 2002) | `atropos.track_record.sharpe_standard_error` | live_verdict prereg divergence |
| Walk-forward purging/embargo (López de Prado) | `atropos.walkforward_framework` | stage-4 ML chain |
| Betting-against-beta factor context (Frazzini-Pedersen 2014) | `atropos.instrument_map` cross_sectional_factor family | BAB/BAC-class books: factor premia ≠ alpha |
| Half-normal identity for abs()-structures | **no code — documentation only** | understanding carry/flip books (prose below) |

## The half-normal identity (documentation-only, on purpose)

For a book whose per-period P&L is an abs()-structure (carry and
flip books: the payoff is symmetric by construction), the expected
absolute return is E[|r|]/σ = √(2/π) ≈ 0.798 per period. Annualized
on a daily convention that is √(2/π · 252) ≈ 12.66 — and 12.66 is the
SIGNATURE of a costless abs() structure, NOT an edge. Reading it as
one is the oldest self-deception in this class of books.

It is also explicitly NOT a Sharpe. Sharpe divides the mean by the
FULL σ of the stream; the half-normal identity divides the expected
absolute value by σ and applies to the |r| distribution, whose mean
is σ·√(2/π), not σ. The two numbers answer different questions: a
costless abs() book can show a "12.66" while its true Sharpe is
exactly zero once costs enter. There is no function for this on
purpose — the moment it becomes a metric, someone gates on it, and a
structural signature is not evidence of edge. When you meet a carry or
flip book, compute E[|r|]/σ yourself and ask whether the annualized
number is just the signature.

## Per-trade vs per-period conventions

Two granularities live side by side, and mixing them is the A/C''
class of error:

- **Trade streams** (`atropos.backtest_metrics.per_trade_sharpe`, R-multiple
  trades from the replay or the paper log) are NEVER √n-annualized:
  trades are not independent periods, and an annualized per-trade
  Sharpe is a fabricated number. `per_trade_sharpe` does not
  annualize — that is the contract.
- **Per-period streams** (daily/threshold EV streams feeding
  `atropos.portfolio.portfolio_gates` or the Monte Carlo envelope in
  `atropos.ensemble_pipeline`) are the only streams an annualized
  reading may be taken on, and even there the active-period convention
  applies (`active_days` counts finite non-zero returns;
  CAGR/Calmar use the CALENDAR span, not the active count —
  KNOWN-ISSUES §16).

The A/C'' lessons, verbatim from the journal: A-daily was judged on a
daily booked-R feed at the daily threshold (the feed the strategy
decides at); C''-daily — the daily aggregate of a 15m OB-retest leg —
was VOID by construction (feed granularity must match decision
granularity, KNOWN-ISSUES §12): collapsing sub-daily decisions to one
number per day is signal loss, not a time-frame test. The trade
battery runs on trades; the daily threshold runs on periods. Never
feed one the other's stream.

## The units law

One risk unit — **1R = 1% of equity** — is the bookkeeping constant
across the whole stack:

- Every position enters at exactly one risk unit (`atropos.portfolio`
  replay, `atropos.backtesting` slot model): R-multiples are
  commensurable across legs, feeds and systems.
- Equity metrics compound R×0.01: a stream of R-multiples converts to
  equity fractions by multiplying each R by 0.01 and compounding —
  that is what the portfolio gates' drawdown floor reads
  (`atropos.portfolio.portfolio_gates` on the realized equity).
- **Additive vs multiplicative:** trade-level aggregates (mean R per
  trade, per-trade Sharpe, pooled R arrays across legs) are ADDITIVE
  in R. Equity-curve aggregates (drawdown, CAGR, Calmar) are
  MULTIPLICATIVE in the compounding R×0.01 stream. For
  concurrent-position aggregates the credited stream is per-period
  EV (what the DSR gate reads), while the ledger rows stay in R —
  never sum R across concurrent positions into an equity number, and
  never compound a per-trade mean into a drawdown.

## Metrics never gate

The standing law (KNOWN-ISSUES §16): **a performance metric never
gates, never overrides, never appears in an issue.** Sharpe 2.3 on 71
active days is an insignificant mean with outliers, not an edge — the
kill protocol answers "can this be killed", not "how good is the
return". Sortino returns None below 30 observations; `sharpe_ci` is a
noisy directional read-out, never a significance test; hit_rate ships
only next to its payoff context inside `atropos.evaluation_metrics.payoff_stats`.
Gates are preregistered thresholds (`dsr_min`, `dd_floor`,
`importance_rho_min`); metrics are read-outs next to the verdict,
owned by the modules in the table above.
