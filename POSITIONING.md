# Positioning

One page on what Atropos is, what it is not, and how its two halves
relate. Read this before building on top of it.

**What Atropos does.** Atropos is a hypothesis-killing framework: a
library you point at your own candidate signal and that returns exactly
one verdict. The kill-protocol is the product: freeze a preregistration,
run the 7-stage adversarial pipeline (null controls, specification
curve, robustness battery, walk-forward, Monte Carlo risk, predictive
evaluation, cross-sectional tests), read the verdict — from `SURVIVES`
to `COST_DEATH` — and write the tombstone. The ML chain
(`run_ml_kill_protocol`) runs the same doctrine on a feature dataset:
expanding-block walk-forward (purge + embargo) and four gates. For the
rare signal that survives, Atropos also ships a reference backtest
pipeline — simulate → rank → admission → metrics
(`atropos.backtesting`, `atropos.ranking`, `atropos.backtest_metrics`)
— which asks the next question: is the survivor *tradeable* at all
once one slot at a time, real taker costs, and a past-only ranking
oracle are imposed on it? The portfolio layer (`atropos.portfolio`)
asks the question after that: does the book still hold up with
concurrent positions — slot admission with replacement,
correlation-aware admission (a candidate whose recent return proxy is
|Spearman|-correlated with an open position's above the cap never
enters), and portfolio-level gates (deflated Sharpe on the daily EV
stream, drawdown floor, same-side concentration) over the realized
equity.

**What Atropos does not do.** It does not optimize strategy parameters
or search the space of profitable configurations — a candidate that
survived is not tuned here, and tuning after a verdict would start a
new hypothesis anyway. The backtest pipeline IS a backtester — but not a general-purpose
one.  It is a falsification-first replay of the survivor under one
fixed, documented cost model: bar-level stop/target/time exits with a
gap-through-stop scratch contract, a one-slot state machine, and
slot-admission policies.  What it deliberately does not provide: no
portfolio construction, no brokerage integration, no parameter sweeps,
no market data handling — those live in general-purpose engines
(vectorbt / backtrader / qlib), and mixing the two roles would
compromise the falsification-first contract. The portfolio layer
sharpens rather than weakens this line: portfolio SIMULATION is not
portfolio CONSTRUCTION — `atropos.portfolio` replays a fixed slot cap
to gate a stream; it does not optimize weights, allocate capital, or
trade live, and every position enters at exactly one risk unit. And nothing here guarantees
profit: `SURVIVES` means "not killed by every attack we ran", never
"trading-ready", and a passing backtest is one more non-kill, not a
promise.

**How the two halves relate.** The kill-protocol is the methodological
core (falsification first); the backtest pipeline is the reference
check of the survivor's tradeability and nothing more. Both are a
library: Atropos ships functions, not an app — you import them into
YOUR pipeline and keep your own data, execution and sizing layers.

| Task | Use |
|---|---|
| Kill a rule-based hypothesis | `run_full_pipeline` — how-to 02 |
| Kill an ML hypothesis | `run_ml_kill_protocol` — how-to 05 |
| Judge rule + ML in one verdict | `atropos.kill_chain.run_full_kill_chain` |
| Check a survivor's tradeability | `docs/examples/backtest_pipeline.py` (simulate → rank → admission → metrics) — how-to 06 |
| Compare rankers honestly | `atropos.ranking` + `RANKING_METRICS` — how-to 07 |
| Choose a slot admission policy | `atropos.backtesting.run_admission_policy` — how-to 08 |
| Compare portfolio admission policies / portfolio gates | `atropos.portfolio` (`run_portfolio_replay`, `run_portfolio`) — how-to 09 |

**The one-paragraph pitch** (and the honest version of it):

> "Atropos is a falsification-first backtest framework. It does not
> replace vectorbt — it replaces the set of scripts quant researchers
> write AROUND vectorbt to check that their Sharpe is not an artifact.
> The portfolio-aware layer closes the 'what about a portfolio of 100
> names' question without claiming portfolio construction. The novelty
> is honest verdicts and portfolio-level robustness gates, which
> competitors do not have. Game-changing? No. Closing a real gap? Yes."

Do not upgrade "closing a real gap" to "game-changing" in any pitch:
falsification is hygiene, not discovery.  It makes research more
honest, but it does not create new alpha — that is a different class
of contribution (a new methodology like DSR, a new data class, new
infrastructure).

**What would have to happen to earn "game-changing"** (all three are
earned by adoption and time, none by code):

1. One external case where Atropos killed a hypothesis that
   competitors passed — proof of value.
2. One external case where Atropos confirmed a hypothesis that
   competitors killed falsely — proof of calibration (it can refuse
   to kill live edges, not only kill everything).
3. A community that adopts the verdict language — "my signal did not
   pass the DSR gate in Atropos" as a phrase other people use.

Until all three exist, the honest sentence is: **"It does not change
the rules of the game — it makes the game more honest."**

