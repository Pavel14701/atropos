# Positioning

One page on what Atropos is, what it is not, and why it is a
falsification toolkit and not a compiler. Read this before building on
top of it.

**What Atropos is.** Atropos — falsification toolkit for trading
hypotheses: backtest kill protocol, feature decomposition, paper
parity, with machine-checkable instrument routing. Three tiers, one
doctrine — a hypothesis is guilty until it survives every attack, and
the product is VERDICTS, not strategies.

- **Backtest kill protocol** — `atropos.ensemble_pipeline`
  (`run_full_pipeline`), the core of the backtest tier: a candidate
  signal enters a 7-stage adversarial pipeline (null controls,
  specification curve, robustness battery, walk-forward, Monte Carlo
  risk, predictive evaluation, cross-sectional tests) and leaves with
  exactly one verdict. Thresholds are frozen in a preregistration
  before the data is touched; the authoritative verdict list lives in
  the module docstring — docs must not re-enumerate it. The ML chain
  (`atropos.ml_training.run_ml_kill_protocol`) runs the same doctrine
  on a feature dataset; `atropos.kill_chain.run_full_kill_chain`
  unifies both. Survival is not the end: for the rare survivor the
  reference backtest pipeline (`atropos.backtesting`,
  `atropos.ranking`, `atropos.backtest_metrics`) asks whether it is
  *tradeable* at all, and `atropos.portfolio` asks whether the book
  holds up with concurrent positions.
- **Feature decomposition** — `atropos.orthogonal.orthogonal_check`,
  orchestrated by `atropos.feature_scan.feature_analysis`: does the
  candidate carry information beyond known controls? Orthogonal to
  the kill chain, not a stage of it.
- **Paper parity** — `atropos.paper_protocol.run_paper_protocol`: the
  forward test of the IMPLEMENTATION — does paper execution match the
  frozen spec? A `PARITY_FAIL` indicts the paper engine or its feed;
  it never indicts the hypothesis.
- **Instrument routing** — `atropos.instrument_map` (`PLAN_REGISTRY`):
  strategy families map to batteries. The family is declared in the
  prereg/config as `strategy_family`; machine-checkable — enforced in
  the pipeline when declared (a forbidden tool or stage raises),
  flagged in audit mode otherwise.

**What Atropos is not.** Three explicit out-of-scope surfaces:

- **Data ingestion** — no market-data handling, no live feeds; the
  data contract (`atropos.data_contract`) verifies and aligns data
  YOU supply.
- **Execution engine** — no brokerage integration, no order routing,
  no live trading; the replay is a falsification-first reference
  under one fixed cost model (general-purpose engines — vectorbt /
  backtrader / qlib — live on the other side of that line).
- **Capital allocation** — no portfolio construction, no weights, no
  capital deployment; every position enters at exactly one risk unit
  (portfolio SIMULATION is not portfolio CONSTRUCTION).

It also does not optimize strategy parameters or search the space of
profitable configurations — tuning after a verdict starts a new
hypothesis anyway. And nothing here guarantees profit: `SURVIVES`
means "not killed by every attack we ran", never "trading-ready".

**Why "falsification toolkit" and not "compiler".** A compiler turns
an idea into a running strategy — it claims strategy production and
live deployment. Atropos claims neither: it produces VERDICTS
(SURVIVED / KILLED are the product), and every tier after the kill
protocol only sharpens the evidence. The re-entry condition is
explicit, not rhetorical: **"compiler" returns when a real-data paper
run reaches SURVIVES_TO_LIVE — trigger, not vibe.** Until then the
honest name is falsification toolkit.

**The four honest limits of the paper protocol** (read before quoting
its outputs):

1. `tools_used` is a declaration, not a measurement — parity validates
   the declaration against the family, not against actual execution.
2. `audit_report` on runner outputs without saved stages returns `[]`.
3. `SURVIVES_TO_LIVE` has not yet been achieved on real data — the
   token exists, the evidence does not.
4. Spec correctness is out of scope — the protocol validates that
   paper matches frozen, not that frozen is right (frozen correctness
   goes through `run_full_pipeline` on the backtest feed).

**How the tiers relate.** The kill protocol is the methodological core
(falsification first); feature decomposition and paper parity widen
the evidence, not the claims. All of it is a library: Atropos ships
functions, not an app — you import them into YOUR pipeline and keep
your own data, execution and sizing layers.

| Task | Use |
|---|---|
| Kill a rule-based hypothesis | `run_full_pipeline` — how-to 02 |
| Kill an ML hypothesis | `run_ml_kill_protocol` — how-to 05 |
| Judge rule + ML in one verdict | `atropos.kill_chain.run_full_kill_chain` |
| Decompose a candidate's information | `atropos.orthogonal.orthogonal_check` — KNOWN-ISSUES §11 |
| Check a survivor's tradeability | `docs/examples/backtest_pipeline.py` — how-to 06 |
| Compare rankers honestly | `atropos.ranking` + `RANKING_METRICS` — how-to 07 |
| Choose a slot admission policy | `atropos.backtesting.run_admission_policy` — how-to 08 |
| Compare portfolio policies / gates | `atropos.portfolio` — how-to 09 |
| Route a strategy family to its instruments | `atropos.instrument_map.PLAN_REGISTRY` (`strategy_family`) — skill 19 |
| Forward-test paper execution vs frozen spec | `atropos.paper_protocol.run_paper_protocol` — skill 20 |
| Portfolio kill protocol calibration | `docs/calibration.md` — public-data progression, blind-run protocol |

**The one-paragraph pitch** (and the honest version of it):

> "Atropos is a falsification-first backtest framework. It does not
> replace vectorbt — it replaces the set of scripts quant researchers
> write AROUND vectorbt to check that their Sharpe is not an artifact.
> The portfolio-aware layer closes the 'what about a portfolio of 100
> names' question without claiming portfolio construction. The novelty
> is honest verdicts and portfolio-level robustness gates (designed
> for a niche that general-purpose engines do not cover).
> Game-changing? No. Closing a real gap? Yes."

Do not upgrade "closing a real gap" to "game-changing" in any pitch:
falsification is hygiene, not discovery — it makes research more
honest, but it does not create new alpha. Until external adoption
earns it (a kill competitors passed, a calibration competitors got
wrong, a community using the verdict language), the honest sentence
is: **"It does not change the rules of the game — it makes the game
more honest."**
