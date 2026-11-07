# ATROPOS — a system for killing crap hypotheses

## A portable hypothesis-testing framework for quant finance

> **Quick paths.** User (apply it to your own project) → `docs/how-to/01`.
> Terms and definitions → `GLOSSARY.md`. Context for an AI agent → `atropos/agent/`
> (pip package) or the `atropos agent-context` command. ML hypothesis (walk-forward
> engine) → `docs/how-to/` (skill 04 / `run_ml_kill_protocol`). Reference reading —
> start with §1–§3 (philosophy, pipeline, laws); the rest as needed.

> **Purpose.** A self-contained protocol for testing factor and trading hypotheses,
> built around falsification rather than confirmation. An agent with zero context
> runs the stages of §2 in order and gets a reproducible verdict. The document is
> not tied to any repository, dataset, or codebase. Calibration: crypto markets
> (spot, perps, futures); applicability: any market and asset class. All examples
> in this document are anonymized.

**Notation.** `forward_return` — the forward return of the instrument over the
signal horizon. `t` — the t-statistic of the corresponding coefficient. Default
survival threshold: `|t| >= 2`. `20bp` — 0.20%, the baseline cost of a round-trip
trade (fee + spread + slippage + funding/borrow).

---

## §1 Philosophy

**1.1. This is falsification infrastructure, not a winner hunt.**
The framework's job is not to find a strategy that "works" but to kill a hypothesis
as early and as cheaply as possible. Each stage is a filter; the stages are ordered
by increasing cost of testing. A good day is not a signal found but a hypothesis
killed cheaply and documented.

**1.2. The graveyard is the product.**
Every documented death is institutional memory: it saves the next researcher weeks
of dead-end work and keeps capital from re-entering a known dead end. A hypothesis
without a death record counts as untested. A hypothesis "in progress" that has not
passed the stages has no right to consume capital.

**1.3. Motto.**
"We are not looking for the grail — we built a machine that kills losing hypotheses
before they take the capital."

**1.4. Every rule has been paid for with deaths.**
No rule in this document is theoretical. Each is backed by five or more documented
hypothesis deaths across different markets and asset classes. If a rule seems too
harsh — that is an argument for it, not against: the harshness is its function.

**1.5. The burden of proof lies on the hypothesis.**
A hypothesis is guilty (noise) until proven innocent. The researcher is not the
hypothesis's lawyer but a prosecutor with a checklist. The phrase "let's try again
with different parameters" means: a new hypothesis has been born, and it must pass
the entire pipeline from stage 1, with zero credit.

**1.6. Skipping a stage = protocol violation.**
An "obviously strong" signal does not skip stages. It is precisely the obviously
strong signals that most often turn out to be repackaging of known factors or a
single calendar year dressed up as alpha.

---

## §2 The Canonical Pipeline — 7 Stages

The order is mandatory; skipping any stage = protocol violation. For each stage we
fix: the question, the procedure, the kill criteria, an anonymized real example,
and the typical mistake. Death at any stage is final for the hypothesis in its
current formulation: a graveyard entry (§8) is mandatory; re-entry under the same
name is forbidden.

**Implementation map.** The stages above are the researcher's protocol; the
shipped pipeline (`atropos.ensemble_pipeline.run_full_pipeline`) automates the
statistical core as its own seven stages, in increasing order of cost:
null controls (stages 2 and 7 here: permutation/placebo, momentum and
known-feature repackaging, subperiod concentration, calendar fixed effects) →
specification curve (stage 4: does ANY reasonable spec keep `|t| >= 2` after
controls) → robustness battery (stage 5) → walk-forward (stage 6, expanding) →
Monte Carlo risk envelope (the cost line, law 3, plus drawdown tails) →
predictive evaluation (purged CV, feature-importance stability, deflated
Sharpe) → optional cross-sectional panel tests. The premise check (stage 1)
and the raw-signal sanity look (stage 3) stay with the researcher — a machine
cannot check that an instrument exists or that the sign matches the
prediction's rationale. Cost baselines differ by layer, by design: this law's
20bp round trip is the doc line, the backtesting engine's taker default is
25bp round trip (`2 * comm + gen_slip`, `atropos.backtesting.CostModel`), and
the pipeline's stage-5 default nets 10bp one-way per unit turnover
(`cost.cost_bps` in `atropos.ensemble_pipeline.DEFAULT_CONFIG`).

### Stage 1. PREMISE-CHECK — does the condition exist in the target market?

**Stage question:** do the structural premises of the hypothesis exist in the
target market — data, instruments, microstructure, the ability to hold both sides?

**Procedure:**
1. List the premises: "field X exists", "instrument Y is available",
   "history covers at least one full cycle".
2. Check each one against facts: the field present in the data; the instrument
   available (perp/option/borrow/leverage); history depth; liquidity of both legs;
   funding and session regimes.
3. Record the amount of available history and make sure it is enough for all
   later stages: 200 null permutations and the thirds split.

**Kill criteria:** a premise is missing or false → PREMISE-DEATH; data unavailable
or history shorter than the minimum required → DATA-BLOCKED. Stop. Do not build
infrastructure for a DATA-BLOCKED signal (§7, prohibition 6).

**Real example (anonymized):** factor X — a low-risk construction on an illiquid
asset: the premise "the short leg is available" failed. Death before any
statistics, zero compute spent.

**Typical mistake:** importing premises from another market ("but it worked on
equities"): borrow, order-book depth, trading sessions, and funding in crypto are
structured fundamentally differently.

### Stage 2. NULL-CONTROL — does the pipeline produce signal from noise?

**Stage question:** does our pipeline produce a "signal" out of pure noise?

**Procedure:**
1. Do not look at the real statistic yet. First, the null.
2. 200 times: shuffle the signal series (permutation) or substitute placebo
   dates; run the FULL pipeline, same filters and same clustering; collect the
   distribution of the statistic.
3. Compare the real statistic against the distribution: it must lie above the
   95th percentile of the null distribution.

**Kill criteria:** the real statistic does not beat the null distribution →
NULL-DEATH, stop. If the pipeline consistently finds "signals" in shuffled data,
the pipeline itself is broken (lookahead, leakage): fix the pipeline, not the
hypothesis.

**Real example:** factor X: raw t = 5.5 looked convincing, but the 95th percentile
of the placebo distribution turned out to be 4.8 — the real effect was on the edge
of noise.

**Typical mistake:** running the null on a truncated pipeline (with some filters
dropped) or using 20 shuffles instead of 200 — the percentile estimate floats
together with the verdict.

### Stage 3. SIGNAL TEST — raw effect, sign, stability across years

**Stage question:** is there a raw unconditional effect, directed as predicted,
and does it live longer than a year or two?

**Procedure:**
1. Baseline check: a quintile sort or the raw regression
   `forward_return ~ signal`.
2. Record four numbers: effect size, t, the sign against the prediction, the
   share of positive years.
3. Estimate the monotonicity of returns across quintiles — stage 5 will need it.

**Kill criteria:** the sign is opposite to the prediction with no a priori grounds
→ INVERTED-DEATH; `|t| < 2` → death; the effect holds in only 1–2 years → flag as
suspected calendar contamination and move on: stage 7 delivers the final verdict.

**Real example:** factor X: raw t = 5.5, 8 of 10 years positive — stage passed,
died later on incrementality. Contrast: factor Y: raw t = 3.1, but the sign
contradicts the entire known literature → INVERTED-DEATH, no mercy.

**Typical mistake:** celebrating the raw t as "it works". The raw t is the most
overrated number in research; without stage 4 it means nothing.

### Stage 4. INCREMENTAL TEST — the main stage

**Stage question:** does the signal add anything beyond what is already known?

**Procedure:**
1. Regression: `forward_return ~ momentum + size + volume + iv_level +
   market_beta + new_signal`, HC0 errors (robust, White-style).
2. The control set is fixed a priori: momentum, size, volume, IV level, market
   beta. Expanding or shrinking the set based on the result is forbidden.
3. Look exclusively at `t(new_signal)`.

**Kill criteria:** `t(new_signal) < 2` after controls = REPACKAGING
(INCREMENTAL-DEATH): the signal is a repackaging of known factors. Stop.

**Real example:** factor X: raw t = 5.5 → after controls t = -0.20.
It turned out to be momentum in expensive clothes.

**Typical mistake:** forgetting the main competitor (testing a "volatility"
signal without an IV-level control), using non-robust errors, or orthogonalizing
the signal up front — orthogonalization hides the contest itself; only an honest
regression with the full control set delivers the verdict.

### Stage 5. ROBUSTNESS BATTERY — 7 tests

**Stage question:** does the effect survive perturbation of arbitrary research
choices?

**Procedure — all 7 tests, no exceptions:**
1. **Threshold 2×:** double and halve the main threshold/parameter — does the
   effect survive?
2. **Window adjacent:** shift the holding window by an adjacent step (±1).
3. **Subperiod thirds:** the effect is significant in each third of the sample
   (at least 2 of 3).
4. **Leave-one-out:** drop one instrument/asset at a time — the effect does not
   vanish under any single drop.
5. **Placebo > 95pctl:** the real statistic is above the 95th percentile of the
   stage-2 null distribution.
6. **Declustering top-1%:** remove the largest cluster of overlapping events
   (top 1% by contribution to the effect) — the effect remains.
7. **Sign monotone:** quintile portfolio returns are monotone in the signal.

**Kill criteria:** fewer than 5 of 7 passed → ROBUSTNESS-DEATH, stop. 5–6 passed
but everything hangs on a single dimension → FRAGILE-DEATH.

**Real example:** factor Z: raw t = 4.2, passed threshold and window, but died on
declustering — the whole effect was one weekly catastrophe.

**Typical mistake:** soft scoring ("t = 1.7 — nearly significant"). The battery is
pass/fail on every test; one leniency devalues all seven tests at once.

### Stage 6. WALK-FORWARD — out of sample, with frozen decisions

**Stage question:** does the effect work out of sample, when no decision was made
looking at the future?

**Procedure:**
1. Expanding train→test: fit on ALL history before the test segment — apply
   to the next segment. Every later fold trains on strictly more data.
2. All thresholds and parameters are frozen at train time; on test they are
   applied as-is, no revising.
3. Purge the rows between train end and test start whose forward-return
   labels overlap the test window (purge gap = label horizon), so train
   cannot read the test period through overlapping labels.

**Why expanding, not rolling.** Earlier revisions of this document banned
expanding windows; that doctrine is obsolete and this paragraph supersedes it.
For a falsification stage, expanding is the correct shape: the question is
whether the edge survives out of sample as evidence accumulates, and more
accumulated history means more held-out evidence and a sharper verdict — a
rolling window would discard data and make the test WEAKER exactly when
confidence should grow. Rolling also smuggles in a free hyperparameter (the
window length) that is itself tunable — a fresh overfitting surface with no
falsification benefit. Expanding has no such knob: the initial training length
only seeds the first fold. The reference implementation
(`atropos.walkforward_framework`) offers no rolling mode at all; it additionally
de-duplicates overlapping test slices so fold frequency cannot manufacture
significance.

**Kill criteria:** shrinkage > 2× (the test result is worse than half the train
result) → OVERFIT-DEATH, stop. The effect's sign flips on test → death, no
discussion.

**Real example:** factor W: train t = 3.9 → test t = 1.1; the parameters
(threshold + cooldown) had been tuned on the full sample → OVERFIT-DEATH.

**Typical mistake:** "tweaking one parameter" between train and test, or
evaluating walk-forward on overlapping windows and counting correlated wins as
independent (see §7, prohibition 8).

### Stage 7. CALENDAR CONTROL — calendar contamination

**Stage question:** is the effect just one calendar year or one regime dressed up
as alpha?

**Procedure:**
1. Repeat the stage-4 incremental regression, adding year dummies and regime
   flags (bull/bear, high/low volatility).
2. Compare `t(new_signal)` before and after the addition.

**Kill criteria:** if t drops below 2 → calendar contamination → INCREMENTAL-DEATH
(calendar variant), stop. A calendar edge does not transfer to new days
(§3, law 5).

**Real example:** factor V: t = 2.8 → with year dummies t = 0.9; the whole effect
is one bull year.

**Typical mistake:** tuning the set of regime flags "until t survives" — that is
fitting the controls to rescue the hypothesis, disguised as a control.

---

## §3 The Six Systematic Laws

The laws are generalizations paid for with deaths. Each follows the format:
statement → mechanism → three anonymized examples → protocol.

### Law 1. Description ≠ prediction

**Statement.** Statistics describing the past distribution of a variable
("in 10 of 10 regime days there was X") describe the current regime and predict
nothing about new days.

**Mechanism.** A descriptive statistic is conditioned on knowing the regime is
underway. At the regime boundary the conditional distribution changes; the
regime's duration itself is no argument for its continuation — often the opposite.

**Examples:**
- factor A: hit-rate 10/10 inside the regime, zero predictive power on holdout
  days;
- factor B: vol-of-vol described the crash days perfectly, no forward return;
- factor C: the correlation regime described flawlessly, the forward direction —
  a coin flip.

**Protocol.** Any descriptive statistic must pass a forward test of its
prediction on days outside the sample the description was computed on. Failed →
DECORATION-DEATH.

### Law 2. Incrementality is the deciding test

**Statement.** The raw t means nothing; the only verdict is the new signal's t
after the known controls.

**Mechanism.** A new signal almost always correlates with known factors
(momentum, size, volume, IV, beta). The regression divides the explanatory pie;
whoever owns the pie keeps the t.

**Examples.** Three candidates with raw t > 5 died at t < 1 after controls:
- factor A: raw 5.8 → 0.4;
- factor B: raw 5.5 → -0.20;
- factor C: raw 6.1 → 0.9.

**Protocol.** The raw t never appears on its own. Stage 4 is mandatory; the
control set is fixed a priori. `t < 2` after controls = REPACKAGING.

### Law 3. The 20bp cost line

**Statement.** If the effect's amplitude per round trip is below ~20bp of costs,
the hypothesis is dead before any statistics.

**Mechanism.** Fee + spread + slippage + funding/borrow. An effect smaller than
the cost of harvesting it cannot be monetized at any level of statistical
significance.

**Examples (18+ deaths on this line):**
- factor A: mean edge 8bp per round trip at daily turnover → COST-DEATH;
- factor B: intraday reversal, 12bp gross against 20bp of costs;
- factor C: 6bp of edge at the rebalance frequency the construction itself
  requires.

**Protocol.** The cost model is computed BEFORE statistics (§7, prohibition 5).
Median gross edge < 20bp → COST-DEATH: past all stages, straight to the graveyard.

### Law 4. The risk premium in crypto is negative

**Statement.** Factors that pay a risk premium in equities (lottery-likeness,
high risk, high IV, illiquidity) are often inverted in crypto: risky assets lose.

**Mechanism.** A market dominated by retail with a taste for lottery payouts
overpays for the "casino" and pays no risk premium in the academic sense; there
is no institutional demand for "risk factors". A sign imported from another asset
class is an assumption, not a fact.

**Examples:**
- factor A: max-ret/lottery factor — the sign is opposite to the academic
  literature;
- factor B: the premium for high IV is negative;
- factor C: the low-vol anomaly in crypto is stronger, not weaker, than in
  equities.

**Protocol.** The sign check at stage 3 is mandatory (§7, prohibition 4). An
inversion with an a priori rationale is not a failed test but a separate
"inverse" hypothesis that itself passes the entire pipeline from stage 1.

### Law 5. An annual regime = calendar contamination

**Statement.** A rule found in the regime of a single calendar year ("in a year
like this it worked") does not transfer to new days.

**Mechanism.** One year = one realization of one regime; year dummies absorb it.
The market does not remember calendar labels; the regime's uniqueness does not
replicate.

**Examples:**
- factor A: the entire effect in a single year; with year dummies t 2.8 → 0.9;
- factor B: the "month of year" rule died in the very next year;
- factor C: a regime flag fitted on one episode is dead on the next.

**Protocol.** Stage 7 is always run. Calendar terms in scoring are forbidden
(§7, prohibition 1). To survive, the effect must live in at least 2 of 3
subperiods.

### Law 6. A vol forecast does not monetize

**Statement.** A good volatility forecast does not by itself turn into PnL. Two
independent confirmations: the forecast works — the sizing does not.

**Mechanism.** A vol forecast improves the description of the distribution, but
the gain from sizing is a second-order effect (the variance of terminal wealth),
and costs and construction constraints eat it. The Sharpe gain from vol timing is
small against the same effort spent on direction.

**Examples:**
- factor A: the vol forecast has decent quality, the Sharpe of the
  vol-targeted portfolio did not change;
- factor B: a regime vol predictor — no sizing was found that beats the constant
  benchmark after costs;
- factor C: a fear-index-style predictor: high accuracy, zero monetization.

**Protocol.** Vol hypotheses are tested only with an explicit sizing rule against
benchmark sizing, after costs. Forecast accuracy by itself is not success.

---

## §4 The Death Taxonomy — 18 Terms

| Term | Definition | Example (one line) |
|---|---|---|
| NULL-DEATH | The real statistic is indistinguishable from the distribution of shuffled data. | t = 3.2 with placebo 95th percentile = 4.1 |
| COST-DEATH | Gross edge below the cost line of ~20bp per round trip. | edge of 12bp at daily turnover |
| INCREMENTAL-DEATH | After known controls the new signal's t falls below 2: repackaging. | raw 5.5 → 0.1 after momentum+IV |
| PREMISE-DEATH | A structural premise of the hypothesis does not exist in the target market. | the short leg is unavailable |
| RISK-DEATH | The effect is just a premium for a known risk; it disappears once the exposure is controlled. | the premium vanished after controlling for beta and crash days |
| DATA-BLOCKED | The hypothesis cannot be tested: no data, or history too short or dirty. | an option chain is needed, only spot exists |
| ROBUSTNESS-DEATH | Battery failed: fewer than 5 of 7 tests passed. | dies when the threshold is doubled |
| PARADIGM-DEATH | The hypothesis was true in a paradigm that has ended (structural break). | the effect exists before the market institutionalizes, zero after |
| INSTRUMENT-BLOCKED | The effect exists but is unreachable with the available instruments. | the basis trade needs a perp, only spot exists |
| OVERFIT-DEATH | Walk-forward shrinkage > 2×: the parameters memorized train. | train t 3.9 → test t 1.1 |
| PERSISTENCE-DEATH | The effect depleted over time (publication, adaptation, arbitrage). | worked for two years, then flat |
| INVERTED-DEATH | The sign is opposite to the prediction: the hypothesis is wrong in this formulation. | "high IV wins" — it loses |
| FRAGILE-DEATH | The effect lives only in a narrow band of parameters/windows. | alive at threshold 0.8–1.2, dead at 1.3 |
| LEVERAGE-DEATH | The gross result is positive, but the path (drawdown/vol) liquidates the position before the premium is paid. | a 60% drawdown at 3× leverage |
| ANTI-SIGNAL-DEATH | The signal predicts correctly, but trading against it is unexecutable. | the crash signal fires when exit is impossible |
| DECAY-DEATH | The signal's half-life is shorter than the execution delay. | the signal lives 30 seconds, execution takes 5 minutes |
| DECORATION-DEATH | A beautiful description with no predictive power. | 10/10 inside the regime, zero on new days |
| CONSTRUCTION-CLOSED | The trading construction is impossible as stated. | the weights require tomorrow's open today |

Note. PERSISTENCE-DEATH is about the aging of a known effect; DECAY-DEATH is
about the signal's intrinsic speed against the execution delay; PARADIGM-DEATH is
about the structural break of an entire class of hypotheses.

### What a kill verdict means

The code emits tokens in the underscore form; this document's doctrine
uses the hyphen form (see the mapping below). Every run reports
`verdict_class`, one of four:

- **DEAD** — absence of edge proven on this feed. A conclusion: act on
  it. Only two paths qualify: the battery baseline is catastrophically
  dead (mean ≤ 0 with power / no active days), and `CS_NO_EDGE` when
  the config target is `"xs"`.
- **NEEDS_ANALYSIS** — a specific gate was informative but did not
  prove absence of edge: investigate the mechanism. The most common
  outcome.
- **INCONCLUSIVE** — the test could not run: fix the feed, the model
  or the window and re-run.
- **SURVIVES** — this protocol on this feed could not kill it. NOT
  proof of truth.

Every kill reports five facts: (1) which gate fired, (2) on what feed
(anchor, n_obs, active_days), (3) which specific test failed, (4)
whether the failure is informative, (5) what would change the verdict.
A verdict without all five facts is not a verdict.

Code token ↔ doctrine term (rows of the table above): `ROBUSTNESS_DEATH`
↔ ROBUSTNESS-DEATH (the battery row), `OVERFIT_DEATH` ↔ OVERFIT-DEATH
(the walk-forward shrink row), `DEGENERATE_FEED` /
`CONSTANT_PREDICTIONS` ≈ DATA-BLOCKED (the feed or model could not be
tested), `PLACEBO_VIOLATION` ≈ the NULL-DEATH family (the effect
survives its own manipulation).

**Booked-R caveat.** A feed whose signal is the exposure sign and whose
returns are booked R is circular for a signal-vs-return protocol: the
"prediction" is verified against its own P&L bookkeeping. Such legs
need per-trade or predictor-form ingestion — e.g. the SYSTEM-9 legs
ported in `system9_legs.py`; that ingestion is separate work.

---

## §5 The Red Thread

One line — the reason this document exists:

```
Premise → Null → Signal → INCREMENTAL → Robustness → Walk-forward → Calendar → Verdict → Graveyard | Prereg
```

- Each arrow is a filter; the cheapest killers come first.
- Movement is forward-only: returning to an earlier stage is forbidden.
- Death at any segment is final for the hypothesis in its current formulation.
- A surviving hypothesis has two outcomes, and both are honest: preregistration
  and forward (§8), then adopt or reject.

---

## §6 What to Do If... — the Decision Tree

```
Start: hypothesis formulated (what, where, sign, horizon, construction)
  |
  |-- Stage 1: premise missing / no data ------------> DATA-BLOCKED ---------> STOP
  |-- Stage 2: null not beaten (t inside noise) -----> NULL-DEATH -----------> STOP
  |-- Stage 3: sign against the prediction ----------> INVERTED-DEATH -------> STOP
  |-- Stage 4: incremental t < 2 ---------------------> REPACKAGING -----------> STOP
  |                                                   (INCREMENTAL-DEATH)
  |-- Stage 5: robustness < 5/7 ----------------------> ROBUSTNESS-DEATH ------> STOP
  |-- Stage 6: WF shrink > 2x ------------------------> OVERFIT-DEATH ---------> STOP
  |-- Stage 7: calendar t drops < 2 ------------------> INCREMENTAL-DEATH -----> STOP
  |                                                   (calendar variant)
  |
  `-- All passed ---> PREREG (§8) ---> forward test on live data
                          |-- confirmed --------> ADOPT (with the prereg number)
                          `-- not confirmed ----> graveyard entry -> STOP
```

Canonical forks:

- **If the premise does not exist** → DATA-BLOCKED, stop. No infrastructure gets
  built.
- **If the null is not beaten** → NULL-DEATH, stop. The pipeline is fixed first,
  not the hypothesis.
- **If incremental t < 2** → REPACKAGING (INCREMENTAL-DEATH), stop.
- **If robustness < 5/7** → ROBUSTNESS-DEATH, stop.
- **If WF shrink > 2×** → OVERFIT-DEATH, stop.
- **If calendar t drops below 2** → INCREMENTAL-DEATH (calendar variant), stop.
- **If everything passed** → prereg → forward → adopt or reject. There is no
  third option.

Every STOP ends with a graveyard entry in the §8 form. A missing entry is a
protocol violation, not an act of mercy.

---

## §7 What NOT to Do — 8 Prohibitions

1. **NO calendar terms in scoring.** No month, no year, no regime label enters
   the signal function. A calendar edge does not transfer to new days
   (§3, law 5); the check is stage 7.
2. **NO parameter tuning after the data.** Any change after seeing the result =
   a new hypothesis, the entire pipeline from stage 1. One sample — one promise.
3. **NO "one run and done".** Always the battery (stage 5) and the null control
   (stage 2). A single run is statistically indistinguishable from a lucky peek.
4. **NO equity factors without a sign check.** In crypto the risk premium is
   negative (§3, law 4): a sign from the equities literature is an assumption,
   not a fact.
5. **NO signals without a cost model.** The cost line is computed before
   statistics (§3, law 3): an effect smaller than its costs is dead at any t.
6. **NO infrastructure for DATA-BLOCKED signals.** No data — no pipeline.
   Building an engine for an untestable hypothesis is investing in your own
   grave.
7. **NO regime statistics without a prediction test.** Description ≠ prediction
   (§3, law 1): every descriptive statistic must prove forward strength on new
   days.
8. **NO tests on overlapping forward returns.** Overlap inflates t: the
   effective sample is shorter than the visible one. Non-overlapping windows, or
   an honest clustering correction, plus the stage-5 declustering.

---

## §8 The Graveyard and Preregistration Protocol

### 8.1. Graveyard Entry — Required Fields

1. Hypothesis name (anonymized) and date of death.
2. Stage of death and death type from the §4 taxonomy.
3. The minimal numbers: raw t; incremental t (if reached); the failed test.
4. Cause of death in one line.
5. "Resurrection": what new data or conditions would make a reopen possible.
6. "Anti-recidivism": which §7 prohibition a re-entry would violate.

### 8.2. Graveyard Rules

- Death is final for the formulation. The same idea with a modified construction
  is a new hypothesis: pipeline from stage 1 and a new preregistration.
- Before starting a new hypothesis, the agent must read the relevant graveyard
  entries. A repeat death for the same reason is logged separately and counts as
  a process error, not a hypothesis error.

### 8.3. Preregistration — Required Fields

1. Hypothesis: what, where, sign, horizon.
2. Construction: portfolio or regression; all thresholds and parameters frozen.
3. Cost model: the cost line per round trip.
4. Forward window: length and the success criterion (e.g. `|t| >= 2` on
   non-overlapping forward returns).
5. The adopt/reject rule, formulated before the forward starts.

### 8.4. The Capital Rule

Only preregistered hypotheses earn the right to forward capital. Only a graveyard
entry closes a dead hypothesis. An adopted strategy without a preregistration
number does not exist.

---

## §9 Reading Verdicts After the Backtest Pipeline

The protocol's `SURVIVES` is a falsification statement: every attack in
§2 failed to kill the hypothesis. It is NOT a tradeability statement.
"Not killed" leaves the economic question open — one slot at a time, a
real cost line, and the mechanics of turning a stream of candidate
signals into an admitted book are nowhere tested by the seven stages.

That is the job of the backtest pipeline — the reference replay that
runs AFTER the protocol on the survivor: `simulate_signal` (isolated
net R under the taker model, with the gap-through-stop scratch
contract) → `run_position_state_machine` (which signals become trades
when one slot is occupied) → `rank_within_groups` (which candidate of a
decision point is picked, trained past-only) → `run_admission_policy`
(FCFS vs REPLACE under a slot cap) → the per-trade metrics
(`per_trade_sharpe`, `pooled_stats`, `bucketed_sharpe`). It speaks its
own language — `mrr` for the ranking oracle, `per_trade_sharpe` (never
annualized) and `max_drawdown_r` for the book — and a survivor can fail
here too: a `SURVIVES` signal whose admitted book pays the cost line
nowhere is not tradeable, whatever its t-statistic was.

The two verdict chains are not interchangeable, and neither can stand in
for the other. A kill verdict is not a backtest report; a backtest
report is not a survival verdict (its metrics say nothing about
incrementality, null controls or calendar contamination). Protocol
first, pipeline second — read each verdict in its own vocabulary, and
pass a survivor through both before any capital decision.
Reference: `POSITIONING.md`, `docs/how-to/06-run-backtest-pipeline.md`.

---

*End of document. Stage order is mandatory, the graveyard is the product, the
verdict comes only from the protocol.*
