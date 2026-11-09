# How-to 02. Reading pipeline verdicts

> **Estimated time:** 15 minutes (the examples take ~5 seconds in total)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.12; pandas/numpy come as dependencies. Bring
> your own hypothesis data — the examples here are synthetic.

## Why

`run_full_pipeline` runs a signal through up to seven adversarial stages
and returns exactly one verdict. This how-to is the decoder: what each
verdict means, which stage issued it, and what to do next.
Self-contained: no need to read `FRAMEWORK.md`. Below you also get the
vocabulary of the second kill chain, ML (`run_ml_kill_protocol`: three
verdicts, four gates), and the unified matrix for when both run.

Import (the `atropos` package does not re-export functions at the top
level — use the module path):

```python
from atropos.ensemble_pipeline import run_full_pipeline
```

Input: `data` — a dict with `returns` (forward returns) and `signal`
(same index); optional `features` (a DataFrame of known controls),
`mom7`/`mom30`, `hypothesis` (a description string). For the
cross-sectional mode: `panel_signal`/`panel_returns` (a DataFrame: rows
are dates, columns are assets). The second argument is
`hypothesis_config` (overrides on top of `DEFAULT_CONFIG`).

The output is a dict with keys:

| Key | Contents |
|-----|----------|
| `overall_verdict` | one of the pipeline's verdict tokens — the authoritative list lives in the `atropos.ensemble_pipeline` module docstring (docs must not re-enumerate it); the table below decodes the tokens a first run most often returns |
| `killed_by_stage` | the killing stage number, or `None` |
| `kill_reason` | the reason in one line (or `None`) |
| `stages` | a list of 7 dicts `{stage, name, status, summary, results}` |
| `config` | a JSON-compatible copy of the actual config |
| `n_obs`, `date_range`, `hypothesis`, `runtime_seconds` | run context |

## The verdicts, decoded

The authoritative verdict list — including the panel-only
`SURVIVES_CS_ONLY` forms and the non-informative tokens — lives in
exactly one place, the `atropos.ensemble_pipeline` module docstring;
docs must not re-enumerate it. This table is a decoder for the tokens
a first run most often returns, not the enumeration.

| Verdict | Issued by | What it means | What to do next |
|---------|-----------|---------------|-----------------|
| `SURVIVES` | all stages PASS | "not killed yet" — **not confirmation**, but the absence of a kill | record a SURVIVED tombstone; use it as justification; monitor on new snapshots |
| `SURVIVES_WITH_WARNINGS` | all stages PASS, WARNs present | survived with caveats (undecided controls, turnover, deflated Sharpe) | write the WARNs into the tombstone; decide whether the caveats are compatible with the decision |
| `REPACKAGING` | stage 1 (momentum/known-features) or 2 | the signal adds nothing beyond known momentum/features — repackaging | do not revisit without a new mechanism; a new construction = a new hypothesis |
| `ARTIFACT` | stage 1 (the other controls) | the effect is an artifact of alignment / placebo / sub-period concentration / calendar | check the data mechanics; resurrection only via the tombstone conditions |
| `DEAD` | stage 3 | the rule is untradeable or a catastrophic robustness-battery failure (7 tests) | the rule is unimplementable as-is; a different rule = a new hypothesis |
| `NO_OOS_EDGE` | stage 4 or 6 | no out-of-sample edge (walk-forward / purged CV) | do not return on the same data; a new snapshot only per the tombstone conditions |
| `COST_DEATH` | stage 5 | the gross envelope passes but net-of-cost Sharpe does not: costs ate the edge | revisit only the execution economics (maker mode, less frequent rebalancing); conditions in the tombstone |
| `RISK` | stage 5 | a risk-envelope breach: the mDD tail OR the gross Sharpe floor | incompatible with the risk mandate; different sizing/mandate = a new hypothesis |
| `CS_NO_EDGE` | stage 7 | cross-sectional tests failed (IC / net spread / quantile monotonicity) | for panel families: the family is closed in this formulation |

## Stage statuses

| Status | Meaning |
|--------|---------|
| `PASS` | the stage passed |
| `WARN` | passed with a caveat: some checks inside the stage came back undecided or borderline (see `summary`) |
| `FAIL` | the hypothesis was killed at this stage; all following stages are `SKIPPED` |
| `SKIPPED` | the stage did not run: killed earlier, OR an optional stage without data for it (stage 7 without a panel; stages 1–6 in panel-only mode) |

`INCONCLUSIVE` is **not** a stage status: it lives inside the results.
Individual stage-1 controls and the stage-7 overall return
`INCONCLUSIVE` when a check is not computable (e.g. the signal is
exactly collinear with a control — "collinear with controls"), and
stage 2 reports `INCONCLUSIVE` in its summary when a lenient `SURVIVES`
call is backed by fewer than 5 testable specs — a multiple search too
small to trust. Upstairs this surfaces as a stage WARN, not a kill:
"could not check" ≠ "killed". That is the meaning of INCONCLUSIVE
across the whole framework — too little data or non-computability does
not kill a hypothesis.

The seven stages in order: 1 `null_control_suite` (6 null controls),
2 `incremental_regression` (specification curve), 3
`robustness_battery` (7 tests), 4 `walkforward_framework`,
5 `monte_carlo` (the net-of-cost Sharpe/mDD envelope), 6
`predictive_model_eval` (purged CV, deflated Sharpe), 7
`cross_sectional` (optional, when a panel is present).

## The ML chain (run_ml_kill_protocol)

The framework has a second kill chain — a walk-forward ML engine over a
feature dataset (`features` → binary `target`), not `signal` → outcome.
It has its own verdict vocabulary: `SURVIVES`, `DEAD(<gate>)`,
`UNDERPOWERED`. How to assemble the
dataset and run it — how-to 05
(`docs/how-to/05-run-ml-kill-protocol.md`); this section is the decoder.

```python
from atropos.ml_training import run_ml_kill_protocol
```

| Verdict | Issued by | What it means | What to do next |
|---------|-----------|---------------|-----------------|
| `SURVIVES` | every requested model passed all enabled gates at `n_events >= n_min_events` | "not killed yet" — not confirmation, but the absence of a kill | record a SURVIVED tombstone with the numbers (EV, DSR, rho, n_events); monitor on new snapshots |
| `DEAD(<gate>)` | the first triggered gate of the first model that did not survive | the edge did not survive this gate; the name in brackets is the cause | read the `reason` of the gate entry in `kills` / `models[name]["kills"]`; resurrection conditions in the tombstone |
| `UNDERPOWERED` | the model has `n_events < n_min_events` (default 500 declustered events) | **undecided, not a kill**: the sample cannot power any gate decision, gate flags are informational | add data and re-run; in the unified matrix this is INCONCLUSIVE, not death |

The four gates and their thresholds (`checks.*` in the config):

| Gate | What it catches | Default threshold |
|------|-----------------|-------------------|
| `incremental` | repackaging of the linear part: the paired-bootstrap t of the EV difference between the model's taken trades and the logit battery (features + your `battery_controls`) | `incremental_t` = 2.0 |
| `cost_line` | a book that fails to pay its costs: merged-OOS EV of taken trades below the bar | `profit_bar` = 0.0040 (40bp) |
| `importance_stability` | "signal out of noise": per-fold permutation rankings of the features do not reproduce | `importance_rho_min` = 0.5 |
| `deflated_sharpe` | multiple testing: the deflated Sharpe of the daily EV stream | `dsr_min` = 0.95 |

Each gate entry is `{check, enabled, triggered, inconclusive,
reason}`. The INCONCLUSIVE semantics are the same as across the whole
framework: "could not check" ≠ "killed" (the typical case is incremental
without `battery_controls`). Careful: SURVIVES with a non-computable
incremental means the incremental edge is undemonstrated, not passed.
The `DEAD(<gate>)` verdict names the first triggered gate in the order
`incremental` → `cost_line` → `importance_stability` →
`deflated_sharpe`, not the "loudest" cause.

## Both chains (run_full_kill_chain)

The rule pipeline and the ML engine can be run through one wrapper
(`rule_data` + optional `ml_dataset`); it is a thin aggregator with no
gates of its own: `unified` is a pure function of the two verdicts, and
a kill by either chain decides everything. "Alive" = `SURVIVES` /
`SURVIVES_WITH_WARNINGS`; `[...]` is `unified["source"]`.

```python
from atropos.kill_chain import run_full_kill_chain
```

| Rule verdict | ML verdict | unified |
|--------------|------------|---------|
| alive | (not run) | `SURVIVES` |
| kill | (not run) | `KILLED` [rule] |
| alive | `SURVIVES` | `SURVIVES` |
| alive | `DEAD(...)` | `KILLED` [ml] |
| alive | `UNDERPOWERED` | `INCONCLUSIVE` [ml] |
| kill | `SURVIVES` | `KILLED` [rule] |
| kill | `DEAD(...)` | `KILLED` [rule+ml] |
| kill | `UNDERPOWERED` | `KILLED` [rule] |

ML `UNDERPOWERED` maps to INCONCLUSIVE — undecided is not death: a new
snapshot with more power may decide the hypothesis. How to assemble both
datasets and what to do with each outcome — how-to 05; reference:
`docs/api/atropos/kill_chain.html`.

## Worked examples (all runs are real)

### Example 1: a real edge → SURVIVES

A synthetic world: a hidden AR(0.5) driver predicts the forward return;
the signal = driver + 30% momentum "bleed"; momentum is a confounder in
the returns. This is what an honestly planted edge looks like.

```python
import numpy as np
import pandas as pd

from atropos.ensemble_pipeline import run_full_pipeline


def make_planted_edge(n_obs=900, seed=4):
    rng = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    mom_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        mom[pos] = 0.5 * mom[pos - 1] + mom_shocks[pos]
    mom /= float(np.std(mom))
    driver = np.zeros(n_obs)
    driver_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        driver[pos] = 0.5 * driver[pos - 1] + driver_shocks[pos]
    driver /= float(np.std(driver))
    signal = driver + 0.3 * mom
    fwd = driver + 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(fwd, index=idx),
        "signal": pd.Series(signal, index=idx),
        "hypothesis": "persistent order-flow state z predicts next-period return",
    }


res = run_full_pipeline(make_planted_edge())
print("verdict      :", res["overall_verdict"])
print("killed_by    :", res["killed_by_stage"])
for st in res["stages"]:
    print(f"  {st['stage']} {st['name']:<24} {st['status']:<7} {st['summary'][:95]}")
```

Example output:

```text
verdict      : SURVIVES
killed_by    : None
  1 null_control_suite       PASS    6/6 controls passed
  2 incremental_regression   PASS    SURVIVES: max|t|=55.94 across 10 specs (10 with |t|>=2, 5 controlled, sign_consistent=True)
  3 robustness_battery       PASS    ROBUST: 7/7 tests passed (baseline threshold=0.089)
  4 walkforward_framework    PASS    PASS: wf_t=26.56 direction=0.853 shrink=0.88 (22 folds, no_peeking=True, mode=expanding, purge=
  5 monte_carlo              PASS    net SR p5/p50/p95 = 3.85/4.85/5.82 | mDD p5/p50/p95 = -15.7%/-9.9%/-5.9% | breakeven 8093bps (c
  6 predictive_model_eval    PASS    [model=ols] purged-CV IC=0.875 | wf_eval pnl_t=26.31 hit=0.847 | sign acc=0.847 | DSR p=0.000
  7 cross_sectional          SKIPPED not run: optional stage (no panel data)
```

How to read: every serial stage is PASS and there is no panel data —
stage 7 is SKIPPED as optional (`killed_by_stage=None`, which is why the
SKIPPED summary says "optional stage", not "killed earlier"). SURVIVES
here means "not killed yet", not "proven": weak edge quality on your
real data will look far more modest than these synthetic numbers, and
that is normal.

### Example 2: pure noise → killed at stage 1

```python
import numpy as np
import pandas as pd

from atropos.ensemble_pipeline import run_full_pipeline

rng = np.random.default_rng(11)
n_obs = 900
idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
data = {
    "returns": pd.Series(rng.standard_normal(n_obs), index=idx),
    "signal": pd.Series(rng.standard_normal(n_obs), index=idx),
    "hypothesis": "claimed predictor - actually pure noise",
}
res = run_full_pipeline(data)
print("verdict      :", res["overall_verdict"])
print("killed_by    :", res["killed_by_stage"])
print("kill_reason  :", res["kill_reason"])
s1 = res["stages"][0]
print(s1["summary"])
for name, ctl in s1["results"].items():
    print(f"  {name:<24} {ctl['verdict']:<13} p={ctl['p_value']:.4f}")
```

Example output:

```text
verdict      : REPACKAGING
killed_by    : 1
kill_reason  : the new signal adds nothing beyond known momentum / known features
0/6 controls passed; failed: shuffled_permutation, placebo_placebo, momentum_control, known_features_control, subperiod_stability, calendar_control
  shuffled_permutation     FAIL          p=0.4776
  placebo_placebo          FAIL          p=0.6219
  momentum_control         FAIL          p=0.4803
  known_features_control   FAIL          p=0.4803
  subperiod_stability      FAIL          p=0.2090
  calendar_control         FAIL          p=0.5000
```

How to read: noise passes no control (p-values ~0.5 — exactly what
should happen under the null). The stage-1 verdict is refined: if
`momentum_control` or `known_features_control` is among the failures,
the verdict is `REPACKAGING`, otherwise `ARTIFACT`. Noise fails those
too (it has no incremental t after controls), hence REPACKAGING.
Stages 2–7 are SKIPPED ("not run: candidate killed earlier"): the
pipeline kills on the first FAIL and does not spend resources on a
corpse.

### Example 3: COST_DEATH vs RISK — the stage-5 matrix

The same real edge (example 1), two different economies. Stage 5
computes the bootstrap envelope of Sharpe and max drawdown net-of-cost
and attributes the breach with a matrix:

- **An mDD breach** (the drawdown tail is worse than the limit) → always
  `RISK`;
- **The gross Sharpe floor is also breached** (no edge even before
  costs) → `RISK`;
- **Only the net Sharpe is breached while the gross passes and the mDD
  tail is intact** → `COST_DEATH` (`cost_breach=True` in the stage's
  `results`).

That is, the mDD breach dominates: COST_DEATH is possible only with an
intact drawdown tail.

```python
from atropos.ensemble_pipeline import run_full_pipeline

# ... make_planted_edge as in example 1 ...
edge = make_planted_edge()

# (a) the same edge but thin returns (~2bp/day) + prohibitive 140bps
thin = dict(edge)
thin["returns"] = edge["returns"] * 0.02
res_cost = run_full_pipeline(thin, {"cost": {"cost_bps": 140.0}})
print("COST_DEATH case")
print("  verdict    :", res_cost["overall_verdict"],
      "| killed_by:", res_cost["killed_by_stage"])
s5 = res_cost["stages"][4]
print("  stage 5    :", s5["status"], "|", s5["summary"])
print("  cost_breach:", s5["results"]["cost_breach"])

# (b) full edge amplitude but a strict risk mandate: mDD tail no worse than -5%
res_risk = run_full_pipeline(edge, {"mc": {"worst_allowed_mdd_p5": -0.05}})
print("RISK case")
print("  verdict    :", res_risk["overall_verdict"],
      "| killed_by:", res_risk["killed_by_stage"])
s5 = res_risk["stages"][4]
print("  stage 5    :", s5["status"], "|", s5["summary"])
print("  cost_breach:", s5["results"]["cost_breach"])
```

Example output:

```text
COST_DEATH case
  verdict    : COST_DEATH | killed_by: 5
  stage 5    : FAIL | FAILED: net SR p5=-0.52 <= floor 0.00 while gross p5=3.85 (cost 140bps vs breakeven 162bps)
  cost_breach: True

RISK case
  verdict    : RISK | killed_by: 5
  stage 5    : FAIL | FAILED: mDD p5=-15.7% < limit -5.0%
  cost_breach: False
```

How to read: in case (a) the gross envelope passes with a huge margin
(gross p5=3.85) — the edge is real, but 140bps one-way eat it entirely
(net p5=-0.52). The cause is lethal but "economic": `COST_DEATH` says
"fix the execution, not the signal". In case (b), at default costs, the
drawdown tail (mDD p5=-15.7%) breaches the -5% mandate —
`cost_breach=False`, verdict `RISK`: even a cost-free edge is
incompatible with this risk mandate. Same stage, opposite prescriptions:
that is the point of cause attribution inside stage 5.

### Example 4: REPACKAGING at stage 1 vs stage 2

Refining the `REPACKAGING` verdict (see example 2): it is issued both
when the stage-1 factor controls fail (`momentum_control` /
`known_features_control`) and when the stage-2 specification curve dies
("no reasonable specification holds |t| >= 2" → repackaging of known
information). Telling them apart is simple: `killed_by_stage` is 1 or 2;
`stages[0]["repackaging"]` holds the bool flag for the first case.

## The conservative philosophy in one paragraph

A stage PASS is not confirmation of the hypothesis but the absence of a
kill at that stage; `SURVIVES` = "not killed yet". INCONCLUSIVE ("could
not check": collinearity, too few panel dates, degenerate input) is not
a kill: missing data does not bury a hypothesis, it leaves it open until
a new snapshot. The FPR controls are calibrated (~0.95 fail rate under
pure noise), and the stages are ordered defense-in-depth from cheap
kills to expensive ones. The first FAIL decides everything — everything
after it is SKIPPED, so no "almost passed" illusion is created.

## What next

- How to get an ML verdict and run both chains at once — how-to 05
  (`docs/how-to/05-run-ml-kill-protocol.md`).
- How to turn a verdict into a tombstone — how-to 03
  (`docs/how-to/03-read-graveyard.md`).
- How to extend the stage-1 null controls — how-to 04
  (`docs/how-to/04-add-null-control.md`).
- Reference: `docs/api/atropos/ensemble_pipeline.html`,
  `DEFAULT_CONFIG` — same place; the ML engine —
  `docs/api/atropos/ml_training.html`.
