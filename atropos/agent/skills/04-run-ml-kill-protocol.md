---
name: 04-run-ml-kill-protocol
description: Run an ML hypothesis (features→target) through atropos.ml_training.run_ml_kill_protocol and read the verdict SURVIVES / DEAD(<gate>) / UNDERPOWERED.
when_to_use: for ML hypotheses: features→target, after prereg
version: 0.1.0
---

# SKILL 04: RUN ML KILL PROTOCOL — the walk-forward ML chain

> Core skill of the Atropos agent. Public API:
> `atropos.ml_training.run_ml_kill_protocol(features, target,
> exec_returns, times_us, *, config=None, battery_controls=None)
> -> dict`. Expanding-block walk-forward CV with purge + embargo;
> every model from the open `MODEL_REGISTRY` (8 built-ins +
> `"ensemble"` after `import atropos.ml_ensemble`) goes through one
> path fit → OOS probabilities → threshold → EV and the four gates.

## WHEN THE ML CHAIN INSTEAD OF RULE-BASED

The rule chain (`02-run-pipeline`) tests a RULE: signal → outcome.
The ML chain is for a hypothesis formulated as a CLASSIFIER:
features → binary event (`target` 0/1), the decision a probability
threshold over the EV of taken trades. The discipline does not
change: graveyard check and prereg before data are the same skills
(`03`, `01`); only the run engine changes. The ML chain does NOT
replace the rule chain "on top": these are two separate runs of one
hypothesis, each with its own verdict.

## PROCEDURE

1. **Prereg before data.** `from atropos import generate_prereg`;
   kill criteria are the engine gates as numbers (profit_bar, the DSR
   minimum, the event floor), frozen before the data is pulled.
2. **Graveyard check.** Skill `03-read-graveyard`: search by
   prediction + kill criterion, not by title.
3. **Assemble the dataset** (a caller-defined contract): `features`
   (DataFrame, chronological rows), `target` (0/1 int),
   `exec_returns` (executable per-event returns; costs ALREADY netted
   by the caller), `times_us` (int64 microseconds, monotone) — one
   row per event. `battery_controls` — columns of known linear
   controls for the incremental gate's linear-battery comparator.
4. **One run.** `config` is merged over `DEFAULT_TRAINING_CONFIG`
   (key-wise); thresholds/gates from the prereg. Multiple models
   (`models=[...]`) = one CV pass and a COMPARISON in
   `report["models"]` — the engine never picks a winner.
5. **Read the verdict.** `report["verdict"]` — one of the three below;
   per-model details in `report["models"][name]`.

```python
from atropos.ml_training import run_ml_kill_protocol

report = run_ml_kill_protocol(
    features, target, exec_returns, times_us,       # dataset from step 3
    config={"profit_bar": 0.004},                   # on top of DEFAULT
    battery_controls=["ctrl1", "ctrl2"],            # known controls
)
print(report["verdict"], report["n_events"], sorted(report["models"]))
```

## GATES (4) — WHAT EACH ONE CHECKS

| Gate | Check |
|---|---|
| `incremental` | the model's EV on taken trades beats the linear-battery logit (full-procedure bootstrap CI of taken-trade EV vs the linear battery) |
| `cost_line` | merged-OOS EV on taken trades >= `profit_bar` |
| `importance_stability` | per-fold permutation importances rank consistently (median pairwise Spearman >= `importance_rho_min`) |
| `deflated_sharpe` | deflated Sharpe (Bailey & Lopez de Prado 2014) of the daily EV stream >= `dsr_min` |

## VERDICTS (3) — WHAT TO DO FOR EACH

| Verdict | Meaning | Agent action |
|---|---|---|
| `SURVIVES` | all enabled gates passed, enough events | tombstone `SURVIVED`: a conditional candidate with gates, not a leg of a system |
| `DEAD(<gate>)` | the first triggered gate of the first model that failed to survive | tombstone `KILLED`, spell out the gate from the parentheses and record the numbers from `report["kills"]` |
| `UNDERPOWERED` | fewer than `n_min_events` declustered events | NOT a death: the hypothesis is undecided. tombstone `INCONCLUSIVE` with a power plan (more data / a wider snapshot); re-running on the same snapshot is forbidden |

## TRAPS

- Do not tune `threshold_grid` (or any thresholds/gates) after a look
  at the result — that is result shopping; thresholds are frozen in
  the prereg.
- `battery_controls` = the known LINEAR controls from the prereg;
  without them the incremental gate is honestly inconclusive (not a
  kill, but no proof of incrementality either).
- The ensemble is enabled by a side-effect import:
  `import atropos.ml_ensemble` registers `"ensemble"` in
  `MODEL_REGISTRY` plus its defaults; before the import the model is
  not in the registry.
- UNDERPOWERED does not "revive" the hypothesis and does not permit a
  re-run with new thresholds: a new decision = a new prereg (see
  `01`).

## NEXT

Tombstone + a row in `INDEX.csv` — as in `02-run-pipeline.md`.
Need one verdict across the rule+ML chains together —
`atropos.kill_chain.run_full_kill_chain` (a thin aggregate: a kill by
any chain = KILLED; UNDERPOWERED with no death = INCONCLUSIVE).
