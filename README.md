# Atropos

Atropos is a hypothesis-killing **library** for quantitative research:
you import it into your own research pipeline, call its functions on
your data, and get back verdict dicts you act on. A candidate signal
enters a 7-stage adversarial pipeline — null controls,
specification curve, robustness battery, walk-forward, Monte Carlo risk,
predictive evaluation, cross-sectional tests — and leaves with exactly
one verdict. The goal is not to confirm your idea but to kill it
cheaply: thresholds are frozen in a preregistration before the data is
touched, and every verdict is written down. Verdicts: `SURVIVES`,
`SURVIVES_WITH_WARNINGS`, `REPACKAGING`, `ARTIFACT`, `DEAD`,
`NO_OOS_EDGE`, `COST_DEATH`, `RISK`, `CS_NO_EDGE`; stages report
`PASS`/`WARN`/`FAIL`/`SKIPPED`/`INCONCLUSIVE`. Survival is not the end:
`atropos.track_record.live_verdict` judges live PnL against the prereg.

## Install

Python 3.11+; numpy/pandas/scipy only. `uv sync` — or `pip install -e .`

Fresh repo? `atropos init` generates the hypothesis workspace
(`GRAVEYARD.md`, `hypotheses/INDEX.csv`) — existing files are never overwritten.

## First verdict in five minutes

A "fresh" signal that is really just observed momentum plus noise —
killed at stage 1 (run from the repo root with `uv run python`):

```python
import numpy as np
import pandas as pd
from atropos import run_full_pipeline

rng = np.random.default_rng(4)
n_obs = 900
mom = np.zeros(n_obs)
for pos in range(1, n_obs):
    mom[pos] = 0.5 * mom[pos - 1] + rng.standard_normal()
mom /= mom.std()

signal = mom + 0.4 * rng.standard_normal(n_obs)  # "new alpha"
returns = 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
features = pd.DataFrame({"mom_noisy": mom + 0.1 * rng.standard_normal(n_obs)})

res = run_full_pipeline(
    {"signal": signal, "returns": returns, "features": features}
)
print(res["overall_verdict"], "-", res["kill_reason"])
# REPACKAGING - the new signal adds nothing beyond known momentum / known features
```

Fuller walkthrough (prereg first, two worlds, verdict dissection):
`notebooks/01-quickstart.ipynb`.

## Kill it with a model too

The ML wave adds a second killer under the same doctrine.
`atropos.ml_training.run_ml_kill_protocol` walks an expanding-block
walk-forward (purge + embargo) over a dataset you build — a feature
frame, a 0/1 target, per-event executable returns, int64-microsecond
timestamps — fits any model from the open `MODEL_REGISTRY` (gbm,
logit, rf, extratrees, hist_gbm, mlp, knn, qda; `import
atropos.ml_ensemble` adds a stacked `ensemble`), and gates it with
configurable checks: incremental EV vs a linear battery, cost line,
importance stability, deflated Sharpe. Verdicts: `SURVIVES`,
`DEAD(<gate>)`, `UNDERPOWERED` — undecided, reported, not a kill.
A planted nonlinear edge for the engine — a quadrant interaction
`f1*f2` hidden in the exec returns, the DSR gate off because the toy
sample spans too few days to power it:

```python
import numpy as np
import pandas as pd
from atropos.ml_training import run_ml_kill_protocol

rng = np.random.default_rng(0)
hidden = rng.standard_normal((600, 3))
features = pd.DataFrame(
    {"f1": hidden[:, 0], "f2": hidden[:, 1], "f3": hidden[:, 2]}
)
# planted nonlinear edge: an f1*f2 interaction drives the exec return
exec_ret = 0.01 * np.sign(
    hidden[:, 0] * hidden[:, 1]
) + 0.003 * rng.standard_normal(600)
report = run_ml_kill_protocol(
    features,
    (exec_ret > 0.004).astype(np.int64),  # 0/1 target
    exec_ret,  # per-event returns, net of costs
    np.arange(600, dtype=np.int64) * 3_600_000_000,  # us stamps
    config={
        "k_blocks": 3,
        "bootstrap_b": 100,
        "n_min_events": 5,
        "checks": {"deflated_sharpe": False},
        "model_params": {"gbm": {"n_estimators": 25}},
    },
)
print(report["verdict"], round(report["ev_model"], 4))
# SURVIVES 0.0096 - the gbm's taken trades clear every enabled gate
```

One call judges both chains: `atropos.kill_chain.run_full_kill_chain`
takes `rule_data` plus an optional `ml_dataset` and returns rule, ml
and unified verdicts — KILLED if either chain kills, SURVIVES only if
both survive, INCONCLUSIVE when the data cannot decide.

## Demos

```bash
uv run python -m atropos.ensemble_pipeline   # 10 cases, verdict matrix
uv run python -m atropos.track_record        # live PnL vs prereg
uv run python -m atropos.ml_training         # planted edge, 8-model registry, documented-kill reproduction (VRP panic, calibrated to a closed-corpus tombstone)
uv run python -m atropos.kill_chain          # rule x ML unified verdict
```

Every module ships a runnable demo: `uv run python -m atropos.<module>`.

## Read next

1. `docs/how-to/` — step-by-step guides, including the ML kill
   protocol (`run_ml_kill_protocol`).
2. `GLOSSARY.md` — prereg, kill criteria, COD, graveyard.
3. `FRAMEWORK.md` §1-3 — philosophy, pipeline, six laws.

Committed demo transcripts (what the demos actually print) live in
`docs/examples/`.

## For AI agents

`atropos agent-setup` installs the agent context (AGENTS.md + skills);
`atropos agent-context` prints it; `atropos agent-doctor` verifies it.

## Contributing

Gates: `ruff`, `mypy --strict`, AST convention tests,
`uv run python -m pytest` — see `CONTRIBUTING.md`; developer rulebook:
`contributor/AGENTS.md`. License: MIT (`LICENSE`).
