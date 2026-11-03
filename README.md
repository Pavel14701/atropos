# Atropos

Atropos is a hypothesis-killing framework for quantitative research: a
candidate signal enters a 7-stage adversarial pipeline — null controls,
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

## Demos

```bash
uv run python -m atropos.ensemble_pipeline   # 10 cases, verdict matrix
uv run python -m atropos.track_record        # live PnL vs prereg
```

Every module ships a runnable demo: `uv run python -m atropos.<module>`.

## Read next

1. `docs/how-to/` — step-by-step guides.
2. `GLOSSARY.md` — prereg, kill criteria, COD, graveyard.
3. `FRAMEWORK.md` §1-3 — philosophy, pipeline, six laws (in Russian;
   code and demos are English).

## For AI agents

`atropos agent-setup` installs the agent context (AGENTS.md + skills);
`atropos agent-context` prints it; `atropos agent-doctor` verifies it.

## Contributing

Gates: `ruff`, `mypy --strict`, AST convention tests,
`uv run python -m pytest` — see `CONTRIBUTING.md`; developer rulebook:
`contributor/AGENTS.md`. License: MIT (`LICENSE`).
