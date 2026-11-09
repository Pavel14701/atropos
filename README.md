# Atropos

**Atropos — falsification toolkit for trading hypotheses: backtest kill
protocol, feature decomposition, paper parity, with machine-checkable
instrument routing.**

Three tiers, one doctrine — a hypothesis is guilty until it survives
every attack:

- **Backtest kill protocol** (`atropos.ensemble_pipeline.run_full_pipeline`):
  a candidate signal enters a 7-stage adversarial pipeline — null
  controls, specification curve, robustness battery, walk-forward,
  Monte Carlo risk, predictive evaluation, cross-sectional tests — and
  leaves with exactly one verdict. Thresholds are frozen in a
  preregistration before the data is touched; the kill protocol is the
  core of the backtest tier, and survival is not the end:
  `atropos.track_record.live_verdict` judges live PnL against the
  prereg. For the rare survivor, the reference backtest pipeline
  (`atropos.backtesting`, `atropos.ranking`, `atropos.backtest_metrics`)
  and the portfolio layer (`atropos.portfolio`) ask the next questions —
  tradeable at all, and does the book hold up with concurrent positions.
  The authoritative verdict list lives in
  `atropos/ensemble_pipeline.py`'s module docstring — docs must not
  re-enumerate it; every report also carries `verdict_class`
  (DEAD / NEEDS_ANALYSIS / INCONCLUSIVE / SURVIVES), `informative`,
  `interpretation`, `next_step`.
- **Feature decomposition** (`atropos.orthogonal.orthogonal_check`,
  orchestrated by `atropos.feature_scan.feature_analysis`): does the
  candidate carry information beyond known controls — decomposition
  orthogonal to the kill chain, not a stage of it.
- **Paper parity** (`atropos.paper_protocol.run_paper_protocol`): the
  forward test of the IMPLEMENTATION — does paper execution match the
  frozen spec? A `PARITY_FAIL` indicts the paper engine or its feed,
  never the hypothesis.

## Instrument routing

A strategy family dictates which instruments — battery tools and
pipeline stages — may judge it. Strategy families map to batteries in
`atropos/instrument_map.py`; the family is declared in the prereg/config
as `strategy_family`, and the mapping is machine-checkable: enforced in
the pipeline when declared (a forbidden tool or stage raises), flagged
in audit mode otherwise. See skill 19
(`atropos agent-context --skill=19-instrument-routing`).

## Install

`pip install atropos-quant` — Python 3.12; numpy/pandas/scipy/scikit-learn (lightgbm optional for the `ranking` extra). From a clone: `uv sync` — or `pip install -e .`

Fresh repo? `atropos init` generates the hypothesis workspace
(`GRAVEYARD.md`, `hypotheses/INDEX.csv`) — existing files are never overwritten.

> **Name note:** the DISTRIBUTION is `atropos-quant` (`pip install atropos-quant`), the IMPORT is `atropos` (`from atropos import ...`) -- same split as scikit-learn/sklearn or beautifulsoup4/bs4.  The CLI command is `atropos-quant` (the plain `atropos` command belongs to an unrelated bioinformatics package).

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
uv run python -m atropos.ml_training         # planted edge, 9-model registry (8 base + ensemble), documented-kill reproduction (VRP panic, calibrated to a closed-corpus tombstone)
uv run python -m atropos.kill_chain          # rule x ML unified verdict
uv run python -m atropos.portfolio           # multi-slot replay, correlation cap, portfolio gates
```

Most modules ship runnable demos: `uv run python -m atropos.<module>`.

The backtest layer's golden path — one script wiring the full reference
pipeline: `docs/examples/backtest_pipeline.py`
(`uv run python docs/examples/backtest_pipeline.py`): 161 candidate
signals → 63 slots → 56 ranked groups, mrr 0.778 vs 0.556 (gbm vs
random, deterministic seed).

## Read next

1. `POSITIONING.md` — what Atropos is, what it is not, why
   "falsification toolkit" and not "compiler".
2. `docs/methodology.md` — the method → single-home table: every
   statistical claim the docs make lives in exactly one function.
3. `docs/how-to/` — step-by-step guides, including the ML kill
   protocol (`run_ml_kill_protocol`), the backtest pipeline
   (how-to 06-08) and the portfolio layer (how-to 09).
4. `GLOSSARY.md` — prereg, kill criteria, COD, graveyard, the backtest
   pipeline's vocabulary.
5. `FRAMEWORK.md` §1-3 — philosophy, pipeline, six laws.
6. `docs/calibration.md` — how the portfolio kill gates are calibrated
   on public data, and the blind-run protocol for private systems.

Committed demo transcripts (what the demos actually print) live in
`docs/examples/`.

A note on STRATEGY_MAP: the shipped `atropos/instrument_map.py` registry
covers the shipped families; a deployed system carries its own map —
declare its family in the prereg/config (`strategy_family`) and route
against its own registry.

## For AI agents

`atropos agent-setup` installs the agent context (AGENTS.md + skills);
`atropos agent-context` prints it; `atropos agent-doctor` verifies it.
Skills 19-21 cover instrument routing, paper parity and units — see
`atropos agent-context --list-skills`.

## Contributing

Gates: `ruff`, `mypy --strict`, AST convention tests,
`uv run python -m pytest` — see `CONTRIBUTING.md`; developer rulebook:
`contributor/AGENTS.md`. License: MIT (`LICENSE`).
