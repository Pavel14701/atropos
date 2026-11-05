# How-to 05. The ML kill protocol: walk-forward gates over a feature dataset

> **Estimated time:** 30–40 minutes (the snippet runs take ~2 minutes in
> total)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.11+; scikit-learn comes as a project dependency.
> Bring your own hypothesis data — here it is a synthetic seeded world
> (reproducible byte-for-byte). The snippets run in sequence in one
> session: from step 4 on they reuse the variables of step 3.

## When to use the ML chain, not rule-based

The framework kills hypotheses with two chains. Rule-based
(`run_full_pipeline`, how-to 02) takes a finished rule — `signal` →
outcome — and runs it through seven adversarial stages. The ML chain
(`run_ml_kill_protocol`) takes a feature space — `features` → a binary
`target` — and checks whether it produces executable trades that survive
four gates in expanding-block walk-forward CV (purge + embargo between
train and test).

| | rule-based (`run_full_pipeline`) | ML (`run_ml_kill_protocol`) |
|---|---|---|
| Input | `signal` + `returns`: a rule → an outcome | `features` (a DataFrame) + a 0/1 `target`: features → a classifier |
| What is frozen in the prereg | the rule construction: thresholds, windows, normalization | the probability-threshold grid, the gates and their thresholds, the economics (`profit_bar`, costs), the model and its parameters |
| Who decides | 7 stages: null controls, the specification curve, the robustness battery, walk-forward, the MC envelope, purged CV, cross-sectional | 4 gates over the out-of-sample book: `incremental`, `cost_line`, `importance_stability`, `deflated_sharpe` |
| Typical question | "is this signal tradeable net-of-cost OOS?" | "is there signal in the features beyond the linear battery, and does it survive all the gates?" |
| Verdicts | 9: `SURVIVES` … `CS_NO_EDGE` | 3: `SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED` |

Rule of thumb: if the hypothesis is stated as a concrete entry/exit rule
— rule-based; if the effect is nonlinear by construction (feature
interactions, regimes) or no rule is formulated yet and there are only
features — the ML chain. Unsure — run both: `run_full_kill_chain`
(section below) aggregates the two verdicts, and a kill by either chain
kills the hypothesis.

Self-contained: no need to read `FRAMEWORK.md` before this how-to.

Import (the `atropos` package does not re-export functions at the top
level — use module paths):

```python
from atropos.ml_training import run_ml_kill_protocol
```

## Step 1. Preregistration — the same discipline

An ML hypothesis is registered with the same `generate_prereg` as a
rule-based one (full walkthrough — how-to 01), before the first data
touch. A convenient anchor: the engine's four gates are ready-made kill
criteria — just copy them into the prereg verbatim, with the default
thresholds from `DEFAULT_TRAINING_CONFIG`.

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

params = {
    "name": "quadrant_interaction_ml",
    "hypothesis": (
        "A boosted-tree classifier on hourly features f1..f4 finds "
        "executable long entries whose taken-trade EV clears the "
        "40bp profit bar after 20bp round-trip costs, incrementally "
        "to a linear battery on known controls."
    ),
    "construction": [
        "features f1..f4, causal transforms only; target = exec > 40bp",
        "exec = fwd_return - 20bp; walk-forward k_blocks=6, purge 2h, "
        "embargo 24h",
    ],
    "nulls": [
        "the edge is linear: the logit battery on ctrl1/ctrl2 trades "
        "the same book",
        "per-fold feature rankings are noise (unstable importance)",
    ],
    "kill_criteria": [
        {
            "criterion": "paired bootstrap t of taken-trade EV vs "
                         "linear battery",
            "comparison": "<",
            "threshold": 2.0,
        },
        "OOS EV of taken trades < 0.0040",
        "median pairwise Spearman of per-fold importances < 0.5",
        "deflated Sharpe of the daily EV stream < 0.95",
    ],
    "power_calc": {"N_required": 500, "alpha": 0.05, "power": 0.80},
    "hash": "b2c3d4e5",
}

problems = validate_prereg(params)
print(problems)
assert problems == []
doc = generate_prereg(params, date="2026-11-05")
print("\n".join(doc.splitlines()[:4]))
```

Example output:

```text
[]
# Preregistration: quadrant_interaction_ml

- **Frozen on:** 2026-11-05
- **Spec hash:** `b2c3d4e5`
```

Note: `N_required: 500` in the power calc is the same floor of
independent events as the engine's `n_min_events: 500`; the four kill
criteria mirror the four gates of step 5. Commit the
`PREREG_<name>_<date>.md` file before the data pull (step 4 of how-to
01).

## Step 2. The graveyard check

The graveyard is checked before the hypothesis is proposed (how-to 03).
If there is no layout yet — generate the skeleton: the command creates
only missing files and never overwrites existing ones.

```bash
uv run atropos init
grep -i "quadrant\|classifier\|ensemble" hypotheses/INDEX.csv \
  || echo "no matches: graveyard check clean"
```

Example output on an empty layout (Windows; POSIX paths use `/`):

```text
  created GRAVEYARD.md
  created hypotheses\INDEX.csv
  created hypotheses\README.md
init: workspace ready; files are never overwritten - edit them manually after init.
no matches: graveyard check clean
```

A repeated `uv run atropos init` on the same layout prints
`skipped (exists)` for each of the three files and changes nothing — the
command is idempotent (ran). The search result ("FOUND_NOTHING") goes
into the prereg.

## Step 3. Assemble the dataset: the contract + mini synthetics

The dataset is four components, validated before any `fit` (the
`ValueError` names the component):

| Component | Contract | Example validation error |
|---|---|---|
| `features` | a DataFrame of float columns, rows in chronological order; columns listed in `battery_controls` are excluded from the models' features and fed to the linear battery | `feature columns must be finite floats (no NaN/inf)` |
| `target` | 1-D integers, exactly 0/1 — one row per event | `target must be binary 0/1, got [0, 2]` |
| `exec_returns` | 1-D finite floats; costs are **subtracted by the caller** (`exec = fwd - cost`; `cost_rt` in the config is a reference entry for the caller, the engine itself subtracts nothing) | `exec_returns must be 1-D finite floats` |
| `times_us` | int64 microseconds, monotonically non-decreasing (the walk-forward axis) | `times_us must be monotonically non-decreasing` |
| `battery_controls` | names of `features` columns | `battery_controls not found in features: ['mom30']` |

Duplicate stamps are not an error: they collapse when counting
independent events — `decluster_count` counts clusters with a gap of at
least an hour, not rows.

This how-to's mini synthetics: hourly µs stamps, a planted
nonlinearity — the interaction `tanh(f1*f2/2)`, invisible to any linear
model — plus a soft linear `f3` (the battery sees it) and noise.

```python
import numpy as np
import pandas as pd


def build_world(n_rows: int, seed: int = 2026):
    """Seeded hourly world: planted tanh interaction + linear + noise."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    exec_returns = (
        0.012 * np.tanh(raw[:, 0] * raw[:, 1] / 2.0)
        + 0.0035 * raw[:, 2]
        + 0.0024 * rng.standard_normal(n_rows)
    )
    features = pd.DataFrame(
        {
            "f1": raw[:, 0], "f2": raw[:, 1],
            "f3": raw[:, 2], "f4": raw[:, 3],
            "ctrl1": controls[:, 0], "ctrl2": controls[:, 1],
        }
    )
    times_us = (
        int(np.datetime64("2024-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_rows, dtype=np.int64) * 3_600_000_000
    )
    target = (exec_returns > 0.004).astype(np.int64)  # exec > profit_bar
    return features, target, exec_returns, times_us


features, target, exec_returns, times_us = build_world(1500)
print(features.shape, round(float(target.mean()), 3),
      target.dtype, bool((np.diff(times_us) >= 0).all()))
```

Example output:

```text
(1500, 6) 0.264 int64 True
```

Your own data must satisfy the same contract: every row is an event with
a known exec outcome; the typical mistake is forgetting to subtract
costs from `exec_returns` (the engine will see an edge that does not
exist: the gates are calibrated for already-net returns).

Validation before fit — four typical errors (run on the world above):

```python
cases = {}
with_nan = features.copy()
with_nan.loc[3, "f1"] = np.nan
cases["NaN feature"] = (with_nan, target, exec_returns, times_us, None)
cases["non-binary target"] = (
    features, target.astype(float) * 2, exec_returns, times_us, None)
bad_times = times_us.copy()
bad_times[10] = bad_times[9] - 1
cases["non-monotone times"] = (
    features, target, exec_returns, bad_times, None)
cases["missing control"] = (
    features, target, exec_returns, times_us, ["mom30"])
for label, (feats, tgt, ex, ts, bat) in cases.items():
    try:
        run_ml_kill_protocol(feats, tgt, ex, ts, battery_controls=bat)
    except ValueError as err:
        print(f"{label}: {err}")
```

Example output:

```text
NaN feature: feature columns must be finite floats (no NaN/inf)
non-binary target: target must be binary 0/1, got [0, 2]
non-monotone times: times_us must be monotonically non-decreasing
missing control: battery_controls not found in features: ['mom30']
```

## Step 4. One run

```python
report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict     :", report["verdict"])
print(f"n_rows={report['n_rows']}  n_events={report['n_events']}  "
      f"(min {report['n_min_events']}, powered={report['powered']})")
print("EV model/baseline/battery (bp):",
      round(report["ev_model"] * 1e4, 1),
      round(report["ev_baseline"] * 1e4, 1),
      round(report["ev_battery"] * 1e4, 1))
print("DSR:", report["deflated_sharpe"],
      " rho:", report["importance_stability"])
for gate in report["kills"]:
    print(f"  {gate['check']:<20} enabled={gate['enabled']} "
          f"triggered={gate['triggered']}")
for fold in report["folds"]:
    print(f"  fold {fold['fold']}: train={fold['train_rows']} "
          f"test={fold['test_rows']} "
          f"purge+embargo dropped={fold['n_dropped_purge_embargo']}")
```

Example output (~2 seconds):

```text
verdict     : UNDERPOWERED
n_rows=1500  n_events=249  (min 500, powered=False)
EV model/baseline/battery (bp): 72.7 59.3 61.8
DSR: 0.9999999825663033  rho: 0.7999999999999999
  incremental          enabled=True triggered=False
  cost_line            enabled=True triggered=False
  importance_stability enabled=True triggered=False
  deflated_sharpe      enabled=True triggered=False
  fold 1: train=226 test=250 purge+embargo dropped=23
  fold 2: train=476 test=250 purge+embargo dropped=23
  fold 3: train=726 test=250 purge+embargo dropped=23
  fold 4: train=976 test=250 purge+embargo dropped=23
  fold 5: train=1226 test=251 purge+embargo dropped=23
```

How to read. The CV structure at `k_blocks=6`: five test blocks of 250
rows each; the oldest block is never tested (it has nothing to train
on); before each test block, purge + embargo drop 23 train rows. All
four gates passed — but there are only 249 independent events < 500:
nothing to decide with, verdict `UNDERPOWERED`. This is **neither death
nor survival** (the table in step 5): at an event shortfall the gate
flags are informational. The honest answer is to add data and re-run.

With a multi-model run (`"models": [...]`) the `powered` flag speaks
for the whole run: it requires EVERY registry model to clear the event
floor, so one thin secondary (say, 21 events) flags the run as
undecided even when the primary cleared it comfortably with 1169
events. The demo output names the offending models explicitly;
per-model `SURVIVES` verdicts stay per-model — never cherry-pick a
survivor out of an undecided run.

One reading note on `deflated_sharpe`: it is a CDF probability in
[0, 1]. A value of exactly `1.0` is the saturation cap (z large enough
that the normal CDF rounds to one — common for strong demo worlds),
and tiny values like `1e-129` are honest extreme CDF-tail values for
a dead book -- finite and representable, not numerical faults.

### Add data and re-run

```python
features, target, exec_returns, times_us = build_world(6000)

report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict     :", report["verdict"])
print(f"n_rows={report['n_rows']}  n_events={report['n_events']}  "
      f"(min {report['n_min_events']}, powered={report['powered']})")
print("EV model/baseline/battery (bp):",
      round(report["ev_model"] * 1e4, 1),
      round(report["ev_baseline"] * 1e4, 1),
      round(report["ev_battery"] * 1e4, 1))
print("DSR:", report["deflated_sharpe"],
      " rho:", report["importance_stability"])
print("incremental :", report["incremental"])
print("baseline    :", report["baseline"])
print("battery     :", report["battery"])
print("dsr_detail  :", report["dsr_detail"])
```

Example output (~6 seconds):

```text
verdict     : SURVIVES
n_rows=6000  n_events=843  (min 500, powered=True)
EV model/baseline/battery (bp): 80.7 70.3 70.3
DSR: 1.0  rho: 1.0
incremental : {'gbm': {'t_vs_battery': 139.04437804450353, 't_vs_baseline': 141.4843135373026, 'skipped_draws': 0}}
baseline    : {'ev': 0.007034936819590529, 'n_events': 193}
battery     : {'available': True, 'ev': 0.007034659597109983, 'n_events': 191}
dsr_detail  : {'n_days': 209, 'var_sr': 0.44557527574926054, 'sr': {'gbm': 1.9903409535233292, 'baseline': 0.8501631731846993, 'battery': 0.8188189422471551}}
```

How to read: 843 independent taken trades — the 500 floor is cleared,
the gate decisions have something to stand on. `incremental` is the
paired-bootstrap t of the EV difference against the battery: 139.0
against a threshold of 2.0 (on a synthetic world the margin is huge; on
real data it will be orders of magnitude more modest — and that is
normal). `deflated_sharpe` is computed on the daily EV stream: 209
days, with the multiple-testing discount coming from the SR variance of
the whole family of the run's streams (`var_sr`, `dsr_detail`). The
battery and baseline in this report are logits (features + controls and
features without controls, respectively); their narrow books (191/193
events) are the norm: a logit cannot cut out a high-confidence
sub-book.

The report carries 18 keys: verdicts (`verdict`, `kills`, `models`,
`powered`, `n_min_events`), scale (`n_rows`, `n_events`, `folds`),
linear comparators (`baseline`, `battery`, `incremental`, `ev_model`,
`ev_baseline`, `ev_battery`), the DSR family (`dsr_detail`,
`deflated_sharpe`, `importance_stability`), and a json-safe copy of the
actual config (`config`). The flat keys alias the primary model
(`config["model"]`); the other models' verdicts live in
`report["models"][name]` — the engine never chooses between them (see
the "Models" section).

## Step 5. Interpretation

### The three verdicts

| Verdict | Issued by | What it means | What to do |
|---|---|---|---|
| `SURVIVES` | every requested model passed all enabled gates at `n_events >= n_min_events` | "not killed yet" — not confirmation, but the absence of a kill | a SURVIVED tombstone with the numbers; monitor on new snapshots |
| `DEAD(<gate>)` | the first triggered gate of the first non-surviving model | the edge did not survive this gate; the name in brackets is the cause | read the gate table: a lethal cause or an "economic" one; resurrection conditions in the tombstone |
| `UNDERPOWERED` | the model has `n_events < n_min_events` (default 500) | **undecided, not a kill**: the sample cannot power any gate decision, gate flags are informational | add data (step 4: 249 events → a re-run on 6000 rows → a decidable run); in the unified matrix this is INCONCLUSIVE |

### The four gates

| Gate | What it catches | Configuration (`checks.*` and nearby) |
|---|---|---|
| `incremental` | repackaging of the linear part: the paired-bootstrap t of the EV difference between the model's taken trades and the logit battery (features + your `battery_controls`) not above the threshold | `incremental: true`, threshold `incremental_t` (2.0) |
| `cost_line` | a book that fails to pay its costs: merged-OOS EV of taken trades below the bar | `cost_line: true`, `profit_bar` (0.0040 = 40bp) |
| `importance_stability` | "signal out of noise": per-fold permutation rankings of the features do not reproduce (the median pairwise Spearman below the bar) | `importance_stability: true`, `importance_rho_min` (0.5) |
| `deflated_sharpe` | multiple testing: the deflated Sharpe (Bailey & Lopez de Prado 2014) of the daily EV stream below the floor | `deflated_sharpe: true`, `dsr_min` (0.95), `n_trials_ds` (5) |

Each gate entry is `{check, enabled, triggered, inconclusive,
reason}`. `INCONCLUSIVE` — "could not check", not a kill. The typical
case: incremental without `battery_controls`.

```python
no_battery = run_ml_kill_protocol(
    features.drop(columns=["ctrl1", "ctrl2"]),
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
)
print("verdict:", no_battery["verdict"], "| battery:", no_battery["battery"])
print(no_battery["kills"][0])
```

Example output:

```text
verdict: SURVIVES | battery: {'available': False, 'ev': None, 'n_events': 0}
{'check': 'incremental', 'enabled': True, 'triggered': False, 'inconclusive': True, 'reason': 'battery controls not supplied: incremental edge over a linear battery is not decidable in this run'}
```

How to read: formally `SURVIVES` — no gate triggered. But incremental is
INCONCLUSIVE: the incremental edge over the linear battery is
**undemonstrated** in this run, not passed. Such a run is unfit as
justification for the hypothesis — drop `battery_controls` deliberately
or do not drop them at all.

### Noise must die: DEAD(<gate>)

```python
rng_noise = np.random.default_rng(7)
noise_exec = 0.008 * np.sign(rng_noise.standard_normal(len(times_us)))
noise_target = (noise_exec > 0.004).astype(np.int64)

noise_report = run_ml_kill_protocol(
    features,
    noise_target,
    noise_exec,
    times_us,
    config={"model": "logit"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict:", noise_report["verdict"])
print("triggered:",
      [g["check"] for g in noise_report["kills"] if g["triggered"]])
print("EV model (bp):", round(noise_report["ev_model"] * 1e4, 2),
      " n_events:", noise_report["n_events"])
```

Example output:

```text
verdict: DEAD(cost_line)
triggered: ['cost_line', 'importance_stability', 'deflated_sharpe']
EV model (bp): -0.17  n_events: 2864
```

How to read: pure noise under a logit trades a wide book (2864 events —
there is power, a fake UNDERPOWERED is ruled out), and an EV of −0.17bp
does not pay the 40bp bar. The verdict names the **first triggered gate
in order** (`incremental` → `cost_line` → `importance_stability` →
`deflated_sharpe`), not the "loudest" cause: incremental did not trigger
for this model/battery pair — both arms trade nearly identical wide
books, and the bootstrap difference of their EVs on this seed came out
slightly in the model's favor. The other three gates catch noise
independently; any one of them is enough for death.

## Models: the registry of 8 + ensemble

| Name | Bias axis |
|---|---|
| `logit` | the linear comparator (also the baseline and battery arm) |
| `gbm` | boosting, exact splits (the engine default) |
| `hist_gbm` | boosting, histogram splits |
| `rf` | bagging, optimized splits |
| `extratrees` | bagging, random splits (decorrelation) |
| `mlp` | a neural network (32×16, adam) |
| `knn` | local neighborhoods (distance-weighted) |
| `qda` | a generative quadratic surface |

The ninth name, `ensemble`, appears in the registry only after importing
`atropos.ml_ensemble` — a documented side effect:

```python
import atropos.ml_ensemble  # noqa: F401  (side effect, see "Traps")
from atropos.ml_training import MODEL_REGISTRY

print(sorted(MODEL_REGISTRY))
```

Example output:

```text
['ensemble', 'extratrees', 'gbm', 'hist_gbm', 'knn', 'logit', 'mlp', 'qda', 'rf']
```

Several models are compared in one CV pass (`models` instead of
`model`); the engine never chooses between them — the verdicts live in
`report["models"]`, and the run verdict echoes the first non-survivor.
The ensemble is configured with flat kwargs in
`model_params["ensemble"]` (members, weights, the combination rule
`mean`/`weighted`/`rank_mean`/`stacking`; honest past-only OOF — see the
module docstring):

```python
report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={
        "models": ["gbm", "ensemble"],
        "model_params": {
            "ensemble": {"members": ["gbm", "rf"],
                         "combination": "mean"},
        },
    },
    battery_controls=["ctrl1", "ctrl2"],
)
print("run verdict:", report["verdict"])
for name, entry in report["models"].items():
    print(f"  {name:<9} {entry['verdict']:<25} "
          f"EV={round(entry['ev'] * 1e4, 1)}bp  "
          f"events={entry['n_events']}")
```

Example output (~15 seconds):

```text
run verdict: SURVIVES
  gbm       SURVIVES                  EV=80.7bp  events=843
  ensemble  SURVIVES                  EV=87.4bp  events=664
```

Config overrides are merged key-by-key at every level: specifying
`{"checks": {"dsr_min": 0.95}}` means explicitly freezing the default
floor without touching the other gates. Example: the same run but with
the `rf` model and its default parameters:

```python
rf_report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "rf", "checks": {"dsr_min": 0.95}},
    battery_controls=["ctrl1", "ctrl2"],
)
print("rf default run:", rf_report["verdict"],
      " n_events:", rf_report["n_events"])

rf_report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={
        "model": "rf",
        "model_params": {"rf": {"min_samples_leaf": 1, "max_depth": 7}},
    },
    battery_controls=["ctrl1", "ctrl2"],
)
print("rf sharp run  :", rf_report["verdict"],
      " n_events:", rf_report["n_events"],
      " EV:", round(rf_report["ev_model"] * 1e4, 1), "bp")
```

Example output:

```text
rf default run: UNDERPOWERED  n_events: 100
rf sharp run  : SURVIVES  n_events: 537  EV: 90.2 bp
```

How to read: the default `rf` (`min_samples_leaf=50`) takes a narrow
book — 100 independent events < 500; `UNDERPOWERED` is not about model
quality but about the power of its book; finer leaves give 537 events
and a decidable run. Model capacity is as much a part of the frozen
construction as the threshold grid: fix it in the prereg.

## Traps

- **Do not tune `threshold_grid` after the data.** The grid
  (0.30…0.70, step 0.05) and the gate thresholds are part of the
  construction frozen in the prereg (step 1). Each fold's threshold is
  chosen honestly on the train fold (argmax EV over the grid); narrowing
  the grid or moving `profit_bar` after a look at OOS is a new
  hypothesis with a new prereg, not an edit of the old one.
- **UNDERPOWERED is not death.** "Undecided": add data and re-run
  (step 4). But it is not survival either: the graveyard gets
  INCONCLUSIVE, the unified matrix yields `INCONCLUSIVE [ml]`; at an
  event shortfall the gate flags are not interpreted.
- **`battery_controls` are known linear controls.** Momentums, known
  factors — what would "explain" the effect linearly. They are excluded
  from the models' feature set and fed to the logit battery; do not put
  your target constructs there. Without them incremental is
  INCONCLUSIVE and the run is formally SURVIVES with an undemonstrated
  incremental edge (the snippet in step 5).
- **Importing `atropos.ml_ensemble` is a side effect.** Without it,
  `model="ensemble"` fails before any fit:
  `ValueError: unknown model 'ensemble': available ['extratrees',
  'gbm', 'hist_gbm', 'knn', 'logit', 'mlp', 'qda', 'rf']` (ran); with
  it, the registry has the ninth name. Keep the import explicit at the
  top of the script. And remember: `rank_mean`/`normalize_scores`
  flatten the score distribution toward uniform — a selective threshold
  stops selecting a narrow book. On this how-to's world, a rank_mean
  ensemble of `[gbm, rf]` trades the whole book (1500 events,
  EV 62.4bp) and dies at `incremental`, while the default `mean` over
  probabilities survives (87.4bp) — the default is no accident (ran;
  see `DEFAULT_ENSEMBLE_CONFIG`).
- **The engine does not pick a model.** `models: [...]` is a comparison,
  not a tournament: each member has its own verdict in
  `report["models"]`, and the run verdict echoes the first
  non-survivor. There is no and can be no "best" model: selecting on
  OOS is the same tuning as editing the grid.

## Both chains at once: run_full_kill_chain

The wrapper always runs the rule pipeline (7 stages) on `rule_data`
(`returns` + `signal`) and, if an `ml_dataset` is given (`features`,
`target`, `exec_returns`, `times_us`), the ML engine on it. It is a thin
aggregator, not a third engine: no new gates, `unified` is a pure
function of the two verdicts (source: `rule` / `ml` / `rule+ml`).

| Rule verdict | ML verdict | unified |
|---|---|---|
| alive | (not run) | `SURVIVES` |
| kill | (not run) | `KILLED` [rule] |
| alive | `SURVIVES` | `SURVIVES` |
| alive | `DEAD(...)` | `KILLED` [ml] |
| alive | `UNDERPOWERED` | `INCONCLUSIVE` [ml] |
| kill | `SURVIVES` | `KILLED` [rule] |
| kill | `DEAD(...)` | `KILLED` [rule+ml] |
| kill | `UNDERPOWERED` | `KILLED` [rule] |

("alive" = `SURVIVES` / `SURVIVES_WITH_WARNINGS`; kill — any other rule
verdict; `[...]` is `unified["source"]`.)

```python
from atropos.kill_chain import run_full_kill_chain

rule_rng = np.random.default_rng(4)
idx = pd.date_range("2024-01-01", periods=900, freq="B")
rule_data = {
    "returns": pd.Series(0.01 * rule_rng.standard_normal(900), index=idx),
    "signal": pd.Series(rule_rng.standard_normal(900), index=idx),
    "hypothesis": "ml how-to: rule chain sanity (pure noise)",
}
ml_dataset = {
    "features": features, "target": target,
    "exec_returns": exec_returns, "times_us": times_us,
}
out = run_full_kill_chain(rule_data, ml_dataset)
u = out["unified"]
print("rule:", out["rule"]["overall_verdict"], "| ml:", out["ml"]["verdict"])
print("unified:", u["verdict"], "[", u["source"], "]", "|", u["reason"])
```

Example output (the guide's longest snippet, ~half a minute):

```text
rule: REPACKAGING | ml: SURVIVES
unified: KILLED [ rule ] | rule chain killed: REPACKAGING
```

How to read: the rule chain killed the noise as repackaging at stage 1;
the ML chain survived on the planted nonlinear world — but a kill by
either chain decides everything: `KILLED [rule]`. To see
`unified SURVIVES`, both chains must be alive (all four quadrants of the
matrix are run and asserted in the module's demo:
`uv run python -m atropos.kill_chain`).

## Module demo: a documented-kill reproduction

`uv run python -m atropos.ml_training` runs five asserted cases; the
fifth is a statistics-calibrated reproduction of a documented
tombstone (closed donor corpus, "VRP-panic"): a synthetic world
calibrated to the tombstone's marginal statistics must come back
`DEAD(...)` with both documented cause-of-death gates triggered
(cost_line + incremental).

The committed transcript of the full module demo (all five cases, including this one) lives in `docs/examples/ml_training_demo.log`; regenerate it with `uv run python -m atropos.ml_training`.

## Checklist

- [ ] The prereg with the four gates as kill criteria is committed
      BEFORE the data (law 1); `validate_prereg` returned `[]`.
- [ ] The graveyard check is written into the prereg (how-to 03).
- [ ] The dataset passed the contract validation: finite float features,
      a binary target, monotone int64 µs; costs subtracted from
      `exec_returns` by the caller.
- [ ] `battery_controls` are known linear controls; incremental in the
      report is not INCONCLUSIVE.
- [ ] `n_events >= n_min_events` — otherwise UNDERPOWERED: add data, do
      not interpret the gate flags.
- [ ] `threshold_grid`, `checks.*`, `model_params` were not changed
      after the first look at OOS (laws 4/5).
- [ ] The verdict is recorded in a tombstone (how-to 03) with the
      numbers: EV, DSR, rho, n_events, and fold statistics — the entry's
      minimal numbers.

## What next

- Reading the rule verdicts — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Tombstones and the graveyard registry — how-to 03
  (`docs/how-to/03-read-graveyard.md`).
- Preregistration — how-to 01 (`docs/how-to/01-preregister.md`).
- Reference: `docs/api/atropos/ml_training.html` (the engine and
  `DEFAULT_TRAINING_CONFIG`), `docs/api/atropos/ml_ensemble.html`,
  `docs/api/atropos/kill_chain.html`, `docs/api/atropos/workspace.html`.
