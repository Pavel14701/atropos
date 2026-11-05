# Atropos Agent Context — package layer (default layer)

How to USE the Atropos library. This is the package layer: the user
layer (`%APPDATA%/atropos/agent` / `~/.config/atropos/agent`) and the
project layer (`./.atropos/agent`) may replace, extend, or disable
these files and skills. Diagnostics: `atropos agent-doctor`.

## Hypothesis workflow (5 steps)

1. **Graveyard check.** Before anything: search `hypotheses/graveyard/`
   and `INDEX.csv` by prediction and kill criterion (not by title).
   A `KILLED` duplicate with no new mechanism is not proposed. Skill:
   `03-read-graveyard`. No graveyard in the project? `atropos init`
   generates the skeleton (existing files are never overwritten).
2. **Preregister before data.** The prereg is frozen BEFORE any touch
   of the data. Kill criteria — hard "operator + number" rules only.
   Skill: `01-preregister`.
3. **Run pipeline.** One run on the frozen spec; the first FAIL decides
   everything. Skill: `02-run-pipeline`.
   **ML hypothesis?** If the hypothesis is a classifier (features →
   target 0/1, threshold + EV) rather than a signal → outcome rule,
   the discipline is the same (graveyard + prereg before data), the
   engine is different: skill `04-run-ml-kill-protocol`
   (`atropos.ml_training.
   run_ml_kill_protocol`): walk-forward CV, 4 gates, verdicts
   `SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED` (undecided — NOT a
   death). One run, one verdict, as in the rule chain.
4. **Interpret.** PASS ≠ confirmation: `SURVIVES` = a conditional
   candidate with gates, not a leg of a system. One run, one verdict.
5. **Record tombstone.** Every hypothesis ends with a tombstone and a
   row in `INDEX.csv`, including your own uncomfortable verdicts.

## API (as implemented)

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

# params-dict of 7 required fields; run validate_prereg(params) first.
doc = generate_prereg(params)  # -> markdown string

from atropos.ensemble_pipeline import run_full_pipeline

# data: returns, signal, features, mom7/mom30, hypothesis,
# panel_signal/panel_returns; config: on top of DEFAULT_CONFIG.
report = run_full_pipeline(data, config)
report["overall_verdict"]  # SURVIVES | SURVIVES_WITH_WARNINGS |
# REPACKAGING | ARTIFACT | DEAD | NO_OOS_EDGE | COST_DEATH | RISK |
# CS_NO_EDGE (+ INCONCLUSIVE stages); killed_by_stage, kill_reason.

from atropos.track_record import live_verdict
# production stage for a deployed SURVIVES; LIVE_OK ≠ confirmation.
```

## Skills (10)

Core: `01-preregister`, `02-run-pipeline` (the 9-verdict table),
`03-read-graveyard`, `04-run-ml-kill-protocol` (the ML chain:
features→target, walk-forward, gates). Stage 1-7 protocols:
`11-premise-check`, `12-null-control`, `13-signal-test`,
`14-incremental-test` (the decisive one), `15-robustness-battery`,
`16-walkforward-calendar`. The order is fixed:
1 → 2 → 3 → 4 → 5 → 6, each stage cheaper than the next. List:
`atropos agent-context --list-skills`; text: `--skill=NAME`.

## NEVER (do not ever)

- Do not look into the data before the prereg is frozen — no
  "previews".
- Do not change kill criteria after the first touch of the data.
- Do not re-run with tweaked parameters (result shopping); a change of
  intent = a new hypothesis, a new ID.
- Do not read PASS/SURVIVES as confirmation of an edge.
- Do not re-propose a `KILLED` hypothesis without a new mechanism.
- Do not leave a hypothesis without a verdict: without a tombstone it
  counts as open.
- Do not edit or delete tombstones: the graveyard is append-only.

## Overriding this layer

- Files: `AGENTS.md`/`CLAUDE.md` in a higher layer replace;
  `X.append.md` extends; `X.disabled` disables the base (appends
  remain).
- Skills: `skills/{name}.md` replaces/adds; `{name}.append.md`
  extends; `skills/.disabled` (a list of names) and
  `skills/.disabled-all` disable. Higher layer wins: project > user >
  package.
- Setup: `atropos agent-setup --scope=project --dry-run`.
