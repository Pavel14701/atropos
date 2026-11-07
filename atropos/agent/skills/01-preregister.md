---
name: 01-preregister
description: Freeze the hypothesis with a preregistration via atropos.prereg_generator BEFORE any touch of the data.
when_to_use: Before the first look at the data for a new hypothesis; when reopening a buried family with a new mechanism.
version: 0.1.0
---

# SKILL 01: PREREGISTER — freeze the hypothesis before data
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Core skill of the Atropos agent. Public API:
> `atropos.prereg_generator.generate_prereg(params: dict) -> str` and
> `validate_prereg(params) -> list[str]`. An empty problem list = you
> may generate. Law: the prereg is committed BEFORE the data is
> pulled.

## PURPOSE

Turn the hypothesis into an immutable document with a numeric kill
criterion before the data gets a chance to hint at anything. A prereg
without a numeric threshold is not a prereg — it is an intention.

## PROCEDURE

1. **Assemble the params-dict.** Seven required fields
   (`REQUIRED_FIELDS`): `name`, `hypothesis`, `construction` (a str or
   a list of steps), `nulls` (a str or a list), `kill_criteria` (a
   list), `power_calc` (a str / list / dict), `hash` (the hash of the
   frozen spec).
2. **Kill criteria — hard directional rules only.** Dict form:
   `{"criterion": ..., "threshold": <number>, "comparison": "<" | "<="
   | ">" | ">=" | "=="}`; str form — an operator pinned to a number
   (`"incremental t < 2"`). The threshold is written before the test
   and never changed afterwards.
3. **Validate.** `validate_prereg(params)` must return `[]`. Any
   problem is fixed in the params, not ignored.
4. **Generate and commit.** `generate_prereg(params)` — a markdown
   document (date optional, ISO format). The document is
   committed/saved BEFORE the data is pulled. No workspace layout —
   `atropos init` creates the skeleton without overwriting existing
   files.
5. **Reopening.** For a buried family — a new dated prereg that
   discloses the closed history and the new mechanism; otherwise it is
   mining.

## EXAMPLE

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

params = {
    "name": "H-0007-mom30-echo",
    "hypothesis": "mom30 predicts the 5d forward-return",
    "construction": ["zscore(mom30)", "top-tercile long only"],
    "nulls": ["no predictive power vs a shuffled signal"],
    "kill_criteria": [
        {"criterion": "incremental t", "threshold": 2.0,
         "comparison": "<"},
        {"criterion": "placebo percentile", "threshold": 95.0,
         "comparison": "<"},
    ],
    "power_calc": "N=(2*sigma/EV)^2 ~= 600 observations for t~2",
    "hash": "sha256:<hash-frozen-spec>",
}
assert validate_prereg(params) == []
doc = generate_prereg(params)  # markdown, finished before data
```

## NEVER

- Do not look into the data before the prereg is frozen — including
  "just glancing at the spread".
- Do not change thresholds/directions after the first run: a change of
  intent = a new hypothesis with a new ID.
- Do not leave a kill criterion without a number: "looks too weak" is
  not a criterion.

## NEXT

Prereg frozen → `02-run-pipeline.md` (the run and the verdict).
Resurrection conditions of other families → `03-read-graveyard.md`.
