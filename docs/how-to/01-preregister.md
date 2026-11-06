# How-to 01. Preregistering a hypothesis: from params to the PREREG file

> **Estimated time:** 10 minutes
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync` done), Python 3.12. Data access is **not needed** for this
> how-to — and is **forbidden** until the end of step 4 (law 1:
> preregister before data).

## Why

A preregistration is a freeze of the protocol BEFORE the data:
prediction, construction, nulls, and kill criteria are written down
before the first data touch, and committing the file becomes the
protocol's hash. The tool is `atropos.prereg_generator`: it will not let
you register a "soft" kill criterion without an operator and a number.

Self-contained: no need to read `FRAMEWORK.md` before this how-to.

Import (the `atropos` package does not re-export functions at the top
level — use module paths):

```python
from atropos.prereg_generator import generate_prereg, validate_prereg
```

## Step 1. Assemble `params` — 6 required fields

| Field | Type | What it is |
|-------|------|------------|
| `name` | `str` | short hypothesis name (a slug; goes into the title) |
| `hypothesis` | `str` | the prediction in one phrase: what, sign, horizon |
| `construction` | `str` or `list[str]` | how the signal is built: thresholds, windows, normalization |
| `nulls` | `str` or `list[str]` | null explanations of the effect (repackaging, concentration, …) |
| `kill_criteria` | non-empty `list` of `str`/`dict` | numeric kill thresholds — see below |
| `power_calc` | `str`, `list[str]`, or `dict` | power calculation: N, alpha, power |

The spec **hash is not an input**: `generate_prereg` computes it
automatically — sha256 over the canonical JSON of the frozen fields
(first 8 hex chars; `atropos.prereg_generator.spec_hash`) — so the
document always carries the hash of the spec it actually renders.
Editing any frozen field provably changes the hash (tamper evidence);
a stale `hash` entry left in `params` is tolerated by validation and
ignored at render time.

A kill criterion is a hard directional rule, in two forms:

- **string**: an operator plus a number inside the text, e.g.
  `"IC < 0.05 on 500 obs"` or `"share of PnL from best subperiod > 0.6"`;
- **dict**: `{"criterion": str, "threshold": number, "comparison": one of
  < <= > >= == ~}` — `comparison` is **required**, there is no default.

A bare number in prose ("wait 200 days") freezes nothing and will be
rejected. The `~` operator is allowed only as an explicit
diagnostic/approximate marker; for hard kills use directional operators.

## Step 2. Run `validate_prereg` down to an empty list

The function returns a list of problems; `[]` = valid. Never proceed to
generation while the list is non-empty — `generate_prereg` raises
`ValueError` on invalid params.

### Valid example (example output)

```python
from atropos.prereg_generator import validate_prereg

params = {
    "name": "overnight_imbalance_edge",
    "hypothesis": (
        "Overnight order-flow imbalance predicts the next-day return "
        "incrementally to 7d and 30d momentum."
    ),
    "construction": [
        "signal = zscore(overnight_order_flow_imbalance)",
        "enter at next open, exit at next close",
    ],
    "nulls": [
        "the edge is repackaged 7d/30d momentum",
        "the edge is concentrated in a single calendar year",
    ],
    "kill_criteria": [
        {
            "criterion": "|t| of the new signal after mom7 + mom30 controls",
            "comparison": "<",
            "threshold": 2.0,
        },
        "share of total PnL from the best subperiod > 0.6",
    ],
    "power_calc": {"N_required": 2600, "alpha": 0.05, "power": 0.80},
}

problems = validate_prereg(params)
print(problems)
assert problems == []
```

Example output:

```text
[]
```

### Invalid example 1 — prose instead of a criterion (example output)

```python
bad_wait = {
    "name": "x", "hypothesis": "h", "construction": "c", "nulls": "n",
    "power_calc": "p",
    "kill_criteria": ["wait 200 days before final check"],
}
print(validate_prereg(bad_wait))
```

Example output:

```text
["kill_criteria[0]: no comparison operator bound to a numeric threshold found in 'wait 200 days before final check' (e.g. 'IC < 0.05')"]
```

"Wait another 200 days" is not a kill criterion: no operator is bound to
a number, so the threshold can be moved after the data.

### Invalid example 2 — a dict without `comparison` (example output)

```python
bad_dict = dict(bad_wait)
bad_dict["kill_criteria"] = [
    {"criterion": "t-stat after controls", "threshold": 2.0},
]
print(validate_prereg(bad_dict))
```

Example output:

```text
["kill_criteria[0]: missing required 'comparison' (one of ['<', '<=', '==', '>', '>=', '~'])"]
```

A threshold without direction is a soft switch: it is unclear whether
`2.0` kills at `t < 2` or at `t > 2`. Hence `comparison` is required and
never defaulted.

## Step 3. Generate the document

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

# ... params as in step 2 ...

assert validate_prereg(params) == []
doc = generate_prereg(params, date="2026-11-03")
print(doc)
```

Example output (abridged; the full document is from your run):

```markdown
# Preregistration: overnight_imbalance_edge

- **Frozen on:** 2026-11-03
- **Spec hash:** `31711471`

## Hypothesis

Overnight order-flow imbalance predicts the next-day return
incrementally to 7d and 30d momentum.

## Construction

- signal = zscore(overnight_order_flow_imbalance)
- enter at next open, exit at next close

## Null hypotheses

**H1:** the edge is repackaged 7d/30d momentum
**H2:** the edge is concentrated in a single calendar year

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | \|t\| of the new signal after mom7 + mom30 controls | < | 2.0 |
| 2 | share of total PnL from the best subperiod > 0.6 | > | 0.6 |

## Power analysis

- **N_required:** 2600
- **alpha:** 0.05
- **power:** 0.8

## Attestation

This specification (construction, nulls, kill criteria, power) was
frozen before any outcome data was evaluated against it. Post-hoc
changes invalidate the preregistration; any deviation must be logged
as a separate amendment with a new hash.
```

Note row 2 of the kill-criteria table: for the string form the generator
extracts the operator and the number itself (">" and "0.6") instead of
inserting a stub. The date: `date=None` uses today; an explicit date
must be ISO (`YYYY-MM-DD`), otherwise `ValueError`. The `Spec hash`
line is computed from the frozen fields — your run's hash will differ
from the one shown, and editing any field in `params` changes it.

## Step 4. Save `PREREG_<name>_<date>.md` and commit BEFORE the data

```python
from pathlib import Path

out = Path("PREREG_overnight_imbalance_edge_2026-11-03.md")
out.write_text(doc, encoding="utf-8")
print("saved:", out.resolve())
```

Example output (the path depends on your working directory):

```text
saved: <workspace>/PREREG_overnight_imbalance_edge_2026-11-03.md
```

If the workspace does not exist yet (no `GRAVEYARD.md` and no
`hypotheses/` directory), generate the skeleton with `atropos init` —
the command creates only missing files and never overwrites existing
ones; edit them manually afterwards.

Put the file in your project's preregistration directory (in the
methodology's layout — `hypotheses/preregistrations/`) and **commit it
before any data pull**. The commit is the protocol's hash: editing after
the first data touch invalidates the test; a changed intent = a new
hypothesis with a new ID, not an edit of this one.

## Step 5. The laws checklist (before opening the data)

- [ ] Graveyard check done, search results written into the prereg
      (see how-to 03).
- [ ] Every kill criterion = operator + number (string) or
      `{criterion, threshold, comparison}` (dict); `validate_prereg`
      returned `[]`.
- [ ] `comparison` is explicit in every dict criterion.
- [ ] Data snapshot and seed are frozen (law 3: reproducibility).
- [ ] `PREREG_<name>_<date>.md` is committed BEFORE the data pull
      (law 1).
- [ ] Thresholds are never changed again (law 4: kill criteria upfront;
      law 5: no post-hoc edits).

## What next

- How to read the pipeline's result — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Where the verdict is recorded — how-to 03
  (`docs/how-to/03-read-graveyard.md`).
- Function reference — `docs/api/atropos/prereg_generator.html`.
