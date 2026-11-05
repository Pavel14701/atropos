# How-to 03. Reading and growing the hypothesis graveyard

> **Estimated time:** 10 minutes
>
> **You will need:** your project with the framework deployed and at
> least one graveyard entry. No atropos code needed — just git, grep,
> and csv. No layout yet? Generate the skeleton: `atropos init`.

## Why

Law 6 of the framework: **every hypothesis gets a verdict**. Without a
tombstone a hypothesis counts as open, and open items accumulate
duplicates. The graveyard is not a museum of failures but a **product of
the process**: the accumulated knowledge base of which hypothesis
families are already closed and under what conditions a closure may be
revisited.

It is important to understand the boundary: the `atropos` repository is
the killing toolkit (the pipeline, the controls, the prereg generator).
The graveyard itself is a **methodological artifact of your project**:
it lives in your `hypotheses/` layout, not in the framework.

If there is no layout yet, generate the skeleton: `atropos init` — the
command creates `GRAVEYARD.md`, `hypotheses/INDEX.csv`, and
`hypotheses/README.md` only if they are missing; existing files are
never overwritten.

## The directory layout (per METHODOLOGY.md)

```text
[PROJECT_ROOT]/
├── hypotheses/
│   ├── preregistrations/      # H-NNNN: preregs (before data)
│   ├── code/                  # H-NNNN: test code
│   ├── graveyard/             # H-NNNN: tombstones (verdicts), append-only
│   └── INDEX.csv              # registry of all hypotheses
└── ...
```

The identity rule: one hypothesis = one `H-NNNN` = at least three
artifacts (prereg, code, tombstone).

## Tombstone structure

Required fields (per the framework's graveyard protocol):

1. **ID and verdict** — `H-NNNN` and `KILLED | SURVIVED | INCONCLUSIVE`,
   the verdict date, the author.
2. **COD (cause of death)** — the stage and the type of death. The type
   comes from the atropos pipeline taxonomy: `REPACKAGING`, `ARTIFACT`,
   `DEAD`, `NO_OOS_EDGE`, `COST_DEATH`, `RISK`, `CS_NO_EDGE` (or
   `SURVIVED` / `INCONCLUSIVE` for entries that were not killed).
3. **Prediction** — what was predicted (falsifiable, from the prereg).
4. **The minimal numbers** — the actual metric value against the
   threshold (raw t; incremental t if reached; the failed test), the
   data snapshot, the seed.
5. **Cause of death in one line** — for future readers.
6. **Resurrection conditions** — which new data or conditions would make
   a revisit possible.
7. **Do not try again** — which prohibition a second attempt would
   violate.

An example entry (from this how-to's demo layout):

```markdown
# H-0002 — KILLED (COST_DEATH)
- Hypothesis: cross-sectional imbalance spread on the top-20 symbols
- Prediction: net-of-cost SR p5 > 0 at 10bps round-trip
- COD: stage 5, monte-carlo — net SR p5=-0.52 against a breakeven of 8bps
- Verdict: KILLED, 2026-03-02, snapshot 2026-02-01, seed 0
- Resurrection conditions: costs < 8bps one-way (maker execution)
  OR a holding horizon >= 5 days on the same signal
- Do not try again: the same signal with taker execution
```

The `INDEX.csv` registry row:

```csv
h_id,title,tags,prereg_date,snapshot,verdict,verdict_date,tombstone_path
H-0002,xs-imbalance-spread,cross-sectional|costs|orderflow,2026-02-20,2026-02-01,KILLED,2026-03-02,graveyard/H-0002.md
```

## How to search before a new hypothesis (the graveyard check)

Pipeline stage 2, mandatory before the prereg. Hypothesis identity is a
match of **prediction + kill criterion**, not of the title: the same
idea under a new name is a duplicate; a `KILLED` hypothesis is not
re-proposed without a new mechanism. Search three ways:

1. **By tags** — tags are mandatory in the prereg; search the registry
   by them:

```bash
grep -i "orderflow" hypotheses/INDEX.csv
```

Example output on this how-to's demo layout (3 tombstones:
H-0001 orderflow/daily, H-0002 cross-sectional/costs, H-0003 funding):

```text
H-0001,overnight-ofi-edge,orderflow|momentum|daily,2026-02-01,2026-01-15,KILLED,2026-02-10,graveyard/H-0001.md
H-0002,xs-imbalance-spread,cross-sectional|costs|orderflow,2026-02-20,2026-02-01,KILLED,2026-03-02,graveyard/H-0002.md
```

2. **By COD / death family** — which constructions already died the same
   way:

```bash
grep -il "momentum_control\|REPACKAGING" hypotheses/graveyard/*.md
```

Example output:

```text
hypotheses/graveyard/H-0001.md
```

3. **Reading what you found** — for a duplicate candidate, compare the
   prediction and the kill criterion:

```bash
sed -n '1,4p' hypotheses/graveyard/H-0001.md
```

Example output:

```text
# H-0001 — KILLED (REPACKAGING)
- Hypothesis: overnight order-flow imbalance predicts the next-day return
- Prediction: |t| >= 2 after mom7/mom30 controls on 500 obs
- COD (cause of death): stage 1, momentum_control — t=0.73 after mom7+mom30
```

The search results are recorded in the new prereg: what was searched, by
which tags, what was found ("FOUND_NOTHING" or "DUPLICATE_OF_H-XXXX").
"I remember it was never tested" is not acceptable evidence.

## How to read resurrection conditions

Resurrection is not "a new argument appeared" but "the data or
conditions recorded in the tombstone BEFORE this point changed".
Practical rules:

- **A new data snapshot ≠ a new argument.** A snapshot is legitimate
  only if the tombstone explicitly names its class in the resurrection
  conditions (e.g. ">= 180 days of funding history" or "full order-book
  depths"). The same snapshot with the same construction = the same dead
  hypothesis.
- **A new execution condition** (costs, horizon) is legitimate only if
  it is in the resurrection-conditions list (like "maker execution" in
  H-0002). Otherwise it is result shopping — tuning toward the desired
  outcome.
- **A changed construction = a new hypothesis.** The same idea with a
  different window/threshold/normalization does not "resurrect" the old
  one: it is a new `H-NNNN`, a new prereg, the pipeline from stage 1.
  Dying again for the same reason is a process error, not a hypothesis
  error, and is recorded separately.
- **INCONCLUSIVE entries are not graves:** the hypothesis is open;
  reopening on a new snapshot is allowed per its own resurrection
  condition ("collect the missing data").

## How to add your own tombstone (append-only)

1. Write `hypotheses/graveyard/H-NNNN.md` following the structure above.
   `NNNN` is the next free number.
2. Add a row to `hypotheses/INDEX.csv` (the format is above).
3. Never edit or delete existing tombstones — the graveyard is
   append-only. An error in an entry is fixed by a new entry referencing
   the old one, not by editing the old one.

Adding an entry (an actual run on the demo layout):

```bash
cat > hypotheses/graveyard/H-0004.md <<'EOF'
# H-0004 — KILLED (ARTIFACT)
- Hypothesis: volume-spike reversal on 5m bars
- Prediction: mean event-day return in the top-5% volume bucket < 0 at p < 0.05
- COD: stage 1, placebo_placebo — p=0.61, random days give the same
- Verdict: KILLED, 2026-04-05, snapshot 2026-03-15, seed 7
- Resurrection conditions: a new snapshot with tick-accurate volumes
  (the current snapshot aggregates with double counting)
- Do not try again: the same event set on aggregated volumes
EOF
printf '%s\n' \
  'H-0004,volume-spike-reversal,microstructure|volume|events,2026-03-25,2026-03-15,KILLED,2026-04-05,graveyard/H-0004.md' \
  >> hypotheses/INDEX.csv
```

Reconciling the registry with the graveyard directory (a CI check from
the methodology: every entry with a tombstone has a registry row, and
vice versa):

```python
import csv
from pathlib import Path

root = Path("hypotheses")
rows = list(csv.DictReader((root / "INDEX.csv").open(encoding="utf-8")))
listed = {row["h_id"] for row in rows}
buried = {p.stem for p in (root / "graveyard").glob("H-*.md")}
print("in index but no tombstone:", sorted(listed - buried) or "none")
print("tombstone missing from index:", sorted(buried - listed) or "none")
assert listed == buried
```

Example output after adding H-0004:

```text
in index but no tombstone: none
tombstone missing from index: none
```

In CI this becomes a separate step (together with prereg lint); a
nightly job can re-run `SURVIVED` hypotheses on a new snapshot — a check
that the survivors have not died on fresh data.

## Checklist

- [ ] Graveyard check before every new hypothesis: tags → COD → compare
      prediction + kill criterion; the results go into the prereg.
- [ ] The tombstone contains all required fields, including the COD from
      the pipeline taxonomy and the resurrection conditions.
- [ ] A row added to `INDEX.csv`; the registry matches the directory.
- [ ] Existing tombstones were not edited (append-only).
- [ ] No hypothesis stays open past the open-days limit — a verdict is
      mandatory for every `H-NNNN`, including your own inconvenient ones.

## What next

- How to get a verdict for the tombstone — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- How to write a prereg with the graveyard-check results — how-to 01
  (`docs/how-to/01-preregister.md`).
