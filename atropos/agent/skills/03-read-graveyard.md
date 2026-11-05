---
name: 03-read-graveyard
description: Graveyard check — search for structural duplicates among killed hypotheses and read the resurrection conditions.
when_to_use: BEFORE proposing any new hypothesis and before its preregistration; whenever in doubt "has this been tested already?".
version: 0.1.0
---

# SKILL 03: READ GRAVEYARD — the graveyard before thinking

> Core skill of the Atropos agent. The graveyard check comes FIRST,
  before the prereg: a hypothesis that dies at the graveyard costs 0
  data queries. The graveyard is append-only: tombstones are never
  edited or deleted.

## PURPOSE

Do not propose what has already been refuted. Identity is defined by
the prediction and the kill criterion, not by the title: "a mom30 echo
with a 1.5σ threshold" and "momentum carry at 1.5 sigma" are one
hypothesis.

## PROCEDURE

1. **Find the registry.** `hypotheses/INDEX.csv` (h_id, title, tags,
   verdict) and `hypotheses/graveyard/H-NNNN.md`. No registry in the
   project — `atropos init` generates the skeleton (never overwrites
   existing files).
2. **Search by mechanics, not by title.** Tags / prediction / kill
   criterion; a structural duplicate = matching prediction and kill
   criterion.
3. **Read the tombstone in full.** Required fields: cause of death (a
   single primary taxonomy term), key numbers, the kill criterion the
   hypothesis died on, and **resurrection conditions** — the terms for
   reopening (a new argument / new data / a new instrument; "none
   known" = honest).
4. **Decision.** A `KILLED` duplicate with no new mechanism → do not
   propose it, that is mining. A new mechanism exists OR the data
   snapshot has grown wider → reopening: a new dated prereg that
   discloses the closed history (skill `01-preregister.md`).
5. **3-strikes.** Three kill mechanics on one family = permanent
   close: successors are admitted with prior <= 20% and a single run.

## WRITING TO THE GRAVEYARD (once a verdict is in)

Every hypothesis ends with a tombstone and a row in `INDEX.csv`,
including your own uncomfortable ones: without a tombstone a
hypothesis counts as open. Into the tombstone: h_id, prediction, kill
criterion, verdict, key numbers from the report (`overall_verdict`,
`killed_by_stage`, `kill_reason`), resurrection conditions, date.

## NEVER

- Do not rely on memory: "I remember it was never tested" is not a
  graveyard-check result; record what you searched and by which tags.
- Do not edit or delete tombstones.
- Do not re-propose a `KILLED` hypothesis without a new mechanism.

## NEXT

Graveyard clean / a new mechanism found → `01-preregister.md`.
Run and verdict → `02-run-pipeline.md`.
Stage 1-7 protocol skills: `11-premise-check.md` ...
`16-walkforward-calendar.md`.
