# Atropos — Hypothesis-Killing Framework — Universal Agent Instructions

> **Context beacon.** This is the DEVELOPER context of Atropos
> (pytest/ruff/mypy/conventions). For USING the framework with an AI
> agent, see `atropos/agent/` (pip-installed) or run `atropos agent-context`.

A document for any AI agent and any analyst: Claude, Cursor, Windsurf,
Copilot, Codex, GPT agents, custom scripts. Tool-agnostic: nothing here
is specific to a particular agent. Before use, fill in the
`[PLACEHOLDER]` placeholders with your project's values.

---

## 1. Project Overview — fill in for your project

- Project: `[PROJECT_NAME]`
- Project goal: `[PROJECT_GOAL]`
- Repository root: `[PROJECT_ROOT]`
- Data directory: `[DATA_DIR]`; current snapshot: `[SNAPSHOT_DATE]`
- Process owner (questions and escalations): `[OWNER]`
- Primary methodology document: `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` —
  read before starting any work.

## 2. Before Proposing Any Hypothesis

Mandatory protocol, strictly in order:

1. **Graveyard check.** Search `hypotheses/graveyard/` and
   `hypotheses/INDEX.csv`: has this hypothesis, or a structurally
   identical one, been tested? Identity = matching prediction and kill
   criterion, not a matching title. A `KILLED` hypothesis is not
   re-proposed without a new mechanism.
2. **Read the pipeline.** Section 3 below and
   `docs/HYPOTHESIS_KILLING_FRAMEWORK.md`.
3. **Prereg before data.** The preregistration (prereg) is written and
   committed BEFORE any data is touched: first
   `hypotheses/preregistrations/H-NNNN.md`, then the data pull.

Graveyard search results are recorded in the prereg: what was searched,
by which tags, what was found. "I remember it was never tested" is not
acceptable evidence.

## 3. The 7-Stage Pipeline

| # | Stage | Agent action | Stage kill criterion |
|---|-------|--------------|----------------------|
| 1 | Formulation | State a falsifiable prediction | An unfalsifiable prediction — kill on the spot |
| 2 | Graveyard check | Search the killed and the open (section 2) | A structural duplicate of a `KILLED` hypothesis — kill |
| 3 | Preregistration | Write `H-NNNN.md`: prediction, numeric kill criterion, snapshot, seed, tags | No measurable kill criterion — kill |
| 4 | Data | Load the frozen snapshot, verify the schema against the data dictionary | The snapshot does not cover the test — `INCONCLUSIVE`; substituting data is forbidden |
| 5 | Test | Implement the test in `hypotheses/code/H-NNNN_*.py` plus a pytest test | The test fails a sanity case — kill |
| 6 | Verdict | Compare the metric against the pre-registered threshold; account for multiple comparisons | The metric is worse than the threshold — `KILLED` |
| 7 | Tombstone | Write `graveyard/H-NNNN.md` and a row in `INDEX.csv` | Without a tombstone the hypothesis counts as open (law 6) |

Outcomes: `KILLED` — refuted; `SURVIVED` — survived, and only now may it
justify a decision; `INCONCLUSIVE` — not enough data, reopening on a new
snapshot is allowed.

## 4. The 6 Systematic Laws — DO NOT VIOLATE

1. **Preregister before data** — no data is touched before the prereg is
   committed.
2. **One hypothesis — one ID** — one hypothesis = one `H-NNNN` = prereg
   + code + tombstone; duplicate IDs are forbidden.
3. **Reproducibility** — fixed seed, frozen snapshot, pinned
   dependencies; the result must reproduce from the command in the
   prereg.
4. **Kill criteria upfront** — the numeric refutation threshold is
   written before the test and never changed afterwards.
5. **No post-hoc edits** — after the first data touch the prereg is not
   edited; a changed intent = a new hypothesis.
6. **Every hypothesis gets a verdict** — every `H-NNNN` ends with a
   tombstone and a row in `INDEX.csv`, including your own inconvenient
   hypotheses.

## 5. Code Conventions

- Language: Python 3.11+; package manager and environment — **uv**
  (`uv sync`, commands via `uv run`); versions pinned in
  `pyproject.toml` + `uv.lock`.
- Style: `ruff check` + `ruff format` — a strict rule set, 78 columns:
  mandatory docstrings (pydocstyle), performance constructs (perflint),
  ambiguous-name bans (E741). Zero warnings before a commit.
- Types: `mypy` in **strict** mode — all code and tests; type errors are
  never suppressed without the exact code in the ignore.
- Names: single-letter identifiers are banned everywhere (except the
  `_` throwaway); enforced by the AST conventions test alongside ruff.
- Tests: `pytest`; every hypothesis test gets `tests/test_h_NNNN.py`;
  hypothesis code without a test is not merged.
- Tests never touch the network and never depend on wall-clock time —
  only `[DATA_DIR]` snapshots.
- Data: `[DATA_DIR]/snapshots/` is immutable; a new pull = a new
  snapshot, editing an existing one is forbidden.
- Naming: `hypotheses/code/H-NNNN_<slug>.py`; metric output on stdout
  strictly in the format
  `H-NNNN: metric=[VALUE] threshold=[THRESHOLD] verdict=[KILLED|SURVIVED|INCONCLUSIVE]`.

## 6. Key Files Index — fill in for your project

The canonical `hypotheses/` layout (preregistrations, code,
graveyard, INDEX.csv) is a framework convention: scaffold with
`atropos init` (files are generated, never overwritten).

| File | Purpose |
|------|---------|
| `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` | Full methodology: stages, laws, formats |
| `hypotheses/preregistrations/H-NNNN.md` | Preregs — written before data |
| `hypotheses/code/H-NNNN_*.py` | Hypothesis test code |
| `hypotheses/graveyard/H-NNNN.md` | Tombstones — append-only |
| `hypotheses/INDEX.csv` | Registry of all hypotheses |
| `[DATA_DIR]/snapshots/[SNAPSHOT_DATE]/` | Data snapshots |
| `[DATA_DIR]/DATA_DICTIONARY.md` | Data dictionary |
| `[PROJECT_SPECIFIC_FILE]` | `[PURPOSE]` — add your project's own rows |

## 7. What NOT to Do — 10 prohibitions

1. Do not look at the data before the prereg is committed.
2. Do not change kill criteria after the first data touch.
3. Do not use the same data snapshot to conceive a hypothesis and to
   test it (data leakage).
4. Do not ignore multiple comparisons across test families.
5. Do not declare `SURVIVED` without a pre-registered numeric threshold.
6. Do not edit or delete tombstones — the graveyard is append-only.
7. Do not run tests without a fixed seed.
8. Do not skip the graveyard check "from memory".
9. Do not keep a hypothesis open longer than `[MAX_OPEN_DAYS]` days —
   a verdict is mandatory.
10. Do not tune code, thresholds, or wording toward a desired result
    (result shopping).

## 8. When to Stop and Ask

- The kill criterion cannot be reduced to a number — discuss the
  wording with `[OWNER]`, do not improvise on the fly.
- Data outside `[DATA_DIR]` is needed — request access, do not route
  around the restriction.
- An ambiguous duplicate candidate is found — ask before spending
  resources on the test.

When speed and discipline conflict, discipline wins.

---

## Appendix A. New Hypothesis Checklist

- [ ] Graveyard check done, results written into the prereg
- [ ] `H-NNNN` is free — the ID is the next in sequence
- [ ] Prereg filled in: prediction, kill criterion, snapshot, seed, tags
- [ ] Prereg committed BEFORE the data pull
- [ ] Test covered by a pytest test, `ruff check` clean
- [ ] Verdict recorded: tombstone + a row in `INDEX.csv`

## Appendix B. INDEX.csv Row Format

```csv
h_id,title,tags,prereg_date,snapshot,verdict,verdict_date,tombstone_path
H-0001,Example title,tag1|tag2,2026-01-10,2026-01-01,KILLED,2026-01-12,graveyard/H-0001.md
```

## Appendix C. Preregistration Template

```markdown
# H-[NNNN]: [TITLE]
- Author: [AUTHOR], date: [DATE], tags: [TAGS]
- Prediction: [MEASURABLE_FALSIFIABLE_PREDICTION]
- Kill criteria: [METRIC] < [THRESHOLD] on snapshot [SNAPSHOT_DATE], seed [SEED]
- Graveyard check: [WHAT_WAS_SEARCHED], result: [FOUND_NOTHING | DUPLICATE_OF_H-XXXX]
```

Document priority: laws (section 4) > prohibitions (section 7) >
appendix templates > convenience. On conflict — stop and ask `[OWNER]`.
