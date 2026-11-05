# Atropos — Hypothesis-Killing Library — Claude-Specific Instructions

A Claude specialization of `AGENTS.md` (Claude Code, ZCode, and
compatible CLI agents). The base rules live in `AGENTS.md`; this file
carries only the differences and the binding of rules to concrete tools.
Fill in the `[PLACEHOLDER]` placeholders with your project's values.

---

## 1. First Action and Project Overview

Before any work, read in this order: `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` → `AGENTS.md` → this file. If the primary document is missing — stop and report: the method is not deployed (deployment is described in `METHODOLOGY.md`).

Project values: name `[PROJECT_NAME]`, data `[DATA_DIR]`, snapshot `[SNAPSHOT_DATE]`, owner `[OWNER]`, environment `[VENV]`. Where this file diverges from `AGENTS.md` on tools, this file wins; on process, the laws of `AGENTS.md` win.

## 2. Before Proposing a Hypothesis — which tools to use

1. **Graveyard check.** Read `hypotheses/INDEX.csv`; then Grep
   `hypotheses/graveyard/` for the keywords and tags of the future
   hypothesis (not titles). Write the search results into the prereg.
2. **Read the pipeline.** Read `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` —
   the table of 7 stages and kill criteria.
3. **Prereg before data.** Write `hypotheses/preregistrations/H-NNNN.md`
   → commit via Bash → only then any data pull or reading.

## 3. Pipeline and Outcomes

Stages: `Formulation → Graveyard check → Preregistration → Data → Test → Verdict → Tombstone`. Stages are never skipped or reordered. The full kill-criterion table is in `AGENTS.md`, section 3.

Handling outcomes:
- `KILLED` — Write the tombstone + a row in `INDEX.csv`; a short summary
  to the user: what was killed and why.
- `SURVIVED` — a tombstone listing every check; the decision to act on
  it belongs to the user, not the agent.
- `INCONCLUSIVE` — a tombstone stating which data was missing; reopening
  on a new snapshot is allowed.

## 4. The 6 Laws — do not violate

1. Preregister before data. 2. One hypothesis — one ID. 3. Reproducibility (seed + snapshot + requirements). 4. Kill criteria upfront. 5. No post-hoc edits. 6. Every hypothesis gets a verdict. Full wording in `AGENTS.md`, section 4.

## 5. Tools and Conventions

- Create and edit files only with Write/Edit; do not assemble files via
  Bash heredocs.
- Reading with Read (not `cat`, `head`, `tail`); searching with
  Grep/Glob: search first, then targeted reads.
- All commands through uv in the project environment (`uv sync`,
  `uv run <cmd>`); the global interpreter is never used.
- Tests: Bash `uv run pytest -q` before every commit; hypothesis code
  without a test is not merged.
- Style: Bash `uv run ruff check .` and `uv run ruff format .`; 78
  columns, docstrings, perflint, E741; configuration in `pyproject.toml`;
  zero warnings.
- Types: Bash `uv run mypy` — strict mode, all code and tests;
  single-letter identifiers are banned (AST test
  `tests/test_conventions.py`, except `_`).
- Running a hypothesis test: Bash `python hypotheses/code/H-NNNN_<slug>.py`;
  stdout strictly:
  `H-NNNN: metric=[VALUE] threshold=[THRESHOLD] verdict=[KILLED|SURVIVED|INCONCLUSIVE]`.
- If pytest fails, fix the test code; weakening a kill criterion is
  forbidden (law 4).
- `[DATA_DIR]/snapshots/` is immutable: a new pull = a new snapshot;
  overwriting is forbidden.
- Commits only on an explicit user request; without a request, do not
  work on `main`.

A standard session, in order:

```bash
uv run pytest -q                        # 1. baseline is green
uv run ruff check . && uv run ruff format --check .   # 2. style is clean
uv run mypy                             # 3. types (strict)
# 4. Write prereg → commit → data pull → run the test → Write tombstone → INDEX.csv row → commit
```

## 6. Key Files

The canonical `hypotheses/` layout is a library convention:
scaffold with `atropos init` (files are generated, never overwritten).

| File | Role |
|------|------|
| `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` | Methodology — read first |
| `AGENTS.md` | Base agent rules; this file is a layer on top of them |
| `hypotheses/preregistrations/H-NNNN.md` | Preregs — Write before any data |
| `hypotheses/code/H-NNNN_*.py` | Hypothesis test code |
| `hypotheses/graveyard/H-NNNN.md` | Tombstones — Write only, append-only |
| `hypotheses/INDEX.csv` | Registry — one row per hypothesis |
| `[DATA_DIR]/DATA_DICTIONARY.md` | Data schema — the source of truth before the pull |
| `[PROJECT_SPECIFIC_FILE]` | `[PURPOSE]` — add your project's own files |

## 7. Claude-Specific Prohibitions (on top of the 10 prohibitions of `AGENTS.md`)

1. Do not use subagents to bypass pipeline stages; every stage runs in
   the main context with visible artifacts.
2. Do not edit the prereg via Edit after the first data touch — only a
   new hypothesis (law 5).
3. Do not "improve" kill criteria when porting the prereg into code —
   the code follows the prereg verbatim.
4. Do not open data before the prereg, even "just to check the schema" —
   the schema comes from `DATA_DICTIONARY.md`.
5. Do not start a data pull with an uncommitted prereg.
6. Do not end a session with a hypothesis lacking a verdict: either a
   tombstone, or an explicit OPEN status and a plan.

---

Priority: laws (`AGENTS.md`, section 4) > prohibitions (section 7 of
both files) > speed and convenience. On conflict — stop and ask
`[OWNER]`.
