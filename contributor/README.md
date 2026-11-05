# contributor/ — Atropos developer context

This is where the project's dev discipline lives: code conventions, the
quality gates (pytest / ruff / mypy), and the rules for agents working
in this repository. This is NOT a guide to using the framework — see
the links at the bottom.

## Files

- `AGENTS.md` — universal agent instructions: the hypothesis pipeline,
  the 6 laws, plus dev discipline (uv, ruff, mypy, the AST conventions
  test).
- `CLAUDE.md` — a Claude specification layered on `AGENTS.md`: binds the
  rules to concrete tools (Read/Write/Edit/Bash); an add-on, not a
  replacement.

## Quality gates (run from the repository root)

```bash
uv run python -m pytest --tb=no        # tests: rc=0 is mandatory
uv run python -m ruff check .          # lint: 78 columns, docstrings, E741, PERF
uv run python -m ruff format --check . # formatting
uv run python -m mypy                  # types: strict, all code and tests
```

## Run pattern: isolation and no flakiness

- Do not combine `-q` with a pipe (`| head`, `| grep`): the pytest
  summary gets swallowed and a failure is only visible via the return
  code. Either full output, or an explicit capture of the return code
  (`rc=$?`) — and judge by that.
- Tests never touch the network and never depend on wall-clock time —
  only frozen data snapshots and seeds.

## Why single-letter identifiers are banned

A 1-character name tells the reader nothing; ruff only catches `l/I/O`
(E741), the remaining single-letter names are banned by the AST test
`tests/test_conventions.py` (it scans `atropos/*.py` and `tests/*.py`).
The only exception is the `_` throwaway. TypedDict fields have a strict
whitelist of `{"n", "B"}` (the historical public API of `bootstrap_ci`):
any other single-letter field (`q`) is banned like an ordinary variable.

## Where to go next

- Using the framework with an AI agent → `atropos/agent/`
  (a pip package) or the `atropos agent-context` command.
- Methodology → `FRAMEWORK.md`; deployment → `METHODOLOGY.md`.
- Templates (prereg, post-mortem, agent briefing) → `TEMPLATES.md`.
