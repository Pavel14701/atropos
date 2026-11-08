# Contributing: the laws

Discipline beats speed. A PR merges when the gates are clean, the
tests are honest, and no law is bent. The full rulebook for framework
developers lives in `CONTRIBUTING.md` and `contributor/AGENTS.md`;
this page is the short version with the laws that surprise people.

## Running the suite

```bash
uv run python -m pytest               # full suite
uv run python -m pytest -m "not slow" # skip @slow calibration tests
uv run ruff check . && uv run ruff format --check .
uv run python -m mypy                 # strict
```

Python 3.12, pinned in `uv.lock`; set up with `uv sync`. Zero
warnings before a PR. Tests never touch the network and never read
the wall clock: fixed seeds, fixed dates, reproducible draws.

## The DOCTEST LAW

Every `__all__` symbol carries a runnable doctest — a docstring
without a doctest is a docstring without a contract. The examples are
executed in the suite (`tests/test_docstring_contract.py` enforces
the coverage), so the documented call signature and the shown output
are pinned like any other test: if your change moves a doctest's
output, that is a review item, never a silent re-pin. Documented
`ValueError` behavior is tested too, mostly as doctest tracebacks.

## The MUTATION LAW

Mutation runners (mutants, fault injection, deliberate-break probes)
run ONLY on committed code and only through backup-and-restore: copy
the file, break it, run, restore the copy. NEVER `git checkout` /
`git restore` to undo an experiment — those commands discard
uncommitted work around your experiment too, and a mutation run that
eats a colleague's WIP is not an experiment, it is data loss. If the
run crashes before the restore, restore from the backup, then verify
with `git diff` that the tree is exactly as you found it.

## The ERROR-PATH LAW

Every validation rung is pinned by a `pytest.raises` test (or an
equivalent doctest traceback): each `ValueError` a constructor or
entry point raises — bad role, negative rate, unknown combination,
deferred family, failed lookahead check — has a test that feeds the
bad input and asserts the message's identifying part. The v15
coverage-driven error-path tests sweep for unpinned raise sites; a
new validation branch is born with its pin, not "covered later".

## Conventions

- **Frozen dataclasses + `__post_init__`.** Specs and policies are
  `@dataclass(frozen=True)`; every field is validated loudly at
  construction (`__post_init__` raising `ValueError` with the field
  named). Invalid state cannot exist past the constructor.
- **TypedDict reports, no pydantic.** Result dicts are contracts
  declared as `TypedDict` (every key, every outcome — see
  `ValidationReport`, `DataManifest`, `FeatureAnalysisReport`,
  `TradeRecord`); keys are snapshotted in
  `tests/test_api_contract.py`, so adding a key is a deliberate diff.
  pydantic is not a dependency and stays out.
- **78 columns** in code and docstrings (ruff `E501`; docs markdown
  is exempt — it wraps in the renderer).
- **mypy `--strict`** on `atropos/` and `tests/`; a suppression needs
  a specific error code and a reason.
- **No single-letter names.** Ruff catches `l`/`I`/`O`;
  `tests/test_conventions.py` (AST) catches the rest. `_` for
  throwaways is the only exception.
- **Google-convention docstrings** (ruff pydocstyle `D`, convention
  google), summary on the first line. New modules ship an asserted
  `run_all_examples()` demo and a populated `__all__`.

## Adding a module

Sweep `CONTRIBUTING.md` § "Adding a module" before your first
`atropos/<name>.py`: the meta-tests enforce the demo, the doctest
coverage, the no-`np.roll` lookahead ban and the shared-helper rule
automatically, so a new module is born under full law or not at all.
