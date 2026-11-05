# Contributing

Discipline beats speed. The bar for merging is: clean gates, honest
tests, no lookaheads, no moved goalposts.

## Before you contribute

Set up with `uv sync` (Python 3.11+, pinned in `uv.lock`). All commands
run through `uv run` from the repo root:

```bash
uv run ruff check . && uv run ruff format --check .   # style
uv run python -m mypy                                 # types (strict)
uv run python -m pytest                               # full suite
uv run python -m pytest -m "not slow"                 # skip @slow calibration
```

Zero warnings before a PR. The rules that surprise newcomers:

- **78 columns, everywhere** — code and docstrings; Google-convention
  docstrings are required (ruff `D` rules).
- **mypy `--strict`** on `atropos/` and `tests/` — no untyped code, no
  blanket ignores; a suppression needs a specific error code.
- **No single-letter identifiers** — ruff catches `l`/`I`/`O`, the AST
  test (`tests/test_conventions.py`) catches the rest; `_` for
  throwaways is the only exception.
- **Tests never touch the network and never read the wall clock** —
  fixed seeds and fixed dates only; every random draw is reproducible.
- **Seeded numbers are a contract.** Golden, determinism and sweep tests
  pin exact metric values and verdicts. If your change moves a pinned
  number — even in the "better" direction — that is a review item
  (explain why the old value was wrong), never a silent re-pin.
- **`xfail_strict = true`** — an XPASS is a failure. If your change
  makes a pinned xfail pass, promote it (see KNOWN-ISSUES.md) instead
  of loosening it.

## Adding a module (`atropos/<name>.py`)

The meta-tests sweep every module automatically, so a new module is
born under full law:

- **Demo must assert.** `python -m atropos.<name>` runs a `run_all_examples()`
  demo whose outcomes are asserted, not printed and admired
  (`tests/test_meta_architecture.py`). The demo runner stays out of
  `__all__` (API-contract test).
- **No `np.roll`** — it wraps future values into the past, a lookahead;
  lag with an explicit NaN-padded shift.
- **No swallowed exceptions** — `except SomeError: pass` hides bugs;
  use narrow typed fallbacks with a reason, or let it raise.
- **Helpers have one home** — shared input-coercion helpers live in
  `atropos/_utils.py`; copies are removed on sight (meta-test enforces).
- **Public functions get a docstring example** — every `__all__` entry
  carries a runnable example; doctests run in the suite, and documented
  `ValueError` behavior is actually tested (`tests/test_docstring_contract.py`).
- Result dicts are contracts: keys are snapshotted in
  `tests/test_api_contract.py` — adding a key means updating the
  snapshot deliberately.

**Documentation language.** All documentation is English-only by
decision: the pre-translation Russian versions of FRAMEWORK.md,
METHODOLOGY.md, TEMPLATES.md, CASES_AND_FLAGS.md, contributor
notes and the how-to guides were replaced (not mirrored) in the
translation wave. Do not re-introduce Russian prose.

## Adding an ML model (the registry)

The ML engine's fleet is open, and the contract tests sweep every
entry automatically — a new model is born under full law:

- **Register the factory** in `atropos/ml_training.py`:
  `MODEL_REGISTRY["name"] = make_name`. Signature:
  `make_name(**params, seed: int)` — keyword arguments from the
  model's `model_params` block plus a `seed`, returning an unfitted
  sklearn-style estimator with `fit` / `predict_proba` that is
  deterministic for a fixed seed (import-side registration like
  `atropos.ml_ensemble` does for `"ensemble"` also works).
- **Ship defaults**: a non-empty
  `DEFAULT_TRAINING_CONFIG["model_params"]["name"]` block — the
  coverage test fails otherwise, and the bundled families' blocks
  are pinned against silent drift.
- **Four contract families pick the entry up automatically**
  (parametrized over `sorted(MODEL_REGISTRY)` in
  `tests/test_ml_training.py`): determinism (two same-seed fits
  agree bit-for-bit), probability validity, beats-logit on a planted
  nonlinearity, and the full engine pass. If your model structurally
  cannot pass one family, skip it only with the documented
  structural reason in `_PLANTED_BEATS_SKIP` (see `knn` and
  KNOWN-ISSUES.md §1c) — never weaken the check.

## Adding a hypothesis (the user workflow)

The framework's whole point; walk the doctrine, not around it:

1. **Prereg before data.** Build the params dict (name, hypothesis,
   construction, nulls, kill criteria, power calc, spec hash), freeze it
   with `atropos.prereg_generator.generate_prereg`, and commit the
   markdown *before* touching any data.
2. **One run, one verdict.** Call `atropos.run_full_pipeline` on a
   fixed snapshot with a fixed seed. The dict takes `returns` +
   `signal` (series mode, optional `features`), or `panel_signal` +
   `panel_returns` DataFrames (panel mode). Thresholds set in the prereg
   are the thresholds used — no post-hoc edits.
3. **Record the verdict.** SURVIVES earns a live track record
   (`atropos.track_record.live_verdict` against the prereg); anything
   else gets a tombstone in your graveyard with COD and resurrection
   conditions (no graveyard yet? scaffold it with `atropos init`). Formats and the full pipeline doctrine: `FRAMEWORK.md`
   (§2 pipeline, §8 graveyard protocol) and `TEMPLATES.md`.

## For framework developers

The developer rulebook (review standards, test layers, perf budgets)
lives in `contributor/AGENTS.md`. Deep methodology docs
(`FRAMEWORK.md`, `METHODOLOGY.md`, `CASES_AND_FLAGS.md`) are in Russian;
code, docstrings, tests and demos are English — keep new code English.
Case-study excerpts quoted in skills and docs come from a closed
donor corpus that is not shipped; the portability test
(`tests/test_portability.py`) fails the suite on any machine-specific
trace or donor-repo reference — never paste real logs, paths or user
names into the tree.
