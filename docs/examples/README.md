# docs/examples — committed demo transcripts

Committed outputs of the package demos, so a reader can see what the
library produces without running anything first.

## ml_training_demo.log

Captured with:

```bash
uv run python -m atropos.ml_training
```

Deterministic (frozen seeds): the five case **verdicts** are the
contract and are pinned by tests; float details may differ across BLAS
builds. Case 5 is a statistics-calibrated reproduction of a documented
tombstone from the closed donor corpus ("VRP panic") — it demonstrates
the engine reproducing a kill verdict, not a re-test of the original
data.

Regenerate and diff by hand; a durable drift check lives in
`tests/test_examples.py`.
