---
name: 02-run-pipeline
description: Run the hypothesis through atropos.ensemble_pipeline.run_full_pipeline and read one of the 14 verdicts correctly.
when_to_use: When the prereg is frozen and the data is pulled; for any decisive run on the frozen spec.
version: 0.1.0
---

# SKILL 02: RUN PIPELINE — one run, one verdict
> Code canonical. Map: FRAMEWORK.md:69-81 ("Implementation map"). Deviations: KNOWN-ISSUES §8.

> Core skill of the Atropos agent. Public API:
> `atropos.ensemble_pipeline.run_full_pipeline(data: dict,
> config: dict | None) -> dict`. Stages run in adversarial order:
> the first FAIL kills the candidate, the remaining stages are
> SKIPPED.

## PROCEDURE

1. **Assemble the data-dict.** Keys: `returns` (forward returns, a
   series), `signal` (same index), optionally `features` (a DataFrame
   of known controls), `mom7` / `mom30` (observed momentum factors;
   otherwise trailing sums), `hypothesis` (description). Panel mode:
   `panel_signal` / `panel_returns` (DataFrame: dates x assets) enable
   stage 7; panel inputs alone (stages 1-6 SKIPPED).
2. **Config on top of defaults.** `config` is merged over
   `DEFAULT_CONFIG`; notable keys: `cost.cost_bps` (one-way cost in
   bps, subtracted at stage 5), `model` (`"ols"` | `"knn"` |
   `"kernel"` | a callable factory, stage 6).
3. **One run.** No parameter tuning after the result; the seed is
   pinned inside the configs.
4. **Read the report.** `overall_verdict`, `killed_by_stage`,
   `kill_reason`, `stages` (per-stage dicts), `runtime_seconds`; plus
   the verdict metadata: `verdict_class` (DEAD / NEEDS_ANALYSIS /
   INCONCLUSIVE / SURVIVES), `informative`, `interpretation`,
   `next_step`, `what_would_change`, `failed_test`, `active_days`.

## EXAMPLE

```python
from atropos.ensemble_pipeline import run_full_pipeline

report = run_full_pipeline(
    {"returns": fwd, "signal": sig, "features": controls,
     "hypothesis": "H-0007 mom30 echo"},
    {"cost": {"cost_bps": 10.0}},
)
print(report["overall_verdict"], report.get("kill_reason"))
```

## VERDICT TABLE (14) — WHAT TO DO FOR EACH

| Verdict | Meaning | Agent action |
|---|---|---|
| `SURVIVES` | all stages passed, no warnings | tombstone `SURVIVED`. PASS ≠ confirmation: survived = a conditional candidate. For a deployed strategy include `track_record.live_verdict` |
| `SURVIVES_WITH_WARNINGS` | passed, but with warnings | tombstone `SURVIVED` with the warning list from `stages` |
| `REPACKAGING` | the signal is a proxy of a known feature (stages 1-2) | tombstone `KILLED`: the effect is real but borrowed; the family closes, successors start with prior <= 20% and a single run |
| `ARTIFACT` | artifact: subperiod concentration / placebo / calendar (stage 1) | tombstone `KILLED`, record the artifact mechanics |
| `DEAD` | absence of edge PROVEN on this feed: battery baseline catastrophically dead (mean <= 0 with power / no active days), or `CS_NO_EDGE` at config target `xs` | tombstone `KILLED`; stop pursuing on this feed — DEAD is not transferable across feeds |
| `ROBUSTNESS_DEATH` | robustness battery (stage 3): a specific structural test FAILED while the baseline held | tombstone `KILLED`; name the failed test and the mechanism it implies (smoothing inversion / thirds sign-flip / LOO non-persistence) |
| `PLACEBO_VIOLATION` | battery placebo lag-k produced non-zero edge (stage 3) — causality red flag | tombstone `KILLED`; audit the feed for lookahead (alignment, point-in-time semantics, warmup); do not trade until explained |
| `OVERFIT_DEATH` | walk-forward shrink gate (stage 4): OOS/IS shrink beyond the cap | tombstone `KILLED`; record the shrink ratio and whether the IS regime overlaps the OOS |
| `NO_OOS_EDGE` | no out-of-sample predictive edge: model-eval (stage 6) | tombstone `KILLED`; a FEATURE question — execution, microstructure, regime timing, carry |
| `COST_DEATH` | gross alive, net-of-cost dies (stage 5) | tombstone `KILLED`; record the `cost_bps` at which it dies |
| `RISK` | drawdown-tail or gross-Sharpe-floor breach (stage 5) | tombstone `KILLED` with the risk numbers |
| `CS_NO_EDGE` | cross-sectional IC / quantile spread empty (stage 7); DEAD class only at config target `xs`, default `ts` -> NEEDS_ANALYSIS | tombstone `KILLED` if XS was the target |
| `DEGENERATE_FEED` / `CONSTANT_PREDICTIONS` | UNINFORMATIVE: the test could not run (zero-variance / all-NaN feed pre-flight, or degenerate walk-forward `preds_std`) — the strategy is not judged | tombstone `INCONCLUSIVE`; fix the feed / model spec and re-run on a new snapshot |

`INCONCLUSIVE` at a stage (too little data, degenerate inputs) is an
honest outcome: the overall result is `INCONCLUSIVE`, reopening is
allowed only on a new data snapshot.

## NEVER

- Do not re-run "with slightly different parameters" after a verdict —
  one run, one verdict; a revision = a new hypothesis, a new ID.
- Do not read PASS as confirmation of an edge: a survivor is a
  conditional candidate with gates, not a leg of a system.
- Do not skip stages with manual scripts "for speed": the stage order
  itself is the protection.

## NEXT

Verdict recorded → tombstone + a row in the registry (see
`01-preregister.md` and `03-read-graveyard.md`). `SURVIVES` in
production → `atropos.track_record.live_verdict` (LIVE_OK /
LIVE_KILLED / INCONCLUSIVE; LIVE_OK is NOT confirmation of an edge).
