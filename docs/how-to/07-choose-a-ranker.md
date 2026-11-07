# How-to 07. Choosing a ranker: pointwise vs lambdarank, and reading RANKING_METRICS

> **Estimated time:** 15–20 minutes (the snippets run ~3 minutes in
> total; lightgbm only in the last step)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.12. The candidate world here is synthetic
> (seeded, reproducible). `lightgbm` is an OPTIONAL dependency — only
> step 5 needs it (`uv sync --extra ranking`, or
> `uv pip install lightgbm`); every other path runs
> without it.

## What the ranking layer does

`atropos.ranking` picks the BEST candidate of each decision point
(group = one day / one bar) by expected value, trained strictly on
PAST groups. Two oracles share that schedule:

| Oracle | Call | Semantics | Dependency |
|---|---|---|---|
| pointwise `P(win)` | `rank_within_groups` | any registry classifier, features → binary win flag; the default and the recommendation | none extra |
| lambdarank | `rank_with_lambdarank` | the donor `LGBMRanker` port (`objective="lambdarank"`, graded relevance, fixed hyper-parameters) | `lightgbm`, imported lazily inside the call |

Group `t` is scored by a model fitted on groups `< t - embargo_groups`
(the embargo keeps the same-regime neighbor out of the train fit), so
every score is past-only by construction. Honest suitability marks per
registry model — information, not a gate — live in `RANKER_SUITABILITY`:

```python
from atropos.ranking import RANKER_SUITABILITY

print(RANKER_SUITABILITY)
```

Example output:

```text
{'extratrees': 'medium', 'gbm': 'high', 'hist_gbm': 'high', 'knn': 'low', 'logit': 'high', 'mlp': 'medium', 'qda': 'low', 'rf': 'medium'}
```

Reading rule: `high` = stable per-candidate probabilities on small
groups (`gbm`, `logit`, `hist_gbm`); `medium` = usable, especially
after `rank_normalize_group` (`rf`, `extratrees`, `mlp`); `low` =
noisy probabilities on small candidate groups (`knn`, `qda`) — for
groups under ~10 candidates prefer the `high` marks. The registry
accepts every model; nothing is refused, the marks are advisory.

Import paths (module paths work in every release; the package also
re-exports the entry points at the top level, so
`from atropos import rank_within_groups` works too):

```python
import numpy as np
import pandas as pd

from atropos.ranking import (
    RANKER_SUITABILITY,
    RANKING_METRICS,
    ev_threshold,
    paired_bootstrap_diff,
    rank_within_groups,
)
```

## Step 1. A candidate world and three rankers

80 days x 6 candidates; one feature (`edge`) really drives the latent
win probability, two are pure noise. A candidate "wins" stochastically
by its latent quality — the rankers must recover that from past days
only:

```python
def build_rank_world(n_groups: int = 80, per_group: int = 6, seed: int = 9):
    """80 days x 6 candidates; 'edge' drives the latent win probability."""
    rng = np.random.default_rng(seed)
    rows, labels, groups = [], [], []
    quality = rng.standard_normal((n_groups, per_group))
    for day in range(n_groups):
        latent = quality[day]
        p_win = 1.0 / (1.0 + np.exp(-2.0 * latent))
        wins = (rng.random(per_group) < p_win).astype(np.int64)
        for pos in range(per_group):
            rows.append(
                {
                    "edge": latent[pos] + 0.4 * rng.standard_normal(),
                    "noise_a": rng.standard_normal(),
                    "noise_b": rng.standard_normal(),
                }
            )
            labels.append(int(wins[pos]))
            groups.append(day)
    return (
        pd.DataFrame(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


frame, labels, day_id = build_rank_world()
print(frame.shape, "win rate", round(float(labels.mean()), 3))


def group_mrr(report: dict) -> list[float]:
    """Per-group mrr@2 of a rank_within_groups report (RAW scores)."""
    vals = []
    for grp in report["groups"]:
        ideal = [
            cid
            for cid, _ in sorted(
                zip(grp.candidates, grp.labels, strict=True),
                key=lambda pair: (-pair[1], pair[0]),
            )
        ]
        scored = [
            cid
            for cid, _ in sorted(
                zip(grp.candidates, grp.scores, strict=True),
                key=lambda pair: -pair[1],
            )
        ]
        vals.append(RANKING_METRICS["mrr"](ideal, scored, top_k=2))
    return vals


books = {}
for name in ("gbm", "logit", "knn"):
    report = rank_within_groups(
        frame,
        labels,
        day_id,
        model=name,
        top_s=1,
        # the FIRST scored group trains on one group (6 rows): size the
        # smallest early window (docstring warning, made explicit here)
        config={"model_params": {"knn": {"n_neighbors": 5}}},
    )
    mrrs = group_mrr(report)
    books[name] = dict(zip(report["scored_group_ids"], mrrs))
    print(
        f"{name:<6} n_scored={report['n_scored']}"
        f" n_skipped={report['n_skipped']}"
        f" mean mrr@2={float(np.mean(mrrs)):.3f}"
    )
```

Example output (~1 minute: every scored group refits its model):

```text
(480, 3) win rate 0.504
gbm    n_scored=78 n_skipped=0 mean mrr@2=0.404
logit  n_scored=78 n_skipped=0 mean mrr@2=0.397
knn    n_scored=78 n_skipped=0 mean mrr@2=0.385
```

How to read. With 6 candidates, a random ordering scores mrr@2 = 0.25,
so all three beat chance — but the ordering matches the suitability
marks (`knn` last, as advertised: with `n_neighbors=5` on the 6-row
cold-start windows its probabilities are noisy). Two structural notes:

- **`n_scored=78`, not 80**: the first `embargo_groups + 1` days have
  no usable past and are skipped before scoring; `n_skipped` counts
  only additional single-class-window skips (0 here).
- **Size the factory params for the smallest early window.** The first
  scored group trains on ONE group (6 rows): the explicit
  `n_neighbors=5` above is the docstring's warning made explicit.

## Step 2. The `low` marks are real: qda on the same world

```python
# qda, same world: the cold-start window kills the fit (the docstring
# warning made real -- more samples than features, per class, needed)
try:
    rank_within_groups(frame, labels, day_id, model="qda", top_s=1)
except Exception as err:  # noqa: BLE001 (guide demo: show, don't swallow)
    print("qda ->", type(err).__name__, ":", str(err)[:110])
```

Example output:

```text
qda -> LinAlgError : The covariance matrix of class 1 is not full rank. When using `solver='svd'` the number of samples in each cla
```

How to read: on 6-candidate groups the first train window holds ~1-2
positive rows against 3 features — QDA's covariance is rank-deficient
and the fit errors propagate LOUDLY (by design; nothing is silently
swallowed). qda/knn carry their `low` marks for reasons you can
reproduce in one minute.

## Step 3. Reading RANKING_METRICS right

The registry (`atropos.ranking.RANKING_METRICS`) holds `ndcg@k`,
`precision@k`, `mrr`, `spearman` — each takes
`(ideal_order, scored_order, top_k)` candidate-id strings and returns
one float per group. Three rules, all pinned:

- **RAW scores.** Metrics consume the raw model ordering: within one
  group raw and normalized scores induce the same order, and the
  semantics are pinned to raw so callers never have to guess. The
  ideal order comes from `CandidateGroup.labels` (NaN = untradeable;
  the `group_mrr` helper above breaks label ties deterministically by
  candidate id).
- **Scored groups only.** Metrics run over `report["groups"]` — the
  groups the oracle actually scored — never over the skipped ones.
- **`mrr` is single-winner**: relevance is `ideal_order[0]` (the
  trading reading: one candidate wins the group); `spearman` needs
  `top_k >= 2`; `top_k` is capped at the group size. Register your own
  metric the same open way: `RANKING_METRICS["ap@k"] = my_fn`.

## Step 4. Compare two rankers honestly (intersection + paired bootstrap)

`n_skipped` is per-run: two rankers can end up with different
`scored_group_ids` sets, and comparing metrics over unequal sets lets
the skip pattern fake a difference. Compare on the INTERSECTION, and
attach a CI with the paired bootstrap:

```python
common = sorted(set(books["gbm"]) & set(books["knn"]))
print("intersection gbm & knn:", len(common), "of",
      len(books["gbm"]), "/", len(books["knn"]))
diff = paired_bootstrap_diff(
    np.array([books["gbm"][g] for g in common]),
    np.array([books["knn"][g] for g in common]),
    n_boot=2000,
    seed=7,
)
print(diff)
```

Example output:

```text
intersection gbm & knn: 78 of 78 / 78
{'diff': 0.019230769230769232, 'lo': -0.057692307692307696, 'hi': 0.08990384615384528, 'p_le0': 0.3355, 'n': 78}
```

How to read: on this small world the gbm-vs-knn mean edge (0.019 mrr)
is WITHIN NOISE — the 95% CI spans zero, `p_le0` ≈ 0.34. That is the
bootstrap doing its job: a 0.02 head-to-head difference on 78 groups
is not a finding. The arms here are per-group floats of the SAME
groups (NaN pairs are dropped, rows resampled jointly); the caller owns
the group set being comparable — `embargo_until` is carried on
`CandidateGroup` but NOT enforced inside the bootstrap by default
(pinned in the docstring; `atropos.ranking.compare_rankers_on_intersection`
makes the intersection the default path in one call). Opt-in
enforcement exists: `paired_bootstrap_diff(..., enforce_embargo=True)`
drops embargo-violating pairs when the fold metadata is supplied and
warns when the flag is set without it (KNOWN-ISSUES §5b).

## Step 5. When to reach for the lambdarank oracle

Pointwise P(win) is the recommendation whenever a binary "did the
candidate win" label exists and the group is small — every registry
model supports it, and the marks above tell you which ones behave.
The lambdarank oracle earns its extra dependency when you have GRADED
utilities (not just 0/1) and care about the ORDER inside the top-k
more than the win flag; it ports the donor corpus' fixed
hyper-parameters so the oracle is reproducible. `lightgbm` is imported
lazily inside the call — the module and every other path run without
it.

```python
from atropos.ranking import rank_with_lambdarank

lambda_report = rank_with_lambdarank(
    frame, labels.astype(float), day_id, top_s=1
)
lambda_mrrs = group_mrr(lambda_report)
print(
    f"lambdarank n_scored={lambda_report['n_scored']}"
    f" n_skipped={lambda_report['n_skipped']}"
    f" mean mrr@2={float(np.mean(lambda_mrrs)):.3f}"
)
```

Example output (~1 minute, needs `lightgbm`):

```text
lambdarank n_scored=78 n_skipped=0 mean mrr@2=0.378
```

How to read: on this world the listwise oracle does not beat the
pointwise gbm (0.378 vs 0.404) — no surprise, the label IS a win flag
and the groups are tiny. The two oracles rank by different semantics;
compare their picks through `RANKING_METRICS` as in step 4, never
assert them equal, and remember the engine never picks between them —
the caller decides, per prereg.

## Step 6. The EV threshold: two look-alike functions

`ev_threshold` (`atropos.ranking`) and `pick_threshold`
(`atropos.ml_training`) both scan a probability grid for the best
EV cut — and are deliberately NOT interchangeable:

| | `ev_threshold` (ranking) | `pick_threshold` (ml_training) |
|---|---|---|
| taking rule | `proba >= threshold` (inclusive) | `proba > threshold` (strict) |
| grid | the grid PLUS the all-trades floor `-inf` | the bare grid; no-trade falls back to the HIGHEST grid value |
| returns | `(threshold, ev, n_taken)` | the threshold float only |

```python
from atropos.ml_training import pick_threshold

proba = np.array([0.2, 0.6, 0.8])
exec_vals = np.array([0.12, 0.03, -0.06])
grid = (0.5, 0.7)
print("pick_threshold :", pick_threshold(proba, exec_vals, grid=grid))
print("ev_threshold   :", ev_threshold(proba, exec_vals, grid=grid))
```

Example output:

```text
pick_threshold : 0.5
ev_threshold   : (-inf, 0.03, 3)
```

How to read: same inputs, opposite cuts. `pick_threshold` returns 0.5 —
a cut whose taken book EVs at -0.015 (it never sees the all-trades
floor). `ev_threshold` adds the floor: taking everything EVs +0.03, no
grid cut beats it, so the answer is `(-inf, 0.03, 3)`. The difference
is pinned by tests; use the ranking variant when "maybe take nothing"
is an honest answer, the ML-engine variant where the bare grid is the
preregistered contract.

## Traps

- **knn/qda on small groups.** The `low` suitability marks are the
  docstring's honest shape: distance-weighted kNN probabilities
  compress toward the base rate and QDA needs more samples than
  features per class — both reproduce their failure modes in minutes
  (steps 1-2). For groups under ~10 candidates: `gbm`, `logit`,
  `hist_gbm`.
- **Fit errors on the cold-start window are loud, by design.** The
  first scored group trains on `embargo_groups + 1`-deep past at most
  one group wide; size `model_params` for THAT window (step 1).
- **Do not compare rankers over unequal scored sets.** Intersect
  `scored_group_ids` first, then bootstrap (step 4).
- **Do not feed admission the raw scores these metrics read.** Metrics
  want RAW; admission wants `rank_normalize_group` output (how-to 06,
  step 5). Both rules are pinned; mixing them is the classic leak.
- **The engine never picks a ranker.** `RANKER_SUITABILITY` is
  information, not a gate; your prereg freezes the choice before the
  data.

## Checklist

- [ ] The oracle choice (pointwise model or lambdarank) is frozen in
      the prereg, with `model_params` sized for the smallest early
      window.
- [ ] Ranker comparisons run on the INTERSECTION of
      `scored_group_ids`, with `paired_bootstrap_diff` CIs, not naked
      means.
- [ ] Metrics read RAW scores of SCORED groups only; admission gets
      `rank_normalize_group` output.
- [ ] `ev_threshold` vs `pick_threshold` chosen deliberately (floor +
      tuple vs bare grid + float), not by autocomplete.
- [ ] lightgbm imported only where lambdarank runs; every other path
      stays dependency-free.

## What next

- Wiring the oracle into the full pipeline — how-to 06
  (`docs/how-to/06-run-backtest-pipeline.md`).
- Admission policies and reading their output — how-to 08
  (`docs/how-to/08-read-admission-results.md`).
- The ML kill protocol (where `pick_threshold` lives) — how-to 05
  (`docs/how-to/05-run-ml-kill-protocol.md`).
- What the backtest layer is (and is not) — `POSITIONING.md`.
