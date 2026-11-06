# How-to 08. Reading admission results: FCFS vs REPLACE, ties, and the raw-score trap

> **Estimated time:** 10 minutes (the snippets run in ~1 second)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.11+. No data needed — the candidate stream
> here is a hand-built four-trade example; every number is real
> output.

## What admission replays

The ranking oracle picks candidates per decision point (how-to 07);
the admission layer decides which of them GET TO TRADE under a slot
cap. `atropos.backtesting.run_admission_policy` replays a candidate
stream against `cap` concurrent slots under one of three policies:

| Policy | Slots full, newcomer arrives | Replaced trade contributes | Reject reason |
|---|---|---|---|
| `fcfs` | reject (first-come-first-served) | — | `capacity` |
| `replace_low` | if its score STRICTLY beats the worst open score: close the worst, take the newcomer | `0` (conservative: a mid-flight close is assumed to give back) | `inferior_score` |
| `replace_high` | same selection rule as `replace_low` | its full isolated `r_pess` (optimistic bound) | `inferior_score` |

Import (module paths; these sub-module paths stay valid in every
release, and the package also re-exports the entry points at the top
level, so `from atropos import run_admission_policy` works equally
well):

```python
from atropos.backtesting import run_admission_policy
```

## Step 1. One stream, three policies

A hand-built stream: the best trade (`s=0.91`) arrives while both
slots are busy; the worst (`s=0.40`) arrives only after slot 2 frees.
Scores below are already within-group normalized (step 4 shows why
that matters):

```python
trades = [
    # arrives first, holds slot 1 for a long time
    {"ts_entry": 0, "ts_exit": 30, "s": 0.62, "r_pess": 1.0, "candidate_id": 0},
    # fills slot 2, exits early
    {"ts_entry": 1, "ts_exit": 8, "s": 0.55, "r_pess": 0.4, "candidate_id": 1},
    # the BEST trade: arrives while both slots are busy -> FCFS rejects it
    {"ts_entry": 2, "ts_exit": 20, "s": 0.91, "r_pess": 1.8, "candidate_id": 2},
    # arrives once slot 2 has freed (ts 8): FCFS takes it despite the
    # worst score -- this is how the best trade ends up in the reject pile
    {"ts_entry": 8, "ts_exit": 12, "s": 0.40, "r_pess": -0.5, "candidate_id": 3},
]

fcfs = run_admission_policy(trades, cap=2, policy="fcfs")
print("fcfs summary :", fcfs["summary"])
print("fcfs accepted:", [(t["candidate_id"], t["s"]) for t in fcfs["accepted"]])
print("fcfs rejected:", [(t["candidate_id"], t["s"], t["reason"]) for t in fcfs["rejected"]])
print("fcfs equity  :", fcfs["final_equity_r"], "R; best-trade r_pess:", trades[2]["r_pess"], "R")

for policy in ("replace_low", "replace_high"):
    report = run_admission_policy(trades, cap=2, policy=policy)
    print(
        f"{policy:<12}:", report["summary"],
        "replaced:", [(t["candidate_id"], t["s"]) for t in report["replaced"]],
    )
```

Example output:

```text
fcfs summary : {'n': 3, 'ev': 0.9, 'dd': 0.5, 'replaced': 0}
fcfs accepted: [(0, 0.62), (1, 0.55), (3, 0.4)]
fcfs rejected: [(2, 0.91, 'capacity')]
fcfs equity  : 0.9 R; best-trade r_pess: 1.8 R
replace_low : {'n': 3, 'ev': 2.8, 'dd': 0.0, 'replaced': 1} replaced: [(1, 0.55)]
replace_high: {'n': 3, 'ev': 3.2, 'dd': 0.0, 'replaced': 1} replaced: [(1, 0.55)]
```

How to read:

- **FCFS admits by arrival, not by quality**: candidates 0, 1, 3 get
  the slots; the best trade (0.91, `r_pess` 1.8) dies with reason
  `capacity` and the worst (0.40, `r_pess` -0.5) trades because it
  happened to arrive when a slot was free.
- **REPLACE-low** evicts the worst open score (candidate 1, 0.55) when
  the 0.91 newcomer beats it STRICTLY; the evicted trade contributes 0
  (a mid-flight close is assumed to give back), so the book books
  `1.0 + 1.8 = 2.8R`.
- **REPLACE-high** makes the same selection but credits the evicted
  trade's full isolated `r_pess` (0.4) — `2.8 + 0.4 = 3.2R`, the
  optimistic bound. The spread `2.8 … 3.2` is the honest range of the
  replacement's value; reality lands somewhere between.

The output dict keys: `policy`, `cap`, `accepted` (trade dicts),
`rejected` (trade dicts + `reason`), `replaced` (trade dicts +
`reason="replaced"`), `equity_curve` (event-based cumulative R — R is
credited at each trade's `ts_exit`, so the drawdown is not flattered
by daily aggregation), `max_drawdown_r`, `final_equity_r`, and
`summary` (the compact `{"n", "ev", "dd", "replaced"}` row).

## Step 2. The donor finding: FCFS rejects better

The engine's robustness checks carried a measured result across from
the donor corpus (pinned in the docstring): on a 1040-trade stream,
**first-come-first-served capping rejected BETTER trades** — the
rejected mean `r_pess` was `+0.484R` vs `+0.441R` for the accepted,
a pure capacity loss whose reject rate drifted from 16% to 24% as
candidate density grew. Step 1 reproduces the mechanism in one glance:
under FCFS the accepted book means `+0.3R` per trade (1.0 + 0.4 - 0.5
over 3 trades) while the rejected pile holds `+1.8R`.

The REPLACE policies exist because of this finding. Two honest
caveats:

- **It is a measured tendency, not a law.** On the golden-path world of
  how-to 06 (a synthetic losing book) the picture flipped: the FCFS
  rejected book averaged -1.662R vs -1.520R for the accepted. In a
  negative-EV stream "more rejects" is the better deal. Measure on
  YOUR stream — that is exactly what `rejected`/`accepted` mean values
  are for.
- **Replacement raises turnover**: every eviction is a round trip the
  policy assumed away at 0R (low) or at full isolated R (high). Neither
  assumption is a fill you saw. Read the `2.8 … 3.2` band as the
  optimistic/pessimistic envelope, not as a forecast.

## Step 3. Ties: strict beats, candidate_id order

Two pinned rules make equal-score situations deterministic:

- **Equal score never displaces.** A newcomer must STRICTLY beat the
  worst open score; at equality it is rejected with
  `reason="inferior_score"`.
- **The worst holder is the lowest score, ties broken by
  `candidate_id`** (ascending; the stream arrival position when
  absent), so "which of two equal-scored holders would go" resolves
  deterministically.

```python
tie = [
    {"ts_entry": 0, "ts_exit": 30, "s": 0.50, "r_pess": 1.0, "candidate_id": 0},
    {"ts_entry": 1, "ts_exit": 30, "s": 0.50, "r_pess": 1.0, "candidate_id": 1},
    {"ts_entry": 2, "ts_exit": 40, "s": 0.50, "r_pess": 2.0, "candidate_id": 2},
]
tied = run_admission_policy(tie, cap=2, policy="replace_low")
print("tie accepted:", [(t["candidate_id"], t["s"]) for t in tied["accepted"]])
print("tie rejected:", [(t["candidate_id"], t["s"], t["reason"]) for t in tied["rejected"]])
```

Example output:

```text
tie accepted: [(0, 0.5), (1, 0.5)]
tie rejected: [(2, 0.5, 'inferior_score')]
```

How to read: candidate 2 carries the same 0.50 as both holders — not a
strict improvement, so it is rejected even though a "replace the worst"
contest would have been ambiguous. Deterministic, no order dependence,
no discretionary picks.

## Step 4. Never feed raw probabilities

Admission compares candidates from DIFFERENT model groups, so the
`s` score must be a within-group rank-normalized value
(`atropos.ranking.rank_normalize_group`, re-exported from
`atropos.backtesting`): best of group → 1.0, worst → 0.0, ties share
the average rank position, a single-candidate group → exactly `0.5`
(neutral in a replacement contest). Raw cross-model probabilities — a
logit's 0.9 next to a kNN's 0.7 — are not comparable and turn every
replacement decision into noise.

```python
import numpy as np

from atropos.ranking import rank_normalize_group

raw_scores = np.array([0.83, 0.83, 0.41])  # two-way raw tie
print("raw       :", raw_scores.tolist())
print("normalized:", rank_normalize_group(raw_scores).tolist())
print("single-candidate group ->", rank_normalize_group(np.array([0.37])).tolist())
```

Example output:

```text
raw       : [0.83, 0.83, 0.41]
normalized: [0.75, 0.75, 0.0]
single-candidate group -> [0.5]
```

How to read: the raw tie maps to 0.75/0.75 (the AVERAGE of the tied
rank positions), so a tie broken for display can never leak into the
admission scale. If a `CandidateGroup` frame is used upstream, pass
its scores through this call — never the raw `scores` column.

## Traps

- **Mixing policies across a prereg.** The admission policy is part of
  the frozen construction: pick `fcfs` / `replace_low` /
  `replace_high` before the replay, and do not swap after seeing the
  book.
- **Reading `ev` alone.** `summary["ev"]` is the final equity of the
  accepted stream; `dd` is its event-based drawdown. A policy that
  trades more will also draw down more — read the pair, and remember
  replaced trades contribute 0 (`replace_low`) or their isolated
  `r_pess` (`replace_high`).
- **Assuming FCFS is always worse.** The donor finding says FCFS
  rejected better on ONE measured corpus; on a negative-EV stream the
  reject-heavy policy wins (how-to 06, step 6 shows the flip). The
  point of the replay is to measure your stream, not to inherit a
  conclusion.
- **Feeding raw `predict_proba` output as `s`.** Normalize per group
  first (step 4); the docstring refuses nothing — it is your leak to
  make.
- **Equal scores do not displace.** If your construction NEEDS the
  newcomer to win ties, that is a different (and now untested) policy —
  do not nudge scores to fake it; freeze the policy in the prereg.

## What next

- The full pipeline around this step (simulate → rank → normalize →
  admit → metrics) — how-to 06
  (`docs/how-to/06-run-backtest-pipeline.md`).
- Choosing the oracle whose scores feed `s` — how-to 07
  (`docs/how-to/07-choose-a-ranker.md`).
- Where admission sits in the framework — `POSITIONING.md`,
  `FRAMEWORK.md` §9.
- Module reference: the `atropos.backtesting` docstring (policies,
  contracts, cost model constants).
