# How-to 06. Running the backtest pipeline: from signals to an admitted book

> **Estimated time:** 20–25 minutes (the snippets run ~1 minute in
> total)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.12; numpy/pandas/scipy come as dependencies.
> The data here is a synthetic seeded OHLC world (reproducible
> byte-for-byte). The snippets run in sequence in one session: from
> step 2 on they reuse the variables of step 1.

## When to reach for this

The kill-protocol answers "is the hypothesis dead?"; a `SURVIVES`
verdict leaves the next question open: **is the survivor tradeable at
all** once one slot at a time, a real taker cost model, and a
ranking oracle are imposed on it? The backtest pipeline answers that
with a reference replay:

```
simulate -> rank -> normalize -> admit -> metrics
```

It is a library you wire into your own pipeline (see `POSITIONING.md`
for the boundary: this is not vectorbt/backtrader/qlib, and it does not
optimize parameters). This how-to walks the golden path step by step;
the committed one-file version is `docs/examples/backtest_pipeline.py` —
run it first to see the whole transcript:

```bash
uv run python docs/examples/backtest_pipeline.py
```

Example output (deterministic, seed=41):

```text
==============================================================================
STAGE 1: isolated replay of 161 candidate signals
==============================================================================
decided trades: 161/161 (mean r_pess -1.529R, mean r_opt -1.151R)

==============================================================================
STAGE 2: position state machine (one slot, no reverse)
==============================================================================
slots filled: 63 of 161 signals

==============================================================================
STAGE 3: rank candidates within days (gbm, past-only)
==============================================================================
groups scored: 56  selections: 98

==============================================================================
STAGE 4: RANKING_METRICS — gbm vs random on the same groups
==============================================================================
  ndcg@k       gbm 0.800  random 0.805
  precision@k  gbm 0.852  random 0.815
  mrr          gbm 0.759  random 0.556
  spearman     gbm 0.185  random 0.185

==============================================================================
STAGE 5: maker-entry vs market entry (limit at fill - 0.25 ATR, wait 4 bars)
==============================================================================
maker r_pess=+0.000R  market r_pess=-1.779R  (maker fills are path-deterministic; unfilled zone -> NaN)

backtest pipeline: all stage checks passed
```

The rest of this how-to rebuilds the same path from smaller pieces.
Imports (module paths — they work in every release, and the package
also re-exports the entry points at the top level, so
`from atropos import simulate_signal` works equally well):

```python
import numpy as np
import pandas as pd

from atropos.backtesting import (
    CostModel,
    run_admission_policy,
    run_position_state_machine,
    simulate_signal,
)
from atropos.backtest_metrics import pooled_stats
from atropos.ml_training import rolling_z
from atropos.ranking import (
    RANKING_METRICS,
    rank_normalize_group,
    rank_within_groups,
)

HOUR_US = 3_600_000_000
```

## Step 1. A world, candidate signals, one isolated sim

A seeded trend + panic OHLC world (hourly bars; two panic bursts so
stops, time exits and the gap-through-stop contract all get exercised),
and candidate signals from causal z-score momentum crossings on both
sides:

```python
def build_world(n_bars: int = 2600, seed: int = 41) -> pd.DataFrame:
    """Trend + panic hourly OHLC world (deterministic, seed=41)."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate(
        [
            np.full(n_bars // 3, 0.0004),
            np.full(n_bars // 3, -0.0006),
            np.full(n_bars - 2 * (n_bars // 3), 0.0004),
        ]
    )
    close = 100.0 + np.cumsum(drift + 0.004 * rng.standard_normal(n_bars))
    for start in (n_bars // 3 + 20, 2 * (n_bars // 3) + 40):
        close[start : start + 8] -= np.linspace(0.8, 0.1, 8)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.standard_normal(n_bars)) * 0.3 + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    atr = pd.Series(high - low).rolling(20).mean().bfill().to_numpy()
    stamps = (
        int(np.datetime64("2024-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_bars, dtype=np.int64) * HOUR_US
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "ts_us": stamps,
        }
    )


def make_signals(world: pd.DataFrame, seed: int = 41) -> pd.DataFrame:
    """Candidate signals: causal z-score momentum crossings, both sides."""
    z = rolling_z(world["close"].to_numpy(), 48)
    rng = np.random.default_rng(seed)
    rows = []
    last_band = 0.0
    for pos in range(48, len(world)):
        crossed = (z[pos] > 1.5 and last_band <= 1.5) or (
            z[pos] < -1.5 and last_band >= -1.5
        )
        last_band = z[pos]
        if not crossed or not np.isfinite(z[pos]):
            continue
        rows.append(
            {
                "ts_us": int(world["ts_us"].iloc[pos]),
                "entry_pos": pos + 1,  # fill at next bar open (generator rule)
                "side": "long" if z[pos] > 0 else "short",
                "priority": int(round(abs(z[pos]) * 10)),
                "atr": float(world["atr"].iloc[pos + 1]),
                "sl": float(
                    world["close"].iloc[pos]
                    - np.sign(z[pos]) * 1.2 * world["atr"].iloc[pos + 1]
                ),
                "tp": float(
                    world["close"].iloc[pos]
                    + np.sign(z[pos]) * 2.0 * world["atr"].iloc[pos + 1]
                ),
                "noise": rng.standard_normal(),
            }
        )
    return pd.DataFrame(rows)


world = build_world()
signals = make_signals(world)
```

Each candidate is simulated in isolation through the taker cost model.
`simulate_signal` returns a `SimResult`: optimistic and pessimistic net
R and the exit bar:

```python
first = signals.iloc[0]
sim = simulate_signal(
    world["open"].to_numpy(),
    world["high"].to_numpy(),
    world["low"].to_numpy(),
    world["close"].to_numpy(),
    int(first["entry_pos"]),
    str(first["side"]),
    float(first["sl"]),
    float(first["tp"]),
    hold_bars=48,
    atr_value=float(first["atr"]),
    cost=CostModel(),
)
print("candidate signals:", len(signals))
print(
    "first signal -> exit_idx", sim["exit_idx"],
    "r_opt", round(sim["r_opt"], 3),
    "r_pess", round(sim["r_pess"], 3),
)
```

Example output:

```text
candidate signals: 161
first signal -> exit_idx 78 r_opt -1.381 r_pess -1.779
```

### Two engine contracts worth knowing

Both probes run on a six-bar hand-made world so the numbers stay
readable:

```python
opens = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0])
highs = opens + 0.5
lows = opens - 0.5
closes = opens + 0.2
```

**Gap-through-stop.** If the entry bar already OPENS beyond the stop,
the trade is a scratch at costs (exit on the entry bar) — never a +R
windfall in gap-distance units:

```python
normal = simulate_signal(
    opens, highs, lows, closes, 1, "long",
    stop_price=99.0, target_price=104.0, hold_bars=4, atr_value=1.0,
)
print("normal long :", normal["exit_idx"], round(normal["r_opt"], 3),
      round(normal["r_pess"], 3))

scratch = simulate_signal(
    opens, highs, lows, closes, 1, "long",
    stop_price=102.0,  # bar 1 opens at 101 < stop: gapped through overnight
    target_price=106.0, hold_bars=4, atr_value=1.0,
)
print("gap-through :", scratch["exit_idx"], round(scratch["r_opt"], 3),
      round(scratch["r_pess"], 3))
```

Example output:

```text
normal long : 4 0.448 0.372
gap-through : 1 -0.253 -0.404
```

How to read: the normal long walks to its time exit at bar 4; the
gapped long exits on the ENTRY bar at costs only (-0.253 / -0.404) —
the overnight gap never becomes a win in gap-distance units.

**Force-close.** With `allow_reverse=True`, an opposite-side signal
force-closes the open trade at its decision bar; the row's isolated
`exit_idx`/`r_pess` stay STALE — read the recomputed `force_close_r`:

```python
probe = pd.DataFrame(
    {
        "decision_idx": [0, 3],
        "side": ["long", "short"],
        "exit_idx": [5, 5],
        "stop_price": [98.5, 104.0],
        "target_price": [104.0, 98.5],
    }
)
slots = run_position_state_machine(
    probe,
    allow_reverse=True,
    opens=opens,
    highs=highs,
    lows=lows,
    closes=closes,
)
print(
    slots[["side", "status", "exit_idx", "force_exit_idx", "force_close_r"]]
    .to_string(float_format=lambda v: f"{v:.3f}")
)
```

Example output:

```text
    side status  exit_idx  force_exit_idx  force_close_r
0   long  taken         5           3.000          0.697
1  short  taken         5             NaN            NaN
```

How to read: the long's isolated `exit_idx` says 5, but it was
force-closed at bar 3 — `force_close_r` (0.697) is the honest P&L; the
stale `exit_idx` 5 is bookkeeping of the never-happened isolated exit.

## Step 2. The state machine: which signals become trades

Live, one slot is occupied at a time; the dataset builder above
simulated every candidate in isolation. `run_position_state_machine`
walks the signals chronologically and labels each `taken`/`skipped`
(decision bar fills at `decision_idx + 1` by the generator rule;
`cooldown` blocks decision bars after an exit):

```python
def replay_signals(world: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Replay every candidate in isolation through the cost model."""
    records = []
    for _, row in signals.iterrows():
        entry = int(row["entry_pos"])
        if entry >= len(world):
            continue
        sim = simulate_signal(
            world["open"].to_numpy(),
            world["high"].to_numpy(),
            world["low"].to_numpy(),
            world["close"].to_numpy(),
            entry,
            str(row["side"]),
            float(row["sl"]),
            float(row["tp"]),
            hold_bars=48,
            atr_value=float(row["atr"]),
            cost=CostModel(),
        )
        records.append(
            {
                **row.to_dict(),
                "r_opt": sim["r_opt"],
                "r_pess": sim["r_pess"],
                "exit_idx": sim["exit_idx"],
            }
        )
    frame = pd.DataFrame(records)
    for column in ("r_opt", "r_pess", "exit_idx", "priority", "atr"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


replayed = replay_signals(world, signals)
decided = replayed[replayed["exit_idx"] >= 0]
print(
    f"decided trades: {len(decided)}/{len(replayed)} "
    f"(mean r_pess {decided['r_pess'].mean():+.3f}R, "
    f"mean r_opt {decided['r_opt'].mean():+.3f}R)"
)
slot_input = signals.rename(columns={"entry_pos": "decision_idx"})
exit_by_stamp = replayed.set_index("ts_us")["exit_idx"]
exits = exit_by_stamp.loc[slot_input["ts_us"]].to_numpy().astype(int)
fill_idx = slot_input["decision_idx"].to_numpy() + 1
exits = np.where(exits == -1, -1, np.maximum(exits, fill_idx))
slot_input["exit_idx"] = exits
slots = run_position_state_machine(slot_input, cooldown=2)
taken = slots[slots["status"] == "taken"]
skipped = slots[slots["status"] == "skipped"]
print(f"slots filled: {len(taken)} of {len(signals)} signals")
print("skip reasons:", skipped["reason"].value_counts().to_dict())
```

Example output:

```text
decided trades: 161/161 (mean r_pess -1.529R, mean r_opt -1.151R)
slots filled: 63 of 161 signals
skip reasons: {'in_position': 94, 'cooldown': 3, 'same_bar_after_exit': 1}
```

How to read: 161 candidates collapse into 63 trades — the slot
discipline is the point. The skip reasons name the mechanism:
`in_position` (the slot was busy), `cooldown` (blocked by the 2-bar
cooldown), `same_bar_after_exit` (a signal decided on the exit bar;
set `same_bar_reentry=True` if your construction allows it).

## Step 3. Rank candidates within days (past-only)

Candidates are grouped by calendar day of the decision stamp; the
oracle is a registry model fitted on the EXPANDING PAST ONLY
(`embargo_groups=1` keeps the same-regime neighbor out of each train
fit). Labels: a candidate "wins" if its realized `r_pess` was positive
(`NaN` would mark an untradeable candidate). The report's `groups` are
`CandidateGroup` objects carrying RAW model scores:

```python
signals = signals.sort_values("ts_us").reset_index(drop=True)
day_id = (signals["ts_us"] // (24 * HOUR_US)).to_numpy()
labels = (
    (replayed.set_index("ts_us").loc[signals["ts_us"], "r_pess"] > 0)
    .astype(np.int64)
    .to_numpy()
)
feature_frame = signals[["priority", "noise", "atr"]].rename(
    columns={"atr": "atr_at_signal"}
)
report = rank_within_groups(
    feature_frame,
    labels,
    day_id,
    model="gbm",
    top_s=2,
    embargo_groups=1,
    seed=42,
)
group = [g for g in report["groups"] if len(g.candidates) >= 4][-1]
print(
    f"groups scored: {report['n_scored']}  skipped: {report['n_skipped']}"
    f"  selections: {len(report['selected'])}"
)
print(
    "last 4-candidate group:", group.group_id, group.candidates,
    "labels", group.labels.tolist(),
)
print("  raw model probabilities:", [f"{v:.2e}" for v in group.scores.tolist()])
```

Example output:

```text
groups scored: 56  skipped: 30  selections: 83
last 4-candidate group: 19804 ['125', '126', '127', '128'] labels [0.0, 0.0, 0.0, 0.0]
  raw model probabilities: ['3.03e-05', '4.63e-08', '4.40e-06', '2.42e-08']
```

How to read: the first `embargo_groups + 1` days have no usable past
(30 more days were skipped honest — a single-class train window picks
nothing), so 80 candidate days reduce to 56 scored groups. The raw
scores are tiny (a ~2% base rate) and INCOMPARABLE across models —
exactly why the next-but-one step normalizes them.

## Step 4. Score the ranking: RANKING_METRICS vs random

Metrics read the RAW scores (within a group raw and normalized scores
induce the same order) and take candidate-id strings:

```python
rng = np.random.default_rng(7)
for metric_name, metric in RANKING_METRICS.items():
    gbm_vals, random_vals = [], []
    for grp in report["groups"]:
        ideal = [
            cid
            for cid, _ in sorted(
                zip(grp.candidates, grp.labels, strict=True),
                key=lambda pair: -pair[1],
            )
        ]
        if len(ideal) < 2:
            continue
        scored_gbm = [
            cid
            for cid, _ in sorted(
                zip(grp.candidates, grp.scores, strict=True),
                key=lambda pair: -pair[1],
            )
        ]
        scored_random = rng.permutation(ideal).tolist()
        gbm_vals.append(metric(ideal, scored_gbm, top_k=2))
        random_vals.append(metric(ideal, scored_random, top_k=2))
    print(
        f"  {metric_name:<12} gbm {np.mean(gbm_vals):.3f}  "
        f"random {np.mean(random_vals):.3f}"
    )
```

Example output:

```text
  ndcg@k       gbm 0.800  random 0.805
  precision@k  gbm 0.852  random 0.815
  mrr          gbm 0.759  random 0.556
  spearman     gbm 0.185  random 0.185
```

How to read: the gbm oracle clearly beats random on `mrr` (0.759 vs
0.556) and `precision@k`, is at par on `ndcg@k` (graded gains are flat
in this world) and `spearman` (the head is too shallow). One metric
never tells the story — read all four, and compare rankers on the
INTERSECTION of their `scored_group_ids` (how-to 07). On a synthetic
world these margins mean little; on your data they are the honest
measure of the oracle.

## Step 5. Normalize scores: the admission seam

Raw cross-model probabilities are incomparable (a logit 0.9 and a kNN
0.7 mean different things); admission compares candidates from
DIFFERENT model groups only on within-group rank positions:

```python
norm = rank_normalize_group(group.scores)
print("raw       :", [f"{v:.2e}" for v in group.scores.tolist()])
print("normalized:", [round(float(v), 3) for v in norm.tolist()])
print("single-candidate group ->", rank_normalize_group(np.array([0.37])).tolist())
```

Example output:

```text
raw       : ['3.03e-05', '4.63e-08', '4.40e-06', '2.42e-08']
normalized: [1.0, 0.333, 0.667, 0.0]
single-candidate group -> [0.5]
```

How to read: the mapping is the average rank position, ties shared —
a tie can never leak into the normalized scale. A single-candidate
group maps to exactly 0.5: neutral, so it neither wins nor loses a
replacement contest.

## Step 6. Admit trades and read the book

Build the admission stream: every scored candidate with its normalized
score, isolated `r_pess`, and entry/exit stamps from the world. Then
replay two policies under the same cap:

```python
by_ts = replayed.set_index("ts_us")
trades = []
for grp in report["groups"]:
    gnorm = rank_normalize_group(grp.scores)
    for pos_in_group, cid in enumerate(grp.candidates):
        sig = signals.iloc[int(cid)]
        rep = by_ts.loc[sig["ts_us"]]
        entry_bar = int(sig["entry_pos"])
        exit_bar = max(int(rep["exit_idx"]), entry_bar + 1)
        trades.append(
            {
                "ts_entry": int(world["ts_us"].iloc[entry_bar]),
                "ts_exit": int(world["ts_us"].iloc[exit_bar]),
                "s": float(gnorm[pos_in_group]),
                "r_pess": float(rep["r_pess"]),
            }
        )
print("candidate trades for admission:", len(trades))
fcfs = run_admission_policy(trades, cap=2, policy="fcfs")
swap = run_admission_policy(trades, cap=2, policy="replace_low")
print(
    "fcfs        : accepted", len(fcfs["accepted"]),
    " rejected", len(fcfs["rejected"]),
    " ev", round(fcfs["final_equity_r"], 1), "R",
    " dd", round(fcfs["max_drawdown_r"], 1), "R",
)
print(
    "replace_low : accepted", len(swap["accepted"]),
    " rejected", len(swap["rejected"]),
    " replaced", len(swap["replaced"]),
    " ev", round(swap["final_equity_r"], 1), "R",
)
print(
    "fcfs rejected book : mean r_pess",
    round(float(np.mean([t["r_pess"] for t in fcfs["rejected"]])), 3),
)
print(
    "fcfs accepted book : mean r_pess",
    round(float(np.mean([t["r_pess"] for t in fcfs["accepted"]])), 3),
)
pooled = pooled_stats(
    [t["r_pess"] for t in swap["accepted"]],
    [t["ts_exit"] for t in swap["accepted"]],
)
print(
    "replace_low pooled: n={n} mean={mean:+.3f} dd={dd:.1f}R "
    "sharpe_per_trade={sharpe_per_trade:.3f}".format(**pooled)
)
```

Example output:

```text
candidate trades for admission: 98
fcfs        : accepted 72  rejected 26  ev -109.4 R  dd 109.4 R
replace_low : accepted 82  rejected 16  replaced 10  ev -113.8 R
fcfs rejected book : mean r_pess -1.662
fcfs accepted book : mean r_pess -1.52
replace_low pooled: n=82 mean=-1.536 dd=124.0R sharpe_per_trade=-2.167
```

How to read, three honest notes:

- **This synthetic world loses at taker costs** (mean `r_pess` ≈ -1.5R
  per trade): the point of the golden path is the PIPELINE mechanics,
  not the money. A real survivor gets here only after the kill-protocol.
- **The admission policies differ mechanically, not morally**: replace_low
  admitted 10 more trades by evicting worst-score holders (evicted
  trades contribute 0). In a losing world that means more losses booked;
  in a winning world it means more wins captured. Measure on your
  stream (how-to 08), do not assume either policy is "better".
- **The metrics are per-trade**: `sharpe_per_trade` is never annualized
  (-2.17 is a real per-trade number for a losing book, and it reads
  ~10x worse than the annualized Sharpe would — that is the point);
  the honest annualization is `bucketed_sharpe`, and `pooled_stats`
  returns both plus the naive t-stat and the R drawdown.

## Traps

- **Do not feed admission raw probabilities.** `s` must be a
  WITHIN-GROUP normalized score (`rank_normalize_group`); raw
  cross-model `predict_proba` outputs are not comparable and the
  replacement contests become noise.
- **Do not compare rankers over unequal scored sets.** `n_skipped` is
  per-run; compare on the intersection of `scored_group_ids` (how-to
  07).
- **Do not read the annualized Sharpe of a trade stream.**
  `sharpe_per_trade` is per-trade by contract; annualizing a trade
  sequence with `sqrt(n_trades)` inflates the number by an order of
  magnitude. Use `bucketed_sharpe` with timestamps.
- **`embargo_until` is carried, not enforced.** If your fold scheduler
  promises embargo windows, enforce them where you build the per-group
  values (the paired bootstrap of how-to 07 will not do it for you).
- **A passing backtest is not a verdict.** This pipeline is the
  reference check of a SURVIVOR; it does not replace the kill-protocol
  and cannot resurrect a killed hypothesis.

## Checklist

- [ ] The kill-protocol verdict is on file first; the backtest pipeline
      runs on a survivor, not on a hope.
- [ ] Every candidate replayed in isolation through the SAME
      `CostModel` you intend to trade with.
- [ ] The state machine contract respected: fills at
      `decision_idx + 1`, exits ≥ fill, force-closes read via
      `force_close_r`.
- [ ] Ranking labels carry `NaN` for untradeable candidates; the oracle
      is past-only (`embargo_groups >= 1` by default).
- [ ] Admission scores normalized per group; cap and policy frozen in
      the prereg before the replay.
- [ ] Book metrics read per-trade (`pooled_stats`), annualization only
      via `bucketed_sharpe`.

## What next

- Reading admission output: FCFS vs REPLACE, ties, the donor finding —
  how-to 08 (`docs/how-to/08-read-admission-results.md`).
- Choosing the ranking oracle and comparing rankers — how-to 07
  (`docs/how-to/07-choose-a-ranker.md`).
- Where this fits (and what it is not) — `POSITIONING.md`.
- The one-file version: `docs/examples/backtest_pipeline.py`.
