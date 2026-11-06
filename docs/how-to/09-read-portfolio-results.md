# How-to 09. Reading portfolio results: concurrent slots, the correlation cap, and the three gates

> **Estimated time:** 15 minutes (the snippets run in ~2 seconds)
>
> **You will need:** the atropos repository with a set-up environment
> (`uv sync`), Python 3.12. No data needed — the world here is a
> seeded three-instrument OHLC panel, and every number below is real
> output. The snippets run in sequence in one session: from step 2 on
> they reuse the variables of step 1.

## What the portfolio layer adds

How-to 08 replayed one slot cap over an abstract candidate stream.
`atropos.portfolio` generalizes the discipline to N concurrent slots
over a real OHLC panel and adds the two controls the single-slot
engine cannot express: correlation-aware admission (a candidate whose
recent return proxy is |Spearman|-correlated with an open position's
above the cap never enters) and portfolio-level gates. Two entry
points: `run_portfolio_replay` (the multi-slot replay: ledgers,
exposure, event equity — no gates, `portfolio_verdict`
`"not_evaluated"`) and `run_portfolio` (the full conveyor: replay +
gates + verdict — the function a user calls).

The admission policies are the same vocabulary as how-to 08, plus
portfolio-only reject reasons:

| Policy | Slots full, newcomer arrives | Displaced trade credits | Reject reason |
|---|---|---|---|
| `fcfs` | reject (first-come-first-served) | — | `capacity` |
| `replace_low` | if its score STRICTLY beats the worst open score: close the worst, take the newcomer | `0` (a mid-flight close is assumed to give back) | `inferior_score` |
| `replace_high` | same selection rule as `replace_low` | its full isolated `r_pess` | `inferior_score` |

Reasons beyond the policy: `correlation_cap` (the filter runs BEFORE
any policy decision — a correlated candidate never enters, free slot
or not) and `cooldown` (every slot is blocked after an exit). The
REPLACE policies exist because of the donor finding (how-to 08,
step 2): on a measured 1040-trade stream, FCFS capping rejected
BETTER trades (+0.484R rejected mean vs +0.441R accepted). It is a
tendency, not a law — in this how-to's own world FCFS ends at +0.011R
while `replace_low` ends at -0.069R. Measure on YOUR stream.

```python
from atropos.portfolio import (
    run_portfolio_replay,
    run_portfolio,
    filter_by_correlation,
)
```

## Step 1. Build the inputs: normalized scores, an OHLC panel, R proxies

A signal frame needs `decision_idx`, `side`, `instrument`, `sl`,
`tp`, `score`; optional `priority`, `atr_entry`, `risk_ref`, and
`r_proxy` (required when `corr_cap` is set). Two contracts matter:

- **`score` must be within-group rank-normalized**
  (`atropos.ranking.rank_normalize_group`, how-to 08 step 4) — the
  replay compares candidates across model groups, and raw
  cross-model probabilities are not comparable. The hand-set scores
  below stand in for normalized output.
- **`r_proxy` is the instrument's GROWING return series from ONE
  common anchor through the row's decision bar** (here: returns
  since bar 0) — the replay extends open positions' snapshots to the
  candidate's bar from the candidate's own cell, so the correlation
  verdict pairs identical days by construction.

The price panel is a DataFrame with a 2-level `(field, instrument)`
column MultiIndex: one OHLC quadruple per instrument on one shared
bar grid. The world: `BBB` is a near-clone of `AAA` (correlation-cap
material), `CCC` runs its own independent walk.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_bars = 120
factor = rng.normal(0.0, 0.4, n_bars)
closes = {
    "AAA": 100.0 + np.cumsum(factor + rng.normal(0.0, 0.2, n_bars)),
    "BBB": 100.0 + np.cumsum(0.97 * factor + rng.normal(0.0, 0.1, n_bars)),  # AAA clone
    "CCC": 100.0 + np.cumsum(rng.normal(0.0, 0.4, n_bars)),  # independent walk
}

fields = ("open", "high", "low", "close")
ohlc = {field: {} for field in fields}
for name, close_series in closes.items():
    opens = np.concatenate(([100.0], close_series[:-1]))
    wick = np.abs(rng.normal(0.0, 0.2, n_bars))
    ohlc["open"][name] = opens
    ohlc["high"][name] = np.maximum(opens, close_series) + wick
    ohlc["low"][name] = np.minimum(opens, close_series) - wick
    ohlc["close"][name] = close_series

prices = pd.DataFrame(
    np.column_stack([ohlc[f][k] for f in fields for k in closes]),
    columns=pd.MultiIndex.from_product([list(fields), list(closes)]),
)

schedule = [  # (decision bar, instrument, side, score -- already normalized)
    (40, "AAA", "long", 0.50),
    (44, "AAA", "long", 0.60),
    (45, "BBB", "long", 0.80),   # clone of the open AAA position
    (45, "CCC", "long", 0.70),   # independent same-bar control
    (52, "CCC", "long", 0.95),   # replacement bait: beats the 0.50 holder
    (80, "CCC", "short", 0.65),  # gives the exposure series both sides
]

rows = []
for decision, instrument, side, score in schedule:
    fill = float(closes[instrument][decision + 1])
    is_long = side == "long"
    history = closes[instrument][: decision + 1]
    rows.append(
        {
            "decision_idx": decision,
            "side": side,
            "instrument": instrument,
            "sl": fill * (0.97 if is_long else 1.03),
            "tp": fill * (1.06 if is_long else 0.94),
            "score": score,
            "atr_entry": 1.0,
            "r_proxy": np.diff(history) / history[:-1],  # returns since bar 0
        }
    )
signals = pd.DataFrame(rows)
print(signals[["decision_idx", "side", "instrument", "score"]].to_string())
```

Example output:

```text
   decision_idx   side instrument  score
0            40   long        AAA   0.50
1            44   long        AAA   0.60
2            45   long        BBB   0.80
3            45   long        CCC   0.70
4            52   long        CCC   0.95
5            80  short        CCC   0.65
```

## Step 2. The replay: slots, reasons, exposure, event equity

```python
replay = run_portfolio_replay(signals, prices, slots=2, policy="fcfs", hold_bars=24)
print(replay["summary"])
print("accepted:", [(r["instrument"], r["decision_idx"], r["slot"]) for r in replay["accepted"]])
print("rejected:", [(r["instrument"], r["decision_idx"], r["reason"]) for r in replay["rejected"]])
print("equity :", replay["equity_curve"].round(3).tolist())
print("gross bars 40-46:", replay["exposure"]["gross"][40:47].tolist())
```

Example output:

```text
{'slots': 2, 'policy': 'fcfs', 'n_accepted': 3, 'n_rejected': 3, 'n_replaced': 0, 'n_correlation_rejects': 0, 'n_unresolved': 0, 'ev': 0.010519049217449006, 'dd': 0.39098077188696534, 'concurrent_max': 2}
accepted: [('AAA', 40, 0), ('AAA', 44, 1), ('CCC', 80, 0)]
rejected: [('BBB', 45, 'capacity'), ('CCC', 45, 'capacity'), ('CCC', 52, 'capacity')]
equity : [-0.309, -0.391, 0.011]
gross bars 40-46: [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
```

How to read:

- **Decisions lead fills by one bar** (the generator rule): AAA
  decided at 40 fills at 41 — the gross slice turns 1.0 at bar 41
  and 2.0 at bar 45, never above the cap.
- **FCFS admits by arrival**: both slots are busy from bar 45 on, so
  BBB (the clone), the same-bar CCC control, and the 0.95-score CCC
  all die with `capacity` — the best-scored candidate of the day is
  in the reject pile, the how-to 08 finding reproduced.
- **The equity curve is EVENT-based**: one point per credited trade
  (here 3), R credited at each trade's exit bar — not a per-bar
  series; aggregation cannot flatter `max_drawdown_r` (0.391).
  Per-bar positions live in `exposure` (`gross` / `net` /
  `concurrent`). Each accepted ledger row extends its input row with
  `slot`, `fill_idx`, `exit_idx`, `r_pess`, `credit_bar`, `credit_r`,
  `replaced`, `force_exit_idx`, `force_close_r`.

## Step 3. REPLACE policies and force_close_r

```python
low = run_portfolio_replay(signals, prices, slots=2, policy="replace_low", hold_bars=24)
high = run_portfolio_replay(signals, prices, slots=2, policy="replace_high", hold_bars=24)
print("low  ev:", round(low["final_equity_r"], 3), "| high ev:", round(high["final_equity_r"], 3))
print("replaced:", [(r["instrument"], r["decision_idx"], r["score"]) for r in low["replaced"]])
for row in low["replaced"]:
    print(
        "displaced:", row["instrument"], "at bar", row["force_exit_idx"],
        "| force_close_r:", round(row["force_close_r"], 3),
    )
print("credits (low):", [(r["instrument"], r["decision_idx"], round(r["credit_r"], 3)) for r in low["accepted"]])
```

Example output:

```text
low  ev: -0.069 | high ev: -0.536
replaced: [('AAA', 40, 0.5), ('AAA', 44, 0.6), ('CCC', 45, 0.7)]
displaced: AAA at bar 45.0 | force_close_r: -0.388
displaced: AAA at bar 45.0 | force_close_r: -0.333
displaced: CCC at bar 52.0 | force_close_r: 0.076
credits (low): [('AAA', 40, 0.0), ('AAA', 44, 0.0), ('BBB', 45, 0.359), ('CCC', 45, 0.0), ('CCC', 52, -0.829), ('CCC', 80, 0.401)]
```

How to read:

- **Replacement cascades**: BBB (0.80) displaces the worst open
  holder AAA (0.50) at bar 45; the same-bar CCC (0.70) displaces the
  next-worst AAA (0.60); the 0.95 CCC later displaces the 0.70 CCC.
  Strictly-higher scores only — ties never displace.
- **`force_close_r` is the honest partial hold**: the displaced
  trade's P&L RECOMPUTED to the replacement bar (AAA closed at
  -0.388 on its way to an isolated -0.309 at bar 64; the row's
  isolated `r_pess`/`exit_idx` stay stale, mirroring the engine's
  contract).
- **The credit follows the policy, not the reconstruction**:
  `replace_low` credits a displaced trade 0; `replace_high` keeps
  its full isolated `r_pess` — here `high - low = -0.468`, exactly
  the sum of the three displaced `r_pess` values (two negative,
  which is why the "optimistic" policy lands LOWER). Read the pair
  as the bookkeeping band on displacement, never as a forecast.
  And scores are not promises: the 0.95-score CCC lost -0.829R.

## Step 4. The full conveyor: gates and the verdict

```python
report = run_portfolio(
    signals, prices, slots=2, policy="fcfs", corr_cap=0.7, hold_bars=24
)
print("verdict (default min_trades=30):", report["portfolio_verdict"])
print("rejected:", [(r["instrument"], r["decision_idx"], r["reason"]) for r in report["rejected"]])

powered = run_portfolio(
    signals, prices, slots=2, policy="fcfs", corr_cap=0.7, hold_bars=24,
    min_trades=3,
)
print("verdict (min_trades=3):", powered["portfolio_verdict"])
for gate in powered["gates"]:
    print(gate["gate"], "| value", round(gate["value"], 3), "| floor", gate["floor"], "|", gate["reason"])
```

Example output:

```text
verdict (default min_trades=30): UNDERPOWERED
rejected: [('AAA', 44, 'correlation_cap'), ('BBB', 45, 'correlation_cap'), ('CCC', 52, 'correlation_cap')]
verdict (min_trades=3): DEAD(deflated_sharpe)
deflated_sharpe | value 0.0 | floor 0.95 | dsr 0.000 below floor 0.950
drawdown_floor | value -0.385 | floor -0.2 | drawdown -0.385 reached the floor -0.200
side_concentration | value 1.0 | floor 0.8 | same-side share 1.000 above cap 0.800
```

How to read:

- **The correlation cap fired three times, for two reasons**: AAA at
  44 stacks a second signal on its own OPEN AAA position (|rho| = 1
  — same-instrument stacking is the extreme case); BBB at 45 is the
  clone of that open position; CCC at 52 stacks on the CCC admitted
  one bar earlier. The independent CCC control on bar 45 was ADMITTED
  — the layer rejects correlation, not co-timing.
- **`UNDERPOWERED` is undecided, not a pass**: only 3 trades were
  credited (the equity curve's length IS the credited-trade count),
  below the default `min_trades=30`. Verdict vocabulary:
  `SURVIVES` / `DEAD(<gate>)` for the first triggered gate /
  `UNDERPOWERED`. The gates, in fixed order (each a dict with
  `gate`, `triggered`, `value`, `floor`, `reason`):

| Gate | Reads | `value` | Default floor | Breach condition |
|---|---|---|---|---|
| `deflated_sharpe` | per-period portfolio EV stream (R credited per `ev_period_bars` bucket) | the DSR | `dsr_min=0.95` | DSR below the floor, or non-finite (fail-closed) |
| `drawdown_floor` | event equity curve | max drawdown (negative number) | `dd_floor=-0.20` | `value <= floor` — REACHING the floor breaches |
| `side_concentration` | `concurrent` + `net` exposure series | max same-side share `0.5 + \|net\| / (2 * concurrent)` | `max_same_side_share=0.80` | share above the cap; skipped honestly (NaN) without exposure data |

- On a random-walk world all three fire: DSR ~ 0 (sparse credits),
  drawdown -0.385 breaches -0.20, and an all-long book has same-side
  share 1.000. `DEAD(deflated_sharpe)` is the expected verdict for
  noise — the gates are doing their job.

## Step 5. The correlation cap: proxies, not probabilities

Why raw model probabilities cannot stand in for `r_proxy`: the cap
compares CO-MOVEMENT of instruments, and a probability measures model
CONFIDENCE — a logit's 0.9 says nothing about whether the instrument
moves with one already open. The proxy is the instrument's per-bar
return series on one common anchor; Spearman (rank) correlation is
deliberate — scale-free, so proxies from differently priced
instruments compare fairly. Contract: the candidate's series ends at
ITS decision bar and every open position's series is extended to that
same bar (the replay does this from the candidate's own cell; calling
the filter standalone, slice the shared anchor yourself).

```python
def proxy_through(name, bar):
    # growing common-anchor series: returns since bar 0 through `bar`
    history = closes[name][: bar + 1]
    return np.diff(history) / history[:-1]

# the replay contract: every proxy ends at the CANDIDATE's decision bar
candidates = pd.DataFrame(
    {
        "instrument": ["BBB", "CCC"],
        "r_proxy": [proxy_through("BBB", 45), proxy_through("CCC", 45)],
    }
)
aaa_open = {"slot_0": proxy_through("AAA", 45)}

print("BBB clone   -> admit:", filter_by_correlation(candidates, 0, aaa_open, corr_cap=0.7))
print("CCC control -> admit:", filter_by_correlation(candidates, 1, aaa_open, corr_cap=0.7))
print("under 20 overlapping bars -> admit:",
      filter_by_correlation(candidates, 0, aaa_open, corr_cap=0.7, min_overlap=50))
```

Example output:

```text
BBB clone   -> admit: False
CCC control -> admit: True
under 20 overlapping bars -> admit: True
```

How to read: the clone of an open position is rejected, the
independent instrument on the same bar admits, and a pair with too
few overlapping bars admits too — `min_overlap` trades must exist
for a correlation estimate, and insufficient data is not correlation
(inconclusive is never a reject).

## Traps

- **Feeding raw `predict_proba` as `score`.** Normalize per group
  first (`rank_normalize_group`); the replay compares across groups.
- **Feeding a model score as `r_proxy`.** The proxy is a RETURN
  stream on a common anchor; the cap reads co-movement, not belief.
- **Reading `UNDERPOWERED` as a pass.** It is undecided — collect
  more trades or lower `min_trades` in the prereg BEFORE the run.
- **Comparing policy `ev`s as forecasts.** `replace_low` /
  `replace_high` are bookkeeping bounds on displacement; their
  spread is honest uncertainty, not a prediction interval. And
  `equity_curve` is event-based — use `exposure` for per-bar.
- **Stacking same-instrument signals under `corr_cap`.** |rho| = 1
  against an open position dies under any cap below 1; pyramiding is
  a different, untested policy — freeze what you test in the prereg.

## What next

- Single-slot admission mechanics, ties, and the donor finding —
  how-to 08 (`docs/how-to/08-read-admission-results.md`).
- The full pipeline around admission (simulate → rank → normalize →
  admit → metrics) — how-to 06
  (`docs/how-to/06-run-backtest-pipeline.md`).
- Where the portfolio layer sits in the framework —
  `POSITIONING.md`.
- Module reference: the `atropos.portfolio` docstring (policies, the
  score contract, gate semantics), or run the demo:
  `uv run python -m atropos.portfolio`.
