---
name: 17-backtest-pipeline
description: Build the full backtest pipeline in Atropos — simulate → state machine → ranking → admission → metrics; use when asked to backtest a signal, compare rankers, or run admission policies
when_to_use: after a hypothesis SURVIVES the kill protocol, to test tradability
version: 0.1.0
---

# SKILL 17: BACKTEST PIPELINE — simulate → rank → admit → metrics

> Tradability test of a SURVIVING candidate. The executable golden
> path is `docs/examples/backtest_pipeline.py`: copy the order, don't
> invent. Public API: `atropos.backtesting` (sim, state machine,
> admission, per-trade metrics) and `atropos.ranking` (ranking
> oracles, `RANKING_METRICS`, `rank_normalize_group`,
> `RANKER_SUITABILITY`).

## WHEN TO APPLY / NOT APPLY

APPLY after a hypothesis SURVIVES the kill protocol (skills `02` /
`04`), to test TRADABILITY: does the signal survive honest replay
one slot at a time, under fees, slippage, gaps, capacity caps and an
admission policy. Atropos is a FILTER and a reference pipeline — it
renders an honest verdict on what is given; it does not make a
candidate pass.

NOT for:

- parameter optimization. Tuning thresholds/parameters to make the
  metrics look better is result shopping; parameters are frozen in
  the prereg (skill `01`). If asked to "optimize parameters" — STOP
  (see below).
- a first verdict. The backtest pipeline never substitutes for the
  kill protocol: no SURVIVES from skills 02/04 → no backtest.
- portfolio construction beyond the three admission policies.

## GOLDEN PATH (7 steps — copy the order, don't invent)

```python
from atropos.backtesting import (
    CostModel,
    run_admission_policy,
    run_position_state_machine,
    simulate_signal,
)
from atropos.backtest_metrics import per_trade_sharpe
from atropos.ranking import (
    RANKING_METRICS,
    rank_normalize_group,
    rank_within_groups,
)
```

1. **Data.** Four 1-D finite equal-length price series
   (`opens/highs/lows/closes`) + a signal table with `ts_us`,
   `entry_idx`, `side`, `stop`, `target`. A synthetic seeded world is
   honest as long as it is marked SYNTHETIC (see `build_ohlc_world`
   in the example).
2. **Isolated replay.** `simulate_signal(opens, highs, lows, closes,
   entry_idx, side, stop_price, target_price, hold_bars, atr_value,
   *, risk_ref=None, cost=DEFAULT_COST)` → `{"r_opt", "r_pess",
   "exit_idx"}`; `exit_idx == -1` means no exit inside the hold
   window and BOTH R values are NaN.
3. **State machine.** `run_position_state_machine(signals_df, *,
   allow_reverse=False, same_bar_reentry=False, cooldown=0, ...)`
   decides which candidates actually trade. Input columns
   `decision_idx` / `side` / `exit_idx` (+ optional `priority`);
   output adds `status` / `reason` / `force_exit_idx` /
   `force_close_r`. A force-close RECOMPUTES the P&L: read
   `force_close_r`, the row's isolated `r_pess`/`exit_idx` are stale.
4. **Rank within groups.** `rank_within_groups(features, target,
   group_ids, *, model="gbm", top_s=1, embargo_groups=1, config=None,
   seed=42)` — expanding past-only P(win) ranking over
   `atropos.ml_training.MODEL_REGISTRY`. Report keys: `n_scored`,
   `n_skipped`, `scored_group_ids`, `selected`, `groups` (each a
   `CandidateGroup` carrying RAW scores).
5. **Normalize.** `rank_normalize_group(scores)` maps ONE group's raw
   scores to average-rank positions in `[0, 1]` (returns a fresh
   array; `CandidateGroup.scores` are never normalized in place).
6. **Admission.** `run_admission_policy(trades, *, cap,
   policy="fcfs")` over trade dicts `ts_entry` / `ts_exit` / `s` (the
   NORMALIZED score) / `r_pess`; policies `fcfs` | `replace_low` |
   `replace_high`. Returns `accepted` / `rejected` / `replaced` /
   `equity_curve` / `max_drawdown_r` / `summary`.
7. **Metrics.** `RANKING_METRICS` (`ndcg@k`, `precision@k`, `mrr`,
   `spearman`; signature `metric(ideal_order, scored_order, top_k)`,
   candidate-id strings, RAW orderings) to compare rankers on the
   INTERSECTION of `scored_group_ids`; per-trade statistics
   (`per_trade_sharpe`, `trade_curve_stats`, `bucketed_sharpe`,
   `pooled_stats`, defined in `atropos.backtest_metrics`) on the
   accepted stream's `r_pess`.

`maker_entry_sim(..., fill_price, atr_value, *, delta_atr=0.25,
wait_bars=4, ...)` is the maker-entry variant of step 2: a limit
`delta_atr * ATR` into the zone from the market-fill reference, at
most `wait_bars` bars to fill; a missed fill contributes exactly
0.0, an unfillable zone returns NaN.  Fills are path-deterministic.

## INVIOLABLE INVARIANTS (violation = invalid backtest)

1. **Past-only + embargo.** Group `t` is scored by a model fitted on
   groups `< t - embargo_groups` only; keep `embargo_groups >= 1`
   (the default) so same-regime neighbours stay out of the train fit.
2. **Groups never cross a fold.** Groups are contiguous
   chronological blocks; a group id reappearing after a later group
   started raises `ValueError`. Rankers under comparison MUST use the
   same group-id set.
3. **Raw ≠ normalized.** `CandidateGroup.scores` are raw model
   probabilities; normalize separately with `rank_normalize_group`;
   the admission policy reads the normalized value (`s`), never raw
   cross-model probabilities.
4. **Deterministic tie-break.** `(score desc, candidate_id asc)` on
   the `CandidateGroup` path, `(score desc, position asc)` on the
   DataFrame path. Never re-sort by anything else.
5. **Seed fixed bit-for-bit.** Pass `seed=` explicitly to
   `rank_within_groups` (the explicit argument wins over any
   `model_seed` in `config`) and report it; the seed is part of the
   verdict.
6. **Gap-through-stop is a scratch.** An entry bar opening beyond the
   stop resolves as an immediate market scratch — never a +1R win in
   gap-distance units. Do not "repair" this pessimism.
7. **`n_skipped` groups are NOT in the metrics.** Compute metrics on
   scored groups only (`scored_only`); when comparing two rankers,
   compare on the INTERSECTION of `scored_group_ids` so unequal skip
   sets cannot fake a difference.
8. **lightgbm is optional.** The pointwise path (`rank_within_groups`)
   runs without it; only the `rank_with_lambdarank` oracle imports
   it, lazily. Never add a hard lightgbm import to a pipeline script.

## ANTI-PATTERNS (each one has invalidated a real backtest)

- Comparing raw `predict_proba` across model groups without
  `rank_normalize_group`: a logit 0.9 and a kNN 0.7 do not mean the
  same thing.
- Shuffling rows before walk-forward ranking: chronology IS the
  contract; shuffled group ids either raise (interleaved blocks) or
  silently redefine "past".
- Hand-reconstructing the P&L of a force-closed trade instead of
  reading `force_close_r` (the machine recomputes it through
  `simulate_signal`; the isolated row values are stale).
- Ignoring `embargo_until` when building the arms of
  `paired_bootstrap_diff`: enforcement belongs to the fold scheduler
  that produced the per-group values; the bootstrap resamples
  whatever it is handed.
- Reading SURVIVES (kill protocol) as tradable: **SURVIVES ≠
  tradable** — this skill exists precisely because they differ.
- Treating `ev_threshold` as an alias of
  `atropos.ml_training.pick_threshold`: inclusive `>=` + the `-inf`
  all-trades floor + an `(thr, ev, n)` tuple, vs strict `>` + a bare
  float.
- Averaging `ndcg@k` / `mrr` over different `n_skipped` sets (or over
  the full scored lists of two runs): compare on the intersection of
  `scored_group_ids`.
- Inventing fees/slippage: the `CostModel(gen_slip, comm, gap,
  e_mult, x_mult)` values come from the user or the prereg; naming
  the engine defaults is allowed only as an explicit, reported choice.

## CHOOSING A RANKER (group size drives the choice)

`RANKER_SUITABILITY` in `atropos/ranking.py` is an honest marking,
not a gate (the registry accepts every model):

| Candidates per group | Use first | Avoid |
|---|---|---|
| >= 10 | `gbm`, `hist_gbm`, `logit` (suitability high) | `knn`, `qda` (low: noisy probabilities on small groups) |
| 3–9 | `logit`, `hist_gbm` | `knn`, `qda`; `mlp` wants more rows |
| < 3 | `logit` | everything else — almost no ranking contrast |

- `extratrees` / `rf` / `mlp` are suitability `medium`: usable,
  especially after `rank_normalize_group`.
- An ensemble is legal only when built from high/medium members.
- `lambdarank` (`rank_with_lambdarank`) is the dedicated graded
  oracle — not a pointwise substitute; lightgbm optional.

## STOP AND ASK THE USER IF

- the **target** is not defined (horizon, what counts as a win, the
  executable window) — never invent a label;
- the **group_id** is not defined (a group = a day? a bar? an event?)
  — the meaning of every ranking metric depends on it;
- there are **no CostModel parameters** (fees/slippage) — do not
  invent; ask for the numbers or an explicit "use the engine
  defaults" decision;
- the **admission policy** is not chosen (`fcfs` vs `replace_low` vs
  `replace_high`) — the choice changes the equity curve and the
  drawdown;
- a group has **< 10 candidates** and the user asks for `knn`/`qda`
  (suitability `low` there) — confirm explicitly;
- the user asks to **"optimize parameters"** — that is result
  shopping; offer a new prereg (skill `01`) instead.

## OUTPUT CONTRACT

Every backtest report must state:

- metrics computed `scored_only` (and on the INTERSECTION of
  `scored_group_ids` for ranker comparisons) + `n_skipped`;
- `admission_policy` and `cap`;
- `ranker` (the `model` name from the ranking report), `seed`,
  `top_s`, `embargo_groups`;
- `cost_model` — the `CostModel` field values actually charged;
- data provenance: real snapshot id, or SYNTHETIC + seed;
- and the line: **SURVIVES ≠ tradable.**
