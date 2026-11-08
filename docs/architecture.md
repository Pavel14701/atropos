# Architecture: five layers, one direction

Data flows one way: a source is materialized under the contract,
legs decide, the Book simulates, strategies compose, analytics
classify. Each layer consumes only the layer below it — analytics
never touch raw frames, the Book never invents costs, legs never hold
state.

## Layer 1 — the data contract (`atropos.data_contract`)

Look-ahead is structurally impossible, not discouraged. A source
normalizes to the `Adapter` protocol (`fields` / `fetch` / `as_of` /
`validate_self` / `extent`; public adapters: `OHLCVAdapter`,
`TidyAdapter`). `FieldMeta` declares each field's `t_known_offset` —
the value labelled `t` is known at `t + t_known_offset` (bar closes 0,
bar opens `-bar_delta`). `materialize(adapter, MaterializeSpec)` cuts
every column at `cutoff = t_decision - t_known_offset - lag *
bar_delta`, then INDEPENDENTLY re-derives the cutoff and verifies the
produced labels against it: a resolver bug flips
`manifest["lookahead_checks"]` to False and `materialize` raises.
The result, `MaterializedData` (outcome/features/controls aligned
as-of, a `ts`/`entity`/`split` index, the `DataManifest`), is the
only bundle framework functions should trust; its timestamp-based
train/test split carries a wall-clock embargo.

## Layer 2 — legs (`atropos.legs`)

A leg is a declaration, not a program. Entry/exit gates are a
Predicate AST (`Compare`/`And`/`Or`/`Not`, `evaluate_predicate`,
`from_predicate_dict`); `LegSpec` layers them (entry gate + Book
throttles, exit mechanisms, instrument field, execution shape,
sizing) and `validate_features_declared` audits the feature contract
in both directions at construction. `LegEngine` is STATELESS:
`evaluate(leg, md, ts) -> LegDecision`, no memory, no clock — the
same leg on the same bar always decides the same way.

## Layer 3 — the Book (`atropos.book`)

The imperative shell around the pure engine. The Book owns what the
engine must not — the open position (one slot per leg), the entry
cooldown, the equity curve in R — walks the decision grid once per
leg and emits one `TradeRecord` per completed trade. Stop/target/time
exits are resolved EXCLUSIVELY through the public
`atropos.backtesting.simulate_intents` (the private scan is never a
path), so "a Book trade == a simulated trade" holds by construction.
`TradeRecord.exit_reason` ("stop"/"target"/"time"/"signal"/"forced"/
"netted", mapped in exactly one place) is the prime diagnostic — read
the reason histogram before the P&L. `run_book` isolates legs (a
fresh Book each); `Book.walk_portfolio` shares ONE position pool and
equity ledger across legs ("sequential"/"priority"/"netting",
`netted_remainder` clamped so a reduction is never a loss).

## Layer 4 — strategy composition (`atropos.strategy`)

A strategy is `tuple[LegSpec, ...] + PortfolioPolicy` — one shape for
one leg or four. `PortfolioPolicy` declares the carry policy
(`borrow_bps_annual` / `funding_bps_annual` are REQUIRED — the carry
policy is never silently defaulted) and the combination:
"independent" re-dispatches to `run_book`; the shared modes delegate
to `walk_portfolio`. `run_strategy` returns one unified result shape
(records, equity, per-leg, exposure, summary). Caps are diagnostics,
not enforcement. `records_to_features` flattens TradeRecords into the
frame the analytic tools consume.

## Layer 5 — analysis (`atropos.feature_scan`, `atropos.orthogonal`)

Analytics consume ONLY record shapes and materialized bundles:
`feature_analysis(md)` chains missingness → clustering →
`orthogonal_check` → bucket scan → classification (one of
`NOISE` / `PROXY` / `SHAPE_UNKNOWN` / `STRONG_UNVERIFIED` /
`REGIME_DEPENDENT` / `STRONG` per representative feature, plus a
recommended action); on a bundle path the manifest is validated
(failed lookahead or battery = fatal). The same tools run on trades:
`records_to_features` output feeds `orthogonal_check` /
`bucket_scan` / `feature_importance` directly.

## Key decisions

1. **Look-ahead is caught, not assumed.** The contract verifies
   every column against an independently recomputed cutoff; a failed
   check kills `materialize` and poisons `feature_analysis`. Nothing
   in the stack trusts "the data was probably fine".
2. **One cost source.** Costs live in the Book's `CostModel` and the
   carry policy on `Book`/`PortfolioPolicy` (REQUIRED rates,
   calendar-derived periods); `ExecutionSpec` carries none, so a
   cost change never mutates a leg, and costs sit INSIDE the
   pessimistic R (`pnl_gross_r` is reconstructed, never measured).
3. **Exploratory never becomes STRONG.** Discovery, importance and
   rule generation are diagnostics in their own report blocks; the
   `EXPLORATORY` cell never merges into the classification, and only
   a VERIFIED regime indicator (with `prereg_hash`) can upgrade a
   feature to `STRONG`.
4. **Distribution `atropos-quant`, import `atropos`.**
   `pip install atropos-quant`, `from atropos import ...`, CLI
   `atropos-quant` (the plain `atropos` command belongs to an
   unrelated bioinformatics package; `import atropos_quant` fails on
   purpose). Same split as scikit-learn/sklearn.
