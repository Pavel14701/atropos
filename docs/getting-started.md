# Getting started: fixture to verdict

One pass through the stack on a tiny synthetic fixture: materialize a
verified bundle, classify its feature, replay a one-leg strategy
through the Book. Every block is executable —
`uv run pytest --doctest-glob="*.md" docs/getting-started.md`.

## 1. Materialize: lookahead is verified, not assumed

An adapter normalizes a source; `materialize` resolves every field
AS-OF onto the decision grid, then VERIFIES each column against an
independently recomputed cutoff — a resolver bug flips the flags below
instead of passing silently. We serve two fields from the in-memory
`_TableAdapter` (the minimal adapter behind the package's own
doctests); the outcome carries `lag=1` (the honest forward label) and
its leading NaN row is dropped, so 8 bars materialize to 7 rows.

>>> import pandas as pd
>>> from atropos.data_contract import MaterializedData, MaterializeSpec
>>> from atropos.data_contract import FieldMeta, FieldSpec, _TableAdapter, materialize
>>> bars = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
>>> mom = pd.Series([0.2, -0.5, 0.8, 0.4, -0.9, 0.1, -0.3, 0.6], index=bars)
>>> booked = pd.Series([0.01, -0.004, 0.012, 0.006, -0.011, 0.002, -0.007, 0.009], index=bars)
>>> fields = {"mom": FieldMeta("mom"), "booked_return": FieldMeta("booked_return")}
>>> adapter = _TableAdapter(fields, {"mom": mom, "booked_return": booked})
>>> md = materialize(
...     adapter,
...     MaterializeSpec(
...         outcome=FieldSpec("booked_return", role="outcome", lag=1),
...         features=(FieldSpec("mom"),), train_frac=0.6,
...         min_train_rows=4, min_test_rows=4, nan_policy="drop",
...     ),
... )
>>> md.manifest["lookahead_checks"]
{'booked_return': True, 'mom': True}
>>> md.manifest["validation"]["passed"]
True

The manifest (`bundle_hash`, splits, hashes) is what a prereg cites.

## 2. Feature analysis: one classification per feature

`feature_analysis` consumes the bundle directly — on this path the
manifest is VALIDATED (a failed lookahead check or validation battery
is fatal). The report chains missingness, clustering, the orthogonal
check and the bucket scan, then classifies each representative feature
into one of six cells (`STRONG`, `REGIME_DEPENDENT`,
`STRONG_UNVERIFIED`, `SHAPE_UNKNOWN`, `PROXY`, `NOISE`) plus a
recommended action under the same key. A 7-row fixture proves the
mechanics; exact-cell pins need the planted 800-row demo world of
`atropos.feature_scan`.

>>> from atropos.feature_scan import feature_analysis
>>> report = feature_analysis(md, n_perm_orth=25, min_bucket_n=0,
...                           label="getting-started")
>>> sorted(report["classification"]) == ["mom"]
True
>>> report["classification"]["mom"] in {
...     "STRONG", "REGIME_DEPENDENT", "STRONG_UNVERIFIED",
...     "SHAPE_UNKNOWN", "PROXY", "NOISE",
... }
True

## 3. The Book: decisions in, TradeRecords out

The Book needs the instrument's OHLC as feature columns (the MVP price
path), so this fixture is a manifest-less `MaterializedData` built
inline — fine for a demo, never for research data. One leg: long when
`mom > 0`, 2-ATR stop, `close` is the instrument. Exits are delegated
to the public `simulate_intents`; `run_book` gives each leg a fresh
Book so legs cannot interact (multi-leg composition:
`atropos.strategy.run_strategy`). Carry rates are REQUIRED — a silent
zero flatters shorts.

>>> from atropos.book import run_book
>>> from atropos.legs import (
...     Compare, EntrySpec, ExitSpec, InstrumentRef, LegSpec, StopSpec,
... )
>>> book_md = MaterializedData(
...     outcome=pd.Series(0.0, index=bars),
...     features=pd.DataFrame({
...         "open": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0],
...         "high": [101.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0],
...         "low": [99.0, 99.0, 100.0, 101.0, 99.0, 98.0, 97.0, 96.0],
...         "close": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0],
...         "mom": [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
...         "atr": [0.5] * 8,
...     }, index=bars),
...     controls=pd.DataFrame(index=bars),
...     index=pd.DataFrame({"ts": bars, "entity": "", "split": "train"}, index=bars),
... )
>>> leg = LegSpec(
...     name="demo_long", direction="long",
...     entry=EntrySpec(Compare("mom", ">", 0.0)),
...     exit=ExitSpec(stop=StopSpec("atr", 2.0)),
...     instrument=InstrumentRef("field", {"field": "close"}),
...     features_used=("mom", "atr", "close"),
...     rationale="demo fixture: momentum entry, 2-ATR stop",
... )
>>> out = run_book(
...     (leg,), book_md,
...     borrow_bps_annual=500.0, funding_bps_annual=100.0,
...     periods_override=252.0,
... )
>>> out["per_leg"]["demo_long"]["n_trades"]
1
>>> record = out["records"]["demo_long"][0]
>>> (record["exit_reason"], record["entry_bar"], record["exit_bar"])
('stop', 2, 4)

`exit_reason` is the diagnostic to read first: this leg's only trade
died on its stop — the signal never got to pay. `records_to_features`
(from `atropos.strategy`) flattens these records into the
analysis-ready frame every analytic tool consumes.

Next: `docs/architecture.md`, `docs/migration.md` (v11 upgrades),
`CONTRIBUTING.md` (house laws).
