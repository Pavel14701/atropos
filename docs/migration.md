# Migrating from v11 to v15

Four waves sit between them: the v12 verdict taxonomy, the v13 data
contract + legs, the v14 composition + carry, and the v15
distribution rename. Each section lists what changed and what you
must do.

## v12 — verdict taxonomy

The generic `DEAD` verdict is gone. The former catch-all is split
into specific tokens — `ROBUSTNESS_DEATH` (a battery test failed
while the baseline held), `PLACEBO_VIOLATION`, `OVERFIT_DEATH`,
`NO_OOS_EDGE`, `COST_DEATH`, `DEGENERATE_FEED`,
`CONSTANT_PREDICTIONS` — and every report now carries a `verdict_class`
(`DEAD` / `NEEDS_ANALYSIS` / `INCONCLUSIVE` / `SURVIVES`) plus
`informative`, `interpretation` and `next_step`.

Do: match on the specific token or on `verdict_class == "DEAD"`, not
on the bare string `"DEAD"` in `overall_verdict`.

## v13 — data contract + legs

The framework gained a floor. `MaterializedData`
(`atropos.data_contract.materialize`) is the verified bundle:
as-of-resolved columns with `FieldMeta.t_known_offset` declarations,
per-field `lookahead_checks`, and a `DataManifest` whose
`bundle_hash` is what a preregistration cites. The look-ahead ban is
structural — verified, not assumed.

`atropos.legs` adds the Predicate AST (`Compare`/`And`/`Or`/`Not`),
the layered `LegSpec`, the stateless `LegEngine`, and the Book layer
(`atropos.book`: `Book`, `run_book`, `TradeRecord` with
`exit_reason` diagnostics). `atropos.adapters` ships `OHLCVAdapter`
and `TidyAdapter`.

Do: `feature_analysis(md)` now accepts a `MaterializedData` directly
— on that path the manifest is VALIDATED (a failed lookahead check or
validation battery is fatal; a missing manifest is a loud warning).
The bare-Series call still works but proves nothing about alignment;
`to_pipeline_data` is a LEGACY bridge that warns on every call.

## v14 — composition + carry

`Book.walk_portfolio` walks many legs over one decision grid with
shared state: `combination="sequential"` / `"priority"` /
`"netting"` (opposite intents close and are consumed; the closing
record's `exit_reason` is `"netted"` with a clamped
`netted_remainder` — a reduction is never a loss). `"independent"`
stays `run_book` (fresh Book per leg). `atropos.strategy` composes
`StrategySpec` = legs + `PortfolioPolicy`.

**BREAKING: carry parameters are REQUIRED.** `Book(...)`,
`run_book(...)` and `PortfolioPolicy(...)` now demand
`borrow_bps_annual` and `funding_bps_annual` (keyword-only, no
defaults). A silent zero would make every short look better than it
is; `0` is still allowed but warns loudly. `funding_bps_annual` is
sign-carried (negative legal); carry periods derive from the
manifest's calendar or `periods_override` — never a blind 365.

Do: add the two rates to every `Book` / `run_book` /
`PortfolioPolicy` call. If your old code implicitly assumed zero
carry, say so out loud with `0.0` and keep the warning.

## v15 — distribution rename

The PyPI distribution is `atropos-quant` (the plain `atropos` name is
taken by an unrelated bioinformatics package). The import is
unchanged — `from atropos import ...` — and the CLI command moves to
`atropos-quant`. `import atropos_quant` raises `ImportError` on
purpose.

Do: `pip install atropos-quant` (or `pip install -e .` from a clone);
keep every import as `atropos`; call the CLI as `atropos-quant`.
