# docs/api — auto-generated reference

Everything in this directory except this README is **generated** by
pdoc from the package docstrings; do not edit or review it by hand.
Regenerate from the repository root with the command below (pdoc runs
in an ephemeral `--with pdoc` environment; project dependencies are
unchanged). Modules are listed explicitly: a bare `pdoc atropos`
renders only the package page (verified on pdoc 16.0.0), so every
submodule is passed by name.

```bash
uv run --with pdoc python -m pdoc atropos atropos._utils atropos.adapters atropos.agent atropos.backtesting atropos.bootstrap_ci atropos.book atropos.cli atropos.cost_model atropos.cross_sectional atropos.data_contract atropos.ensemble_pipeline atropos.evaluation_metrics atropos.feature_scan atropos.incremental_regression atropos.instrument_map atropos.kill_chain atropos.legs atropos.ml_ensemble atropos.ml_training atropos.monte_carlo atropos.null_control_suite atropos.orthogonal atropos.paper_protocol atropos.portfolio atropos.portfolio_protocol atropos.power_calculator atropos.predictive_model_eval atropos.prereg_generator atropos.robustness_battery atropos.ranking atropos.strategy atropos.track_record atropos.walkforward_framework atropos.workspace -o docs/api
```

The command lists all package modules (31 as of the instrument-routing wave, including the agent subpackage and the v13/v14 modules: adapters, book, data_contract, feature_scan, legs, orthogonal, paper_protocol, strategy). A new `atropos/` module is
added to this list too — otherwise its pages never appear in or
update `docs/api`.

The entry point is `docs/api/index.html` (redirect to the package
page).

CI checks freshness: the `pdoc-freshness` job in
`.github/workflows/ci.yml` regenerates the reference into a temp
dir and diffs it against the committed `docs/api/` -- a stale
reference is a red CI.  When you add a public module, add it to the
regen command in the job AND here, then regenerate locally and
commit.
