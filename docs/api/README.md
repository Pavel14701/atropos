# docs/api — auto-generated reference

Everything in this directory except this README is **generated** by
pdoc from the package docstrings; do not edit or review it by hand.
Regenerate from the repository root with the command below (pdoc runs
in an ephemeral `--with pdoc` environment; project dependencies are
unchanged). Modules are listed explicitly: a bare `pdoc atropos`
renders only the package page (verified on pdoc 16.0.0), so every
submodule is passed by name.

```bash
uv run --with pdoc python -m pdoc atropos atropos._utils atropos.bootstrap_ci atropos.cli atropos.cost_model atropos.cross_sectional atropos.ensemble_pipeline atropos.evaluation_metrics atropos.incremental_regression atropos.kill_chain atropos.ml_ensemble atropos.ml_training atropos.monte_carlo atropos.null_control_suite atropos.power_calculator atropos.predictive_model_eval atropos.prereg_generator atropos.robustness_battery atropos.track_record atropos.walkforward_framework atropos.workspace -o docs/api
```

The command lists all 20 package modules. A new `atropos/` module is
added to this list too — otherwise its pages never appear in or
update `docs/api`.

The entry point is `docs/api/index.html` (redirect to the package
page).
