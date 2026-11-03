# docs/api — авто-генерируемая документация

Содержимое этого каталога (кроме данного README) — **авто-генерат**
pdoc по докстрингам пакета `atropos`; вручную его не редактируют и не
правят. Регенерация — из корня репозитория командой ниже (pdoc
подключается эфемерным окружением `--with pdoc`, зависимости проекта не
меняются; модули перечислены явно, потому что `atropos/__init__.py`
экспортирует только `__version__`):

```bash
uv run --with pdoc python -m pdoc atropos atropos._utils atropos.bootstrap_ci atropos.cost_model atropos.cross_sectional atropos.ensemble_pipeline atropos.evaluation_metrics atropos.incremental_regression atropos.monte_carlo atropos.null_control_suite atropos.power_calculator atropos.predictive_model_eval atropos.prereg_generator atropos.robustness_battery atropos.track_record atropos.walkforward_framework -o docs/api
```

Точка входа — `docs/api/index.html` (редирект на страницу пакета
`atropos.html`); страницы модулей лежат в `docs/api/atropos/`.
