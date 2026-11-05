# docs/api — авто-генерируемая документация

Содержимое этого каталога (кроме данного README) — **авто-генерат**
pdoc по докстрингам пакета `atropos`; вручную его не редактируют и не
правят. Регенерация — из корня репозитория командой ниже (pdoc
подключается эфемерным окружением `--with pdoc`, зависимости проекта
не меняются). Модули перечислены явно: голое `pdoc atropos` рендерит
только страницу пакета (`atropos.html`) — подмодули в обход `__all__`
pdoc 16 сам не подхватывает.

```bash
uv run --with pdoc python -m pdoc atropos atropos._utils atropos.bootstrap_ci atropos.cli atropos.cost_model atropos.cross_sectional atropos.ensemble_pipeline atropos.evaluation_metrics atropos.incremental_regression atropos.kill_chain atropos.ml_ensemble atropos.ml_training atropos.monte_carlo atropos.null_control_suite atropos.power_calculator atropos.predictive_model_eval atropos.prereg_generator atropos.robustness_battery atropos.track_record atropos.walkforward_framework atropos.workspace -o docs/api
```

Команда перечисляет все 20 модулей пакета. Новый модуль `atropos/`
добавляется и в этот список (иначе его страницы в `docs/api` не
появятся и не обновятся).

Точка входа — `docs/api/index.html` (редирект на страницу пакета
`atropos.html`); страницы модулей лежат в `docs/api/atropos/`.
