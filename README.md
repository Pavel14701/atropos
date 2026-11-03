# ATROPOS
## Фреймворк убийства гипотез в крипто-кванте

> **Атропос** (Ἄτροπος) — мойра, перерезающая нить жизни. Каждая гипотеза
> входит в пайплайн живой и выходит ровно с одним вердиктом:
> `SURVIVED` · `KILLED` · `INCONCLUSIVE`.

> Проверь могилы → прогони pipeline → один прогон → один вердикт → graveyard или prereg.

## Пакет `atropos`

13 модулей (Python 3.11+, numpy/pandas/scipy), каждый — самодостаточный
инструмент убийства; демо: `python -m atropos.<модуль>`.

| Модуль | Что делает |
|---|---|
| `prereg_generator` | Пререгистрация из dict: валидация + замороженный markdown |
| `power_calculator` | Мощность теста, detectable effect, t-статистика partial R² |
| `evaluation_metrics` | Sharpe (bootstrap CI), max drawdown, Calmar, deflated Sharpe, profit factor |
| `bootstrap_ci` | Percentile / block / stationary bootstrap CI |
| `cost_model` | Издержки: net-доходности, turnover, breakeven bps, COST-DEATH тест |
| `null_control_suite` | 6 нулевых контролов: permutation, placebo, momentum, known features, subperiods, calendar |
| `incremental_regression` | Incremental test + specification curve (HC0/Newey-West) |
| `robustness_battery` | Батарея из 7 тестов: пороги, окна, subperiods, placebo lag, LOO, declustering, monotonicity |
| `walkforward_framework` | Walk-forward с антиподглядыванием (no-peeking) и purge/embargo на стыке train/test |
| `monte_carlo` | GBM-пути, block-bootstrap распределения Sharpe и maxDD |
| `predictive_model_eval` | Purged K-fold CV, walk-forward OOS, стабильность важностей; встроенные модели: OLS, kNN, Nadaraya-Watson + подключаемые фабрики (sklearn-обёртки без новой зависимости) |
| `cross_sectional` | Кросс-секция: IC по датам, квантильные long-short спреды net-of-cost, монотонность по квантилям |
| `ensemble_pipeline` | Весь конвейер одним вызовом: до 7 стадий, вердикт по первой же смерти |

## Инструментарий

- **uv** — менеджер пакетов и окружение: `uv sync`, все команды через `uv run`
- **ruff** — жёсткий линтинг: 78 колонок, докстринги (pydocstyle/google),
  запрет неоднозначных имён (E741), ускорение конструкций (PERF); полный
  набор — в `pyproject.toml`
- **mypy --strict** — строгая типизация всего кода и тестов
- **pytest** — тест на каждый модуль + AST-тест конвенций
  (`tests/test_conventions.py`: запрет однобуквенных идентификаторов,
  кроме `_`)
- **Запрет однобуквенных переменных** — ruff ловит `l/I/O`, AST-тест — все остальные

```bash
uv sync                                        # окружение
uv run ruff check . && uv run ruff format --check .   # стиль
uv run mypy                                    # типы (strict)
uv run pytest                                  # тесты
uv run python -m atropos.ensemble_pipeline     # демо полного конвейера
```

## Документы

| Файл | Что внутри |
|---|---|
| FRAMEWORK.md | Философия + Pipeline 7 этапов + 6 законов + таксономия смерти (18 терминов) |
| TEMPLATES.md | 5 шаблонов: prereg, скрин, батарея робастности, post-mortem, брифинг агента |
| CASES_AND_FLAGS.md | 8+ красных флагов + 5 case studies (глубокий разбор) + чек-лист агента (7 пунктов) |
| METHODOLOGY.md | Разворачивание фреймворка с нуля в любом проекте |
| skills/ | 6 исполняемых протоколов-скиллов пайплайна |

## Правило

**Любая гипотеза** → открой FRAMEWORK.md → проверь graveyard → прогони pipeline (до 7 стадий: контролы → спек-кривая → батарея → walk-forward → risk/cost-конверт net-of-cost → модельный OOS → кросс-секция) → один прогон → один вердикт → graveyard или prereg. Ноль параметров после данных. Ноль календарей в скоринге. Ноль «может быть пригодится». Вердикты включают `COST_DEATH` (edge съеден издержками: gross проходит, net — нет) и `CS_NO_EDGE` (кросс-секционный спред не выживает).

## Источник

deterministic-trading-engine repo · 16 раундов марафона · ~89 направлений · 130+ могил · 6 систематических законов · PREREG + graveyard + response doc
