---
name: 04-run-ml-kill-protocol
description: Прогнать ML-гипотезу (features→target) через atropos.ml_training.run_ml_kill_protocol и прочитать вердикт SURVIVES / DEAD(<gate>) / UNDERPOWERED.
when_to_use: for ML hypotheses: features→target, after prereg
version: 0.1.0
---

# SKILL 04: RUN ML KILL PROTOCOL — walk-forward ML-цепочка

> Core-скилл агента Atropos. Публичный API:
> `atropos.ml_training.run_ml_kill_protocol(features, target,
> exec_returns, times_us, *, config=None, battery_controls=None)
> -> dict`. Expanding-block walk-forward CV с purge + embargo;
> каждая модель из открытого `MODEL_REGISTRY` (8 встроенных +
> `"ensemble"` после `import atropos.ml_ensemble`) идёт через один
> путь fit → OOS-вероятности → порог → EV и четыре гейта.

## КОГДА ML-ЦЕПОЧКА ВМЕСТО RULE-BASED

Rule-цепочка (`02-run-pipeline`) проверяет ПРАВИЛО: signal → outcome.
ML-цепочка — когда гипотеза формулируется как КЛАССИФИКАТОР:
признаки → бинарное событие (`target` 0/1), решение — вероятностный
порог с EV взятых сделок. Дисциплина не меняется: graveyard check и
prereg до данных — те же скиллы (`03`, `01`); меняется только движок
прогона. ML-цепочка НЕ заменяет rule-цепочку «поверх»: это два разных
прогона одной гипотезы, каждый со своим вердиктом.

## PROCEDURE (порядок)

1. **Prereg до данных.** `from atropos import generate_prereg`;
   kill-критерии — гейты движка числами (profit_bar, DSR-минимум,
   event-floor), заморожены до выгрузки данных.
2. **Graveyard check.** Skill `03-read-graveyard`: поиск по
   prediction + kill-критерию, не по заголовку.
3. **Собрать датасет** (контракт caller-defined): `features`
   (DataFrame, хронологические строки), `target` (0/1 int),
   `exec_returns` (исполнимые per-event returns; издержки УЖЕ вычтены
   вызывающим), `times_us` (int64 микросекунды, monotone) — по строке
   на событие. `battery_controls` — колонки известных линейных
   контролов для linear-battery компаратора incremental-гейта.
4. **Один прогон.** `config` мерджится поверх
   `DEFAULT_TRAINING_CONFIG` (key-wise); пороги/гейты из пререга.
   Несколько моделей (`models=[...]`) = один CV-проход и СРАВНЕНИЕ в
   `report["models"]` — движок никогда не выбирает победителя.
5. **Прочитать вердикт.** `report["verdict"]` — один из трёх ниже;
   per-model детали в `report["models"][name]`.

```python
from atropos.ml_training import run_ml_kill_protocol

report = run_ml_kill_protocol(
    features, target, exec_returns, times_us,       # датасет из шага 3
    config={"profit_bar": 0.004},                   # поверх DEFAULT
    battery_controls=["ctrl1", "ctrl2"],            # известные контроли
)
print(report["verdict"], report["n_events"], sorted(report["models"]))
```

## ГЕЙТЫ (4) — ЧТО КАЖДЫЙ ПРОВЕРЯЕТ

| Гейт | Проверка |
|---|---|
| `incremental` | EV взятых сделок модели бьёт linear-battery logit (paired bootstrap t >= incremental_t) |
| `cost_line` | merged-OOS EV взятых сделок >= `profit_bar` |
| `importance_stability` | per-fold перестановочные важности ранжируются согласованно (медианный pairwise Spearman >= `importance_rho_min`) |
| `deflated_sharpe` | deflated Sharpe (Bailey & Lopez de Prado 2014) дневного EV-потока >= `dsr_min` |

## ВЕРДИКТЫ (3) — ЧТО ДЕЛАТЬ ДЛЯ КАЖДОГО

| Вердикт | Смысл | Действие агента |
|---|---|---|
| `SURVIVES` | все включённые гейты пройдены, событий достаточно | tombstone `SURVIVED`: условный кандидат с гейтами, не leg системы |
| `DEAD(<gate>)` | первый сработавший гейт первой невыжившей модели | tombstone `KILLED`, расшифровать гейт из скобок и записать числа из `report["kills"]` |
| `UNDERPOWERED` | меньше `n_min_events` declustered событий | НЕ смерть: гипотеза нерешённая (undecided). tombstone `INCONCLUSIVE` с планом мощности (больше данных / шире снимок); повторный прогон на том же снимке запрещён |

## ЛОВУШКИ

- Не подгонять `threshold_grid` (и любые пороги/гейты) после взгляда
  на результат — это result shopping; пороги замораживаются в прeregе.
- `battery_controls` = известные ЛИНЕЙНЫЕ контроли из пререга; без
  них incremental-гейт честно inconclusive (не kill, но и не
  доказательство инкрементальности).
- Ensemble включается side-effect-импортом:
  `import atropos.ml_ensemble` регистрирует `"ensemble"` в
  `MODEL_REGISTRY` и его дефолты; до импорта модели в реестре нет.
- UNDERPOWERED не «оживляет» гипотезу и не разрешает перезапуск с
  новыми порогами: новое решение = новый prereg (см. `01`).

## ДАЛЬШЕ

Tombstone + строка в `INDEX.csv` — как в `02-run-pipeline.md`.
Нужен один вердикт по rule+ML цепочкам вместе —
`atropos.kill_chain.run_full_kill_chain` (тонкий агрегат: kill любой
цепочки = KILLED; UNDERPOWERED без смерти = INCONCLUSIVE).
