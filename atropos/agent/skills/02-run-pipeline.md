---
name: 02-run-pipeline
description: Прогнать гипотезу через atropos.ensemble_pipeline.run_full_pipeline и правильно прочитать один из 9 вердиктов.
when_to_use: Когда пререг зафиксирован и данные выгружены; для любого решающего прогона по замороженной спеке.
version: 0.1.0
---

# SKILL 02: RUN PIPELINE — один прогон, один вердикт

> Core-скилл агента Atropos. Публичный API:
> `atropos.ensemble_pipeline.run_full_pipeline(data: dict,
> config: dict | None) -> dict`. Стадии идут в adversarial-порядке,
> первый FAIL убивает кандидата, остальные стадии — SKIPPED.

## PROCEDURE (порядок)

1. **Собрать data-dict.** Ключи: `returns` (forward-returns, серия),
   `signal` (тот же индекс), опционально `features` (DataFrame известных
   контролов), `mom7` / `mom30` (наблюдаемые momentum-факторы; иначе
   трейлинг-суммы), `hypothesis` (описание). Панельный режим:
   `panel_signal` / `panel_returns` (DataFrame: даты x активы) включают
   стадию 7; одни — panel-only (стадии 1-6 SKIPPED).
2. **Конфиг поверх дефолтов.** `config` мерджится поверх
   `DEFAULT_CONFIG`; заметные ключи: `cost.cost_bps` (односторонние
   издержки, bps, вычитаются на стадии 5), `model` (`"ols"` | `"knn"` |
   `"kernel"` | callable-factory, стадия 6).
3. **Один прогон.** Без подбора параметров после результата; seed
   зафиксирован внутри конфигов.
4. **Прочитать отчёт.** `overall_verdict`, `killed_by_stage`,
   `kill_reason`, `stages` (по-стадийные dict'ы), `runtime_seconds`.

## EXAMPLE (пример)

```python
from atropos.ensemble_pipeline import run_full_pipeline

report = run_full_pipeline(
    {"returns": fwd, "signal": sig, "features": controls,
     "hypothesis": "H-0007 mom30 echo"},
    {"cost": {"cost_bps": 10.0}},
)
print(report["overall_verdict"], report.get("kill_reason"))
```

## ТАБЛИЦА ВЕРДИКТОВ (9) — ЧТО ДЕЛАТЬ ДЛЯ КАЖДОГО

| Вердикт | Смысл | Действие агента |
|---|---|---|
| `SURVIVES` | все стадии прошли, предупреждений нет | tombstone `SURVIVED`. PASS ≠ подтверждение: выжил = условный кандидат. Для deployed-стратегии включить `track_record.live_verdict` |
| `SURVIVES_WITH_WARNINGS` | прошли, но с предупреждениями | tombstone `SURVIVED` с перечнем предупреждений из `stages` |
| `REPACKAGING` | сигнал — прокси известной фичи (стадии 1-2) | tombstone `KILLED`: эффект реален, но чужой; семья закрывается, преемники — с prior <= 20% и одним прогоном |
| `ARTIFACT` | артефакт: субпериодная концентрация / плацебо / календарь (стадия 1) | tombstone `KILLED`, записать механику артефакта |
| `DEAD` | батарея робастности (стадия 3) | tombstone `KILLED`, перечислить сломанные тесты |
| `NO_OOS_EDGE` | нет out-of-sample эджа: walk-forward / model-eval (стадии 4, 6) | tombstone `KILLED`; сжатие vs discovery — в надгробие |
| `COST_DEATH` | гросс жив, net-of-cost умирает (стадия 5) | tombstone `KILLED`; зафиксировать `cost_bps`, при котором умирает |
| `RISK` | пробой drawdown-хвоста или гросс-Sharpe-пола (стадия 5) | tombstone `KILLED` с риск-числами |
| `CS_NO_EDGE` | кросс-секционный IC / квантильный спред пуст (стадия 7) | tombstone `KILLED` |

`INCONCLUSIVE` на стадии (мало данных, вырожденные входы) — честный
исход: итог `INCONCLUSIVE`, переоткрытие допустимо только на новом
снимке данных.

## NEVER (запреты)

- Не перезапускать «с чуть другими параметрами» после вердикта —
  один прогон, один вердикт; пересмотр = новая гипотеза, новый ID.
- Не читать PASS как подтверждение эджа: выживший — условный
  кандидат с гейтами, не leg системы.
- Не пропускать стадии ручными скриптами «для скорости»: порядок
  стадий и есть защита.

## ДАЛЬШЕ

Вердикт записан → tombstone + строка в реестре (см. `01-preregister.md`
и `03-read-graveyard.md`). `SURVIVES` в production →
`atropos.track_record.live_verdict` (LIVE_OK / LIVE_KILLED /
INCONCLUSIVE; LIVE_OK — НЕ подтверждение эджа).
