# Atropos Agent Context — пакетный слой (default layer)

Как ПОЛЬЗОВАТЬСЯ фреймворком Atropos. Это пакетный слой: user-слой
(`%APPDATA%/atropos/agent` / `~/.config/atropos/agent`) и project-слой
(`./.atropos/agent`) могут заменять, дополнять или отключать эти файлы
и скиллы. Диагностика: `atropos agent-doctor`.

## Рабочий цикл гипотезы (5 шагов)

1. **Graveyard check.** До всего: поиск в `hypotheses/graveyard/` и
   `INDEX.csv` по предсказанию и kill-критерию (не по заголовку).
   `KILLED`-дубликат без новой механики не выдвигается. Скилл:
   `03-read-graveyard`.
2. **Preregister до данных.** Пререгистрация фиксируется ДО любого
   обращения к данным. Kill-критерии — только жёсткие правила
   «оператор + число». Скилл: `01-preregister`.
3. **Run pipeline.** Один прогон по замороженной спеке; первый FAIL
   решает всё. Скилл: `02-run-pipeline`.
4. **Interpret.** PASS ≠ подтверждение: `SURVIVES` = условный кандидат
   с гейтами, не leg системы. Один прогон — один вердикт.
5. **Record tombstone.** Каждая гипотеза заканчивается надгробием и
   строкой в `INDEX.csv`, включая собственные неудобные вердикты.

## API (фактический)

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

# params-dict из 7 обязательных полей; сначала validate_prereg(params).
doc = generate_prereg(params)  # -> markdown-строка

from atropos.ensemble_pipeline import run_full_pipeline

# data: returns, signal, features, mom7/mom30, hypothesis,
# panel_signal/panel_returns; config: поверх DEFAULT_CONFIG.
report = run_full_pipeline(data, config)
report["overall_verdict"]  # SURVIVES | SURVIVES_WITH_WARNINGS |
# REPACKAGING | ARTIFACT | DEAD | NO_OOS_EDGE | COST_DEATH | RISK |
# CS_NO_EDGE (+ INCONCLUSIVE-стадии); killed_by_stage, kill_reason.

from atropos.track_record import live_verdict
# production-этап для deployed SURVIVES; LIVE_OK ≠ подтверждение.
```

## Скиллы (9)

Core: `01-preregister`, `02-run-pipeline` (таблица 9 вердиктов),
`03-read-graveyard`. Протоколы этапов 1-7: `11-premise-check`,
`12-null-control`, `13-signal-test`, `14-incremental-test` (решающий),
`15-robustness-battery`, `16-walkforward-calendar`. Порядок фиксирован:
1 → 2 → 3 → 4 → 5 → 6, каждый этап дешевле следующего. Список:
`atropos agent-context --list-skills`; текст: `--skill=NAME`.

## NEVER (не делать никогда)

- Не смотреть в данные до фиксации пререга — никакие «превью».
- Не менять kill-критерии после первого обращения к данным.
- Не перезапускать прогон с подправленными параметрами (result
  shopping); смена замысла = новая гипотеза, новый ID.
- Не читать PASS/SURVIVES как подтверждение эджа.
- Не выдвигать `KILLED`-гипотезу повторно без новой механики.
- Не держать гипотезу без вердикта: без tombstone она открыта.
- Не редактировать и не удалять надгробия: кладбище append-only.

## Переопределение этого слоя

- Файлы: `AGENTS.md`/`CLAUDE.md` в старшем слое заменяют; `X.append.md`
  дополняют; `X.disabled` отключает базу (append-ы остаются).
- Скиллы: `skills/{name}.md` заменяет/добавляет; `{name}.append.md`
  дополняет; `skills/.disabled` (список имён) и `skills/.disabled-all`
  отключают. Старший слой: project > user > package.
- Настройка: `atropos agent-setup --scope=project --dry-run`.
