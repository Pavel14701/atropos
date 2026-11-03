# AGENTS.md — Универсальные инструкции агента
# Atropos — Hypothesis-Killing Framework — Universal Agent Instructions

> **Context beacon.** This is the DEVELOPER context of Atropos
> (pytest/ruff/mypy/conventions). For USING the framework with an AI
> agent, see `atropos/agent/` (pip-installed) or run `atropos agent-context`.

Документ для любого ИИ-агента и любого аналитика: Claude, Cursor, Windsurf, Copilot, Codex, GPT-агенты, кастомные скрипты. Инструментонезависим: никакой специфики конкретного агента здесь нет. Перед использованием заполните плейсхолдеры `[PLACEHOLDER]` значениями проекта. Ключевые термины продублированы на английском.

---

## 1. Обзор проекта (Project Overview) — заполнить на проект

- Проект: `[PROJECT_NAME]`
- Цель проекта: `[PROJECT_GOAL]`
- Корень репозитория: `[PROJECT_ROOT]`
- Каталог данных: `[DATA_DIR]`; актуальный снимок: `[SNAPSHOT_DATE]`
- Владелец процесса (вопросы и эскалации): `[OWNER]`
- Главный документ методологии: `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` — прочитать до начала любой работы.

## 2. Перед выдвижением любой гипотезы (Before Proposing Any Hypothesis)

Обязательный протокол, строго по порядку:

1. **Graveyard check.** Поиск по `hypotheses/graveyard/` и `hypotheses/INDEX.csv`: проверялась ли эта или структурно идентичная гипотеза. Идентичность = совпадают предсказание (prediction) и kill-критерий, а не заголовок. `KILLED`-гипотеза не выдвигается повторно без новой механики.
2. **Прочитать пайплайн.** Раздел 3 ниже и `docs/HYPOTHESIS_KILLING_FRAMEWORK.md`.
3. **Prereg before data.** Пререгистрация (preregistration) пишется и коммитится ДО любого обращения к данным: сначала `hypotheses/preregistrations/H-NNNN.md`, потом выгрузка.

Результаты поиска кладбища фиксируются в пререгистрации: что искали, по каким тегам, что нашли. «Я помню, что не проверяли» — не основание.

## 3. Пайплайн из 7 стадий (The 7-Stage Pipeline)

| # | Стадия | Действие агента | Kill-критерий стадии |
|---|--------|-----------------|----------------------|
| 1 | Formulation | Сформулировать фальсифицируемое предсказание (falsifiable prediction) | Предсказание неопровержимо — kill на месте |
| 2 | Graveyard check | Поиск среди убитых и открытых (раздел 2) | Структурный дубликат `KILLED` — kill |
| 3 | Preregistration | Записать `H-NNNN.md`: prediction, числовой kill-критерий, snapshot, seed, теги | Нет измеримого kill-критерия — kill |
| 4 | Data | Загрузить зафиксированный снимок, сверить схему со словарём данных | Снимок не покрывает проверку — `INCONCLUSIVE`; подменять данные нельзя |
| 5 | Test | Реализовать тест `hypotheses/code/H-NNNN_*.py` + pytest-тест | Тест не проходит контрольный (sanity) случай — kill |
| 6 | Verdict | Сравнить метрику с заранее записанным порогом; учесть multiple comparisons | Метрика хуже порога — `KILLED` |
| 7 | Tombstone | Записать `graveyard/H-NNNN.md` и строку в `INDEX.csv` | Без надгробия гипотеза считается открытой (закон 6) |

Исходы (outcomes): `KILLED` — опровергнута; `SURVIVED` — выжила и только теперь может обосновывать решение; `INCONCLUSIVE` — не хватило данных, допустимо переоткрытие на новом снимке.

## 4. Шесть системных законов (The 6 Systematic Laws) — НЕ НАРУШАТЬ

1. **Preregister before data** — никакого обращения к данным до коммита пререгистрации.
2. **One hypothesis — one ID** — одна гипотеза = один `H-NNNN` = prereg + code + tombstone; дубликаты ID запрещены.
3. **Reproducibility** — фиксированный seed, зафиксированный snapshot, зафиксированные зависимости; результат обязан воспроизводиться командой из пререгистрации.
4. **Kill criteria upfront** — числовой порог опровержения записан до теста и после не меняется.
5. **No post-hoc edits** — после первого обращения к данным пререгистрация не редактируется; смена замысла = новая гипотеза.
6. **Every hypothesis gets a verdict** — каждая `H-NNNN` завершается tombstone и строкой в `INDEX.csv`, включая собственные неудобные гипотезы.

## 5. Конвенции кода (Code Conventions)

- Язык: Python 3.11+; менеджер пакетов и окружение — **uv** (`uv sync`, команды через `uv run`); версии зафиксированы в `pyproject.toml` + `uv.lock`.
- Стиль: `ruff check` + `ruff format` — жёсткий набор правил, 78 колонок: обязательные докстринги (pydocstyle), ускорение конструкций (perflint), запреты неоднозначных имён (E741). Ноль предупреждений перед коммитом.
- Типы: `mypy` в режиме **strict** — весь код и тесты; ошибки типов не подавляются без точного кода в ignore.
- Имена: однобуквенные идентификаторы запрещены везде (кроме выброса `_`); проверяется AST-тестом конвенций наряду с ruff.
- Тесты: `pytest`; на каждый тест гипотезы — `tests/test_h_NNNN.py`; код гипотезы без теста не мержится.
- Тесты не ходят в сеть и не зависят от внешнего времени — только снимки `[DATA_DIR]`.
- Данные: `[DATA_DIR]/snapshots/` неизменяемы (immutable); новая выгрузка = новый снимок, правка старого запрещена.
- Именование: `hypotheses/code/H-NNNN_<slug>.py`; вывод метрик в stdout строго в формате `H-NNNN: metric=[VALUE] threshold=[THRESHOLD] verdict=[KILLED|SURVIVED|INCONCLUSIVE]`.

## 6. Индекс ключевых файлов (Key Files Index) — заполнить на проект

| Файл | Назначение |
|------|------------|
| `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` | Полная методология: стадии, законы, форматы |
| `hypotheses/preregistrations/H-NNNN.md` | Пререгистрации — пишутся до данных |
| `hypotheses/code/H-NNNN_*.py` | Код тестов гипотез |
| `hypotheses/graveyard/H-NNNN.md` | Надгробия (tombstones) — append-only |
| `hypotheses/INDEX.csv` | Реестр всех гипотез |
| `[DATA_DIR]/snapshots/[SNAPSHOT_DATE]/` | Снимки данных |
| `[DATA_DIR]/DATA_DICTIONARY.md` | Словарь данных |
| `[PROJECT_SPECIFIC_FILE]` | `[PURPOSE]` — допишите строки своего проекта |

## 7. Чего не делать (What NOT to Do) — 10 запретов

1. Не смотреть в данные до коммита пререгистрации.
2. Не менять kill-критерии после первого обращения к данным.
3. Не использовать один снимок данных и для рождения гипотезы, и для её проверки (data leakage).
4. Не игнорировать множественные сравнения (multiple comparisons) в семействах тестов.
5. Не объявлять `SURVIVED` без заранее записанного числового порога.
6. Не редактировать и не удалять надгробия — кладбище append-only.
7. Не запускать тесты без зафиксированного seed.
8. Не пропускать graveyard check «по памяти».
9. Не держать гипотезу открытой дольше `[MAX_OPEN_DAYS]` дней — вердикт обязателен.
10. Не подстраивать код, пороги или формулировки под желаемый результат (result shopping).

## 8. Когда остановиться и спросить (When to Stop and Ask)

- Kill-критерий не сводится к числу — обсудить формулировку с `[OWNER]`, не изобретать на ходу.
- Нужны данные вне `[DATA_DIR]` — запросить доступ, не обходить ограничение.
- Найден неоднозначный кандидат в дубликаты — спросить до траты ресурсов на тест.

При конфликте скорости и дисциплины побеждает дисциплина.

---

## Приложение A. Чек-лист новой гипотезы (New Hypothesis Checklist)

- [ ] Graveyard check выполнен, результаты вписаны в пререгистрацию
- [ ] `H-NNNN` свободен — ID следующий по порядку
- [ ] Пререгистрация заполнена: prediction, kill-критерий, snapshot, seed, теги
- [ ] Пререгистрация закоммичена ДО выгрузки данных
- [ ] Тест покрыт pytest-тестом, `ruff check` чист
- [ ] Вердикт записан: tombstone + строка в `INDEX.csv`

## Приложение B. Формат строки INDEX.csv (Registry Row Format)

```csv
h_id,title,tags,prereg_date,snapshot,verdict,verdict_date,tombstone_path
H-0001,Example title,tag1|tag2,2026-01-10,2026-01-01,KILLED,2026-01-12,graveyard/H-0001.md
```

## Приложение C. Шаблон пререгистрации (Preregistration Template)

```markdown
# H-[NNNN]: [TITLE]
- Автор: [AUTHOR], дата: [DATE], теги: [TAGS]
- Prediction: [MEASURABLE_FALSIFIABLE_PREDICTION]
- Kill criteria: [METRIC] < [THRESHOLD] на снимке [SNAPSHOT_DATE], seed [SEED]
- Graveyard check: [WHAT_WAS_SEARCHED], результат: [FOUND_NOTHING | DUPLICATE_OF_H-XXXX]
```

Приоритет документов: законы раздела 4 > запреты раздела 7 > шаблоны приложений > удобство. При противоречии — стоп и вопрос `[OWNER]`.
