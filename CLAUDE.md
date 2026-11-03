# CLAUDE.md — Инструкции для Claude
# Atropos — Hypothesis-Killing Framework — Claude-Specific Instructions

Claude-специализация `AGENTS.md` (Claude Code, ZCode и совместимые CLI-агенты). Базовые правила — в `AGENTS.md`; здесь только отличия и привязка правил к конкретным инструментам (tools). Плейсхолдеры `[PLACEHOLDER]` заполнить значениями проекта.

---

## 1. Первое действие и обзор проекта (First Action and Overview)

До любой работы прочитать в этом порядке: `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` → `AGENTS.md` → этот файл. Если главный документ отсутствует — остановиться и сообщить: фреймворк не развёрнут (развёртывание описано в `METHODOLOGY.md`).

Значения проекта: имя `[PROJECT_NAME]`, данные `[DATA_DIR]`, снимок `[SNAPSHOT_DATE]`, владелец `[OWNER]`, окружение `[VENV]`. Если этот файл расходится с `AGENTS.md` в инструментах — приоритет у этого файла; в процессе — приоритет у законов `AGENTS.md`.

## 2. Перед выдвижением гипотезы — какие инструменты использовать

1. **Graveyard check.** Read `hypotheses/INDEX.csv`; затем Grep по `hypotheses/graveyard/` по ключевым словам и тегам будущей гипотезы (не по заголовкам). Результаты поиска вписать в пререгистрацию.
2. **Прочитать пайплайн.** Read `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` — таблица 7 стадий и kill-критериев.
3. **Prereg before data.** Write `hypotheses/preregistrations/H-NNNN.md` → закоммитить через Bash → и только затем любая выгрузка или чтение данных.

## 3. Пайплайн и исходы (Pipeline and Outcomes)

Стадии: `Formulation → Graveyard check → Preregistration → Data → Test → Verdict → Tombstone`. Стадии не пропускаются и не переупорядочиваются. Полная таблица kill-критериев — `AGENTS.md`, раздел 3.

Обработка исходов:
- `KILLED` — Write tombstone + строка в `INDEX.csv`; короткое резюме пользователю: что убито и почему.
- `SURVIVED` — tombstone со списком всех проверок; решение о применении принимает пользователь, не агент.
- `INCONCLUSIVE` — tombstone с указанием, каких данных не хватило; переоткрытие допустимо на новом снимке.

## 4. Шесть законов — не нарушать (The 6 Laws)

1. Preregister before data. 2. One hypothesis — one ID. 3. Reproducibility (seed + snapshot + requirements). 4. Kill criteria upfront. 5. No post-hoc edits. 6. Every hypothesis gets a verdict. Полные формулировки — `AGENTS.md`, раздел 4.

## 5. Инструменты и конвенции (Tools and Conventions)

- Создание и правка файлов — только Write/Edit; не собирать файлы через heredoc в Bash.
- Чтение — Read (не `cat`, `head`, `tail`); поиск — Grep/Glob: сначала поиск, затем точечное чтение.
- Все команды — через uv в окружении проекта (`uv sync`, `uv run <cmd>`); глобальный интерпретатор не используется.
- Тесты: Bash `uv run pytest -q` перед каждым коммитом; код гипотезы без теста не мержится.
- Стиль: Bash `uv run ruff check .` и `uv run ruff format .`; 78 колонок, докстринги, perflint, E741; конфигурация в `pyproject.toml`; ноль предупреждений.
- Типы: Bash `uv run mypy` — режим strict, весь код и тесты; однобуквенные идентификаторы запрещены (AST-тест `tests/test_conventions.py`, кроме `_`).
- Запуск теста гипотезы: Bash `python hypotheses/code/H-NNNN_<slug>.py`; вывод в stdout строго: `H-NNNN: metric=[VALUE] threshold=[THRESHOLD] verdict=[KILLED|SURVIVED|INCONCLUSIVE]`.
- Падает pytest — чинить код теста; ослаблять kill-критерий запрещено (закон 4).
- Снимки `[DATA_DIR]/snapshots/` неизменяемы: новая выгрузка = новый снимок, перезапись запрещена.
- Коммиты — только по явному запросу пользователя; без запроса не работать в `main`.

Типовая последовательность одной сессии (standard session):

```bash
uv run pytest -q                        # 1. база зелёная
uv run ruff check . && uv run ruff format --check .   # 2. стиль чист
uv run mypy                             # 3. типы (strict)
# 4. Write prereg → commit → выгрузка данных → запуск теста → Write tombstone → строка INDEX.csv → commit
```

## 6. Ключевые файлы (Key Files)

| Файл | Роль |
|------|------|
| `docs/HYPOTHESIS_KILLING_FRAMEWORK.md` | Методология фреймворка — читать первым |
| `AGENTS.md` | Базовые правила агента; этот файл — надстройка над ними |
| `hypotheses/preregistrations/H-NNNN.md` | Пререгистрации — Write до любых данных |
| `hypotheses/code/H-NNNN_*.py` | Код тестов гипотез |
| `hypotheses/graveyard/H-NNNN.md` | Надгробия — только Write, append-only |
| `hypotheses/INDEX.csv` | Реестр — строка на каждую гипотезу |
| `[DATA_DIR]/DATA_DICTIONARY.md` | Схема данных — источник правды до выгрузки |
| `[PROJECT_SPECIFIC_FILE]` | `[PURPOSE]` — допишите файлы своего проекта |

## 7. Claude-специфичные запреты (сверх 10 запретов из `AGENTS.md`)

1. Не использовать подагентов (subagents) для обхода стадий пайплайна; каждая стадия выполняется в основном контексте с видимыми артефактами.
2. Не править пререгистрацию через Edit после первого обращения к данным — только новая гипотеза (закон 5).
3. Не «улучшать» kill-критерии при переносе prereg в код — код следует prereg дословно.
4. Не открывать данные до prereg даже «для проверки схемы» — схема берётся из `DATA_DICTIONARY.md`.
5. Не запускать выгрузку данных при незакоммиченной пререгистрации.
6. Не завершать сессию с гипотезой без вердикта: либо tombstone, либо явный статус OPEN и план.

---

Приоритет: законы (`AGENTS.md`, раздел 4) > запреты (разделы 7 обоих файлов) > скорость и удобство. При конфликте — стоп и вопрос `[OWNER]`.
