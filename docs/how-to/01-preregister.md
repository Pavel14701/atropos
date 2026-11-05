# How-to 01. Пререгистрация гипотезы: от params до PREREG-файла

> **Estimated time:** 10 минут
>
> **You will need:** репозиторий atropos с настроенным окружением
> (`uv sync` выполнен), Python 3.11+. Доступ к данным для этого how-to
> **не нужен** — и до конца шага 4 **запрещён** (закон 1: preregister
> before data).

## Зачем

Пререгистрация — это заморозка protocol ДО данных: предсказание,
конструкция, нули и kill-критерии записываются до первого обращения к
данным, и коммит файла становится хешем протокола. Инструмент —
`atropos.prereg_generator`: он не даст зарегистрировать «мягкий»
kill-критерий без оператора и числа.

Самодостаточно: читать `FRAMEWORK.md` перед этим how-to не нужно.

Импорт (пакет `atropos` не реэкспортирует функции наверх — используйте
пути модулей):

```python
from atropos.prereg_generator import generate_prereg, validate_prereg
```

## Шаг 1. Собрать `params` — 7 обязательных полей

| Поле | Тип | Что это |
|------|-----|---------|
| `name` | `str` | короткое имя гипотезы (slug, попадёт в заголовок) |
| `hypothesis` | `str` | предсказание одной фразой: что, знак, горизонт |
| `construction` | `str` или `list[str]` | как строится сигнал: пороги, окна, нормализация |
| `nulls` | `str` или `list[str]` | нулевые объяснения эффекта (repackaging, концентрация, …) |
| `kill_criteria` | непустой `list` из `str`/`dict` | числовые пороги убийства — см. ниже |
| `power_calc` | `str`, `list[str]` или `dict` | расчёт мощности: N, alpha, power |
| `hash` | `str` | хеш замороженной спецификации |

Kill-критерий — жёсткое направленное правило, двумя способами:

- **строка**: оператор + число внутри текста, напр.
  `"IC < 0.05 on 500 obs"` или `"share of PnL from best subperiod > 0.6"`;
- **dict**: `{"criterion": str, "threshold": число, "comparison": one of
  < <= > >= == ~}` — `comparison` **обязателен**, дефолта нет.

Голое число в прозе («wait 200 days») не замораживает ничего и будет
отклонено. Оператор `~` допускается только как явный
диагностический/приближённый маркер; для жёстких убийств используйте
направленные операторы.

## Шаг 2. Прогнать `validate_prereg` до пустого списка

Функция возвращает список проблем; `[]` = валидно. Никогда не переходите
к генерации, пока список не пуст — `generate_prereg` на невалидных
params бросает `ValueError`.

### Валидный пример (фактический вывод)

```python
from atropos.prereg_generator import validate_prereg

params = {
    "name": "overnight_imbalance_edge",
    "hypothesis": (
        "Overnight order-flow imbalance predicts the next-day return "
        "incrementally to 7d and 30d momentum."
    ),
    "construction": [
        "signal = zscore(overnight_order_flow_imbalance)",
        "enter at next open, exit at next close",
    ],
    "nulls": [
        "the edge is repackaged 7d/30d momentum",
        "the edge is concentrated in a single calendar year",
    ],
    "kill_criteria": [
        {
            "criterion": "|t| of the new signal after mom7 + mom30 controls",
            "comparison": "<",
            "threshold": 2.0,
        },
        "share of total PnL from the best subperiod > 0.6",
    ],
    "power_calc": {"N_required": 2600, "alpha": 0.05, "power": 0.80},
    "hash": "a1b2c3d4",
}

problems = validate_prereg(params)
print(problems)
assert problems == []
```

Фактический вывод:

```text
[]
```

### Невалидный пример 1 — проза вместо критерия (фактический вывод)

```python
bad_wait = {
    "name": "x", "hypothesis": "h", "construction": "c", "nulls": "n",
    "power_calc": "p", "hash": "hh",
    "kill_criteria": ["wait 200 days before final check"],
}
print(validate_prereg(bad_wait))
```

Фактический вывод:

```text
["kill_criteria[0]: no comparison operator bound to a numeric threshold found in 'wait 200 days before final check' (e.g. 'IC < 0.05')"]
```

«Подождём ещё 200 дней» — это не kill-критерий: нет оператора,
связанного с числом, значит порог можно двигать после данных.

### Невалидный пример 2 — dict без `comparison` (фактический вывод)

```python
bad_dict = dict(bad_wait)
bad_dict["kill_criteria"] = [
    {"criterion": "t-stat after controls", "threshold": 2.0},
]
print(validate_prereg(bad_dict))
```

Фактический вывод:

```text
["kill_criteria[0]: missing required 'comparison' (one of ['<', '<=', '==', '>', '>=', '~'])"]
```

Порог без направления — мягкий выключатель: непонятно, убивает `2.0`
при `t < 2` или при `t > 2`. Поэтому `comparison` обязателен и не
подставляется по умолчанию.

## Шаг 3. Сгенерировать документ

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

# ... params как в шаге 2 ...

assert validate_prereg(params) == []
doc = generate_prereg(params, date="2026-11-03")
print(doc)
```

Фактический вывод (сокращённо; полный документ — на вашем прогоне):

```markdown
# Preregistration: overnight_imbalance_edge

- **Frozen on:** 2026-11-03
- **Spec hash:** `a1b2c3d4`

## Hypothesis

Overnight order-flow imbalance predicts the next-day return
incrementally to 7d and 30d momentum.

## Construction

- signal = zscore(overnight_order_flow_imbalance)
- enter at next open, exit at next close

## Null hypotheses

**H1:** the edge is repackaged 7d/30d momentum
**H2:** the edge is concentrated in a single calendar year

## Kill criteria

The hypothesis is KILLED if any of the following holds, measured on the
frozen construction with frozen thresholds:

| # | Criterion | Rule | Threshold |
|---|-----------|------|-----------|
| 1 | \|t\| of the new signal after mom7 + mom30 controls | < | 2.0 |
| 2 | share of total PnL from the best subperiod > 0.6 | > | 0.6 |

## Power analysis

- **N_required:** 2600
- **alpha:** 0.05
- **power:** 0.8

## Attestation

This specification (construction, nulls, kill criteria, power) was
frozen before any outcome data was evaluated against it. Post-hoc
changes invalidate the preregistration; any deviation must be logged
as a separate amendment with a new hash.
```

Обратите внимание на строку 2 таблицы kill-критериев: для строковой
формы генератор сам вытаскивает оператор и число («>» и «0.6»), а не
подставляет заглушку. Дата: `date=None` ставит сегодняшний день;
явная дата должна быть ISO (`YYYY-MM-DD`), иначе `ValueError`.

## Шаг 4. Сохранить `PREREG_<name>_<date>.md` и закоммитить ДО данных

```python
from pathlib import Path

out = Path("PREREG_overnight_imbalance_edge_2026-11-03.md")
out.write_text(doc, encoding="utf-8")
print("saved:", out.resolve())
```

Пример вывода (путь зависит от вашей рабочей директории):

```text
saved: <workspace>/PREREG_overnight_imbalance_edge_2026-11-03.md
```

Если workspace ещё не создан (нет `GRAVEYARD.md` и каталога
`hypotheses/`), сгенерируйте скелет командой `atropos init` — команда
создаёт только отсутствующие файлы и никогда не перезаписывает
существующие; дальше редактируйте их вручную.

Файл положите в каталог пререгистраций вашего проекта (в раскладке
методологии — `hypotheses/preregistrations/`) и **закоммитьте до любой
выгрузки данных**. Коммит = хеш протокола: правка после первого
обращения к данным аннулирует тест; смена замысла = новая гипотеза с
новым ID, а не правка этой.

## Шаг 5. Чек-лист законов (перед тем как открыть данные)

- [ ] Graveyard check сделан, результаты поиска вписаны в prereg
      (см. how-to 03).
- [ ] Каждый kill-критерий = оператор + число (строка) или
      `{criterion, threshold, comparison}` (dict); `validate_prereg`
      вернул `[]`.
- [ ] `comparison` указан явно в каждом dict-критерии.
- [ ] Snapshot данных и seed зафиксированы (закон 3:
      воспроизводимость).
- [ ] Файл `PREREG_<name>_<date>.md` закоммичен ДО выгрузки данных
      (закон 1).
- [ ] Пороги больше не меняются (закон 4: kill criteria upfront;
      закон 5: no post-hoc edits).

## Что дальше

- Как читать результат проверки — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Куда записывается вердикт — how-to 03 (`docs/how-to/03-read-graveyard.md`).
- Справочник функций — `docs/api/atropos/prereg_generator.html`.
