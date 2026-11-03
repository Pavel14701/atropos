---
name: 01-preregister
description: Заморозить гипотезу пререгистрацией через atropos.prereg_generator ДО любого обращения к данным.
when_to_use: Перед первым look в данные по новой гипотезе; при переоткрытии похороненной семьи с новой механикой.
version: 0.1.0
---

# SKILL 01: PREREGISTER — заморозить гипотезу до данных

> Core-скилл агента Atropos. Публичный API:
> `atropos.prereg_generator.generate_prereg(params: dict) -> str` и
> `validate_prereg(params) -> list[str]`. Пустой список проблем = можно
> генерировать. Закон: prereg коммитится ДО выгрузки данных.

## PURPOSE (цель)

Превратить гипотезу в неизменяемый документ с числовым kill-критерием
до того, как данные получат шанс что-то подсказать. Пререг без
числового порога — не пререг, а намерение.

## PROCEDURE (порядок)

1. **Собрать params-dict.** Семь обязательных полей (`REQUIRED_FIELDS`):
   `name`, `hypothesis`, `construction` (str или список шагов),
   `nulls` (str или список), `kill_criteria` (список), `power_calc`
   (str / список / dict), `hash` (хэш замороженной спеки).
2. **Kill-критерии — только жёсткие направленные правила.** Форма dict:
   `{"criterion": ..., "threshold": <число>, "comparison": "<" | "<="
   | ">" | ">=" | "=="}`; форма str — оператор, привязанный к числу
   (`"incremental t < 2"`). Порог записан до теста и после не меняется.
3. **Валидировать.** `validate_prereg(params)` обязан вернуть `[]`.
   Любая проблема чинится в params, а не игнорируется.
4. **Сгенерировать и зафиксировать.** `generate_prereg(params)` —
   markdown-документ (дата опциональна, формат ISO). Документ
   коммитится/сохраняется ДО выгрузки данных.
5. **Переоткрытие.** Для похороненной семьи — новый датированный пререг,
   раскрывающий закрытую историю и новую механику; иначе это майнинг.

## EXAMPLE (пример)

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

params = {
    "name": "H-0007-mom30-echo",
    "hypothesis": "mom30 предсказывает forward-return на 5d",
    "construction": ["zscore(mom30)", "top-tercile long only"],
    "nulls": ["нет предсказательной силы vs shuffled-сигнал"],
    "kill_criteria": [
        {"criterion": "incremental t", "threshold": 2.0,
         "comparison": "<"},
        {"criterion": "placebo percentile", "threshold": 95.0,
         "comparison": "<"},
    ],
    "power_calc": "N=(2*sigma/EV)^2 ~= 600 наблюдений для t~2",
    "hash": "sha256:<hash-frozen-spec>",
}
assert validate_prereg(params) == []
doc = generate_prereg(params)  # markdown, закончен до данных
```

## NEVER (запреты)

- Не смотреть в данные до фиксации пререга — включая «просто глянуть
  на разброс».
- Не менять пороги/направления после первого прогона: смена замысла =
  новая гипотеза с новым ID.
- Не оставлять kill-критерий без числа: «слишком слабо» — не критерий.

## ДАЛЬШЕ

Пререг зафиксирован → `02-run-pipeline.md` (прогон и вердикт).
Условия воскрешения чужих семей — `03-read-graveyard.md`.
