# How-to 04. Добавление нового null-контроля в stage 1

> **Estimated time:** 45–60 минут (контроль + тесты + калибровка)
>
> **You will need:** репозиторий atropos с окружением (`uv sync`),
> права на правку `atropos/**` (в отличие от how-to 01–03), чистые
> `uv run ruff check .`, `uv run mypy` (strict) и `uv run pytest -q`.
> Это контрибьюторский how-to: он меняет код библиотеки.

## Что такое null-контроль и какой у него контракт

Stage 1 (`atropos.null_control_suite`) — шесть adversarial-контролей:
`shuffled_permutation`, `placebo_placebo`, `momentum_control`,
`known_features_control`, `subperiod_stability`, `calendar_control`.
Каждый возвращает словарь строго одного вида (схема, не код):

```text
{
    "stat": float,        # статистика контроля (t, корреляция, доля...)
    "p_value": float,     # p-value этой статистики
    "verdict": "PASS" | "FAIL" | "INCONCLUSIVE",
    # ...контроль-специфичные ключи (n, r2, HC0, ...)...
    "reason": str,        # опционально: почему вычислить не удалось
}
```

Семантика:

- **PASS** — гипотеза пережила этот контроль. Порог жёсткий:
  `PASS` требует `p_value < 0.05` строго (`p_value == 0.05` — FAIL).
- **FAIL** — контроль убил гипотезу.
- **INCONCLUSIVE** — контроль **невычислим**, чаще всего design
  rank-deficient (сигнал в точности коллинеарен контролю). Это не PASS
  и не FAIL: pseudo-inverse сфабриковал бы правдоподобный t-stat ровно
  в тех случаях, которые suite и должен ловить. Текущая конвенция
  `reason` — `"collinear with controls"`.

INCONCLUSIVE-семантика важна и для пустого выравнивания:
`atropos._utils.align_inputs` при пустом результате dropna возвращает
пустые массивы (не бросает исключение) — публичный контроль обязан
перевести это в честный INCONCLUSIVE, а не упасть в глубине регрессии
и не молча убить гипотезу на нуле наблюдений.

## Шаги добавления

1. **Контроль** — функция в `atropos/null_control_suite.py`, экспорт
   через `__all__` (алфавитный порядок).
2. **Регистрация в оркестраторе** — `atropos/ensemble_pipeline.py`,
   функция `_stage_1_null_controls`: импорт, вызов в словарь
   `results`, и три места, где зашита старая конфигурация:
   - строка summary жёстко делит на 6:
     `f"{len(results) - len(failed)}/6 controls passed"` — обновить
     делитель (или вычислять из `len(results)`);
   - `_KILL_REASONS[1]` перечисляет семейства контролей в прозе
     («alignment / placebo / concentration / calendar») — дополнить;
   - уточнение вердикта: если ваш контроль ловит переупаковку, его имя
     надо добавить в кортеж `("momentum_control", "known_features_control")`
     в `repackaging = ...` — иначе его FAIL даст `ARTIFACT`, а не
     `REPACKAGING`.
3. **Демо ассертит** — `run_all_examples()` в конце
   `null_control_suite.py` это layer-0 тест: посаженный эдж печатает
   PASS по всем контролям, а хвост демо это **ассертит**, плюс
   trash-кейсы. Ваш контроль обязан участвовать в обоих блоках.
4. **Юнит-тесты** — `tests/test_null_control_suite.py`: посаженный
   случай → PASS; trash-случай → FAIL или INCONCLUSIVE; граничные
   случаи (строгость порога, degenerate-вход).
5. **Калибровочный тест (FPR под нулём)** —
   `tests/test_calibration.py::test_null_controls_fail_fraction_under_pure_noise`:
   под чистым шумом контроль должен FAIL примерно в 95% сидов (PASS
   требует p < 0.05), окно ассерта `[0.85, 1.0]` на 150 сидах.
   Добавьте свой контроль в цикл и в словарь `fails`. Если ваш PASS-гейт
   устроен иначе (как у `subperiod_stability`, где гейт — знаковая
   консистентность + cap концентрации), обосновайте своё окно в
   докстринге теста и pinned измерением.

## Мини-пример: скелет контроля (прогнанный)

Контроль-пример `convexity_control`: убивает «эдж», который является
артефактом выпуклости — регрессия `outcome ~ const + signal + signal²`,
статистика — t линейного члена после квадратичного. Скелет показывает
все обязательные элементы контракта: helpers из `_utils`,
INCONCLUSIVE на degenerate-входе, строгий порог.

```python
"""Skeleton example control: does a linear edge survive its own square?

outcome ~ const + signal + signal^2: if the "edge" is a convexity
artifact (big |moves| both pay), the linear term dies after the square
enters the model.  Follows the null_control_suite contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from atropos._utils import align_inputs, ols_hc0

_ALPHA = 0.05


def convexity_control(outcome: Any, signal: Any) -> dict[str, Any]:
    out, sig = align_inputs(outcome, signal)
    design = np.column_stack([np.ones(out.size), sig, sig**2])
    res = ols_hc0(out, design)
    t_lin = float(res["t"][1])
    p_lin = float(res["p"][1])
    degenerate = bool(res["rank_deficient"]) or not np.isfinite(t_lin)
    result: dict[str, Any] = {
        "stat": t_lin,
        "p_value": p_lin,
        "verdict": (
            "INCONCLUSIVE" if degenerate
            else ("PASS" if p_lin < _ALPHA else "FAIL")
        ),
        "n": int(res["n"]),
        "r2": float(res["r2"]),
        "HC0": True,
    }
    if degenerate:
        result["reason"] = "collinear with controls"
    return result
```

Демо-блок к нему (посаженный эдж / trash / degenerate / калибровка) —
фактический вывод прогона:

```text
linear edge   : PASS stat=11.40
convexity only: FAIL stat=-1.26
degenerate    : INCONCLUSIVE reason=collinear with controls
null FAIL fraction: 97/100 = 0.97
skeleton demo OK
```

Что здесь видно:

- линейный эдж проходит (t=11.40 после квадратичного члена);
- чистая выпуклость умирает (линейный член не нужен: t=-1.26 → FAIL);
- нулевая дисперсия сигнала → rank-deficient design → INCONCLUSIVE с
  `reason`, а не сфабрикованный t;
- под чистым шумом FAIL в 97/100 сидов — калибровка ~0.95 держится.

## Мета-ловушки (проверяется CI)

- **Никаких приватных копий хелперов.** Выравнивание и OLS — только из
  `atropos._utils` (`align_inputs`, `ols_hc0`). Копия `_align` в
  модуле — это дрейф поведения между копиями: первый же рефактор
  выравнивания разъедет ваш контроль относительно остальных. Пустое
  выравнивание (см. выше) переводится в INCONCLUSIVE.
- **Демо ассертит, а не просто печатает.** `run_all_examples()` —
  layer-0 верификация: регрессия роняет демо, а не печатает тихо
  неверный вердикт. Ваш контроль добавляется и в печать, и в ассерты.
- **Докстринг-контракт** (`tests/test_docstring_contract.py`): каждая
  функция, экспортированная через `__all__`, обязана нести `>>>`
  пример в докстринге; доктесты реально исполняются; `Raises:
  ValueError` должен быть обеспечен реальным `raise`; `Args:` обязан
  совпадать с `inspect.signature`.
- **Имена**: однобуквенные идентификаторы запрещены везде, кроме `_` —
  проверяется AST-тестом (`tests/test_conventions.py`), ruff это не
  ловит.
- **Стиль и типы**: `ruff` (78 колонок, pydocstyle, perflint, E741),
  `mypy` strict — ноль предупреждений; тесты гипотезного кода не ходят
  в сеть и не зависят от времени.
- **Никаких lookahead-лагов**: лаги строятся через `shift` (NaN-голова
  падает в выравнивании), никогда `np.roll` — он заворачивает хвост в
  голову и протекает будущим в лаг.

## Чек-лист

- [ ] Контроль возвращает `{stat, p_value, verdict, ...}` с INCONCLUSIVE
      на невычислимом входе и `reason="collinear with controls"`.
- [ ] Порог жёсткий: PASS ⇔ `p_value < 0.05`.
- [ ] Хелперы из `atropos._utils`, локальных копий нет.
- [ ] Регистрация в `_stage_1_null_controls` + делитель summary +
      текст `_KILL_REASONS[1]` + (если ловит repackaging) кортеж
      уточнения вердикта.
- [ ] Демо печатает и **ассертит** новый контроль, trash-кейс включён.
- [ ] Юнит-тесты в `tests/test_null_control_suite.py`.
- [ ] FPR-калибровка в `tests/test_calibration.py` под нулём ~0.95.
- [ ] `uv run ruff check .`, `uv run mypy`, `uv run pytest -q` — чисто.

## Что дальше

- Как вердикты стадии 1 читаются в отчёте пайплайна — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Справочник по существующим контролям:
  `docs/api/atropos/null_control_suite.html`.
