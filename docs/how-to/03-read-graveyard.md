# How-to 03. Чтение и пополнение кладбища гипотез

> **Estimated time:** 10 минут
>
> **You will need:** ваш проект с развёрнутым фреймворком (раскладка
> `hypotheses/` по методологии) и хотя бы одна запись на кладбище.
> Код atropos не нужен — только git, grep и csv.

## Зачем

Закон 6 фреймворка: **каждая гипотеза получает вердикт**. Без
надгробия гипотеза считается открытой, а открытое накапливает
дубликаты. Кладбище — не музей неудач, а **продукт процесса**: это
накопленная база знаний о том, какие семейства гипотез уже закрыты и
при каких условиях закрытие можно пересматривать.

Важно понимать границу: репозиторий `atropos` — это инструменты
убийства (пайплайн, контроли, генератор prereg). Само кладбище —
**методологический артефакт вашего проекта**: он живёт в вашей
раскладке `hypotheses/`, а не в библиотеке.

## Раскладка каталога (по METHODOLOGY.md)

```text
[PROJECT_ROOT]/
├── hypotheses/
│   ├── preregistrations/      # H-NNNN: пререгистрации (до данных)
│   ├── code/                  # H-NNNN: код тестов
│   ├── graveyard/             # H-NNNN: надгробия (вердикты), append-only
│   └── INDEX.csv              # реестр всех гипотез
└── ...
```

Правило идентичности: одна гипотеза = один `H-NNNN` = минимум три
артефакта (prereg, code, tombstone).

## Структура надгробия (tombstone)

Обязательные поля (по протоколу кладбища фреймворка):

1. **ID и вердикт** — `H-NNNN` и `KILLED | SURVIVED | INCONCLUSIVE`,
   дата вердикта, автор.
2. **COD — cause of death** — стадия и тип смерти. Тип берётся из
   таксономии пайплайна atropos: `REPACKAGING`, `ARTIFACT`, `DEAD`,
   `NO_OOS_EDGE`, `COST_DEATH`, `RISK`, `CS_NO_EDGE` (или `SURVIVED` /
   `INCONCLUSIVE` для невзятых записей).
3. **Prediction** — что предсказывалось (фальсифицируемое, из prereg).
4. **Минимальные цифры** — фактическое значение метрики против порога
   (raw t; incremental t, если дошли; проваленный тест), снимок данных,
   seed.
5. **Причина смерти одной строкой** — чтение на будущее.
6. **Resurrection conditions** — какие новые данные или условия
   сделали бы пересмотр возможным.
7. **Не пробовать снова** — какой запрет нарушил бы повторный заход.

Пример записи (из демо-раскладки этого how-to):

```markdown
# H-0002 — KILLED (COST_DEATH)
- Гипотеза: cross-sectional imbalance spread на топ-20 символов
- Prediction: net-of-cost SR p5 > 0 при 10bps round-trip
- COD: stage 5, monte-carlo — net SR p5=-0.52 при breakeven 8bps
- Вердикт: KILLED, 2026-03-02, снимок 2026-02-01, seed 0
- Resurrection conditions: издержки < 8bps one-way (maker-исполнение)
  ИЛИ горизонт удержания >= 5 дней на том же сигнале
- Не пробовать снова: тот же сигнал с taker-исполнением
```

Строка реестра `INDEX.csv`:

```csv
h_id,title,tags,prereg_date,snapshot,verdict,verdict_date,tombstone_path
H-0002,xs-imbalance-spread,cross-sectional|costs|orderflow,2026-02-20,2026-02-01,KILLED,2026-03-02,graveyard/H-0002.md
```

## Как искать перед новой гипотезой (graveyard check)

Стадия 2 пайплайна, обязательная до пререгистрации. Идентичность
гипотез — это совпадение **prediction + kill-критерия**, а не
заголовка: одна и та же идея под новым названием — дубликат, `KILLED`
не выдвигается повторно без новой механики. Ищите тремя способами:

1. **По тегам** — теги обязательны в prereg, ищите по ним в реестре:

```bash
grep -i "orderflow" hypotheses/INDEX.csv
```

Фактический вывод на демо-раскладке этого how-to (3 надгробия:
H-0001 orderflow/daily, H-0002 cross-sectional/costs, H-0003 funding):

```text
H-0001,overnight-ofi-edge,orderflow|momentum|daily,2026-02-01,2026-01-15,KILLED,2026-02-10,graveyard/H-0001.md
H-0002,xs-imbalance-spread,cross-sectional|costs|orderflow,2026-02-20,2026-02-01,KILLED,2026-03-02,graveyard/H-0002.md
```

2. **По COD / семейству смерти** — какие конструкции уже умирали так же:

```bash
grep -il "momentum_control\|REPACKAGING" hypotheses/graveyard/*.md
```

Фактический вывод:

```text
hypotheses/graveyard/H-0001.md
```

3. **Чтение найденного** — у кандидата в дубликаты сверяете prediction
   и kill-критерий:

```bash
sed -n '1,4p' hypotheses/graveyard/H-0001.md
```

Фактический вывод:

```text
# H-0001 — KILLED (REPACKAGING)
- Гипотеза: overnight order-flow imbalance предсказывает next-day return
- Prediction: |t| >= 2 после контролов mom7/mom30 на 500 obs
- COD (cause of death): stage 1, momentum_control — t=0.73 после mom7+mom30
```

Результаты поиска фиксируются в новой пререгистрации: что искали, по
каким тегам, что нашли («FOUND_NOTHING» или «DUPLICATE_OF_H-XXXX»).
«Я помню, что не проверяли» — не основание.

## Как читать resurrection conditions

Воскрешение — не «появился новый аргумент», а «изменились данные или
условия, записанные в надгробии ДО этого момента». Практические
правила:

- **Новый снимок данных ≠ новый аргумент.** Снимок легитимен, только
  если tombstone прямо называет его класс в resurrection conditions
  (например: «>= 180 дней funding-истории» или «полные глубины
  стакана»). Тот же снимок с той же конструкцией = та же мёртвая
  гипотеза.
- **Новое условие исполнения** (издержки, горизонт) легитимно, только
  если оно в списке resurrection conditions (как «maker-исполнение» в
  H-0002). Иначе это result shopping — подгонка под желаемый исход.
- **Изменённая конструкция = новая гипотеза.** Та же идея с другим
  окном/порогом/нормализацией не «воскрешает» старую: это новый
  `H-NNNN`, новая пререгистрация, пайплайн со стадии 1. Повторная
  смерть по той же причине — ошибка процесса, а не гипотезы, и
  фиксируется отдельно.
- **INCONCLUSIVE-записи** — не могилы: гипотеза открыта, переоткрытие
  на новом снимке допустимо по её собственному resurrection-условию
  («набрать недостающие данные»).

## Как добавить своё надгробие (append-only)

1. Напишите `hypotheses/graveyard/H-NNNN.md` по структуре выше.
   `NNNN` — следующий свободный номер.
2. Добавьте строку в `hypotheses/INDEX.csv` (формат — выше).
3. Никогда не редактируйте и не удаляйте существующие надгробия —
   кладбище append-only. Ошибка в записи лечится новой записью со
   ссылкой на старую, а не правкой старой.

Добавление записи (фактический прогон на демо-раскладке):

```bash
cat > hypotheses/graveyard/H-0004.md <<'EOF'
# H-0004 — KILLED (ARTIFACT)
- Гипотеза: volume-spike reversal на 5m-барах
- Prediction: mean event-day return в топ-5% объёма < 0 при p < 0.05
- COD: stage 1, placebo_placebo — p=0.61, случайные дни дают то же
- Вердикт: KILLED, 2026-04-05, снимок 2026-03-15, seed 7
- Resurrection conditions: новый снимок с tick-точными объёмами
  (текущий snapshot агрегирует двойной печатью)
- Не пробовать снова: тот же event-сет на агрегированных объёмах
EOF
printf '%s\n' \
  'H-0004,volume-spike-reversal,microstructure|volume|events,2026-03-25,2026-03-15,KILLED,2026-04-05,graveyard/H-0004.md' \
  >> hypotheses/INDEX.csv
```

Сверка реестра с каталогом кладбища (CI-проверка из методологии:
у каждой записи с tombstone есть строка реестра, и наоборот):

```python
import csv
from pathlib import Path

root = Path("hypotheses")
rows = list(csv.DictReader((root / "INDEX.csv").open(encoding="utf-8")))
listed = {row["h_id"] for row in rows}
buried = {p.stem for p in (root / "graveyard").glob("H-*.md")}
print("in index but no tombstone:", sorted(listed - buried) or "none")
print("tombstone missing from index:", sorted(buried - listed) or "none")
assert listed == buried
```

Фактический вывод после добавления H-0004:

```text
in index but no tombstone: none
tombstone missing from index: none
```

В CI это транслируется в отдельный шаг (вместе с lint пререгистраций);
ночная задача может пересчитывать `SURVIVED`-гипотезы на новом снимке —
проверка, что выжившие не умерли на свежих данных.

## Чек-лист

- [ ] Graveyard check перед каждой новой гипотезой: теги → COD →
      сверка prediction + kill-критерия; результаты — в prereg.
- [ ] Tombstone содержит все обязательные поля, включая COD из
      таксономии пайплайна и resurrection conditions.
- [ ] Строка добавлена в `INDEX.csv`, реестр совпадает с каталогом.
- [ ] Существующие надгробия не редактировались (append-only).
- [ ] Незакрытых гипотез нет дольше лимита открытых дней — вердикт
      обязателен для каждой `H-NNNN`, включая свои неудобные.

## Что дальше

- Как получить вердикт для tombstone — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Как оформить prereg с результатами graveyard check — how-to 01
  (`docs/how-to/01-preregister.md`).
