# How-to 02. Чтение вердиктов пайплайна

> **Estimated time:** 15 минут (примеры суммарно считаются ~5 секунд)
>
> **You will need:** репозиторий atropos с настроенным окружением
> (`uv sync`), Python 3.11+; pandas/numpy идут зависимостями. Данные
> своих гипотез подставляете свои — здесь примеры синтетические.

## Зачем

`run_full_pipeline` прогоняет сигнал через до семи adversarial-стадий и
возвращает ровно один вердикт. Этот how-to — переводчик: что означает
каждый вердикт, какая стадия его вынесла, и что делать дальше.
Самодостаточно: читать `FRAMEWORK.md` не нужно. Ниже — и словарь
второй цепочки убийств, ML (`run_ml_kill_protocol`: три вердикта,
четыре гейта), и unified-матрица, когда прогоняются обе.

Импорт (пакет `atropos` не реэкспортирует функции наверх — используйте
путь модуля):

```python
from atropos.ensemble_pipeline import run_full_pipeline
```

Вход: `data` — словарь с `returns` (forward-доходности) и `signal`
(тот же индекс); опционально `features` (DataFrame известных
контролей), `mom7`/`mom30`, `hypothesis` (строка-описание). Для
cross-sectional режима: `panel_signal`/`panel_returns` (DataFrame:
строки — даты, столбцы — активы). Второй аргумент — `hypothesis_config`
(overrides поверх `DEFAULT_CONFIG`).

Выход — словарь с ключами:

| Ключ | Содержимое |
|------|-----------|
| `overall_verdict` | один из 9 вердиктов (таблица ниже) |
| `killed_by_stage` | номер стадии-убийцы или `None` |
| `kill_reason` | причина одной строкой (или `None`) |
| `stages` | список из 7 словарей `{stage, name, status, summary, results}` |
| `config` | JSON-совместимая копия фактического конфига |
| `n_obs`, `date_range`, `hypothesis`, `runtime_seconds` | контекст прогона |

## 9 вердиктов

| Вердикт | Кто выносит | Что значит | Что делать дальше |
|---------|-------------|------------|-------------------|
| `SURVIVES` | все стадии PASS | «пока не убито» — **не подтверждение**, а отсутствие убийства | записать tombstone SURVIVED; использовать как обоснование; мониторить на новых снимках |
| `SURVIVES_WITH_WARNINGS` | все стадии PASS, есть WARN | выжила с оговорками (неопределившиеся контролы, turnover, deflated Sharpe) | выписать WARN-и в tombstone; решить, совместимы ли оговорки с решением |
| `REPACKAGING` | стадия 1 (momentum/known-features) или 2 | сигнал добавляет ничего сверх известного момента/фич — переупаковка | не пересматривать без новой механики; новая конструкция = новая гипотеза |
| `ARTIFACT` | стадия 1 (остальные контроли) | эффект = артефакт выравнивания / плацебо / концентрации по подпериодам / календаря | проверить механику данных; воскресление — только по условиям tombstone |
| `DEAD` | стадия 3 | правило нетрейдится или катастрофа батареи робастности (7 тестов) | правило нереализуемо в текущем виде; другое правило = новая гипотеза |
| `NO_OOS_EDGE` | стадия 4 или 6 | нет out-of-sample эджа (walk-forward / purged CV) | не возвращаться на тех же данных; новый снимок — по условиям tombstone |
| `COST_DEATH` | стадия 5 | gross-конверт проходит, а net-of-cost Sharpe нет: эдж съеден издержками | пересматривать только экономику исполнения (maker-режим, реже ребаланс); условия — в tombstone |
| `RISK` | стадия 5 | breach риск-конверта: mDD-хвост ИЛИ gross Sharpe-пол | несовместимо с риск-мандатом; другой скейлинг/мандат = новая гипотеза |
| `CS_NO_EDGE` | стадия 7 | cross-sectional тесты провалены (IC / net spread / монотонность квантилей) | для панельных семейств: семейство закрыто в этой формулировке |

## Статусы стадий

| Статус | Значение |
|--------|----------|
| `PASS` | стадия пройдена |
| `WARN` | пройдена с оговоркой: какие-то проверки внутри стадии не определились или на грани (см. `summary`) |
| `FAIL` | гипотеза убита на этой стадии; все следующие стадии — `SKIPPED` |
| `SKIPPED` | стадия не запускалась: убита раньше, ИЛИ опциональная стадия без данных для неё (стадия 7 без панели; стадии 1–6 в panel-only режиме) |

`INCONCLUSIVE` — **не** статус стадии: он живёт внутри результатов.
Отдельные контроли стадии 1 и overall стадии 7 возвращают
`INCONCLUSIVE`, когда проверка невычислима (например, сигнал в точности
коллинеарен контролю — «collinear with controls»). Наверх это
транслируется как WARN стадии, не как kill: «не смогли проверить» ≠
«убито». Это и есть смысл INCONCLUSIVE во всём фреймворке — мало
данных или невычислимость не убивают гипотезу.

Семь стадий по порядку: 1 `null_control_suite` (6 null-контролей),
2 `incremental_regression` (specification curve), 3
`robustness_battery` (7 тестов), 4 `walkforward_framework`,
5 `monte_carlo` (конверт Sharpe/mDD net-of-cost), 6
`predictive_model_eval` (purged CV, deflated Sharpe), 7
`cross_sectional` (опциональная, при наличии панели).

## ML-цепочка (run_ml_kill_protocol)

У фреймворка вторая цепочка убийств — walk-forward ML-движок поверх
признакового датасета (`features` → бинарный `target`), а не
`signal` → исход. У неё собственный словарь вердиктов: три, а не
девять. Как собрать датасет и прогнать — how-to 05
(`docs/how-to/05-run-ml-kill-protocol.md`); здесь — переводчик.

```python
from atropos.ml_training import run_ml_kill_protocol
```

| Вердикт | Кто выносит | Что значит | Что делать дальше |
|---------|-------------|------------|-------------------|
| `SURVIVES` | каждая запрошенная модель прошла все включённые гейты при `n_events >= n_min_events` | «пока не убито» — не подтверждение, а отсутствие убийства | записать tombstone SURVIVED с числами (EV, DSR, rho, n_events); мониторить на новых снимках |
| `DEAD(<gate>)` | первый сработавший гейт первой невыжившей модели | эдж не пережил этот гейт; имя в скобках — причина | читать `reason` записи гейта в `kills` / `models[name]["kills"]`; resurrection-условия — в tombstone |
| `UNDERPOWERED` | у модели `n_events < n_min_events` (дефолт 500 declustered-событий) | **undecided, not a kill**: выборка не запитывает ни одно гейт-решение, флаги гейтов — информационные | добрать данных и перегнать; в unified-матрице это INCONCLUSIVE, не смерть |

Четыре гейта и их пороги (`checks.*` в конфиге):

| Гейт | Что ловит | Дефолтный порог |
|------|-----------|-----------------|
| `incremental` | переупаковку линейной части: paired-bootstrap t разницы EV взятых сделок модели и логит-батареи (фичи + ваши `battery_controls`) | `incremental_t` = 2.0 |
| `cost_line` | книгу, не окупающую издержки: merged-OOS EV взятых сделок ниже плиты | `profit_bar` = 0.0040 (40bp) |
| `importance_stability` | «сигнал из шума»: per-fold permutation-ранжировки признаков не воспроизводятся | `importance_rho_min` = 0.5 |
| `deflated_sharpe` | множественное тестирование: deflated Sharpe дневного EV-потока | `dsr_min` = 0.95 |

Каждая запись гейта — `{check, enabled, triggered, inconclusive,
reason}`. Семантика INCONCLUSIVE та же, что во всём фреймворке:
«не смогли проверить» ≠ «убито» (типовой случай — incremental без
`battery_controls`). Осторожно: SURVIVES при невычислимом
incremental означает, что инкрементальный эдж не доказан, а не
пройден. Вердикт `DEAD(<gate>)` называет первый сработавший гейт по
порядку `incremental` → `cost_line` → `importance_stability` →
`deflated_sharpe`, а не «самую громкую» причину.

## Обе цепочки (run_full_kill_chain)

Rule-конвейер и ML-движок можно прогнать одной обёрткой
(`rule_data` + опциональный `ml_dataset`); это тонкий агрегатор без
своих гейтов: `unified` — чистая функция двух вердиктов, и kill
любой цепочки решает всё. «Жива» = `SURVIVES` /
`SURVIVES_WITH_WARNINGS`; `[...]` — `unified["source"]`.

```python
from atropos.kill_chain import run_full_kill_chain
```

| rule-вердикт | ml-вердикт | unified |
|--------------|------------|---------|
| жива | (не запускалась) | `SURVIVES` |
| kill | (не запускалась) | `KILLED` [rule] |
| жива | `SURVIVES` | `SURVIVES` |
| жива | `DEAD(...)` | `KILLED` [ml] |
| жива | `UNDERPOWERED` | `INCONCLUSIVE` [ml] |
| kill | `SURVIVES` | `KILLED` [rule] |
| kill | `DEAD(...)` | `KILLED` [rule+ml] |
| kill | `UNDERPOWERED` | `KILLED` [rule] |

ML `UNDERPOWERED` транслируется в INCONCLUSIVE — undecided is not
death: новый снимок с большей мощностью может решить гипотезу. Как
собрать оба датасета и что делать с каждым исходом — how-to 05;
справочник: `docs/api/atropos/kill_chain.html`.

## Разобранные примеры (все прогоны фактические)

### Пример 1: настоящий эдж → SURVIVES

Синтетический мир: скрытый AR(0.5)-драйвер предсказывает forward-return,
сигнал = драйвер + 30% «кровотечения» момента, момент — конфаундер в
доходности. Так выглядит честно посаженный эдж.

```python
import numpy as np
import pandas as pd

from atropos.ensemble_pipeline import run_full_pipeline


def make_planted_edge(n_obs=900, seed=4):
    rng = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    mom_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        mom[pos] = 0.5 * mom[pos - 1] + mom_shocks[pos]
    mom /= float(np.std(mom))
    driver = np.zeros(n_obs)
    driver_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        driver[pos] = 0.5 * driver[pos - 1] + driver_shocks[pos]
    driver /= float(np.std(driver))
    signal = driver + 0.3 * mom
    fwd = driver + 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(fwd, index=idx),
        "signal": pd.Series(signal, index=idx),
        "hypothesis": "persistent order-flow state z predicts next-period return",
    }


res = run_full_pipeline(make_planted_edge())
print("verdict      :", res["overall_verdict"])
print("killed_by    :", res["killed_by_stage"])
for st in res["stages"]:
    print(f"  {st['stage']} {st['name']:<24} {st['status']:<7} {st['summary'][:95]}")
```

Фактический вывод:

```text
verdict      : SURVIVES
killed_by    : None
  1 null_control_suite       PASS    6/6 controls passed
  2 incremental_regression   PASS    SURVIVES: max|t|=55.94 across 10 specs (10 with |t|>=2, 5 controlled, sign_consistent=True)
  3 robustness_battery       PASS    ROBUST: 7/7 tests passed (baseline threshold=0.089)
  4 walkforward_framework    PASS    PASS: wf_t=26.56 direction=0.853 shrink=0.88 (22 folds, no_peeking=True, mode=expanding, purge=
  5 monte_carlo              PASS    net SR p5/p50/p95 = 3.85/4.85/5.82 | mDD p5/p50/p95 = -15.7%/-9.9%/-5.9% | breakeven 8093bps (c
  6 predictive_model_eval    PASS    [model=ols] purged-CV IC=0.875 | wf_eval pnl_t=26.31 hit=0.847 | sign acc=0.847 | DSR p=0.000
  7 cross_sectional          SKIPPED not run: optional stage (no panel data)
```

Как читать: все серии-стадии PASS, панельных данных нет — стадия 7
SKIPPED как опциональная (`killed_by_stage=None`, поэтому summary у
SKIPPED именно «optional stage», а не «killed earlier»). SURVIVES здесь
означает «пока не убито», не «доказано»: слабое эдж-качество в ваших
реальных данных будет выглядеть гораздо скромнее этих синтетических
чисел, и это нормально.

### Пример 2: чистый шум → убита на стадии 1

```python
import numpy as np
import pandas as pd

from atropos.ensemble_pipeline import run_full_pipeline

rng = np.random.default_rng(11)
n_obs = 900
idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
data = {
    "returns": pd.Series(rng.standard_normal(n_obs), index=idx),
    "signal": pd.Series(rng.standard_normal(n_obs), index=idx),
    "hypothesis": "claimed predictor - actually pure noise",
}
res = run_full_pipeline(data)
print("verdict      :", res["overall_verdict"])
print("killed_by    :", res["killed_by_stage"])
print("kill_reason  :", res["kill_reason"])
s1 = res["stages"][0]
print(s1["summary"])
for name, ctl in s1["results"].items():
    print(f"  {name:<24} {ctl['verdict']:<13} p={ctl['p_value']:.4f}")
```

Фактический вывод:

```text
verdict      : REPACKAGING
killed_by    : 1
kill_reason  : the new signal adds nothing beyond known momentum / known features
0/6 controls passed; failed: shuffled_permutation, placebo_placebo, momentum_control, known_features_control, subperiod_stability, calendar_control
  shuffled_permutation     FAIL          p=0.4776
  placebo_placebo          FAIL          p=0.6219
  momentum_control         FAIL          p=0.4803
  known_features_control   FAIL          p=0.4803
  subperiod_stability      FAIL          p=0.2090
  calendar_control         FAIL          p=0.5000
```

Как читать: шум не проходит ни один контроль (p-values ~0.5 — ровно
то, что должно быть под нулём). Вердикт стадии 1 уточняется: если среди
провалившихся есть `momentum_control` или `known_features_control`,
итог — `REPACKAGING`, иначе `ARTIFACT`. Шум проваливает и их тоже
(инкрементального t после контролов у него нет), поэтому REPACKAGING.
Стадии 2–7 — SKIPPED («not run: candidate killed earlier»): пайплайн
убивает на первом FAIL и не тратит ресурс на труп.

### Пример 3: COST_DEATH против RISK — матрица стадии 5

Один и тот же настоящий эдж (пример 1), две разные экономики. Стадия 5
считает bootstrap-конверт Sharpe и max-drawdown net-of-cost и
разбирает причину breach по матрице:

- **mDD-брешь** (хвост просадки хуже лимита) → всегда `RISK`;
- **gross Sharpe-пол тоже пробит** (эджа нет даже до издержек) → `RISK`;
- **только net Sharpe пробит при проходящем gross и целом хвосте mDD**
  → `COST_DEATH` (в `results` стадии будет `cost_breach=True`).

То есть mDD-брешь доминирует: COST_DEATH возможен только при целом
хвосте просадок.

```python
from atropos.ensemble_pipeline import run_full_pipeline

# ... make_planted_edge как в примере 1 ...
edge = make_planted_edge()

# (a) тот же эдж, но тонкие доходности (~2bp/день) + запретительные 140bps
thin = dict(edge)
thin["returns"] = edge["returns"] * 0.02
res_cost = run_full_pipeline(thin, {"cost": {"cost_bps": 140.0}})
print("COST_DEATH case")
print("  verdict    :", res_cost["overall_verdict"],
      "| killed_by:", res_cost["killed_by_stage"])
s5 = res_cost["stages"][4]
print("  stage 5    :", s5["status"], "|", s5["summary"])
print("  cost_breach:", s5["results"]["cost_breach"])

# (b) полная амплитуда эджа, но строгий риск-мандат: хвост mDD не хуже -5%
res_risk = run_full_pipeline(edge, {"mc": {"worst_allowed_mdd_p5": -0.05}})
print("RISK case")
print("  verdict    :", res_risk["overall_verdict"],
      "| killed_by:", res_risk["killed_by_stage"])
s5 = res_risk["stages"][4]
print("  stage 5    :", s5["status"], "|", s5["summary"])
print("  cost_breach:", s5["results"]["cost_breach"])
```

Фактический вывод:

```text
COST_DEATH case
  verdict    : COST_DEATH | killed_by: 5
  stage 5    : FAIL | FAILED: net SR p5=-0.52 <= floor 0.00 while gross p5=3.85 (cost 140bps vs breakeven 162bps)
  cost_breach: True

RISK case
  verdict    : RISK | killed_by: 5
  stage 5    : FAIL | FAILED: mDD p5=-15.7% < limit -5.0%
  cost_breach: False
```

Как читать: в случае (a) gross-конверт проходит с огромным запасом
(gross p5=3.85) — эдж настоящий, но 140bps one-way съедают его целиком
(net p5=-0.52). Причина смертельна, но «экономическая»: `COST_DEATH`
говорит «меняйте исполнение, а не сигнал». В случае (b) при дефолтных
издержках хвост просадки (mDD p5=-15.7%) пробивает мандат -5% —
`cost_breach=False`, вердикт `RISK`: даже бесплатный по издержкам эдж
несовместим с этим риск-мандатом. Одинаковая стадия — противоположные
рецепты: это и есть задача атрибуции причины внутри стадии 5.

### Пример 4: REPACKAGING на стадии 1 vs стадия 2

Уточнение вердикта `REPACKAGING` (см. пример 2): он выносится и когда
фактор-контроли стадии 1 провалены (`momentum_control` /
`known_features_control`), и когда умирает specification curve стадии 2
(«ни одна разумная спецификация не держит |t| >= 2» → переупаковка
известной информации). Различить просто: `killed_by_stage` равен 1 или
2; в `stages[0]["repackaging"]` лежит bool-флаг первого случая.

## Консервативная философия в одном абзаце

PASS стадии — не подтверждение гипотезы, а отсутствие убийства на этой
стадии; `SURVIVES` = «пока не убито». INCONCLUSIVE («не смогли
проверить»: коллинеарность, мало дат в панели, degenerate-вход) — не
kill: недостающие данные не хоронят гипотезу, а оставляют её открытой
до нового снимка. FPR-контроли калиброваны (~0.95 fail-rate под чистым
шумом), стадии выстроены defense-in-depth от дешёвых убийств к
дорогим. Первый FAIL решает всё — то, что идёт после него, SKIPPED и
не создаёт видимость «почти прошло».

## Что дальше

- Как получить ML-вердикт и прогнать обе цепочки разом — how-to 05
  (`docs/how-to/05-run-ml-kill-protocol.md`).
- Как оформить вердикт в tombstone — how-to 03
  (`docs/how-to/03-read-graveyard.md`).
- Как расширить null-контроли стадии 1 — how-to 04
  (`docs/how-to/04-add-null-control.md`).
- Справочник: `docs/api/atropos/ensemble_pipeline.html`,
  `DEFAULT_CONFIG` — там же; ML-движок —
  `docs/api/atropos/ml_training.html`.
