# How-to 05. ML-kill-протокол: walk-forward гейты над feature-датасетом

> **Estimated time:** 30–40 минут (прогоны сниппетов суммарно ~2
> минуты)
>
> **You will need:** репозиторий atropos с настроенным окружением
> (`uv sync`), Python 3.11+; scikit-learn идёт зависимостью проекта.
> Данные своих гипотез подставляете свои — здесь синтетический
> seeded-мир (воспроизводим байт-в-байт). Сниппеты выполняются
> подряд в одной сессии: начиная с шага 4 они переиспользуют
> переменные шага 3.

## Когда ML-цепочка, а не rule-based

Фреймворк убивает гипотезы двумя цепочками. Rule-based
(`run_full_pipeline`, how-to 02) принимает готовое правило —
`signal` → исход — и прогоняет его через семь adversarial-стадий.
ML-цепочка (`run_ml_kill_protocol`) принимает признаковое
пространство — `features` → бинарный `target` — и проверяет,
порождает ли оно исполнимые сделки, которые переживают четыре гейта
в expanding-block walk-forward CV (purge + embargo между train и
test).

| | rule-based (`run_full_pipeline`) | ML (`run_ml_kill_protocol`) |
|---|---|---|
| Вход | `signal` + `returns`: правило → исход | `features` (DataFrame) + `target` 0/1: признаки → классификатор |
| Что заморожено в prereg | конструкция правила: пороги, окна, нормализация | сетка порогов вероятности, гейты и их пороги, экономика (`profit_bar`, издержки), модель и её параметры |
| Кто решает | 7 стадий: null-контроли, specification curve, батарея робастности, walk-forward, MC-конверт, purged CV, cross-sectional | 4 гейта над out-of-sample книгой: `incremental`, `cost_line`, `importance_stability`, `deflated_sharpe` |
| Типичный вопрос | «этот сигнал торгуется net-of-cost OOS?» | «в признаках есть сигнал сверх линейной батареи, и он выживает после всех гейтов?» |
| Вердикты | 9: `SURVIVES` … `CS_NO_EDGE` | 3: `SURVIVES` / `DEAD(<gate>)` / `UNDERPOWERED` |

Практическое правило: гипотеза формулируется как конкретное правило
входа/выхода — rule-based; эффект по построению нелинейный
(взаимодействия признаков, режимы) или правило ещё не сформулировано,
а есть только признаки — ML-цепочка. Не уверены — прогоните обе:
`run_full_kill_chain` (раздел ниже) агрегирует оба вердикта, и
убийство любой цепочки убивает гипотезу.

Самодостаточно: читать `FRAMEWORK.md` перед этим how-to не нужно.

Импорт (пакет `atropos` не реэкспортирует функции наверх —
используйте пути модулей):

```python
from atropos.ml_training import run_ml_kill_protocol
```

## Шаг 1. Пререгистрация — та же дисциплина

ML-гипотеза регистрируется тем же `generate_prereg`, что и
rule-based (полный разбор — how-to 01), до первого обращения к
данным. Удобная опора: четыре гейта движка — это готовые
kill-критерии, их достаточно переписать в prereg дословно, с
дефолтными порогами из `DEFAULT_TRAINING_CONFIG`.

```python
from atropos.prereg_generator import generate_prereg, validate_prereg

params = {
    "name": "quadrant_interaction_ml",
    "hypothesis": (
        "A boosted-tree classifier on hourly features f1..f4 finds "
        "executable long entries whose taken-trade EV clears the "
        "40bp profit bar after 20bp round-trip costs, incrementally "
        "to a linear battery on known controls."
    ),
    "construction": [
        "features f1..f4, causal transforms only; target = exec > 40bp",
        "exec = fwd_return - 20bp; walk-forward k_blocks=6, purge 2h, "
        "embargo 24h",
    ],
    "nulls": [
        "the edge is linear: the logit battery on ctrl1/ctrl2 trades "
        "the same book",
        "per-fold feature rankings are noise (unstable importance)",
    ],
    "kill_criteria": [
        {
            "criterion": "paired bootstrap t of taken-trade EV vs "
                         "linear battery",
            "comparison": "<",
            "threshold": 2.0,
        },
        "OOS EV of taken trades < 0.0040",
        "median pairwise Spearman of per-fold importances < 0.5",
        "deflated Sharpe of the daily EV stream < 0.95",
    ],
    "power_calc": {"N_required": 500, "alpha": 0.05, "power": 0.80},
    "hash": "b2c3d4e5",
}

problems = validate_prereg(params)
print(problems)
assert problems == []
doc = generate_prereg(params, date="2026-11-05")
print("\n".join(doc.splitlines()[:4]))
```

Фактический вывод:

```text
[]
# Preregistration: quadrant_interaction_ml

- **Frozen on:** 2026-11-05
- **Spec hash:** `b2c3d4e5`
```

Заметьте: `N_required: 500` в power-расчёте — это тот же пол
независимых событий, что `n_min_events: 500` движка; четыре
kill-критерия зеркалят четыре гейта шага 5. Файл
`PREREG_<name>_<date>.md` закоммитьте до выгрузки данных (шаг 4
how-to 01).

## Шаг 2. Graveyard check

Кладбище проверяется до выдвижения гипотезы (how-to 03). Если
раскладки ещё нет — сгенерируйте скелет: команда создаёт только
отсутствующие файлы и никогда не перезаписывает существующие.

```bash
uv run atropos init
grep -i "quadrant\|classifier\|ensemble" hypotheses/INDEX.csv \
  || echo "no matches: graveyard check clean"
```

Фактический вывод на пустой раскладке (Windows; на POSIX пути с `/`):

```text
  created GRAVEYARD.md
  created hypotheses\INDEX.csv
  created hypotheses\README.md
init: workspace ready; files are never overwritten - edit them manually after init.
no matches: graveyard check clean
```

Повторный `uv run atropos init` на той же раскладке печатает
`skipped (exists)` для каждого из трёх файлов и ничего не меняет —
команда идемпотентна (прогнано). Результат поиска («FOUND_NOTHING»)
вписывается в prereg.

## Шаг 3. Собрать датасет: контракт + мини-синтетика

Датасет — четыре компонента, валидируемые до любого `fit` (ошибка
`ValueError` называется по имени компонента):

| Компонент | Контракт | Пример ошибки валидации |
|---|---|---|
| `features` | DataFrame float-колонок, хронологические строки; колонки из `battery_controls` исключаются из фич моделей и добавляются линейной батарее | `feature columns must be finite floats (no NaN/inf)` |
| `target` | 1-D целые, ровно 0/1 — по строке на событие | `target must be binary 0/1, got [0, 2]` |
| `exec_returns` | 1-D конечные float; издержки **вычтены вызывающим** (`exec = fwd - cost`; `cost_rt` в конфиге — справочная запись для вызывающего, движок сам ничего не вычитает) | `exec_returns must be 1-D finite floats` |
| `times_us` | int64 микросекунды, монотонно неубывающие (ось walk-forward) | `times_us must be monotonically non-decreasing` |
| `battery_controls` | имена колонок `features` | `battery_controls not found in features: ['mom30']` |

Дубликаты штампов не ошибка: они схлопываются при подсчёте
независимых событий — `decluster_count` считает кластеры с зазором
не меньше часа, а не строки.

Мини-синтетика этого how-to: hourly µs-штампы, посаженная
нелинейность — взаимодействие `tanh(f1*f2/2)`, невидимое для любых
линейных моделей, — плюс мягкий линейный `f3` (его батарея видит)
и шум.

```python
import numpy as np
import pandas as pd


def build_world(n_rows: int, seed: int = 2026):
    """Seeded hourly world: planted tanh interaction + linear + noise."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    exec_returns = (
        0.012 * np.tanh(raw[:, 0] * raw[:, 1] / 2.0)
        + 0.0035 * raw[:, 2]
        + 0.0024 * rng.standard_normal(n_rows)
    )
    features = pd.DataFrame(
        {
            "f1": raw[:, 0], "f2": raw[:, 1],
            "f3": raw[:, 2], "f4": raw[:, 3],
            "ctrl1": controls[:, 0], "ctrl2": controls[:, 1],
        }
    )
    times_us = (
        int(np.datetime64("2024-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_rows, dtype=np.int64) * 3_600_000_000
    )
    target = (exec_returns > 0.004).astype(np.int64)  # exec > profit_bar
    return features, target, exec_returns, times_us


features, target, exec_returns, times_us = build_world(1500)
print(features.shape, round(float(target.mean()), 3),
      target.dtype, bool((np.diff(times_us) >= 0).all()))
```

Фактический вывод:

```text
(1500, 6) 0.264 int64 True
```

Свои данные должны удовлетворять тому же контракту: каждый ряд —
событие с известным exec-исходом; типичная ошибка — забыть вычесть
издержки из `exec_returns` (движок увидит эдж, которого нет:
гейты откалиброваны под уже-нетто доходности).

Валидация до fit — четыре типовые ошибки (прогон на мире выше):

```python
cases = {}
with_nan = features.copy()
with_nan.loc[3, "f1"] = np.nan
cases["NaN feature"] = (with_nan, target, exec_returns, times_us, None)
cases["non-binary target"] = (
    features, target.astype(float) * 2, exec_returns, times_us, None)
bad_times = times_us.copy()
bad_times[10] = bad_times[9] - 1
cases["non-monotone times"] = (
    features, target, exec_returns, bad_times, None)
cases["missing control"] = (
    features, target, exec_returns, times_us, ["mom30"])
for label, (feats, tgt, ex, ts, bat) in cases.items():
    try:
        run_ml_kill_protocol(feats, tgt, ex, ts, battery_controls=bat)
    except ValueError as err:
        print(f"{label}: {err}")
```

Фактический вывод:

```text
NaN feature: feature columns must be finite floats (no NaN/inf)
non-binary target: target must be binary 0/1, got [0, 2]
non-monotone times: times_us must be monotonically non-decreasing
missing control: battery_controls not found in features: ['mom30']
```

## Шаг 4. Один прогон

```python
report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict     :", report["verdict"])
print(f"n_rows={report['n_rows']}  n_events={report['n_events']}  "
      f"(min {report['n_min_events']}, powered={report['powered']})")
print("EV model/baseline/battery (bp):",
      round(report["ev_model"] * 1e4, 1),
      round(report["ev_baseline"] * 1e4, 1),
      round(report["ev_battery"] * 1e4, 1))
print("DSR:", report["deflated_sharpe"],
      " rho:", report["importance_stability"])
for gate in report["kills"]:
    print(f"  {gate['check']:<20} enabled={gate['enabled']} "
          f"triggered={gate['triggered']}")
for fold in report["folds"]:
    print(f"  fold {fold['fold']}: train={fold['train_rows']} "
          f"test={fold['test_rows']} "
          f"purge+embargo dropped={fold['n_dropped_purge_embargo']}")
```

Фактический вывод (~2 секунды):

```text
verdict     : UNDERPOWERED
n_rows=1500  n_events=249  (min 500, powered=False)
EV model/baseline/battery (bp): 72.7 59.3 61.8
DSR: 0.9999999825663033  rho: 0.7999999999999999
  incremental          enabled=True triggered=False
  cost_line            enabled=True triggered=False
  importance_stability enabled=True triggered=False
  deflated_sharpe      enabled=True triggered=False
  fold 1: train=226 test=250 purge+embargo dropped=23
  fold 2: train=476 test=250 purge+embargo dropped=23
  fold 3: train=726 test=250 purge+embargo dropped=23
  fold 4: train=976 test=250 purge+embargo dropped=23
  fold 5: train=1226 test=251 purge+embargo dropped=23
```

Как читать. Структура CV при `k_blocks=6`: пять тестовых блоков по
250 строк; самый старый блок никогда не тестируется (ему не на чём
учиться); перед каждым тест-блоком purge + embargo выкидывают по
23 train-строки. Все четыре гейта пройдены — но независимых событий
249 < 500: решать нечем, вердикт `UNDERPOWERED`. Это **не смерть и
не выживание** (таблица в шаге 5): флаги гейтов при недоборе
событий — информационные. Честный ответ — добрать данных и
перегнать.

### Добрать данных и перегнать

```python
features, target, exec_returns, times_us = build_world(6000)

report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict     :", report["verdict"])
print(f"n_rows={report['n_rows']}  n_events={report['n_events']}  "
      f"(min {report['n_min_events']}, powered={report['powered']})")
print("EV model/baseline/battery (bp):",
      round(report["ev_model"] * 1e4, 1),
      round(report["ev_baseline"] * 1e4, 1),
      round(report["ev_battery"] * 1e4, 1))
print("DSR:", report["deflated_sharpe"],
      " rho:", report["importance_stability"])
print("incremental :", report["incremental"])
print("baseline    :", report["baseline"])
print("battery     :", report["battery"])
print("dsr_detail  :", report["dsr_detail"])
```

Фактический вывод (~6 секунд):

```text
verdict     : SURVIVES
n_rows=6000  n_events=843  (min 500, powered=True)
EV model/baseline/battery (bp): 80.7 70.3 70.3
DSR: 1.0  rho: 1.0
incremental : {'gbm': {'t_vs_battery': 139.04437804450353, 't_vs_baseline': 141.4843135373026, 'skipped_draws': 0}}
baseline    : {'ev': 0.007034936819590529, 'n_events': 193}
battery     : {'available': True, 'ev': 0.007034659597109983, 'n_events': 191}
dsr_detail  : {'n_days': 209, 'var_sr': 0.44557527574926054, 'sr': {'gbm': 1.9903409535233292, 'baseline': 0.8501631731846993, 'battery': 0.8188189422471551}}
```

Как читать: 843 независимых взятых сделки — пол в 500 преодолён,
гейт-решениям есть опереться. `incremental` — paired-bootstrap t
разницы EV против батареи: 139.0 при пороге 2.0 (на синтетическом
мире запас огромный, в реальных данных он будет на порядки скромнее
— и это нормально). `deflated_sharpe` считается по дневному
EV-потоку: 209 дней, скидка на множественное тестирование — через
дисперсию SR всей семьи потоков прогона (`var_sr`, `dsr_detail`).
Батарея и baseline в этом отчёте — логиты (фичи + контроли и фичи
без контролов соответственно); их узкие книги (191/193 события) —
норма: логит не умеет срезать высокоуверенную подкнигу.

Отчёт несёт 18 ключей: вердикты (`verdict`, `kills`, `models`,
`powered`, `n_min_events`), масштаб (`n_rows`, `n_events`, `folds`),
линейные компараторы (`baseline`, `battery`, `incremental`,
`ev_model`, `ev_baseline`, `ev_battery`), DSR-семья (`dsr_detail`,
`deflated_sharpe`, `importance_stability`) и json-safe копия
фактического конфига (`config`). Плоские ключи алиасят первичную
модель (`config["model"]`); вердикты остальных моделей живут в
`report["models"][name]` — движок никогда не выбирает между ними
(см. раздел «Модели»).

## Шаг 5. Интерпретация

### Три вердикта

| Вердикт | Кто выносит | Что значит | Что делать |
|---|---|---|---|
| `SURVIVES` | каждая запрошенная модель прошла все включённые гейты при `n_events >= n_min_events` | «пока не убито» — не подтверждение, а отсутствие убийства | tombstone SURVIVED с числами; мониторить на новых снимках |
| `DEAD(<gate>)` | первый сработавший гейт первой невыжившей модели | эдж не пережил этот гейт; имя в скобках — причина | читать таблицу гейтов: смертельная причина или «экономическая»; resurrection-условия — в tombstone |
| `UNDERPOWERED` | у модели `n_events < n_min_events` (дефолт 500) | **undecided, not a kill**: выборка не может запитать ни одно гейт-решение, флаги гейтов — информационные | добрать данных (шаг 4: 249 событий → перегон на 6000 строках → решаемый прогон); в unified-матрице это INCONCLUSIVE |

### Четыре гейта

| Гейт | Что ловит | Конфигурация (`checks.*` и рядом) |
|---|---|---|
| `incremental` | переупаковку линейной части: paired-bootstrap t разницы EV взятых сделок модели и логит-батареи (фичи + ваши `battery_controls`) не выше порога | `incremental: true`, порог `incremental_t` (2.0) |
| `cost_line` | книгу, не окупающую издержки: merged-OOS EV взятых сделок ниже плиты | `cost_line: true`, `profit_bar` (0.0040 = 40bp) |
| `importance_stability` | «сигнал из шума»: per-fold permutation-ранжировки признаков не воспроизводятся (медианная парная Spearman ниже плиты) | `importance_stability: true`, `importance_rho_min` (0.5) |
| `deflated_sharpe` | множественное тестирование: deflated Sharpe (Bailey & Lopez de Prado 2014) дневного EV-потока ниже floor | `deflated_sharpe: true`, `dsr_min` (0.95), `n_trials_ds` (5) |

Каждая запись гейта — `{check, enabled, triggered, inconclusive,
reason}`. `INCONCLUSIVE` — «не смогли проверить», не kill. Типовой
случай: incremental без `battery_controls`.

```python
no_battery = run_ml_kill_protocol(
    features.drop(columns=["ctrl1", "ctrl2"]),
    target,
    exec_returns,
    times_us,
    config={"model": "gbm"},
)
print("verdict:", no_battery["verdict"], "| battery:", no_battery["battery"])
print(no_battery["kills"][0])
```

Фактический вывод:

```text
verdict: SURVIVES | battery: {'available': False, 'ev': None, 'n_events': 0}
{'check': 'incremental', 'enabled': True, 'triggered': False, 'inconclusive': True, 'reason': 'battery controls not supplied: incremental edge over a linear battery is not decidable in this run'}
```

Как читать: формально `SURVIVES` — ни один гейт не сработал. Но
incremental INCONCLUSIVE: инкрементальный эдж над линейной батареей
в этом прогоне **не доказан**, а не пройден. Такой прогон не годится
как обоснование гипотезы — подавите `battery_controls` осознанно или
не подавляйте вовсе.

### Шум обязан умереть: DEAD(<gate>)

```python
rng_noise = np.random.default_rng(7)
noise_exec = 0.008 * np.sign(rng_noise.standard_normal(len(times_us)))
noise_target = (noise_exec > 0.004).astype(np.int64)

noise_report = run_ml_kill_protocol(
    features,
    noise_target,
    noise_exec,
    times_us,
    config={"model": "logit"},
    battery_controls=["ctrl1", "ctrl2"],
)
print("verdict:", noise_report["verdict"])
print("triggered:",
      [g["check"] for g in noise_report["kills"] if g["triggered"]])
print("EV model (bp):", round(noise_report["ev_model"] * 1e4, 2),
      " n_events:", noise_report["n_events"])
```

Фактический вывод:

```text
verdict: DEAD(cost_line)
triggered: ['cost_line', 'importance_stability', 'deflated_sharpe']
EV model (bp): -0.17  n_events: 2864
```

Как читать: чистый шум под логитом торгует широкую книгу (2864
события — мощность есть, подделка UNDERPOWERED исключена), EV
−0.17bp не окупает плиту 40bp. Вердикт называет **первый сработавший
гейт по порядку** (`incremental` → `cost_line` →
`importance_stability` → `deflated_sharpe`), а не «самую громкую»
причину: incremental у этой пары модель/батарея не сработал — обе
руки торгуют почти одинаковые широкие книги, и bootstrap-разница их
EV на этом сиде вышла чуть в пользу модели. Остальные три гейта
ловят шум независимо; для смерти достаточно любого одного.

## Модели: реестр ×8 + ensemble

| Имя | Ось смещения (bias axis) |
|---|---|
| `logit` | линейный компаратор (он же baseline- и battery-рука) |
| `gbm` | бустинг, точные сплиты (дефолт движка) |
| `hist_gbm` | бустинг, гистограммные сплиты |
| `rf` | бэггинг, оптимизированные сплиты |
| `extratrees` | бэггинг, случайные сплиты (декорреляция) |
| `mlp` | нейросеть (32×16, adam) |
| `knn` | локальное соседство (distance-weighted) |
| `qda` | генеративная квадратичная поверхность |

Девятое имя, `ensemble`, появляется в реестре только после импорта
`atropos.ml_ensemble` — это документированный side effect:

```python
import atropos.ml_ensemble  # noqa: F401  (side effect, см. «Ловушки»)
from atropos.ml_training import MODEL_REGISTRY

print(sorted(MODEL_REGISTRY))
```

Фактический вывод:

```text
['ensemble', 'extratrees', 'gbm', 'hist_gbm', 'knn', 'logit', 'mlp', 'qda', 'rf']
```

Несколько моделей сравниваются в одном CV-проходе (`models` вместо
`model`); движок никогда не выбирает между ними — вердикты лежат в
`report["models"]`, а вердикт прогона эхом первого невыжившего.
Ансамбль настраивается плоскими kwargs в
`model_params["ensemble"]` (члены, веса, правило комбинации
`mean`/`weighted`/`rank_mean`/`stacking`; honest past-only OOF —
см. докстринг модуля):

```python
report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={
        "models": ["gbm", "ensemble"],
        "model_params": {
            "ensemble": {"members": ["gbm", "rf"],
                         "combination": "mean"},
        },
    },
    battery_controls=["ctrl1", "ctrl2"],
)
print("run verdict:", report["verdict"])
for name, entry in report["models"].items():
    print(f"  {name:<9} {entry['verdict']:<25} "
          f"EV={round(entry['ev'] * 1e4, 1)}bp  "
          f"events={entry['n_events']}")
```

Фактический вывод (~15 секунд):

```text
run verdict: SURVIVES
  gbm       SURVIVES                  EV=80.7bp  events=843
  ensemble  SURVIVES                  EV=87.4bp  events=664
```

Config-оверрайды мержатся по ключам на всех уровнях: указать
`{"checks": {"dsr_min": 0.95}}` — значит явным образом заморозить
дефолтный floor, не трогая остальные гейты. Пример: тот же прогон,
но моделью `rf` с её дефолтными параметрами:

```python
rf_report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={"model": "rf", "checks": {"dsr_min": 0.95}},
    battery_controls=["ctrl1", "ctrl2"],
)
print("rf default run:", rf_report["verdict"],
      " n_events:", rf_report["n_events"])

rf_report = run_ml_kill_protocol(
    features,
    target,
    exec_returns,
    times_us,
    config={
        "model": "rf",
        "model_params": {"rf": {"min_samples_leaf": 1, "max_depth": 7}},
    },
    battery_controls=["ctrl1", "ctrl2"],
)
print("rf sharp run  :", rf_report["verdict"],
      " n_events:", rf_report["n_events"],
      " EV:", round(rf_report["ev_model"] * 1e4, 1), "bp")
```

Фактический вывод:

```text
rf default run: UNDERPOWERED  n_events: 100
rf sharp run  : SURVIVES  n_events: 537  EV: 90.2 bp
```

Как читать: дефолтный `rf` (`min_samples_leaf=50`) берёт узкую
книгу — 100 независимых событий < 500, `UNDERPOWERED` не про качество
модели, а про мощность её книги; более мелкие листья дают 537
событий и решаемый прогон. Ёмкость модели — такая же часть
замороженной конструкции, как сетка порогов: фиксируйте её в prereg.

## Ловушки

- **Не подгоняйте `threshold_grid` после данных.** Сетка
  (0.30…0.70, шаг 0.05) и пороги гейтов — часть конструкции,
  замороженная в prereg (шаг 1). Порог каждого фолда честно
  выбирается по train-фолду (argmax EV по сетке); сузили сетку или
  подвинули `profit_bar` после взгляда на OOS — это новая гипотеза
  с новым prereg, не правка старой.
- **UNDERPOWERED — не смерть.** «undecided»: добавить данных и
  перегнать (шаг 4). Но и не выживание: в кладбище пишется
  INCONCLUSIVE, в unified-матрице это `INCONCLUSIVE [ml]`; флаги
  гейтов при недоборе событий не интерпретируются.
- **`battery_controls` — известные линейные контроли.** Моментумы,
  известные факторы — то, что «объяснило бы» эффект линейно. Они
  исключаются из фич-набора моделей и добавляются логит-батарее;
  не кладите туда целевые конструкты. Без них incremental
  INCONCLUSIVE и прогон формально SURVIVES при недоказанном
  инкрементальном эдже (сниппет в шаге 5).
- **Импорт `atropos.ml_ensemble` — side effect.** Без него
  `model="ensemble"` падает до любого fit:
  `ValueError: unknown model 'ensemble': available ['extratrees',
  'gbm', 'hist_gbm', 'knn', 'logit', 'mlp', 'qda', 'rf']` (прогнано);
  с ним в реестре девятое имя. Держите импорт явным наверху скрипта.
  И помните: `rank_mean`/`normalize_scores` выпрямляют распределение
  скоров в равномерное — селективный порог перестаёт отбирать узкую
  книгу. На мире этого how-to rank_mean-ансамбль из `[gbm, rf]`
  торгует всю книгу (1500 событий, EV 62.4bp) и умирает на
  `incremental`, тогда как дефолтный `mean` поверх вероятностей
  выживает (87.4bp) — дефолт не случайность (прогнано; см.
  `DEFAULT_ENSEMBLE_CONFIG`).
- **Движок не выбирает модель.** `models: [...]` — сравнение, не
  турнир: у каждого члена свой вердикт в `report["models"]`, а
  вердикт прогона — эхо первого невыжившего. «Лучшей» модели нет и
  быть не может: выбор по OOS — та же подгонка, что и правка сетки.

## Обе цепочки разом: run_full_kill_chain

Обёртка всегда гонит rule-конвейер (7 стадий) по `rule_data`
(`returns` + `signal`) и, если дан `ml_dataset` (`features`,
`target`, `exec_returns`, `times_us`), ML-движок по нему. Это тонкий
агрегатор, не третий движок: никаких новых гейтов, `unified` —
чистая функция двух вердиктов (source: `rule` / `ml` / `rule+ml`).

| rule-вердикт | ml-вердикт | unified |
|---|---|---|
| жива | (не запускалась) | `SURVIVES` |
| kill | (не запускалась) | `KILLED` [rule] |
| жива | `SURVIVES` | `SURVIVES` |
| жива | `DEAD(...)` | `KILLED` [ml] |
| жива | `UNDERPOWERED` | `INCONCLUSIVE` [ml] |
| kill | `SURVIVES` | `KILLED` [rule] |
| kill | `DEAD(...)` | `KILLED` [rule+ml] |
| kill | `UNDERPOWERED` | `KILLED` [rule] |

(«жива» = `SURVIVES` / `SURVIVES_WITH_WARNINGS`; kill — любой другой
rule-вердикт; `[...]` — `unified["source"]`.)

```python
from atropos.kill_chain import run_full_kill_chain

rule_rng = np.random.default_rng(4)
idx = pd.date_range("2024-01-01", periods=900, freq="B")
rule_data = {
    "returns": pd.Series(0.01 * rule_rng.standard_normal(900), index=idx),
    "signal": pd.Series(rule_rng.standard_normal(900), index=idx),
    "hypothesis": "ml how-to: rule chain sanity (pure noise)",
}
ml_dataset = {
    "features": features, "target": target,
    "exec_returns": exec_returns, "times_us": times_us,
}
out = run_full_kill_chain(rule_data, ml_dataset)
u = out["unified"]
print("rule:", out["rule"]["overall_verdict"], "| ml:", out["ml"]["verdict"])
print("unified:", u["verdict"], "[", u["source"], "]", "|", u["reason"])
```

Фактический вывод (самый долгий сниппет гайда, ~полминуты):

```text
rule: REPACKAGING | ml: SURVIVES
unified: KILLED [ rule ] | rule chain killed: REPACKAGING
```

Как читать: rule-цепочка убила шум как переупаковку на стадии 1,
ML-цепочка на посаженном нелинейном мире выжила — но kill любой
цепочки решает всё: `KILLED [rule]`. Чтобы увидеть `unified
SURVIVES`, живыми должны быть обе цепочки (все четыре квадранта
матрицы прогнаны и заассерены в демо модуля:
`uv run python -m atropos.kill_chain`).

## Чек-лист

- [ ] Prereg с четырьмя гейтами как kill-критериями закоммичен ДО
      данных (закон 1); `validate_prereg` вернул `[]`.
- [ ] Graveyard check вписан в prereg (how-to 03).
- [ ] Датасет прошёл валидацию контракта: конечные float-фичи,
      бинарный target, монотонные int64 µs; издержки вычтены из
      `exec_returns` вызывающим.
- [ ] `battery_controls` — известные линейные контроли; incremental
      в отчёте не INCONCLUSIVE.
- [ ] `n_events >= n_min_events` — иначе UNDERPOWERED: добрать
      данных, флаги гейтов не интерпретировать.
- [ ] `threshold_grid`, `checks.*`, `model_params` не менялись
      после первого взгляда на OOS (закон 4/5).
- [ ] Вердикт записан в tombstone (how-to 03) с числами: EV, DSR,
      rho, n_events и fold-статистика — минимальные цифры записи.

## Что дальше

- Чтение rule-вердиктов — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Tombstone и реестр кладбища — how-to 03
  (`docs/how-to/03-read-graveyard.md`).
- Пререгистрация — how-to 01 (`docs/how-to/01-preregister.md`).
- Справочник: `docs/api/atropos/ml_training.html` (движок и
  `DEFAULT_TRAINING_CONFIG`), `docs/api/atropos/ml_ensemble.html`,
  `docs/api/atropos/kill_chain.html`, `docs/api/atropos/workspace.html`.
