"""Wall-clock performance smoke tests for the hot paths.

WARNING: wall-clock smoke, generous margins - catches
order-of-magnitude regressions only.  This is NOT a benchmark suite:
no pytest-benchmark, no percentiles, no microsecond claims.  Each
budget was measured once on a dev machine (fact noted in every
docstring) and then multiplied by ~8 or floored at a generous round
constant, so a slow CI runner still passes; run-to-run wall-clock
noise is expected and absorbed by the margins.  The tests exist to
fail loudly when a code path turns quadratic or ~10x slower -
nothing finer.

Recorded caveat (review): the fixed budgets below catch CATASTROPHES
(an order-of-magnitude blow-up on any runner), not regressions on
fast runners - a path that doubles from 0.04s to 0.08s still passes
every budget by design.  Two-tier budgets (hard fail at the
catastrophe level, soft warning at a regression level) are a
deliberate follow-up, not an oversight.

All tests are marked ``slow``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.cross_sectional import run_cross_sectional_tests
from atropos.ensemble_pipeline import (
    _make_panel,
    _make_synthetic,
    run_full_pipeline,
)
from atropos.ml_training import run_ml_kill_protocol
from atropos.monte_carlo import monte_carlo_paths
from atropos.ranking import rank_within_groups
from atropos.robustness_battery import run_battery
from atropos.walkforward_framework import walkforward

# Mirror of tests/test_ensemble_pipeline.REDUCED_CONFIG (CI-budget
# knobs: same keys as DEFAULT_CONFIG, fewer draws/sims).  Duplicated
# here on purpose: performance tests must not import other test
# modules.
FAST_PIPELINE_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}

# Battery run with the DEFAULT bootstrap budget (n_boot=1000) so the
# dominant cost - 1000 block resamples of n observations - is inside
# the measured runtime at both sizes.
BATTERY_PARAMS: dict[str, Any] = {
    "threshold": 0.0,
    "window": 10,
    "n_boot": 1000,
    "block": 20,
    "seed": 0,
}


def _synthetic_edge_frame(n_rows: int, seed: int) -> pd.DataFrame:
    """AR(1) signal with a correlated one-period-forward outcome.

    Deterministic for a fixed seed; the persistent signal keeps the
    default walk-forward OLS path non-degenerate at every fold.
    """
    rng = np.random.default_rng(seed)
    signal = np.zeros(n_rows)
    shocks = rng.standard_normal(n_rows)
    for pos in range(1, n_rows):
        signal[pos] = 0.4 * signal[pos - 1] + shocks[pos]
    noise = rng.standard_normal(n_rows)
    return pd.DataFrame({"signal": signal, "outcome": 0.5 * signal + noise})


@pytest.mark.slow
def test_walkforward_5000_rows_smoke() -> None:
    """Walkforward over 5000 rows (train 500 / test 100 / step 100).

    Budget: measured fact ~0.04s (45 expanding OLS folds + the
    full-sample reference fit); fact x 8 = 0.32s, floored at the
    mandated generous constant 10.0s (never stricter than 10s).
    Wall-clock smoke, generous margins - catches order-of-magnitude
    regressions only.
    """
    frame = _synthetic_edge_frame(5000, seed=42)
    start = time.perf_counter()
    result = walkforward(frame, train_window=500, test_window=100, step=100)
    elapsed = time.perf_counter() - start
    assert result["n_folds"] == 45
    assert elapsed < 10.0, (
        f"walkforward(5000 rows) took {elapsed:.2f}s, budget 10s "
        "(measured fact ~0.04s): likely a super-linear fold/regression"
    )


@pytest.mark.slow
def test_full_pipeline_1500_obs_smoke() -> None:
    """run_full_pipeline on genuine_edge n_obs=1500, REDUCED_CONFIG.

    1500 observations is ~4x the usual test size, so all seven stages
    run on a larger-than-normal series (a genuine edge keeps every
    stage live - nothing is skipped or killed early).
    Budget: measured fact ~0.12s; fact x 8 = 0.94s, rounded up to the
    generous constant 1.5s.  Wall-clock smoke, generous margins -
    catches order-of-magnitude regressions only.
    """
    data = _make_synthetic("genuine_edge", n_obs=1500, seed=4)
    start = time.perf_counter()
    result = run_full_pipeline(data, FAST_PIPELINE_CONFIG)
    elapsed = time.perf_counter() - start
    assert result["killed_by_stage"] is None
    assert elapsed < 1.5, (
        f"run_full_pipeline(n_obs=1500) took {elapsed:.2f}s, budget "
        "1.5s (measured fact ~0.12s): one of the seven stages got "
        "much slower"
    )


@pytest.mark.slow
def test_monte_carlo_paths_20k_smoke() -> None:
    """monte_carlo_paths(100.0, 0.05, 0.2, 20000 paths, 252 steps).

    Budget: measured fact ~0.12s for the single vectorized
    (20000, 252) log-GBM simulation; fact x 8 = 0.93s, rounded up to
    the generous constant 1.5s.  Wall-clock smoke, generous margins -
    catches order-of-magnitude regressions only (e.g. a per-path
    Python loop replacing the vectorized cumsum).
    """
    start = time.perf_counter()
    paths = monte_carlo_paths(
        100.0, 0.05, 0.2, n_paths=20000, n_steps=252, seed=0
    )
    elapsed = time.perf_counter() - start
    assert paths.shape == (20000, 252)
    assert elapsed < 1.5, (
        f"monte_carlo_paths(20000x252) took {elapsed:.2f}s, budget "
        "1.5s (measured fact ~0.12s)"
    )


@pytest.mark.slow
def test_cross_sectional_500_dates_100_assets_smoke() -> None:
    """run_cross_sectional_tests on a 500-date x 100-asset panel.

    Panel built with ``atropos.ensemble_pipeline._make_panel`` (the
    canonical synthetic panel generator); a planted genuine edge
    keeps all three killers doing real work.
    Budget: measured fact ~0.28s; fact x 8 = 2.2s, rounded up to the
    generous constant 3.0s.  Wall-clock smoke, generous margins -
    catches order-of-magnitude regressions only.
    """
    panel = _make_panel("genuine_edge", n_dates=500, n_assets=100, seed=13)
    start = time.perf_counter()
    outcome = run_cross_sectional_tests(
        panel["panel_signal"], panel["panel_returns"]
    )
    elapsed = time.perf_counter() - start
    assert outcome["overall_verdict"] == "PASS"
    assert elapsed < 3.0, (
        f"run_cross_sectional_tests(500x100) took {elapsed:.2f}s, "
        "budget 3s (measured fact ~0.28s): likely a per-date Python "
        "loop gone quadratic"
    )


@pytest.mark.slow
def test_battery_quadratic_growth_guard_smoke() -> None:
    """run_battery wall-clock at n=2000 stays under 4x the n=1000 run.

    Guards against quadratic blowup (ratio ~4); n^1.5 (2.83) and
    n log n (2.1) pass by design - this is a catastrophe guard, not
    a linearity proof.  Anti-pattern: linear work (the default
    n_boot=1000 block bootstrap resamples n rows per replicate)
    predicts a ratio near 2 - measured 1.9..2.1 across trials; a
    quadratic slip (an accidentally nested loop over observations)
    lands at ~4x.  Budget 4.0x doubles the measured baseline as a
    soft margin for single-shot timer noise.  Wall-clock smoke,
    generous margins - catches quadratic blow-ups, not 10%
    regressions.
    """
    base_frame = _synthetic_edge_frame(1000, seed=7)
    double_frame = _synthetic_edge_frame(2000, seed=7)

    def timed_run(frame: pd.DataFrame) -> float:
        start = time.perf_counter()
        run_battery(frame["signal"], frame["outcome"], BATTERY_PARAMS)
        return time.perf_counter() - start

    ratio = timed_run(double_frame) / timed_run(base_frame)
    assert ratio < 4.0, (
        f"run_battery wall-clock ratio n=2000/n=1000 = {ratio:.2f} "
        "(linear expects ~2, quadratic ~4): the battery picked up a "
        "super-linear term"
    )


# ---------------------------------------------------------------------------
# big-data benchmarks (10k-row engine runs and the 5k-signal replay)
# ---------------------------------------------------------------------------
# ML-engine benchmark config: the full gate battery at CI-budget knobs
# - logit (the fast linear arm), 4 purge/embargo blocks, 100 paired
# bootstrap replicates (the default is 2000), 200 lbfgs iterations.
# Gate mechanics are the defaults; only the compute shrinks.
ML_BENCH_CONFIG: dict[str, Any] = {
    "model": "logit",
    "k_blocks": 4,
    "bootstrap_b": 100,
    "model_params": {"logit": {"max_iter": 200}},
}

# Ranking benchmark gbm params: the registry default (400 boosting
# stages) measures ~73s for this world - beyond the mandated 60s
# budget - so the benchmark runs the shrunk params used by the
# ranking test layer (30 stages, depth 2; identical mechanics).
RANKING_BENCH_CONFIG: dict[str, Any] = {
    "model_params": {"gbm": {"n_estimators": 30, "max_depth": 2}},
}

_HOUR_US = 3_600_000_000


def _hourly_stamps(n_rows: int) -> np.ndarray:
    """Monotone int64 microsecond stamps one hour apart."""
    return np.arange(n_rows, dtype=np.int64) * _HOUR_US


def _ml_dataset_10k(seed: int = 11) -> dict[str, Any]:
    """Seeded 10000-row planted world for the ML engine benchmark.

    Mirrors the canonical test_ml_training dataset recipe (duplicated
    on purpose: performance tests must not import other test
    modules): a quadrant interaction the linear battery cannot see,
    a linear f3 term, noise controls, hourly stamps.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((10000, 4))
    controls = rng.standard_normal((10000, 2))
    exec_ret = (
        0.012 * np.tanh(feats[:, 0] * feats[:, 1] / 2.0)
        + 0.0035 * feats[:, 2]
        + 0.0024 * rng.standard_normal(10000)
    )
    features = pd.DataFrame(
        {
            "f1": feats[:, 0],
            "f2": feats[:, 1],
            "f3": feats[:, 2],
            "f4": feats[:, 3],
            "ctrl1": controls[:, 0],
            "ctrl2": controls[:, 1],
        }
    )
    labels = (exec_ret > 0.004).astype(np.int64)
    return {
        "features": features,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": _hourly_stamps(10000),
        "battery_controls": ["ctrl1", "ctrl2"],
    }


def _ranking_world_10k(
    n_groups: int = 50, group_size: int = 200, seed: int = 17
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Seeded 10000-row ranking world (50 groups x 200 candidates).

    Mirrors the ranking test layer's planted world (duplicated on
    purpose: performance tests must not import other test modules):
    one latent-quality feature carries the signal, three are noise.
    """
    rng = np.random.default_rng(seed)
    rows = n_groups * group_size
    quality = np.tile(np.linspace(-1.5, 1.5, group_size), n_groups)
    shift = np.repeat(rng.normal(0.0, 0.4, n_groups), group_size)
    utility = 0.02 * (quality + shift) + 0.006 * rng.standard_normal(rows)
    features = pd.DataFrame(
        {
            "quality": quality,
            "noise_one": rng.standard_normal(rows),
            "noise_two": rng.standard_normal(rows),
            "noise_three": rng.standard_normal(rows),
        }
    )
    target = (utility > 0).astype(np.int64)
    group_ids = np.repeat(np.arange(n_groups), group_size)
    return features, target, group_ids


@pytest.mark.slow
def test_ml_training_10k_rows_under_120s() -> None:
    """run_ml_kill_protocol on 10000 rows (logit, k_blocks=4, b=100).

    The full engine walk: expanding folds over 10k hourly rows, logit
    fit/predict per fold, threshold grid + EV, paired block bootstrap
    (b=100), per-fold permutation importances, deflated Sharpe and
    the linear battery comparator.
    Budget: generous CI budget 120s - measured fact ~0.10s on the dev
    machine (fact x 8 floors far below it; CI-hardware timing for
    this 10k-row run is untested, so the ceiling stays intentionally
    huge).  Wall-clock smoke, generous margins - catches
    order-of-magnitude regressions only.
    """
    data = _ml_dataset_10k()
    start = time.perf_counter()
    report = run_ml_kill_protocol(
        data["features"],
        data["target"],
        data["exec_returns"],
        data["times_us"],
        config=ML_BENCH_CONFIG,
        battery_controls=data["battery_controls"],
    )
    elapsed = time.perf_counter() - start
    assert report["n_rows"] == 10000
    verdict = report["verdict"]
    assert (
        verdict == "SURVIVES"
        or verdict.startswith("DEAD(")
        or verdict == "UNDERPOWERED"
    ), verdict
    assert elapsed < 120.0, (
        f"run_ml_kill_protocol(10000 rows) took {elapsed:.2f}s, "
        "budget 120s (measured fact ~0.10s): likely a super-linear "
        "fold/bootstrap/importance path"
    )


@pytest.mark.slow
def test_portfolio_replay_5k_signals_under_60s() -> None:
    """run_portfolio_replay over 5000 synthetic signals (slots=3).

    ``atropos.portfolio`` is built by a parallel agent and may not
    exist yet - ``importorskip`` keeps this benchmark an honest SKIP
    until the module lands (then 5000 seeded signals replay through
    3 parallel slots).  Budget: generous CI budget 60s; measured fact
    pending (the module is absent at the time this benchmark was
    written).  Wall-clock smoke, generous margins - catches
    order-of-magnitude regressions only.
    """
    portfolio = pytest.importorskip("atropos.portfolio")
    run_portfolio_replay = portfolio.run_portfolio_replay
    rng = np.random.default_rng(7)
    n_signals = 5000
    n_bars = n_signals + 200
    close = 100.0 + np.cumsum(0.0004 + 0.004 * rng.standard_normal(n_bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.standard_normal(n_bars)) * 0.3 + 0.05
    prices = pd.DataFrame(
        {
            ("open", "SPY"): open_,
            ("high", "SPY"): np.maximum(open_, close) + spread,
            ("low", "SPY"): np.minimum(open_, close) - spread,
            ("close", "SPY"): close,
        }
    )
    prices.columns = pd.MultiIndex.from_tuples(
        [tuple(col) for col in prices.columns]
    )
    entries = np.sort(
        rng.choice(np.arange(100, n_bars - 60), n_signals, replace=False)
    )
    atr = 0.5
    signals = pd.DataFrame(
        {
            "decision_idx": entries,
            "side": np.where(rng.random(n_signals) > 0.5, "long", "short"),
            "instrument": "SPY",
            "sl": close[entries]
            - np.where(rng.random(n_signals) > 0.5, 1.0, -1.0) * atr,
            "tp": close[entries]
            + np.where(rng.random(n_signals) > 0.5, 2.0, -2.0) * atr,
            "score": rng.random(n_signals),
        }
    )
    start = time.perf_counter()
    report = run_portfolio_replay(signals, prices, slots=3)
    elapsed = time.perf_counter() - start
    assert report is not None
    assert elapsed < 60.0, (
        f"run_portfolio_replay(5000 signals, slots=3) took "
        f"{elapsed:.2f}s, budget 60s (measured fact pending - the "
        "module was absent when this benchmark was written)"
    )


@pytest.mark.slow
def test_ranking_10k_groups_under_60s() -> None:
    """rank_within_groups on 10000 rows, 50 groups (gbm, top_s=1).

    The pointwise oracle's full path: every scored group refits the
    gbm on its expanding embargoed past (48 scored groups, up to
    ~9800 train rows) and ranks its 200 candidates.
    Budget: generous CI budget 60s - measured fact ~4.0s with the
    shrunk gbm params below (the registry default's 400 boosting
    stages measures ~73s and would sit beyond the budget; identical
    mechanics, only the compute shrinks).  Wall-clock smoke,
    generous margins - catches order-of-magnitude regressions only.
    """
    features, target, group_ids = _ranking_world_10k()
    start = time.perf_counter()
    report = rank_within_groups(
        features,
        target,
        group_ids,
        model="gbm",
        top_s=1,
        seed=42,
        config=RANKING_BENCH_CONFIG,
    )
    elapsed = time.perf_counter() - start
    assert report["n_rows"] == 10000
    assert report["n_groups"] == 50
    assert report["n_scored"] >= 45
    assert elapsed < 60.0, (
        f"rank_within_groups(10000 rows x 50 groups, gbm) took "
        f"{elapsed:.2f}s, budget 60s (measured fact ~4.0s): likely "
        "a per-group refit gone super-linear"
    )
