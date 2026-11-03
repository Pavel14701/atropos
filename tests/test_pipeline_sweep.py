"""Multi-seed sweep of the ensemble pipeline: test the tests.

The pipeline's job is to kill candidates; these tests check the killer
itself across MANY data seeds instead of the single demo seeds the rest
of the suite exercises.  Both failure modes are covered: over-kill (a
genuine planted edge must SURVIVE on every swept seed - a pipeline that
murders real research is broken) and under-kill (pure noise must get a
kill verdict on every swept seed - noise that sneaks through becomes a
published "strategy").  All tests are marked ``slow``; each finishes far
inside the 60 s slow budget at the reduced CI config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.ensemble_pipeline import _make_synthetic, run_full_pipeline

# CI-budget config identical to tests/test_ensemble_pipeline.py: the
# same knobs as DEFAULT_CONFIG, fewer draws/sims so a sweep of seeds
# stays fast at n_obs=400.
REDUCED_CONFIG: dict[str, Any] = {
    "n_perm_draws": 30,
    "battery": {"n_boot": 100},
    "mc": {"n_sim": 100},
    "importance": {"n_bootstrap": 10},
}

# The sweep seeds were fixed a priori (the first ten data seeds); the
# rest of the suite only ever exercises the seed=4 / seed=13 worlds.
SWEEP_SEEDS: tuple[int, ...] = tuple(range(10))

# any verdict that means "candidate killed"; CS_NO_EDGE needs panel
# data the series-only sweep never supplies.
KILL_VERDICTS: frozenset[str] = frozenset(
    {"REPACKAGING", "DEAD", "NO_OOS_EDGE", "RISK", "COST_DEATH", "ARTIFACT"}
)


def _make_noise_world(seed: int) -> dict[str, Any]:
    """Pure-noise candidate: signal independent of returns (n=400).

    Same construction as the single-seed noise test in
    tests/test_ensemble_pipeline.py, parametrized by seed so the sweep
    covers fresh draws instead of the one showcased rng stream.
    """
    rng = np.random.default_rng(5_000 + seed)
    n_obs = 400
    index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    return {
        "returns": pd.Series(rng.standard_normal(n_obs) * 0.01, index=index),
        "signal": pd.Series(rng.standard_normal(n_obs), index=index),
        "hypothesis": "pure noise must die",
    }


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "FINDING (2026-11-03, review T5): the FAST genuine edge "
        "(raw AR(0.5) driver) dies at stage 3 on sweep seeds 2 and 7 "
        "(robustness battery DEAD at n_obs=400 with REDUCED_CONFIG). "
        "seed 2: 5/7 battery tests pass but window_sensitivity is "
        "catastrophic - the window-smoothed mean return degrades "
        "monotonically (w=5 mu=+0.07, w=10 mu=-0.03, w=15 mu=-0.11): "
        "smoothing INVERTS the signal->return alignment because the "
        "driver's memory (~2 periods) is shorter than the smoothing "
        "windows (5/10/15); a smoothed high reads as 'the state was "
        "high over the whole window' -> mean-reversion anti-signal. "
        "The battery is RIGHT to kill this: the class is not "
        "smoothing-robust. Review verdict on the three options: "
        "(1) accept the kill - honest, pinned here; (2) extend the "
        "generator with a second, genuinely SLOW edge class - DONE, "
        "see kind='slow_genuine_edge' and "
        "test_edge_class_contract_fast_vs_slow below, which proves "
        "the battery PASSES slow-robust edges on all 10 seeds; "
        "(3) soften the window_sensitivity rule - REJECTED "
        "(p-hacking the killer). seed 7 detail unchanged: the "
        "baseline itself FAILs (2/7, threshold=-0.252) with "
        "catastrophic subperiod_thirds and declustering."
    ),
    strict=True,
)
def test_genuine_edge_survives_on_every_sweep_seed() -> None:
    """Genuine planted edge must SURVIVE on all 10 swept seeds.

    Over-kill check: the single-seed demo tests (always seed=4) cannot
    distinguish a calibrated pipeline from one tuned to its showcase
    seed.  Every swept seed must come back with a SURVIVES* verdict; a
    dying seed is a finding about the pipeline, never licence to loosen
    the assertion (the currently dying seeds are documented in the
    xfail reason above).
    Budget: ~0.5 s measured (10 REDUCED_CONFIG runs, n_obs=400).
    """
    for seed in SWEEP_SEEDS:
        result = run_full_pipeline(
            _make_synthetic("genuine_edge", n_obs=400, seed=seed),
            REDUCED_CONFIG,
        )
        assert result["overall_verdict"].startswith("SURVIVES"), (
            f"seed={seed}: verdict={result['overall_verdict']} "
            f"killed_by_stage={result['killed_by_stage']} "
            f"kill_reason={result['kill_reason']}"
        )


@pytest.mark.slow
def test_edge_class_contract_fast_vs_slow() -> None:
    """Positive contract: the battery PASSES slow-robust edges.

    The review's fast/slow edge-class contract.  The FAST class
    (kind='genuine_edge', raw AR(0.5) driver, memory shorter than the
    battery's 5/10/15 smoothing windows) is honestly killed by
    window_sensitivity on a minority of seeds - that finding is pinned
    in test_genuine_edge_survives_on_every_sweep_seed above and MUST
    stay an xfail (softening the rule was rejected as p-hacking).  The
    fix was to EXTEND the generator, not weaken the killer:
    kind='slow_genuine_edge' averages the same driver into a slow
    state whose information spans a multi-row window, so battery
    smoothing overlaps the information instead of washing past it -
    the edge attenuates but never inverts.  This test asserts the
    positive side: the slow class must SURVIVE* on EVERY swept seed.
    A dying seed here would be a NEW finding about the battery
    over-killing slow-robust edges - report it, never loosen this
    assert.
    Budget: ~0.5 s measured (10 REDUCED_CONFIG runs, n_obs=400).
    """
    for seed in SWEEP_SEEDS:
        result = run_full_pipeline(
            _make_synthetic("slow_genuine_edge", n_obs=400, seed=seed),
            REDUCED_CONFIG,
        )
        assert result["overall_verdict"].startswith("SURVIVES"), (
            f"seed={seed}: slow_genuine_edge verdict="
            f"{result['overall_verdict']} "
            f"killed_by_stage={result['killed_by_stage']} "
            f"kill_reason={result['kill_reason']}"
        )


@pytest.mark.slow
def test_pure_noise_is_killed_on_every_sweep_seed() -> None:
    """Pure noise must get a kill verdict on all 10 swept seeds.

    Under-kill check: which stage does the killing is free - measured
    9/10 seeds die as REPACKAGING at stage 1; seed 7 passes stage 1 by
    multiple-testing luck (every control p-value lands at 0.02-0.03,
    including the 1/31 permutation floor at n_perm_draws=30) and the
    battery then kills it at stage 3 as DEAD.  The requirement is only
    that EVERY seed ends in a kill verdict with ``killed_by_stage`` set.
    Budget: ~0.2 s measured (10 early-dying REDUCED_CONFIG runs).
    """
    for seed in SWEEP_SEEDS:
        result = run_full_pipeline(_make_noise_world(seed), REDUCED_CONFIG)
        assert result["overall_verdict"] in KILL_VERDICTS, (
            f"seed={seed}: noise survived as "
            f"{result['overall_verdict']} "
            f"(killed_by_stage={result['killed_by_stage']})"
        )
        assert result["killed_by_stage"] is not None, (
            f"seed={seed}: kill verdict {result['overall_verdict']} "
            "without a killing stage"
        )


@pytest.mark.slow
def test_surviving_demo_run_reports_no_failed_stage() -> None:
    """Self-application: SURVIVES* implies no stage status is FAIL.

    Light version of the doctrine's self-check: the overall verdict
    must be the AND of its parts.  A run that prints SURVIVES while any
    stage carries FAIL would mean the orchestrator ignores its own
    stages, so on the canonical surviving demo run (genuine_edge,
    seed=4) every stage must report PASS / WARN / SKIPPED.
    Budget: ~0.1 s measured (one REDUCED_CONFIG run).
    """
    result = run_full_pipeline(
        _make_synthetic("genuine_edge", n_obs=400, seed=4), REDUCED_CONFIG
    )
    assert result["overall_verdict"].startswith("SURVIVES")
    for stage in result["stages"]:
        assert stage["status"] in {"PASS", "WARN", "SKIPPED"}, (
            f"stage {stage['stage']} ({stage['name']}) reports "
            f"{stage['status']} under {result['overall_verdict']}"
        )
