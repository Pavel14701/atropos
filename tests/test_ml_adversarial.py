"""Adversarial-integrity tests for the walk-forward ML engine.

Four adversarial angles on ``atropos.ml_training``, each attacking a
structural promise of the engine rather than a numeric answer:

* purge/embargo integrity -- clustered events with sub-purge gaps,
  events exactly on the purge and embargo boundaries (+/- 1us),
  duplicated stamps and a huge gap: the keep-mask must equal its
  definitional formula, keep nothing inside the burned pre-test zone,
  and treat the boundary stamp itself as inclusive (``<=``);
* fold invariants -- ``fold_edges`` / ``assign_folds`` must partition
  the unique stamps chronologically (no gaps, no overlaps, no empty
  interior blocks) and refuse clustered times below the two-per-block
  floor with the documented ``ValueError``;
* paired worlds -- destroying the planted dependence (target shuffle)
  must kill every registry model's verdict on three seeds, and
  bootstrap-resampled exec worlds must keep every EV statistic inside
  the honest bounds of the underlying returns;
* config adversarial -- disabled gates are respected (pure noise
  SURVIVES with everything off), an unattainable importance threshold
  refuses or triggers, a missing battery is INCONCLUSIVE (never a
  silent pass), a battery over ALL columns is refused, non-integer /
  structurally broken configs raise messaged ``ValueError`` before any
  model is fit, and the power gate dominates: a thin sample is
  UNDERPOWERED no matter which gates are on or off.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.ml_training import (
    MODEL_REGISTRY,
    assign_folds,
    fold_edges,
    paired_bootstrap_t_stat,
    purge_embargo_mask,
    run_ml_kill_protocol,
)

HOUR_US = 3_600_000_000

_FAST_BASE: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"gbm": {"n_estimators": 30}},
}

_ALL_OFF_CHECKS: dict[str, Any] = {
    "incremental": False,
    "cost_line": False,
    "importance_stability": False,
    "deflated_sharpe": False,
}


def _planted_world(n_rows: int = 1600, seed: int = 3) -> dict[str, Any]:
    """Hourly world with the planted nonlinear edge (see the demo)."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    exec_ret = (
        0.012 * np.tanh(feats[:, 0] * feats[:, 1] / 2.0)
        + 0.0035 * feats[:, 2]
        + 0.0024 * rng.standard_normal(n_rows)
    )
    frame = pd.DataFrame(
        {
            "f1": feats[:, 0],
            "f2": feats[:, 1],
            "f3": feats[:, 2],
            "f4": feats[:, 3],
            "ctrl1": controls[:, 0],
            "ctrl2": controls[:, 1],
        }
    )
    stamps = np.arange(n_rows, dtype=np.int64) * HOUR_US
    labels = (exec_ret > 0.004).astype(np.int64)
    return {
        "features": frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["ctrl1", "ctrl2"],
    }


def _noise_world(n_rows: int = 1600, seed: int = 7) -> dict[str, Any]:
    """Symmetric zero-mean noise world: a powered, honest DEAD."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    exec_ret = 0.008 * np.sign(rng.standard_normal(n_rows))
    frame = pd.DataFrame(
        {
            "f1": feats[:, 0],
            "f2": feats[:, 1],
            "f3": feats[:, 2],
            "f4": feats[:, 3],
            "ctrl1": controls[:, 0],
            "ctrl2": controls[:, 1],
        }
    )
    stamps = np.arange(n_rows, dtype=np.int64) * HOUR_US
    labels = (exec_ret > 0.004).astype(np.int64)
    return {
        "features": frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["ctrl1", "ctrl2"],
    }


def _run(dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Run the engine once over a dataset dict with the given config."""
    return run_ml_kill_protocol(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        config=config,
        battery_controls=dataset["battery_controls"],
    )


# ---------------------------------------------------------------------------
# purge / embargo integrity on adversarial timestamps
# ---------------------------------------------------------------------------
def _adversarial_stamps(test_start: int) -> np.ndarray:
    """Clustered, boundary-hugging, duplicated and gapped train stamps.

    Micro-second-spaced clusters (gaps far below any purge window),
    stamps exactly on the purge boundary and +/- 1us around it, the
    same for the embargo boundary, duplicate boundary stamps and one
    far-past stamp across a huge gap -- in NO particular order, since
    the mask is row-aligned, not sortedness-sensitive.
    """
    purge_offset = (1 + 2) * HOUR_US  # horizon 1h + purge 2h defaults
    embargo_offset = 24 * HOUR_US
    offsets = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,  # tight micro-second cluster
            -purge_offset,
            -purge_offset + 1,
            -purge_offset - 1,  # exact purge boundary +/- 1us
            -embargo_offset,
            -embargo_offset + 1,
            -embargo_offset - 1,  # exact embargo boundary +/- 1us
            -500 * HOUR_US,  # far past across a huge gap
            -purge_offset,
            -embargo_offset,  # duplicated boundary stamps
        ],
        dtype=np.int64,
    )
    return test_start + offsets


@pytest.mark.parametrize(
    ("horizon_h", "purge_h", "embargo_h"),
    [(1, 2, 24), (3, 10, 1), (2, 2, 2), (1, 48, 200)],
)
def test_purge_embargo_mask_matches_formula_on_adversarial_stamps(
    horizon_h: int, purge_h: int, embargo_h: int
) -> None:
    """The mask equals its definitional formula on hostile stamps.

    Keep iff ``t + (horizon + purge) hours <= test_start`` AND
    ``t + embargo hours <= test_start`` -- recomputed independently
    here and compared elementwise.  Semantically, nothing inside the
    burned pre-test zone (``t > test_start - burn``) may survive, and
    everything older than the burn is kept, for whichever of the two
    windows dominates in this parameterization.
    """
    test_start = 1000 * HOUR_US
    stamps = _adversarial_stamps(test_start)
    mask = purge_embargo_mask(
        stamps,
        test_start,
        horizon_h=horizon_h,
        purge_h=purge_h,
        embargo_h=embargo_h,
    )
    expected = (stamps + (horizon_h + purge_h) * HOUR_US <= test_start) & (
        stamps + embargo_h * HOUR_US <= test_start
    )
    assert mask.dtype == np.bool_
    assert mask.shape == stamps.shape
    assert np.array_equal(mask, expected)

    burn_us = max(horizon_h + purge_h, embargo_h) * HOUR_US
    assert not bool(mask[stamps > test_start - burn_us].any())
    assert bool(mask[stamps + burn_us <= test_start].all())


def test_purge_embargo_boundary_is_inclusive_and_duplicate_safe() -> None:
    """A stamp exactly on the boundary is kept; +1us is dropped.

    Both regimes: the embargo boundary when the embargo dominates and
    the purge boundary when the purge window dominates.  Duplicate
    boundary stamps must receive identical mask entries (the mask is
    per-row; equal stamps cannot diverge).
    """
    test_start = 1000 * HOUR_US

    embargo_edge = test_start - 24 * HOUR_US
    stamps_embargo = np.array(
        [embargo_edge - 1, embargo_edge, embargo_edge + 1], dtype=np.int64
    )
    mask_embargo = purge_embargo_mask(
        stamps_embargo, test_start, horizon_h=1, purge_h=2, embargo_h=24
    )
    assert mask_embargo.tolist() == [True, True, False]

    purge_edge = test_start - 3 * HOUR_US
    stamps_purge = np.array(
        [purge_edge - 1, purge_edge, purge_edge + 1], dtype=np.int64
    )
    mask_purge = purge_embargo_mask(
        stamps_purge, test_start, horizon_h=1, purge_h=2, embargo_h=1
    )
    assert mask_purge.tolist() == [True, True, False]

    duplicates = np.array(
        [purge_edge, purge_edge, purge_edge], dtype=np.int64
    )
    mask_dup = purge_embargo_mask(
        duplicates, test_start, horizon_h=1, purge_h=2, embargo_h=1
    )
    assert len(set(mask_dup.tolist())) == 1


# ---------------------------------------------------------------------------
# fold invariants: chronological partition without holes or overlaps
# ---------------------------------------------------------------------------
def _clustered_adversarial_stamps() -> np.ndarray:
    """Three tight clusters plus scattered stamps, chronologically."""
    clusters = (
        10 * HOUR_US + np.arange(40, dtype=np.int64),  # 1us apart
        500 * HOUR_US + np.arange(40, dtype=np.int64),
        900 * HOUR_US + np.arange(40, dtype=np.int64),
    )
    scattered = np.arange(0, 30, dtype=np.int64) * 40 * HOUR_US
    stamps = np.concatenate([scattered, *clusters])
    return np.sort(stamps)


@pytest.mark.parametrize("n_blocks", [2, 3, 4, 5, 6])
def test_fold_edges_partition_on_clustered_stamps(n_blocks: int) -> None:
    """Edges chop the unique stamps into a gap-free chronological cover.

    On clustered + scattered stamps: ``n_blocks + 1`` strictly
    increasing edges anchored at the first and last unique stamp,
    ``assign_folds`` labels every row inside ``[edge_j, edge_{j+1})``,
    sorted stamps get non-decreasing fold ids, and every block is
    populated (an empty interior block would mean a lost slice of the
    timeline).
    """
    stamps = _clustered_adversarial_stamps()
    uniq = np.unique(stamps)
    edges = fold_edges(stamps, n_blocks)
    folds = assign_folds(stamps, edges)

    assert len(edges) == n_blocks + 1
    assert all(low < high for low, high in pairwise(edges))
    assert edges[0] == int(uniq[0])
    assert edges[-1] == int(uniq[-1])
    assert set(folds.tolist()) == set(range(n_blocks))
    assert np.all(np.diff(folds) >= 0)
    for block in range(n_blocks):
        block_stamps = stamps[folds == block]
        assert block_stamps.size > 0
        assert int(block_stamps.min()) >= edges[block]
        if block < n_blocks - 1:
            assert int(block_stamps.max()) < edges[block + 1]
        else:
            # the timeline's final edge is itself a stamp: the last
            # block is closed at ``edges[-1]``
            assert int(block_stamps.max()) == edges[-1]


def test_fold_edges_rejects_times_below_two_per_block_floor() -> None:
    """Fewer than ``2 * n_blocks`` unique stamps is a loud refusal."""
    three_unique = np.repeat(np.array([10, 20, 30], dtype=np.int64), 5)
    with pytest.raises(ValueError, match="unique stamps"):
        fold_edges(three_unique, 2)
    with pytest.raises(ValueError, match="unique stamps"):
        fold_edges(np.array([], dtype=np.int64), 2)


def test_assign_folds_half_open_edges_and_duplicate_consistency() -> None:
    """Half-open blocks: a stamp equal to an inner edge opens the next.

    Pinned on the docstring example shape: ``[e, e, e)`` semantics --
    stamp ``e_1`` belongs to block 1, the final edge stamp to the last
    block, and duplicate stamps always share one fold.
    """
    edges = [0, 10, 20, 30]
    stamps = np.array([0, 9, 10, 19, 20, 29, 30], dtype=np.int64)
    assert assign_folds(stamps, edges).tolist() == [0, 0, 1, 1, 2, 2, 2]
    duplicates = np.array([10, 10, 10], dtype=np.int64)
    assert len(set(assign_folds(duplicates, edges).tolist())) == 1


# ---------------------------------------------------------------------------
# paired worlds: destroyed dependence must die, resampled worlds stay sane
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("permutation_seed", [101, 202, 303])
def test_shuffled_target_never_survives(permutation_seed: int) -> None:
    """Destroying the target-feature link kills every registry model.

    The features and exec returns stay fixed; permuting the target
    severs the dependence the models are supposed to learn.  In one
    CV pass over the WHOLE registry (import-time snapshot, so new
    entries are covered automatically) no model may come back
    SURVIVES on any of the three shuffles, and the run verdict must
    be a DEAD -- an engine that mint an edge out of shuffled labels
    is the adversarial failure this layer exists to catch.
    """
    dataset = _planted_world()
    shuffler = np.random.default_rng(permutation_seed)
    shuffled = {
        **dataset,
        "target": shuffler.permutation(dataset["target"]),
    }
    config = {
        **_FAST_BASE,
        "models": sorted(MODEL_REGISTRY),
    }
    report = _run(shuffled, config)
    assert set(report["models"]) == set(MODEL_REGISTRY)
    for name, entry in report["models"].items():
        assert entry["verdict"] != "SURVIVES", (name, permutation_seed)
    assert report["verdict"].startswith("DEAD")


@pytest.mark.parametrize("boot_seed", [11, 22, 33, 44, 55])
def test_bootstrap_exec_worlds_keep_ev_inside_honest_bounds(
    boot_seed: int,
) -> None:
    """Bootstrap exec worlds keep every EV statistic honest.

    Resampling exec returns with replacement (the labels re-derived
    from the world's own exec stream, exactly as the dataset contract
    prescribes) must keep the engine's taken-trade EV inside
    ``[min(exec), max(exec)]`` of that world -- an EV outside the
    support of the returns it averages is fabricated -- and within
    one full exec range of the base world's EV.  Each world is
    deterministic: rerunning it reproduces the report exactly.
    """
    dataset = _planted_world()
    config = {**_FAST_BASE, "model": "logit"}
    base_report = _run(dataset, config)
    assert base_report["ev_model"] is not None

    sampler = np.random.default_rng(boot_seed)
    world_exec = dataset["exec_returns"][
        sampler.integers(0, dataset["exec_returns"].size, 1600)
    ]
    world = {
        **dataset,
        "exec_returns": world_exec,
        "target": (world_exec > 0.004).astype(np.int64),
    }
    report = _run(world, config)
    ev_model = report["ev_model"]
    assert ev_model is not None
    assert float(world_exec.min()) <= ev_model <= float(world_exec.max())
    assert abs(ev_model - float(base_report["ev_model"])) <= float(
        dataset["exec_returns"].max()
    ) - float(dataset["exec_returns"].min())
    repeat = _run(world, config)
    assert repeat == report


# ---------------------------------------------------------------------------
# config adversarial: gates respect their toggles, configs refuse loudly
# ---------------------------------------------------------------------------
def test_all_gates_off_pure_noise_survives() -> None:
    """With every gate disabled the engine must let pure noise pass.

    Disabled gates are disabled: on the noise world the verdict is
    SURVIVES, every gate record reports ``enabled=False`` /
    ``triggered=False``, and the run stays powered (the power floor
    is a sample-size property, not a gate someone can switch off).
    """
    report = _run(_noise_world(), {**_FAST_BASE, "checks": _ALL_OFF_CHECKS})
    assert report["verdict"] == "SURVIVES"
    assert report["powered"] is True
    for gate in report["kills"]:
        assert gate["enabled"] is False
        assert gate["triggered"] is False


def test_unattainable_importance_threshold_refuses_and_triggers() -> None:
    """rho_min > 1 is refused; an isolated importance gate still kills.

    ``importance_rho_min = 1.01`` is structurally impossible (Spearman
    rho is bounded by 1) and the config guard refuses it with a
    messaged ``ValueError`` naming the key.  With rho_min exactly 1.0
    and the earlier gates disabled, the noise world's unstable feature
    rankings trigger the gate: verdict ``DEAD(importance_stability)``.
    """
    dataset = _noise_world()
    with pytest.raises(ValueError, match="importance_rho_min"):
        _run(dataset, {**_FAST_BASE, "checks": {"importance_rho_min": 1.01}})

    isolated = {
        **_FAST_BASE,
        "checks": {
            "incremental": False,
            "cost_line": False,
            "deflated_sharpe": False,
            "importance_rho_min": 1.0,
        },
    }
    report = _run(dataset, isolated)
    assert report["verdict"] == "DEAD(importance_stability)"
    importance_gate = next(
        gate
        for gate in report["kills"]
        if gate["check"] == "importance_stability"
    )
    assert importance_gate["triggered"] is True
    assert report["importance_stability"] is not None
    assert report["importance_stability"] < 1.0


def test_battery_absence_is_inconclusive_and_full_battery_refuses() -> None:
    """No battery: INCONCLUSIVE, never a silent pass or a silent kill.

    Without battery columns the incremental gate is enabled, NOT
    triggered and flagged inconclusive with a reason; the planted
    world still runs and survives on the remaining gates.  Naming
    EVERY feature column a battery control leaves no model feature,
    which the dataset contract refuses loudly.
    """
    dataset = _planted_world()
    no_battery = {**dataset, "battery_controls": None}
    report = _run(no_battery, _FAST_BASE)
    incremental_gate = next(
        gate for gate in report["kills"] if gate["check"] == "incremental"
    )
    assert incremental_gate["enabled"] is True
    assert incremental_gate["triggered"] is False
    assert incremental_gate["inconclusive"] is True
    assert incremental_gate["reason"]
    assert report["battery"]["available"] is False
    assert report["verdict"] == "SURVIVES"

    full_battery = {
        **dataset,
        "battery_controls": list(dataset["features"].columns),
    }
    with pytest.raises(ValueError, match="non-control column"):
        _run(full_battery, _FAST_BASE)


_INVALID_CONFIGS: dict[str, dict[str, Any]] = {
    "k_blocks_float": {"k_blocks": 4.5},
    "k_blocks_string": {"k_blocks": "4"},
    "k_blocks_below_minimum": {"k_blocks": 1},
    "bootstrap_b_float": {"bootstrap_b": 100.5},
    "n_min_events_float": {"n_min_events": 5.5},
    "checks_toggle_not_bool": {"checks": {"incremental": "off"}},
    "checks_not_a_dict": {"checks": "all-off"},
    "checks_threshold_nan": {"checks": {"incremental_t": float("nan")}},
    "model_unknown": {"model": "no_such_model"},
    "models_empty_list": {"models": []},
    "models_not_strings": {"models": ["gbm", 7]},
    "grid_empty": {"threshold_grid": ()},
    "grid_not_increasing": {"threshold_grid": (0.5, 0.2)},
    "grid_outside_unit": {"threshold_grid": (0.0, 0.5)},
    "model_params_not_nested": {"model_params": {"gbm": 30}},
    "profit_bar_negative": {"profit_bar": -0.001},
}


@pytest.mark.parametrize("case_name", sorted(_INVALID_CONFIGS))
def test_invalid_config_refuses_loudly(case_name: str) -> None:
    """Non-integer / structurally broken configs fail before any fit."""
    with pytest.raises(ValueError) as excinfo:
        _run(_planted_world(), {**_FAST_BASE, **_INVALID_CONFIGS[case_name]})
    assert str(excinfo.value), "a refusal must explain itself"


def test_non_integer_bootstrap_params_refuse_loudly() -> None:
    """``validate_int`` guards the engine's integer knobs everywhere."""
    exec_vals = np.array([0.01, -0.005, 0.006, 0.004, -0.002, 0.008])
    taken = np.array([True] * 3 + [False] * 3)
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_t_stat(
            exec_vals,
            taken,
            exec_vals,
            taken,
            n_boot=7.5,  # type: ignore[arg-type]
            seed=0,
        )
    with pytest.raises(ValueError, match="seed"):
        paired_bootstrap_t_stat(
            exec_vals, taken, exec_vals, taken, n_boot=10, seed=-1
        )


def test_k_blocks_beyond_available_unique_stamps_refuses() -> None:
    """More blocks than the timeline can carry refuses before fitting."""
    dataset = _planted_world(n_rows=300, seed=5)
    with pytest.raises(ValueError, match="unique stamps"):
        _run(dataset, {**_FAST_BASE, "k_blocks": 200})


# ---------------------------------------------------------------------------
# UNDERPOWERED dominance: the power gate outranks every gate toggle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "checks",
    [_ALL_OFF_CHECKS, {"deflated_sharpe": False}],
    ids=["all_off", "fast"],
)
@pytest.mark.parametrize("world_kind", ["planted", "noise"])
def test_underpowered_dominates_gate_toggles(
    world_kind: str, checks: dict[str, Any]
) -> None:
    """A thin sample is UNDERPOWERED whatever the gates say about it.

    Thirty declustered events against the default floor of 500: the
    verdict is UNDERPOWERED with all gates on AND with all gates off,
    ``powered`` is False, and per-model verdicts echo it -- gate flags
    stay informational (the noise world's gates DO trigger) but can
    never turn a starved sample into a DEAD or a SURVIVES.
    """
    dataset = (
        _planted_world(n_rows=480, seed=11)
        if world_kind == "planted"
        else _noise_world(n_rows=480, seed=13)
    )
    report = _run(
        dataset, {**_FAST_BASE, "n_min_events": 500, "checks": checks}
    )
    assert report["n_events"] < report["n_min_events"]
    assert report["powered"] is False
    assert report["verdict"] == "UNDERPOWERED"
    assert report["n_min_events"] == 500
    for name, entry in report["models"].items():
        assert entry["verdict"] == "UNDERPOWERED", name
