"""No-lookahead metamorphic tests for the walk-forward ML engine.

The honest-CV promise of ``atropos.ml_training``: no fold's artifacts
may depend on data from that fold's future.  These tests mutate the
future on purpose and assert the past does not move:

* fold geometry -- scrambling features/target/exec from the FIRST test
  start onward leaves ``fold_edges`` / ``assign_folds`` /
  ``purge_embargo_mask`` bitwise identical and every geometry field of
  every fold record unchanged (the fold machinery is a pure function
  of the timestamps);
* first test fold -- mutating everything STRICTLY AFTER the first
  test block leaves the first fold's report record (threshold / taken
  / EV of the model and both linear comparators) and its permutation
  importances bitwise identical, parametrized over the whole
  :data:`MODEL_REGISTRY` at import time (registry-growth safe);
* ``rolling_z`` -- strictly trailing window: future values never move
  past outputs bitwise, a value's own window reproduces from the
  prefix alone, and touching one value moves exactly its downstream
  window (the mutation has teeth);
* ``ev_of_taken`` / ``pick_threshold`` -- fold-level statistics, not
  causal streams: jointly permuting rows changes nothing, while the
  documented non-causality (the threshold consumes the WHOLE train
  fold, so truncation may move it) is pinned explicitly.

The stacking stage (``atropos.ml_ensemble``) is covered by the final
test: a tail mutation of the dataset leaves every earlier fold of the
full ``run_ensemble_report`` bit-identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.ml_ensemble import run_ensemble_report
from atropos.ml_training import (
    MODEL_REGISTRY,
    assign_folds,
    ev_of_taken,
    fold_edges,
    pick_threshold,
    purge_embargo_mask,
    rolling_z,
    run_ml_kill_protocol,
)

HOUR_US = 3_600_000_000
_N_ROWS = 2000
_WINDOWS = (8, 24, 96)
_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)
_FOLD_GEOMETRY_KEYS = (
    "fold",
    "train_rows",
    "test_rows",
    "test_start",
    "n_dropped_purge_embargo",
)

# minimal-but-honest engine config: identical gate mechanics to the
# defaults, compute shrunk to unit-test scale (the DSR gate needs more
# OOS days than a 2000-row hourly toy world spans)
_MINIMAL_CONFIG: dict[str, Any] = {
    "k_blocks": 4,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"gbm": {"n_estimators": 30}},
}


def _planted_dataset(n_rows: int = _N_ROWS, seed: int = 17) -> dict[str, Any]:
    """Seeded hourly world whose target loads on a planted nonlinear edge.

    Mirrors the engine's demo world: the quadrant interaction
    ``tanh(f1*f2/2)`` is invisible to the linear battery, the mild f3
    term keeps the comparators honest, and two noise columns feed the
    battery controls.
    """
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


def _mutate_from(
    dataset: dict[str, Any], start: int, seed: int
) -> dict[str, Any]:
    """Copy the dataset with every value column scrambled from ``start``.

    Timestamps are the ONLY column left untouched, so any change in a
    run's fold geometry or past-fold artifacts can only come from a
    lookahead bug, never from the mutation channel itself.
    """
    mutated = {
        **dataset,
        "features": dataset["features"].copy(),
        "target": dataset["target"].copy(),
        "exec_returns": dataset["exec_returns"].copy(),
    }
    rng = np.random.default_rng(seed)
    n_rows = int(mutated["features"].shape[0])
    mutated["features"].iloc[start:] = (
        rng.standard_normal((n_rows - start, mutated["features"].shape[1]))
        * 7.0
    )
    mutated["exec_returns"][start:] = -3.0 * mutated["exec_returns"][start:]
    mutated["target"][start:] = 1 - mutated["target"][start:]
    return mutated


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
# fold geometry: value mutations never move the walk-forward skeleton
# ---------------------------------------------------------------------------
def test_future_value_mutation_spares_fold_geometry_bitwise() -> None:
    """Scrambling values from the first test start leaves geometry fixed.

    The fold machinery (edges, fold assignment, purge/embargo masks)
    may depend on the timestamps only.  Mutating features, target and
    exec returns from the first test-block boundary onward must leave
    every geometry field of every fold record identical, while the
    first fold's model outcomes DO move -- proving the mutation had
    teeth and the comparison is not vacuous.
    """
    dataset = _planted_dataset()
    stamps = dataset["times_us"]
    n_blocks = int(_MINIMAL_CONFIG["k_blocks"])
    edges = fold_edges(stamps, n_blocks)
    folds = assign_folds(stamps, edges)
    boundary = int(np.flatnonzero(folds >= 1)[0])
    assert stamps[boundary] == edges[1]

    mutated = _mutate_from(dataset, boundary, seed=29)
    base_report = _run(dataset, _MINIMAL_CONFIG)
    mutated_report = _run(mutated, _MINIMAL_CONFIG)

    # fold primitives: bitwise identical on identical timestamps
    assert fold_edges(mutated["times_us"], n_blocks) == edges
    assert np.array_equal(assign_folds(mutated["times_us"], edges), folds)
    for fold_no in range(1, n_blocks):
        train_rows = np.flatnonzero(folds < fold_no)
        mask = purge_embargo_mask(stamps[train_rows], edges[fold_no])
        mask_mutated = purge_embargo_mask(
            mutated["times_us"][train_rows], edges[fold_no]
        )
        assert np.array_equal(mask, mask_mutated)

    # geometry of every fold record: unchanged
    for base_fold, mutated_fold in zip(
        base_report["folds"], mutated_report["folds"], strict=True
    ):
        for key in _FOLD_GEOMETRY_KEYS:
            assert base_fold[key] == mutated_fold[key], (key, base_fold)

    # teeth: the first test fold's outcomes moved under the mutation
    assert (
        base_report["folds"][0]["models"]
        != mutated_report["folds"][0]["models"]
    )


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_future_mutation_spares_first_test_fold(model_name: str) -> None:
    """First test fold's OOS artifacts are immune to later-row mutations.

    With ``k_blocks=4`` the first test fold is fold 1; everything from
    fold 2 onward is mutated (features scrambled, exec negated x3,
    target flipped).  Fold 1 trained on fold 0 and tested on fold 1 --
    strictly before the mutation point -- so its full report record
    (threshold / taken / EV for the model, baseline and battery) and
    its permutation importance vector must be IDENTICAL, for every
    model of the registry.  Later folds and the overall report must
    move (the mutation has teeth).
    """
    dataset = _planted_dataset()
    config = {**_MINIMAL_CONFIG, "model": model_name}
    n_blocks = int(config["k_blocks"])
    folds = assign_folds(
        dataset["times_us"], fold_edges(dataset["times_us"], n_blocks)
    )
    mutation_start = int(np.flatnonzero(folds >= 2)[0])
    assert mutation_start > 0

    mutated = _mutate_from(dataset, mutation_start, seed=41)
    base_report = _run(dataset, config)
    mutated_report = _run(mutated, config)

    assert base_report["folds"][0] == mutated_report["folds"][0]
    base_model = base_report["models"][model_name]
    mutated_model = mutated_report["models"][model_name]
    assert (
        base_model["importance_by_fold"][0]
        == mutated_model["importance_by_fold"][0]
    )

    # teeth: the report as a whole and some later fold moved
    assert base_report != mutated_report
    assert any(
        base_report["folds"][idx] != mutated_report["folds"][idx]
        for idx in range(1, len(base_report["folds"]))
    )


# ---------------------------------------------------------------------------
# rolling_z: strictly trailing window, bitwise causal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", _WINDOWS)
def test_rolling_z_future_never_moves_past(window: int) -> None:
    """Values beyond a row's trailing window never move its z bitwise.

    For several cut points, everything strictly after ``slot +
    window`` is permuted and scaled: the z prefix up to ``slot +
    window`` must be BITWISE identical (an equality, not a tolerance --
    any difference is leakage from the future; ``equal_nan`` because
    the honest NaN warm-up head is part of the contract).  The
    trailing-window property is checked from the other side too: the
    prefix alone reproduces every z value.  Bumping one value moves
    exactly its downstream window and nothing before it.
    """
    values = _planted_dataset()["features"]["f1"].to_numpy()
    zed = rolling_z(values, window)
    rng = np.random.default_rng(31)
    for slot in (window, 100, 999, 1500):
        mutated = values.copy()
        tail = mutated[slot + window :]
        assert tail.size > 0
        mutated[slot + window :] = rng.permutation(tail) * 3.7 - 1.1
        zed_mutated = rolling_z(mutated, window)
        assert np.array_equal(
            zed[: slot + window], zed_mutated[: slot + window], equal_nan=True
        )
        # strictly trailing: the prefix alone reproduces every z
        assert np.array_equal(
            zed[: slot + 1],
            rolling_z(values[: slot + 1], window),
            equal_nan=True,
        )
    # teeth: touching values[1000] moves z from 1000 on, never before
    bumped = values.copy()
    bumped[1000] += 5.0
    zed_bumped = rolling_z(bumped, window)
    assert zed_bumped[1000] != zed[1000]
    assert np.array_equal(zed_bumped[:1000], zed[:1000], equal_nan=True)


# ---------------------------------------------------------------------------
# fold-level statistics: permutation symmetry, documented non-causality
# ---------------------------------------------------------------------------
def test_ev_and_threshold_are_permutation_symmetric_fold_stats() -> None:
    """Joint row permutations leave EV and the chosen threshold fixed.

    ``ev_of_taken`` is a mean over taken rows and ``pick_threshold``
    argmaxes per-threshold EVs, so jointly permuting (proba, exec)
    rows cannot change either (the EV means may drift a final-ulp by
    summation order; the threshold, a grid value, cannot move at all
    unless two levels tie to within that dust).
    """
    rng = np.random.default_rng(37)
    proba = np.clip(0.5 + 0.2 * rng.standard_normal(300), 0.01, 0.99)
    exec_vals = 0.01 * rng.standard_normal(300)
    taken = proba > 0.45
    assert int(taken.sum()) > 1
    order = rng.permutation(proba.size)

    ev_base = ev_of_taken(exec_vals, taken)
    ev_perm = ev_of_taken(exec_vals[order], taken[order])
    assert ev_perm == pytest.approx(ev_base, rel=1e-12, abs=1e-15)

    threshold_base = pick_threshold(proba, exec_vals, grid=_GRID)
    threshold_perm = pick_threshold(
        proba[order], exec_vals[order], grid=_GRID
    )
    assert threshold_perm == threshold_base


def test_pick_threshold_is_documented_non_causal_on_truncation() -> None:
    """The threshold consumes the whole train fold, not a causal prefix.

    This pins the intended semantics (train-fold honesty, NOT a time
    stream): truncating the tail of the train fold CAN move the
    selected threshold.  With six mediocre trades followed by six
    strong high-probability trades the full fold picks the grid level
    0.40 (argmax EV, ties to the lowest), while the prefix alone can
    only take trades at 0.30.  The mirror-image invariance that DOES
    hold -- permutations -- lives in the test above.
    """
    proba = np.array([0.35] * 6 + [0.90] * 6)
    exec_vals = np.array([-0.01] * 6 + [0.05] * 6)
    full = pick_threshold(proba, exec_vals, grid=_GRID)
    prefix = pick_threshold(proba[:6], exec_vals[:6], grid=_GRID)
    assert full == 0.40
    assert prefix == 0.30
    assert full != prefix


# ---------------------------------------------------------------------------
# stacking stub: pending module, skip until it lands
# ---------------------------------------------------------------------------
def test_ensemble_report_tail_mutation_spares_earlier_folds() -> None:
    """Full run_ensemble_report: future mutation is invisible upstream.

    The ensemble adapter runs the SAME engine walk-forward, so its
    OOF stacking plus every earlier fold record (thresholds, taken
    masks, EVs of the model and the battery, importances) must be
    bit-identical when every value column is scrambled strictly after
    the first test block.  This extends the engine-level metamorphic
    contract to the composite (the unit-level OOF-honesty invariant
    lives in test_ml_ensemble.py).
    """
    dataset = _planted_dataset(seed=17)
    config = _MINIMAL_CONFIG
    base = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={
            "members": ["gbm", "rf"],
            "combination": "stacking",
            "meta_learner": "logreg",
        },
        config=config,
        battery_controls=dataset["battery_controls"],
    )
    # fold 1's record covers its TRAIN and TEST rows, both of which
    # lie before fold 2's test block -- the mutation starts there
    first_test = int(
        np.flatnonzero(
            assign_folds(
                dataset["times_us"],
                fold_edges(dataset["times_us"], config["k_blocks"]),
            )
            == 2
        )[0]
    )
    mutated = _mutate_from(dataset, first_test, seed=99)
    mutated_run = run_ensemble_report(
        mutated["features"],
        mutated["target"],
        mutated["exec_returns"],
        dataset["times_us"],
        ensemble={
            "members": ["gbm", "rf"],
            "combination": "stacking",
            "meta_learner": "logreg",
        },
        config=config,
        battery_controls=dataset["battery_controls"],
    )
    assert base["folds"][0] == mutated_run["folds"][0]
    assert (
        base["models"]["ensemble"]["importance_by_fold"][0]
        == mutated_run["models"]["ensemble"]["importance_by_fold"][0]
    )
