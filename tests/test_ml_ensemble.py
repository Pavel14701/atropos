"""Primary test layer for the registry ensemble composite.

Covers: :func:`rank_normalize` average-rank ties and exact values
(equal scores share one rank, whatever their input order), the three
algebraic combinations (mean / weighted / rank_mean) against
hand-replicated member probabilities, weight normalization, the
past-only OOF matrix (shape, row indices, the min-train guard and the
CRITICAL honest nesting metamorphic invariant -- mutating a future
tile never changes earlier tiles' OOF scores), ensemble config
validation, the stacking meta-learner (ridge probability fail-closed,
OOF-only training contract, executable meta-vs-in-sample proof), the
registry-registration contract, the full engine path for
``model="ensemble"`` and end-to-end report determinism.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import atropos.ml_ensemble as ml_ensemble_module
import numpy as np
import pandas as pd
import pytest
from atropos.ml_ensemble import (
    DEFAULT_ENSEMBLE_CONFIG,
    StackingMeta,
    build_ensemble,
    normalize_ensemble_config,
    oof_score_matrix,
    rank_normalize,
    run_ensemble_report,
)
from atropos.ml_training import MODEL_REGISTRY, run_ml_kill_protocol

HOUR_US = 3_600_000_000

_ENGINE_REPORT_KEYS = frozenset(
    {
        "verdict",
        "kills",
        "n_events",
        "n_rows",
        "folds",
        "models",
        "baseline",
        "battery",
        "incremental",
        "dsr_detail",
        "ev_model",
        "ev_baseline",
        "ev_battery",
        "deflated_sharpe",
        "importance_stability",
        "powered",
        "n_min_events",
        # additive (P3 close-out): the engine report now echoes the
        # prereg anchor; the ensemble report inherits it verbatim
        "prereg_hash",
        "config",
    }
)
_NAMESPACE_KEYS = frozenset(
    {
        "members",
        "combination",
        "normalize_scores",
        "weights",
        "meta_learner",
        "meta_folds",
        "seed",
        "oof_rows",
        "oof_correlations",
        "meta_coefficients",
    }
)

_ALLOWED_VERDICTS = {
    "SURVIVES",
    "DEAD(incremental)",
    "DEAD(cost_line)",
    "DEAD(importance_stability)",
    "DEAD(deflated_sharpe)",
    "UNDERPOWERED",
}


def _hourly_stamps(n_rows: int) -> np.ndarray:
    """Monotone int64 microsecond stamps one hour apart."""
    return np.arange(n_rows, dtype=np.int64) * HOUR_US


def _planted_dataset(n_rows: int = 700, seed: int = 5) -> dict[str, Any]:
    """Seeded planted-nonlinearity world (own generator).

    No imports from sibling test files: the generator lives here.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_rows, 4))
    controls = rng.standard_normal((n_rows, 2))
    exec_ret = (
        0.012 * np.tanh(feats[:, 0] * feats[:, 1] / 2.0)
        + 0.0035 * feats[:, 2]
        + 0.0024 * rng.standard_normal(n_rows)
    )
    feature_frame = pd.DataFrame(
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
        "features": feature_frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": _hourly_stamps(n_rows),
        "battery_controls": ["ctrl1", "ctrl2"],
    }


def _oof_world(n_rows: int = 240, seed: int = 4) -> tuple[np.ndarray, ...]:
    """Chronological (feats, labels) pair with a learnable structure."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_rows, 3))
    signal = 0.5 * feats[:, 0] + 0.4 * feats[:, 0] * feats[:, 1]
    labels = (signal + rng.standard_normal(n_rows) > 0.0).astype(np.int64)
    return feats, labels


def _member_positive(
    name: str,
    member_kwargs: dict[str, Any],
    seed: int,
    *,
    feats: np.ndarray,
    labels: np.ndarray,
    probe: np.ndarray,
) -> np.ndarray:
    """Replicate the composite's member build: factory(**kwargs, seed)."""
    model = MODEL_REGISTRY[name](**{**member_kwargs, "seed": seed})
    model.fit(feats, labels)
    return np.asarray(model.predict_proba(probe), dtype=np.float64)[:, 1]


# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------
def test_rank_normalize_exact_known_values() -> None:
    """Ascending average ranks in [0, 1]; degenerate inputs to zeros."""
    assert rank_normalize(np.array([3.0, 1.0, 2.0])).tolist() == [
        1.0,
        0.0,
        0.5,
    ]
    assert rank_normalize(np.array([5.0, 5.0, 1.0])).tolist() == [
        0.75,
        0.75,
        0.0,
    ]
    assert rank_normalize(np.array([7.0])).tolist() == [0.0]
    assert rank_normalize(np.array([])).tolist() == []


def test_rank_normalize_ties_share_average_rank() -> None:
    """Ties collapse onto the average of their rank positions.

    Exact multi-group check over seven rows (count - 1 == 6): 0.5
    alone at 1-based rank 1 -> 0.0, 1.0 alone at rank 2 -> 1/6, the
    2.0 pair at average rank 3.5 -> 5/12, the 3.0 triple at average
    rank 6 -> 5/6.
    """
    ranks = rank_normalize(np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 0.5]))
    assert ranks.tolist() == [
        1 / 6,
        5 / 12,
        5 / 12,
        5 / 6,
        5 / 6,
        5 / 6,
        0.0,
    ]


def test_rank_normalize_tie_order_invariance() -> None:
    """Average rank is invariant to the ORDER of equal scores.

    Stronger than the old stable-order contract: permuting equal
    scores leaves the whole rank vector unchanged (equal values map to
    equal ranks under every arrangement), so a rank_mean blend cannot
    depend on how ties arrived.
    """
    rng = np.random.default_rng(11)
    base = np.array([5.0, 3.0, 5.0, 1.0, 5.0, 3.0])
    reference = rank_normalize(base)
    # the tied 5.0 trio -> average 1-based rank 5 -> (5-1)/5 = 0.8;
    # the tied 3.0 pair -> average rank 2.5 -> (2.5-1)/5 = 0.3
    assert reference.tolist() == [0.8, 0.3, 0.8, 0.0, 0.8, 0.3]
    for _ in range(16):
        order = rng.permutation(base.size)
        np.testing.assert_array_equal(
            rank_normalize(base[order]), reference[order]
        )
    # and permuting the WHOLE vector permutes the ranks along with it
    np.testing.assert_array_equal(
        np.sort(rank_normalize(base[rng.permutation(base.size)])),
        np.sort(reference),
    )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_default_ensemble_config_defaults() -> None:
    """Default dict stays mergeable; normalize() accepts it as is.

    Every default member must be a registered registry name.
    """
    assert set(DEFAULT_ENSEMBLE_CONFIG) == {
        "members",
        "weights",
        "combination",
        "normalize_scores",
        "meta_learner",
        "meta_folds",
        "seed",
        "member_params",
    }
    assert DEFAULT_ENSEMBLE_CONFIG["members"] == ("gbm", "rf")
    assert DEFAULT_ENSEMBLE_CONFIG["combination"] == "mean"
    assert DEFAULT_ENSEMBLE_CONFIG["normalize_scores"] is False
    assert DEFAULT_ENSEMBLE_CONFIG["meta_folds"] == 3
    for name in DEFAULT_ENSEMBLE_CONFIG["members"]:
        assert name in MODEL_REGISTRY
    assert normalize_ensemble_config().members == ("gbm", "rf")


def test_normalize_ensemble_config_weights_and_passthrough() -> None:
    """Weights normalize to sum 1; other overrides pass through."""
    config = normalize_ensemble_config(
        {
            "members": ["logit", "gbm"],
            "weights": [4.0, 4.0],
            "combination": "weighted",
            "meta_folds": 5,
        }
    )
    assert config.members == ("logit", "gbm")
    assert config.combination == "weighted"
    assert config.meta_folds == 5
    assert config.normalized_weights() == (0.5, 0.5)
    uniform = normalize_ensemble_config({"members": ["logit", "gbm"]})
    assert uniform.weights is None
    assert uniform.normalized_weights() == (0.5, 0.5)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"members": []}, "empty"),
        ({"members": "gbm"}, "non-empty list"),
        ({"members": ["gbm", 42]}, "must be str"),
        ({"members": ["gbm", "nope"]}, "unknown ensemble member"),
        ({"members": ["gbm", "gbm"]}, "duplicate"),
        ({"weights": [1.0]}, "must match members"),
        ({"weights": [-1.0, 1.0]}, ">= 0"),
        ({"weights": [0.0, 0.0]}, "positive total"),
        ({"weights": "heavy"}, "None or a list"),
        ({"combination": "median"}, "combination"),
        ({"meta_learner": "lasso"}, "meta_learner"),
        ({"combination": "stacking"}, "requires meta_learner"),
        ({"meta_folds": 1}, "meta_folds"),
        ({"meta_folds": 2.5}, "integer"),
        ({"seed": -1}, "seed"),
        ({"normalize_scores": "yes"}, "normalize_scores"),
        ({"member_params": "gbm"}, "member_params"),
        ({"member_params": {"nope": {}}}, "unknown member_params"),
        ({"member_params": {"gbm": 3}}, "dict of factory"),
        ({"bogus_key": 1}, "unknown ensemble config"),
    ],
)
def test_config_validation_raises_value_error(
    overrides: dict[str, Any], match: str
) -> None:
    """Garbage config values raise ValueError naming the offender.

    The raise happens before any model is built.
    """
    with pytest.raises(ValueError, match=match):
        normalize_ensemble_config(overrides)


def test_build_ensemble_merges_model_params_and_seed() -> None:
    """build_ensemble merges member kwargs and honors the seed override.

    The merged result is revalidated, so a bad merge fails loudly.
    """
    base = normalize_ensemble_config({"members": ["gbm", "logit"]})
    estimator = build_ensemble(base, {"gbm": {"n_estimators": 5}}, seed=13)
    assert estimator.config.seed == 13
    assert estimator.config.member_params["gbm"] == {"n_estimators": 5}
    with pytest.raises(ValueError, match="unknown member_params"):
        build_ensemble(base, {"nope": {}})


# ---------------------------------------------------------------------------
# combinations: exact values against hand-replicated members
# ---------------------------------------------------------------------------
def _blend_world() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded (feats, labels, probe) triple for combination checks."""
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((220, 3))
    labels = (feats[:, 0] * feats[:, 1] > 0.0).astype(np.int64)
    probe = rng.standard_normal((60, 3))
    return feats, labels, probe


_BLEND_PARAMS: dict[str, dict[str, Any]] = {
    "gbm": {"n_estimators": 12},
    "logit": {"max_iter": 300},
}


def test_mean_combination_matches_member_mean() -> None:
    """combination='mean' equals the plain average of member probas."""
    feats, labels, probe = _blend_world()
    estimator = build_ensemble(
        {
            "members": ["gbm", "logit"],
            "combination": "mean",
            "normalize_scores": False,
            "seed": 9,
            "member_params": _BLEND_PARAMS,
        }
    ).fit(feats, labels)
    got = estimator.predict_proba(probe)[:, 1]
    gbm = _member_positive(
        "gbm",
        _BLEND_PARAMS["gbm"],
        9,
        feats=feats,
        labels=labels,
        probe=probe,
    )
    logit = _member_positive(
        "logit",
        _BLEND_PARAMS["logit"],
        10,
        feats=feats,
        labels=labels,
        probe=probe,
    )
    expected = np.clip((gbm + logit) / 2.0, 1e-12, 1.0 - 1e-12)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_weighted_combination_uses_normalized_weights() -> None:
    """combination='weighted' rescales weights to sum 1 (exact)."""
    feats, labels, probe = _blend_world()
    estimator = build_ensemble(
        {
            "members": ["gbm", "logit"],
            "combination": "weighted",
            "weights": [3.0, 1.0],
            "normalize_scores": False,
            "seed": 9,
            "member_params": _BLEND_PARAMS,
        }
    ).fit(feats, labels)
    got = estimator.predict_proba(probe)[:, 1]
    gbm = _member_positive(
        "gbm",
        _BLEND_PARAMS["gbm"],
        9,
        feats=feats,
        labels=labels,
        probe=probe,
    )
    logit = _member_positive(
        "logit",
        _BLEND_PARAMS["logit"],
        10,
        feats=feats,
        labels=labels,
        probe=probe,
    )
    expected = np.clip((3.0 * gbm + logit) / 4.0, 1e-12, 1.0 - 1e-12)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_rank_mean_combination_ranks_then_averages() -> None:
    """combination='rank_mean' averages rank-normalized columns.

    normalize_scores is a no-op there: ranks of ranks are ranks.
    """
    feats, labels, probe = _blend_world()
    for normalize in (False, True):
        estimator = build_ensemble(
            {
                "members": ["gbm", "logit"],
                "combination": "rank_mean",
                "normalize_scores": normalize,
                "seed": 9,
                "member_params": _BLEND_PARAMS,
            }
        ).fit(feats, labels)
        got = estimator.predict_proba(probe)[:, 1]
        gbm = rank_normalize(
            _member_positive(
                "gbm",
                _BLEND_PARAMS["gbm"],
                9,
                feats=feats,
                labels=labels,
                probe=probe,
            )
        )
        logit = rank_normalize(
            _member_positive(
                "logit",
                _BLEND_PARAMS["logit"],
                10,
                feats=feats,
                labels=labels,
                probe=probe,
            )
        )
        expected = np.clip((gbm + logit) / 2.0, 1e-12, 1.0 - 1e-12)
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_estimator_sklearn_contract_and_unfitted_guard() -> None:
    """predict_proba is a valid (n, 2) matrix; predict is binary.

    predict before fit raises RuntimeError.
    """
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((150, 3))
    labels = (feats[:, 0] > 0.2).astype(np.int64)
    estimator = build_ensemble(
        {
            "members": ["logit", "gbm"],
            "member_params": {"gbm": {"n_estimators": 12}},
        }
    )
    with pytest.raises(RuntimeError, match="fitted"):
        estimator.predict_proba(feats)
    proba = estimator.fit(feats, labels).predict_proba(feats)
    assert proba.shape == (150, 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(np.unique(estimator.predict(feats)).tolist()) <= {0, 1}


def test_estimator_fit_validates_inputs() -> None:
    """Fit rejects bad inputs with ValueError before any member fit.

    Non-binary labels, non-2-D feats and row mismatches each raise.
    """
    with pytest.raises(ValueError, match="binary 0/1"):
        build_ensemble({"members": ["logit"]}).fit(
            np.zeros((10, 2)), np.full(10, 2)
        )
    with pytest.raises(ValueError, match="2-D"):
        build_ensemble({"members": ["logit"]}).fit(
            np.zeros(10), np.zeros(10, dtype=np.int64)
        )
    with pytest.raises(ValueError, match="match feats rows"):
        build_ensemble({"members": ["logit"]}).fit(
            np.zeros((10, 2)), np.zeros(9, dtype=np.int64)
        )


# ---------------------------------------------------------------------------
# past-only OOF matrix
# ---------------------------------------------------------------------------
def test_oof_matrix_shape_and_row_indices() -> None:
    """Tiles after the first boundary are scored in member column order.

    Row indices are strictly increasing and scores stay in [0, 1].
    """
    feats, labels = _oof_world(120)
    matrix, row_ix = oof_score_matrix(
        feats,
        labels,
        ("logit", "gbm"),
        n_folds=3,
        seed=0,
        member_params={"gbm": {"n_estimators": 8}},
    )
    assert matrix.shape == (80, 2)
    assert row_ix.tolist() == list(range(40, 120))
    assert np.all(np.diff(row_ix) > 0)
    assert np.isfinite(matrix).all()
    assert ((matrix >= 0.0) & (matrix <= 1.0)).all()


def test_oof_honest_nesting_future_mutation_is_invisible() -> None:
    """CRITICAL port invariant: past-only nesting is honest.

    Mutating the features AND labels of a later tile must leave
    earlier tiles' OOF scores bit-identical -- each tile is scored by
    members refitted strictly before it, so the future cannot leak
    into the past.
    """
    feats, labels = _oof_world(240)
    params = {"gbm": {"n_estimators": 10}}
    base_matrix, base_rows = oof_score_matrix(
        feats,
        labels,
        ("gbm", "logit"),
        n_folds=4,
        seed=7,
        member_params=params,
    )
    # bounds: linspace(0, 240, 5) -> tiles [60,120), [120,180), [180,240)
    mutated_x = feats.copy()
    mutated_x[180:] += 7.0
    mutated_y = labels.copy()
    mutated_y[180:] = 1 - mutated_y[180:]
    mut_matrix, mut_rows = oof_score_matrix(
        mutated_x,
        mutated_y,
        ("gbm", "logit"),
        n_folds=4,
        seed=7,
        member_params=params,
    )
    assert np.array_equal(base_rows, mut_rows)
    earlier = base_rows < 180
    assert int(earlier.sum()) == 120
    np.testing.assert_array_equal(base_matrix[earlier], mut_matrix[earlier])


def test_oof_matrix_rejects_bad_inputs() -> None:
    """n_folds/seed/members/shape violations raise ValueError."""
    feats, labels = _oof_world(60)
    with pytest.raises(ValueError, match="n_folds"):
        oof_score_matrix(feats, labels, ("logit",), n_folds=1)
    with pytest.raises(ValueError, match=r"2 \* n_folds"):
        oof_score_matrix(feats[:5], labels[:5], ("logit",), n_folds=3)
    with pytest.raises(ValueError, match="unknown ensemble member"):
        oof_score_matrix(feats, labels, ("nope",), n_folds=3)
    with pytest.raises(ValueError, match="match feats rows"):
        oof_score_matrix(feats, labels[:-1], ("logit",), n_folds=3)


@pytest.mark.parametrize(
    ("n_rows", "n_folds", "works"),
    [(96, 3, True), (95, 3, False), (64, 2, True), (63, 2, False)],
)
def test_oof_matrix_min_train_guard(
    n_rows: int, n_folds: int, works: bool
) -> None:
    """Every tile trains on >= 32 rows: n_rows // n_folds >= 32.

    The guard fires during validation, BEFORE any tile member is built
    or fitted -- a tiny dataset can never leak a memorized tile into a
    stacking meta-learner.  The boundary is exact: 32 rows per tile
    works, one row less raises.
    """
    feats, labels = _oof_world(n_rows)
    if works:
        matrix, row_ix = oof_score_matrix(
            feats, labels, ("logit",), n_folds=n_folds, seed=0
        )
        assert matrix.shape == (n_rows - n_rows // n_folds, 1)
        assert row_ix.size == matrix.shape[0]
        assert ((matrix >= 0.0) & (matrix <= 1.0)).all()
    else:
        with pytest.raises(ValueError, match=r"n_rows // n_folds >= 32"):
            oof_score_matrix(
                feats, labels, ("logit",), n_folds=n_folds, seed=0
            )


def test_oof_matrix_min_train_guard_message_names_remedy() -> None:
    """The guard's message says what to change (folds down, data up)."""
    feats, labels = _oof_world(80)
    with pytest.raises(ValueError, match="reduce n_folds or add data"):
        oof_score_matrix(feats, labels, ("logit",), n_folds=4, seed=0)


def test_stacking_estimator_small_dataset_raises_guard() -> None:
    """A stacking ensemble fit on tiny data fails on the OOF guard.

    Documented in EnsembleEstimator.fit: rows too few for honest
    meta_folds tiles raise before the meta-learner is fitted.
    """
    estimator = build_ensemble(
        {
            "members": ["logit"],
            "combination": "stacking",
            "meta_learner": "ridge",
            "meta_folds": 4,
        }
    )
    with pytest.raises(ValueError, match="n_rows // n_folds >= 32"):
        estimator.fit(
            np.zeros((40, 2)), np.array([0, 1] * 20, dtype=np.int64)
        )


# ---------------------------------------------------------------------------
# stacking meta-learner
# ---------------------------------------------------------------------------
def test_stacking_meta_kinds_fit_predict_deterministic() -> None:
    """Both learners blend rows into (n_rows,) scores deterministically.

    Fixed seeds reproduce bit-identical outputs and coefficients.
    """
    rng = np.random.default_rng(3)
    oof = rng.standard_normal((200, 2))
    signal = 0.6 * oof[:, 0] - 0.4 * oof[:, 1]
    target = (signal + rng.standard_normal(200) > 0.0).astype(np.float64)
    for kind in ("logreg", "ridge"):
        meta = StackingMeta(kind, seed=5)
        first = meta.fit(oof, target).predict(oof)
        assert first.shape == (200,)
        assert meta.coefficients().shape == (2,)
        second = StackingMeta(kind, seed=5).fit(oof, target).predict(oof)
        np.testing.assert_array_equal(first, second)


def test_stacking_meta_rejects_bad_kind_and_guards_unfitted() -> None:
    """Unknown kinds raise at construction; unfitted access raises.

    predict/coefficients before fit raise RuntimeError.  The default
    seed is 42 -- the same documented base seed as
    DEFAULT_ENSEMBLE_CONFIG and the engine's defaults.
    """
    with pytest.raises(ValueError, match="meta_learner"):
        StackingMeta("none")
    with pytest.raises(ValueError, match="seed"):
        StackingMeta("logreg", seed=-1)
    with pytest.raises(RuntimeError, match="fitted"):
        StackingMeta("logreg").predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError, match="fitted"):
        StackingMeta("ridge").coefficients()
    with pytest.raises(ValueError, match="1-D"):
        StackingMeta("ridge").fit(np.zeros((4, 2)), np.zeros((2, 2)))
    assert StackingMeta("ridge").seed == 42
    assert StackingMeta("logreg").seed == 42


def test_stacking_meta_ridge_rejects_non_probability_target() -> None:
    """Ridge stacking fail-closes on targets outside [0, 1].

    Ridge predictions travel the clip path as probabilities, so fit
    must reject targets that would let non-probability numbers out.
    logreg stays untouched: it binarizes any numeric target.
    """
    oof = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(
        ValueError, match=r"ridge stacking requires target in \[0, 1\]"
    ):
        StackingMeta("ridge").fit(oof, np.array([100.0, 200.0, 300.0]))
    with pytest.raises(ValueError, match=r"got \[-0.5, 0.5\]"):
        StackingMeta("ridge").fit(oof, np.array([-0.5, 0.0, 0.5]))
    # a NaN target also fails the range check (comparisons are False)
    with pytest.raises(ValueError, match="ridge stacking requires target in"):
        StackingMeta("ridge").fit(oof, np.array([0.0, float("nan"), 1.0]))
    fitted = StackingMeta("ridge").fit(oof, np.array([0.0, 1.0, 1.0]))
    assert fitted.predict(oof).shape == (3,)
    # logreg keeps its documented binarizing semantics
    logit = StackingMeta("logreg").fit(oof, np.array([-5.0, 3.0, 0.0]))
    assert logit.predict(oof).shape == (3,)


def test_stacking_meta_fit_docstring_warns_oof_required() -> None:
    """StackingMeta.fit must document the OOF input requirement.

    The primitive cannot enforce past-only inputs (any caller can fit
    it on anything), so the docstring carries the warning; the
    executable half of the contract is the meta-vs-in-sample test.
    """
    doc = (inspect.getdoc(StackingMeta.fit) or "").lower()
    assert "out-of-fold" in doc or "oof" in doc


def test_stacking_meta_trains_on_oof_not_in_sample_scores() -> None:
    """Executable "no stacking without nesting" proof (TZ item 8).

    Pure-noise labels leave nothing to generalize, but capacious
    members still memorize them IN SAMPLE -- so a meta reading
    in-sample member scores sees near-perfect separation and grows
    large weights.  The estimator fits its meta on the past-only OOF
    matrix instead, where the same members carry no signal: its
    coefficients stay at noise scale.  Both regimes are asserted with
    a documented gap (ridge coefficients, seed 0 world: honest
    ~0.14 vs broken ~2.1, a ~15x separation).

    Attribute access goes through documented API only: ``meta`` and
    ``member_models`` are EnsembleEstimator attributes, and
    ``StackingMeta.coefficients`` is the public coefficient reader.
    """
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((600, 4))
    labels = rng.integers(0, 2, 600).astype(np.int64)
    estimator = build_ensemble(
        {
            "members": ["gbm", "rf"],
            "combination": "stacking",
            "meta_learner": "ridge",
            "meta_folds": 3,
            "seed": 0,
            "member_params": {
                "gbm": {"n_estimators": 80, "max_depth": 5},
                "rf": {"n_estimators": 80, "max_depth": 8},
            },
        }
    ).fit(feats, labels)
    assert estimator.meta is not None  # documented stacking attribute
    honest_max = float(np.abs(estimator.meta.coefficients()).max())
    # the broken counterfactual: the SAME members (already fitted on
    # these rows), meta reading their IN-SAMPLE scores instead
    stacked_in_sample = np.column_stack(
        [
            np.asarray(model.predict_proba(feats), dtype=np.float64)[:, 1]
            for _, model in estimator.member_models
        ]
    )
    broken = StackingMeta("ridge", seed=0).fit(
        stacked_in_sample, labels.astype(np.float64)
    )
    broken_max = float(np.abs(broken.coefficients()).max())
    # threshold gap picked from the observed regimes (seed 0: 0.139
    # vs 2.132); the floors sit well inside each regime so only a
    # nesting break (in-sample scores reaching the meta) or a garbage
    # OOF builder can cross them
    assert honest_max < 0.5, f"meta weight looks in-sample: {honest_max}"
    assert broken_max > 1.0, (
        "in-sample counterfactual did not separate; threshold stale"
    )
    assert broken_max > 5.0 * honest_max


# ---------------------------------------------------------------------------
# registry contract and the full engine path
# ---------------------------------------------------------------------------
def test_import_registers_composite_in_registry() -> None:
    """Importing atropos.ml_ensemble registered the 'ensemble' entry.

    The adapter honours the shared factory contract.  Identity is read
    through the MODULE attribute, not the from-import binding:
    ``importlib.reload(atropos.ml_ensemble)`` (exercised in-suite by
    test_ml_training.py::test_ensemble_params_reload_does_not_overwrite)
    re-executes the ``def`` statement and rebinds module-level
    functions, so a from-import name would compare against a stale
    object while the registry holds the current one.
    """
    assert MODEL_REGISTRY["ensemble"] is ml_ensemble_module.make_ensemble
    feats, labels = _oof_world(120)
    model = MODEL_REGISTRY["ensemble"](
        members=("logit", "gbm"),
        member_params={"gbm": {"n_estimators": 10}},
    )
    proba = model.fit(feats, labels).predict_proba(feats)
    assert proba.shape == (120, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_ensemble_runs_through_engine_via_registry() -> None:
    """model='ensemble' with flat model_params completes the engine path.

    This is the config mapping documented in the module docstring.
    """
    dataset = _planted_dataset(n_rows=700, seed=6)
    report = run_ml_kill_protocol(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        config={
            "model": "ensemble",
            "k_blocks": 3,
            "bootstrap_b": 50,
            "n_min_events": 5,
            "checks": {"deflated_sharpe": False},
            "model_params": {
                "ensemble": {
                    "members": ["logit", "gbm"],
                    "combination": "mean",
                    "member_params": {"gbm": {"n_estimators": 15}},
                }
            },
        },
        battery_controls=dataset["battery_controls"],
    )
    assert "ensemble" in report["models"]
    assert report["models"]["ensemble"]["verdict"] in _ALLOWED_VERDICTS
    assert report["verdict"] in _ALLOWED_VERDICTS


def test_run_ensemble_report_adds_ensemble_namespace() -> None:
    """The report keeps every engine key; adds one 'ensemble' namespace.

    The namespace carries exactly the documented diagnostic keys.
    """
    dataset = _planted_dataset(n_rows=700, seed=10)
    report = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={
            "members": ["logit", "gbm"],
            "combination": "rank_mean",
            "member_params": {"gbm": {"n_estimators": 15}},
        },
        config={
            "k_blocks": 3,
            "bootstrap_b": 50,
            "n_min_events": 5,
            "checks": {"deflated_sharpe": False},
        },
        battery_controls=dataset["battery_controls"],
    )
    assert frozenset(report) == _ENGINE_REPORT_KEYS | {"ensemble"}
    assert frozenset(report["ensemble"]) == _NAMESPACE_KEYS
    namespace = report["ensemble"]
    assert namespace["members"] == ["logit", "gbm"]
    assert frozenset(namespace["weights"]) == {"logit", "gbm"}
    assert math.isclose(sum(namespace["weights"].values()), 1.0)
    assert "logit|gbm" in namespace["oof_correlations"]
    assert namespace["oof_rows"] > 0
    assert namespace["meta_coefficients"] is None
    assert "ensemble" in report["models"]


def test_stacking_ensemble_full_engine_path() -> None:
    """combination='stacking' rides the engine path end to end.

    The report carries the fitted meta coefficients and the
    coefficient-share weights.
    """
    dataset = _planted_dataset(n_rows=800, seed=8)
    report = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={
            "members": ["logit", "gbm"],
            "combination": "stacking",
            "meta_learner": "logreg",
            "meta_folds": 3,
            "member_params": {"gbm": {"n_estimators": 15}},
        },
        config={
            "k_blocks": 3,
            "bootstrap_b": 50,
            "n_min_events": 5,
            "checks": {"deflated_sharpe": False},
        },
        battery_controls=dataset["battery_controls"],
    )
    assert report["models"]["ensemble"]["verdict"] in _ALLOWED_VERDICTS
    namespace = report["ensemble"]
    assert namespace["meta_coefficients"] is not None
    assert frozenset(namespace["meta_coefficients"]) == {"logit", "gbm"}
    assert math.isclose(sum(namespace["weights"].values()), 1.0)


def test_ensemble_seed_defaults_to_engine_model_seed() -> None:
    """The ensemble seed defaults to the engine's model_seed.

    An explicit ensemble seed wins over model_seed.
    """
    dataset = _planted_dataset(n_rows=500, seed=12)
    fast_config: dict[str, Any] = {
        "k_blocks": 3,
        "bootstrap_b": 30,
        "n_min_events": 5,
        "checks": {
            "incremental": False,
            "importance_stability": False,
            "deflated_sharpe": False,
        },
        "model_seed": 7,
    }
    inherited = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={"members": ["logit"]},
        config=fast_config,
    )
    assert inherited["ensemble"]["seed"] == 7
    assert inherited["config"]["model_params"]["ensemble"]["seed"] == 7
    explicit = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={"members": ["logit"], "seed": 11},
        config=fast_config,
    )
    assert explicit["ensemble"]["seed"] == 11


def test_run_ensemble_report_rejects_bad_key_before_engine() -> None:
    """An invalid ensemble key fails loudly BEFORE the engine runs.

    Validation happens on the merged ensemble config first, so no
    walk-forward fold is ever fitted for a config that cannot build.
    """
    dataset = _planted_dataset(n_rows=300, seed=3)
    with pytest.raises(ValueError, match="unknown ensemble config keys"):
        run_ensemble_report(
            dataset["features"],
            dataset["target"],
            dataset["exec_returns"],
            dataset["times_us"],
            ensemble={"combination_typo": "rank_mean"},
            config={
                "k_blocks": 3,
                "bootstrap_b": 30,
                "n_min_events": 5,
                "checks": {"deflated_sharpe": False},
            },
        )


def test_run_ensemble_report_engine_receives_normalized_ensemble() -> None:
    """The engine gets the VALIDATED ensemble view, not a raw merge.

    The ``model_params["ensemble"]`` echo in the report config is the
    normalized ``asdict`` of the validated config: member_params merge
    key-wise (the untouched default member entry survives), weights
    and members are the validated tuples, and the seed echo matches
    the diagnostics namespace.
    """
    dataset = _planted_dataset(n_rows=500, seed=12)
    report = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble={
            "members": ["logit"],
            "weights": [3.0],
            "member_params": {"gbm": {"n_estimators": 5}},
        },
        config={
            "k_blocks": 3,
            "bootstrap_b": 30,
            "n_min_events": 5,
            "checks": {
                "incremental": False,
                "importance_stability": False,
                "deflated_sharpe": False,
            },
        },
    )
    assert "ensemble" in report["models"]
    echo = report["config"]["model_params"]["ensemble"]
    assert echo["members"] == ("logit",)
    assert echo["weights"] == (3.0,)
    # key-wise member_params merge: the override replaced the gbm
    # entry wholesale and the untouched rf default survived
    assert echo["member_params"]["gbm"] == {"n_estimators": 5}
    assert (
        echo["member_params"]["rf"]
        == DEFAULT_ENSEMBLE_CONFIG["member_params"]["rf"]
    )
    assert echo["seed"] == report["ensemble"]["seed"]
    assert echo["meta_folds"] == report["ensemble"]["meta_folds"]
    assert echo["combination"] == report["ensemble"]["combination"]


def test_run_ensemble_report_is_deterministic() -> None:
    """Two identical runs produce bit-identical full reports.

    Diagnostics included, weights and OOF correlations included.
    """
    dataset = _planted_dataset(n_rows=600, seed=12)
    ensemble_cfg: dict[str, Any] = {
        "members": ["logit", "gbm"],
        "combination": "rank_mean",
        "member_params": {"gbm": {"n_estimators": 15}},
    }
    engine_cfg: dict[str, Any] = {
        "k_blocks": 3,
        "bootstrap_b": 50,
        "n_min_events": 5,
        "checks": {"deflated_sharpe": False},
    }
    first = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble=ensemble_cfg,
        config=engine_cfg,
        battery_controls=dataset["battery_controls"],
    )
    second = run_ensemble_report(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        ensemble=ensemble_cfg,
        config=engine_cfg,
        battery_controls=dataset["battery_controls"],
    )
    assert first == second
