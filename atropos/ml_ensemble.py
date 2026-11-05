"""Composite ensemble models over the open :data:`MODEL_REGISTRY`.

    Port of the source protocol's ``engine.ensemble`` package (base /
    combine / meta): :func:`rank_normalize` (average-rank score
    normalization, ties share their mean position), the
score combinations ``mean`` / ``weighted`` / ``rank_mean`` /
``stacking``, the past-only out-of-fold score matrix and the linear
stacking meta-learner -- rebuilt over registry factories instead of
the protocol's private component classes.

Two nesting levels keep stacking honest (source TZ item 8: *no
stacking without nesting*): the walk-forward engine
:func:`atropos.ml_training.run_ml_kill_protocol` is the OUTER level
(every CV fold fits fresh members on strictly earlier rows), and
:func:`oof_score_matrix` is the INNER one (the meta-learner trains on
out-of-fold member scores produced by members fitted strictly before
each time tile).  A meta-learner fitted on in-sample member scores
would read overfit scores and learn garbage weights -- never stack
outside both nestings.

**Import side effect.** Importing this module registers the composite
in the engine's open registry::

    MODEL_REGISTRY["ensemble"] = make_ensemble

and seeds ``DEFAULT_TRAINING_CONFIG["model_params"]["ensemble"]`` with
the default member list, keeping the engine's bundled-defaults
invariant true (every registry name ships a ``model_params`` entry;
``setdefault`` never overwrites an existing one).  ``model=
"ensemble"`` (or ``models=[..., "ensemble"]``) immediately works
through the full engine path (walk-forward, thresholds, EV, all four
gates) with no engine changes.  The registry contract is the only
dependency on :mod:`atropos.ml_training`: every member is built by its
factory as ``factory(**member_params[name], seed=...)``.

**Engine config mapping.** The composite is configured through flat
factory kwargs taken from ``model_params["ensemble"]``::

    config = {
        "model": "ensemble",
        "model_params": {"ensemble": {
            "members": ["gbm", "rf"],      # registry names, in order
            "weights": [2.0, 1.0],         # optional, any scale
            "combination": "rank_mean",    # mean|weighted|rank_mean|stacking
            "normalize_scores": True,      # rank-normalize member scores
            "meta_learner": "none",        # logreg|ridge|none
            "meta_folds": 3,               # past-only OOF tiles
            "member_params": {"gbm": {"n_estimators": 60}},
        }},
    }

Every key is optional (defaults in :data:`DEFAULT_ENSEMBLE_CONFIG`);
unknown keys and unknown member names fail validation loudly.  Member
``i`` is built with seed ``seed + i`` (documented determinism).

Run ``python -m atropos.ml_ensemble`` for the synthetic verification
demo: a planted nonlinear edge through the full engine with a
``["gbm", "rf"]`` rank-mean ensemble, honest OOF shape checks and a
two-run bit-identical determinism assertion.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

from atropos._utils import validate_int, validate_number
from atropos.ml_training import (
    DEFAULT_TRAINING_CONFIG,
    MODEL_REGISTRY,
    run_ml_kill_protocol,
)

__all__ = [
    "DEFAULT_ENSEMBLE_CONFIG",
    "EnsembleConfig",
    "EnsembleEstimator",
    "StackingMeta",
    "build_ensemble",
    "make_ensemble",
    "normalize_ensemble_config",
    "oof_score_matrix",
    "rank_normalize",
    "run_ensemble_report",
]

Combination = Literal["mean", "weighted", "rank_mean", "stacking"]
MetaLearner = Literal["logreg", "ridge", "none"]

_COMBINATIONS: tuple[str, ...] = (
    "mean",
    "weighted",
    "rank_mean",
    "stacking",
)
_META_LEARNERS: tuple[str, ...] = ("logreg", "ridge", "none")
_PROBA_EPS = 1e-12
_HOUR_US = 3_600_000_000
_MIN_TRAIN_FOR_TILES = 32
"""Minimum rows in the first scored OOF tile's training window.

The earliest scored tile of :func:`oof_score_matrix` trains its
members on ``n_rows // n_folds`` rows.  Below this floor the members
effectively memorize that window, so their "out-of-fold" scores carry
no honest generalization signal and a meta-learner fitted on them
learns noise that then flows into every stacked blend -- at shapes
the old ``n_rows >= 2 * n_folds`` check happily accepted.
"""

DEFAULT_ENSEMBLE_CONFIG: dict[str, Any] = {
    "members": ("gbm", "rf"),
    "weights": None,
    "combination": "mean",
    "normalize_scores": False,
    "meta_learner": "none",
    "meta_folds": 3,
    "seed": 42,
    "member_params": {
        "gbm": {"n_estimators": 300, "max_depth": 3},
        "rf": {"n_estimators": 300, "max_depth": 7, "min_samples_leaf": 5},
    },
}
"""Default ensemble config (dict form, house style).

Mirrors :data:`atropos.ml_training.DEFAULT_TRAINING_CONFIG`: a plain
dict that merges key-wise with user overrides, validates loudly and
echoes JSON-safe into reports.  The frozen :class:`EnsembleConfig`
dataclass is the typed snapshot consumers read after validation; the
dict stays the public configuration surface.

The default blend is the plain ``mean`` of member PROBABILITIES (no
rank normalization): the engine's threshold/EV path reads scores on
the probability scale, and rank normalization flattens any blend onto
a uniform distribution -- selective thresholds then take a fixed ~30%
of rows regardless of member sharpness (pinned by the engine's
registry contract tests).  ``rank_mean`` / ``weighted`` stay first
class for rank-mixing use cases; the module demo uses ``rank_mean``.
"""


# ---------------------------------------------------------------------------
# score mechanics (deterministic primitives)
# ---------------------------------------------------------------------------
def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map scores to average-rank positions in ``[0, 1]``.

    Rank space makes heterogeneous score scales comparable before
    blending (tree probabilities and logistic margins live on different
    axes).  Ties share the AVERAGE of their rank positions
    (``scipy.stats.rankdata`` with ``method="average"``): two rows
    scored identically by a member must also blend identically --
    distinct sequential ranks would tilt the blend toward whichever
    tied row happened to sort first.  ``rankdata`` is a deterministic
    function of the value vector alone (equal scores map to equal
    ranks whatever their input order), so the mapping stays inside the
    determinism gate.

    Args:
        scores: 1-D array-like of scores.

    Returns:
        1-D float array of normalized average ranks (higher score =
        higher rank); arrays shorter than two entries map to all
        zeros.

    Example::

        >>> rank_normalize(np.array([3.0, 1.0, 2.0])).tolist()
        [1.0, 0.0, 0.5]
        >>> rank_normalize(np.array([5.0, 5.0, 1.0])).tolist()
        [0.75, 0.75, 0.0]
        >>> rank_normalize(np.array([7.0])).tolist()
        [0.0]
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    count = flat.size
    if count < 2:
        return np.zeros_like(flat)
    ranks = np.asarray(rankdata(flat, method="average"), dtype=np.float64)
    return (ranks - 1.0) / float(count - 1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow warnings)."""
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    odds = np.exp(values[~positive])
    out[~positive] = odds / (1.0 + odds)
    return out


# ---------------------------------------------------------------------------
# configuration: dict surface + frozen dataclass snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleConfig:
    """Validated ensemble blueprint (built by normalize_..._config).

    The dict form (:data:`DEFAULT_ENSEMBLE_CONFIG` + overrides) is the
    public configuration surface -- house style: dict configs merge
    key-wise, validate loudly and echo JSON-safe into reports.  This
    frozen dataclass is the internal typed snapshot every consumer
    reads after :func:`normalize_ensemble_config` has accepted it.

    Attributes:
        members: registry names in build/blend order (unique).
        weights: raw blend weights aligned with ``members``; ``None``
            means uniform.
        combination: score combination rule.
        normalize_scores: rank-normalize member scores before the
            combination step.
        meta_learner: stacking meta-learner kind (unused unless
            ``combination="stacking"``).
        meta_folds: number of past-only OOF tiles for stacking.
        seed: base seed; member ``i`` is built with ``seed + i``.
        member_params: per-member factory kwargs keyed by registry
            name (entries for unselected members are ignored).

    Example::

        >>> EnsembleConfig(members=("gbm",)).members
        ('gbm',)
    """

    members: tuple[str, ...] = ("gbm", "rf")
    weights: tuple[float, ...] | None = None
    combination: Combination = "mean"
    normalize_scores: bool = False
    meta_learner: MetaLearner = "none"
    meta_folds: int = 3
    seed: int = 42
    member_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def normalized_weights(self) -> tuple[float, ...]:
        """Blend weights scaled to sum 1 (uniform when ``None``)."""
        if self.weights is None:
            share = 1.0 / len(self.members)
            return tuple(share for _ in self.members)
        total = float(sum(self.weights))
        return tuple(weight / total for weight in self.weights)


def _validate_members(value: Any) -> tuple[str, ...]:
    """Validate the members list; return it as a unique name tuple.

    Raises:
        ValueError: on a non-list, empty list, non-string entry,
            unknown registry name or duplicate member.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(
            "members must be a non-empty list or tuple of registry "
            f"names, got {value!r}"
        )
    if len(value) == 0:
        raise ValueError("members must not be empty")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"ensemble member names must be str, got {item!r}"
            )
        if item not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown ensemble member {item!r}: available "
                f"{sorted(MODEL_REGISTRY)}"
            )
        if item in names:
            raise ValueError(
                f"duplicate ensemble member {item!r}: members must be unique"
            )
        names.append(item)
    return tuple(names)


def _validate_weights(
    value: Any, members: tuple[str, ...]
) -> tuple[float, ...] | None:
    """Validate raw weights (None passes through as uniform).

    Raises:
        ValueError: on a non-sequence, wrong length, non-number entry,
            negative entry or non-positive total.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "weights must be None or a list/tuple of numbers aligned "
            f"with members, got {value!r}"
        )
    if len(value) != len(members):
        raise ValueError(
            f"weights length {len(value)} must match members "
            f"({len(members)}): {list(members)}"
        )
    values = tuple(
        validate_number(f"weight[{slot}]", weight)
        for slot, weight in enumerate(value)
    )
    if any(weight < 0.0 for weight in values):
        raise ValueError(f"weights must be >= 0, got {list(values)}")
    if sum(values) <= 0.0:
        raise ValueError(
            f"weights must sum to a positive total, got {list(values)}"
        )
    return values


def _validate_member_params(value: Any) -> dict[str, dict[str, Any]]:
    """Validate per-member factory kwargs (registry-name keys).

    Raises:
        ValueError: on a non-dict config or a non-dict kwargs entry;
            keys that name no registry model.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "member_params must map registry names to dicts of factory "
            f"kwargs, got {value!r}"
        )
    for name, entry in value.items():
        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown member_params key {name!r}: available "
                f"{sorted(MODEL_REGISTRY)}"
            )
        if not isinstance(entry, dict):
            raise ValueError(
                f"member_params[{name!r}] must be a dict of factory "
                f"kwargs, got {entry!r}"
            )
    return {name: dict(entry) for name, entry in value.items()}


def normalize_ensemble_config(
    overrides: dict[str, Any] | None = None,
) -> EnsembleConfig:
    """Merge overrides over :data:`DEFAULT_ENSEMBLE_CONFIG` and validate.

    House-style dict configuration: unknown keys, unknown members,
    mismatched weights and bad meta settings raise ``ValueError``
    naming the offender, before any model is built.  Nested
    ``member_params`` dicts update key-wise per member name (a
    per-member dict replaces the default one wholesale, like the
    engine's ``model_params`` merge).

    Args:
        overrides: ensemble config overrides (see
            :data:`DEFAULT_ENSEMBLE_CONFIG` keys); ``None`` takes the
            defaults.

    Returns:
        The frozen validated :class:`EnsembleConfig`.

    Raises:
        ValueError: on any unknown key or contract violation.

    Example::

        >>> config = normalize_ensemble_config(
        ...     {"members": ["logit", "gbm"], "weights": [3.0, 1.0]}
        ... )
        >>> (config.members, config.combination)
        (('logit', 'gbm'), 'mean')
        >>> config.normalized_weights()
        (0.75, 0.25)
        >>> normalize_ensemble_config({"members": []})
        Traceback (most recent call last):
            ...
        ValueError: members must not be empty
    """
    merged: dict[str, Any] = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in DEFAULT_ENSEMBLE_CONFIG.items()
    }
    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_ENSEMBLE_CONFIG))
        if unknown:
            raise ValueError(
                f"unknown ensemble config keys {unknown}: available "
                f"{sorted(DEFAULT_ENSEMBLE_CONFIG)}"
            )
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    members = _validate_members(merged["members"])
    combination = merged["combination"]
    if combination not in _COMBINATIONS:
        raise ValueError(
            f"combination must be one of {_COMBINATIONS}, got {combination!r}"
        )
    meta_learner = merged["meta_learner"]
    if meta_learner not in _META_LEARNERS:
        raise ValueError(
            f"meta_learner must be one of {_META_LEARNERS}, got "
            f"{meta_learner!r}"
        )
    if combination == "stacking" and meta_learner == "none":
        raise ValueError(
            "combination 'stacking' requires meta_learner 'logreg' or 'ridge'"
        )
    normalize_scores = merged["normalize_scores"]
    if not isinstance(normalize_scores, bool):
        raise ValueError(
            f"normalize_scores must be a bool, got {normalize_scores!r}"
        )
    return EnsembleConfig(
        members=members,
        weights=_validate_weights(merged["weights"], members),
        combination=combination,
        normalize_scores=normalize_scores,
        meta_learner=meta_learner,
        meta_folds=validate_int(
            "meta_folds", merged["meta_folds"], minimum=2
        ),
        seed=validate_int("seed", merged["seed"], minimum=0),
        member_params=_validate_member_params(merged["member_params"]),
    )


# ---------------------------------------------------------------------------
# stacking meta-learner and the past-only OOF matrix
# ---------------------------------------------------------------------------
class StackingMeta:
    """Linear stacking meta-learner over per-member score columns.

    Port of the source protocol's ``StackingMeta``: ``logreg`` fits an
    L2 logistic regression on the binarized target ``(y > 0)`` and
    blends through ``decision_function`` (logit margins); ``ridge``
    fits ridge regression on the raw numeric target and blends by
    prediction -- so a ridge target MUST lie in ``[0, 1]`` for its
    predictions to stay valid probabilities (``fit`` rejects anything
    else, fail-closed).  Both are deterministic for a fixed seed; the
    default seed 42 matches :data:`DEFAULT_ENSEMBLE_CONFIG`.

    Example::

        >>> meta = StackingMeta("ridge", seed=0)
        >>> oof = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> targets = np.array([0.0, 1.0, 1.0])
        >>> meta.fit(oof, targets).predict(
        ...     np.array([[0.2, 0.8]])
        ... ).shape
        (1,)
    """

    def __init__(self, kind: str = "logreg", seed: int = 42) -> None:
        """Store the learner kind and seed (nothing is fitted yet).

        The seed defaults to 42 -- the same base seed as
        :data:`DEFAULT_ENSEMBLE_CONFIG` and the engine's
        ``DEFAULT_TRAINING_CONFIG``, so all levels of the stack draw
        from one documented default.

        Raises:
            ValueError: if ``kind`` is neither ``"logreg"`` nor
                ``"ridge"``, or ``seed`` is not an integer >= 0.
        """
        if kind not in ("logreg", "ridge"):
            raise ValueError(
                f"meta_learner must be 'logreg' or 'ridge', got {kind!r}"
            )
        self.kind = kind
        self.seed = validate_int("seed", seed, minimum=0)
        self.model: LogisticRegression | Ridge | None = None

    def fit(self, oof: Any, target: Any) -> StackingMeta:
        """Fit on an OOF score matrix ``(n_rows, n_members)``.

        WARNING: the training scores MUST be out-of-fold (build them
        with :func:`oof_score_matrix`).  This primitive cannot verify
        the past-only contract itself -- that is the caller's
        responsibility -- and fitting it on in-sample member scores
        hands the meta-learner overfit inputs, poisons its weights and
        breaks the no-stacking-without-nesting invariant.

        Args:
            oof: out-of-fold member scores, one column per member.
            target: numeric target of the scored rows (``logreg``
                binarizes it as ``> 0``; ``ridge`` requires values in
                ``[0, 1]`` so its predictions stay on the probability
                scale after the engine's clip path).

        Returns:
            Itself (fitted), sklearn-style.

        Raises:
            ValueError: on shape mismatches, non-finite scores, or a
                ``ridge`` target outside ``[0, 1]``.
        """
        matrix = _as_score_matrix("oof", oof)
        labels = np.asarray(target, dtype=np.float64)
        if labels.ndim != 1 or labels.size != matrix.shape[0]:
            raise ValueError(
                f"target must be 1-D and match oof rows "
                f"({matrix.shape[0]}), got ndim={labels.ndim} "
                f"size={labels.size}"
            )
        if self.kind == "logreg":
            self.model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.seed
            )
            self.model.fit(matrix, (labels > 0.0).astype(np.int64))
        else:
            low = float(np.min(labels))
            high = float(np.max(labels))
            if not (low >= 0.0 and high <= 1.0):
                raise ValueError(
                    "ridge stacking requires target in [0, 1] for "
                    f"probability output, got [{low}, {high}]"
                )
            self.model = Ridge(alpha=1.0)
            self.model.fit(matrix, labels)
        return self

    def predict(self, scores: Any) -> np.ndarray:
        """Blend rows of per-member scores into combined scores.

        Returns:
            ``logreg``: logit margins (``decision_function``);
            ``ridge``: regression predictions.  One value per row.

        Raises:
            RuntimeError: if called before :meth:`fit`.
            ValueError: on a non-2-D or non-finite input.
        """
        if self.model is None:
            raise RuntimeError("StackingMeta is not fitted")
        matrix = _as_score_matrix("scores", scores)
        if self.kind == "logreg":
            return np.asarray(
                self.model.decision_function(matrix), dtype=np.float64
            )
        return np.asarray(self.model.predict(matrix), dtype=np.float64)

    def coefficients(self) -> np.ndarray:
        """Fitted per-member coefficients (1-D, member order).

        Raises:
            RuntimeError: if called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError("StackingMeta is not fitted")
        return np.asarray(self.model.coef_, dtype=np.float64).ravel()


def _as_score_matrix(name: str, values: Any) -> np.ndarray:
    """Coerce to a finite 2-D float matrix, rejecting other shapes.

    Raises:
        ValueError: on non-2-D input or NaN/inf entries.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_rows, n_members), got ndim={arr.ndim}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite floats")
    return arr


def _as_binary_labels(name: str, values: Any) -> np.ndarray:
    """Coerce to int64 0/1 labels, rejecting anything else.

    Raises:
        ValueError: on non-1-D input or labels outside {0, 1}.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={arr.ndim}")
    if not np.isin(arr, (0, 1)).all():
        found = sorted(np.unique(arr).tolist())
        raise ValueError(f"{name} must be binary 0/1, got {found}")
    return arr.astype(np.int64)


def _build_member(
    name: str, member_params: dict[str, dict[str, Any]], seed: int
) -> Any:
    """Build one registry member: ``factory(**kwargs, seed=seed)``.

    Raises:
        ValueError: if ``name`` is not in :data:`MODEL_REGISTRY`.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown ensemble member {name!r}: available "
            f"{sorted(MODEL_REGISTRY)}"
        )
    kwargs = dict(member_params.get(name, {}))
    kwargs.setdefault("seed", seed)
    return MODEL_REGISTRY[name](**kwargs)


def oof_score_matrix(
    feats: Any,
    labels: Any,
    members: Any,
    *,
    n_folds: int = 3,
    seed: int = 42,
    member_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Past-only out-of-fold member scores, ``(matrix, row_ix)``.

    Port of the source protocol's time-tiling OOF builder, adapted to
    the engine's row axis: rows are cut into ``n_folds`` equal tiles BY
    POSITION and every tile is scored by fresh copies of every member
    fitted strictly on the rows BEFORE the tile starts.  ``feats`` MUST
    be chronological -- the walk-forward engine guarantees it (its
    folds hand over train blocks in time order), which makes position
    tiles honest time tiles.

    WARNING (source TZ item 8 -- no stacking without nesting): these
    scores are the ONLY honest input for a stacking meta-learner.  A
    member's scores on its own training rows are overfit, and a
    meta-learner fitted on them learns garbage.  The engine's
    walk-forward is the outer nesting level; this matrix is the inner
    one -- never build a stacked blend outside both.

    Args:
        feats: chronological ``(n_rows, n_features)`` float matrix.
        labels: binary 0/1 labels of the same rows.
        members: non-empty sequence of unique registry names.
        n_folds: number of time tiles, integer >= 2.
        seed: base seed; member ``i`` of every tile is rebuilt with
            ``seed + i`` (identical across tiles, so tiles differ only
            through their training windows).
        member_params: per-member factory kwargs.

    Returns:
        ``(matrix, row_ix)``: the ``(n_scored, n_members)`` OOF score
        matrix and the strictly increasing row indices scored --
        everything after the first tile boundary (the earliest tile
        has no honest past training window).

    Raises:
        ValueError: on invalid members / ``n_folds`` / ``seed``, a
            shape-contract violation, fewer than ``2 * n_folds`` rows
            (which would leave a tile or training window empty), or a
            per-tile training window under ``_MIN_TRAIN_FOR_TILES``
            rows (``n_rows // n_folds < 32``: tiles would fit
            near-memorized members and leak garbage into the
            meta-learner).

    Example::

        >>> feats = np.linspace(-1.0, 1.0, 96).reshape(-1, 1)
        >>> labels = np.tile(np.array([0, 1]), 48)
        >>> matrix, row_ix = oof_score_matrix(
        ...     feats, labels, ("logit",), n_folds=3, seed=0
        ... )
        >>> (matrix.shape, int(row_ix[0]), int(row_ix[-1]))
        ((64, 1), 32, 95)
    """
    member_tuple = _validate_members(members)
    n_folds = validate_int("n_folds", n_folds, minimum=2)
    seed = validate_int("seed", seed, minimum=0)
    params = _validate_member_params(member_params)
    matrix_x = _as_score_matrix("feats", feats)
    targets = _as_binary_labels("labels", labels)
    if matrix_x.shape[0] != targets.size:
        raise ValueError(
            f"labels must match feats rows ({matrix_x.shape[0]}), got "
            f"{targets.size}"
        )
    n_rows = int(targets.size)
    if n_rows < 2 * n_folds:
        raise ValueError(
            f"stacking needs n_rows >= 2 * n_folds ({2 * n_folds}), got "
            f"{n_rows}"
        )
    if n_rows // n_folds < _MIN_TRAIN_FOR_TILES:
        raise ValueError(
            f"stacking needs n_rows // n_folds >= "
            f"{_MIN_TRAIN_FOR_TILES} (got {n_rows // n_folds}); "
            f"reduce n_folds or add data"
        )
    bounds = np.linspace(0, n_rows, n_folds + 1).astype(int)
    per_member: list[list[np.ndarray]] = [[] for _ in member_tuple]
    row_blocks: list[np.ndarray] = []
    for tile in range(1, n_folds):
        low = int(bounds[tile])
        high = int(bounds[tile + 1])
        row_blocks.append(np.arange(low, high, dtype=np.int64))
        for slot, name in enumerate(member_tuple):
            model = _build_member(name, params, seed + slot)
            model.fit(matrix_x[:low], targets[:low])
            positive = np.asarray(
                model.predict_proba(matrix_x[low:high]), dtype=np.float64
            )[:, 1]
            per_member[slot].append(positive)
    matrix = np.column_stack(
        [np.concatenate(blocks) for blocks in per_member]
    )
    return matrix, np.concatenate(row_blocks)


# ---------------------------------------------------------------------------
# the composite estimator and its factories
# ---------------------------------------------------------------------------
def _combine_scores(
    config: EnsembleConfig,
    stacked: np.ndarray,
    meta: StackingMeta | None,
) -> np.ndarray:
    """Apply the configured combination rule to member score columns."""
    if config.combination == "mean":
        return stacked.mean(axis=1)
    if config.combination == "weighted":
        weights = np.asarray(config.normalized_weights(), dtype=np.float64)
        return stacked @ weights
    if config.combination == "rank_mean":
        ranked = np.column_stack(
            [
                rank_normalize(stacked[:, slot])
                for slot in range(stacked.shape[1])
            ]
        )
        return ranked.mean(axis=1)
    if meta is None:  # pragma: no cover - guarded by fit()
        raise RuntimeError("stacking meta is not fitted")
    return meta.predict(stacked)


def _squash_to_probability(
    combined: np.ndarray, config: EnsembleConfig
) -> np.ndarray:
    """Map combined scores to valid positive-class probabilities.

    Averages of probabilities/ranks and ridge stacking predictions are
    clipped into ``[0, 1]``; logreg stacking margins pass through the
    logistic function.  A tiny epsilon clip keeps log-loss finite on
    saturated predictions.
    """
    if config.combination == "stacking" and config.meta_learner == "logreg":
        values = _sigmoid(combined)
    else:
        values = np.asarray(combined, dtype=np.float64).clip(0.0, 1.0)
    return values.clip(_PROBA_EPS, 1.0 - _PROBA_EPS)


class EnsembleEstimator:
    """sklearn-style composite over registry members (the "ensemble").

    Fits every member of :data:`MODEL_REGISTRY` named in the config on
    the same rows, then blends their positive-class scores by the
    configured rule (``mean`` / ``weighted`` / ``rank_mean`` /
    ``stacking``) into ``predict_proba`` / ``predict``.  Built by
    :func:`build_ensemble` / :func:`make_ensemble`; the import-time
    registry registration carries this class through the whole engine
    path unchanged.

    Attributes:
        config: the validated :class:`EnsembleConfig`.
        member_models: fitted ``(name, estimator)`` pairs in member
            order (empty before ``fit``).
        meta: the fitted :class:`StackingMeta` (stacking only).
        fitted_: False until :meth:`fit` completes.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        """Store the validated config; nothing is fitted yet."""
        self.config = config
        self.member_models: list[tuple[str, Any]] = []
        self.meta: StackingMeta | None = None
        self.fitted_ = False

    def fit(self, feats: Any, labels: Any) -> EnsembleEstimator:
        """Fit every member (and the past-only stacking meta, if any).

        Rows MUST be chronological: the stacking combination builds its
        meta training set with :func:`oof_score_matrix`, whose position
        tiles are honest time tiles only under that guarantee (the
        walk-forward engine provides it).

        Args:
            feats: ``(n_rows, n_features)`` float matrix.
            labels: binary 0/1 labels of the same rows.

        Returns:
            Itself (fitted), sklearn-style.

        Raises:
            ValueError: on shape/label-contract violations or (for
                stacking) rows too few for ``meta_folds`` honest tiles.
        """
        matrix_x = _as_score_matrix("feats", feats)
        targets = _as_binary_labels("labels", labels)
        if matrix_x.shape[0] != targets.size:
            raise ValueError(
                f"labels must match feats rows ({matrix_x.shape[0]}), "
                f"got {targets.size}"
            )
        self.member_models = []
        for slot, name in enumerate(self.config.members):
            model = _build_member(
                name, self.config.member_params, self.config.seed + slot
            )
            model.fit(matrix_x, targets)
            self.member_models.append((name, model))
        self.meta = None
        if self.config.combination == "stacking":
            oof, row_ix = oof_score_matrix(
                matrix_x,
                targets,
                self.config.members,
                n_folds=self.config.meta_folds,
                seed=self.config.seed,
                member_params=self.config.member_params,
            )
            meta = StackingMeta(self.config.meta_learner, self.config.seed)
            meta.fit(oof, targets[row_ix])
            self.meta = meta
        self.fitted_ = True
        return self

    def predict_proba(self, feats: Any) -> np.ndarray:
        """Blended class probabilities, ``(n_rows, 2)``.

        Member scores are (optionally rank-)normalized, combined by
        the configured rule and squashed into ``[0, 1]`` (see
        :func:`_squash_to_probability`).  Note that rank normalization
        of a single row maps to zero -- blend at least two rows.

        Args:
            feats: ``(n_rows, n_features)`` float matrix.

        Returns:
            Positive/negative class probabilities per row.

        Raises:
            RuntimeError: if called before :meth:`fit`.
            ValueError: on a non-2-D or non-finite input.
        """
        if not self.fitted_:
            raise RuntimeError("EnsembleEstimator is not fitted")
        matrix_x = _as_score_matrix("feats", feats)
        columns: list[np.ndarray] = []
        for _, model in self.member_models:
            positive = np.asarray(
                model.predict_proba(matrix_x), dtype=np.float64
            )[:, 1]
            if self.config.normalize_scores:
                positive = rank_normalize(positive)
            columns.append(positive)
        stacked = np.column_stack(columns)
        combined = _combine_scores(self.config, stacked, self.meta)
        positive = _squash_to_probability(combined, self.config)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, feats: Any) -> np.ndarray:
        """Class labels at the 0.5 probability boundary.

        Args:
            feats: ``(n_rows, n_features)`` float matrix.

        Returns:
            int64 0/1 labels per row.

        Raises:
            RuntimeError: if called before :meth:`fit`.
        """
        proba = self.predict_proba(feats)
        return (proba[:, 1] > 0.5).astype(np.int64)


def build_ensemble(
    config: EnsembleConfig | dict[str, Any] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    seed: int | None = None,
) -> EnsembleEstimator:
    """Composite factory: an estimator from a config plus members.

    ``config`` may be a validated :class:`EnsembleConfig` or a dict of
    overrides over :data:`DEFAULT_ENSEMBLE_CONFIG` (``None`` takes the
    defaults); ``model_params`` are per-member factory kwargs merged
    over the config's ``member_params``; ``seed`` overrides the config
    seed.  Everything is re-validated, so a broken merged value fails
    here -- loudly, before any fitting.

    Args:
        config: ensemble config (dataclass, overrides dict or None).
        model_params: per-member kwargs merged over the config.
        seed: base-seed override.

    Returns:
        An unfitted :class:`EnsembleEstimator`.

    Raises:
        ValueError: on any contract violation of the merged config.

    Example::

        >>> estimator = build_ensemble(
        ...     {"members": ("logit",)},
        ...     {"logit": {"max_iter": 200}},
        ...     seed=0,
        ... )
        >>> feats = np.array([[-1.0], [-0.5], [0.5], [1.0]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> estimator.fit(feats, labels).predict(
        ...     np.array([[0.9], [-0.9]])
        ... ).tolist()
        [1, 0]
    """
    overrides: dict[str, Any] = (
        asdict(config)
        if isinstance(config, EnsembleConfig)
        else dict(config or {})
    )
    if model_params is not None:
        overrides["member_params"] = {
            **overrides.get("member_params", {}),
            **dict(model_params),
        }
    if seed is not None:
        overrides["seed"] = seed
    return EnsembleEstimator(normalize_ensemble_config(overrides))


def make_ensemble(
    *,
    members: Any = DEFAULT_ENSEMBLE_CONFIG["members"],
    weights: Any = DEFAULT_ENSEMBLE_CONFIG["weights"],
    combination: Any = DEFAULT_ENSEMBLE_CONFIG["combination"],
    normalize_scores: Any = DEFAULT_ENSEMBLE_CONFIG["normalize_scores"],
    meta_learner: Any = DEFAULT_ENSEMBLE_CONFIG["meta_learner"],
    meta_folds: Any = DEFAULT_ENSEMBLE_CONFIG["meta_folds"],
    member_params: Any = None,
    seed: Any = DEFAULT_ENSEMBLE_CONFIG["seed"],
) -> EnsembleEstimator:
    """Registry adapter: the composite from flat engine kwargs.

    This is the entry registered as ``MODEL_REGISTRY["ensemble"]`` at
    import time: the engine calls it with the flat kwargs of
    ``config["model_params"]["ensemble"]`` (see the module docstring
    for the mapping); ``member_params=None`` means "no per-member
    overrides" and the ``seed`` kwarg is injected by the engine as its
    ``model_seed`` when the config does not set one.

    Args:
        members: registry names in blend order.
        weights: optional raw blend weights (any positive scale).
        combination: mean|weighted|rank_mean|stacking.
        normalize_scores: rank-normalize member scores first.
        meta_learner: logreg|ridge|none.
        meta_folds: past-only OOF tiles for stacking (>= 2).
        member_params: per-member factory kwargs.
        seed: base seed (member ``i`` gets ``seed + i``).

    Returns:
        An unfitted :class:`EnsembleEstimator`.

    Raises:
        ValueError: on any kwarg violating the ensemble contract.

    Example::

        >>> model = make_ensemble(
        ...     members=("logit",),
        ...     member_params={"logit": {"max_iter": 200}},
        ...     seed=0,
        ... )
        >>> fitted = model.fit(
        ...     np.array([[-1.0], [-0.5], [0.5], [1.0]]),
        ...     np.array([0, 0, 1, 1]),
        ... )
        >>> float(fitted.predict_proba(np.array([[0.9], [-0.9]]))[0, 1]) > 0.5
        True
    """
    kwargs: dict[str, Any] = {
        "members": members,
        "weights": weights,
        "combination": combination,
        "normalize_scores": normalize_scores,
        "meta_learner": meta_learner,
        "meta_folds": meta_folds,
        "seed": seed,
    }
    if member_params is not None:
        kwargs["member_params"] = member_params
    return build_ensemble(kwargs)


# Import side effects (module docstring): registering the composite in
# the engine's open registry makes model="ensemble" work through the
# full walk-forward / EV / gate path with no engine changes, and the
# matching model_params entry keeps the engine's bundled-defaults
# invariant true (every registry name ships model_params defaults).
MODEL_REGISTRY["ensemble"] = make_ensemble
DEFAULT_TRAINING_CONFIG["model_params"].setdefault(
    "ensemble", {"members": DEFAULT_ENSEMBLE_CONFIG["members"]}
)


# ---------------------------------------------------------------------------
# engine wrapper with per-member diagnostics
# ---------------------------------------------------------------------------
def _pair_correlations(
    oof: np.ndarray, members: tuple[str, ...]
) -> dict[str, float | None]:
    """Pairwise Spearman correlations of member OOF score columns."""
    out: dict[str, float | None] = {}
    for first in range(len(members)):
        for second in range(first + 1, len(members)):
            rho = spearmanr(oof[:, first], oof[:, second]).statistic
            value = (
                None
                if rho is None or not math.isfinite(float(rho))
                else float(rho)
            )
            out[f"{members[first]}|{members[second]}"] = value
    return out


def _ensemble_diagnostics(
    features: pd.DataFrame,
    target: Any,
    battery_controls: list[str] | None,
    config: EnsembleConfig,
) -> dict[str, Any]:
    """Per-member diagnostics over a full-timeline past-only OOF pass.

    Runs :func:`oof_score_matrix` once over the dataset's model
    columns (battery controls excluded, chronological rows) with the
    run's member settings, then reports blend weights, the members'
    pairwise OOF score correlations and (stacking) the meta-learner
    coefficients fitted on that matrix.  Too few rows for honest tiles
    (no room for a ``_MIN_TRAIN_FOR_TILES``-row training window per
    tile) yield the documented empty scaffold (``oof_rows=0``) instead
    of an error after the engine has already run.
    """
    controls = list(dict.fromkeys(battery_controls or []))
    model_columns = [
        column for column in features.columns if column not in controls
    ]
    labels = _as_binary_labels("target", target)
    entry: dict[str, Any] = {
        "members": list(config.members),
        "combination": config.combination,
        "normalize_scores": config.normalize_scores,
        "weights": {
            name: float(weight)
            for name, weight in zip(
                config.members, config.normalized_weights(), strict=True
            )
        },
        "meta_learner": config.meta_learner,
        "meta_folds": config.meta_folds,
        "seed": config.seed,
        "oof_rows": 0,
        "oof_correlations": {},
        "meta_coefficients": None,
    }
    if (
        not model_columns
        or labels.size // config.meta_folds < _MIN_TRAIN_FOR_TILES
    ):
        return entry
    matrix_x = features.loc[:, model_columns].to_numpy(dtype=np.float64)
    oof, row_ix = oof_score_matrix(
        matrix_x,
        labels,
        config.members,
        n_folds=config.meta_folds,
        seed=config.seed,
        member_params=config.member_params,
    )
    entry["oof_rows"] = int(oof.shape[0])
    entry["oof_correlations"] = _pair_correlations(oof, config.members)
    if config.combination == "stacking":
        meta = StackingMeta(config.meta_learner, config.seed)
        meta.fit(oof, labels[row_ix].astype(np.float64))
        coefficients = meta.coefficients()
        total = float(np.abs(coefficients).sum())
        shares = (
            np.abs(coefficients) / total
            if total > 0.0
            else np.zeros_like(coefficients)
        )
        entry["weights"] = {
            name: float(share)
            for name, share in zip(config.members, shares, strict=True)
        }
        entry["meta_coefficients"] = {
            name: float(coefficient)
            for name, coefficient in zip(
                config.members, coefficients, strict=True
            )
        }
    return entry


def run_ensemble_report(
    features: pd.DataFrame,
    target: np.ndarray,
    exec_returns: np.ndarray,
    times_us: np.ndarray,
    *,
    ensemble: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    battery_controls: list[str] | None = None,
) -> dict[str, Any]:
    """Run the kill protocol on the ``"ensemble"`` registry composite.

    Thin wrapper over
    :func:`atropos.ml_training.run_ml_kill_protocol` with
    ``model="ensemble"``: the engine report returns with every engine
    key unchanged (verdicts, gates, folds, per-model entries) plus ONE
    additive ``"ensemble"`` namespace of diagnostics -- normalized
    blend ``weights``, pairwise Spearman correlations of the members'
    past-only OOF scores (``oof_correlations``, keys ``"a|b"``), the
    fitted stacking coefficients when stacking
    (``meta_coefficients``), the scored row count and the config echo.

    Precedence: explicit ``ensemble`` kwargs override the ensemble dict
    inside ``config["model_params"]["ensemble"]``, which overrides
    :data:`DEFAULT_ENSEMBLE_CONFIG`.  The ensemble seed defaults to
    the engine's ``model_seed``.  The merged ensemble config is
    validated BEFORE the engine runs (a violation raises with no model
    fitted), and its normalized :func:`dataclasses.asdict` view is
    what reaches ``model_params["ensemble"]`` -- the engine's registry
    adapter re-reads already-valid values instead of re-validating a
    raw merge.

    Args:
        features: DataFrame of float feature columns, chronological
            rows (engine contract).
        target: 0/1 integer labels of the run.
        exec_returns: executable per-event returns (costs netted).
        times_us: monotone int64 microsecond stamps.
        ensemble: ensemble config overrides.
        config: engine config overrides (see the engine).
        battery_controls: columns feeding the linear battery.

    Returns:
        The engine report plus the ``"ensemble"`` namespace.

    Raises:
        ValueError: on an invalid ensemble config or dataset, before
            any model is fit.

    Example::

        A planted nonlinear edge, the DSR gate off (the toy sample
        spans too few days to power it)::

        >>> rng = np.random.default_rng(0)
        >>> hidden = rng.standard_normal((500, 3))
        >>> frame = pd.DataFrame(
        ...     {"f1": hidden[:, 0], "f2": hidden[:, 1], "c1": hidden[:, 2]}
        ... )
        >>> exec_ret = (
        ...     0.01 * np.sign(hidden[:, 0] * hidden[:, 1])
        ...     + 0.003 * rng.standard_normal(500)
        ... )
        >>> stamps = np.arange(500, dtype=np.int64) * 3_600_000_000
        >>> labels = (exec_ret > 0.004).astype(np.int64)
        >>> report = run_ensemble_report(
        ...     frame,
        ...     labels,
        ...     exec_ret,
        ...     stamps,
        ...     ensemble={
        ...         "members": ["logit", "gbm"],
        ...         "combination": "mean",
        ...         "member_params": {"gbm": {"n_estimators": 20}},
        ...     },
        ...     config={
        ...         "k_blocks": 3,
        ...         "bootstrap_b": 50,
        ...         "n_min_events": 5,
        ...         "checks": {"deflated_sharpe": False},
        ...     },
        ...     battery_controls=["c1"],
        ... )
        >>> report["ensemble"]["members"]
        ['logit', 'gbm']
        >>> report["verdict"] != "UNDERPOWERED"
        True
    """
    merged: dict[str, Any] = dict(config or {})
    engine_params = merged.get("model_params")
    from_engine: dict[str, Any] = (
        dict(engine_params.get("ensemble") or {})
        if isinstance(engine_params, dict)
        else {}
    )
    combined: dict[str, Any] = {
        **DEFAULT_ENSEMBLE_CONFIG,
        **from_engine,
        **(ensemble or {}),
    }
    if "seed" not in from_engine and "seed" not in (ensemble or {}):
        combined["seed"] = merged.get(
            "model_seed", DEFAULT_TRAINING_CONFIG["model_seed"]
        )
    normalized = normalize_ensemble_config(combined)
    merged["model"] = "ensemble"
    engine_ensemble = asdict(normalized)
    if isinstance(engine_params, dict):
        merged["model_params"] = {
            **engine_params,
            "ensemble": engine_ensemble,
        }
    else:
        merged["model_params"] = {"ensemble": engine_ensemble}
    report = run_ml_kill_protocol(
        features,
        target,
        exec_returns,
        times_us,
        config=merged,
        battery_controls=battery_controls,
    )
    report["ensemble"] = _ensemble_diagnostics(
        features, target, battery_controls, normalized
    )
    return report


# ---------------------------------------------------------------------------
# synthetic verification demo
# ---------------------------------------------------------------------------
def _demo_world(n_rows: int = 1400, seed: int = 7) -> dict[str, Any]:
    """Seeded synthetic hourly world with a planted nonlinear edge.

    The executable return loads on the quadrant interaction
    ``tanh(f1 * f2 / 2)`` (invisible to any linear model) plus a mild
    linear ``f3`` term the battery CAN see; ``ctrl1`` / ``ctrl2`` are
    noise controls.  Own generator -- seeded, no test-file imports.
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
    stamps = (
        int(np.datetime64("2023-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_rows, dtype=np.int64) * _HOUR_US
    )
    labels = (exec_ret > 0.004).astype(np.int64)
    return {
        "features": feature_frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["ctrl1", "ctrl2"],
    }


_DEMO_MEMBER_PARAMS: dict[str, dict[str, Any]] = {
    "gbm": {"n_estimators": 60, "max_depth": 4},
    "rf": {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 10},
}


def run_all_examples() -> None:
    """Verify the ensemble composite end to end (asserted demo).

    Case 1: the past-only OOF matrix of a planted nonlinear world has
    the honest shape ``(rows after the first tile, n_members)`` with
    strictly increasing row indices.  Case 2: the ``["gbm", "rf"]``
    rank-mean ensemble rides the FULL engine path via the registry
    (walk-forward, thresholds, EV, gates) and must reach a meaningful
    verdict -- never UNDERPOWERED.  Case 3: two identical runs produce
    bit-identical reports (determinism gate).
    """
    print("=" * 78)
    print("CASE 1: past-only OOF score matrix (honest inner nesting)")
    print("=" * 78)
    world = _demo_world()
    model_x = (
        world["features"]
        .drop(columns=world["battery_controls"])
        .to_numpy(dtype=np.float64)
    )
    oof, row_ix = oof_score_matrix(
        model_x,
        world["target"],
        ("gbm", "rf"),
        n_folds=4,
        seed=42,
        member_params=_DEMO_MEMBER_PARAMS,
    )
    print(
        f"oof matrix {oof.shape}, scored rows {row_ix.size} of "
        f"{model_x.shape[0]}"
    )
    print()
    print("=" * 78)
    print("CASE 2: ensemble(['gbm', 'rf'], rank_mean) through the engine")
    print("=" * 78)
    report = run_ensemble_report(
        world["features"],
        world["target"],
        world["exec_returns"],
        world["times_us"],
        ensemble={
            "members": ["gbm", "rf"],
            "combination": "rank_mean",
            "member_params": _DEMO_MEMBER_PARAMS,
        },
        config={
            "k_blocks": 4,
            "bootstrap_b": 200,
            "n_min_events": 5,
            "checks": {"deflated_sharpe": False},
        },
        battery_controls=world["battery_controls"],
    )
    _print_ensemble_report(report)
    print()
    print("=" * 78)
    print("CASE 3: determinism (two identical runs)")
    print("=" * 78)
    repeat = run_ensemble_report(
        world["features"],
        world["target"],
        world["exec_returns"],
        world["times_us"],
        ensemble={
            "members": ["gbm", "rf"],
            "combination": "rank_mean",
            "member_params": _DEMO_MEMBER_PARAMS,
        },
        config={
            "k_blocks": 4,
            "bootstrap_b": 200,
            "n_min_events": 5,
            "checks": {"deflated_sharpe": False},
        },
        battery_controls=world["battery_controls"],
    )
    _verify_demo(oof, row_ix, report, repeat, n_rows=int(model_x.shape[0]))


def _print_ensemble_report(report: dict[str, Any]) -> None:
    """Human-readable one-run summary (demo output only)."""
    entry = report["models"]["ensemble"]
    print(f"verdict     : {report['verdict']}")
    print(
        f"n_rows={report['n_rows']}  n_events={entry['n_events']} "
        f"(min {report['n_min_events']}, powered={report['powered']})"
    )
    print(
        f"baseline EV={_fmt_bp(report['ev_baseline'])}  "
        f"battery EV={_fmt_bp(report['ev_battery'])}  "
        f"ensemble EV={_fmt_bp(entry['ev'])}"
    )
    namespace = report["ensemble"]
    weights = "  ".join(
        f"{name}={weight:.3f}"
        for name, weight in namespace["weights"].items()
    )
    print(f"combination : {namespace['combination']}  weights [{weights}]")
    for pair, rho in sorted(namespace["oof_correlations"].items()):
        shown = "n/a" if rho is None else f"{rho:.3f}"
        print(f"  oof corr {pair:<16} {shown}")
    for gate in entry["kills"]:
        state = (
            "off"
            if not gate["enabled"]
            else (
                "INCONCLUSIVE"
                if gate["inconclusive"]
                else ("TRIGGERED" if gate["triggered"] else "pass")
            )
        )
        print(f"    {gate['check']:<20} {state}")


def _fmt_bp(value: float | None) -> str:
    """Basis-point formatting for demo output."""
    if value is None:
        return "n/a"
    return f"{value * 1e4:.1f}bp"


def _verify_demo(
    oof: np.ndarray,
    row_ix: np.ndarray,
    report: dict[str, Any],
    repeat: dict[str, Any],
    n_rows: int,
) -> None:
    """Assert the demo contract (the demo fails loudly on drift).

    The determinism check compares canonical ``json.dumps`` serializations
    (``sort_keys``, ``default=str``) instead of dict equality, so the
    gate stays valid if the report ever grows numpy-array leaves that
    ``==`` would compare elementwise.
    """
    assert oof.shape == (row_ix.size, 2)
    assert row_ix.size == n_rows - n_rows // 4  # rows after first tile
    assert np.all(np.diff(row_ix) > 0)
    assert "ensemble" in report["models"]
    verdict = report["verdict"]
    assert verdict != "UNDERPOWERED"
    assert verdict == "SURVIVES" or verdict.startswith("DEAD(")
    namespace = report["ensemble"]
    assert namespace["members"] == ["gbm", "rf"]
    assert set(namespace["weights"]) == {"gbm", "rf"}
    assert math.isclose(sum(namespace["weights"].values()), 1.0)
    assert "gbm|rf" in namespace["oof_correlations"]
    assert json.dumps(report, default=str, sort_keys=True) == json.dumps(
        repeat, default=str, sort_keys=True
    )
    print("two runs are bit-identical: all verification checks passed")


if __name__ == "__main__":
    run_all_examples()
