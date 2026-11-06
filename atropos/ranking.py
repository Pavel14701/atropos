"""Candidate ranking layer over the open model registry.

Two oracles pick the BEST candidate within a GROUP (candidate signals
of one day / actions on one bar) by expected value, trained strictly
on PAST groups:

* pointwise-proba ranking -- :func:`rank_within_groups` fits any
  :data:`atropos.ml_training.MODEL_REGISTRY` classifier on past
  groups (features -> ``P(win)``), scores the current group's
  candidates with the predicted probabilities and keeps the
  top-``top_s`` picks.  This is NOT lambdarank: an honest pointwise
  formulation every registry classifier supports out of the box (the
  whole house fleet, never one vendor's ranker).
* lambdarank oracle -- :func:`rank_with_lambdarank` ports the donor
  corpus' ``LGBMRanker`` fit (``objective="lambdarank"``, graded
  relevance, fixed hyper-parameters, ``random_state`` seed).  lightgbm
  is an OPTIONAL dependency, imported lazily inside the function; the
  module imports and every other path runs without it.

Both oracles share the expanding past-only schedule (group ``t`` is
scored by a model fitted on groups ``< t - embargo_groups``), the
deterministic ``(score, candidate_id)`` tie-break, and the
:class:`CandidateGroup` interchange contract consumed by the admission
layer.

Per-group ranking-quality metrics live in the open
:data:`RANKING_METRICS` registry (always computed on RAW model
scores; within one group raw and normalized scores induce the same
ordering):

=================  ====================================================
metric             definition (per group, best candidate first)
=================  ====================================================
ndcg@k             NDCG over graded relevance read off the ideal
                   order; gain ``2**rel - 1``, ``rel = n - 1 - pos``
precision@k        share of the scored top-k sitting in the ideal top-k
mrr                reciprocal position of the single best candidate
                   (``ideal_order[0]``) within the scored top-k
spearman           Spearman correlation of ideal vs scored positions
                   over the scored top-k head (needs ``top_k >= 2``)
=================  ====================================================

:data:`RANKER_SUITABILITY` is an honest suitability marking per
registry model, not a gate: ``low`` = noisy probabilities on small
candidate groups (kNN/QDA) -- for groups under ~10 candidates prefer
``logit`` / ``gbm`` / ``hist_gbm``.  The registry accepts every
model; nothing is refused.

Cross-ranker comparisons run on the scored INTERSECTION, never on the
full lists (``n_skipped`` is per-run, so unequal skip sets would fake
a gap): :func:`compare_rankers_on_intersection` computes every
:data:`RANKING_METRICS` metric per common group for two reports and
attaches a :func:`paired_bootstrap_diff` CI on the difference.  The
paired bootstrap itself takes an opt-in ``enforce_embargo`` that
drops pairs whose ``CandidateGroup.embargo_until`` exceeds the fold
being decided.

Donor-corpus ports kept semantically DISTINCT from their closest
house relatives (the difference is pinned in each docstring):
:func:`ev_threshold` vs
:func:`atropos.ml_training.pick_threshold` (inclusive ``>=`` with an
all-trades floor and a ``(threshold, ev, n)`` payload, vs a bare
strict-``>`` threshold) and :func:`paired_bootstrap_diff` vs
:func:`atropos.ml_training.paired_bootstrap_t_stat` (raw paired
difference CI, no t, vs a taken-flag t-statistic).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from atropos._utils import validate_int, validate_number
from atropos.ml_ensemble import rank_normalize
from atropos.ml_training import DEFAULT_TRAINING_CONFIG, MODEL_REGISTRY

__all__ = [
    "RANKER_SUITABILITY",
    "RANKING_METRICS",
    "BootstrapDiff",
    "CandidateGroup",
    "apply_rule_table",
    "compare_rankers_on_intersection",
    "ev_threshold",
    "fit_rule_table",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "paired_bootstrap_diff",
    "precision_at_k",
    "rank_normalize_group",
    "rank_with_lambdarank",
    "rank_within_groups",
    "spearman_in_group",
]

RankingMetric = Callable[[Sequence[str], Sequence[str], int], float]
"""Metric registry entry: ``metric(ideal_order, scored_order, top_k)``.

Both orders are candidate-id sequences ranking the SAME candidate set
(best first); the return value is one float.  Registering a custom
metric (``RANKING_METRICS["ap@k"] = my_fn``) makes it available to
comparison sweeps with no engine changes -- the same open-registry
pattern as :data:`atropos.ml_training.MODEL_REGISTRY`.
"""

RANKER_SUITABILITY: dict[str, str] = {
    "extratrees": "medium",
    "gbm": "high",
    "hist_gbm": "high",
    "knn": "low",
    "logit": "high",
    "mlp": "medium",
    "qda": "low",
    "rf": "medium",
}
"""Honest suitability marking per registry model -- information, NOT
a gate (the registry accepts every model).

``low`` = noisy probabilities on small candidate groups (kNN/QDA):
for groups under ~10 candidates prefer ``logit`` / ``gbm`` /
``hist_gbm``.  ``medium`` = usable, especially after
:func:`rank_normalize_group`.  ``high`` = stable per-candidate
probabilities on small groups.
"""

_EV_GRID: tuple[float, ...] = tuple(
    round(0.30 + 0.05 * step, 2) for step in range(9)
)
_RULE_TABLE_COLUMNS = ("regime_dir", "side", "rule", "r_net")

_LAMBDA_HYPER: dict[str, Any] = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "label_gain": list(range(13)),
    "verbosity": -1,
}
"""Donor-pinned ``LGBMRanker`` hyper-parameters (protocol.py
``train_ranker``): fixed on purpose so the oracle is reproducible."""


class BootstrapDiff(TypedDict):
    """Result payload of :func:`paired_bootstrap_diff`."""

    diff: float
    lo: float
    hi: float
    p_le0: float
    n: int


@dataclass(frozen=True, eq=False)
class CandidateGroup:
    """One ranking group: the candidates of a single decision point.

    The interchange contract between the ranking layer and the
    admission layer.  ``scores`` are RAW model probabilities -- raw
    cross-model probabilities are not comparable, so run them through
    :func:`rank_normalize_group` before feeding an admission policy
    (scores are never normalized in place; ``rank_normalize_group``
    returns a fresh array).  Within one group raw and normalized
    scores induce the same ordering, so :data:`RANKING_METRICS` may
    consume the raw scores directly.

    ``labels`` are the realized ground-truth utilities (NaN marks an
    untradeable candidate); ``fold_id`` and ``embargo_until`` carry
    fold metadata for downstream schedulers and are NOT enforced
    here.  Value equality is disabled (the ndarray fields make it
    ambiguous): compare field by field.

    Raises:
        ValueError: on an empty candidate list, non-string or
            duplicate candidate ids, length mismatches, non-finite
            scores, non-numeric labels, or a non-integer
            ``fold_id`` / ``embargo_until``.

    Example::

        >>> group = CandidateGroup(
        ...     group_id="2024-01-02",
        ...     candidates=["c_long", "c_short"],
        ...     scores=np.array([0.9, 0.7]),
        ...     labels=np.array([0.012, -0.004]),
        ... )
        >>> group.candidates[0]
        'c_long'
        >>> CandidateGroup(
        ...     group_id="d",
        ...     candidates=["a", "a"],
        ...     scores=np.array([0.1, 0.2]),
        ...     labels=np.array([0.0, 0.0]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: CandidateGroup candidates must be unique, got ['a', 'a']
    """

    group_id: Any
    candidates: list[str]
    scores: np.ndarray
    labels: np.ndarray
    fold_id: int | None = None
    embargo_until: int | None = None

    def __post_init__(self) -> None:
        """Validate every field loudly (the contract lives on the class)."""
        candidates = list(self.candidates)
        if not candidates:
            raise ValueError("CandidateGroup needs at least one candidate")
        if any(not isinstance(item, str) for item in candidates):
            raise ValueError("CandidateGroup candidates must be strings")
        if len(set(candidates)) != len(candidates):
            raise ValueError(
                f"CandidateGroup candidates must be unique, got {candidates}"
            )
        object.__setattr__(self, "candidates", candidates)
        scores = np.asarray(self.scores, dtype=np.float64).ravel()
        if scores.size != len(candidates):
            raise ValueError(
                f"CandidateGroup scores length {scores.size} does not "
                f"match {len(candidates)} candidates"
            )
        if not np.isfinite(scores).all():
            raise ValueError("CandidateGroup scores must be finite")
        object.__setattr__(self, "scores", scores)
        try:
            labels = np.asarray(self.labels, dtype=np.float64).ravel()
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CandidateGroup labels must be a numeric array"
            ) from error
        if labels.size != len(candidates):
            raise ValueError(
                f"CandidateGroup labels length {labels.size} does not "
                f"match {len(candidates)} candidates"
            )
        object.__setattr__(self, "labels", labels)
        for field_name in ("fold_id", "embargo_until"):
            value = getattr(self, field_name)
            if value is not None:
                validate_int(field_name, value, minimum=0)


# ---------------------------------------------------------------------------
# shared validation and selection mechanics
# ---------------------------------------------------------------------------
def _validated_feature_matrix(features: pd.DataFrame) -> np.ndarray:
    """Numeric, finite ``(n_rows, n_features)`` matrix, else ValueError."""
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features frame is empty: nothing to rank")
    # pandas API: exact for ExtensionDtypes too (np.issubdtype crashes
    # on e.g. StringDtype instead of answering False)
    non_numeric = [
        str(column)
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise ValueError(
            f"feature columns must be numeric, got non-numeric {non_numeric}"
        )
    matrix = features.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "feature matrix must be finite (NaN/inf): impute or drop "
            "the offending cells -- registry classifiers reject "
            "non-finite input"
        )
    return matrix


def _validated_binary_target(target: Any, n_rows: int) -> np.ndarray:
    """Finite 0/1 int array of length ``n_rows``, else ValueError."""
    values = np.asarray(target)
    if values.ndim != 1 or values.size != n_rows:
        raise ValueError(
            f"target must be 1-D of length {n_rows}, got shape {values.shape}"
        )
    flat = values.astype(np.float64)
    if not np.isfinite(flat).all() or not np.isin(flat, (0.0, 1.0)).all():
        raise ValueError("target must be binary 0/1 win flags, finite")
    return flat.astype(np.int64)


def _validated_utility_target(target: Any, n_rows: int) -> np.ndarray:
    """Finite 1-D float utilities of length ``n_rows``, else ValueError."""
    values = np.asarray(target)
    if values.ndim != 1 or values.size != n_rows:
        raise ValueError(
            f"target must be 1-D of length {n_rows}, got shape {values.shape}"
        )
    flat = values.astype(np.float64)
    if not np.isfinite(flat).all():
        raise ValueError("target must be finite utilities (no NaN/inf)")
    return flat


def _contiguous_group_blocks(
    group_ids: Any, n_rows: int
) -> list[tuple[Any, np.ndarray]]:
    """``(group_id, row_positions)`` blocks in first-appearance order.

    Groups must be contiguous chronological blocks: rows of one group
    form a single run, and no group may reappear after a later group
    started (interleaved groups would silently redefine "past").
    Ids themselves need not be numerically sorted -- day strings are
    fine; the first-appearance order IS the chronology.
    """
    ids = np.asarray(group_ids)
    if ids.ndim != 1 or ids.size != n_rows:
        raise ValueError(
            f"group_ids must be 1-D of length {n_rows}, got shape {ids.shape}"
        )
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    edges = np.append(starts[1:], n_rows)
    blocks: list[tuple[Any, np.ndarray]] = []
    seen: set[Any] = set()
    for position, start in enumerate(starts):
        raw_id = ids[int(start)]
        try:
            group_id = raw_id.item()
        except AttributeError:
            group_id = raw_id
        if group_id in seen:
            raise ValueError(
                "group_ids must be contiguous chronological blocks; "
                f"group {group_id!r} reappears after a later group "
                "started"
            )
        seen.add(group_id)
        blocks.append((group_id, np.arange(int(start), int(edges[position]))))
    return blocks


def _top_positions(scores: np.ndarray, top_s: int) -> np.ndarray:
    """Indices of the top-``top_s`` scores, ties broken by position."""
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:top_s]


def _merged_rank_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user overrides over :data:`DEFAULT_TRAINING_CONFIG`.

    Dicts merge key-wise at every level (same contract as the
    ``ml_training`` engine), so an override of one model's parameter
    block keeps the other models' defaults.
    """
    merged: dict[str, Any] = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in DEFAULT_TRAINING_CONFIG.items()
    }
    if not config:
        return merged
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            target = merged[key]
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and isinstance(
                    target.get(sub_key), dict
                ):
                    target[sub_key] = {**target[sub_key], **sub_value}
                else:
                    target[sub_key] = sub_value
        else:
            merged[key] = value
    params = merged["model_params"]
    if not isinstance(params, dict) or not all(
        isinstance(entry, dict) for entry in params.values()
    ):
        raise ValueError("model_params must map model names to dicts")
    return merged


def _score_given_groups(
    groups: Sequence[CandidateGroup],
    *,
    top_s: int,
    score_key: str,
) -> dict[str, Any]:
    """Rank already-scored groups: no refitting, schedule irrelevant.

    Selection tie-breaks by ``(score desc, candidate_id asc)``;
    ``selected`` holds candidate ids (str), not row positions.
    """
    group_list = list(groups)
    if not group_list:
        raise ValueError("groups list is empty: nothing to rank")
    for group in group_list:
        if not isinstance(group, CandidateGroup):
            raise ValueError(
                "groups must be CandidateGroup instances, got "
                f"{type(group).__name__}"
            )
    seen: set[Any] = set()
    selected: list[str] = []
    score_lists: list[list[float]] = []
    scored_ids: list[Any] = []
    for group in group_list:
        if group.group_id in seen:
            raise ValueError(
                f"duplicate group_id in groups list: {group.group_id!r}"
            )
        seen.add(group.group_id)
        picks = sorted(
            range(len(group.candidates)),
            key=lambda index: (
                -float(group.scores[index]),
                group.candidates[index],
            ),
        )[:top_s]
        selected.extend(group.candidates[index] for index in picks)
        score_lists.append([float(value) for value in group.scores])
        scored_ids.append(group.group_id)
    return {
        "model": "given",
        "seed": 0,
        "top_s": top_s,
        "embargo_groups": 0,
        "n_groups": len(group_list),
        "n_rows": sum(len(group.candidates) for group in group_list),
        "n_scored": len(scored_ids),
        "n_skipped": 0,
        "selected": selected,
        score_key: score_lists,
        "scored_group_ids": scored_ids,
        "train_group_counts": [],
        "groups": group_list,
    }


# ---------------------------------------------------------------------------
# oracle 1: pointwise-proba ranking over the model registry
# ---------------------------------------------------------------------------
def rank_within_groups(
    features: pd.DataFrame | Sequence[CandidateGroup],
    target: Any | None = None,
    group_ids: Any | None = None,
    *,
    model: str = "gbm",
    top_s: int = 1,
    embargo_groups: int = 1,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Top-``top_s`` candidates per group by registry-model P(win).

    Pointwise-proba ranking -- the primary oracle, open to the whole
    :data:`atropos.ml_training.MODEL_REGISTRY` (plus caller-registered
    factories).  Group ``t`` is scored by a model fitted on the
    EXPANDING PAST ONLY: groups with index ``< t - embargo_groups``
    (the embargo gap keeps same-regime neighbours out of the train
    fit, mirroring the donor fold calendar).  Both input forms are
    supported:

    * ``features`` DataFrame + binary 0/1 ``target`` + ``group_ids``:
      the full fit-and-score path (every scored group refits the
      model on its past); ``selected`` holds global row positions
      (int).
    * a sequence of already-scored :class:`CandidateGroup` groups:
      the scores are taken as given (``model`` / ``seed`` /
      ``config`` / ``embargo_groups`` are not exercised); groups
      with too-small train windows cannot exist here, so every group
      is scored; ``selected`` holds candidate ids (str).

    Ties break by ``(score desc, position asc)`` on the DataFrame
    path and ``(score desc, candidate_id asc)`` on the group path --
    deterministic in both cases.  ``top_s`` is capped at the group
    size.  A scored group whose past training window is single-class
    is skipped honestly (counted in ``n_skipped``, no selection).
    Model fit errors propagate loudly -- size the factory params for
    the SMALLEST early window (e.g. a kNN ``k`` above the first
    window's row count, or QDA's more-samples-than-features rule
    before enough class rows accumulate).

    Args:
        features: numeric feature frame, or ready-scored groups.
        target: binary 0/1 win flags (DataFrame input only).
        group_ids: contiguous chronological group blocks (DataFrame
            input only).
        model: registry name of the pointwise classifier.
        top_s: how many candidates to keep per scored group, >= 1.
        embargo_groups: groups excluded from the train window right
            behind the scored group, >= 0.
        config: overrides over
            :data:`atropos.ml_training.DEFAULT_TRAINING_CONFIG`
            (nested dicts merge key-wise); this oracle consumes
            ``model_params[model]``.
        seed: factory seed (the explicit argument wins over any
            ``model_seed`` in ``config``); factory kwargs default to
            ``seed`` per the registry contract.

    Returns:
        Report dict with the keys ``model``, ``seed``, ``top_s``,
        ``embargo_groups``, ``n_groups``, ``n_rows``, ``n_scored``,
        ``n_skipped``, ``selected``, ``per_group_proba``,
        ``scored_group_ids``, ``train_group_counts`` and ``groups``.
        ``per_group_proba[g][j]`` is the probability of
        ``groups[g].candidates[j]``.

    Raises:
        ValueError: on unknown model, non-numeric / non-finite
            features, non-binary target, interleaved or duplicate
            groups, too few groups for the embargo
            (``> embargo_groups + 1`` needed), a fully single-class
            target, or invalid scalar parameters.

    Example::

        >>> frame = pd.DataFrame(
        ...     {"quality": [float(pos % 3) for pos in range(12)]}
        ... )
        >>> target = np.array([int(pos % 3 == 2) for pos in range(12)])
        >>> groups = np.repeat(np.arange(4), 3)
        >>> report = rank_within_groups(
        ...     frame,
        ...     target,
        ...     groups,
        ...     model="logit",
        ...     top_s=1,
        ...     config={"model_params": {"logit": {"max_iter": 100}}},
        ... )
        >>> report["n_scored"], report["n_skipped"]
        (2, 0)

    Cross-ranker comparability note: ``n_skipped`` counts SKIPPED
    groups of THIS run only (too few rows / single-class).  When
    comparing two rankers via ``RANKING_METRICS``, compare them on the
    INTERSECTION of ``scored_group_ids`` so unequal skip sets cannot
    fake a difference -- :func:`compare_rankers_on_intersection`
    implements exactly this.
        >>> report["selected"]
        [8, 11]
        >>> rank_within_groups(report["groups"], top_s=1)["selected"]
        ['8', '11']
    """
    top_s = validate_int("top_s", top_s)
    embargo_groups = validate_int("embargo_groups", embargo_groups, minimum=0)
    seed = validate_int("seed", seed, minimum=0)
    if isinstance(features, pd.DataFrame):
        if target is None or group_ids is None:
            raise ValueError(
                "target and group_ids are required when features is a "
                "DataFrame"
            )
        matrix = _validated_feature_matrix(features)
        labels = _validated_binary_target(target, matrix.shape[0])
        blocks = _contiguous_group_blocks(group_ids, matrix.shape[0])
        return _rank_by_fitting(
            matrix,
            labels,
            blocks,
            model_name=model,
            top_s=top_s,
            embargo_groups=embargo_groups,
            config=config,
            seed=seed,
        )
    if target is not None or group_ids is not None:
        raise ValueError(
            "target/group_ids must be None when ranking CandidateGroup "
            "groups (scores are already attached)"
        )
    return _score_given_groups(
        groups=features, top_s=top_s, score_key="per_group_proba"
    )


def _rank_by_fitting(
    matrix: np.ndarray,
    labels: np.ndarray,
    blocks: list[tuple[Any, np.ndarray]],
    *,
    model_name: str,
    top_s: int,
    embargo_groups: int,
    config: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Expanding past-only fit-and-score loop shared by the oracles."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {model_name!r}: available "
            f"{sorted(MODEL_REGISTRY)}"
        )
    merged = _merged_rank_config(config)
    params = dict(merged["model_params"].get(model_name, {}))
    params.setdefault("seed", seed)
    factory = MODEL_REGISTRY[model_name]
    n_groups = len(blocks)
    if n_groups < embargo_groups + 2:
        raise ValueError(
            "expanding past-only ranking needs more than "
            f"embargo_groups + 1 = {embargo_groups + 1} groups, got "
            f"{n_groups}"
        )
    selected: list[int] = []
    proba_lists: list[list[float]] = []
    scored_ids: list[Any] = []
    train_counts: list[int] = []
    groups_out: list[CandidateGroup] = []
    skipped = 0
    for position, (group_id, rows) in enumerate(blocks):
        if position < embargo_groups + 1:
            continue  # no usable past beyond the embargo yet
        train_rows = np.concatenate(
            [
                block_rows
                for _, block_rows in blocks[: position - embargo_groups]
            ]
        )
        train_labels = labels[train_rows]
        if np.unique(train_labels).size < 2:
            skipped += 1
            continue
        fitted = factory(**params)
        fitted.fit(matrix[train_rows], train_labels)
        proba = np.asarray(fitted.predict_proba(matrix[rows]))[:, 1]
        picks = _top_positions(proba, top_s)
        selected.extend(int(rows[index]) for index in picks)
        proba_lists.append([float(value) for value in proba])
        scored_ids.append(group_id)
        train_counts.append(position - embargo_groups)
        groups_out.append(
            CandidateGroup(
                group_id=group_id,
                candidates=[str(int(row)) for row in rows],
                scores=proba,
                labels=labels[rows].astype(np.float64),
            )
        )
    if not scored_ids:
        raise ValueError(
            "no group could be scored: every past-only training "
            "window was single-class (check the target or shrink "
            "embargo_groups)"
        )
    return {
        "model": model_name,
        "seed": seed,
        "top_s": top_s,
        "embargo_groups": embargo_groups,
        "n_groups": n_groups,
        "n_rows": int(matrix.shape[0]),
        "n_scored": len(scored_ids),
        "n_skipped": skipped,
        "selected": selected,
        "per_group_proba": proba_lists,
        "scored_group_ids": scored_ids,
        "train_group_counts": train_counts,
        "groups": groups_out,
    }


# ---------------------------------------------------------------------------
# oracle 2: lambdarank (optional lightgbm, lazy import)
# ---------------------------------------------------------------------------
def _import_lightgbm() -> Any:
    """Lazy-import lightgbm (OPTIONAL dependency) or raise with a hint.

    Dynamic ``importlib`` import keeps the module importable (and the
    extras dependency honest) without lightgbm installed; only this
    oracle ever touches it.
    """
    try:
        return importlib.import_module("lightgbm")
    except ImportError as error:
        raise ImportError(
            "lightgbm is an optional dependency: uv add lightgbm "
            "(or pip install lightgbm) to use the lambdarank oracle"
        ) from error


def rank_with_lambdarank(
    features: pd.DataFrame | Sequence[CandidateGroup],
    target: Any | None = None,
    group_ids: Any | None = None,
    *,
    top_s: int = 1,
    embargo_groups: int = 1,
    seed: int = 7,
) -> dict[str, Any]:
    """Top-``top_s`` candidates per group by the lambdarank oracle.

    Port of the donor corpus' ``train_ranker``: an ``LGBMRanker`` with
    ``objective="lambdarank"`` and the fixed hyper-parameters in
    ``_LAMBDA_HYPER``; graded relevance is
    ``clip(round((target + 2) * 2), 0, 12)`` over the caller's
    utility labels (binary 0/1 labels are valid input and stay
    ordinal); rows enter the fit as contiguous per-group blocks; the
    expanding past-only schedule and the tie-break match
    :func:`rank_within_groups`.  The two oracles rank by different
    semantics (graded listwise loss vs pointwise ``P(win)``), so their
    picks are comparable through :data:`RANKING_METRICS`, not asserted
    equal.

    The DataFrame input expects FINITE continuous utilities as
    ``target`` (NaN relevance is undefined).  The
    :class:`CandidateGroup` input takes the attached scores as given,
    exactly like :func:`rank_within_groups`.  ``seed`` feeds
    ``random_state`` (donor default 7).

    lightgbm is an OPTIONAL dependency, imported lazily inside this
    function; the module imports and every other path runs without
    it.

    Args:
        features: numeric feature frame, or ready-scored groups.
        target: finite utility labels (DataFrame input only).
        group_ids: contiguous chronological group blocks (DataFrame
            input only).
        top_s: how many candidates to keep per scored group, >= 1.
        embargo_groups: groups excluded right behind the scored
            group, >= 0.
        seed: ``LGBMRanker`` ``random_state``, integer >= 0.

    Returns:
        Report dict of the same shape as
        :func:`rank_within_groups`, with ``per_group_score`` in place
        of ``per_group_proba`` and ``model == "lambdarank"`` (the
        given-scores path reports ``model == "given"``).

    Raises:
        ImportError: if lightgbm is not installed.
        ValueError: on invalid inputs (same contract as
            :func:`rank_within_groups`, minus the binary-label
            restriction).

    Example::

        >>> report = rank_with_lambdarank(  # doctest: +SKIP
        ...     frame, target, groups, top_s=1, seed=7
        ... )
        >>> report["model"]  # doctest: +SKIP
        'lambdarank'
    """
    top_s = validate_int("top_s", top_s)
    embargo_groups = validate_int("embargo_groups", embargo_groups, minimum=0)
    seed = validate_int("seed", seed, minimum=0)
    if not isinstance(features, pd.DataFrame):
        return _score_given_groups(
            groups=features, top_s=top_s, score_key="per_group_score"
        )
    if target is None or group_ids is None:
        raise ValueError(
            "target and group_ids are required when features is a DataFrame"
        )
    ranker_factory = _import_lightgbm().LGBMRanker
    matrix = _validated_feature_matrix(features)
    utilities = _validated_utility_target(target, matrix.shape[0])
    blocks = _contiguous_group_blocks(group_ids, matrix.shape[0])
    n_groups = len(blocks)
    if n_groups < embargo_groups + 2:
        raise ValueError(
            "expanding past-only ranking needs more than "
            f"embargo_groups + 1 = {embargo_groups + 1} groups, got "
            f"{n_groups}"
        )
    relevance = np.clip(np.round((utilities + 2.0) * 2.0), 0, 12).astype(
        np.int64
    )
    selected: list[int] = []
    score_lists: list[list[float]] = []
    scored_ids: list[Any] = []
    train_counts: list[int] = []
    groups_out: list[CandidateGroup] = []
    skipped = 0
    for position, (group_id, rows) in enumerate(blocks):
        if position < embargo_groups + 1:
            continue  # no usable past beyond the embargo yet
        train_rows = np.concatenate(
            [
                block_rows
                for _, block_rows in blocks[: position - embargo_groups]
            ]
        )
        if train_rows.size < 2:
            skipped += 1
            continue
        included = blocks[: position - embargo_groups]
        group_sizes = [len(block_rows) for _, block_rows in included]
        ranker = ranker_factory(**_LAMBDA_HYPER, random_state=seed)
        ranker.fit(
            matrix[train_rows],
            relevance[train_rows],
            group=group_sizes,
            callbacks=[],
        )
        scores = np.asarray(ranker.predict(matrix[rows]), dtype=np.float64)
        picks = _top_positions(scores, top_s)
        selected.extend(int(rows[index]) for index in picks)
        score_lists.append([float(value) for value in scores])
        scored_ids.append(group_id)
        train_counts.append(position - embargo_groups)
        groups_out.append(
            CandidateGroup(
                group_id=group_id,
                candidates=[str(int(row)) for row in rows],
                scores=scores,
                labels=utilities[rows],
            )
        )
    if not scored_ids:
        raise ValueError(
            "no group could be scored: every past-only training "
            "window was too small (shrink embargo_groups or add groups)"
        )
    return {
        "model": "lambdarank",
        "seed": seed,
        "top_s": top_s,
        "embargo_groups": embargo_groups,
        "n_groups": n_groups,
        "n_rows": int(matrix.shape[0]),
        "n_scored": len(scored_ids),
        "n_skipped": skipped,
        "selected": selected,
        "per_group_score": score_lists,
        "scored_group_ids": scored_ids,
        "train_group_counts": train_counts,
        "groups": groups_out,
    }


# ---------------------------------------------------------------------------
# per-group ranking metrics
# ---------------------------------------------------------------------------
def _ordering_contract(
    ideal_order: Sequence[str], scored_order: Sequence[str]
) -> tuple[list[str], list[str], dict[str, int]]:
    """Validate an (ideal, scored) ordering pair; map ideal positions.

    Raises:
        ValueError: on empty orderings, fewer than two candidates,
            non-string entries, repeated ids, or the two orders not
            ranking the same candidate set.
    """
    ideal = list(ideal_order)
    scored = list(scored_order)
    for name, order in (("ideal_order", ideal), ("scored_order", scored)):
        if not order:
            raise ValueError(f"{name} is empty: quality is undefined")
        if any(not isinstance(item, str) for item in order):
            raise ValueError(f"{name} entries must be candidate-id strings")
        if len(set(order)) != len(order):
            raise ValueError(f"{name} must not repeat a candidate id")
    if len(ideal) != len(scored) or set(ideal) != set(scored):
        raise ValueError(
            "ideal_order and scored_order must rank the same candidate "
            f"set, got lengths {len(ideal)} and {len(scored)}"
        )
    if len(ideal) < 2:
        raise ValueError(
            "ranking metrics need at least two candidates per group"
        )
    return (
        ideal,
        scored,
        {candidate: position for position, candidate in enumerate(ideal)},
    )


def _dcg_at_k(gains: np.ndarray) -> float:
    """Discounted cumulative gain of position-ordered gains."""
    positions = np.arange(1, gains.size + 1, dtype=np.float64)
    return float(np.sum(gains / np.log2(positions + 1.0)))


def ndcg_at_k(
    ideal_order: Sequence[str], scored_order: Sequence[str], top_k: int
) -> float:
    """NDCG@``top_k`` of the scored order against the ideal order.

    Graded relevance is read off the ideal order: the candidate at
    ideal position ``pos`` (0-based) has relevance ``n - 1 - pos`` and
    gain ``2**rel - 1``; DCG discounts position ``i`` (1-based) by
    ``log2(i + 1)``.  ``top_k`` is capped at the group size, so the
    ideal denominator is positive and identical orderings score 1.0.

    Raises:
        ValueError: on invalid orderings (see ``_ordering_contract``)
            or a non-integer ``top_k < 1``.

    Example::

        >>> ideal = ["c1", "c2", "c3"]
        >>> ndcg_at_k(ideal, ["c3", "c2", "c1"], 1)
        0.0
        >>> ndcg_at_k(ideal, ideal, 2)
        1.0
    """
    ideal, scored, ideal_pos = _ordering_contract(ideal_order, scored_order)
    top_k = min(validate_int("top_k", top_k, minimum=1), len(ideal))
    size = len(ideal)
    gains_scored = np.array(
        [
            2.0 ** (size - 1 - ideal_pos[candidate]) - 1.0
            for candidate in scored[:top_k]
        ]
    )
    gains_ideal = np.array(
        [2.0 ** (size - 1 - position) - 1.0 for position in range(top_k)]
    )
    return float(_dcg_at_k(gains_scored) / _dcg_at_k(gains_ideal))


def precision_at_k(
    ideal_order: Sequence[str], scored_order: Sequence[str], top_k: int
) -> float:
    """Share of the scored top-``top_k`` inside the ideal top-``top_k``.

    Relevant = the candidates occupying the ideal top-``top_k`` (the
    classic overlap precision); ``top_k`` is capped at the group size.

    Raises:
        ValueError: on invalid orderings or a non-integer
            ``top_k < 1``.

    Example::

        >>> ideal = ["c1", "c2", "c3"]
        >>> precision_at_k(ideal, ["c3", "c2", "c1"], 1)
        0.0
        >>> precision_at_k(ideal, ["c3", "c2", "c1"], 3)
        1.0
    """
    ideal, scored, _ = _ordering_contract(ideal_order, scored_order)
    top_k = min(validate_int("top_k", top_k, minimum=1), len(ideal))
    head = set(scored[:top_k])
    relevant = set(ideal[:top_k])
    return len(head & relevant) / top_k


def mean_reciprocal_rank(
    ideal_order: Sequence[str], scored_order: Sequence[str], top_k: int
) -> float:
    """Reciprocal position of the single best candidate (MRR@``top_k``).

    Relevance here is single-winner (the trading-domain reading: one
    candidate wins the group): ``ideal_order[0]``.  Returns ``1 /
    rank`` when that candidate sits within the scored top-``top_k``
    and 0.0 otherwise; ``top_k`` is capped at the group size.

    Raises:
        ValueError: on invalid orderings or a non-integer
            ``top_k < 1``.

    Example::

        >>> ideal = ["c1", "c2", "c3"]
        >>> round(mean_reciprocal_rank(
        ...     ideal, ["c3", "c2", "c1"], 3
        ... ), 12)
        0.333333333333
        >>> mean_reciprocal_rank(ideal, ["c3", "c2", "c1"], 1)
        0.0
    """
    ideal, scored, _ = _ordering_contract(ideal_order, scored_order)
    top_k = min(validate_int("top_k", top_k, minimum=1), len(ideal))
    best_candidate = ideal[0]
    for position in range(top_k):
        if scored[position] == best_candidate:
            return 1.0 / (position + 1.0)
    return 0.0


def spearman_in_group(
    ideal_order: Sequence[str], scored_order: Sequence[str], top_k: int
) -> float:
    """Spearman of ideal vs scored positions over the scored top-k head.

    Both orderings rank the same candidate set; the correlation runs
    over the candidates occupying the scored top-``top_k`` (their
    global ideal positions vs their head positions 0..k-1), so a
    perfect head match scores 1.0 and a perfectly inverted head
    scores -1.0.  ``top_k`` is capped at the group size and must stay
    >= 2 for a rank correlation to exist; ``top_k >= n`` degenerates
    to the full-ordering correlation.

    Raises:
        ValueError: on invalid orderings, a non-integer ``top_k < 1``,
            or ``top_k == 1`` (rank correlation undefined on one
            point).

    Example::

        >>> ideal = ["c1", "c2", "c3"]
        >>> spearman_in_group(ideal, ideal, 3)
        1.0
        >>> spearman_in_group(ideal, ["c3", "c2", "c1"], 3)
        -1.0
    """
    ideal, scored, ideal_pos = _ordering_contract(ideal_order, scored_order)
    top_k = min(validate_int("top_k", top_k, minimum=1), len(ideal))
    if top_k < 2:
        raise ValueError(
            "spearman needs top_k >= 2: a rank correlation over one "
            "point is undefined"
        )
    head = scored[:top_k]
    ideal_positions = np.array(
        [ideal_pos[candidate] for candidate in head], dtype=np.float64
    )
    scored_positions = np.arange(top_k, dtype=np.float64)
    correlation = spearmanr(ideal_positions, scored_positions)
    return float(correlation.statistic)


RANKING_METRICS: dict[str, RankingMetric] = {
    "ndcg@k": ndcg_at_k,
    "precision@k": precision_at_k,
    "mrr": mean_reciprocal_rank,
    "spearman": spearman_in_group,
}
"""Open per-group ranking-metric registry: name -> metric (table in
the module docstring).

Metrics consume the RAW model ordering -- pass
``CandidateGroup.scores``-derived orders, not normalized ones: within
one group the two induce the same order, and the registry's semantics
are pinned to raw scores so callers never have to guess.  ``mrr``
treats ``ideal_order[0]`` as the single relevant candidate; ``top_k``
is capped at the group size; ``spearman`` additionally requires
``top_k >= 2``.
"""


# ---------------------------------------------------------------------------
# cross-ranker comparison on the scored intersection
# ---------------------------------------------------------------------------
def _ideal_order_from_labels(
    candidates: list[str], labels: np.ndarray
) -> list[str]:
    """Ideal ordering of one group's candidates (best label first).

    NaN labels (untradeable candidates) sort last; label ties break by
    candidate id ascending -- the same deterministic tie-break the
    scored ordering uses, so even a fully tied group has one
    well-defined ideal order.
    """

    def sort_key(index: int) -> tuple[tuple[int, float], str]:
        label = float(labels[index])
        rank = (0, -label) if np.isfinite(label) else (1, 0.0)
        return rank, candidates[index]

    order = sorted(range(len(candidates)), key=sort_key)
    return [candidates[index] for index in order]


def _scored_order_from_scores(
    candidates: list[str], scores: np.ndarray
) -> list[str]:
    """Scored ordering of one group's candidates (best score first).

    Ties break by candidate id ascending, matching the oracles'
    deterministic ``(score desc, candidate id asc)`` tie-break.  Report
    scores are finite by construction (``CandidateGroup`` validation).
    """
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), candidates[index]),
    )
    return [candidates[index] for index in order]


def _report_score_lists(report: dict[str, Any]) -> list[list[float]]:
    """Per-group score lists of a ranker report (key auto-discovered).

    Pointwise and given-scores reports carry ``per_group_proba`` and
    lambdarank reports ``per_group_score``; a dict with neither is not
    a ranker report this layer can read.
    """
    for score_key in ("per_group_proba", "per_group_score"):
        if score_key in report:
            score_lists = report[score_key]
            if not isinstance(score_lists, list):
                raise ValueError(
                    f"report[{score_key!r}] must be a list of per-group "
                    "score lists"
                )
            return score_lists
    raise ValueError(
        "report carries neither per_group_proba nor per_group_score: "
        "cross-ranker comparison needs two rank_within_groups / "
        "rank_with_lambdarank reports"
    )


def compare_rankers_on_intersection(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    *,
    top_k: int = 2,
) -> dict[str, Any]:
    """Compare two ranker reports on their scored-group INTERSECTION.

    ``n_skipped`` is per-run: two rankers on the same data can end up
    with different ``scored_group_ids`` sets, and comparing
    :data:`RANKING_METRICS` over the full lists lets the skip pattern
    fake a difference.  This helper makes the safe path (pinned in the
    :func:`rank_within_groups` docstring and
    ``docs/how-to/07-choose-a-ranker.md``) the default path: every
    metric is computed per common group for both rankers, and the
    per-group paired differences go through
    :func:`paired_bootstrap_diff` for a CI on ``diff``.

    Args:
        report_a: ranker A's report from :func:`rank_within_groups`
            or :func:`rank_with_lambdarank` (needs ``scored_group_ids``,
            ``groups`` and the per-group score lists).
        report_b: ranker B's report, same contract.
        top_k: head size every metric runs at, >= 2 (``spearman``
            needs at least two head points).

    Returns:
        Dict with one entry per :data:`RANKING_METRICS` name plus
        ``groups_used`` (the common group ids actually compared,
        sorted).  Each metric entry holds ``a`` / ``b`` (mean
        per-group metric of ranker A / B over those groups), ``diff``
        (``a - b``, the bootstrap's mean paired difference), the
        :class:`BootstrapDiff` CI fields ``lo`` / ``hi`` / ``p_le0`` /
        ``n`` bootstrapped on the paired per-group differences, and
        ``n_common`` (the group count the means are taken over).
        Common groups with fewer than two candidates carry no ranking
        contrast and are excluded on both sides consistently, so the
        pairing for the bootstrap survives.

    Raises:
        ValueError: on an empty intersection, a common group ranking
            different candidate sets on the two sides, a report
            without ``scored_group_ids`` / ``groups`` / per-group
            score lists, an intersection with no ranking-usable
            groups, or a non-integer ``top_k < 2``.

    Example::

        Two reports over one scored world, one of which skipped a day
        the other scored: the comparison runs on the intersection.

        >>> day_one = CandidateGroup(
        ...     group_id="d1",
        ...     candidates=["d1_long", "d1_short"],
        ...     scores=np.array([0.9, 0.1]),
        ...     labels=np.array([2.0, 1.0]),
        ... )
        >>> day_two = CandidateGroup(
        ...     group_id="d2",
        ...     candidates=["d2_long", "d2_short"],
        ...     scores=np.array([0.8, 0.2]),
        ...     labels=np.array([1.5, 0.5]),
        ... )
        >>> full_report = rank_within_groups([day_one, day_two], top_s=1)
        >>> trimmed_report = rank_within_groups([day_one], top_s=1)
        >>> compared = compare_rankers_on_intersection(
        ...     full_report, trimmed_report
        ... )
        >>> compared["groups_used"]
        ['d1']
        >>> compared["ndcg@k"]["a"], compared["ndcg@k"]["diff"]
        (1.0, 0.0)
        >>> compared["ndcg@k"]["n_common"], compared["mrr"]["p_le0"]
        (1, 1.0)
    """
    top_k = validate_int("top_k", top_k, minimum=2)
    for report_name, report in (
        ("report_a", report_a),
        ("report_b", report_b),
    ):
        if not isinstance(report, dict):
            raise ValueError(
                f"{report_name} must be a ranker report dict, got "
                f"{type(report).__name__}"
            )
        missing = [
            key for key in ("scored_group_ids", "groups") if key not in report
        ]
        if missing:
            raise ValueError(
                f"{report_name} is missing {missing}: cross-ranker "
                "comparison needs rank_within_groups / "
                "rank_with_lambdarank reports"
            )
    scored_a = list(report_a["scored_group_ids"])
    scored_b = list(report_b["scored_group_ids"])
    common = sorted(set(scored_a) & set(scored_b))
    if not common:
        raise ValueError(
            "rankers share no scored group: the scored_group_ids "
            f"intersection is empty (a scored {len(scored_a)} groups, "
            f"b scored {len(scored_b)}); compare on common groups or "
            "check n_skipped first"
        )
    groups_a = dict(zip(scored_a, report_a["groups"], strict=True))
    groups_b = dict(zip(scored_b, report_b["groups"], strict=True))
    scores_a = dict(zip(scored_a, _report_score_lists(report_a), strict=True))
    scores_b = dict(zip(scored_b, _report_score_lists(report_b), strict=True))
    usable: list[Any] = []
    for group_id in common:
        group_a = groups_a[group_id]
        group_b = groups_b[group_id]
        if set(group_a.candidates) != set(group_b.candidates):
            raise ValueError(
                f"common group {group_id!r} ranks different candidate "
                "sets on the two sides: comparison needs one candidate "
                "universe per group"
            )
        if len(group_a.candidates) >= 2 and len(group_b.candidates) >= 2:
            usable.append(group_id)
    if not usable:
        raise ValueError(
            "no ranking-usable common group: every group in the "
            "intersection has fewer than two candidates, and the "
            "metrics are undefined there"
        )
    comparison: dict[str, Any] = {}
    for metric_name, metric in RANKING_METRICS.items():
        values_a: list[float] = []
        values_b: list[float] = []
        for group_id in usable:
            group_a = groups_a[group_id]
            group_b = groups_b[group_id]
            ideal_a = _ideal_order_from_labels(
                group_a.candidates, group_a.labels
            )
            ideal_b = _ideal_order_from_labels(
                group_b.candidates, group_b.labels
            )
            scored_order_a = _scored_order_from_scores(
                group_a.candidates, np.asarray(scores_a[group_id])
            )
            scored_order_b = _scored_order_from_scores(
                group_b.candidates, np.asarray(scores_b[group_id])
            )
            values_a.append(metric(ideal_a, scored_order_a, top_k))
            values_b.append(metric(ideal_b, scored_order_b, top_k))
        bootstrap = paired_bootstrap_diff(
            np.asarray(values_a, dtype=np.float64),
            np.asarray(values_b, dtype=np.float64),
        )
        comparison[metric_name] = {
            "a": float(np.mean(values_a)),
            "b": float(np.mean(values_b)),
            "diff": bootstrap["diff"],
            "lo": bootstrap["lo"],
            "hi": bootstrap["hi"],
            "p_le0": bootstrap["p_le0"],
            "n": bootstrap["n"],
            "n_common": len(usable),
        }
    comparison["groups_used"] = usable
    return comparison


# ---------------------------------------------------------------------------
# admission-bound score normalization
# ---------------------------------------------------------------------------
def rank_normalize_group(scores: np.ndarray) -> np.ndarray:
    """Within-group rank normalization for admission-bound scores.

    Scores fed to admission must be within-group normalized; raw
    cross-model probabilities are not comparable (a logit 0.9 and a
    kNN 0.7 mean different things).  Maps ONE group's raw scores to
    average-rank positions in ``[0, 1]`` -- best candidate 1.0, worst
    0.0, ties share the AVERAGE of their rank positions so a tie
    never leaks into the normalized scale (scipy ``rankdata`` with
    ``method="average"``, delegated to
    :func:`atropos.ml_ensemble.rank_normalize`).  A single-candidate
    group maps to 0.5 (a group of one candidate is real: a day where
    the filter passed one signal; it carries no ranking contrast, and
    the division by zero of the rank formula must never fire).

    Args:
        scores: 1-D finite raw scores of one group.

    Returns:
        Fresh 1-D float array of normalized scores (the input is
        never mutated).

    Raises:
        ValueError: if ``scores`` is empty or non-finite.

    Example::

        >>> rank_normalize_group(np.array([0.8, 0.9, 0.7])).tolist()
        [0.5, 1.0, 0.0]
        >>> rank_normalize_group(np.array([0.9, 0.9, 0.1])).tolist()
        [0.75, 0.75, 0.0]
        >>> rank_normalize_group(np.array([0.42])).tolist()
        [0.5]
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("cannot normalize an empty group")
    if not np.isfinite(flat).all():
        raise ValueError("scores must be finite before normalization")
    if flat.size == 1:
        return np.array([0.5])
    return rank_normalize(flat)


# ---------------------------------------------------------------------------
# EV-calibrated threshold (donor port)
# ---------------------------------------------------------------------------
def ev_threshold(
    proba: np.ndarray,
    exec_vals: np.ndarray,
    grid: tuple[float, ...] | list[float] = _EV_GRID,
) -> tuple[float, float, int]:
    """EV-calibrated probability threshold: ``(threshold, ev, n)``.

    Donor-corpus port (``engine.model.ranker.ev_threshold``): scans
    the grid PLUS the all-trades floor (``-inf``), takes trades with
    ``proba >= threshold`` (INCLUSIVE, unlike the house
    :func:`atropos.ml_training.pick_threshold`, which uses strict
    ``>`` over the bare grid and returns only the threshold), and
    returns the cut with the best mean executable value.  Ties
    resolve toward the smaller threshold (more trades); pairs with a
    non-finite ``proba`` or ``exec_vals`` entry are dropped.

    NOT an alias of :func:`atropos.ml_training.pick_threshold`:
    inclusive ``>=`` + the all-trades floor ``-inf`` + an ``(thr,
    ev, n)`` tuple, vs strict ``>`` + a bare float -- see the
    Differences test in ``tests/test_ranking.py``.

    Args:
        proba: predicted probabilities of the positive class.
        exec_vals: executable per-event returns of the same rows.
        grid: strictly increasing thresholds inside (0, 1).

    Returns:
        ``(threshold, ev, n_taken)``; the empty-after-drop input
        degrades to ``(-inf, 0.0, 0)``.

    Raises:
        ValueError: on mismatched lengths, an empty grid, a
            non-increasing grid, or grid entries outside (0, 1).

    Example::

        >>> ev_threshold(
        ...     np.array([0.2, 0.6, 0.8]),
        ...     np.array([0.12, 0.03, -0.06]),
        ...     grid=(0.5, 0.7),
        ... )
        (-inf, 0.03, 3)
    """
    proba_arr = np.asarray(proba, dtype=np.float64).ravel()
    exec_arr = np.asarray(exec_vals, dtype=np.float64).ravel()
    if proba_arr.shape != exec_arr.shape:
        raise ValueError(
            f"proba and exec_vals must have equal length, got "
            f"{proba_arr.size} and {exec_arr.size}"
        )
    if not isinstance(grid, (tuple, list)) or len(grid) == 0:
        raise ValueError("grid must be a non-empty tuple or list")
    levels = [validate_number("grid entry", level) for level in grid]
    for low, high in pairwise(levels):
        if not low < high:
            raise ValueError(f"grid must be strictly increasing, got {grid}")
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError("grid entries must lie inside (0, 1)")
    finite = np.isfinite(proba_arr) & np.isfinite(exec_arr)
    values, outcomes = proba_arr[finite], exec_arr[finite]
    if values.size == 0:
        return (-np.inf, 0.0, 0)
    best_threshold = -np.inf
    best_ev = float(outcomes.mean())
    best_n = int(values.size)
    for level in levels:
        mask = values >= level
        count = int(mask.sum())
        if count == 0:
            continue
        ev = float(outcomes[mask].mean())
        if ev > best_ev + 1e-12:
            best_threshold = float(level)
            best_ev = ev
            best_n = count
    return (best_threshold, best_ev, best_n)


# ---------------------------------------------------------------------------
# rule-table gate (donor port, pandas)
# ---------------------------------------------------------------------------
def fit_rule_table(
    train: pd.DataFrame,
    min_trades: int = 30,
    fallback: str = "zone:1.0",
) -> dict[tuple[str, str], str]:
    """Best rule per ``(regime_dir, side)`` bucket by train EV.

    Pandas port of the donor corpus' rule-table gate
    (``engine.model.ranker.fit_rule_table``): ``train`` needs the
    columns ``regime_dir`` / ``side`` / ``rule`` / ``r_net``; per
    bucket the best rule by mean net R wins when it clears
    ``min_trades`` rows (bucket size, NaN ``r_net`` included) and a
    positive EV; otherwise the bucket falls back to ``fallback``.
    EV ties break by rule name ascending -- deterministic.

    Args:
        train: train-fold rows with the four rule-table columns.
        min_trades: minimum bucket size for a rule to be adopted.
        fallback: rule adopted when no rule qualifies.

    Returns:
        Mapping ``(regime_dir, side) -> rule``.

    Raises:
        ValueError: on missing columns, a non-integer
            ``min_trades < 1`` or a non-string ``fallback``.

    Example::

        >>> train = pd.DataFrame({
        ...     "regime_dir": ["up", "up", "up", "up", "down", "down",
        ...                    "down"],
        ...     "side": ["long", "long", "long", "long", "short", "short",
        ...              "short"],
        ...     "rule": ["zone:1.0", "zone:1.0", "atr:2.0", "atr:2.0",
        ...              "zone:1.0", "atr:2.0", "atr:2.0"],
        ...     "r_net": [0.05, 0.02, -0.02, -0.01, 0.01, 0.04, 0.03],
        ... })
        >>> sorted(fit_rule_table(train, min_trades=2).items())
        [(('down', 'short'), 'atr:2.0'), (('up', 'long'), 'zone:1.0')]
    """
    min_trades = validate_int("min_trades", min_trades)
    if not isinstance(fallback, str):
        raise ValueError(f"fallback must be a string, got {fallback!r}")
    missing = [
        column
        for column in _RULE_TABLE_COLUMNS
        if column not in train.columns
    ]
    if missing:
        raise ValueError(f"train is missing columns {missing}")
    frame = train.loc[:, list(_RULE_TABLE_COLUMNS)].copy()
    frame["r_net"] = pd.to_numeric(frame["r_net"], errors="coerce")
    stats = frame.groupby(["regime_dir", "side", "rule"], sort=False).agg(
        trades=("rule", "size"), ev=("r_net", "mean")
    )
    table: dict[tuple[str, str], str] = {}
    for (regime, side), bucket in stats.groupby(
        level=["regime_dir", "side"], sort=False
    ):
        valid = bucket[bucket["ev"].notna()].sort_values(
            ["ev", "rule"], ascending=[False, True], kind="mergesort"
        )
        if (
            valid.empty
            or float(valid.iloc[0]["ev"]) <= 0.0
            or int(valid.iloc[0]["trades"]) < min_trades
        ):
            table[(str(regime), str(side))] = fallback
        else:
            table[(str(regime), str(side))] = str(valid.index[0][2])
    return table


def apply_rule_table(
    df: pd.DataFrame, table: dict[tuple[str, str], str]
) -> pd.DataFrame:
    """Rows whose realized rule matches their bucket's table rule.

    Pandas port of the donor gate: each row's ``(regime_dir, side)``
    is mapped through ``table``; rows without a table entry are
    dropped.  Returns a fresh frame (the input is not mutated).

    Args:
        df: rows with ``regime_dir`` / ``side`` / ``rule`` columns.
        table: mapping from :func:`fit_rule_table`.

    Returns:
        The filtered frame.

    Raises:
        ValueError: on missing columns or an empty table.

    Example::

        >>> table = {
        ...     ("up", "long"): "zone:1.0",
        ...     ("down", "short"): "atr:2.0",
        ... }
        >>> live = pd.DataFrame({
        ...     "regime_dir": ["up", "down", "up"],
        ...     "side": ["long", "short", "short"],
        ...     "rule": ["zone:1.0", "atr:2.0", "zone:1.0"],
        ... })
        >>> apply_rule_table(live, table)["rule"].tolist()
        ['zone:1.0', 'atr:2.0']
    """
    needed = ("regime_dir", "side", "rule")
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns {missing}")
    if not table:
        raise ValueError("table is empty: nothing to apply")
    mapping = {
        f"{regime}|{side}": rule for (regime, side), rule in table.items()
    }
    keys = df["regime_dir"].astype(str) + "|" + df["side"].astype(str)
    chosen = keys.map(mapping)
    keep = chosen.notna() & (chosen == df["rule"].astype(str))
    return df.loc[keep].copy()


# ---------------------------------------------------------------------------
# paired bootstrap of the ranker-vs-ranker difference (donor port)
# ---------------------------------------------------------------------------
def _embargo_keep_mask(
    n_pairs: int,
    current_fold: Any,
    group_ids: Sequence[Any],
    embargo_until_by_group: dict[Any, int],
) -> np.ndarray:
    """Keep-mask excluding pairs embargoed past ``current_fold``.

    One pair per position: the pair at position ``pos`` sits on group
    ``group_ids[pos]`` and is dropped when that group's
    ``embargo_until`` entry exceeds ``current_fold`` (the value is not
    clean for the fold the comparison is made at).  Ids without an
    entry carry no embargo and stay; metadata for ids absent from
    ``group_ids`` is ignored.

    Raises:
        ValueError: on a non-integer ``current_fold`` / embargo entry
            or a ``group_ids`` length not matching ``n_pairs``.
    """
    fold_ref = validate_int("current_fold", current_fold, minimum=0)
    id_list = list(group_ids)
    if len(id_list) != n_pairs:
        raise ValueError(
            f"group_ids length {len(id_list)} does not match "
            f"{n_pairs} per-group pairs"
        )
    keep = np.ones(n_pairs, dtype=bool)
    for position, group_key in enumerate(id_list):
        until = embargo_until_by_group.get(group_key)
        if until is None:
            continue
        until = validate_int("embargo_until entry", until, minimum=0)
        if until > fold_ref:
            keep[position] = False
    return keep


def paired_bootstrap_diff(
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 7,
    enforce_embargo: bool = False,
    current_fold: int | None = None,
    group_ids: Sequence[Any] | None = None,
    embargo_until_by_group: dict[Any, int] | None = None,
) -> BootstrapDiff:
    """Bootstrap CI for ``mean(arm_a) - mean(arm_b)`` (paired groups).

    Donor-corpus port (``engine.model.ranker.paired_bootstrap_diff``):
    ``arm_a`` / ``arm_b`` are per-group values of two rankers on the
    SAME groups (e.g. per-group NDCG from :data:`RANKING_METRICS`);
    NaN pairs are dropped and row indices resampled jointly.  Returns
    the mean difference, the 2.5 / 97.5 percentiles and the fraction
    of draws with ``diff <= 0`` (a one-sided pseudo-p-value).

    NOT an alias of
    :func:`atropos.ml_training.paired_bootstrap_t_stat` (raw paired
    diff CI + ``p_le0``, no t-stat): that variant resamples
    ``(exec, taken-flag)`` pairs and reports a t-statistic; this one
    reports the raw paired-difference CI with no t.

    Embargo: by default enforcement stays with the fold scheduler that
    produced the per-group values (the carried
    ``CandidateGroup.embargo_until`` is not consulted -- the
    historical caller-owned contract, bit-identical).  With
    ``enforce_embargo=True`` the helper applies the minimal check
    itself: pairs whose group's ``embargo_until`` exceeds
    ``current_fold`` are dropped from BOTH arms before resampling, so
    values not yet clean for the fold being decided cannot leak into
    the CI.  The result stays honest about it: ``n`` counts only the
    pairs that entered the resampling.

    Args:
        arm_a: per-group metric values of ranker A.
        arm_b: per-group metric values of ranker B (same groups).  The
            caller is responsible for the group set being comparable.
        n_boot: number of bootstrap draws, integer >= 2.
        seed: RNG seed, integer >= 0.
        enforce_embargo: opt-in embargo enforcement; False (default)
            keeps the caller-owned scheduling behavior exactly.
        current_fold: the fold the comparison is made at, integer
            >= 0; required when ``enforce_embargo`` is True.
        group_ids: group id of each arm position, aligned with
            ``arm_a`` / ``arm_b``; required when ``enforce_embargo``
            is True.
        embargo_until_by_group: mapping group id -> its
            ``embargo_until`` fold; ids without an entry carry no
            embargo; required when ``enforce_embargo`` is True.

    Returns:
        :class:`BootstrapDiff` (``diff``, ``lo``, ``hi``, ``p_le0``,
        ``n``); the empty-after-drop input (NaN pairs or
        embargo-violating pairs) degrades to an all-zero payload with
        ``p_le0 == 1.0``.

    Raises:
        ValueError: if ``n_boot`` is not an integer >= 2 or ``seed``
            is not an integer >= 0; if the embargo inputs are given
            without ``enforce_embargo=True`` or are incomplete under
            it; if ``group_ids`` does not match the arm length.

    Example::

        >>> arm = np.array([0.4, 0.6, 0.5])
        >>> identical = paired_bootstrap_diff(arm, arm, n_boot=50, seed=7)
        >>> identical["diff"], identical["lo"], identical["hi"]
        (0.0, 0.0, 0.0)
        >>> identical["p_le0"], identical["n"]
        (1.0, 3)
        >>> embargoed = paired_bootstrap_diff(
        ...     arm,
        ...     arm,
        ...     n_boot=50,
        ...     seed=7,
        ...     enforce_embargo=True,
        ...     current_fold=1,
        ...     group_ids=["d1", "d2", "d3"],
        ...     embargo_until_by_group={"d3": 4},
        ... )
        >>> embargoed["n"]
        2
    """
    n_boot = validate_int("n_boot", n_boot, minimum=2)
    seed = validate_int("seed", seed, minimum=0)
    values_a = np.asarray(arm_a, dtype=np.float64).ravel()
    values_b = np.asarray(arm_b, dtype=np.float64).ravel()
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"arm_a and arm_b must have equal length, got "
            f"{values_a.size} and {values_b.size}"
        )
    embargo_keep_mask: np.ndarray | None = None
    if enforce_embargo:
        if (
            current_fold is None
            or group_ids is None
            or embargo_until_by_group is None
        ):
            raise ValueError(
                "enforce_embargo=True needs current_fold, group_ids "
                "and embargo_until_by_group: which fold is decided, "
                "which pair sits on which group, which groups are "
                "embargoed until when"
            )
        embargo_keep_mask = _embargo_keep_mask(
            int(values_a.size),
            current_fold,
            group_ids,
            embargo_until_by_group,
        )
    elif (
        current_fold is not None
        or group_ids is not None
        or embargo_until_by_group is not None
    ):
        raise ValueError(
            "current_fold / group_ids / embargo_until_by_group only "
            "apply with enforce_embargo=True (passing them without "
            "the flag is a dead configuration)"
        )
    rng = np.random.default_rng(seed)
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    if embargo_keep_mask is not None:
        finite = finite & embargo_keep_mask
    values_a, values_b = values_a[finite], values_b[finite]
    size = int(values_a.size)
    if size == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_le0": 1.0, "n": 0}
    diffs = values_a - values_b
    indices = rng.integers(0, size, size=(n_boot, size))
    draws = diffs[indices].mean(axis=1)
    return {
        "diff": float(diffs.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "p_le0": float((draws <= 0).mean()),
        "n": size,
    }
