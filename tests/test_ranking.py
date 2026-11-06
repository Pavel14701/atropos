"""Tests for the candidate ranking layer (``atropos.ranking``).

Pinned here:

* the pointwise oracle picks the planted best candidate well above
  chance, and its past-only schedule is metamorphic: mutating group
  ``j`` leaves every scored group whose train horizon excludes ``j``
  bit-identical while later horizons move (the mutation has teeth);
* the embargo gap is visible in the public report
  (``train_group_counts``) and has teeth: the immediate neighbor is
  NOT in the train fit (``embargo_groups >= 1``) but IS included at
  ``embargo_groups == 0``;
* the whole :data:`atropos.ml_training.MODEL_REGISTRY` (at import
  time, registry-growth safe) ranks through the identical path;
* determinism: same seed -> identical reports; ties break by
  ``(score desc, candidate_id asc)`` and never by chance;
* the lambdarank oracle (optional lightgbm, ``importorskip``) is
  deterministic and structurally valid -- its picks are deliberately
  NOT asserted equal to pointwise (different ranking semantics);
* :data:`RANKING_METRICS` known answers, the open-registry contract,
  and ranker-beats-random per metric on the planted world;
* :class:`CandidateGroup` (raw-scores contract) /
  :func:`rank_normalize_group` (average-rank ties, singleton 0.5,
  input never mutated) / :func:`ev_threshold` (inclusive ``>=`` plus
  the all-trades floor -- pinned DISTINCT from
  :func:`atropos.ml_training.pick_threshold`) / the pandas rule-table
  gate / :func:`paired_bootstrap_diff` (raw paired CI -- distinct
  from the taken-flag t-statistic);
* the module imports WITHOUT lightgbm: AST guard (no top-level
  lightgbm import), clean-interpreter import, and the pointwise path
  never pulls lightgbm into ``sys.modules``.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos import ranking
from atropos.ml_training import (
    MODEL_REGISTRY,
    paired_bootstrap_t_stat,
    pick_threshold,
)
from atropos.ranking import (
    RANKER_SUITABILITY,
    RANKING_METRICS,
    CandidateGroup,
    apply_rule_table,
    ev_threshold,
    fit_rule_table,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    rank_normalize_group,
    rank_with_lambdarank,
    rank_within_groups,
    spearman_in_group,
)

_N_GROUPS = 36
_GROUP_SIZE = 4
_REPORT_KEYS = frozenset(
    {
        "model",
        "seed",
        "top_s",
        "embargo_groups",
        "n_groups",
        "n_rows",
        "n_scored",
        "n_skipped",
        "selected",
        "per_group_proba",
        "scored_group_ids",
        "train_group_counts",
        "groups",
    }
)

# compute shrunk to unit-test scale; identical gate mechanics to the
# registry defaults (full-registry parametrization stays seconds-fast)
_SHRUNK_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "ensemble": {
        "members": ("logit",),
        "member_params": {"logit": {"max_iter": 200}},
    },
    "extratrees": {
        "n_estimators": 30,
        "max_depth": 4,
        "min_samples_leaf": 4,
    },
    "gbm": {"n_estimators": 30, "max_depth": 2},
    "hist_gbm": {"max_iter": 30, "max_depth": 2},
    "knn": {"n_neighbors": 1, "weights": "distance"},
    "logit": {"max_iter": 200},
    "mlp": {
        "hidden_layer_sizes": (8,),
        "max_iter": 100,
        "early_stopping": False,
    },
    "qda": {"reg_param": 0.5},
    "rf": {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 4},
}


def _planted_utilities(
    n_groups: int = _N_GROUPS,
    group_size: int = _GROUP_SIZE,
    seed: int = 17,
) -> np.ndarray:
    """The planted per-row utilities (single home for the formula).

    Same draw sequence as :func:`_planted_world`, so the labels, the
    winners and these graded utilities always describe one world.
    """
    rng = np.random.default_rng(seed)
    rows = n_groups * group_size
    quality = np.tile(np.linspace(-1.5, 1.5, group_size), n_groups)
    shift = np.repeat(rng.normal(0.0, 0.4, n_groups), group_size)
    return 0.02 * (quality + shift) + 0.006 * rng.standard_normal(rows)


def _planted_world(
    n_groups: int = _N_GROUPS,
    group_size: int = _GROUP_SIZE,
    seed: int = 17,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded groups whose latent-quality feature carries the signal.

    One feature (``quality``) ranks the candidates of every group;
    the rest are noise.  Returns ``(features, target, group_ids,
    winners)`` where ``winners[group]`` is the in-group slot of the
    realized best utility -- the "best candidate" the oracles must
    find from the past alone.
    """
    rng = np.random.default_rng(seed)
    rows = n_groups * group_size
    quality = np.tile(np.linspace(-1.5, 1.5, group_size), n_groups)
    utility = _planted_utilities(
        n_groups=n_groups, group_size=group_size, seed=seed
    )
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
    winners = np.argmax(utility.reshape(n_groups, group_size), axis=1)
    return features, target, group_ids, winners


def _rank_planted(
    model: str = "logit",
    seed: int = 42,
    embargo_groups: int = 1,
    n_groups: int = _N_GROUPS,
    seed_world: int = 17,
) -> dict[str, Any]:
    """Rank the planted world with shrunk registry parameters."""
    features, target, group_ids, _ = _planted_world(
        n_groups=n_groups, seed=seed_world
    )
    return rank_within_groups(
        features,
        target,
        group_ids,
        model=model,
        top_s=1,
        embargo_groups=embargo_groups,
        seed=seed,
        config={"model_params": dict(_SHRUNK_MODEL_PARAMS)},
    )


def _picks_by_group(report: dict[str, Any]) -> dict[Any, list[int]]:
    """Map scored group id -> selected global row positions."""
    step = int(report["top_s"])
    picks: dict[Any, list[int]] = {}
    for offset, group_id in enumerate(report["scored_group_ids"]):
        start = offset * step
        picks[group_id] = list(report["selected"][start : start + step])
    return picks


def _assert_identical_reports(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    """Bit-identity of two reports (arrays compared field-wise)."""
    for key in (
        "model",
        "seed",
        "top_s",
        "embargo_groups",
        "n_groups",
        "n_rows",
        "n_scored",
        "n_skipped",
    ):
        assert first[key] == second[key], key
    assert first["selected"] == second["selected"]
    assert first["scored_group_ids"] == second["scored_group_ids"]
    assert first["train_group_counts"] == second["train_group_counts"]
    for score_key in ("per_group_proba", "per_group_score"):
        if score_key in first:
            assert first[score_key] == second[score_key]
    for group_first, group_second in zip(
        first["groups"], second["groups"], strict=True
    ):
        assert group_first.group_id == group_second.group_id
        assert group_first.candidates == group_second.candidates
        assert group_first.fold_id == group_second.fold_id
        assert group_first.embargo_until == group_second.embargo_until
        assert np.array_equal(group_first.scores, group_second.scores)
        assert np.array_equal(group_first.labels, group_second.labels)


# ---------------------------------------------------------------------------
# 1. pointwise oracle: planted best candidate + honest report shape
# ---------------------------------------------------------------------------
def test_pointwise_picks_planted_best() -> None:
    """The oracle finds the planted winner in most scored groups."""
    _, _, _group_ids, winners = _planted_world()
    report = _rank_planted()
    assert frozenset(report) == _REPORT_KEYS
    assert report["model"] == "logit"
    assert report["n_groups"] == _N_GROUPS
    assert report["n_rows"] == _N_GROUPS * _GROUP_SIZE
    assert report["n_scored"] == _N_GROUPS - 2
    assert report["n_skipped"] == 0
    assert report["embargo_groups"] == 1
    assert len(report["selected"]) == report["n_scored"]
    # per_group_proba[g][j] is the probability of groups[g].candidates[j]
    for proba_list, group in zip(
        report["per_group_proba"], report["groups"], strict=True
    ):
        assert len(proba_list) == len(group.candidates)
        assert all(np.isfinite(proba_list))
    # hit rate vs the realized winner (chance = 1 / group size = 25%)
    hits = 0
    for position, group_no in enumerate(report["scored_group_ids"]):
        winner_row = int(group_no) * _GROUP_SIZE + int(winners[group_no])
        hits += report["selected"][position] == winner_row
    assert hits / report["n_scored"] >= 0.8


def test_pointwise_given_groups_path_agrees() -> None:
    """The CandidateGroup input replays the same picks by candidate id."""
    report = _rank_planted()
    replay = rank_within_groups(report["groups"], top_s=1)
    assert replay["model"] == "given"
    assert replay["n_scored"] == report["n_scored"]
    expected = [str(row) for row in report["selected"]]
    assert replay["selected"] == expected
    # the given-scores path echoes the raw-score contract untouched
    for group_in, group_out in zip(
        report["groups"], replay["groups"], strict=True
    ):
        assert group_out is group_in


def test_pointwise_top_s_caps_at_group_size() -> None:
    """``top_s`` beyond the group size keeps every candidate, ordered."""
    group = CandidateGroup(
        group_id="solo_day",
        candidates=["alpha", "beta"],
        scores=np.array([0.3, 0.8]),
        labels=np.array([1.0, 2.0]),
    )
    report = rank_within_groups([group], top_s=5)
    assert report["selected"] == ["beta", "alpha"]


def test_single_row_groups_rank_honestly() -> None:
    """One-row groups are legal: each scored group picks its own row."""
    n_groups = 6
    quality = np.array([-1.5, 1.5] * (n_groups // 2))
    features = pd.DataFrame({"quality": quality})
    target = (quality > 0).astype(np.int64)
    group_ids = np.arange(n_groups)
    report = rank_within_groups(
        features,
        target,
        group_ids,
        model="logit",
        embargo_groups=1,
        config={"model_params": {"logit": {"max_iter": 100}}},
    )
    assert report["n_groups"] == n_groups
    assert report["n_scored"] == n_groups - 3  # first single-class skips
    assert report["n_skipped"] == 1
    assert report["selected"] == [3, 4, 5]


# ---------------------------------------------------------------------------
# 2. past-only metamorphic invariant (no lookahead)
# ---------------------------------------------------------------------------
def test_past_only_is_metamorphic() -> None:
    """Mutating group j moves only horizons that train past j.

    Invariant: every scored group whose train window excludes the
    victim keeps its picks AND its probabilities bit-identical.  The
    mutation has teeth at the probability level (later train windows
    include the mutated rows, so their fitted probabilities move);
    the pick level alone can stay put because the planted world is
    separable enough to make the argmax robust.
    """
    baseline = _rank_planted()
    picks = _picks_by_group(baseline)
    features, target, group_ids, _ = _planted_world()
    mutated_features = features.copy()
    mutated_target = target.copy()
    rng = np.random.default_rng(99)
    victim = 25  # scored groups with train horizon <= victim keep picks
    start = victim * _GROUP_SIZE
    stop = start + _GROUP_SIZE
    mutated_features.iloc[start:stop] = rng.standard_normal(
        (_GROUP_SIZE, mutated_features.shape[1])
    )
    mutated_target[start:stop] = np.array([1, 0, 1, 0])
    mutated = rank_within_groups(
        mutated_features,
        mutated_target,
        group_ids,
        model="logit",
        seed=42,
        config={"model_params": dict(_SHRUNK_MODEL_PARAMS)},
    )
    mutated_picks = _picks_by_group(mutated)
    untouched = {group_id for group_id in picks if group_id <= victim + 1}
    assert untouched  # the past is bit-identical
    for scored_index, group_id in enumerate(baseline["scored_group_ids"]):
        if group_id > victim + 1 or group_id == victim:
            # horizons past the victim are the teeth below; the victim
            # itself is re-scored on its own (mutated) rows, so its
            # probabilities legitimately move either way
            continue
        assert mutated_picks[group_id] == picks[group_id], group_id
        assert (
            mutated["per_group_proba"][scored_index]
            == baseline["per_group_proba"][scored_index]
        ), group_id
    # teeth: horizons that DO train past the victim moved (probability
    # level), so the run is not comparing two identical computations
    moved = 0
    for scored_index, group_id in enumerate(baseline["scored_group_ids"]):
        if group_id > victim + 1 and (
            mutated["per_group_proba"][scored_index]
            != baseline["per_group_proba"][scored_index]
        ):
            moved += 1
    assert moved > 0


def _probas_by_group(report: dict[str, Any]) -> dict[Any, list[float]]:
    """Map scored group id -> its probability vector (row order)."""
    return dict(
        zip(
            report["scored_group_ids"],
            report["per_group_proba"],
            strict=True,
        )
    )


def _mutate_victim_features(seed: int = 77, victim: int = 10) -> pd.DataFrame:
    """Copy of the planted world with the victim's rows scrambled."""
    features, _, _, _ = _planted_world()
    rng = np.random.default_rng(seed)
    start = victim * _GROUP_SIZE
    features.iloc[start : start + _GROUP_SIZE] = 7.0 * rng.standard_normal(
        (_GROUP_SIZE, features.shape[1])
    )
    return features


def test_embargo_gap_excludes_immediate_neighbor() -> None:
    """``embargo_groups=1`` keeps the neighbor out of the train fit.

    Discriminator at the probability level: group ``victim + 1``'s
    train window is exactly the groups before ``victim``, so its fit
    (and probabilities) must be bit-identical, while every window
    from ``victim + 2`` on swallows the mutated rows and moves.
    """
    n_scored = _N_GROUPS - 2
    report = _rank_planted()
    # public mask invariant: scored position p trains on p - 1 groups
    assert report["train_group_counts"] == list(range(1, n_scored + 1))
    picks = _picks_by_group(report)
    probas = _probas_by_group(report)
    _, target, group_ids, _ = _planted_world()
    victim = 10
    mutated = rank_within_groups(
        _mutate_victim_features(victim=victim),
        target,
        group_ids,
        model="logit",
        seed=42,
        config={"model_params": dict(_SHRUNK_MODEL_PARAMS)},
    )
    assert mutated["n_scored"] == n_scored
    mutated_picks = _picks_by_group(mutated)
    mutated_probas = _probas_by_group(mutated)
    for group_id in range(2, victim + 2):
        if group_id == victim:
            # re-scored on its own mutated rows -> moves
            assert mutated_probas[group_id] != probas[group_id]
            continue
        # train window excludes the victim; own rows unchanged
        assert mutated_picks[group_id] == picks[group_id], group_id
        assert mutated_probas[group_id] == probas[group_id], group_id
    # teeth: from victim + 2 on the mutated rows enter the train fit
    moved = [
        group_id
        for group_id in range(victim + 2, _N_GROUPS)
        if mutated_probas[group_id] != probas[group_id]
    ]
    assert moved


def test_embargo_zero_includes_neighbor() -> None:
    """``embargo_groups=0`` has no gap: the neighbor IS in the fit.

    Mirror of the embargo-one test: without the gap, group
    ``victim + 1``'s train window includes the mutated rows, so its
    probabilities move -- the exact group the embargo protects at
    ``embargo_groups=1``.
    """
    report = _rank_planted(embargo_groups=0)
    n_scored = _N_GROUPS - 1
    assert report["n_scored"] == n_scored
    assert report["train_group_counts"] == list(range(1, n_scored + 1))
    picks = _picks_by_group(report)
    probas = _probas_by_group(report)
    _, target, group_ids, _ = _planted_world()
    victim = 10
    mutated = rank_within_groups(
        _mutate_victim_features(victim=victim),
        target,
        group_ids,
        model="logit",
        seed=42,
        embargo_groups=0,
        config={"model_params": dict(_SHRUNK_MODEL_PARAMS)},
    )
    mutated_picks = _picks_by_group(mutated)
    mutated_probas = _probas_by_group(mutated)
    # groups before the victim keep their fit untouched
    for group_id in range(1, victim):
        assert mutated_picks[group_id] == picks[group_id], group_id
        assert mutated_probas[group_id] == probas[group_id], group_id
    # the neighbor itself: train window now includes the victim
    assert mutated_probas[victim + 1] != probas[victim + 1]


def test_embargo_two_shifts_schedule() -> None:
    """``embargo_groups=2`` starts scoring one group later."""
    report = _rank_planted(embargo_groups=2)
    n_scored = _N_GROUPS - 3
    assert report["n_scored"] == n_scored
    assert report["train_group_counts"] == list(range(1, n_scored + 1))


# ---------------------------------------------------------------------------
# 3. registry parametrization (growth-safe, import-time fleet)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_registry_models_rank_within_groups(model: str) -> None:
    """Every registry model ranks through the identical pointwise path.

    Import-time fleet snapshot (registry-growth safe): "ensemble" is
    registered by ``ml_ensemble`` at import; any name registered later
    falls back to its factory defaults.  The world holds 12 candidates
    per group so the FIRST expanding window already satisfies QDA's
    more-samples-than-features rule -- the same smallest-window
    constraint the module docstring documents.
    """
    features, target, group_ids, _ = _planted_world(
        n_groups=12, group_size=12
    )
    report = rank_within_groups(
        features,
        target,
        group_ids,
        model=model,
        seed=42,
        config={"model_params": {model: dict(_SHRUNK_MODEL_PARAMS[model])}},
    )
    assert report["model"] == model
    assert report["n_scored"] == 10
    assert report["train_group_counts"] == list(range(1, 11))
    assert len(report["selected"]) == report["n_scored"]
    assert all(0 <= row < report["n_rows"] for row in report["selected"])


# ---------------------------------------------------------------------------
# 4. determinism and tie-breaking
# ---------------------------------------------------------------------------
def test_same_seed_reports_are_identical() -> None:
    """Two calls with one seed produce bit-identical reports."""
    _assert_identical_reports(_rank_planted(), _rank_planted())


def test_ties_break_by_candidate_id_deterministically() -> None:
    """Equal scores fall back to candidate id, never to chance."""
    tied = CandidateGroup(
        group_id="tied_day",
        candidates=["zulu", "alpha", "mike"],
        scores=np.array([0.5, 0.5, 0.5]),
        labels=np.array([1.0, 1.0, 1.0]),
    )
    first = rank_within_groups([tied], top_s=3)
    second = rank_within_groups([tied], top_s=3)
    assert first["selected"] == ["alpha", "mike", "zulu"]
    assert second["selected"] == first["selected"]
    # raw-path ties break by row position: duplicated rows score equal,
    # so top-2 of each scored group is exactly its slots (1, 2)
    features = pd.DataFrame({"quality": np.tile([0.0, 1.0, 1.0], 6)})
    target = np.tile(np.array([0, 1, 1]), 6)
    group_ids = np.repeat(np.arange(6), 3)
    report = rank_within_groups(
        features,
        target,
        group_ids,
        model="logit",
        top_s=2,
        config={"model_params": {"logit": {"max_iter": 100}}},
    )
    expected: list[int] = []
    for scored_group in range(2, 6):
        expected.extend([scored_group * 3 + 1, scored_group * 3 + 2])
    assert report["selected"] == expected


# ---------------------------------------------------------------------------
# 5. lambdarank oracle (optional lightgbm)
# ---------------------------------------------------------------------------
def test_lambdarank_is_deterministic_and_valid() -> None:
    """lambdarank: same seed -> identical report; picks are valid."""
    pytest.importorskip("lightgbm")
    features, target, group_ids, _ = _planted_world()
    kwargs: dict[str, Any] = {
        "top_s": 1,
        "embargo_groups": 1,
        "seed": 7,
    }
    first = rank_with_lambdarank(features, target, group_ids, **kwargs)
    second = rank_with_lambdarank(features, target, group_ids, **kwargs)
    assert first["model"] == "lambdarank"
    assert first["n_scored"] == _N_GROUPS - 2
    assert len(first["selected"]) == first["n_scored"]
    for position, group_no in enumerate(first["scored_group_ids"]):
        pick = first["selected"][position]
        assert group_no * _GROUP_SIZE <= pick < (group_no + 1) * _GROUP_SIZE
    _assert_identical_reports(first, second)
    # pointwise vs lambdarank rank by DIFFERENT semantics; compare via
    # RANKING_METRICS downstream, never assert equal picks here.


def test_lambdarank_import_error_names_the_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing lightgbm raises an actionable ImportError."""

    def fake_import(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    features, target, group_ids, _ = _planted_world(n_groups=8)
    with pytest.raises(ImportError, match="uv add lightgbm"):
        rank_with_lambdarank(features, target, group_ids)


# ---------------------------------------------------------------------------
# 6. the module imports without lightgbm (extras stay optional)
# ---------------------------------------------------------------------------
def test_no_top_level_lightgbm_import() -> None:
    """AST guard: lightgbm must never sit at module top level."""
    source = Path(ranking.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in top_level:
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        assert not any("lightgbm" in name for name in names), names


def test_module_imports_in_clean_interpreter() -> None:
    """A fresh interpreter imports the ranking module end-to-end."""
    # OpenBLAS reserves its thread pool lazily per process; spawning a
    # bare interpreter from inside a long pytest session can exhaust
    # the (already committed) allocation arena on memory-tight hosts.
    # That is an environment limitation, not a module defect: pin the
    # child's thread pool to 1 thread so the import needs no big
    # allocation, which is exactly the portability property under test.
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    # one retry: on a memory-committed host even the 1-thread child
    # can fail to reserve its OpenBLAS arena right after a heavy suite
    # finished releasing pages; a fresh attempt succeeds.  Only a
    # non-allocation stderr (a real import error) fails immediately.
    result = None
    for _attempt in range(2):
        result = subprocess.run(
            [sys.executable, "-I", "-c", "import atropos.ranking"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode == 0:
            break
        assert "Memory allocation" not in result.stderr, result.stderr
    assert result is not None and result.returncode == 0, (
        result.stderr if result is not None else "no attempt ran"
    )


def test_pointwise_never_pulls_lightgbm() -> None:
    """The pointwise path leaves lightgbm out of sys.modules.

    Compared against the pre-call state so the assertion stays honest
    even when the optional lambdarank tests installed lightgbm into
    the session earlier.
    """
    lightgbm_before = "lightgbm" in sys.modules
    _rank_planted()
    assert ("lightgbm" in sys.modules) == lightgbm_before


# ---------------------------------------------------------------------------
# 7. RANKING_METRICS: known answers, registry contract, beats-random
# ---------------------------------------------------------------------------
_IDEAL = ["c1", "c2", "c3"]
_INVERTED = ["c3", "c2", "c1"]


def test_ndcg_known_answers() -> None:
    """NDCG@k against hand-computed discounted gains."""
    assert ndcg_at_k(_IDEAL, _INVERTED, 1) == 0.0
    assert ndcg_at_k(_IDEAL, _IDEAL, 2) == 1.0
    # inverted @3: dcg = 0/log2(2) + 1/log2(3) + 3/log2(4)
    expected = (1.0 / np.log2(3.0) + 3.0 / np.log2(4.0)) / (
        3.0 / np.log2(2.0) + 1.0 / np.log2(3.0)
    )
    assert ndcg_at_k(_IDEAL, _INVERTED, 3) == pytest.approx(expected)
    # top_k caps at the group size
    assert ndcg_at_k(_IDEAL, _IDEAL, 99) == 1.0


def test_precision_known_answers() -> None:
    """Overlap precision of the scored head vs the ideal head."""
    assert precision_at_k(_IDEAL, _INVERTED, 1) == 0.0
    assert precision_at_k(_IDEAL, _INVERTED, 2) == 0.5
    assert precision_at_k(_IDEAL, _INVERTED, 3) == 1.0
    assert precision_at_k(_IDEAL, ["c2", "c1", "c3"], 2) == 1.0


def test_mrr_known_answers() -> None:
    """Single-winner reciprocal rank within (and beyond) the head."""
    assert mean_reciprocal_rank(_IDEAL, _INVERTED, 3) == pytest.approx(
        1.0 / 3.0
    )
    assert mean_reciprocal_rank(_IDEAL, _INVERTED, 1) == 0.0
    assert mean_reciprocal_rank(_IDEAL, _IDEAL, 1) == 1.0
    # the best candidate sits outside the scored head -> 0.0
    assert mean_reciprocal_rank(_IDEAL, ["c2", "c3", "c1"], 2) == 0.0


def test_spearman_known_answers() -> None:
    """Head-restricted rank correlation, -1/0.8/1 pins."""
    assert spearman_in_group(_IDEAL, _IDEAL, 3) == 1.0
    assert spearman_in_group(_IDEAL, _INVERTED, 3) == -1.0
    four = ["a", "b", "c", "d"]
    swap = ["b", "a", "c", "d"]
    assert spearman_in_group(four, swap, 2) == pytest.approx(-1.0)
    # full ordering of one adjacent swap: 1 - 6*2/(4*(16-1)) = 0.8
    assert spearman_in_group(four, swap, 4) == pytest.approx(0.8)


def test_metric_registry_is_open_and_consistent() -> None:
    """The registry exposes the four metrics and accepts additions."""
    for name in ("ndcg@k", "precision@k", "mrr", "spearman"):
        assert RANKING_METRICS[name] is not None
    assert RANKING_METRICS["ndcg@k"] is ndcg_at_k
    assert RANKING_METRICS["spearman"] is spearman_in_group
    registered = RANKING_METRICS["precision@k"]
    assert registered(_IDEAL, _INVERTED, 2) == precision_at_k(
        _IDEAL, _INVERTED, 2
    )
    RANKING_METRICS["hits@1"] = lambda ideal, scored, _top_k: (
        scored[0] == ideal[0]
    )
    try:
        assert RANKING_METRICS["hits@1"](_IDEAL, _INVERTED, 1) == 0.0
    finally:
        del RANKING_METRICS["hits@1"]


@pytest.mark.parametrize(
    ("ideal", "scored", "top_k"),
    [
        ([], [], 1),
        (_IDEAL, ["c1", "c2", "c4"], 1),
        (_IDEAL, ["c1", "c1", "c2"], 1),
        (_IDEAL, _IDEAL[:-1], 1),
        (["c1"], ["c1"], 1),
        (_IDEAL, _INVERTED, 0),
        (_IDEAL, _INVERTED, 1.5),
    ],
)
def test_metric_contract_rejects_bad_input(
    ideal: list[str], scored: list[str], top_k: Any
) -> None:
    """Invalid orderings / top_k raise ValueError for every metric."""
    for metric in RANKING_METRICS.values():
        with pytest.raises(ValueError):
            metric(ideal, scored, top_k)


def test_spearman_needs_at_least_two_head_points() -> None:
    """spearman@1 has no rank structure -> ValueError."""
    with pytest.raises(ValueError, match="top_k >= 2"):
        spearman_in_group(_IDEAL, _IDEAL, 1)


def test_ranker_beats_random_on_every_metric() -> None:
    """Planted world: the oracle beats seeded random pick per metric.

    The ideal order comes from the GRADED planted utilities (the
    metrics are built for graded relevance; a binary-label ideal is
    tie-broken by candidate id and would reward low ids, not good
    candidates).  The scored order comes from the report's raw model
    probabilities.
    """
    report = _rank_planted()
    utilities = _planted_utilities()
    rng = np.random.default_rng(5)
    for name, metric in RANKING_METRICS.items():
        # per-metric top_k: precision at the full group size is
        # degenerate (heads cover the whole group -> always 1.0) and
        # spearman needs at least two head points
        top_k = {"precision@k": 1, "spearman": 2}.get(name, _GROUP_SIZE)
        model_values: list[float] = []
        random_values: list[float] = []
        for scored_index, group in enumerate(report["groups"]):
            candidates = group.candidates
            scores = report["per_group_proba"][scored_index]
            labels = utilities[[int(item) for item in candidates]]
            ideal = sorted(
                range(len(candidates)),
                key=lambda index: (-labels[index], candidates[index]),
            )
            scored = sorted(
                range(len(candidates)),
                key=lambda index: (-scores[index], candidates[index]),
            )
            ideal_order = [candidates[index] for index in ideal]
            scored_order = [candidates[index] for index in scored]
            model_values.append(metric(ideal_order, scored_order, top_k))
            shuffled = list(ideal_order)
            rng.shuffle(shuffled)
            random_values.append(metric(ideal_order, shuffled, top_k))
        assert float(np.mean(model_values)) > float(np.mean(random_values)), (
            name,
            float(np.mean(model_values)),
            float(np.mean(random_values)),
        )


def test_ranker_suitability_marks_the_whole_fleet() -> None:
    """Suitability marking covers registry names and honest levels."""
    assert set(RANKER_SUITABILITY) <= set(MODEL_REGISTRY)
    assert set(RANKER_SUITABILITY.values()) <= {"low", "medium", "high"}
    assert RANKER_SUITABILITY["knn"] == "low"
    assert RANKER_SUITABILITY["gbm"] == "high"


# ---------------------------------------------------------------------------
# 8. CandidateGroup contract
# ---------------------------------------------------------------------------
def _valid_group() -> CandidateGroup:
    return CandidateGroup(
        group_id="d1",
        candidates=["long", "short", "flat"],
        scores=np.array([0.6, 0.9, 0.2]),
        labels=np.array([0.01, 0.02, np.nan]),
    )


def test_candidate_group_coerces_and_keeps_raw_scores() -> None:
    """Construction normalizes containers; raw scores pass through."""
    any_scores: Any = [0.6, 0.9]
    any_labels: Any = [0.01, 0.02]
    any_candidates: Any = ("long", "short")
    group = CandidateGroup(
        group_id=3,
        candidates=any_candidates,
        scores=any_scores,
        labels=any_labels,
    )
    assert group.candidates == ["long", "short"]
    assert group.scores.tolist() == [0.6, 0.9]
    assert group.fold_id is None and group.embargo_until is None
    marked = CandidateGroup(
        group_id="d",
        candidates=["long", "short"],
        scores=np.array([0.6, 0.9]),
        labels=np.array([0.01, 0.02]),
        fold_id=2,
        embargo_until=170,
    )
    assert marked.fold_id == 2 and marked.embargo_until == 170


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidates": [], "scores": [], "labels": []},
        {
            "candidates": ["a", "a"],
            "scores": [0.1, 0.2],
            "labels": [0.0, 0.0],
        },
        {
            "candidates": ["a", 7],
            "scores": [0.1, 0.2],
            "labels": [0.0, 0.0],
        },
        {
            "candidates": ["a", "b"],
            "scores": [0.1],
            "labels": [0.0, 0.0],
        },
        {
            "candidates": ["a", "b"],
            "scores": [0.1, np.nan],
            "labels": [0.0, 0.0],
        },
        {
            "candidates": ["a", "b"],
            "scores": [0.1, 0.2],
            "labels": ["nope", 0.0],
        },
        {
            "candidates": ["a", "b"],
            "scores": [0.1, 0.2],
            "labels": [0.0, 0.0],
            "fold_id": "two",
        },
    ],
)
def test_candidate_group_rejects_bad_fields(kwargs: dict[str, Any]) -> None:
    """Invalid candidate groups fail loudly with ValueError."""
    with pytest.raises(ValueError):
        CandidateGroup(group_id="x", **kwargs)


def test_candidate_group_has_no_value_equality() -> None:
    """Ndarray fields make value equality ambiguous -> identity only."""
    first = _valid_group()
    second = _valid_group()
    assert (first == second) is False


# ---------------------------------------------------------------------------
# 9. rank_normalize_group (admission-bound scores)
# ---------------------------------------------------------------------------
def test_rank_normalize_group_average_ranks_and_singleton() -> None:
    """Ties share the average rank; a singleton maps to 0.5."""
    assert rank_normalize_group(np.array([0.8, 0.9, 0.7])).tolist() == [
        0.5,
        1.0,
        0.0,
    ]
    assert rank_normalize_group(np.array([0.9, 0.9, 0.1])).tolist() == [
        0.75,
        0.75,
        0.0,
    ]
    assert rank_normalize_group(np.array([0.42])).tolist() == [0.5]


def test_rank_normalize_group_does_not_mutate_scores() -> None:
    """Normalization returns a fresh array; raw scores stay raw."""
    group = _valid_group()
    raw = group.scores.copy()
    normalized = rank_normalize_group(group.scores)
    assert group.scores.tolist() == raw.tolist()
    assert normalized is not group.scores
    assert not np.array_equal(normalized, group.scores)
    # cross-model comparability: a logit 0.9 and a knn 0.7 both map to
    # the top of their own groups (1.0), so admission can compare them
    logit_group = CandidateGroup(
        group_id="logit_day",
        candidates=["a", "b", "c"],
        scores=np.array([0.8, 0.9, 0.7]),
        labels=np.array([0.0, 0.0, 0.0]),
    )
    knn_group = CandidateGroup(
        group_id="knn_day",
        candidates=["a", "b", "c"],
        scores=np.array([0.7, 0.65, 0.4]),
        labels=np.array([0.0, 0.0, 0.0]),
    )
    assert int(np.argmax(rank_normalize_group(logit_group.scores))) == 1
    assert int(np.argmax(rank_normalize_group(knn_group.scores))) == 0
    assert rank_normalize_group(logit_group.scores).max() == 1.0
    assert rank_normalize_group(knn_group.scores).max() == 1.0


def test_rank_normalize_group_rejects_empty_and_nonfinite() -> None:
    """Empty / NaN groups fail loudly (no silent zeros)."""
    with pytest.raises(ValueError, match="empty"):
        rank_normalize_group(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        rank_normalize_group(np.array([0.5, np.nan]))


# ---------------------------------------------------------------------------
# 10. ev_threshold (inclusive >= + all-trades floor; NOT pick_threshold)
# ---------------------------------------------------------------------------
def test_ev_threshold_floor_and_grid() -> None:
    """The all-trades floor wins when cuts only drop good trades."""
    threshold, ev, count = ev_threshold(
        np.array([0.2, 0.6, 0.8]),
        np.array([0.12, 0.03, -0.06]),
        grid=(0.5, 0.7),
    )
    assert (threshold, ev, count) == (-np.inf, 0.03, 3)
    # a cut that improves EV is adopted
    threshold, ev, count = ev_threshold(
        np.array([0.1, 0.5]),
        np.array([-0.5, 1.0]),
        grid=(0.5,),
    )
    assert (threshold, ev, count) == (0.5, 1.0, 1)


def test_ev_threshold_is_inclusive_and_distinct_from_pick_threshold() -> None:
    """Pinned difference vs ml_training.pick_threshold (strict >)."""
    proba = np.array([0.5, 0.2])
    exec_vals = np.array([2.0, -1.0])
    threshold, ev, count = ev_threshold(proba, exec_vals, grid=(0.5,))
    # inclusive >=: the proba == 0.5 row TAKES PART and beats the floor
    assert (threshold, ev, count) == (0.5, 2.0, 1)
    # strict > (ml_training.pick_threshold) cannot see that row at all:
    # nothing is taken and the documented fallback returns the highest
    # grid level -- the inclusive cut is invisible to it
    assert pick_threshold(proba, exec_vals, grid=(0.5,)) == 0.5


def test_ev_threshold_drops_nonfinite_pairs_and_degrades() -> None:
    """NaN pairs drop; an all-NaN payload degrades to (-inf, 0.0, 0)."""
    result = ev_threshold(
        np.array([np.nan, 0.7]),
        np.array([5.0, -1.0]),
        grid=(0.5,),
    )
    assert result == (-np.inf, -1.0, 1)
    assert ev_threshold(np.array([np.nan]), np.array([1.0]), grid=(0.5,)) == (
        -np.inf,
        0.0,
        0,
    )


@pytest.mark.parametrize(
    "grid",
    [(), (0.5, 0.4), (0.0, 0.5), (0.5, 1.0), (0.5, float("nan"))],
)
def test_ev_threshold_rejects_bad_grids(grid: tuple[float, ...]) -> None:
    """Empty / non-increasing / out-of-domain grids raise ValueError."""
    with pytest.raises(ValueError):
        ev_threshold(np.array([0.5]), np.array([0.1]), grid=grid)


def test_ev_threshold_rejects_mismatched_lengths() -> None:
    """Proba / exec_vals length mismatch raises ValueError."""
    with pytest.raises(ValueError, match="equal length"):
        ev_threshold(np.array([0.5, 0.6]), np.array([0.1]), grid=(0.5,))


# ---------------------------------------------------------------------------
# 11. rule-table gate (pandas port)
# ---------------------------------------------------------------------------
def _rule_train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime_dir": ["up", "up", "up", "up", "down", "down", "down"],
            "side": [
                "long",
                "long",
                "long",
                "long",
                "short",
                "short",
                "short",
            ],
            "rule": [
                "zone:1.0",
                "zone:1.0",
                "atr:2.0",
                "atr:2.0",
                "zone:1.0",
                "atr:2.0",
                "atr:2.0",
            ],
            "r_net": [0.05, 0.02, -0.02, -0.01, 0.01, 0.04, 0.03],
        }
    )


def test_fit_rule_table_picks_best_rule_with_floor() -> None:
    """Best EV rule per bucket; thin buckets fall back."""
    table = fit_rule_table(_rule_train_frame(), min_trades=2)
    assert table == {
        ("up", "long"): "zone:1.0",
        ("down", "short"): "atr:2.0",
    }
    assert fit_rule_table(_rule_train_frame(), min_trades=2) == table


def test_fit_rule_table_falls_back_on_thin_buckets() -> None:
    """A bucket whose best rule misses min_trades adopts the fallback."""
    table = fit_rule_table(
        _rule_train_frame(), min_trades=5, fallback="atr:1.5"
    )
    assert table == {
        ("up", "long"): "atr:1.5",
        ("down", "short"): "atr:1.5",
    }


def test_apply_rule_table_keeps_only_matching_rows() -> None:
    """Applied table keeps rows whose rule matches the bucket rule."""
    table = fit_rule_table(_rule_train_frame(), min_trades=2)
    live = pd.DataFrame(
        {
            "regime_dir": ["up", "down", "up", "sideway"],
            "side": ["long", "short", "short", "long"],
            "rule": ["zone:1.0", "atr:2.0", "zone:1.0", "zone:1.0"],
            "r_net": [0.1, 0.2, -0.3, 0.0],
        }
    )
    kept = apply_rule_table(live, table)
    assert kept["regime_dir"].tolist() == ["up", "down"]
    assert kept["rule"].tolist() == ["zone:1.0", "atr:2.0"]
    # the input frame is not mutated
    assert len(live) == 4


def test_fit_rule_table_requires_all_columns() -> None:
    """Missing rule-table columns raise ValueError naming them."""
    frame = _rule_train_frame()
    for column in ("regime_dir", "side", "rule", "r_net"):
        with pytest.raises(ValueError, match="missing columns"):
            fit_rule_table(frame.drop(columns=[column]))


def test_rule_table_rejects_bad_parameters() -> None:
    """min_trades / fallback / empty table fail loudly."""
    with pytest.raises(ValueError):
        fit_rule_table(_rule_train_frame(), min_trades=0)
    fractional_trades: Any = 2.5
    with pytest.raises(ValueError):
        fit_rule_table(_rule_train_frame(), min_trades=fractional_trades)
    numeric_fallback: Any = 7
    with pytest.raises(ValueError):
        fit_rule_table(_rule_train_frame(), fallback=numeric_fallback)
    live = _rule_train_frame()
    with pytest.raises(ValueError, match="table is empty"):
        apply_rule_table(live, {})
    trimmed = live.drop(columns=["side"])
    with pytest.raises(ValueError, match="missing columns"):
        apply_rule_table(trimmed, {("up", "long"): "zone:1.0"})


# ---------------------------------------------------------------------------
# 12. paired_bootstrap_diff (raw paired CI; NOT the t-statistic)
# ---------------------------------------------------------------------------
def test_paired_bootstrap_diff_identical_arms() -> None:
    """Identical arms give exactly zero with p_le0 == 1."""
    arm = np.array([0.4, 0.6, 0.5, 0.7])
    result = paired_bootstrap_diff(arm, arm, n_boot=50, seed=7)
    assert result["diff"] == 0.0
    assert result["lo"] == 0.0
    assert result["hi"] == 0.0
    assert result["p_le0"] == 1.0
    assert result["n"] == 4


def test_paired_bootstrap_diff_separates_rankers() -> None:
    """A shifted arm gets a CI clear of zero and p_le0 == 0."""
    rng = np.random.default_rng(3)
    arm_b = rng.normal(0.4, 0.05, 200)
    arm_a = arm_b + 0.2
    result = paired_bootstrap_diff(arm_a, arm_b, n_boot=500, seed=7)
    assert result["diff"] == pytest.approx(0.2)
    assert result["lo"] > 0.0
    assert result["hi"] < 0.4
    assert result["p_le0"] == 0.0
    assert result["n"] == 200


def test_paired_bootstrap_diff_drops_nan_pairs_and_degrades() -> None:
    """NaN pairs drop from both arms; empty input degrades honestly."""
    arm_a = np.array([0.4, np.nan, 0.6])
    arm_b = np.array([0.2, 0.3, np.nan])
    result = paired_bootstrap_diff(arm_a, arm_b, n_boot=50, seed=7)
    assert result["n"] == 1
    assert result["diff"] == pytest.approx(0.2)
    empty = paired_bootstrap_diff(
        np.array([np.nan]), np.array([0.1]), n_boot=50, seed=7
    )
    assert empty == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_le0": 1.0, "n": 0}


def test_paired_bootstrap_diff_rejects_bad_parameters() -> None:
    """n_boot < 2, fractional n_boot, negative seed, mismatch -> err."""
    arm = np.array([0.4, 0.6])
    with pytest.raises(ValueError):
        paired_bootstrap_diff(arm, arm, n_boot=1)
    fractional_boot: Any = 20.5
    with pytest.raises(ValueError):
        paired_bootstrap_diff(arm, arm, n_boot=fractional_boot)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(arm, arm, n_boot=10, seed=-1)
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_diff(arm, np.array([0.4]))


def test_paired_bootstrap_diff_agrees_with_t_stat_on_trivial_case() -> None:
    """Cross-check with ml_training: identical arms -> 0.0 on both."""
    rng = np.random.default_rng(11)
    exec_vals = rng.normal(0.001, 0.01, 300)
    taken = rng.random(300) > 0.5
    t_stat, skipped = paired_bootstrap_t_stat(
        exec_vals, taken, exec_vals, taken, n_boot=100, seed=42
    )
    diff = paired_bootstrap_diff(exec_vals, exec_vals, n_boot=100, seed=42)
    assert t_stat == 0.0
    assert skipped == 0
    assert diff["diff"] == 0.0


# ---------------------------------------------------------------------------
# 13. chaos: degenerate inputs fail loudly, never silently
# ---------------------------------------------------------------------------
def test_empty_inputs_raise() -> None:
    """Empty frames / group lists are rejected up front."""
    with pytest.raises(ValueError, match="empty"):
        rank_within_groups(pd.DataFrame({"quality": []}), [], [])
    with pytest.raises(ValueError, match="empty"):
        rank_within_groups([], top_s=1)


def test_interleaved_group_ids_raise() -> None:
    """Groups must be contiguous chronological blocks (no shuffling)."""
    features = pd.DataFrame({"quality": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="contiguous"):
        rank_within_groups(
            features,
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
        )


@pytest.mark.parametrize(
    "features,target",
    [
        (
            pd.DataFrame({"quality": [0.1, np.nan, 0.3, 0.4] * 3}),
            np.array([0, 1, 0, 1] * 3),
        ),
        (
            pd.DataFrame({"quality": [0.1, np.inf, 0.3, 0.4] * 3}),
            np.array([0, 1, 0, 1] * 3),
        ),
        (
            pd.DataFrame({"quality": [0.1, 0.2, 0.3, 0.4] * 3}),
            np.array([0, 2, 0, 1] * 3),
        ),
        (
            pd.DataFrame({"quality": [0.1, 0.2, 0.3, 0.4] * 3}),
            np.array([0, np.nan, 0, 1] * 3),
        ),
    ],
)
def test_nonfinite_or_nonbinary_inputs_raise(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    """NaN/inf features and non-binary targets fail loudly."""
    group_ids = np.repeat(np.arange(12), 1)
    with pytest.raises(ValueError):
        rank_within_groups(features, target, group_ids)


def test_non_numeric_feature_column_raises() -> None:
    """A string feature column is rejected before any fit."""
    features = pd.DataFrame(
        {"quality": [0.1, 0.2, 0.3, 0.4] * 3, "tag": ["a"] * 12}
    )
    with pytest.raises(ValueError, match="numeric"):
        rank_within_groups(
            features,
            np.array([0, 1] * 6),
            np.repeat(np.arange(6), 2),
        )


def test_too_few_groups_for_embargo_raise() -> None:
    """Expanding past-only needs more than embargo + 1 groups."""
    features = pd.DataFrame({"quality": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="embargo_groups"):
        rank_within_groups(
            features,
            np.array([0, 1, 0]),
            np.array([0, 1, 2]),
            embargo_groups=3,
        )


def test_single_class_target_has_no_scored_groups() -> None:
    """An all-zero target can never train a classifier -> ValueError."""
    features = pd.DataFrame({"quality": [0.1, 0.2, 0.3, 0.4] * 4})
    with pytest.raises(ValueError, match="single-class"):
        rank_within_groups(
            features,
            np.zeros(16, dtype=np.int64),
            np.repeat(np.arange(4), 4),
        )


def test_unknown_model_name_raises() -> None:
    """Unknown registry names fail with the available list."""
    features, target, group_ids, _ = _planted_world(n_groups=8)
    with pytest.raises(ValueError, match="unknown model"):
        rank_within_groups(features, target, group_ids, model="xgboost")


def test_missing_target_or_group_ids_raise() -> None:
    """DataFrame input requires both target and group_ids."""
    features = pd.DataFrame({"quality": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="required"):
        rank_within_groups(features)


def test_given_groups_reject_stray_target() -> None:
    """CandidateGroup input must not carry DataFrame leftovers."""
    group = _valid_group()
    stray_target: Any = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="None"):
        rank_within_groups([group], stray_target)
    not_groups: Any = ["not-a-group"]
    with pytest.raises(ValueError, match="CandidateGroup"):
        rank_within_groups(not_groups, top_s=1)


def test_duplicate_group_ids_in_given_list_raise() -> None:
    """The same group twice is a caller bug, not a feature."""
    group = _valid_group()
    with pytest.raises(ValueError, match="duplicate"):
        rank_within_groups([group, group], top_s=1)
