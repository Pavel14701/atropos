"""Feature-scan pins: planted worlds, provably failable both ways.

Every world is a fixed-seed (42-45) synthetic with a planted effect,
so each test below fails loudly under the mutation it targets.  The
docstring of each test states WHAT it pins and WHICH mutation must be
caught by it (the mutation plan; every mutation is applied -> fail ->
restored before commit).

Checkpoint note: this file grows with the module -- A1 lands tests
1-9 + the two classification pins (proxy / shape-unknown) + the
schema freeze; A2 adds 10-14 (regime + threshold), A3 adds 15-22
(auto tools).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.feature_scan import (
    FeatureAnalysisReport,
    bucket_scan,
    correlation_cluster,
    feature_analysis,
    interaction_scan,
    missingness_scan,
    rule_fit,
    threshold_scan,
)

_N_ROWS = 800

_FA_REPORT_KEYS = frozenset(
    {
        "label",
        "seed",
        "n_effective",
        "orthogonal_t_gate",
        "missingness",
        "clustering",
        "orthogonal",
        "buckets",
        "interaction",
        "rules",
        "regime",
        "threshold",
        "discovery",
        "importance",
        "rule_generation",
        "skipped_features",
        "classification",
        "recommended_actions",
        "notes",
    }
)


def _draws(seed: int, count: int, n_rows: int = _N_ROWS) -> list[np.ndarray]:
    """``count`` independent standard-normal columns from one rng."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n_rows) for _ in range(count)]


def _uniform(seed: int, n_rows: int = _N_ROWS) -> np.ndarray:
    """One uniform(0, 1) column (bucket worlds need full-range x)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, n_rows)


# ---------------------------------------------------------------------------
# Group A -- defensive scans
# ---------------------------------------------------------------------------
def test_nan_not_random_detected() -> None:
    """NaN rows carrying outcome information are flagged (test 1).

    World: the feature is NaN exactly where the outcome is high, so
    the observed rows systematically miss the right tail.  Pins:
    nan_fraction > gate, ks_p significant, nan_outcome_differs True,
    warn True.  Mutation fact: dropping the KS comparison (or OR-ing
    the warn without it) flips nan_outcome_differs/warn to False and
    this test fails.
    """
    real, _unused_null, control, noise = _draws(42, 4, 500)
    # control enters the outcome at weight 1.0 so the NaN mask (drawn
    # from the outcome threshold) links to the control well above 0.3
    outcome_values = 0.5 * real + 1.0 * control + noise
    idx = pd.RangeIndex(500)
    outcome = pd.Series(outcome_values, index=idx)
    flagged = pd.Series(real, index=idx)
    flagged[outcome_values > 1.0] = np.nan
    report = missingness_scan(
        outcome,
        {"flagged": flagged},
        {"gamma_mom": pd.Series(control, index=idx)},
        nan_threshold=0.05,
        label="nan-linked",
    )
    entry = report["features"]["flagged"]
    assert entry["nan_fraction"] > 0.05
    assert np.isfinite(entry["ks_p"])
    assert entry["ks_p"] < 0.05
    assert entry["nan_outcome_differs"] is True
    assert entry["warn"] is True
    # the NaN indicator is outcome-driven, hence control-driven too
    assert np.isfinite(entry["nan_correlates_with"]["gamma_mom"])
    assert abs(entry["nan_correlates_with"]["gamma_mom"]) >= 0.3


def test_nan_random_not_flagged() -> None:
    """Random NaN at the same fraction stays silent (test 2).

    World: 15% of rows missing uniformly at random -- above the warn
    gate, so the warn decision itself is exercised.  Pins:
    nan_outcome_differs False and warn False.  Mutation fact: a warn
    that fires on fraction alone (dropping the differs-or-linked
    conjunct) makes warn True here and fails the test.
    """
    _unused_real, _unused_null, control, noise = _draws(43, 4, 500)
    outcome_values = 0.4 * control + noise
    idx = pd.RangeIndex(500)
    outcome = pd.Series(outcome_values, index=idx)
    rng = np.random.default_rng(44)
    scattered = pd.Series(control, index=idx)
    scattered[rng.random(500) < 0.15] = np.nan
    report = missingness_scan(
        outcome,
        {"scattered": scattered},
        {"gamma_mom": pd.Series(control, index=idx)},
        nan_threshold=0.05,
        label="nan-random",
    )
    entry = report["features"]["scattered"]
    assert entry["nan_fraction"] > 0.05
    assert entry["nan_outcome_differs"] is False
    assert entry["warn"] is False


def test_cluster_representative_correct() -> None:
    """Cluster count, Bonferroni split and representatives (test 3).

    World: chained correlations -- a1~a2 at 0.9, a1~b1 at 0.65,
    b1~b2 at 0.65 -- so at the 0.8 cut the clusters are {a1,a2},
    {b1}, {b2} (n_effective 3, Bonferroni 1/60) while a silent
    tightening of the cut to 0.5 collapses everything into one
    cluster.  Also pins the explicit representative override and its
    three failure modes.  Mutation fact: hard-coding threshold=0.5
    (or clustering on n_raw) changes n_effective / bonferroni_threshold
    and fails the assertions.
    """
    base, extra_a, extra_b1, extra_b2 = _draws(42, 4, 600)
    outcome_values = base + 0.5 * extra_b2
    idx = pd.RangeIndex(600)
    outcome = pd.Series(outcome_values, index=idx)
    a1 = pd.Series(base, index=idx)
    a2 = pd.Series(0.9 * base + 0.436 * extra_a, index=idx)
    b1 = pd.Series(0.65 * base + 0.76 * extra_b1, index=idx)
    b2 = pd.Series(0.65 * b1.to_numpy() + 0.76 * extra_b2, index=idx)
    report = correlation_cluster(
        {"a1": a1, "a2": a2, "b1": b1, "b2": b2},
        outcome,
        threshold=0.8,
        label="clusters",
    )
    assert report["n_raw"] == 4
    assert report["n_effective"] == 3
    assert report["bonferroni_threshold"] == pytest.approx(0.05 / 3)
    assert sorted(map(sorted, report["clusters"])) == [
        ["a1", "a2"],
        ["b1"],
        ["b2"],
    ]
    assert report["representatives"]["a1"] == "a1"
    assert report["representatives"]["a2"] == "a1"
    assert report["representatives"]["b1"] == "b1"
    assert report["representatives"]["b2"] == "b2"
    # explicit override within the same cluster is accepted
    explicit = correlation_cluster(
        {"a1": a1, "a2": a2, "b1": b1, "b2": b2},
        outcome,
        threshold=0.8,
        representatives={"a2": "a1"},
    )
    assert explicit["representatives"]["a2"] == "a1"
    # failure modes raise loudly
    with pytest.raises(ValueError, match="unknown names"):
        correlation_cluster(
            {"a1": a1}, outcome, representatives={"ghost": "a1"}
        )
    with pytest.raises(ValueError, match="not a member"):
        correlation_cluster(
            {"a1": a1, "a2": a2, "b1": b1, "b2": b2},
            outcome,
            threshold=0.8,
            representatives={"b2": "a1"},
        )
    with pytest.raises(ValueError, match="every"):
        correlation_cluster(
            {"a1": a1, "a2": a2, "b1": b1, "b2": b2},
            outcome,
            threshold=0.8,
            representatives={"a1": "a1"},
        )


# ---------------------------------------------------------------------------
# Group B -- core analysis
# ---------------------------------------------------------------------------
def test_bucket_scan_planted_effect() -> None:
    """Bucket profile of a planted linear effect (test 4).

    World: outcome = 2 * feat + noise, quantile buckets.  Pins:
    monotone True via the Spearman gate, p_perm at the permutation
    floor, strictly rising bucket means, CI bracketing the mean.
    Mutation fact: an inverted bucket-mean arm (or a spearman sign
    flip) breaks the rising sequence; a broken rng seed changes
    p_perm.
    """
    feat = _uniform(42)
    _unused, noise = _draws(43, 2)
    outcome_values = 2.0 * feat + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = bucket_scan(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        n_perm=50,
        seed=42,
        label="planted",
    )
    profile = report["features"]["feat"]
    assert profile["monotone"] is True
    assert profile["spearman_p"] <= 0.05
    assert profile["spearman"] > 0.0
    assert profile["p_perm"] <= 0.05
    means = [cell["mean_outcome"] for cell in profile["buckets"]]
    assert means == sorted(means)
    assert all(
        low <= mean <= high
        for low, mean, high in zip(
            [cell["ci_lo"] for cell in profile["buckets"]],
            means,
            [cell["ci_hi"] for cell in profile["buckets"]],
            strict=True,
        )
    )


def test_bucket_scan_sign_inversion() -> None:
    """An inverted effect reads negative and points Q1 > Q2 (test 5).

    World: outcome = -2 * feat + noise.  Pins: spearman negative,
    q2_mean < q1_mean, net_r_t negative.  Mutation fact: the
    mean-arm direction (q2 minus q1) or a spearman sign flip breaks
    the negative triple; this is the sign-inversion mutation catcher
    for bucket_scan.
    """
    feat = _uniform(42)
    _unused, noise = _draws(43, 2)
    outcome_values = -2.0 * feat + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = bucket_scan(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        n_perm=50,
        seed=42,
        label="inverted",
    )
    profile = report["features"]["feat"]
    assert profile["spearman"] < 0.0
    assert profile["monotone"] is True  # monotone != positive
    q_split = profile["q_split"]
    assert q_split["q2_mean"] < q_split["q1_mean"]
    assert q_split["net_r_t"] < 0.0


def test_q1_q2_split() -> None:
    """A magnitude-only effect separates the three signals (test 6).

    World: outcome = feat * (2 + noise) with norm_r = outcome / feat
    -- the effect lives ONLY in raw magnitude: hit rate is flat in f
    (the noise term decides the sign), the normalized outcome is flat
    in f (2 + noise regardless of bucket).  Pins: net_r_signal True,
    hit_rate_signal False, norm_r_signal False.  Mutation fact:
    swapping the Q pools (Q2 minus Q1 -> Q1 minus Q2) flips
    net_r_t's sign; a hit-rate signal wired to net_r_t breaks the
    False on hit_rate_signal.
    """
    feat = _uniform(42)
    _unused, wiggle = _draws(43, 2)
    outcome_values = feat * (2.0 + wiggle)
    norm_values = outcome_values / feat
    idx = pd.RangeIndex(_N_ROWS)
    report = bucket_scan(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        norm_r=pd.Series(norm_values, index=idx),
        n_perm=50,
        seed=42,
        label="magnitude-only",
    )
    q_split = report["features"]["feat"]["q_split"]
    assert q_split["net_r_signal"] is True
    assert q_split["hit_rate_signal"] is False
    assert q_split["norm_r_signal"] is False
    # without a supplied norm_r the normalized signal mirrors net_r
    mirror = bucket_scan(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        n_perm=50,
        seed=42,
    )
    assert mirror["norm_r_is_outcome"] is True
    assert (
        mirror["features"]["feat"]["q_split"]["net_r_signal"]
        is (mirror["features"]["feat"]["q_split"]["norm_r_signal"])
    )
    assert any("mirrors" in note for note in mirror["notes"])


def test_interaction_null() -> None:
    """An additive world reads no interaction (test 7).

    World: outcome = a + b + noise on 600 rows (12 cells, ~50 rows
    each -- above min_cell_n).  Pins: status OK, no low-power cells,
    interaction_present False.  Mutation fact: inverting the F
    comparison (p_f > alpha) or dropping the power guard makes
    interaction_present True here and fails the test.
    """
    feat_a = _uniform(42, 800)
    feat_b = _uniform(43, 800)
    _unused, noise = _draws(47, 2, 800)
    outcome_values = feat_a + feat_b + noise
    idx = pd.RangeIndex(800)
    report = interaction_scan(
        pd.Series(outcome_values, index=idx),
        pd.Series(feat_a, index=idx),
        pd.Series(feat_b, index=idx),
        min_cell_n=30,
        name_a="feat_a",
        name_b="feat_b",
        label="additive",
    )
    assert report["status"] == "OK"
    assert report["feature_a"] == "feat_a"
    assert report["feature_b"] == "feat_b"
    assert report["n_cells"] == 12
    assert report["low_power_cells"] == []
    assert report["interaction_present"] is False


def test_rule_fit_overfit() -> None:
    """A train-only rule reads unstable; a both-splits rule wins (8).

    World: outcome rewards rule_train on the train split only and
    PENALIZES it on the test split (so delta_test is negative by
    construction, not by noise), while rule_both is rewarded on both
    splits.  Pins: statuses valid/unstable, best_valid_rule names the
    valid rule, insufficient_n for a starved rule.  Mutation fact:
    dropping the delta_test > 0 conjunct from `stable` (or the
    best-rule filter reading delta_train) makes best_valid_rule name
    the train-only rule and fails the test.
    """
    feat = _uniform(42, 600)
    noise_train = _draws(43, 1, 600)[0]
    train_mask = np.arange(600) < 400
    rule_train = feat > np.median(feat)
    rule_both = np.arange(600) % 2 == 0
    rule_starved = np.arange(600) == 5
    outcome_values = (
        0.8 * rule_train * train_mask
        - 0.8 * rule_train * ~train_mask
        + 1.0 * rule_both
        + noise_train
    )
    idx = pd.RangeIndex(600)
    report = rule_fit(
        pd.Series(outcome_values, index=idx),
        {
            "rule_train": rule_train,
            "rule_both": rule_both,
            "rule_starved": rule_starved,
        },
        train_idx=np.arange(400),
        test_idx=np.arange(400, 600),
        min_n_train=50,
        min_n_test=20,
        label="overfit",
    )
    assert report["rules"]["rule_train"]["status"] == "unstable"
    assert report["rules"]["rule_train"]["delta_train"] > 0.0
    assert report["rules"]["rule_train"]["delta_test"] < 0.0
    assert report["rules"]["rule_both"]["status"] == "valid"
    assert report["rules"]["rule_starved"]["status"] == "insufficient_n"
    assert report["best_valid_rule"] == "rule_both"
    # guard rails: split overlap and NaN outcome fail loudly
    with pytest.raises(ValueError, match="overlap"):
        rule_fit(
            pd.Series(outcome_values, index=idx),
            {"rule_train": rule_train},
            train_idx=np.arange(400),
            test_idx=np.arange(300, 600),
            min_n_train=50,
            min_n_test=20,
        )
    with pytest.raises(ValueError, match="NaN"):
        nan_outcome = pd.Series(outcome_values, index=idx)
        nan_outcome.iloc[10] = np.nan
        rule_fit(
            nan_outcome,
            {"rule_train": rule_train},
            train_idx=np.arange(400),
            test_idx=np.arange(400, 600),
            min_n_train=50,
            min_n_test=20,
        )


def test_bonferroni_multiple() -> None:
    """The derived gate uses clusters, not raw columns (test 9).

    World: 8 uncorrelated features, one planted at a MARGINAL effect
    (HC0 t = 2.57, thirds-stable, row-spearman significant).  With 8
    clusters the derived gate is |t| >= norm.isf(0.05/8/2) = 2.73, so
    the marginal feature is NOT detected and classifies PROXY.
    Mutation fact: deriving the gate from alpha alone (1.96) -- or
    from any fixed count below 8 clusters -- detects the marginal
    feature (t 2.57 clears 1.96 with stable thirds and monotone
    ranks) and the PROXY assertion fails.
    """
    marginal, null_a, null_b, null_c = _draws(42, 4)
    noise_one, noise_two, null_rest, control = _draws(43, 4)
    null_extra = _draws(44, 1)[0]
    noise_out = _draws(51, 1)[0]
    outcome_values = 0.4 * control + 0.11 * marginal + noise_out
    idx = pd.RangeIndex(_N_ROWS)
    feature_map = {
        "marginal": pd.Series(marginal, index=idx),
        "null_a": pd.Series(null_a, index=idx),
        "null_b": pd.Series(null_b, index=idx),
        "null_c": pd.Series(null_c, index=idx),
        "null_d": pd.Series(noise_one, index=idx),
        "null_e": pd.Series(noise_two, index=idx),
        "null_f": pd.Series(null_rest, index=idx),
        "null_g": pd.Series(null_extra, index=idx),
    }
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        feature_map,
        {"c1": pd.Series(control, index=idx)},
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="bonferroni",
    )
    assert report["n_effective"] == 8
    assert 2.7 < report["orthogonal_t_gate"] < 2.8
    marginal_t = report["orthogonal"]["full"]["t_stat"]["marginal"]
    assert 2.4 < marginal_t < 2.6  # above 1.96, below the 2.73 gate
    assert report["orthogonal"]["per_feature_detected"]["marginal"] is False
    assert report["classification"]["marginal"] == "PROXY"
    assert report["recommended_actions"]["marginal"] == "discard"
    for name in ("null_a", "null_b", "null_c", "null_d"):
        assert report["classification"][name] == "PROXY"


# ---------------------------------------------------------------------------
# Group E -- composition through the orchestrator
# ---------------------------------------------------------------------------
def test_proxy_not_bucketed() -> None:
    """A control-twin feature is PROXY and never bucketed (test 23).

    World: twin = control + small noise (|pearson| ~ 0.98) beside a
    planted real feature.  Pins: the collinearity screen note, twin
    classified PROXY with action discard, twin absent from the
    bucket scan (buckets run on DETECTED features only).  Mutation
    fact: bucketing every representative instead of the detected set
    puts twin into buckets["features"] and fails the test.
    """
    real, _unused_null, control, noise_one = _draws(42, 4)
    noise_two = _draws(45, 1)[0]
    outcome_values = 0.5 * real + 0.4 * control + noise_one
    idx = pd.RangeIndex(_N_ROWS)
    twin_values = control + 0.2 * noise_two
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {
            "real": pd.Series(real, index=idx),
            "twin": pd.Series(twin_values, index=idx),
        },
        {"c1": pd.Series(control, index=idx)},
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="proxy",
    )
    collinear_notes = [
        note
        for note in report["orthogonal"]["facts"]["notes"]
        if "collinear" in note
    ]
    assert collinear_notes != []
    assert report["orthogonal"]["per_feature_detected"]["real"] is True
    assert report["orthogonal"]["per_feature_detected"]["twin"] is False
    assert report["skipped_features"]["twin"] == "proxy"
    assert report["classification"]["twin"] == "PROXY"
    assert report["recommended_actions"]["twin"] == "discard"
    assert "twin" not in report["buckets"]["features"]
    assert "real" in report["buckets"]["features"]
    assert report["classification"]["real"] == "STRONG_UNVERIFIED"


def test_u_shape_not_monotone() -> None:
    """A pure-U world fails the Spearman monotone gate (test 25b).

    World: outcome = 4 * (feat - 0.5)^2 + noise -- symmetric around
    the middle, so the row-level rank relation is flat.  Pins:
    monotone False driven by spearman_p > 0.05 (the spearman path of
    the gate; the low-power path is pinned in
    test_shape_unknown_classification).  Mutation fact: a monotone
    that ignores spearman_p (or gates on rho's absolute value being
    large instead of p's significance) makes monotone True here.
    """
    feat = _uniform(42)
    _unused, noise = _draws(43, 2)
    outcome_values = 4.0 * (feat - 0.5) ** 2 + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = bucket_scan(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        n_perm=50,
        seed=42,
        label="u-shape",
    )
    profile = report["features"]["feat"]
    assert profile["spearman_p"] > 0.05
    assert profile["monotone"] is False
    assert profile["low_power_buckets"] == []
    # the bucket means themselves trace the U: ends above the middle
    means = [cell["mean_outcome"] for cell in profile["buckets"]]
    assert means[0] > means[len(means) // 2]
    assert means[-1] > means[len(means) // 2]


def test_shape_unknown_classification() -> None:
    """Detected + non-monotone classifies SHAPE_UNKNOWN (test 25).

    World: a strongly detected linear feature scanned with an
    impossible min_bucket_n floor -- every bucket reads low-power,
    the monotone verdict flips False (the power path of the same
    gate the U world pins), and the cell lands SHAPE_UNKNOWN with
    action investigate_interaction.  Mutation fact: a classification
    that reads only the Spearman p (dropping the low-power conjunct)
    -- or maps monotone-False to STRONG_UNVERIFIED -- fails the
    cell/action assertions.
    """
    real, _unused_null, control, noise = _draws(42, 4)
    outcome_values = 0.6 * real + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=1000,
        label="shape-unknown",
    )
    assert report["orthogonal"]["per_feature_detected"]["alpha"] is True
    profile = report["buckets"]["features"]["alpha"]
    assert profile["monotone"] is False
    assert profile["low_power_buckets"] == [0, 1, 2, 3, 4]
    assert report["classification"]["alpha"] == "SHAPE_UNKNOWN"
    assert report["recommended_actions"]["alpha"] == (
        "investigate_interaction"
    )


# ---------------------------------------------------------------------------
# Checkpoint A2 -- regime + threshold
# ---------------------------------------------------------------------------
def _regime_indicator(
    values: np.ndarray,
    *,
    name: str = "phase",
    definition: str = "planted phase split",
    prereg_hash: str | None = "frozen-hash-1",
) -> dict[str, Any]:
    """Pack a regime indicator dict for the orchestrator."""
    return {
        "name": name,
        "definition": definition,
        "prereg_hash": prereg_hash,
        "values": pd.Series(values, index=pd.RangeIndex(len(values))),
    }


def test_regime_flip_detected() -> None:
    """A regime flip classifies REGIME_DEPENDENT (test 10).

    World: the effect exists ONLY on the interleaved 'trend' rows
    (every second row carries 3x the feature effect; the 'chop' rows
    are pure noise).  Interleaving keeps every sample third carrying
    the effect, so the orthogonal thirds gate passes while the
    per-regime split still separates them.  With a verified indicator
    the feature is globally detected + monotone, monotone in 'trend'
    but not in 'chop' -> REGIME_DEPENDENT with the regime-filter
    action.  Mutation fact: a classification that reads only
    same_sign (the chop phase's tiny spearman can carry any sign) --
    or skips the per-regime monotone flags -- upgrades this world to
    STRONG and fails the test.
    """
    real, _unused_null, control, noise = _draws(42, 4)
    phase = np.where(np.arange(_N_ROWS) % 2 == 0, "trend", "chop")
    outcome_values = 3.0 * real * (phase == "trend") + 0.4 * control + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        regime_indicator=_regime_indicator(
            phase,
            definition="first half trend, second half chop",
        ),
        min_regime_n=100,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="regime-flip",
    )
    assert report["regime"] is not None
    assert report["regime"]["status"] == "OK"
    flags = report["regime"]["features"]["alpha"]
    assert sorted(flags) == ["chop", "trend"]
    assert flags["trend"]["monotone"] is True
    assert flags["chop"]["monotone"] is False
    assert report["classification"]["alpha"] == "REGIME_DEPENDENT"
    assert report["recommended_actions"]["alpha"] == (
        "build_rule_with_regime_filter"
    )


def test_unverified_regime_not_strong() -> None:
    """An unfrozen indicator can never upgrade to STRONG (test 11).

    World: the feature works in BOTH regimes (stable), but the
    indicator carries no prereg_hash.  Pins: regime status
    REGIME_UNVERIFIED, the note names the hash requirement, and the
    cell stays STRONG_UNVERIFIED even though every per-regime flag is
    monotone.  Mutation fact: assigning STRONG on all-monotone flags
    WITHOUT checking the indicator's status fails the first
    classification assert.
    """
    real, _unused_null, control, noise = _draws(42, 4)
    rng = np.random.default_rng(45)
    phase = np.where(rng.random(_N_ROWS) < 0.5, 0, 1)
    outcome_values = 2.0 * real + 0.4 * control + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        regime_indicator=_regime_indicator(
            phase.astype(float), prereg_hash=None
        ),
        min_regime_n=100,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="unverified-regime",
    )
    assert report["regime"] is not None
    assert report["regime"]["status"] == "REGIME_UNVERIFIED"
    assert report["regime"]["prereg_hash"] is None
    flags = report["regime"]["features"]["alpha"]
    assert all(entry["monotone"] for entry in flags.values())
    assert report["classification"]["alpha"] == "STRONG_UNVERIFIED"
    assert report["recommended_actions"]["alpha"] == (
        "prereg_regime_indicator_then_retest"
    )
    assert any(
        "REGIME_UNVERIFIED" in note and "alpha" in note
        for note in report["notes"]
    )


def test_regime_min_n_guard() -> None:
    """A starved regime is excluded, not silently pooled (test 12).

    World: stable effect, three regimes (300 / 400 / 100 rows), floor
    at 150.  Pins: the third regime gets its entry (with its real n)
    but is excluded from the sign comparison via an explicit note,
    and the two qualifying regimes still upgrade the cell to STRONG.
    Mutation fact: dropping the min_n guard (or the exclusion note)
    fails the note/qualifying asserts.
    """
    real, _unused_null, control, noise = _draws(42, 4)
    phase = np.where(
        np.arange(_N_ROWS) < 300,
        0.0,
        np.where(np.arange(_N_ROWS) < 700, 1.0, 2.0),
    )
    outcome_values = 2.0 * real + 0.4 * control + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        regime_indicator=_regime_indicator(phase),
        min_regime_n=150,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="regime-min-n",
    )
    assert report["regime"] is not None
    flags = report["regime"]["features"]["alpha"]
    assert sorted(flags) == ["0.0", "1.0", "2.0"]
    assert flags["2.0"]["n"] == 100
    assert any(
        "regimes excluded" in note and "'2.0'" in note
        for note in report["regime"]["notes"]
    )
    assert report["classification"]["alpha"] == "STRONG"
    assert report["recommended_actions"]["alpha"] == "build_rule"


def test_strong_classification() -> None:
    """Verified regime + stable effect upgrades the cell to STRONG (24).

    World: the feature works in both phases of an independent coin
    split, the indicator carries a prereg_hash.  Pins: status OK, the
    STRONG cell, the build_rule action.  Mutation fact: downgrading
    STRONG to STRONG_UNVERIFIED unconditionally (ignoring the verified
    flags) -- or gating on the indicator's presence instead of its
    status -- fails the cell/action asserts.
    """
    real, _unused_null, control, noise = _draws(42, 4)
    rng = np.random.default_rng(45)
    phase = np.where(rng.random(_N_ROWS) < 0.5, "calm", "storm")
    outcome_values = 2.0 * real + 0.4 * control + noise
    idx = pd.RangeIndex(_N_ROWS)
    report = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        regime_indicator=_regime_indicator(
            phase, name="vol_phase", definition="independent coin flip"
        ),
        min_regime_n=100,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="strong-world",
    )
    assert report["regime"] is not None
    assert report["regime"]["status"] == "OK"
    assert report["regime"]["regime_name"] == "vol_phase"
    assert report["classification"]["alpha"] == "STRONG"
    assert report["recommended_actions"]["alpha"] == "build_rule"


def test_threshold_plateau_detected() -> None:
    """A wide ridge reads plateau and not edge (test 13).

    World: outcome rewards feat < 0.65; the sweep sits ABOVE the
    boundary (0.7..0.9), so every threshold keeps most of the reward
    and the deltas fall off gently -- both edges hold >= 70% of the
    best delta_test.  Pins: plateau True, edge False, optimal_range
    spanning the whole sweep, every point valid; the orchestrator
    path (threshold_specs + template with {feature}/{t}) reproduces
    the same verdict.  Mutation fact: flipping the plateau share
    (0.7 -> 0.9) makes the last edge fall short and fails the test.
    """
    feat = _uniform(42)
    _unused, noise = _draws(43, 2)
    outcome_values = 1.0 * (feat < 0.65) + 0.1 * noise
    idx = pd.RangeIndex(_N_ROWS)
    report = threshold_scan(
        pd.Series(outcome_values, index=idx),
        pd.Series(feat, index=idx),
        "feat < {t}",
        [0.7, 0.75, 0.8, 0.85, 0.9],
        feature_name="feat",
        train_idx=np.arange(500),
        test_idx=np.arange(500, _N_ROWS),
        min_n_kept=30,
        label="plateau",
    )
    assert report["feature"] == "feat"
    assert report["op"] == "<"
    assert all(point["status"] == "valid" for point in report["points"])
    assert report["plateau_detected"] is True
    assert report["edge_detected"] is False
    assert report["optimal_range"] == [0.7, 0.9]
    orchestrated = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"feat": pd.Series(feat, index=idx)},
        {"c1": pd.Series(noise, index=idx)},
        threshold_specs={"feat": [0.7, 0.8, 0.9]},
        threshold_template="{feature} < {t}",
        train_idx=np.arange(500),
        test_idx=np.arange(500, _N_ROWS),
        min_n_kept=30,
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="plateau-orchestrated",
    )
    assert orchestrated["threshold"] is not None
    sweep = orchestrated["threshold"]["feat"]
    assert sweep["plateau_detected"] is True
    assert sweep["op"] == "<"


def test_threshold_edge_detected() -> None:
    """A single-threshold spike reads edge and not plateau (test 14).

    World: the reward lives in a narrow band (0.4 < feat < 0.6) with
    penalty zones at both tails; deltas over [0.2, 0.4, 0.6, 0.8]
    rise monotonically to the LAST threshold (+1.0) while the first
    edge sits at -1.0.  Pins: edge True, plateau False,
    optimal_range collapsing to the single spike threshold.
    Mutation fact: an edge rule that compares the MAX edge instead
    of the FIRST -- or drops the negative-delta branch -- misses the
    -1.0 leading edge and fails the test.
    """
    feat = _uniform(42)
    _unused, noise = _draws(43, 2)
    outcome_values = (
        1.0 * ((feat > 0.4) & (feat < 0.6))
        - 1.0 * (feat <= 0.2)
        - 1.0 * (feat >= 0.8)
        + 0.15 * noise
    )
    idx = pd.RangeIndex(_N_ROWS)
    report = threshold_scan(
        pd.Series(outcome_values, index=idx),
        pd.Series(feat, index=idx),
        "feat < {t}",
        [0.2, 0.4, 0.6, 0.8],
        feature_name="feat",
        train_idx=np.arange(500),
        test_idx=np.arange(500, _N_ROWS),
        min_n_kept=30,
        label="edge",
    )
    deltas = [point["delta_test"] for point in report["points"]]
    assert deltas[-1] == max(deltas)
    assert deltas[0] < 0.0
    assert report["edge_detected"] is True
    assert report["plateau_detected"] is False
    assert report["optimal_range"] == [0.8, 0.8]


def test_threshold_guards() -> None:
    """Template and sweep guards fail loudly (guards for 13/14).

    Pins: a template without {t}, an unknown operator, a feature
    token mismatch, an over-cap sweep (8 > 7) and an under-length
    sweep (1 < 2) all raise ValueError.  Mutation fact: dropping any
    guard makes the corresponding block silently accept garbage.
    """
    feat = _uniform(42)
    outcome = pd.Series(feat, index=pd.RangeIndex(_N_ROWS))
    feature_col = pd.Series(feat, index=pd.RangeIndex(_N_ROWS))
    with pytest.raises(ValueError, match="placeholder"):
        threshold_scan(
            outcome,
            feature_col,
            "feat <",
            [0.3, 0.6],
            feature_name="feat",
            train_idx=np.arange(500),
            test_idx=np.arange(500, _N_ROWS),
        )
    with pytest.raises(ValueError, match="operator must be"):
        threshold_scan(
            outcome,
            feature_col,
            "feat == {t}",
            [0.3, 0.6],
            feature_name="feat",
            train_idx=np.arange(500),
            test_idx=np.arange(500, _N_ROWS),
        )
    with pytest.raises(ValueError, match="names feature"):
        threshold_scan(
            outcome,
            feature_col,
            "other > {t}",
            [0.3, 0.6],
            feature_name="feat",
            train_idx=np.arange(500),
            test_idx=np.arange(500, _N_ROWS),
        )
    with pytest.raises(ValueError, match="prereg cap"):
        threshold_scan(
            outcome,
            feature_col,
            "feat < {t}",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            feature_name="feat",
            train_idx=np.arange(500),
            test_idx=np.arange(500, _N_ROWS),
        )
    with pytest.raises(ValueError, match="at least 2"):
        threshold_scan(
            outcome,
            feature_col,
            "feat < {t}",
            [0.5],
            feature_name="feat",
            train_idx=np.arange(500),
            test_idx=np.arange(500, _N_ROWS),
        )


def test_feature_analysis_schema() -> None:
    """The FeatureAnalysisReport key set is frozen (schema guard).

    Guards accidental key drift in either direction: an added or
    removed key at the top level fails here first.  The A2/A3 blocks
    must stay present-and-None until their functions land, and the
    notes must say so.
    """
    real, _unused_null, control, noise = _draws(42, 4, 400)
    outcome_values = 0.5 * real + 0.4 * control + noise
    idx = pd.RangeIndex(400)
    report: FeatureAnalysisReport = feature_analysis(
        pd.Series(outcome_values, index=idx),
        {"alpha": pd.Series(real, index=idx)},
        {"c1": pd.Series(control, index=idx)},
        n_perm_orth=25,
        n_perm_bucket=25,
        min_bucket_n=0,
        label="schema",
    )
    assert frozenset(report) == _FA_REPORT_KEYS
    assert report["label"] == "schema"
    assert report["seed"] == 42
    assert report["regime"] is None
    assert report["threshold"] is None
    assert report["discovery"] is None
    assert report["importance"] is None
    assert report["rule_generation"] is None
    assert report["rules"] is None
    assert any("A3" in note for note in report["notes"])
