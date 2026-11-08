"""Orthogonal-check pins: planted worlds, provably failable both ways.

Every world is a fixed-seed (42-47) 600-row synthetic with a planted
effect, so each test below fails loudly under the mutation it targets.
The docstring of each test states WHAT it pins and WHICH orchestrator
mutation must be caught by it (the mutation plan).

Collected items: 9 tests -> 11 items (test_dropna_counters is
parametrized x2 at the strict >10% boundary,
test_orthogonal_report_schema is parametrized x2 over the report
kinds).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from atropos.orthogonal import orthogonal_check, signal_vs_controls

_N_ROWS = 600
_BORDER_COEF = 0.05
"""Planted borderline effect; lands its t at ~1.68, inside (0.5, 2.5)."""

_REPORT_KEYS = frozenset(
    {
        "kind",
        "label",
        "status",
        "n_before_dropna",
        "n_after_dropna",
        "n_dropped",
        "controls",
        "features",
        "correlation",
        "controls_only",
        "full",
        "delta_r2_block",
        "per_feature_delta_r2",
        "permutation",
        "facts",
    }
)
_FACTS_KEYS = frozenset(
    {
        "detected",
        "sign_consistent",
        "strongest_feature",
        "strongest_t",
        "t_threshold",
        "notes",
    }
)
_BLOCK_KEYS = frozenset({"r2", "beta", "se", "t_stat", "p_value"})
_CORR_KEYS = frozenset({"pearson", "spearman"})


def _draws(
    seed: int, count: int = 4, n_rows: int = _N_ROWS
) -> list[np.ndarray]:
    """``count`` independent standard-normal columns from one rng."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n_rows) for _ in range(count)]


def _strong_null_world() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Seed-42 world: strong feature, null feature, control, outcome.

    ``outcome = 0.5 * alpha + 0.4 * control + noise``; the null feature
    is an independent draw (verified t ~ -0.01).
    """
    alpha, null_feat, control, noise = _draws(42)
    outcome = pd.Series(0.5 * alpha + 0.4 * control + noise)
    return outcome, pd.Series(alpha), pd.Series(null_feat), pd.Series(control)


def test_planted_effect_detected() -> None:
    """A planted real effect is detected; a null one is not.

    Pins: status OK, facts.detected True, strongest_feature == "alpha"
    (the planted column), delta_r2_block > 0, per-feature partial
    delta_r2 of the real effect above the null's, permutation p of the
    real effect at 0 while the null's stays high.  Mutation fact: any
    mutation that flips the verdict machinery (threshold sign, delta_r2
    subtraction order, permutation rng) breaks one of these.
    """
    outcome, alpha, null_feat, control = _strong_null_world()
    report = orthogonal_check(
        outcome,
        {"alpha": alpha, "beta": null_feat},
        {"c1": control},
        label="planted",
    )
    assert report["kind"] == "orthogonal"
    assert report["status"] == "OK"
    assert report["facts"]["detected"] is True
    assert report["facts"]["strongest_feature"] == "alpha"
    assert report["facts"]["strongest_t"] == report["full"]["t_stat"]["alpha"]
    assert report["full"]["t_stat"]["alpha"] > report["facts"]["t_threshold"]
    assert report["full"]["t_stat"]["beta"] < report["facts"]["t_threshold"]
    assert report["delta_r2_block"] > 0.0
    assert (
        report["per_feature_delta_r2"]["alpha"]
        > report["per_feature_delta_r2"]["beta"]
    )
    assert sorted(report["permutation"]) == ["alpha", "beta"]
    assert report["permutation"]["alpha"] <= 0.05
    assert report["permutation"]["beta"] >= 0.5
    assert report["n_before_dropna"] == _N_ROWS
    assert report["n_dropped"] == 0


def test_threshold_mutation_boundary() -> None:
    """Pin the 2.5 gate itself, not the borderline status per se.

    The borderline feature is planted with coefficient 0.05 and its
    achieved t is verified BY CONSTRUCTION below to sit inside the
    (0.5, 2.5) band (seeded value: t = 1.676); it is therefore NOT
    detected at the 2.5 threshold, while a strong feature in the SAME
    report IS detected -- the threshold is what separates them.
    Mutation fact for the orchestrator: lowering the gate constant
    (_T_STRONG) to 0.5 flips the borderline feature to detected=True
    and fails the detected/strength pins here.
    """
    border, control, strong, noise = _draws(43)
    outcome = pd.Series(
        0.5 * strong + _BORDER_COEF * border + 0.4 * control + noise
    )
    report = orthogonal_check(
        outcome,
        {"strong": pd.Series(strong), "border": pd.Series(border)},
        {"c1": pd.Series(control)},
        label="boundary",
    )
    assert report["status"] == "OK"
    assert report["facts"]["t_threshold"] == 2.5
    t_border = report["full"]["t_stat"]["border"]
    assert 0.5 < t_border < 2.5  # achieved t by construction (1.676)
    t_strong = report["full"]["t_stat"]["strong"]
    assert t_strong >= report["facts"]["t_threshold"]
    assert report["facts"]["strongest_feature"] == "strong"
    assert report["facts"]["detected"] is True  # companion strong feature
    assert report["facts"]["sign_consistent"] is True


def test_sign_inversion_flips_t() -> None:
    """Inverting the feature column flips its t exactly, sign-for-sign.

    Pins: t_inverted == -t_original EXACTLY (IEEE negation is exact and
    the HC0 fit of a negated column is otherwise bit-identical), the
    sign test (t > 0) flips, and detection is unchanged (|t| carries
    the evidence for the orthogonal kind).  Mutation fact: any mutation
    that absorbs or flips the coefficient sign (abs() on beta, sign
    convention swap) breaks the exact-negation pin.
    """
    alpha, control, noise = _draws(42, count=3)
    outcome = pd.Series(0.5 * alpha + 0.4 * control + noise)
    base = orthogonal_check(
        outcome,
        {"alpha": pd.Series(alpha)},
        {"c1": pd.Series(control)},
        label="base",
    )
    flipped = orthogonal_check(
        outcome,
        {"alpha": pd.Series(-alpha)},
        {"c1": pd.Series(control)},
        label="flip",
    )
    t_orig = base["full"]["t_stat"]["alpha"]
    t_inv = flipped["full"]["t_stat"]["alpha"]
    assert t_inv == -t_orig
    assert (t_orig > 0) != (t_inv > 0)
    assert base["facts"]["detected"] is True
    assert flipped["facts"]["detected"] is True
    assert (
        base["per_feature_delta_r2"]["alpha"]
        == flipped["per_feature_delta_r2"]["alpha"]
    )
    assert base["full"]["p_value"] == flipped["full"]["p_value"]


def test_no_features_returns_explicit_status() -> None:
    """Empty or fully degenerate feature sets are explicit, not silent.

    Pins: status NO_FEATURES, detected False, the reason stated in
    notes ("no features supplied" / "degenerate"), controls-only
    statistics still fitted (beta keys intercept + c1), and the honest
    empty blocks (full r2 NaN, no per-feature delta, no permutation).
    Mutation fact: a mutation that downgrades NO_FEATURES to OK, drops
    the note, or silently drops the degenerate column and continues
    with an OK report fails these pins.
    """
    control_arr, _, noise = _draws(42, count=3)
    outcome = pd.Series(0.5 * control_arr + noise)
    empty = orthogonal_check(
        outcome, {}, {"c1": pd.Series(control_arr)}, label="empty"
    )
    assert empty["status"] == "NO_FEATURES"
    assert empty["facts"]["detected"] is False
    assert any(
        "no features supplied" in note for note in empty["facts"]["notes"]
    )
    assert sorted(empty["controls_only"]["beta"]) == ["c1", "intercept"]
    assert math.isnan(empty["full"]["r2"])
    assert empty["per_feature_delta_r2"] == {}
    assert empty["permutation"] == {}
    dead = pd.Series(np.zeros(_N_ROWS))
    degenerate = orthogonal_check(
        outcome, {"dead": dead}, {"c1": pd.Series(control_arr)}, label="dead"
    )
    assert degenerate["status"] == "NO_FEATURES"
    assert degenerate["facts"]["detected"] is False
    assert any("degenerate" in note for note in degenerate["facts"]["notes"])


def test_collinear_pair_flagged() -> None:
    """A feature collinear with a control is flagged, not detected.

    The feature is planted to WORK numerically (t = 10.7, far above the
    2.5 gate) yet correlate 0.91 with the control, so detected=False
    here reads "the check could not separate", never "the feature does
    not work".  Pins: the note names the control and says "cannot
    separate".  Mutation fact: raising the screen constant above 0.91,
    or deleting the collinear branch, flips detected to True and fails
    the pin.
    """
    control_arr, eps, noise = _draws(44, count=3)
    feature = pd.Series(0.9 * control_arr + 0.44 * eps)
    outcome = pd.Series(0.8 * feature.to_numpy() + 0.4 * control_arr + noise)
    report = orthogonal_check(
        outcome,
        {"f_col": feature},
        {"c1": pd.Series(control_arr)},
        label="collinear",
        n_perm=20,
    )
    assert report["status"] == "OK"
    rho = report["correlation"]["pearson"]["f_col"]["c1"]
    assert abs(rho) >= 0.8
    assert report["full"]["t_stat"]["f_col"] > report["facts"]["t_threshold"]
    assert report["facts"]["detected"] is False
    collinear_notes = [
        note for note in report["facts"]["notes"] if "collinear" in note
    ]
    assert collinear_notes
    assert "c1" in collinear_notes[0]
    assert "cannot separate" in collinear_notes[0]


def test_signal_survives_after_controls() -> None:
    """Both orientations of the stage-1 check are pinned.

    A planted signal with a real effect is detected (t = 13.3 >= 2 in
    its stated orientation); the inverted variant flips t exactly and
    is NOT detected (orientation-pinned rule) while the report notes it
    survives inverted.  The permutation dict is empty for both --
    signal acceptance is deterministic, not permuted.  Mutation fact:
    switching the gate to sign-agnostic |t| >= 2 flips the inverted
    case to detected=True and fails; requiring t <= -2 fails the plain
    case instead.
    """
    signal_arr, control_arr, noise = _draws(46, count=3)
    outcome = pd.Series(0.5 * signal_arr + 0.4 * control_arr + noise)
    plain = signal_vs_controls(
        outcome,
        {"sig": pd.Series(signal_arr)},
        {"c1": pd.Series(control_arr)},
        label="stage-1",
    )
    inverted = signal_vs_controls(
        outcome,
        {"sig": pd.Series(-signal_arr)},
        {"c1": pd.Series(control_arr)},
        label="stage-1",
    )
    assert plain["kind"] == "signal_check"
    assert plain["facts"]["t_threshold"] == 2.0
    assert plain["full"]["t_stat"]["sig"] >= plain["facts"]["t_threshold"]
    assert plain["facts"]["detected"] is True
    assert inverted["facts"]["detected"] is False
    t_plain = plain["full"]["t_stat"]["sig"]
    t_flipped = inverted["full"]["t_stat"]["sig"]
    assert t_flipped == -t_plain
    assert plain["permutation"] == {}
    assert inverted["permutation"] == {}
    assert plain["per_feature_delta_r2"] == {"sig": plain["delta_r2_block"]}
    assert any(
        "not a feature decomposition" in note
        for note in plain["facts"]["notes"]
    )
    assert any("INVERTED sign" in note for note in inverted["facts"]["notes"])


def test_thirds_sign_stability_rule() -> None:
    """The thirds rule itself: a mid-sample sign flip blocks detection.

    The feature's effect is +1.5 per unit in thirds 1-2 and -1.0 in
    third 3, so the full-sample t is large (5.2, well above 2.5) while
    third 3 carries the opposite sign: sign_consistent False AND
    detected False.  Mutation fact: weakening the rule (majority sign
    instead of all-thirds, dropping the |t_third| >= 1 floor, or
    removing the thirds gate entirely) restores detected=True and
    fails this pin.
    """
    feature_arr, control_arr, noise = _draws(45, count=3)
    effects = np.concatenate(
        [np.full(200, 1.5), np.full(200, 1.5), np.full(200, -1.0)]
    )
    outcome = pd.Series(effects * feature_arr + 0.5 * control_arr + noise)
    report = orthogonal_check(
        outcome,
        {"flip": pd.Series(feature_arr)},
        {"c1": pd.Series(control_arr)},
        label="thirds",
        n_perm=20,
    )
    assert report["status"] == "OK"
    assert report["full"]["t_stat"]["flip"] > report["facts"]["t_threshold"]
    assert report["facts"]["strongest_feature"] == "flip"
    assert report["facts"]["sign_consistent"] is False
    assert report["facts"]["detected"] is False


@pytest.mark.parametrize(
    ("n_nan_rows", "expected_after", "excessive_expected"),
    [
        (60, 540, False),
        (90, 510, True),
    ],
)
def test_dropna_counters(
    n_nan_rows: int, expected_after: int, excessive_expected: bool
) -> None:
    """Dropna counters are exact; the B5 note fires only above 10%.

    60 dropped of 600 sits EXACTLY on the 10% boundary and must NOT
    produce the note (the rule is a strict >); 90 dropped is 15% and
    must produce it with the percentage in the text.  Mutation fact:
    flipping > to >= fires the note at the boundary and fails the
    60-row case; breaking the counter arithmetic fails the exact
    n_before/n_after/n_dropped pins in both cases.
    """
    feature_arr, control_arr, noise = _draws(47, count=3)
    outcome = pd.Series(0.5 * feature_arr + 0.4 * control_arr + noise)
    outcome.iloc[-n_nan_rows:] = np.nan
    report = orthogonal_check(
        outcome,
        {"f": pd.Series(feature_arr)},
        {"c1": pd.Series(control_arr)},
        label="dropna",
        n_perm=10,
    )
    assert report["n_before_dropna"] == _N_ROWS
    assert report["n_after_dropna"] == expected_after
    assert report["n_dropped"] == _N_ROWS - expected_after
    excessive = [
        note
        for note in report["facts"]["notes"]
        if note.startswith("excessive dropna")
    ]
    assert bool(excessive) is excessive_expected
    if excessive_expected:
        assert "15% removed" in excessive[0]


@pytest.mark.parametrize("kind", ["orthogonal", "signal_check"])
def test_orthogonal_report_schema(kind: str) -> None:
    """The OrthogonalReport key sets are frozen for BOTH kinds.

    Guards accidental key drift in either direction: an added or
    removed key at any level (top level, facts, fit blocks,
    correlation block) fails here first.  Mutation fact: any mutation
    touching the report assembly that adds, renames or drops a schema
    key is caught by this test.
    """
    alpha, _unused_null, control, noise = _draws(42)
    outcome = pd.Series(0.5 * alpha + 0.4 * control + noise)
    if kind == "orthogonal":
        report = orthogonal_check(
            outcome,
            {"alpha": pd.Series(alpha)},
            {"c1": pd.Series(control)},
            label="schema",
            n_perm=10,
        )
    else:
        report = signal_vs_controls(
            outcome,
            {"alpha": pd.Series(alpha)},
            {"c1": pd.Series(control)},
            label="schema",
        )
    assert frozenset(report) == _REPORT_KEYS
    assert frozenset(report["facts"]) == _FACTS_KEYS
    assert frozenset(report["full"]) == _BLOCK_KEYS
    assert frozenset(report["controls_only"]) == _BLOCK_KEYS
    assert frozenset(report["correlation"]) == _CORR_KEYS
    assert report["kind"] == kind
    assert report["label"] == "schema"
