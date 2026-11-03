"""Tests for atropos.incremental_regression: the repackaging kill rule.

A genuinely new predictor must keep |t| >= 2 after known controls and
across the specification curve; a repackaged one must die.  Verdict
taxonomy of ``incremental_test``: SURVIVES / REPACKAGING-DEATH
(controls present, null incremental t) / NULL-DEATH (no controls, null
t) / INCONCLUSIVE (unidentifiable model).  All data is planted with
known coefficients so the verdicts are structural.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.incremental_regression import (
    DEFAULT_TRANSFORMS,
    curve_summary,
    incremental_test,
    run_all_examples,
    specification_curve,
)


def _genuine_edge(
    n_obs: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Outcome driven by the new signal and one control, plus noise."""
    gen = np.random.default_rng(seed)
    controls = pd.DataFrame(
        {
            "ctrl_a": gen.standard_normal(n_obs),
            "ctrl_b": gen.standard_normal(n_obs),
        }
    )
    signal = gen.standard_normal(n_obs)
    outcome = (
        0.4 * signal
        + 0.5 * controls["ctrl_a"].to_numpy()
        + gen.standard_normal(n_obs)
    )
    return outcome, controls, signal


def test_incremental_test_genuine_signal_survives() -> None:
    outcome, controls, signal = _genuine_edge(400, 95)
    result = incremental_test(outcome, controls, signal, robust=True)
    assert result["verdict"] == "SURVIVES"
    assert abs(result["t_new"]) >= 2.0
    assert result["delta_r2"] > 0.0
    assert result["r2_full"] >= result["r2_restricted"]
    assert result["n"] == 400
    assert result["robust"] is True


def test_incremental_test_repackaged_momentum_dies() -> None:
    gen = np.random.default_rng(96)
    n_obs = 400
    momentum = np.zeros(n_obs)
    shock = gen.standard_normal(n_obs)
    for day in range(1, n_obs):
        momentum[day] = 0.6 * momentum[day - 1] + 0.4 * shock[day]
    # lagged control via a NaN-headed shift (np.roll would wrap FUTURE
    # values into the head of the series - a lookahead leak)
    mom12 = np.concatenate([np.full(12, np.nan), momentum[:-12]])
    controls = pd.DataFrame(
        {
            "mom_noisy": momentum + gen.standard_normal(n_obs) * 0.1,
            "mom12": mom12,
        }
    )
    outcome = 0.35 * momentum + gen.standard_normal(n_obs)
    signal = momentum + gen.standard_normal(n_obs) * 0.35
    result = incremental_test(outcome, controls, signal, robust=True)
    assert result["verdict"] == "REPACKAGING-DEATH"
    assert abs(result["t_new"]) < 2.0
    assert result["delta_r2"] >= 0.0
    # the NaN head of the lagged control is dropped, never wrapped
    assert result["n"] == n_obs - 12


def test_incremental_test_accepts_none_controls() -> None:
    gen = np.random.default_rng(97)
    signal = gen.standard_normal(300)
    outcome = 0.4 * signal + gen.standard_normal(300)
    result = incremental_test(outcome, None, signal)
    assert result["verdict"] == "SURVIVES"
    assert abs(result["t_new"]) >= 2.0
    assert result["n"] == 300
    # a constant-only restricted model explains nothing
    assert result["r2_restricted"] == pytest.approx(0.0, abs=1e-12)


def test_exact_collinearity_is_inconclusive_not_death() -> None:
    """Exact signal/control collinearity must not fabricate a verdict.

    ``signal = 2 * ctrl_a`` makes the full design rank-deficient.  The
    old pinv-based fit invented a plausible-looking minimum-norm
    ``t_new`` here - an arbitrary number that could pass or fail the
    gate by numerical accident.  (With EXACT collinearity the verdict
    is deterministic INCONCLUSIVE; visible added noise keeps the design
    full-rank and lands in REPACKAGING-DEATH instead.)
    """
    gen = np.random.default_rng(101)
    n_obs = 400
    controls = pd.DataFrame({"ctrl_a": gen.standard_normal(n_obs)})
    signal = 2.0 * controls["ctrl_a"].to_numpy()  # exact collinearity
    outcome = 0.5 * controls["ctrl_a"].to_numpy() + gen.standard_normal(n_obs)
    result = incremental_test(outcome, controls, signal, robust=True)
    assert result["verdict"] == "INCONCLUSIVE"
    assert not np.isfinite(result["t_new"])
    assert result["n"] == n_obs  # rows are kept; the MODEL is degenerate


def test_null_death_when_no_controls_and_no_effect() -> None:
    """Without controls a null t is NULL-DEATH, not a repackaging."""
    gen = np.random.default_rng(102)
    n_obs = 500
    signal = gen.standard_normal(n_obs)
    outcome = gen.standard_normal(n_obs)  # unrelated to the signal
    result = incremental_test(outcome, None, signal)
    assert result["verdict"] == "NULL-DEATH"
    assert abs(result["t_new"]) < 2.0
    # an empty control set behaves like no controls at all
    empty_ctrls = pd.DataFrame(index=range(n_obs))
    result_empty = incremental_test(outcome, empty_ctrls, signal)
    assert result_empty["verdict"] == "NULL-DEATH"


def test_saturated_design_is_inconclusive() -> None:
    """Saturated design (const + 2 controls + signal on 3 rows).

    The old pinv path invented minimum-norm t-stats here.
    """
    gen = np.random.default_rng(103)
    n_obs = 3
    controls = pd.DataFrame(
        {
            "ctrl_a": gen.standard_normal(n_obs),
            "ctrl_b": gen.standard_normal(n_obs),
        }
    )
    signal = gen.standard_normal(n_obs)
    outcome = gen.standard_normal(n_obs)
    result = incremental_test(outcome, controls, signal)
    assert result["verdict"] == "INCONCLUSIVE"
    assert not np.isfinite(result["t_new"])


def test_constant_target_is_inconclusive() -> None:
    """A constant outcome makes every t meaningless noise-of-epsilon."""
    gen = np.random.default_rng(104)
    n_obs = 200
    signal = gen.standard_normal(n_obs)
    outcome = np.full(n_obs, 0.7)
    result = incremental_test(outcome, None, signal)
    assert result["verdict"] == "INCONCLUSIVE"
    result_ctrl = incremental_test(
        outcome, pd.DataFrame({"ctrl_a": gen.standard_normal(n_obs)}), signal
    )
    assert result_ctrl["verdict"] == "INCONCLUSIVE"


def test_all_nan_alignment_returns_inconclusive() -> None:
    """Empty alignment (all rows dropped as NaN) must not crash or kill."""
    n_obs = 50
    signal = np.random.default_rng(105).standard_normal(n_obs)
    fwd_ret = np.full(n_obs, np.nan)
    result = incremental_test(fwd_ret, None, signal)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["n"] == 0
    assert not np.isfinite(result["t_new"])
    table = specification_curve(
        fwd_ret,
        [],
        signal,
        specs_to_try=[{"controls": None, "transform": "raw"}],
    )
    assert table["verdict"].iloc[0] == "INCONCLUSIVE"
    assert int(table["n"].iloc[0]) == 0


def test_specification_curve_default_grid_shape() -> None:
    outcome, controls, signal = _genuine_edge(300, 93)
    table = specification_curve(outcome, [controls], signal)
    assert len(table) == 10  # 5 transforms x (no controls + one set)
    assert set(table.columns) == {
        "spec",
        "transform",
        "controls",
        "n",
        "t_new",
        "p_new",
        "delta_r2",
        "verdict",
    }
    assert set(table["transform"]) == set(DEFAULT_TRANSFORMS)
    assert set(table["controls"]) == {"none", "set0"}
    assert list(table["spec"]) == list(range(10))


def test_specification_curve_custom_specs_are_keyword_only() -> None:
    outcome, controls, signal = _genuine_edge(300, 94)
    specs: list[dict[str, Any]] = [
        {"controls": 0, "transform": "rank"},
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.05},
    ]
    table = specification_curve(
        outcome, [controls], signal, specs_to_try=specs, robust=False
    )
    assert len(table) == 2
    assert list(table["transform"]) == ["rank", "winsorize"]
    assert list(table["controls"]) == ["set0", "none"]
    with pytest.raises(TypeError):
        specification_curve(  # type: ignore[call-arg]
            outcome, [controls], signal, specs
        )


def test_specification_curve_rejects_invalid_controls_index() -> None:
    """A mistyped control index must raise instead of an uncontrolled spec.

    Falling back to ``ctrl=None`` on a bad index would silently
    substitute the hypothesis under test.
    """
    outcome, controls, signal = _genuine_edge(300, 106)
    for bad_index in (5, -1, "set0", True):
        specs: list[dict[str, Any]] = [
            {"controls": bad_index, "transform": "raw"}
        ]
        with pytest.raises(ValueError, match="not a valid index"):
            specification_curve(
                outcome, [controls], signal, specs_to_try=specs
            )


def test_zscore_of_constant_signal_is_inconclusive() -> None:
    """sd=0 zscore must yield NaN (INCONCLUSIVE), not a zero vector.

    The old transform returned zeros, turning the "zscore" spec into a
    regression on a constant - a dummy spec that silently changed the
    hypothesis under test.
    """
    gen = np.random.default_rng(107)
    n_obs = 300
    signal = np.full(n_obs, 1.5)  # constant: sd == 0 after alignment
    outcome = gen.standard_normal(n_obs)
    specs = [
        {"controls": None, "transform": "zscore"},
        {"controls": None, "transform": "raw"},
    ]
    table = specification_curve(outcome, [], signal, specs_to_try=specs)
    for row_no in range(2):
        assert not np.isfinite(float(table["t_new"].iloc[row_no]))
        assert table["verdict"].iloc[row_no] == "INCONCLUSIVE"
    # ... and the degenerate specs do not count towards curve survival
    summary = curve_summary(table)
    assert summary["n_specs"] == 0
    assert summary["n_surviving"] == 0


def test_curve_summary_applies_the_kill_rule() -> None:
    surviving = curve_summary(pd.DataFrame({"t_new": [3.0, 0.5, -0.1]}))
    assert surviving["verdict"] == "SURVIVES"
    assert surviving["n_specs"] == 3
    assert surviving["n_surviving"] == 1
    assert surviving["max_abs_t"] == pytest.approx(3.0)
    assert surviving["min_abs_t"] == pytest.approx(0.1)
    assert surviving["sign_consistent"] is False

    dead = curve_summary(pd.DataFrame({"t_new": [1.5, -1.9, 0.2]}))
    assert dead["verdict"] == "REPACKAGING-DEATH"
    assert dead["n_surviving"] == 0
    assert dead["max_abs_t"] == pytest.approx(1.9)

    aligned = curve_summary(pd.DataFrame({"t_new": [2.5, 3.0]}))
    assert aligned["verdict"] == "SURVIVES"
    assert aligned["n_surviving"] == 2
    assert aligned["sign_consistent"] is True
    assert aligned["t_rule"] == 2.0


def test_curve_summary_sign_consistent_handles_exact_zero_t() -> None:
    """np.sign(0.0) == 0.0 must not fake sign inconsistency.

    An exact 0.0 t-stat carries no sign information: consistency is
    judged on the nonzero entries; no nonzero entries at all is
    trivially consistent.
    """
    zero_first = curve_summary(pd.DataFrame({"t_new": [0.0, 2.5, 3.0]}))
    assert zero_first["sign_consistent"] is True
    mixed = curve_summary(pd.DataFrame({"t_new": [2.5, 0.0, -3.0]}))
    assert mixed["sign_consistent"] is False
    all_zero = curve_summary(pd.DataFrame({"t_new": [0.0, 0.0]}))
    assert all_zero["sign_consistent"] is True


def test_curve_on_genuine_edge_survives_every_spec() -> None:
    outcome, controls, signal = _genuine_edge(300, 98)
    table = specification_curve(outcome, [controls], signal, robust=True)
    summary = curve_summary(table)
    assert summary["verdict"] == "SURVIVES"
    assert summary["n_specs"] == 10
    assert summary["n_surviving"] == 10
    assert summary["max_abs_t"] >= 2.0
    assert (table["verdict"] == "SURVIVES").all()


def test_robust_flag_variants_both_run_on_tiny_data() -> None:
    gen = np.random.default_rng(92)
    n_obs = 60
    controls = pd.DataFrame(
        {
            "ctrl_a": gen.standard_normal(n_obs),
            "ctrl_b": gen.standard_normal(n_obs),
        }
    )
    signal = gen.standard_normal(n_obs)
    outcome = (
        0.5 * signal
        + 0.3 * controls["ctrl_a"].to_numpy()
        + gen.standard_normal(n_obs)
    )
    result_robust = incremental_test(outcome, controls, signal, robust=True)
    result_classic = incremental_test(outcome, controls, signal, robust=False)
    assert result_robust["robust"] is True
    assert result_classic["robust"] is False
    for result in (result_robust, result_classic):
        assert np.isfinite(result["t_new"])
        assert result["verdict"] == "SURVIVES"
        assert result["n"] == n_obs


def test_winsorization_strength_changes_the_estimate() -> None:
    gen = np.random.default_rng(91)
    driver = gen.standard_normal(400)
    outcome = 0.4 * driver + gen.standard_normal(400)
    signal = driver + gen.standard_normal(400) * 0.3
    signal[0] = 60.0  # one extreme outlier dominates the raw scale
    specs = [
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.01},
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.45},
    ]
    table = specification_curve(outcome, [], signal, specs_to_try=specs)
    assert list(table["transform"]) == ["winsorize", "winsorize"]
    tight = float(table["t_new"].iloc[0])
    loose = float(table["t_new"].iloc[1])
    assert np.isfinite(tight)
    assert np.isfinite(loose)
    assert abs(tight - loose) > 0.5


def test_specification_curve_rejects_out_of_range_winsor_pct() -> None:
    """winsor_pct outside (0, 0.5) inverts the winsorize quantile bounds.

    At ``p >= 0.5`` the clip bounds cross (``lo > hi``) and the signal
    silently collapses onto the upper bound - a degenerate transform
    that must raise ``ValueError`` up front, both for the shared
    top-level parameter and for a per-spec override.
    """
    outcome, controls, signal = _genuine_edge(300, 108)
    for bad_pct in (0.0, 0.5, 0.75, -0.05):
        with pytest.raises(ValueError, match="winsor_pct"):
            specification_curve(
                outcome, [controls], signal, winsor_pct=bad_pct
            )
    specs: list[dict[str, Any]] = [
        {"controls": None, "transform": "winsorize", "winsor_pct": 0.6}
    ]
    with pytest.raises(ValueError, match="spec 0"):
        specification_curve(outcome, [controls], signal, specs_to_try=specs)
    # values inside the open interval are accepted (0.45 < 0.5)
    ok_table = specification_curve(
        outcome, [controls], signal, winsor_pct=0.45
    )
    assert len(ok_table) == 10


def test_curve_summary_counts_controlled_survivors() -> None:
    rng = np.random.default_rng(21)
    n_days = 600
    control = rng.standard_normal(n_days)
    # the "signal" carries no information beyond the known control
    echo = control + 0.2 * rng.standard_normal(n_days)
    fwd = control + rng.standard_normal(n_days)
    table = specification_curve(
        fwd, [pd.DataFrame({"control": control})], echo
    )
    summary = curve_summary(table)
    # lenient verdict survives on uncontrolled specs alone ...
    assert summary["verdict"] == "SURVIVES"
    assert summary["n_surviving"] >= 1
    # ... but the controlled-survivor counter exposes the repackaging
    assert summary["n_surviving_controlled"] == 0


def _demo_dataset() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int
]:
    """Exact replica of ``run_all_examples``' seeded dataset (rng 3).

    The demo itself asserts only seed-independent properties; the
    exact seeded numbers it used to assert in-line live in the
    ``test_seeded_regression_demo_*`` tests below.
    """
    gen = np.random.default_rng(3)
    n_days = 900
    mom = np.zeros(n_days)
    shock = gen.standard_normal(n_days)
    for day in range(1, n_days):
        mom[day] = 0.6 * mom[day - 1] + 0.4 * shock[day]
    driver = gen.standard_normal(n_days)
    mom12 = np.concatenate([np.full(12, np.nan), mom[:-12]])
    fwd_a = 0.20 * driver + 0.30 * mom + gen.standard_normal(n_days)
    sig_a = driver + 0.5 * mom
    mom_df = pd.DataFrame(
        {
            "mom_noisy": mom + gen.standard_normal(n_days) * 0.1,
            "mom12": mom12,
        },
        index=range(n_days),
    )
    fwd_b = 0.35 * mom + gen.standard_normal(n_days)
    sig_b = mom + gen.standard_normal(n_days) * 0.35
    return fwd_a, sig_a, fwd_b, sig_b, mom_df, n_days


def test_seeded_regression_demo_genuine_edge_exact_values() -> None:
    """Seed lock for demo A: t_new ~ 4.95, every spec survives."""
    fwd_a, sig_a, _, _, mom_df, _ = _demo_dataset()
    result = incremental_test(fwd_a, mom_df, sig_a, robust=True)
    assert result["t_new"] == pytest.approx(4.946119933465835, rel=1e-12)
    assert 4.0 <= result["t_new"] <= 6.0  # the demo's former range
    assert result["delta_r2"] == pytest.approx(0.02791115342423045, rel=1e-12)
    table = specification_curve(fwd_a, [mom_df], sig_a, robust=True)
    summary = curve_summary(table)
    assert summary["n_surviving"] == 10  # every spec in the grid
    assert summary["n_surviving_controlled"] == 5  # all controlled specs
    assert summary["sign_consistent"] is True
    assert (table["verdict"] == "SURVIVES").all()


def test_seeded_regression_demo_repackaged_death_exact_values() -> None:
    """Seed lock for demo B: t_new ~ 0.71, uncontrolled-only survival."""
    _, _, fwd_b, sig_b, mom_df, _ = _demo_dataset()
    result = incremental_test(fwd_b, mom_df, sig_b, robust=True)
    assert result["t_new"] == pytest.approx(0.7066140579170657, rel=1e-12)
    assert 0.0 <= result["t_new"] <= 1.5  # the demo's former range
    table = specification_curve(fwd_b, [mom_df], sig_b, robust=True)
    summary = curve_summary(table)
    assert summary["verdict"] == "SURVIVES"  # lenient by design
    assert summary["n_surviving"] == 5  # uncontrolled specs only
    assert summary["n_surviving_controlled"] == 0  # repackaging flag
    assert summary["sign_consistent"] is True


def test_run_all_examples_demo_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Layer-0: the demo's internal asserts hold and both cases print."""
    run_all_examples()
    output = capsys.readouterr().out
    assert "incremental_regression demo A: genuine edge" in output
    assert "incremental_regression demo B: repackaged momentum" in output
    assert "verdict=SURVIVES" in output
    assert "verdict=REPACKAGING-DEATH" in output
