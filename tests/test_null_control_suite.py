"""Tests for atropos.null_control_suite: adversarial null controls.

Every test plants an effect whose strength is known by construction and
asserts the documented contract: PASS for a genuine edge, FAIL for pure
noise, repackaged momentum, cloned features or concentrated evidence,
and INCONCLUSIVE when the signal is EXACTLY collinear with a control
(no pseudo-inverse may fabricate a t-stat there).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from atropos.null_control_suite import (
    calendar_control,
    known_features_control,
    momentum_control,
    placebo_placebo,
    run_all_examples,
    shuffled_permutation,
    subperiod_stability,
)


def _ar_momentum(n_obs: int, persistence: float, seed: int) -> np.ndarray:
    """AR(1) factor playing the role of the momentum confounder."""
    gen = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    shock = gen.standard_normal(n_obs)
    for day in range(1, n_obs):
        mom[day] = (
            persistence * mom[day - 1] + (1.0 - persistence) * shock[day]
        )
    return mom


def test_shuffled_permutation_rejects_planted_alignment() -> None:
    gen = np.random.default_rng(7)
    signal = gen.standard_normal(300)
    outcome = signal + 0.5 * gen.standard_normal(300)
    result = shuffled_permutation(signal, outcome, n_draws=200, seed=0)
    assert result["verdict"] == "PASS"
    assert result["p_value"] <= 0.05
    assert result["stat"] > 0.8
    assert result["percentile"] >= 99.0
    assert result["n"] == 300
    assert result["n_draws"] == 200


def test_shuffled_permutation_keeps_null_on_pure_noise() -> None:
    gen = np.random.default_rng(21)
    signal = gen.standard_normal(300)
    outcome = gen.standard_normal(300)
    result = shuffled_permutation(signal, outcome, n_draws=200, seed=5)
    assert result["verdict"] == "FAIL"
    assert result["p_value"] > 0.05


def test_placebo_placebo_detects_planted_event_days() -> None:
    gen = np.random.default_rng(13)
    day_index = pd.date_range("2022-01-03", periods=300, freq="B")
    outcome = pd.Series(gen.standard_normal(300), index=day_index)
    event_positions = [40, 120, 199]
    outcome.iloc[event_positions] += 5.0
    result = placebo_placebo(event_positions, outcome, n_draws=200, seed=0)
    assert result["verdict"] == "PASS"
    assert result["p_value"] <= 0.05
    assert result["n_events"] == 3
    assert result["stat"] > 4.0
    assert result["placebo_mean_abs"] < 1.0
    # non-integer day labels must resolve through the Series index
    by_label = placebo_placebo(
        day_index[event_positions], outcome, n_draws=200, seed=0
    )
    assert by_label["verdict"] == "PASS"
    assert by_label["n_events"] == 3


def test_placebo_placebo_fails_on_ordinary_days() -> None:
    gen = np.random.default_rng(14)
    day_index = pd.date_range("2022-01-03", periods=300, freq="B")
    outcome = pd.Series(gen.standard_normal(300), index=day_index)
    result = placebo_placebo([10, 150, 250], outcome, n_draws=200, seed=0)
    assert result["verdict"] == "FAIL"
    assert result["p_value"] > 0.05
    assert result["n_events"] == 3


def test_momentum_control_absorbs_repackaged_momentum() -> None:
    momentum = _ar_momentum(400, 0.7, 101)
    gen = np.random.default_rng(102)
    repackaged = momentum + gen.standard_normal(400) * 0.35
    fwd_ret = 0.35 * momentum + gen.standard_normal(400)
    mom7 = momentum + gen.standard_normal(400) * 0.1
    # lagged control via shift (NaN head dropped by alignment), never
    # np.roll which leaks future momentum into the lag
    mom12 = pd.Series(momentum).shift(12).to_numpy(dtype=float)
    result = momentum_control(fwd_ret, repackaged, mom7, mom12, robust=True)
    assert result["verdict"] == "FAIL"
    assert abs(result["stat"]) < 2.0
    assert result["HC0"] is True


def test_momentum_control_keeps_independent_signal() -> None:
    momentum = _ar_momentum(400, 0.7, 103)
    gen = np.random.default_rng(104)
    driver = gen.standard_normal(400)
    fwd_ret = 0.20 * driver + 0.25 * momentum + gen.standard_normal(400)
    mom7 = momentum + gen.standard_normal(400) * 0.1
    mom12 = pd.Series(momentum).shift(12).to_numpy(dtype=float)
    result = momentum_control(fwd_ret, driver, mom7, mom12, robust=True)
    assert result["verdict"] == "PASS"
    assert abs(result["stat"]) >= 2.0
    classic = momentum_control(fwd_ret, driver, mom7, mom12, robust=False)
    assert classic["HC0"] is False
    assert classic["verdict"] == "PASS"


def test_momentum_control_inconclusive_when_signal_is_control() -> None:
    # regression: sig EXACTLY equal to a control -> identical design
    # columns -> rank-deficient -> INCONCLUSIVE, never a pseudo-inverse
    # PASS that would wave perfect repackaging through
    momentum = _ar_momentum(400, 0.7, 105)
    gen = np.random.default_rng(106)
    fwd_ret = 0.35 * momentum + gen.standard_normal(400)
    mom7 = momentum + gen.standard_normal(400) * 0.1
    mom12 = pd.Series(momentum).shift(12).to_numpy(dtype=float)
    result = momentum_control(fwd_ret, mom7, mom7, mom12, robust=True)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "collinear with controls"
    assert np.isnan(result["stat"])
    assert np.isnan(result["p_value"])


def test_known_features_control_separates_clone_from_new_info() -> None:
    gen = np.random.default_rng(15)
    feat_one = gen.standard_normal(400)
    features = pd.DataFrame(
        {"feat_one": feat_one, "feat_two": gen.standard_normal(400)}
    )
    fwd_ret = 0.5 * feat_one + gen.standard_normal(400)

    clone = 0.9 * feat_one + gen.standard_normal(400) * 0.5
    cloned = known_features_control(fwd_ret, clone, features)
    assert cloned["verdict"] == "FAIL"
    assert abs(cloned["stat"]) < 2.0
    assert cloned["n_features"] == 2

    driver = gen.standard_normal(400)
    fwd_new = 0.4 * driver + 0.5 * feat_one + gen.standard_normal(400)
    original = known_features_control(fwd_new, driver, features)
    assert original["verdict"] == "PASS"
    assert abs(original["stat"]) >= 2.0


def test_known_features_control_inconclusive_on_exact_clone() -> None:
    # regression: signal EXACTLY equal to a known feature -> identical
    # design columns -> rank-deficient -> INCONCLUSIVE with a reason,
    # never a fabricated t-stat
    gen = np.random.default_rng(25)
    feat_one = gen.standard_normal(400)
    features = pd.DataFrame(
        {"feat_one": feat_one, "feat_two": gen.standard_normal(400)}
    )
    fwd_ret = 0.5 * feat_one + gen.standard_normal(400)
    result = known_features_control(fwd_ret, feat_one, features)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "collinear with controls"
    assert np.isnan(result["stat"])
    assert np.isnan(result["p_value"])
    assert result["n_features"] == 2


def test_known_features_control_keeps_small_scale_feature() -> None:
    """A sd~1e-7 informative feature stays in the control set.

    Regression: the ABSOLUTE variance floor (``var > 1e-12``) dropped
    a genuinely informative sd ~1e-7 feature (var ~1e-14) -- fail-open,
    because with every small-scale feature dropped the control
    silently degenerated into the very univariate regression it was
    meant to gate.  The relative floor (``1e-12`` x the strongest
    feature's variance) keeps small-scale controls in the design, so
    ``n_features`` counts them and the incremental t-stat is measured
    against the controls that exist.
    """
    gen = np.random.default_rng(31)
    n_obs = 400
    small = gen.standard_normal(n_obs) * 1e-7  # var ~1e-14 < 1e-12
    features = pd.DataFrame(
        {"small": small, "other": gen.standard_normal(n_obs) * 1e-7}
    )
    driver = gen.standard_normal(n_obs)
    fwd_ret = 0.4 * driver + 0.5 * small + gen.standard_normal(n_obs)
    result = known_features_control(fwd_ret, driver, features)
    assert result["n_features"] == 2  # old code: 0 (both features dropped)
    assert result["verdict"] == "PASS"
    assert abs(result["stat"]) >= 2.0


def test_known_features_control_all_constant_features_dropped() -> None:
    """A feature set constant at any scale leaves an empty control set.

    The ``var_max == 0`` guard of the relative floor: features frozen
    at a tiny scale (values ~1e-9, variance exactly 0) are all
    constant, the scale-free ptp check keeps none of them, and the
    univariate design still evaluates (no divide-by-zero anchoring the
    floor at zero variance).
    """
    gen = np.random.default_rng(32)
    features = pd.DataFrame({"const": np.full(300, 1e-9)})
    fwd_ret = gen.standard_normal(300)
    signal = gen.standard_normal(300)
    result = known_features_control(fwd_ret, signal, features)
    assert result["n_features"] == 0
    assert np.isfinite(result["stat"])
    assert result["verdict"] == "FAIL"  # noise signal cannot pass


def test_shuffled_permutation_strict_alpha_boundary() -> None:
    # unified strict thresholds: p_value == alpha (0.05 exactly) is a
    # FAIL, not a PASS.  n_draws=19 with ge=0 gives p = (1+0)/20 = 0.05.
    signal = np.arange(300, dtype=float)
    outcome = np.arange(300, dtype=float)  # |corr| = 1, no shuffle ties
    result = shuffled_permutation(signal, outcome, n_draws=19, seed=0)
    assert result["p_value"] == pytest.approx(0.05)
    assert result["verdict"] == "FAIL"


def test_shuffled_permutation_degenerate_gives_nan_percentile() -> None:
    # zero-variance signal: nothing was measured, so the percentile
    # must be NaN with a note instead of a plausible-looking 0.0
    gen = np.random.default_rng(26)
    flat_signal = np.ones(300)
    outcome = gen.standard_normal(300)
    result = shuffled_permutation(flat_signal, outcome, seed=0)
    assert result["verdict"] == "FAIL"
    assert np.isnan(result["stat"])
    assert np.isnan(result["percentile"])
    assert "note" in result
    assert "degenerate" in result["note"]


def test_placebo_placebo_strict_alpha_boundary() -> None:
    # same strict-boundary contract as the permutation control: with a
    # huge planted effect no placebo draw matches (ge=0), so n_draws=19
    # yields p = 1/20 = 0.05 exactly -> FAIL
    gen = np.random.default_rng(27)
    outcome = gen.standard_normal(300)
    outcome[[10, 20, 30]] += 50.0
    result = placebo_placebo([10, 20, 30], outcome, n_draws=19, seed=0)
    assert result["p_value"] == pytest.approx(0.05)
    assert result["verdict"] == "FAIL"


def test_placebo_placebo_counts_dropped_out_of_range_events() -> None:
    gen = np.random.default_rng(28)
    outcome = pd.Series(gen.standard_normal(300))
    result = placebo_placebo([10, 999, -5, 150], outcome, seed=0)
    assert result["n_events"] == 2
    assert result["n_out_of_range"] == 2
    # in-range behaviour is unchanged when nothing is dropped
    clean = placebo_placebo([10, 150], outcome, seed=0)
    assert clean["n_events"] == 2
    assert clean["n_out_of_range"] == 0


def test_subperiod_stability_separates_spread_from_concentrated() -> None:
    concentrated_ev = np.zeros(300)
    concentrated_ev[:100] = 1.0
    concentrated = subperiod_stability(
        concentrated_ev, periods=3, n_perm=200, seed=0
    )
    assert concentrated["verdict"] == "FAIL"
    assert concentrated["stat"] == pytest.approx(1.0)
    assert concentrated["sign_consistent"] is True
    assert concentrated["period_effects"] == pytest.approx([100.0, 0.0, 0.0])

    gen = np.random.default_rng(17)
    spread_ev = 0.2 + gen.standard_normal(300) * 0.05
    spread = subperiod_stability(spread_ev, periods=3, n_perm=200, seed=0)
    assert spread["verdict"] == "PASS"
    # default is now the AUTO cap (1/3 + 0.15 = 0.4833..), tighter than
    # the legacy 0.60; the spread evidence (measured max_share ~0.348)
    # still clears it
    assert spread["max_share_allowed"] == pytest.approx(1.0 / 3.0 + 0.15)
    assert spread["stat"] < spread["max_share_allowed"]
    assert spread["sign_consistent"] is True


def test_subperiod_stability_needs_minimum_data() -> None:
    gen = np.random.default_rng(18)
    daily_ev = 0.2 + gen.standard_normal(20) * 0.05
    result = subperiod_stability(daily_ev, periods=3, n_perm=200, seed=0)
    assert result["verdict"] == "FAIL"
    assert "insufficient" in str(result.get("note", ""))


def test_subperiod_stability_share_cap_is_strict() -> None:
    # unified strict thresholds: max_share == max_share_allowed is a
    # FAIL, only strictly-below passes
    concentrated_ev = np.zeros(300)
    concentrated_ev[:100] = 1.0
    at_edge = subperiod_stability(
        concentrated_ev,
        periods=3,
        max_share_allowed=1.0,  # exactly the observed max_share
        n_perm=200,
        seed=0,
    )
    assert at_edge["stat"] == pytest.approx(1.0)
    assert at_edge["verdict"] == "FAIL"


def test_subperiod_stability_default_cap_is_auto_scaled() -> None:
    # deliberate default change (review case R4): the default cap is
    # now the AUTO value 1/periods + 0.15, NOT the legacy fixed 0.60 -
    # a kill gate must be conservative by default.  Pinned numbers:
    # 0.4833.. at periods=3, 0.35 at periods=5.
    concentrated_ev = np.zeros(300)
    concentrated_ev[:100] = 1.0
    at_three = subperiod_stability(concentrated_ev, periods=3, seed=0)
    assert at_three["max_share_allowed"] == pytest.approx(1.0 / 3.0 + 0.15)
    at_five = subperiod_stability(concentrated_ev, periods=5, seed=0)
    assert at_five["max_share_allowed"] == pytest.approx(0.35)


def test_subperiod_stability_auto_cap_scales_with_periods() -> None:
    # five chunks with shares [0.5, 0.125 x 4]: one chunk holds half
    # the evidence.  The legacy fixed 0.60 cap waves that through for
    # periods=5; the auto cap 1/periods + 0.15 = 0.35 (the default)
    # kills it.
    daily_ev = np.concatenate(
        [np.full(60, 2.0), np.full(240, 0.5)]
    )  # chunk sums: [120, 30, 30, 30, 30] -> max_share 0.5
    default = subperiod_stability(daily_ev, periods=5, n_perm=200, seed=0)
    assert default["max_share_allowed"] == pytest.approx(0.35)
    assert default["stat"] == pytest.approx(0.5)
    assert default["verdict"] == "FAIL"

    explicit_none = subperiod_stability(
        daily_ev, periods=5, max_share_allowed=None, n_perm=200, seed=0
    )
    # explicit None is the default again: same seed, same result
    assert explicit_none == default


def test_subperiod_stability_fixed_cap_warns_for_four_plus_periods() -> None:
    # legacy opt-in: an explicitly fixed cap at periods >= 4 cannot
    # scale with periods (fair share 1/periods) and is anti-
    # conservative, so it must raise RuntimeWarning
    daily_ev = np.concatenate([np.full(60, 2.0), np.full(240, 0.5)])
    with pytest.warns(RuntimeWarning, match="anti-conservative"):
        legacy = subperiod_stability(
            daily_ev, periods=5, max_share_allowed=0.60, n_perm=200, seed=0
        )
    assert legacy["max_share_allowed"] == pytest.approx(0.60)
    assert legacy["stat"] == pytest.approx(0.5)
    assert legacy["verdict"] == "PASS"  # meaninglessly lenient at p=5


def test_subperiod_stability_fixed_cap_silent_below_four_periods() -> None:
    # a fixed cap is calibrated for periods=3 (the legacy 0.60): an
    # explicit value there is legitimate and must NOT warn
    concentrated_ev = np.zeros(300)
    concentrated_ev[:100] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        result = subperiod_stability(
            concentrated_ev,
            periods=3,
            max_share_allowed=0.60,
            n_perm=200,
            seed=0,
        )
    assert result["max_share_allowed"] == pytest.approx(0.60)
    assert result["stat"] == pytest.approx(1.0)
    assert result["verdict"] == "FAIL"


def test_calendar_control_contract_on_planted_edge() -> None:
    gen = np.random.default_rng(19)
    day_index = pd.date_range("2022-01-03", periods=400, freq="B")
    signal = pd.Series(gen.standard_normal(400), index=day_index)
    fwd_ret = pd.Series(
        0.3 * signal.to_numpy() + gen.standard_normal(400), index=day_index
    )
    dummies = pd.get_dummies(day_index.year).to_numpy(dtype=float)
    result = calendar_control(fwd_ret, signal, dummies)
    assert set(result) >= {
        "stat",
        "p_value",
        "verdict",
        "t_plain",
        "p_plain",
        "n_year_dummies",
        "n",
    }
    assert result["verdict"] == "PASS"
    assert result["n_year_dummies"] == 2
    assert result["n"] == 400


def test_calendar_control_survives_complete_one_hot_block() -> None:
    # pd.get_dummies(year) is a complete one-hot partition: summing to
    # the intercept.  The trap is defused by dropping the last dummy,
    # the signal keeps a real t-stat, and n_year_dummies still reports
    # the pre-drop count.
    gen = np.random.default_rng(23)
    day_index = pd.date_range("2022-01-03", periods=600, freq="B")
    signal = pd.Series(gen.standard_normal(600), index=day_index)
    fwd_ret = pd.Series(
        0.3 * signal.to_numpy() + gen.standard_normal(600), index=day_index
    )
    dummies = pd.get_dummies(day_index.year).to_numpy(dtype=float)
    assert dummies.shape[1] == 3
    result = calendar_control(fwd_ret, signal, dummies)
    assert result["verdict"] == "PASS"
    assert result["n_year_dummies"] == 3
    assert np.isfinite(result["stat"])


def test_calendar_control_inconclusive_when_signal_is_a_dummy() -> None:
    # regression: signal EXACTLY equal to a dummy column -> the FE
    # design is rank-deficient -> INCONCLUSIVE, never a fabricated PASS
    gen = np.random.default_rng(24)
    dummies = pd.get_dummies(np.repeat([0, 1], 200)).to_numpy(dtype=float)
    signal = dummies[:, 0].copy()
    fwd_ret = 0.3 * signal + gen.standard_normal(400)
    result = calendar_control(fwd_ret, signal, dummies)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "collinear with controls"
    assert np.isnan(result["stat"])


def test_calendar_control_rejects_non_dummy_columns() -> None:
    gen = np.random.default_rng(29)
    signal = gen.standard_normal(400)
    fwd_ret = gen.standard_normal(400)
    constant_half = np.full((400, 1), 0.5)
    with pytest.raises(ValueError, match="0/1 dummies"):
        calendar_control(fwd_ret, signal, constant_half)
    continuous = gen.standard_normal((400, 1))
    with pytest.raises(ValueError, match="0/1 dummies"):
        calendar_control(fwd_ret, signal, continuous)


def test_seeded_regression_demo_permutation_stat() -> None:
    """Seed lock for the demo's permutation stat (rng 42, n_days 800).

    The demo asserts only the property (finite Pearson r in (0, 1));
    the exact seeded magnitude lives here.
    """
    gen = np.random.default_rng(42)
    n_days = 800
    mom = np.zeros(n_days)
    shock = gen.standard_normal(n_days)
    for day in range(1, n_days):
        mom[day] = 0.7 * mom[day - 1] + 0.3 * shock[day]
    driver = gen.standard_normal(n_days)
    gen.standard_normal(n_days)  # mom7 noise (demo consumes it)
    gen.standard_normal(n_days)  # mom12 noise (demo consumes it)
    new_signal = driver + 0.5 * mom
    fwd = 0.20 * driver + 0.25 * mom + gen.standard_normal(n_days)
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    res_perm = shuffled_permutation(
        pd.Series(new_signal, index=idx),
        pd.Series(fwd, index=idx),
        n_draws=200,
        seed=0,
    )
    assert res_perm["stat"] == pytest.approx(0.2741827545181678, rel=1e-12)
    assert res_perm["p_value"] == pytest.approx(
        0.004975124378109453, rel=1e-12
    )


def test_run_all_examples_demo_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Layer-0: the planted edge prints PASS on all six controls.

    The seventh PASS is the auto-scale block's legacy-0.60 row (an
    explicit opt-in that prints its RuntimeWarning); the auto-cap row
    (now the default) on the same data prints FAIL.
    """
    run_all_examples()
    output = capsys.readouterr().out
    assert output.count("verdict=PASS") == 7
    assert "[1 shuffled_permutation]" in output
    assert "[6 calendar_control]" in output
    assert "5b subperiod_stability legacy 0.60 (opt-in)" in output
    assert "5b subperiod_stability auto cap (default)" in output
    assert "warning: fixed max_share_allowed=0.6 is anti-conservative" in (
        output
    )
