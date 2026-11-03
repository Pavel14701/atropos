"""Tests for atropos.power_calculator analytic power and MC agreement.

Checks known-answer t-statistics, monotonicity of required N and of the
detectable effect, agreement between ``simulate_power`` and the
closed-form noncentral-t power, ``check_design_rule`` output, and the
``ValueError`` guard rails.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from atropos.power_calculator import (
    check_design_rule,
    detectable_effect,
    power_for_effect,
    run_all_examples,
    simulate_power,
    t_stat_for_partial_r2,
)
from scipy import stats


def analytic_power(partial_r2: float, n_obs: int, n_predictors: int) -> float:
    """Closed-form two-sided t-test power at alpha = 0.05."""
    df = n_obs - n_predictors - 1
    f2 = partial_r2 / (1.0 - partial_r2)
    ncp = float(np.sqrt(f2 * n_obs))
    t_crit = float(stats.t.ppf(0.975, df))
    return float(
        stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    )


def test_t_stat_for_partial_r2_known_answer() -> None:
    # df = 11 - 1 - 1 = 9, so t = sqrt(0.5 * 9 / 0.5) = 3 exactly
    assert t_stat_for_partial_r2(0.5, n_obs=11, n_predictors=1) == 3.0
    assert t_stat_for_partial_r2(0.1, 200, 5) < t_stat_for_partial_r2(
        0.3, 200, 5
    )
    degenerate = t_stat_for_partial_r2(0.5, n_obs=10, n_predictors=10)
    assert np.isnan(degenerate)


def test_power_for_effect_monotone_in_effect_size() -> None:
    n_strong = power_for_effect(0.01, n_predictors=25)
    n_mid = power_for_effect(0.005, n_predictors=25)
    n_weak = power_for_effect(0.003, n_predictors=25)
    assert n_strong < n_mid < n_weak
    # documented design point: R2 = 0.003 needs roughly N ~ 2600
    assert 2000 < n_weak < 3500


def test_power_for_effect_result_reaches_target_power() -> None:
    n_needed = power_for_effect(0.02, n_predictors=5)
    assert analytic_power(0.02, n_needed, 5) >= 0.80
    # one observation fewer must drop below the target: N is minimal
    assert analytic_power(0.02, n_needed - 1, 5) < 0.80


def test_detectable_effect_monotone_in_n_obs() -> None:
    r2_small_n = detectable_effect(500, n_predictors=25)
    r2_mid_n = detectable_effect(2000, n_predictors=25)
    r2_big_n = detectable_effect(8000, n_predictors=25)
    assert r2_small_n > r2_mid_n > r2_big_n


def test_detectable_effect_result_matches_target_power() -> None:
    r2_min = detectable_effect(2000, n_predictors=25)
    assert analytic_power(r2_min, 2000, 25) == pytest.approx(0.80, abs=0.01)


def test_simulate_power_agrees_with_analytic_and_monotone() -> None:
    mid = simulate_power(0.02, n_obs=200, n_predictors=5, n_sims=300, seed=0)
    strong = simulate_power(
        0.10, n_obs=200, n_predictors=5, n_sims=300, seed=0
    )
    assert mid == pytest.approx(analytic_power(0.02, 200, 5), abs=0.15)
    assert strong > mid


def test_check_design_rule_reports_expected_fields() -> None:
    rule = check_design_rule(n_obs=2000, n_sims=100)
    assert set(rule) == {
        "n",
        "n_predictors",
        "partial_r2",
        "t_analytic",
        "r2_needed_for_t_equals_2",
        "empirical_power",
        "alpha",
        "note",
    }
    assert rule["n"] == 2000
    assert rule["alpha"] == 0.05
    assert rule["t_analytic"] == t_stat_for_partial_r2(0.003, 2000, 25)
    assert rule["t_analytic"] == pytest.approx(2.44, abs=0.1)
    assert rule["r2_needed_for_t_equals_2"] == pytest.approx(
        4.0 / (4.0 + 1974)
    )
    assert 0.0 <= rule["empirical_power"] <= 1.0
    assert "N=2000" in rule["note"]


def test_invalid_designs_raise_value_error() -> None:
    with pytest.raises(ValueError, match=r"partial_r2 must be in \(0, 1\)"):
        power_for_effect(0.0, n_predictors=25)
    with pytest.raises(ValueError, match="n_predictors must be >= 0"):
        power_for_effect(0.3, n_predictors=-1)
    with pytest.raises(ValueError, match="exceeds"):
        power_for_effect(1e-12, n_predictors=25)
    with pytest.raises(
        ValueError, match=r"alpha and power must be in \(0, 1\)"
    ):
        detectable_effect(2000, n_predictors=25, alpha=1.5)
    with pytest.raises(ValueError, match=r"n must exceed n_predictors"):
        detectable_effect(10, n_predictors=25)
    with pytest.raises(ValueError, match=r"n must exceed n_predictors"):
        simulate_power(0.05, n_obs=10, n_predictors=25)
    # a negative control count used to pass and fabricate extra df
    with pytest.raises(ValueError, match="n_predictors must be >= 0"):
        detectable_effect(2000, n_predictors=-1)
    with pytest.raises(ValueError, match="n_predictors must be >= 0"):
        detectable_effect(2000, n_predictors=-25)


def test_zero_predictors_is_a_valid_no_controls_design() -> None:
    # n_predictors=0 ("no controls") is valid in both entry points
    n_needed = power_for_effect(0.3, n_predictors=0)
    assert n_needed > 0
    min_r2_none = detectable_effect(2000, n_predictors=0)
    assert 0.0 < min_r2_none < 1.0
    # more controls cost degrees of freedom: the detectable effect grows
    min_r2_few = detectable_effect(200, n_predictors=1)
    min_r2_many = detectable_effect(200, n_predictors=50)
    assert min_r2_few < min_r2_many
    est_none = simulate_power(0.3, n_obs=100, n_predictors=0, n_sims=50)
    assert 0.0 <= est_none <= 1.0


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (
            lambda: power_for_effect(
                0.3,
                n_predictors=2.5,  # type: ignore[arg-type]
            ),
            "n_predictors",
        ),
        (
            lambda: detectable_effect(
                2000.5,  # type: ignore[arg-type]
                25,
            ),
            "n_obs",
        ),
        (
            lambda: detectable_effect(
                2000,
                n_predictors=2.5,  # type: ignore[arg-type]
            ),
            "n_predictors",
        ),
        (
            lambda: simulate_power(
                0.3,
                n_obs=200.5,  # type: ignore[arg-type]
                n_predictors=5,
            ),
            "n_obs",
        ),
        (
            lambda: simulate_power(
                0.3,
                n_obs=200,
                n_predictors=2.5,  # type: ignore[arg-type]
            ),
            "n_predictors",
        ),
        (
            lambda: check_design_rule(
                n_obs=2000.5  # type: ignore[arg-type]
            ),
            "n_obs",
        ),
        (
            lambda: check_design_rule(
                n_predictors=2.5  # type: ignore[arg-type]
            ),
            "n_predictors",
        ),
    ],
)
def test_integer_parameters_reject_fractional_values(
    call: Callable[[], object], name: str
) -> None:
    """Non-integer design dimensions raise ValueError.

    Regression class: ``int(value)`` coercion used to truncate
    ``n_obs=2000.5`` / ``n_predictors=2.5`` silently, shifting every
    degrees-of-freedom computation.
    """
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        call()


def test_run_all_examples_demo_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Layer-0: the demo's internal asserts hold on the design point."""
    run_all_examples()
    output = capsys.readouterr().out
    assert "power_calculator demo" in output
    assert "N required = " in output
    assert "min partial R2 = " in output
    assert "check_design_rule():" in output
