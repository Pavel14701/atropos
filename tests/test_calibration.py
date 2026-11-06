"""Calibration tests: p-values must behave like p-values.

A statistical layer over the unit tests: under a true null every
p-value producing machinery must reject at its nominal rate and spread
its p-values roughly uniformly.  Miscalibration is a class-of-bug
detector: peeking leaks, broken benchmarks and double counting all
show up as inflated rejection rates long before any single input
fails.  All tests here are marked ``slow``; each runs in seconds on
the fixed seeds below (total well under 30 seconds).

ML-engine calibration (P3 close-out): the walk-forward engine's
incremental gate and its deflated-Sharpe p-values get the same
null-world treatment (100 seeded noise worlds through the full
``run_ml_kill_protocol`` path).  Both probes are currently
ANTI-CONSERVATIVE -- documented findings pinned as ``xfail(strict)``:
the day the engine calibration is fixed, the tests XPASS and strict
semantics force the markers to be removed, never the asserts.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.bootstrap_ci import block_bootstrap_ci
from atropos.evaluation_metrics import deflated_sharpe
from atropos.ml_training import run_ml_kill_protocol
from atropos.null_control_suite import (
    calendar_control,
    placebo_placebo,
    shuffled_permutation,
    subperiod_stability,
)
from atropos.power_calculator import _power_at, simulate_power
from atropos.walkforward_framework import walkforward
from scipy import stats

HOUR_US = 3_600_000_000


def _ml_null_world(seed: int) -> dict[str, Any]:
    """One seeded pure-noise world through the ML kill protocol.

    ``logit`` on three noise features against a linear battery on the
    same features plus two noise controls, symmetric +/-0.8% executable
    returns, hourly stamps (n=2000, ~83 days): a TRUE null for every
    gate -- no incremental edge, no cost-clearing book, no stable
    importances, no DSR.  Deterministic: fixed seeds at every layer.
    Returns the logit model entry's gate records and DSR p-value.
    """
    rng = np.random.default_rng(80_000 + seed)
    n_rows = 2000
    feats = rng.standard_normal((n_rows, 3))
    controls = rng.standard_normal((n_rows, 2))
    exec_ret = 0.008 * np.sign(rng.standard_normal(n_rows))
    feature_frame = pd.DataFrame(
        {
            "f1": feats[:, 0],
            "f2": feats[:, 1],
            "f3": feats[:, 2],
            "c1": controls[:, 0],
            "c2": controls[:, 1],
        }
    )
    labels = (exec_ret > 0.004).astype(np.int64)
    stamps = np.arange(n_rows, dtype=np.int64) * HOUR_US
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = run_ml_kill_protocol(
            feature_frame,
            labels,
            exec_ret,
            stamps,
            config={
                "model": "logit",
                "k_blocks": 3,
                "bootstrap_b": 100,
                "n_min_events": 5,
                "prereg_hash": "calibration-null",
                "checks": {
                    "deflated_sharpe": True,
                    "importance_stability": False,
                },
            },
            battery_controls=["c1", "c2"],
        )
    entry = report["models"]["logit"]
    gates = {gate["check"]: gate for gate in entry["kills"]}
    return {
        "incremental": gates["incremental"],
        "dsr_p": entry["deflated_sharpe"],
        "n_events": entry["n_events"],
    }


@pytest.mark.slow
def test_shuffled_permutation_null_pvalues_roughly_uniform() -> None:
    """True null: permutation p-values are ~U(0,1), KS p > 0.01.

    200 independent null pairs (n=250, 200 draws each; ~3 s measured).
    KS rejects both a stuck-at-tiny and a stuck-at-1 p-value mass; the
    rejection-rate window 1-10% brackets the nominal 5% while leaving
    binomial room (measured: KS p ~ 0.16, rate 3.5%).
    """
    pvalues = np.empty(200)
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        signal = rng.standard_normal(250)
        outcome = rng.standard_normal(250)
        pvalues[seed] = shuffled_permutation(
            signal, outcome, n_draws=200, seed=2000 + seed
        )["p_value"]
    assert bool(np.all(np.isfinite(pvalues)))
    ks_pvalue = float(stats.kstest(pvalues, "uniform").pvalue)
    rejection_rate = float(np.mean(pvalues < 0.05))
    assert ks_pvalue > 0.01
    assert 0.01 <= rejection_rate <= 0.10


@pytest.mark.slow
def test_deflated_sharpe_on_pure_noise_does_not_overreject() -> None:
    """Pure noise, n_trials=10: p < 0.05 in under 10% of 100 seeds.

    The multiple-testing benchmark (expected max SR of 10 unit-variance
    trials, ~1.58 annualized) sits far above what 3 years of noise can
    produce, so a correct implementation rejects essentially never
    (measured: 0/100).  The median-p floor (> 0.80, measured 0.994)
    additionally catches a silently dropped benchmark, which would
    drag the median towards 0.5.
    """
    pvalues = np.empty(100)
    for seed in range(100):
        noise = np.random.default_rng(3000 + seed).standard_normal(750)
        pvalues[seed] = deflated_sharpe(noise * 0.01, n_trials=10)["p_value"]
    assert float(np.mean(pvalues < 0.05)) < 0.10
    assert float(np.median(pvalues)) > 0.80


@pytest.mark.slow
def test_walkforward_null_wf_t_is_calibrated() -> None:
    """True null, step == test_window: wf_t behaves like N(0,1).

    NOTE: the brief asked for median |wf_t| < 0.5, but that threshold
    is statistically wrong for a CALIBRATED t-statistic: the median of
    |N(0,1)| is 0.674, so a median below 0.5 would signal over-
    shrinkage, not honesty.  The assertion brackets the theoretical
    value instead (measured median over the fixed 50 seeds: 0.715;
    nominal P(|t| >= 2) = 4.6%, measured 6%).  An upper median bound
    of 0.95 and a 15% cap on |wf_t| >= 2 still catch peeking leaks,
    which push the statistic far above the unit-normal scale.
    Budget: 50 walk-forward runs of n=400, ~1 s measured.
    """
    magnitudes = np.empty(50)
    for seed in range(50):
        rng = np.random.default_rng(4000 + seed)
        frame = pd.DataFrame(
            {
                "signal": rng.standard_normal(400),
                "outcome": rng.standard_normal(400),
            }
        )
        magnitudes[seed] = abs(
            walkforward(frame, train_window=150, test_window=25, step=25)[
                "wf_t"
            ]
        )
    median_magnitude = float(np.median(magnitudes))
    assert 0.40 < median_magnitude < 0.95
    assert float(np.mean(magnitudes >= 2.0)) < 0.15


@pytest.mark.slow
def test_block_bootstrap_ci_coverage_for_mean() -> None:
    """95% block-bootstrap CI covers the true mean 90-98% of seeds.

    200 replications of n=200 normal draws with n_boot=800, block=10
    (~0.5 s measured; coverage 93%).  A coverage far below 90% means
    the interval is too narrow (dependence mishandled); far above 98%
    it is uselessly wide.  The window deliberately allows the mild
    under-coverage known for percentile bootstrap intervals.
    """
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(50000 + seed)
        sample = rng.normal(0.001, 0.01, 200)
        interval = block_bootstrap_ci(
            sample, n_boot=800, block=10, stat=np.mean, seed=60000 + seed
        )
        if interval["ci_low"] <= 0.001 <= interval["ci_high"]:
            hits += 1
    coverage = hits / 200
    assert 0.90 <= coverage <= 0.98


@pytest.mark.slow
def test_null_controls_fail_fraction_under_pure_noise() -> None:
    """True null: each control FAILs ~95% of the time at alpha=0.05.

    FPR direction of the null-control calibration: PASS needs p < 0.05,
    so under pure noise a control must FAIL about 95% of the time.  A
    FAIL fraction far below 0.85 means the control passes garbage
    (anti-conservative); exactly 1.0 means it never passes anything
    (broken or stuck on the degenerate path).  momentum_control /
    known_features_control are excluded: they need their own factor
    data, not just noise.  Measured on the fixed 150 seeds:
    shuffled_permutation 142/150 = 0.947, placebo_placebo 145/150 =
    0.967, calendar_control 137/150 = 0.913 (600 business days spanning
    three years -> two year dummies, so the fixed-effects path is
    actually exercised).  subperiod_stability is structurally different
    and gets its own window: its PASS gate is sign consistency across
    all three chunks (a 1-in-4 event under a symmetric null) AND
    max_share below the DEFAULT auto cap 1/3 + 0.15 = 0.4833 (pinned
    measured P(max_share < 0.4833) ~= 0.32; the legacy fixed 0.60 cap
    had P ~= 2/3), so its design-implied null FAIL rate is
    1 - (1/4)(0.32) ~= 0.92 (measured 138/150 = 0.920, re-pinned after
    the deliberate default change from 0.60 to auto-scale), not
    the p-value 95%; the window still fails if the sign check is
    dropped (FAIL -> ~0.68) or the cap is loosened back toward the
    legacy 0.60 (-> ~0.83).
    Budget: ~5 s measured (150 seeds x 4 controls, 200 draws each).
    """
    n_seeds = 150
    fails = {
        "shuffled_permutation": 0,
        "placebo_placebo": 0,
        "subperiod_stability": 0,
        "calendar_control": 0,
    }
    cal_index = pd.date_range("2022-01-03", periods=600, freq="B")
    year_dummies = pd.get_dummies(cal_index.year, drop_first=True).to_numpy(
        dtype=float
    )
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        outcome = rng.standard_normal(250)
        signal = rng.standard_normal(250)
        event_positions = rng.choice(250, size=20, replace=False)
        daily_ev = rng.standard_normal(600)
        outcome_cal = rng.standard_normal(600)
        signal_cal = rng.standard_normal(600)
        if (
            shuffled_permutation(
                signal, outcome, n_draws=200, seed=20_000 + seed
            )["verdict"]
            == "FAIL"
        ):
            fails["shuffled_permutation"] += 1
        if (
            placebo_placebo(
                event_positions, outcome, n_draws=200, seed=20_001 + seed
            )["verdict"]
            == "FAIL"
        ):
            fails["placebo_placebo"] += 1
        if (
            subperiod_stability(
                daily_ev, periods=3, n_perm=200, seed=20_002 + seed
            )["verdict"]
            == "FAIL"
        ):
            fails["subperiod_stability"] += 1
        if (
            calendar_control(
                pd.Series(outcome_cal, index=cal_index),
                pd.Series(signal_cal, index=cal_index),
                year_dummies,
            )["verdict"]
            == "FAIL"
        ):
            fails["calendar_control"] += 1
    for name in (
        "shuffled_permutation",
        "placebo_placebo",
        "calendar_control",
    ):
        fail_fraction = fails[name] / n_seeds
        assert 0.85 <= fail_fraction <= 1.0, (name, fail_fraction)
    subperiod_fraction = fails["subperiod_stability"] / n_seeds
    assert 0.85 <= subperiod_fraction <= 0.97, subperiod_fraction


@pytest.mark.slow
def test_shuffled_permutation_power_on_planted_edge() -> None:
    """Planted edge: shuffled_permutation PASSes >= 80% of 200 seeds.

    Power direction of the calibration: a control that rejects real
    edges is as useless as one that passes garbage.  The planted world
    (outcome = 0.5*driver + noise, signal = driver + noise, n=250)
    carries corr(signal, outcome) ~= 0.32 (t ~= 5), so an honest
    permutation control must essentially always PASS: measured 200/200
    = 1.000 on the fixed seeds.  The 0.80 floor tolerates a modest
    power loss without hiding a broken implementation (a verdict stuck
    at FAIL scores 0.0).
    Budget: ~5 s measured (200 seeds x 200 permutation draws).
    """
    passes = 0
    for seed in range(200):
        rng = np.random.default_rng(30_000 + seed)
        driver = rng.standard_normal(250)
        outcome = 0.5 * driver + rng.standard_normal(250)
        signal = driver + rng.standard_normal(250)
        if (
            shuffled_permutation(
                signal, outcome, n_draws=200, seed=40_000 + seed
            )["verdict"]
            == "PASS"
        ):
            passes += 1
    assert passes / 200 >= 0.80


@pytest.mark.slow
def test_simulate_power_calibrates_against_analytic_power() -> None:
    """simulate_power matches the analytic noncentral-t power (<0.05).

    The right contract for a Monte Carlo estimator is statistical
    equivalence / calibration against the closed form, not bit
    identity: ``simulate_power`` draws random non-orthonormal designs,
    so its rejection frequency only CONVERGES to the analytic power of
    the private ``atropos.power_calculator._power_at`` (imported here
    deliberately - a test may reach a private helper).  Four design
    points spanning n_obs, partial R2, n_predictors and alpha; 2000
    sims each give a binomial s.e. <= 0.011, so the 0.05 window is a
    >4-sigma gate.  Measured diffs on the fixed seeds: -0.014, -0.002,
    -0.017, -0.001; budget ~2 s measured (well under the 30 s slow
    budget).
    """
    cases = (
        (2000, 0.003, 25, 0.05),
        (500, 0.01, 5, 0.05),
        (500, 0.02, 5, 0.05),
        (1000, 0.0025, 10, 0.10),
    )
    for case_idx, (n_obs, partial_r2, n_pred, alpha) in enumerate(cases):
        analytic = _power_at(partial_r2, n_obs, n_pred, alpha)
        empirical = simulate_power(
            partial_r2,
            n_obs,
            n_pred,
            alpha=alpha,
            n_sims=2000,
            seed=71_000 + case_idx,
        )
        assert abs(empirical - analytic) < 0.05, (
            n_obs,
            partial_r2,
            n_pred,
            alpha,
            analytic,
            empirical,
        )


# ---------------------------------------------------------------------------
# ML engine calibration (P3 close-out): the walk-forward gates under null
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "FINDING (anti-conservative, P3 close-out): the incremental "
        "gate false-passes ~43-46 of 100 pure-noise worlds -- the "
        "paired bootstrap t of the EV difference between two thin "
        "EV-argmax books is over-dispersed under the null (measured "
        "P(t>2) ~= 0.45, P(|t|>2) ~= 0.9, max t 177 at b=100 / 728 at "
        "b=2000): row resampling does not reproduce the winner's-curse "
        "selection of the fold thresholds, so the observed EV "
        "difference rides into the draws and inflates t.  Among the "
        "21 powered worlds (>= 500 events) it is still 7-8/21.  The "
        "assert below is the CALIBRATION CONTRACT (<= 10%); do not "
        "weaken it -- fix the statistic (e.g. post-selection or "
        "book-stable resampling) and remove the marker."
    )
)
def test_ml_engine_incremental_gate_fpr_under_noise_null() -> None:
    """True null: the incremental gate must kill noise, FPR <= 10%.

    The gate's job on this world is to TRIGGER (no incremental edge
    over the battery); a world where it stays off is a false pass --
    the engine waving a noise model through the incrementality
    criterion.  100 seeded worlds, logit vs a linear battery with two
    noise controls (measured on the fixed seeds: 43/100 false passes
    at bootstrap_b=100, 46/100 at the production b=2000; the two
    configs bracket the rate, the config below freezes the fast one).
    Budget: ~4 s measured (100 engine runs at reduced capacity), well
    under the 120 s budget.
    """
    false_passes = 0
    for seed in range(100):
        world = _ml_null_world(seed)
        gate = world["incremental"]
        assert isinstance(gate["enabled"], bool)
        if not gate["triggered"] and not gate["inconclusive"]:
            false_passes += 1
    assert false_passes <= 10, false_passes


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "FINDING (P3 close-out): the engine's deflated-Sharpe CDF "
        "values are not uniform under the null -- KS p ~= 0.001 over "
        "the 98 finite null-world p-values (median ~= 0.37, i.e. "
        "skewed low / over-rejecting), although the rejection rate at "
        "0.05 stays under the 10% bound (~0.08 measured).  The KS "
        "assert below is the CALIBRATION CONTRACT; when the DSR "
        "standardization on short zero-padded daily streams is fixed, "
        "the test XPASSes and strict semantics force the marker out."
    )
)
def test_ml_engine_deflated_sharpe_pvalues_ks_uniform_under_null() -> None:
    """True null: engine DSR p-values ~U(0,1), KS p > 0.01, FPR <= 10%.

    The engine reports the DSR CDF value as ``deflated_sharpe``; under
    a true null that number IS a p-value and must spread roughly
    uniformly.  100 seeded noise worlds through the full engine path
    (logit, daily EV streams over ~83 zero-padded days, n_trials=5
    against the run's own three-stream family).  Two contracts: the
    rejection rate at 0.05 stays <= 10%, and the KS uniformity holds
    at p > 0.01.  Budget: ~4 s measured (100 engine runs at reduced
    capacity), well under the 120 s budget.
    """
    pvalues: list[float] = []
    for seed in range(100):
        world = _ml_null_world(seed)
        dsr_p = world["dsr_p"]
        if dsr_p is not None:
            assert 0.0 <= float(dsr_p) <= 1.0
            pvalues.append(float(dsr_p))
    # at most a couple of worlds may leave the book empty (no DSR
    # decision); the uniformity sample must stay representative
    assert len(pvalues) >= 95, len(pvalues)
    rejection_rate = float(np.mean(np.asarray(pvalues) < 0.05))
    assert rejection_rate <= 0.10, rejection_rate
    ks_pvalue = float(stats.kstest(np.asarray(pvalues), "uniform").pvalue)
    assert ks_pvalue > 0.01, ks_pvalue
