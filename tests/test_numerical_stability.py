"""Numerical-stability tests: extremes must fail closed, not lie.

Overflowing equity paths, denormal design columns, skewed fat-tail
samples and bootstrap replicates with a single gigantic outlier are
not exotic -- they are what a live kill pipeline meets on a bad data
day.  Each test pins one rule: the metric either reports a FINITE
value, or reports NaN / raises "not computable"; it must never emit a
plausible-looking infinite or fabricated number that a downstream
verdict could mistake for evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos._utils import ols_hc0, sane_variance_mask
from atropos.bootstrap_ci import (
    block_bootstrap_ci,
    percentile_ci,
    stationary_bootstrap_ci,
)
from atropos.evaluation_metrics import deflated_sharpe
from atropos.track_record import drawdown_floor_check

# 1 + 1e308 rounds to 1e308, so two such returns compound the equity
# to +inf at the THIRD element: the inf sits in the MIDDLE of the path,
# unlike the all-overflow series pinned in tests/test_track_record.py.
_INF_MID_WIPE_TO_NAN = np.array([1e308, 1e308, -1.0])
_INF_MID_WIPE_TO_NEGATIVE = np.array([1e308, 1e308, -2.0])


def test_drawdown_inf_mid_path_fails_closed() -> None:
    """Inf mid-path plus wipe-out fails closed: error or total loss.

    An equity path that hits inf mid-series and then wipes out is
    either an error or a total loss -- never "not breached".  Two
    guard orderings are exercised, both with the inf in the middle of
    the compounded path (``[1.0, 1e308, inf, ...]``):

    * return of exactly ``-1.0`` after the overflow: ``inf * 0`` is
      NaN, the final-equity wipe-out check ``equity[-1] <= 0`` does
      NOT fire on NaN, so the drawdown itself is NaN and
      ``drawdown_floor_check`` must raise "drawdown not computable" --
      a kill barrier fails closed rather than trusting a NaN path;
    * return of ``-2.0`` after the overflow: the compounded final
      equity is ``-inf``, the wipe-out branch fires FIRST and reports
      the documented total loss ``mdd = -1.0`` (breached), which is
      semantically right -- everything was lost -- even though the
      path passed through inf to get there.
    """
    with pytest.raises(ValueError, match="drawdown not computable"):
        drawdown_floor_check(_INF_MID_WIPE_TO_NAN, dd_floor=-0.25)

    verdict = drawdown_floor_check(_INF_MID_WIPE_TO_NEGATIVE, dd_floor=-0.25)
    assert verdict["mdd"] == -1.0
    assert verdict["breached"] is True
    assert verdict["n_obs"] == 3


def test_ols_hc0_denormal_column_is_rank_deficient() -> None:
    """Denormal design columns collapse to rank deficiency + NaN.

    A denormal-scaled design column collapses to rank deficiency and
    the full NaN payload -- no pseudo-inverse t-stats from dust.  The
    subnormal column (values ~1e-320) IS stored as non-zero data (no
    FTZ flushing happens on the way in), but
    ``np.linalg.matrix_rank`` compares singular values RELATIVELY to
    the dominant one (the intercept, sv ~ sqrt(n)): any column scaled
    below ~eps of the dominant direction counts as zero, denormal or
    not -- a normal-range 1e-300 column collapses through the very
    same relative tolerance.  ``ols_hc0`` then flags
    ``rank_deficient=True`` with every statistic NaN, which callers
    must treat as INCONCLUSIVE instead of reading minimum-norm
    t-statistics fabricated by a pseudo-inverse.
    """
    rng = np.random.default_rng(0)
    n_rows = 60
    target = rng.standard_normal(n_rows)
    denormal_col = rng.standard_normal(n_rows) * 1e-320
    assert bool(np.any(denormal_col != 0.0))  # real data, not pre-flushed

    design = np.column_stack([np.ones(n_rows), denormal_col])
    assert np.linalg.matrix_rank(design) == 1  # relative SVD tolerance

    result = ols_hc0(target, design)
    assert result["rank_deficient"] is True
    for key in ("beta", "se", "t", "p"):
        assert bool(np.isnan(result[key]).all()), key
    assert math.isnan(float(result["r2"]))
    assert result["n"] == n_rows
    assert result["df"] == n_rows - 2

    # the same collapse for a NORMAL-range (non-subnormal) column: the
    # tolerance is scale-relative, not a denormal artefact
    tiny_col = rng.standard_normal(n_rows) * 1e-300
    result_scaled = ols_hc0(
        target, np.column_stack([np.ones(n_rows), tiny_col])
    )
    assert result_scaled["rank_deficient"] is True


def test_deflated_sharpe_extreme_skew_never_fabricates() -> None:
    """Extreme skew never fabricates a DSR outside [0, 1].

    A one-giant-outlier sample yields a DSR inside [0, 1] or NaN --
    never a fabricated certainty outside the probability range.  With
    99 calm returns plus a single 1e4-scale outlier the sample skew
    (~10) and kurtosis (~98) sit far outside the normal
    approximation's comfort zone.  The degenerate-vertex branch of the
    same denominator (``denom2 <= 0`` -> NaN via a caller-supplied
    ``sr_annual``) is already pinned in
    ``tests/test_evaluation_metrics.py``; with the SAMPLE Sharpe the
    quadratic denominator stays non-negative in exact arithmetic
    (two-point-like samples sit on its vertex,
    ``(1 - skew * SR / 2)^2 >= 0``), so what remains to pin here is
    that extreme skew cannot push the DSR outside [0, 1] or desync
    ``p_value`` from ``1 - dsr``.
    """
    rng = np.random.default_rng(6)
    rets = np.concatenate([rng.standard_normal(99) * 0.01, [1.0e4]])
    out = deflated_sharpe(rets, n_trials=5)
    dsr = float(out["dsr"])
    assert math.isnan(dsr) or 0.0 <= dsr <= 1.0
    if math.isfinite(dsr):
        assert float(out["p_value"]) == pytest.approx(1.0 - dsr)
    assert math.isfinite(float(out["sr_annual"]))
    assert abs(float(out["skew"])) > 5.0  # the sample IS pathological


def test_bootstrap_ci_single_huge_outlier_stays_finite() -> None:
    """One 1e6-scale outlier keeps the bootstrap CI finite and ordered.

    The CI stays finite (or NaN), never infinite, and the interval
    remains ordered.  The mean of the contaminated sample is dominated
    by the outlier, and any replicate whose block swallows it carries
    a ~1e5-scale mean; both bootstrap variants must still report
    finite bounds (a CI of ``[-inf, inf]`` is a silence failure that a
    verdict stage could mistake for "wide but usable").
    """
    rng = np.random.default_rng(2)
    rets = rng.standard_normal(250) * 0.01
    rets[137] = 1.0e6
    for interval in (
        block_bootstrap_ci(rets, n_boot=400, block=20, stat=np.mean, seed=1),
        stationary_bootstrap_ci(
            rets, n_boot=400, mean_block=20, stat=np.mean, seed=1
        ),
    ):
        point = float(interval["point_estimate"])
        low, high = float(interval["ci_low"]), float(interval["ci_high"])
        assert math.isfinite(point)
        for bound in (low, high):
            assert not math.isinf(bound)
            assert not math.isnan(bound)
        assert low <= point <= high

    # percentile_ci drops non-finite entries instead of quantiling inf
    contaminated = np.array([np.inf, -np.inf, 1.0, 2.0, 3.0, 4.0])
    low, high = percentile_ci(contaminated, 0.95)
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high


def test_sane_variance_mask_separates_dust_from_signal() -> None:
    """Dust-scale sd is dropped, genuine 1e-6 * |mean| sd is kept.

    An sd at rounding-dust scale (1e-16 * |mean|) is dropped while an
    sd at 1e-6 * |mean| is kept -- at every mean scale, plus the
    epsilon floor for zero-mean rows.  The mask is RELATIVE: a row
    with mean 1e8 and sd 1e-8 carries the same dust ratio as mean 1
    and sd 1e-16, and must be dropped identically; a genuine sd of
    1e-6 * |mean| implies a per-period Sharpe of 1e6, huge but honest,
    and stays usable.  Zero-mean rows fall back to an absolute floor
    of ``rel_tolerance * eps`` (~2.2e-28): sd 1e-20 survives it, sd
    1e-30 does not.
    """
    means = np.array([1.0, 1.0, 1.0e8, 1.0e8, 0.0, 0.0])
    dust_sds = np.array([1e-16, 1e-16, 1e-8, 1e-8, 1e-30, 1e-20])
    signal_sds = np.array([1e-6, 1e-6, 1.0e2, 1.0e2, 1e-20, 1e-20])
    # dust rows (indices 0..3, plus the sub-floor zero-mean row 4)
    assert sane_variance_mask(means, dust_sds).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    # genuine dispersion rows survive at every scale
    assert sane_variance_mask(means, signal_sds).tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
    ]
