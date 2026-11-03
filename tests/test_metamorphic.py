"""Metamorphic tests: invariants that need no oracle.

The expected answer is never written down; instead a transformation of
the input (scaling, sign flip, reversal, offset, re-draw with another
seed) must change the output in a exactly predictable way -- or not at
all.  Property-based cases run under the derandomized hypothesis
profile registered in ``tests/conftest.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from atropos._utils import block_indices
from atropos.bootstrap_ci import block_bootstrap_ci, stationary_bootstrap_ci
from atropos.cost_model import turnover_series
from atropos.evaluation_metrics import max_drawdown, sharpe_ratio, turnover
from atropos.power_calculator import detectable_effect, power_for_effect
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

_RETURN_ARRAYS = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=30, max_value=200),
    elements=st.floats(min_value=-1.0, max_value=1.0),
)

_POSITION_ARRAYS = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=200),
    elements=st.floats(min_value=-5.0, max_value=5.0),
)

_COMPOUND_ARRAYS = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=150),
    elements=st.floats(min_value=-0.5, max_value=0.5),
)

# Distinct values only: long runs of one repeated value make bootstrap
# rows constant, where the CI degeneracy filter of sharpe_ratio loses
# scale-equivariance (a rounding-dust sd slips past "sds > 0").  That
# degenerate case is pinned by a dedicated strict-xfail test below.
_SPREAD_ARRAYS = st.lists(
    st.floats(min_value=-1.0, max_value=1.0),
    min_size=30,
    max_size=200,
    unique=True,
).map(np.asarray)


@given(
    n_obs=st.integers(min_value=1, max_value=60),
    n_sims=st.integers(min_value=1, max_value=7),
    block=st.integers(min_value=1, max_value=80),
    seed=st.integers(min_value=0, max_value=10**6),
)
def test_block_indices_domain_shape_and_reproducibility(
    n_obs: int, n_sims: int, block: int, seed: int
) -> None:
    """Shape (n_sims, n_obs), values in [0, n_obs), bitwise same seed.

    ``block`` may exceed ``n_obs`` (it is clamped inside); the index
    domain must stay valid for every parameter combination.
    """
    drawn = block_indices(n_obs, n_sims, block, np.random.default_rng(seed))
    repeat = block_indices(n_obs, n_sims, block, np.random.default_rng(seed))
    assert drawn.shape == (n_sims, n_obs)
    assert int(drawn.min()) >= 0
    assert int(drawn.max()) < n_obs
    assert np.array_equal(drawn, repeat)


@given(
    n_obs=st.integers(min_value=2, max_value=60),
    n_sims=st.integers(min_value=1, max_value=7),
    block=st.integers(min_value=1, max_value=80),
    seed=st.integers(min_value=0, max_value=10**6),
)
def test_block_indices_consecutive_within_blocks(
    n_obs: int, n_sims: int, block: int, seed: int
) -> None:
    """Inside a block, indices step by +1 modulo n_obs.

    The moving-block structure itself: every position that is not the
    last column of a block must be followed by its circular successor,
    in every simulated row.
    """
    effective_block = min(max(block, 1), n_obs)
    drawn = block_indices(n_obs, n_sims, block, np.random.default_rng(seed))
    for position in range(n_obs - 1):
        if (position + 1) % effective_block == 0:
            continue
        expected = (drawn[:, position] + 1) % n_obs
        assert np.array_equal(drawn[:, position + 1], expected)


@given(values=_SPREAD_ARRAYS)
def test_sharpe_scale_invariance_positive_factor(
    values: np.ndarray,
) -> None:
    """SR(x * 3.7) == SR(x) when rf == 0, to 1e-9 relative.

    Scaling all returns by a positive factor scales the mean and the
    standard deviation identically, so the ratio (and its block-
    bootstrap CI, rebuilt on the same seed) must not move.  Measured
    worst-case drift over random inputs is ~1e-14; 1e-9 leaves six
    orders of margin.

    Values are drawn pairwise-distinct: series with long runs of one
    repeated value break the CI filter ``sds > 0`` inside
    ``sharpe_ratio`` itself (see the strict-xfail companion test);
    the point estimate stays invariant even there.
    """
    assume(float(np.std(values, ddof=1)) > 1e-6)
    base = sharpe_ratio(values, seed=9)
    scaled = sharpe_ratio(values * 3.7, seed=9)
    assert math.isclose(base["sr"], scaled["sr"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(
        base["sr_ci"][0], scaled["sr_ci"][0], rel_tol=1e-9, abs_tol=1e-12
    )
    assert math.isclose(
        base["sr_ci"][1], scaled["sr_ci"][1], rel_tol=1e-9, abs_tol=1e-12
    )


def test_sharpe_ci_scale_invariance_on_degenerate_input() -> None:
    """CI scale invariance holds even on two-point degenerate series.

    For values [0, 1, 1, ..., 1] a resampled row made of one repeated
    large value can carry a rounding-dust standard deviation (~9e-16)
    instead of an exact 0.  The ``sane_variance_mask`` guard in
    ``_utils`` (sd must exceed 1e-12 * |mean|) drops such rows at any
    scale, so both the point SR and the bootstrap CI are invariant
    under x -> 3.7*x.  Found by this test as an xfail pin before the
    fix; the pin flipped to a passing test when
    ``_utils.sane_variance_mask`` landed.
    """
    values = np.array([0.0] + [1.0] * 29)
    base = sharpe_ratio(values, seed=9)
    scaled = sharpe_ratio(values * 3.7, seed=9)
    assert base["sr"] == scaled["sr"]
    # scaling legitimately moves the bootstrap percentiles by a
    # final-ulp; only the absurd 1e16-poisoned bound is a bug
    assert scaled["sr_ci"][0] == pytest.approx(base["sr_ci"][0], rel=1e-12)
    assert scaled["sr_ci"][1] == pytest.approx(base["sr_ci"][1], rel=1e-12)
    assert scaled["sr_ci"][1] < 100.0  # was ~6.5e16 before the fix


def test_sharpe_scale_invariance_broken_by_rf_documented() -> None:
    """With rf > 0 scale invariance is broken BY DESIGN, not a bug.

    ``excess = x - rf/periods`` is not equivariant under x -> 3.7*x:
    the risk-free drag becomes proportionally smaller, so the scaled
    stream earns a HIGHER Sharpe.  This test pins that documented
    behaviour so a future "fix" cannot silently change rf semantics;
    the strict invariance above holds only for rf == 0.
    """
    values = np.random.default_rng(1).normal(0.002, 0.01, 500)
    base = sharpe_ratio(values, rf=0.02, seed=3)["sr"]
    scaled = sharpe_ratio(values * 3.7, rf=0.02, seed=3)["sr"]
    assert math.isfinite(base)
    assert scaled - base > 0.05


@given(values=_RETURN_ARRAYS)
def test_sharpe_sign_flip_antisymmetry(values: np.ndarray) -> None:
    """SR(-x) == -SR(x): negating returns negates the Sharpe ratio.

    The negated stream flips the sign of every excess return but not
    its dispersion, so the point estimate, the per-period value and the
    CI (with mirrored bounds) must all negate.  In IEEE arithmetic the
    current implementation is bitwise antisymmetric; 1e-12 relative
    keeps the check robust to summation-order changes.
    """
    assume(float(np.std(values, ddof=1)) > 1e-3)
    base = sharpe_ratio(values, seed=11)
    flip = sharpe_ratio(-values, seed=11)
    assert flip["sr"] == pytest.approx(-base["sr"], rel=1e-12, abs=1e-15)
    assert flip["sr_per_period"] == pytest.approx(
        -base["sr_per_period"], rel=1e-12, abs=1e-15
    )
    assert flip["sr_ci"][0] == pytest.approx(
        -base["sr_ci"][1], rel=1e-12, abs=1e-15
    )
    assert flip["sr_ci"][1] == pytest.approx(
        -base["sr_ci"][0], rel=1e-12, abs=1e-15
    )


@given(values=_POSITION_ARRAYS)
def test_turnover_invariant_under_reversal_negation_offset(
    values: np.ndarray,
) -> None:
    """Reversal, negation and constant offset leave turnover unchanged.

    Turnover depends only on the multiset of absolute position steps:
    reversing the series reverses the diff sequence, negating negates
    each diff, and shifting by a constant cancels inside every diff.
    """
    base = turnover(values)["turnover_per_period"]
    assume(base > 1e-9)
    for transformed in (
        turnover(values[::-1])["turnover_per_period"],
        turnover(-values)["turnover_per_period"],
        turnover(values + 2.5)["turnover_per_period"],
    ):
        assert transformed == pytest.approx(base, rel=1e-9, abs=1e-12)


def test_bootstrap_ci_contains_point_estimate() -> None:
    """The 95% bootstrap CI must bracket the statistic it estimates.

    For the mean statistic the bootstrap distribution is centred on the
    sample mean, so the central 95% interval contains the point
    estimate for both bootstrap variants, on symmetric and skewed
    (shifted exponential) series alike.
    """
    for seed in range(10):
        symmetric = np.random.default_rng(100 + seed).normal(0.001, 0.01, 200)
        skewed = (
            np.random.default_rng(200 + seed).exponential(0.01, 200) - 0.01
        )
        for series in (symmetric, skewed):
            results = (
                block_bootstrap_ci(
                    series, n_boot=1000, block=10, stat=np.mean, seed=seed
                ),
                stationary_bootstrap_ci(
                    series,
                    n_boot=1000,
                    mean_block=8,
                    stat=np.mean,
                    seed=seed,
                ),
            )
            for outcome in results:
                assert (
                    outcome["ci_low"]
                    <= outcome["point_estimate"]
                    <= outcome["ci_high"]
                )


def test_bootstrap_ci_stable_across_seeds() -> None:
    """Large n_boot: another seed moves the CI by under 5% relative.

    With 4000 replicates the 2.5%/97.5% quantile estimates are Monte
    Carlo stable; a seed-to-seed jump above 5% would mean the CI is
    noise, not a confidence statement (measured drift: ~0.8%).
    """
    data = np.random.default_rng(77).normal(0.005, 0.01, 800)
    first = block_bootstrap_ci(data, n_boot=4000, block=20, seed=1)
    second = block_bootstrap_ci(data, n_boot=4000, block=20, seed=2)
    assert abs(first["ci_low"] - second["ci_low"]) < 0.05 * abs(
        first["ci_low"]
    )
    assert abs(first["ci_high"] - second["ci_high"]) < 0.05 * abs(
        first["ci_high"]
    )


@given(values=_COMPOUND_ARRAYS)
def test_max_drawdown_stays_in_unit_range(values: np.ndarray) -> None:
    """Max drawdown lies in [-1, 0] for curves and their reversal.

    Returns bounded in (-0.5, 0.5) keep the compounded equity strictly
    positive, so the drawdown is a well-defined fraction in [-1, 0]
    both forward and time-reversed.
    """
    equity = np.cumprod(1.0 + values)
    assert -1.0 <= max_drawdown(equity)["mdd"] <= 0.0
    assert -1.0 <= max_drawdown(equity[::-1])["mdd"] <= 0.0


def test_max_drawdown_time_asymmetry_documented() -> None:
    """Time reversal does NOT preserve mdd for multiplicative equity.

    Compounding geometry is path-ordered: [1, 3, 2, 4] dips 1/3 below
    its running peak, while the reversed path [4, 2, 3, 1] ends 75%
    below a peak set at the start.  Only the range invariants above
    hold under reversal; the asymmetry itself is pinned here as the
    documented behaviour (a known-answer check, not an invariance).
    """
    curve = np.asarray([1.0, 3.0, 2.0, 4.0])
    assert max_drawdown(curve)["mdd"] == pytest.approx(2.0 / 3.0 - 1.0)
    assert max_drawdown(curve[::-1])["mdd"] == pytest.approx(-0.75)


# ---------------------------------------------------------------------------
# property-fuzz: financial invariants over arbitrary machine-representable
# inputs (derandomized hypothesis profile from tests/conftest.py)
# ---------------------------------------------------------------------------
_ANY_FINITE_FLOAT = st.floats(allow_nan=False, allow_infinity=False)

# Below ~1.6e-7 partial R2 the required N passes power_for_effect's 5e7
# design cap and raises BY DESIGN; 5e-4 stays three orders away from it.
_EFFECT_FLOOR = 0.0005


@given(
    values=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=30, max_value=200),
        elements=_ANY_FINITE_FLOAT,
    )
)
def test_sharpe_ratio_fuzz_finite_or_nan_and_ordered_ci(
    values: np.ndarray,
) -> None:
    """Sharpe is finite or NaN (never inf); CI ordered or both-NaN.

    On arbitrary finite returns ``sr`` is finite or NaN, the CI is
    ordered or both-NaN, and ``n`` counts the finite inputs exactly.
    Arbitrary finite float64 returns, full exponent range included:
    whenever the mean overflows to inf the standard deviation
    overflows with it (inf/inf = NaN), so an infinite Sharpe is
    structurally unreachable; a constant series yields NaN on both
    CI bounds; and ``n`` must count exactly the finite inputs.
    """
    out = sharpe_ratio(values, seed=13)
    sr_value = float(out["sr"])
    assert not math.isinf(sr_value)
    low, high = (float(out["sr_ci"][0]), float(out["sr_ci"][1]))
    if math.isnan(low) or math.isnan(high):
        assert math.isnan(low) and math.isnan(high)
    else:
        assert not math.isinf(low)
        assert not math.isinf(high)
        assert low <= high
    assert out["n"] == int(np.isfinite(values).sum())


@given(
    values=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=200),
        elements=_ANY_FINITE_FLOAT,
    )
)
def test_turnover_series_fuzz_nonnegative_with_pinned_head(
    values: np.ndarray,
) -> None:
    """Turnover is >= 0 everywhere, tvr[0] == 0.0, == manual |diff|.

    One-sided absolute steps cannot go negative by construction, and
    the first period has no previous position to rebalance from, so
    its turnover is pinned to 0 even for a non-zero opening position.
    Steps between opposite-sign 1e308-magnitude positions may overflow
    to +inf, which is still a valid non-negative turnover reading --
    finiteness is NOT the invariant here.  The manual recompute
    concatenates [0.0] with |diff| and must match bitwise (both sides
    clean the same finite inputs).
    """
    tvr = turnover_series(values)
    expected = np.concatenate([[0.0], np.abs(np.diff(values))])
    assert tvr.shape == values.shape
    assert float(tvr[0]) == 0.0
    assert bool((tvr >= 0.0).all())
    assert np.array_equal(tvr, expected)


@given(
    effect_pair=st.tuples(
        st.floats(min_value=_EFFECT_FLOOR, max_value=0.099),
        st.floats(min_value=_EFFECT_FLOOR, max_value=0.099),
    ).map(sorted)
)
def test_power_for_effect_fuzz_monotone_in_effect(
    effect_pair: list[float],
) -> None:
    """Weaker effects need weakly more observations: N(e1) >= N(e2).

    The noncentral-t power at fixed (n, alpha) is monotone increasing
    in partial R2, so the minimum N returned by the doubling +
    bisection search must be monotone decreasing in the effect size
    for every random pair e1 < e2 (both in (5e-4, 0.099); the floor
    keeps the search far below the deliberate 5e7 design cap).
    """
    weak, strong = effect_pair
    assume(weak < strong)
    n_weak = power_for_effect(weak, n_predictors=5)
    n_strong = power_for_effect(strong, n_predictors=5)
    assert n_weak >= n_strong


@given(
    n_pair=st.tuples(
        st.integers(min_value=200, max_value=20000),
        st.integers(min_value=200, max_value=20000),
    ).map(sorted)
)
def test_detectable_effect_fuzz_monotone_decreasing_in_n(
    n_pair: list[int],
) -> None:
    """More observations must lower the detectable effect threshold.

    Power at fixed partial R2 is monotone increasing in N, so the
    minimal detectable effect must be monotone NON-INCREASING in N
    for every random pair N1 < N2 drawn from [200, 20000].  Found by
    this test as an xfail pin: scipy's noncentral-t returns NaN over
    patches of the (N, effect) plane and the bisection inflated the
    threshold up to 138x; ``_power_at`` now falls back to the normal
    limit there and the invariant holds.
    """
    small_n, big_n = n_pair
    assume(small_n < big_n)
    assert detectable_effect(small_n, n_predictors=5) >= detectable_effect(
        big_n, n_predictors=5
    )
