"""Statistical power calculator for incremental-signal tests.

Answers the two questions every hypothesis must face BEFORE evaluation:
  1. Given a target partial R-squared, how many observations are needed?
  2. Given the available N, what is the smallest partial R-squared detectable?

Model: testing ONE new coefficient in a linear regression with ``n_predictors``
other regressors.  Power is computed from the noncentral t distribution with
noncentrality ``ncp = sqrt(f2 * n)``, ``f2 = R2 / (1 - R2)`` (Cohen, 1988).

Design rule of thumb: at N = 2000 a partial R-squared of about 0.002 yields
|t| ~ 2; R-squared of 0.003 yields |t| ~ 2.4 (see :func:`check_design_rule`,
which validates this both analytically and by simulation).

Usage::

    from power_calculator import power_for_effect, detectable_effect

    n_needed = power_for_effect(partial_r2=0.003, n_predictors=25)   # ~2600
    min_r2   = detectable_effect(n=2000, n_predictors=25)            # ~0.0023

Run ``python power_calculator.py`` for a synthetic demo.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

__all__ = [
    "power_for_effect",
    "detectable_effect",
    "t_stat_for_partial_r2",
    "simulate_power",
    "check_design_rule",
]

_MAX_N = 50_000_000


def _power_at(partial_r2: float, n: int, n_predictors: int, alpha: float) -> float:
    """Two-sided power of the single-coefficient t-test at sample size ``n``."""
    df = int(n) - int(n_predictors) - 1
    if df < 1:
        return 0.0
    f2 = partial_r2 / (1.0 - partial_r2)
    ncp = float(np.sqrt(f2 * n))
    t_crit = float(_st.t.ppf(1.0 - alpha / 2.0, df))
    return float(_st.nct.sf(t_crit, df, ncp) + _st.nct.cdf(-t_crit, df, ncp))


def _validate(partial_r2: float, n_predictors: int, alpha: float, power: float) -> None:
    if not (0.0 < partial_r2 < 1.0):
        raise ValueError(f"partial_r2 must be in (0, 1), got {partial_r2}")
    if int(n_predictors) < 1:
        raise ValueError(f"n_predictors must be >= 1, got {n_predictors}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")


def power_for_effect(
    partial_r2: float, n_predictors: int, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Minimum number of observations to detect ``partial_r2`` with target power.

    Example::

        >>> n = power_for_effect(0.003, n_predictors=25)   # doctest: +SKIP
        >>> 2000 < n < 3500                                # doctest: +SKIP
        True
    """
    _validate(partial_r2, n_predictors, alpha, power)
    k = int(n_predictors)
    lo = k + 2
    if _power_at(partial_r2, lo, k, alpha) >= power:
        return lo
    hi = lo * 2
    while _power_at(partial_r2, hi, k, alpha) < power:
        lo = hi
        hi *= 2
        if hi > _MAX_N:
            raise ValueError(
                "required N exceeds 5e7: effect is too small to be detectable"
            )
    for _ in range(60):
        mid = (lo + hi) // 2
        if _power_at(partial_r2, mid, k, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return int(hi)


def detectable_effect(
    n: int, n_predictors: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest partial R-squared detectable with ``n`` observations at target power.

    Example::

        >>> r2 = detectable_effect(2000, n_predictors=25)  # doctest: +SKIP
        >>> 0.001 < r2 < 0.005                             # doctest: +SKIP
        True
    """
    n = int(n)
    k = int(n_predictors)
    if n <= k + 1:
        raise ValueError(f"n must exceed n_predictors + 1 (got n={n}, k={k})")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    lo, hi = 1e-9, 0.9999
    if _power_at(lo, n, k, alpha) >= power:
        return lo
    if _power_at(hi, n, k, alpha) < power:
        return float("nan")  # unreachable at any effect size
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))  # geometric bisection (monotone in r2)
        if _power_at(mid, n, k, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def t_stat_for_partial_r2(partial_r2: float, n: int, n_predictors: int) -> float:
    """Analytic |t| of a coefficient implied by its partial R-squared.

    ``|t| = sqrt(R2 * df / (1 - R2))`` with ``df = n - n_predictors - 1``.

    Example::

        >>> t_stat_for_partial_r2(0.003, n=2000, n_predictors=25)  # doctest: +SKIP
        2.43...
    """
    df = int(n) - int(n_predictors) - 1
    if df <= 0:
        return float("nan")
    r2 = min(max(float(partial_r2), 0.0), 0.999999999)
    return float(np.sqrt(r2 * df / (1.0 - r2)))


def simulate_power(
    partial_r2: float,
    n: int,
    n_predictors: int,
    alpha: float = 0.05,
    n_sims: int = 500,
    seed: int | None = None,
) -> float:
    """Empirical rejection frequency of the t-test via Monte Carlo.

    Regressions are simulated with orthonormal N(0,1) regressors so that the
    planted coefficient implies exactly ``partial_r2``.  Useful to validate the
    analytic power formula.  Keep ``n_sims`` moderate (500-2000).

    Example::

        >>> p = simulate_power(0.003, n=2000, n_predictors=25, n_sims=400)  # doctest: +SKIP
        >>> 0.5 < p < 0.9                                                   # doctest: +SKIP
        True
    """
    _validate(partial_r2, n_predictors, alpha, 0.5)
    n, k = int(n), int(n_predictors)
    df = n - k - 1
    if df <= 0:
        raise ValueError("n must exceed n_predictors + 1")
    rng = np.random.default_rng(seed)
    t_crit = float(_st.t.ppf(1.0 - alpha / 2.0, df))
    beta_true = float(np.sqrt(partial_r2 / (1.0 - partial_r2)))
    ones = np.ones(n)
    rejects = 0
    for _ in range(int(n_sims)):
        x_new = rng.standard_normal(n)
        X_other = rng.standard_normal((n, k))
        y = beta_true * x_new + rng.standard_normal(n)
        X = np.column_stack([ones, x_new, X_other])
        xtx_inv = np.linalg.inv(X.T @ X)
        coef = xtx_inv @ (X.T @ y)
        resid = y - X @ coef
        s2 = float(resid @ resid) / df
        se = float(np.sqrt(s2 * xtx_inv[1, 1]))
        if abs(coef[1] / se) > t_crit:
            rejects += 1
    return rejects / float(n_sims)


def check_design_rule(
    n: int = 2000,
    partial_r2: float = 0.003,
    n_predictors: int = 25,
    alpha: float = 0.05,
    n_sims: int = 400,
) -> dict:
    """Validate the working rule "N=2000 sees t~2 at partial R2~0.003".

    Returns analytic |t|, the exact partial R2 needed for |t| = 2 at this N,
    and the empirical power from :func:`simulate_power`.

    Example::

        >>> out = check_design_rule()                       # doctest: +SKIP
        >>> out["t_analytic"] > 2                           # doctest: +SKIP
        True
    """
    n, k = int(n), int(n_predictors)
    df = n - k - 1
    t_analytic = t_stat_for_partial_r2(partial_r2, n, k)
    r2_for_t2 = 4.0 / (4.0 + df)  # solve t^2 = R2*df/(1-R2) at t = 2
    emp = simulate_power(partial_r2, n, k, alpha=alpha, n_sims=n_sims, seed=0)
    return {
        "n": n,
        "n_predictors": k,
        "partial_r2": float(partial_r2),
        "t_analytic": t_analytic,
        "r2_needed_for_t_equals_2": float(r2_for_t2),
        "empirical_power": emp,
        "alpha": float(alpha),
        "note": (
            f"At N={n} and partial R2={partial_r2:g}, |t| ~ {t_analytic:.2f} "
            f"(rule of thumb says ~2; exact |t|=2 needs R2 ~ {r2_for_t2:.4f}); "
            f"empirical power at this effect size: {emp:.2f}"
        ),
    }


def run_all_examples() -> None:
    """Demo every calculator on the canonical N=2000 / R2=0.003 design."""
    print("=" * 72)
    print("power_calculator demo")
    print("=" * 72)

    n_needed = power_for_effect(0.003, n_predictors=25)
    print(f"power_for_effect(R2=0.003, k=25, power=0.80)  -> N required = {n_needed}")

    min_r2 = detectable_effect(2000, n_predictors=25)
    print(f"detectable_effect(N=2000, k=25, power=0.80)   -> min partial R2 = {min_r2:.5f}")

    t_val = t_stat_for_partial_r2(0.003, n=2000, n_predictors=25)
    print(f"t_stat_for_partial_r2(0.003, N=2000, k=25)    -> |t| = {t_val:.3f}")

    rule = check_design_rule(n_sims=300)
    print("\ncheck_design_rule():")
    for key, val in rule.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    run_all_examples()
