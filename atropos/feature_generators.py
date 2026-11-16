"""Local numpy/scipy feature generators for the ml_predictive family.

This module is the GENERATOR tier of the three-tier falsification
toolkit (backtest / feature / paper + routing): it turns raw price,
return and volume columns into the statistical feature series that
:func:`atropos.feature_scan.feature_analysis` and
:mod:`atropos.orthogonal` ``orthogonal_check`` then judge for the
``ml_predictive`` family.  Every generator is LOCAL (numpy/scipy
arithmetic only -- no arch, statsmodels or hmmlearn) and
DETERMINISTIC: explicit seeds wherever randomness enters, no global
state, no data fetching.

The public generators:

* :func:`kalman_ou_spread` -- time-varying-beta Kalman filter of y
  on x (the classic pairs-trading state-space filter) plus OU
  statistics of the residual spread (rolling z-score and AR(1)
  half-life).
* :func:`garch_sigma_forecast` -- GARCH(1,1) one-step-ahead
  conditional sigma, fit by Gaussian MLE over all bars.
* :func:`hmm_regime` -- compact Gaussian hidden-Markov regime labels
  (pure-numpy Baum-Welch), optionally with log-volume as a second
  emission dimension.

PREREG pointer: ``PREREG_ML_PREDICTIVE_2026-11-16`` froze the
expectations BEFORE any generator ran -- garch features
EXPECTED-ORTHOGONAL to the controls, kalman spread a SUSPECTED PROXY
(it is mechanically built from the same two price legs as the
return), hmm regime question deliberately OPEN.  The expectations
bind the downstream analysis, never this code: nothing here reads
them.

Run ``python -m atropos.feature_generators`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from atropos._utils import ols_hc0, validate_int, validate_number

__all__ = [
    "GarchForecastReport",
    "HmmRegimeReport",
    "KalmanOUSpreadReport",
    "garch_sigma_forecast",
    "hmm_regime",
    "kalman_ou_spread",
]


# ---------------------------------------------------------------------------
# pinned constants (every number named once, documented where declared)
# ---------------------------------------------------------------------------
_KALMAN_PRIOR_BETA = 1.0
"""Prior slope mean at the first usable bar (the neutral hedge)."""

_KALMAN_PRIOR_VAR = 1e6
"""Prior slope variance at the first usable bar (diffuse start)."""

_MIN_OU_PAIRS = 3
"""Finite (spread_t-1, spread_t) pairs below this read a NaN fit."""

_GARCH_MIN_OBS = 100
"""Minimum finite returns before a GARCH MLE is attempted."""

_GARCH_PERSISTENCE_CAP = 0.999
"""Ceiling on alpha + beta (variance-stationarity guard in the MLE)."""

_GARCH_FALLBACK_ALPHA = 0.07
"""Fallback alpha when the optimizer fails (documented constant)."""

_GARCH_FALLBACK_BETA = 0.92
"""Fallback beta when the optimizer fails (persistence 0.99)."""

_GARCH_START_ALPHA = 0.05
"""Alpha share of the sample-moment persistence at the optimizer start."""

_HMM_MIN_OBS = 50
"""Minimum usable bars before a Baum-Welch fit is attempted."""

_HMM_VAR_FLOOR = 1e-12
"""Emission-variance floor (a state may not collapse to zero width)."""

_HMM_TRANSITION_FLOOR = 1e-12
"""Transition floor so no state becomes strictly unreachable."""

_HMM_TRANSITION_BOOST = 0.1
"""Diagonal boost on the uniform transition prior (sticky regimes)."""

_HMM_INIT_JITTER = 1e-8
"""Seeded jitter on the quantile init means (exact-tie breaker)."""

_HMM_RHO_FALLBACK = 0.90
"""Lag-1 autocorrelation stand-in when squared returns are degenerate."""


# ---------------------------------------------------------------------------
# frozen report schemas (every key present on every outcome)
# ---------------------------------------------------------------------------
class KalmanOUSpreadReport(TypedDict):
    """Kalman-filtered hedge ratio plus OU statistics of the spread.

    ``kalman_beta`` / ``kalman_beta_var`` are the per-bar filtered
    slope and its filtered variance (the recursion's uncertainty
    path); ``spread_z`` is the rolling z-score of the spread; each
    array is aligned with the input bars and NaN on non-finite input
    bars.  ``half_life`` is the FULL-SAMPLE scalar OU estimate -- one
    float for the whole series, not a per-bar path.
    """

    kalman_beta: np.ndarray
    kalman_beta_var: np.ndarray
    spread_z: np.ndarray
    half_life: float


class GarchForecastReport(TypedDict):
    """One-step-ahead conditional sigma path plus the fitted params.

    ``garch_sigma_forecast`` is aligned with the input bars (NaN on
    non-finite input bars); index ``t`` carries the conditional sigma
    FOR bar ``t`` computed from data up to ``t - 1`` (index 0 carries
    the ``sqrt(v_0)`` prior).  ``converged`` is False exactly when
    the documented fallback parameters were used.
    """

    garch_sigma_forecast: np.ndarray
    omega: float
    alpha: float
    beta: float
    converged: bool


class HmmRegimeReport(TypedDict):
    """Regime labels plus the fitted Markov chain's last-bar outlook.

    ``hmm_regime`` holds int labels 0..n_states-1 per bar (posterior
    decoding; -1 on masked bars) under the canonical ASCENDING-mean
    ordering: state 0 is the lowest/negative-mean regime, state
    ``n_states - 1`` the highest.  ``hmm_transition_prob`` is the
    smoothed next-bar state distribution after the LAST observed bar
    (``gamma_last @ transition``, sums to 1).
    """

    hmm_regime: np.ndarray
    hmm_transition_prob: np.ndarray
    converged: bool
    loglik: float


# ---------------------------------------------------------------------------
# shared input coercion
# ---------------------------------------------------------------------------
def _validate_positive(name: str, value: float) -> float:
    """Raise ValueError unless ``value`` is a finite number > 0.

    Non-numeric input is re-raised as ``ValueError`` naming the
    parameter, per the shared ``validate_number`` contract.
    """
    number = validate_number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def _to_1d_float(values: Any, name: str) -> np.ndarray:
    """Coerce to a 1-D float ndarray with NaNs preserved in place.

    Unlike :func:`atropos._utils.clean_1d` (which DROPS non-finite
    entries) this keeps every position: the generators must mask, not
    drop, so their outputs stay aligned with the input bars.
    """
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={raw.ndim}")
    return raw


def _validated_seed(seed: int | None) -> None:
    """Validate an optional seed parameter; the value is not consumed.

    The kalman and garch fits are deterministic (no randomness
    enters), so ``seed`` is accepted purely for call-site uniformity
    across the generator family: given a value it must still be a
    non-negative integer, but it never affects any output.
    """
    if seed is not None:
        validate_int("seed", seed, minimum=0)


# ---------------------------------------------------------------------------
# 1. kalman_ou_spread
# ---------------------------------------------------------------------------
def _kalman_slope_filter(
    price_y: np.ndarray,
    price_x: np.ndarray,
    *,
    delta: float,
    r_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the local LevelModel recursion; return (beta, var) paths.

    State: ``beta_t`` is a random walk with variance ``delta``.
    Observation: ``y_t = beta_t * x_t + eps`` with variance ``r_var``.
    Per usable bar: predict (``var + delta``), standard Kalman gain
    update, then store the filtered slope and variance.  Bars where
    either input is non-finite are skipped entirely -- the state
    carries forward from the last usable bar and the outputs stay
    NaN there.
    """
    n_obs = price_y.size
    kalman_beta = np.full(n_obs, np.nan)
    kalman_beta_var = np.full(n_obs, np.nan)
    usable = np.isfinite(price_y) & np.isfinite(price_x)
    beta_state = _KALMAN_PRIOR_BETA
    var_state = _KALMAN_PRIOR_VAR
    for step in range(n_obs):
        if not usable[step]:
            continue
        var_predicted = var_state + delta
        obs_x = float(price_x[step])
        innovation_var = var_predicted * obs_x * obs_x + r_var
        gain = var_predicted * obs_x / innovation_var
        innovation = float(price_y[step]) - beta_state * obs_x
        beta_state = beta_state + gain * innovation
        var_state = (1.0 - gain * obs_x) * var_predicted
        kalman_beta[step] = beta_state
        kalman_beta_var[step] = var_state
    return kalman_beta, kalman_beta_var


def _rolling_zscore(spread: np.ndarray, z_window: int) -> np.ndarray:
    """Rolling z-score under pandas min_periods=z_window semantics.

    Position ``t`` is scored against the mean and ddof=1 sample
    standard deviation of the ``z_window`` bars ending at ``t`` (the
    window includes ``t`` itself); the value stays NaN until the
    window holds ``z_window`` FINITE bars, and a zero window std
    reads NaN (a flat window has no z-scale).
    """
    n_obs = spread.size
    z_values = np.full(n_obs, np.nan)
    if n_obs < z_window:
        return z_values
    windows = np.lib.stride_tricks.sliding_window_view(spread, z_window)
    finite = np.isfinite(windows)
    complete = finite.all(axis=1)
    filled = np.where(finite, windows, 0.0)
    means = filled.sum(axis=1) / z_window
    deviations = np.where(finite, windows - means[:, None], 0.0)
    stds = np.sqrt(np.sum(deviations**2, axis=1) / (z_window - 1))
    scored = complete & np.isfinite(stds) & (stds > 0.0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    tail_z = (spread[z_window - 1 :] - means) / safe_stds
    z_values[z_window - 1 :] = np.where(scored, tail_z, np.nan)
    return z_values


def _ou_half_life(spread: np.ndarray) -> float:
    """AR(1) half-life of the spread; NaN when it does not revert.

    OLS of ``spread_t`` on an intercept plus ``spread_{t-1}`` over
    the finite pairs; the intercept absorbs the spread mean, so the
    slope ``phi`` IS the de-meaned AR(1) coefficient.
    ``half_life = -ln(2) / ln(phi)`` for ``0 < phi < 1`` (positive,
    unbounded above as ``phi`` approaches 1); ``phi <= 0`` reads NaN
    (no mean reversion) and ``phi >= 1`` reads NaN (non-stationary
    spread -- no finite half-life exists to clip to).
    """
    finite = np.isfinite(spread)
    pair_mask = finite[:-1] & finite[1:]
    lagged = spread[:-1][pair_mask]
    current = spread[1:][pair_mask]
    if lagged.size < _MIN_OU_PAIRS:
        return float("nan")
    design = np.column_stack([np.ones(lagged.size), lagged])
    phi = float(ols_hc0(current, design)["beta"][1])
    if not np.isfinite(phi) or not 0.0 < phi < 1.0:
        return float("nan")
    return float(-np.log(2.0) / np.log(phi))


def kalman_ou_spread(
    price_y: Any,
    price_x: Any,
    *,
    delta: float = 1e-5,
    r_var: float = 1e-3,
    z_window: int = 60,
    seed: int | None = None,
) -> KalmanOUSpreadReport:
    """Time-varying-beta Kalman filter of y on x, then OU spread stats.

    The classic pairs-trading state-space filter, in local numpy: the
    hedge ratio ``beta_t`` follows a random walk (transition variance
    ``delta``) and the observation is ``y_t = beta_t * x_t + eps``
    (variance ``r_var``), updated with the standard Kalman gain per
    bar.  Initialization: prior ``beta = 1.0`` with a diffuse prior
    variance (:data:`_KALMAN_PRIOR_VAR`), so the first usable bars
    set the ratio rather than inherit it.  The spread is the
    innovation ``y_t - beta_t * x_t`` with the FILTERED (same-bar)
    beta; its rolling z-score and its full-sample AR(1) half-life
    complete the OU reading.

    Args:
        price_y: 1-D dependent price series (positionally aligned).
        price_x: 1-D independent price series, same length as
            ``price_y``.
        delta: random-walk variance of the state (transition noise).
        r_var: observation noise variance of the y-on-x regression.
        z_window: rolling z-score window; a z value requires
            ``z_window`` finite spread bars in its window
            (min_periods=z_window semantics, NaN until full).
        seed: accepted for call-site uniformity across the generator
            family; the filter is deterministic, so the value is
            validated but never affects the output.

    Returns:
        :class:`KalmanOUSpreadReport`: ``kalman_beta`` /
        ``kalman_beta_var`` / ``spread_z`` arrays aligned with the
        inputs (NaN on non-finite input bars, where the filter skips
        and carries its state) and the scalar full-sample
        ``half_life``.

    Raises:
        ValueError: if ``delta`` / ``r_var`` is not a finite positive
            number, ``z_window`` or ``seed`` is not a valid integer,
            an input is not 1-D, or the lengths differ.

    Example::

        >>> rng = np.random.default_rng(11)
        >>> price_x = np.exp(0.03 * np.cumsum(rng.standard_normal(600)))
        >>> ou = np.zeros(600)
        >>> shocks = rng.standard_normal(600)
        >>> for step in range(1, 600):
        ...     ou[step] = 0.9 * ou[step - 1] + 0.05 * shocks[step]
        >>> price_y = 2.0 * price_x + ou
        >>> report = kalman_ou_spread(price_y, price_x)
        >>> abs(float(np.nanmean(report["kalman_beta"][-100:])) - 2.0) < 0.05
        True
        >>> bool(np.isfinite(report["spread_z"][59:]).all())
        True
        >>> 0.0 < report["half_life"] < 20.0
        True

        Non-finite input bars are masked, not dropped -- the output
        keeps the input alignment and reads NaN there:

        >>> gappy = price_y.copy()
        >>> gappy[0] = np.nan
        >>> float(kalman_ou_spread(gappy, price_x)["kalman_beta"][0])
        nan
    """
    checked_delta = _validate_positive("delta", delta)
    checked_r_var = _validate_positive("r_var", r_var)
    checked_window = validate_int("z_window", z_window, minimum=2)
    _validated_seed(seed)
    values_y = _to_1d_float(price_y, "price_y")
    values_x = _to_1d_float(price_x, "price_x")
    if values_y.size != values_x.size:
        raise ValueError(
            "price_y and price_x must have equal length, got"
            f" {values_y.size} and {values_x.size}"
        )
    kalman_beta, kalman_beta_var = _kalman_slope_filter(
        values_y, values_x, delta=checked_delta, r_var=checked_r_var
    )
    spread = values_y - kalman_beta * values_x
    return KalmanOUSpreadReport(
        kalman_beta=kalman_beta,
        kalman_beta_var=kalman_beta_var,
        spread_z=_rolling_zscore(spread, checked_window),
        half_life=_ou_half_life(spread),
    )


# ---------------------------------------------------------------------------
# 2. garch_sigma_forecast
# ---------------------------------------------------------------------------
def _garch_variance_path(
    omega: float,
    alpha: float,
    beta: float,
    rets_sq: np.ndarray,
    initial_var: float,
) -> np.ndarray:
    """Recursion v_t = omega + alpha*r^2_{t-1} + beta*v_{t-1}.

    ``v_0`` is the caller's fixed prior (not a function of the
    parameters), so ``omega`` / ``alpha`` / ``beta`` enter from bar 1
    onward.
    """
    n_obs = rets_sq.size
    cond_var = np.empty(n_obs)
    cond_var[0] = initial_var
    for step in range(1, n_obs):
        cond_var[step] = (
            omega + alpha * rets_sq[step - 1] + beta * cond_var[step - 1]
        )
    return cond_var


def _garch_nll(
    params: np.ndarray,
    rets_sq: np.ndarray,
    initial_var: float,
    penalty: float,
) -> float:
    """Gaussian neg-log-likelihood in (omega, persistence, share) coords.

    ``persistence`` carries ``alpha + beta`` and ``share`` the alpha
    fraction, so ``alpha = persistence * share`` and ``beta =
    persistence * (1 - share)``: the whole constrained region
    (``omega > 0``, ``alpha >= 0``, ``beta >= 0``,
    ``alpha + beta <= 0.999``) becomes pure box bounds for L-BFGS-B.
    A direct (omega, alpha, beta) penalty on the persistence
    constraint was tried and rejected: on step-function volatility
    data the MLE wants persistence at the cap, the line search steps
    onto the flat penalty and dies ABNORMAL exactly at the answer.
    ``penalty`` stands in only for a non-finite or non-positive
    variance path.
    """
    omega = float(params[0])
    persistence = float(params[1])
    alpha = persistence * float(params[2])
    beta = persistence - alpha
    cond_var = _garch_variance_path(omega, alpha, beta, rets_sq, initial_var)
    if not bool(np.all(np.isfinite(cond_var))) or bool(
        np.any(cond_var <= 0.0)
    ):
        return penalty
    return float(
        0.5
        * np.sum(np.log(2.0 * np.pi) + np.log(cond_var) + rets_sq / cond_var)
    )


def _lag1_autocorr(values: np.ndarray) -> float:
    """Lag-1 autocorrelation; the documented fallback when degenerate.

    Fewer than three values, a constant series, or a numerically
    undefined correlation return :data:`_HMM_RHO_FALLBACK` (a typical
    daily persistence) so the optimizer start stays deterministic.
    """
    if values.size < 3 or np.ptp(values) == 0.0:
        return _HMM_RHO_FALLBACK
    rho = float(np.corrcoef(values[1:], values[:-1])[0, 1])
    return rho if np.isfinite(rho) else _HMM_RHO_FALLBACK


def garch_sigma_forecast(
    returns: Any,
    *,
    holdout: int = 250,
    seed: int | None = None,
) -> GarchForecastReport:
    """GARCH(1,1) one-step-ahead conditional sigma via Gaussian MLE.

    Plain maximum likelihood (variance targeting OFF) on EVERY finite
    bar: the recursion ``v_t = omega + alpha*r^2_{t-1} + beta*v_{t-1}``
    with ``v_0`` fixed at the sample variance of the holdout-free
    prefix (the first ``n - holdout`` finite bars; the full sample
    when that prefix has fewer than two bars).  ``holdout`` therefore
    shapes only the ``v_0`` prior, never the likelihood.  Constraints
    ``omega > 0``, ``alpha >= 0``, ``beta >= 0`` and
    ``alpha + beta`` capped at 0.999 are enforced by optimizing in
    (omega, persistence, share) coordinates -- pure box bounds, no
    penalty region (see :func:`_garch_nll`); the start is
    deterministic from sample moments (lag-1 autocorrelation of
    squared returns sets the persistence, a fixed 0.05 share goes to
    ``alpha``, ``omega`` absorbs the remainder of the sample
    variance).

    Alignment: ``garch_sigma_forecast[t]`` is the conditional sigma
    FOR bar ``t`` computed from data up to ``t - 1``; index 0 carries
    ``sqrt(v_0)``.  On optimizer failure the fit falls back to the
    documented constants (alpha=0.07, beta=0.92, ``omega`` = 1% of
    the sample variance) and reports ``converged=False`` -- the
    forecast is still produced, never silently dropped.

    Args:
        returns: 1-D return series (non-finite bars are masked out of
            the fit; their output positions read NaN).
        holdout: size of the held-back tail that the ``v_0`` prefix
            variance excludes; at least 1.
        seed: accepted for call-site uniformity across the generator
            family; the fit is deterministic, so the value is
            validated but never affects the output.

    Returns:
        :class:`GarchForecastReport` with the sigma path aligned to
        the input bars and the fitted ``omega`` / ``alpha`` / ``beta``
        plus the ``converged`` flag.

    Raises:
        ValueError: if ``holdout`` or ``seed`` is not a valid
            integer, the input is not 1-D, fewer than 100 finite
            observations survive, or the finite returns have zero
            variance.

    Example::

        >>> rng = np.random.default_rng(11)
        >>> rets = np.concatenate(
        ...     [0.01 * rng.standard_normal(300),
        ...      0.05 * rng.standard_normal(300)]
        ... )
        >>> report = garch_sigma_forecast(rets)
        >>> report["converged"]
        True
        >>> low = float(np.mean(report["garch_sigma_forecast"][:300]))
        >>> high = float(np.mean(report["garch_sigma_forecast"][300:]))
        >>> high > 2.0 * low
        True
    """
    checked_holdout = validate_int("holdout", holdout, minimum=1)
    _validated_seed(seed)
    raw = _to_1d_float(returns, "returns")
    finite_positions = np.flatnonzero(np.isfinite(raw))
    rets = raw[finite_positions]
    if rets.size < _GARCH_MIN_OBS:
        raise ValueError(
            f"garch_sigma_forecast needs at least {_GARCH_MIN_OBS}"
            f" finite observations, got {rets.size}"
        )
    sample_var = float(np.var(rets, ddof=1))
    if not np.isfinite(sample_var) or sample_var <= 0.0:
        raise ValueError(
            "returns have zero variance; the GARCH likelihood is undefined"
        )
    prefix_size = max(rets.size - checked_holdout, 2)
    initial_var = float(np.var(rets[:prefix_size], ddof=1))
    rets_sq = rets**2
    persistence0 = float(np.clip(_lag1_autocorr(rets_sq), 0.10, 0.98))
    alpha_start = _GARCH_START_ALPHA
    beta_start = float(np.clip(persistence0 - _GARCH_START_ALPHA, 0.10, 0.94))
    share_start = alpha_start / (alpha_start + beta_start)
    omega_floor = max(1e-12 * sample_var, 1e-300)
    omega_start = max(
        sample_var * (1.0 - alpha_start - beta_start), 10.0 * omega_floor
    )
    start = np.array([omega_start, persistence0, share_start])
    penalty = 10.0 * (1.0 + abs(_garch_nll(start, rets_sq, initial_var, 1.0)))
    result = minimize(
        _garch_nll,
        start,
        args=(rets_sq, initial_var, penalty),
        method="L-BFGS-B",
        bounds=[
            (omega_floor, np.inf),
            (0.0, _GARCH_PERSISTENCE_CAP),
            (0.0, 1.0),
        ],
    )
    omega_hat = float(result.x[0])
    persistence_hat = float(result.x[1])
    alpha_hat = persistence_hat * float(result.x[2])
    beta_hat = persistence_hat - alpha_hat
    converged = bool(
        result.success
        and np.isfinite(omega_hat)
        and omega_hat > 0.0
        and alpha_hat >= 0.0
        and beta_hat >= 0.0
        and persistence_hat <= _GARCH_PERSISTENCE_CAP + 1e-9
    )
    if not converged:
        alpha_hat = _GARCH_FALLBACK_ALPHA
        beta_hat = _GARCH_FALLBACK_BETA
        omega_hat = max(
            sample_var * (1.0 - alpha_hat - beta_hat), 10.0 * omega_floor
        )
    cond_var = _garch_variance_path(
        omega_hat, alpha_hat, beta_hat, rets_sq, initial_var
    )
    forecast = np.full(raw.size, np.nan)
    forecast[finite_positions] = np.sqrt(cond_var)
    return GarchForecastReport(
        garch_sigma_forecast=forecast,
        omega=omega_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# 3. hmm_regime
# ---------------------------------------------------------------------------
def _hmm_log_emission(
    obs: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Log Gaussian emission densities (diag covariance), (bars, states)."""
    deviation = obs[:, None, :] - means[None, :, :]
    log_densities = -0.5 * (
        np.log(2.0 * np.pi)
        + np.log(variances)[None, :, :]
        + deviation**2 / variances[None, :, :]
    )
    return np.sum(log_densities, axis=2)


def _hmm_forward_backward(
    log_emission: np.ndarray,
    log_transition: np.ndarray,
    log_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Log-space forward-backward; returns (log_alpha, log_beta, gamma, ll).

    Numerical choice: the recursions run entirely in log space with
    ``scipy.special.logsumexp`` accumulators instead of the Rabiner
    per-bar scaling constants -- no underflow at any series length,
    and the log-likelihood comes out exactly (not as a sum of
    per-bar log scale factors).
    """
    n_time = log_emission.shape[0]
    log_alpha = np.empty_like(log_emission)
    log_alpha[0] = log_start + log_emission[0]
    for step in range(1, n_time):
        log_alpha[step] = (
            logsumexp(log_alpha[step - 1][:, None] + log_transition, axis=0)
            + log_emission[step]
        )
    loglik = float(logsumexp(log_alpha[-1]))
    log_beta = np.zeros_like(log_emission)
    for step in range(n_time - 2, -1, -1):
        log_beta[step] = logsumexp(
            log_transition
            + (log_emission[step + 1] + log_beta[step + 1])[None, :],
            axis=1,
        )
    gamma = np.exp(log_alpha + log_beta - loglik)
    gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    return log_alpha, log_beta, gamma, loglik


def _hmm_e_step(
    obs: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    transition: np.ndarray,
    start_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One E-step: smoothed posteriors, summed xi, log-likelihood."""
    log_emission = _hmm_log_emission(obs, means, variances)
    log_transition = np.log(transition)
    log_alpha, log_beta, gamma, loglik = _hmm_forward_backward(
        log_emission, log_transition, np.log(start_prob)
    )
    log_xi = (
        log_alpha[:-1, :, None]
        + log_transition[None, :, :]
        + log_emission[1:, None, :]
        + log_beta[1:, None, :]
        - loglik
    )
    return gamma, np.exp(log_xi).sum(axis=0), loglik


def _hmm_m_step(
    obs: np.ndarray,
    gamma: np.ndarray,
    xi_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One M-step: (means, variances, transition, start), all floored.

    Variances are floored at :data:`_HMM_VAR_FLOOR` and transition
    entries at :data:`_HMM_TRANSITION_FLOOR` (then renormalized), so
    no state collapses to zero width and none becomes strictly
    unreachable -- the log-space recursions stay finite.
    """
    state_weight = gamma.sum(axis=0)
    weight_safe = np.maximum(state_weight, 1e-12)
    means = np.einsum("ti,td->id", gamma, obs) / weight_safe[:, None]
    deviation = obs[:, None, :] - means[None, :, :]
    variances = np.einsum("ti,tid->id", gamma, deviation**2)
    variances = np.maximum(variances / weight_safe[:, None], _HMM_VAR_FLOOR)
    row_sums = np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
    transition = np.maximum(xi_sum / row_sums, _HMM_TRANSITION_FLOOR)
    transition = transition / transition.sum(axis=1, keepdims=True)
    start_prob = np.maximum(gamma[0], _HMM_TRANSITION_FLOOR)
    return means, variances, transition, start_prob / start_prob.sum()


def _hmm_init(
    obs: np.ndarray, n_states: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic quantile init plus the seeded tie-break jitter.

    Means start at the interior quantiles of the observed data per
    emission dimension (for three states: the -q, median, +q of the
    sample), variances at 1.0, transitions uniform with a 0.1
    diagonal boost, start uniform.  A seeded 1e-8 jitter on the means
    breaks exact ties when quantiles coincide (constant segments):
    with a fixed seed the whole fit stays deterministic, and two
    states can never start out identical.
    """
    levels = np.linspace(0.0, 1.0, n_states + 2)[1:-1]
    means = np.asarray(np.quantile(obs, levels, axis=0), dtype=float)
    means = means + rng.normal(0.0, _HMM_INIT_JITTER, size=means.shape)
    variances = np.ones_like(means)
    transition = np.full((n_states, n_states), 1.0 / n_states)
    transition = transition + np.eye(n_states) * _HMM_TRANSITION_BOOST
    transition = transition / transition.sum(axis=1, keepdims=True)
    start_prob = np.full(n_states, 1.0 / n_states)
    return means, variances, transition, start_prob


def hmm_regime(
    returns: Any,
    volume: Any = None,
    *,
    n_states: int = 3,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> HmmRegimeReport:
    """Compact Gaussian hidden-Markov regime labels (pure numpy EM).

    Baum-Welch over ``n_states`` Gaussian states with diagonal
    covariance; when ``volume`` is given, log-volume joins the
    returns as a second emission dimension.  Initialization is
    deterministic given ``seed``: state means at the interior
    quantiles of the observed data (a seeded 1e-8 jitter breaks
    exact ties), unit variances, uniform transitions with a 0.1
    diagonal boost, uniform start.  EM stops when the
    log-likelihood improves by less than ``tol`` (``converged``) or
    at the ``n_iter`` cap.

    Numerical choice: the forward-backward recursions run in log
    space via ``scipy.special.logsumexp`` (no per-bar scaling
    constants, no underflow; see :func:`_hmm_forward_backward`).
    Labels are posterior-decoded (argmax of the smoothed state
    probability per bar) and states are RELABELED by ascending
    emission mean of the returns dimension, so state 0 is always the
    low/negative regime and state ``n_states - 1`` the high/positive
    one -- a deterministic labeling independent of EM's internal
    state order.  ``hmm_transition_prob`` is the smoothed next-bar
    state distribution after the last observed bar.

    Args:
        returns: 1-D return series driving the first emission.
        volume: optional 1-D volume series, same length as
            ``returns``; log-volume becomes the second emission
            dimension.  Bars with non-finite or non-positive volume
            are masked (the log is undefined there).
        n_states: number of hidden regimes; at least 2.
        n_iter: EM iteration cap; at least 1.
        tol: absolute log-likelihood improvement below which EM
            reports convergence; a finite positive number.
        seed: non-negative seed of the initialization jitter rng.

    Returns:
        :class:`HmmRegimeReport`: int labels per input bar (masked
        bars read -1), the (n_states,) last-bar next-state
        distribution, the convergence flag and the final
        log-likelihood.

    Raises:
        ValueError: if ``n_states`` / ``n_iter`` / ``seed`` is not a
            valid integer, ``tol`` is not a finite positive number,
            an input is not 1-D, lengths differ, or fewer than 50
            usable bars survive masking.

    Example::

        >>> rng = np.random.default_rng(3)
        >>> rets = np.concatenate(
        ...     [-0.01 + 0.003 * rng.standard_normal(200),
        ...      0.000 + 0.003 * rng.standard_normal(200),
        ...      0.010 + 0.003 * rng.standard_normal(200)]
        ... )
        >>> report = hmm_regime(rets)
        >>> report["converged"]
        True
        >>> labels = report["hmm_regime"]
        >>> int(np.argmax(np.bincount(labels[:200], minlength=3)))
        0
        >>> int(np.argmax(np.bincount(labels[200:400], minlength=3)))
        1
        >>> int(np.argmax(np.bincount(labels[400:], minlength=3)))
        2

        A volume column joins the emissions as log-volume:

        >>> volume = np.repeat([2.0e6, 1.0e6, 3.0e6], 200)
        >>> two_dim = hmm_regime(rets, volume)
        >>> abs(float(two_dim["hmm_transition_prob"].sum()) - 1.0) < 1e-9
        True
    """
    checked_states = validate_int("n_states", n_states, minimum=2)
    checked_iter = validate_int("n_iter", n_iter)
    checked_tol = _validate_positive("tol", tol)
    checked_seed = validate_int("seed", seed, minimum=0)
    values_ret = _to_1d_float(returns, "returns")
    usable = np.isfinite(values_ret)
    columns = [values_ret]
    if volume is not None:
        values_vol = _to_1d_float(volume, "volume")
        if values_vol.size != values_ret.size:
            raise ValueError(
                "volume must have the same length as returns, got"
                f" {values_vol.size} and {values_ret.size}"
            )
        usable = usable & np.isfinite(values_vol) & (values_vol > 0.0)
        columns.append(np.log(np.where(usable, values_vol, 1.0)))
    finite_positions = np.flatnonzero(usable)
    if finite_positions.size < _HMM_MIN_OBS:
        raise ValueError(
            f"hmm_regime needs at least {_HMM_MIN_OBS} usable bars"
            f" (finite returns, positive finite volume when given),"
            f" got {finite_positions.size}"
        )
    obs = np.column_stack(columns)[finite_positions]
    rng = np.random.default_rng(checked_seed)
    means, variances, transition, start_prob = _hmm_init(
        obs, checked_states, rng
    )
    loglik_prev = float("-inf")
    converged = False
    gamma: np.ndarray
    loglik = float("-inf")
    for _ in range(checked_iter):
        gamma, xi_sum, loglik = _hmm_e_step(
            obs, means, variances, transition, start_prob
        )
        if abs(loglik - loglik_prev) < checked_tol:
            converged = True
            break
        loglik_prev = loglik
        means, variances, transition, start_prob = _hmm_m_step(
            obs, gamma, xi_sum
        )
    else:
        # iteration cap hit: refresh the posteriors once so the
        # returned labels match the final (post-M-step) parameters
        gamma, _, loglik = _hmm_e_step(
            obs, means, variances, transition, start_prob
        )
    order = np.argsort(means[:, 0])
    rank = np.argsort(order)
    regime = np.full(values_ret.size, -1, dtype=int)
    regime[finite_positions] = rank[np.argmax(gamma, axis=1)]
    transition_sorted = transition[np.ix_(order, order)]
    next_dist = gamma[-1][order] @ transition_sorted
    return HmmRegimeReport(
        hmm_regime=regime,
        hmm_transition_prob=next_dist / next_dist.sum(),
        converged=converged,
        loglik=loglik,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _demo_kalman() -> None:
    """Verified demo: the filter recovers a planted hedge ratio of 2."""
    rng = np.random.default_rng(20261116)
    n_obs = 1000
    price_x = np.exp(0.03 * np.cumsum(rng.standard_normal(n_obs)))
    ou_noise = np.zeros(n_obs)
    shocks = rng.standard_normal(n_obs)
    for step in range(1, n_obs):
        ou_noise[step] = 0.90 * ou_noise[step - 1] + 0.05 * shocks[step]
    price_y = 2.0 * price_x + ou_noise
    report = kalman_ou_spread(price_y, price_x)
    tail_beta = float(np.nanmean(report["kalman_beta"][-200:]))
    n_finite = int(np.isfinite(report["spread_z"]).sum())
    print(
        f"kalman_ou_spread : tail beta={tail_beta:.3f} (true 2.000)"
        f" half_life={report['half_life']:.2f} bars"
        f" spread_z finite={n_finite}/{n_obs}"
    )
    assert abs(tail_beta - 2.0) < 0.05
    assert bool(np.isfinite(report["spread_z"][100:]).all())
    assert 0.0 < report["half_life"] < 30.0


def _demo_garch() -> None:
    """Verified demo: sigma separates a planted two-vol-regime series."""
    rng = np.random.default_rng(7)
    rets = np.concatenate(
        [0.010 * rng.standard_normal(500), 0.050 * rng.standard_normal(500)]
    )
    report = garch_sigma_forecast(rets)
    first_half = float(np.mean(report["garch_sigma_forecast"][:500]))
    second_half = float(np.mean(report["garch_sigma_forecast"][500:]))
    print(
        f"garch_sigma_forecast: sigma[first]={first_half:.4f}"
        f" sigma[second]={second_half:.4f}"
        f" alpha={report['alpha']:.3f} beta={report['beta']:.3f}"
        f" converged={report['converged']}"
    )
    assert report["converged"]
    assert second_half > 2.0 * first_half


def _demo_hmm() -> None:
    """Verified demo: labels separate three planted mean levels."""
    rng = np.random.default_rng(5)
    rets = np.concatenate(
        [
            level + 0.003 * rng.standard_normal(300)
            for level in (-0.010, 0.000, 0.010)
        ]
    )
    report = hmm_regime(rets)
    labels = report["hmm_regime"]
    majority = [
        int(np.argmax(np.bincount(labels[start : start + 300], minlength=3)))
        for start in (0, 300, 600)
    ]
    print(
        f"hmm_regime       : segment majorities={majority}"
        f" loglik={report['loglik']:.1f}"
        f" converged={report['converged']}"
    )
    assert report["converged"]
    assert len(set(majority)) == 3
    assert majority[0] < majority[1] < majority[2]


def run_all_examples() -> None:
    """Demo the three generators on planted synthetic worlds."""
    print("=" * 72)
    print("feature_generators demo: synthetic ml_predictive features")
    print("=" * 72)
    _demo_kalman()
    _demo_garch()
    _demo_hmm()


if __name__ == "__main__":
    run_all_examples()
