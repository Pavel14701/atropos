"""Post-deployment track-record verification: live PnL vs the prereg.

The pipeline ends at SURVIVED, but a survivor is not the truth: the
preregistration exists precisely so that reality can kill the strategy
later.  This module is the production stage of the doctrine:

* :func:`sharpe_standard_error` - Lo (2002) i.i.d. standard error of
  the annualized Sharpe ratio
* :func:`prereg_divergence` - one-sided z-test: is the live Sharpe
  statistically below the preregistered expectation?
* :func:`drawdown_floor_check` - preregistered drawdown barrier,
  monitored continuously, kills immediately
* :func:`live_verdict` - the contract test: LIVE_OK / LIVE_KILLED /
  INCONCLUSIVE from a live return track

Decision discipline: the divergence test fires ONCE, at the
preregistered decision point (``min_observations``); the drawdown
floor is a preregistered path rule (a barrier, not a repeated test),
so it may be watched continuously without multiple-testing inflation.
LIVE_OK is NOT confirmation of the edge - it only means reality has
not killed it yet.

Every function is a pure mapping: data in -> result out.

Run ``python -m atropos.track_record`` for a synthetic demo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "drawdown_floor_check",
    "live_verdict",
    "prereg_divergence",
    "run_all_examples",
    "sharpe_standard_error",
]


def _clean(values: Any) -> np.ndarray:
    """Coerce array-like input to a flat 1-D float array, NaNs dropped.

    Args:
        values: array-like of numbers; non-finite entries are removed
            before any statistic is computed.

    Returns:
        1-D float64 array holding only the finite entries.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _is_constant(values: np.ndarray) -> bool:
    """True for empty or constant series (zero peak-to-peak range).

    Args:
        values: cleaned numeric series.

    Returns:
        Whether every entry is identical.  ``np.std`` of a constant
        float series can carry rounding dust (e.g. 1e-19), so the
        exact peak-to-peak range is used instead.
    """
    return bool(values.size == 0 or np.ptp(values) == 0.0)


def _annualized_sharpe(values: np.ndarray, periods: int) -> float:
    """Annualized Sharpe ``mean / std * sqrt(periods)`` with guards.

    Args:
        values: cleaned return series.
        periods: periods per year (252 for daily data).

    Returns:
        Annualized Sharpe ratio (ddof=1); NaN when there are fewer
        than two observations or the series is constant.
    """
    if values.size < 2 or _is_constant(values):
        return float("nan")
    stdev = float(np.std(values, ddof=1))
    return float(np.mean(values) / stdev * np.sqrt(float(periods)))


def _max_drawdown(values: np.ndarray) -> float:
    """Max peak-to-trough drawdown of a return series (negative).

    Args:
        values: cleaned return series.

    Returns:
        The deepest drawdown as a negative fraction (e.g. ``-0.30``);
        0.0 for an empty or never-drawing series.  A single return of
        ``-1.0`` or worse wipes the equity to zero and yields ``-1.0``.
    """
    if values.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + values)
    if equity[-1] <= 0.0:
        return -1.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(np.min(drawdown))


def sharpe_standard_error(
    sharpe: float, n_obs: int, periods: int = 252
) -> float:
    """Lo (2002) i.i.d. standard error of an annualized Sharpe ratio.

    ``se(SR_pp) = sqrt((1 + SR_pp^2 / 2) / n_obs)``, rescaled by
    ``sqrt(periods)`` to annual units.  Valid under i.i.d. returns;
    autocorrelation widens the true SE, so the divergence test based
    on it is conservative in the strategy's favor.

    Args:
        sharpe: annualized Sharpe ratio estimate.
        n_obs: number of return observations behind it.
        periods: periods per year (252 for daily data).

    Returns:
        Standard error of the annualized Sharpe ratio.

    Raises:
        ValueError: if ``sharpe`` is not finite, ``n_obs`` < 1 or
            ``periods`` < 1.

    Example::

        >>> sharpe_standard_error(1.0, 100, periods=1)   # doctest: +SKIP
        0.12247448713915891
    """
    if not math.isfinite(float(sharpe)):
        raise ValueError(f"sharpe must be finite, got {sharpe}")
    if int(n_obs) < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if int(periods) < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    root = float(np.sqrt(float(periods)))
    sr_pp = float(sharpe) / root
    se_pp = float(np.sqrt((1.0 + 0.5 * sr_pp**2) / float(n_obs)))
    return se_pp * root


def prereg_divergence(
    live_sharpe: float,
    n_live: int,
    expected_sharpe: float,
    n_backtest: int,
    *,
    periods: int = 252,
    kill_z: float = 1.64,
) -> dict[str, Any]:
    """One-sided z-test: live Sharpe below the preregistered Sharpe?

    ``z = (SR_live - SR_prereg) / sqrt(se_live^2 + se_prereg^2)`` with
    Lo (2002) standard errors.  The kill is one-sided: only a live
    track significantly WORSE than the prereg kills; outperformance is
    not evidence of anything (no validation, only falsification).

    Args:
        live_sharpe: annualized Sharpe of the live track.
        n_live: live observations.
        expected_sharpe: annualized Sharpe promised by the prereg
            (the backtest estimate).
        n_backtest: observations behind the backtest estimate.
        periods: periods per year (252 for daily data).
        kill_z: kill threshold in sigmas (one-sided; 1.64 ~= 5%).

    Returns:
        Dict with ``z``, ``se_live``, ``se_expected``, ``kill_z`` and
        ``diverged`` (True when ``z <= -kill_z``).  ``z`` is NaN when
        either standard error is undefined.

    Raises:
        ValueError: on non-finite Sharpe inputs, ``n_live`` or
            ``n_backtest`` < 1, or ``kill_z`` not in (0, 4].

    Example::

        >>> out = prereg_divergence(0.5, 504, 1.5, 1008)  # doctest: +SKIP
        >>> out["diverged"]                               # doctest: +SKIP
        False
    """
    if not math.isfinite(float(live_sharpe)):
        raise ValueError(f"live_sharpe must be finite, got {live_sharpe}")
    if not math.isfinite(float(expected_sharpe)):
        raise ValueError(
            f"expected_sharpe must be finite, got {expected_sharpe}"
        )
    if int(n_live) < 1 or int(n_backtest) < 1:
        raise ValueError(
            "n_live and n_backtest must be >= 1 "
            f"(got {n_live} and {n_backtest})"
        )
    if not 0.0 < float(kill_z) <= 4.0:
        raise ValueError(f"kill_z must be in (0, 4], got {kill_z}")
    se_live = sharpe_standard_error(live_sharpe, n_live, periods)
    se_expected = sharpe_standard_error(expected_sharpe, n_backtest, periods)
    se_total = float(np.hypot(se_live, se_expected))
    if se_total == 0.0 or not math.isfinite(se_total):
        z_stat = float("nan")
    else:
        z_stat = (float(live_sharpe) - float(expected_sharpe)) / se_total
    return {
        "z": z_stat,
        "se_live": se_live,
        "se_expected": se_expected,
        "kill_z": float(kill_z),
        "diverged": bool(math.isfinite(z_stat) and z_stat <= -float(kill_z)),
    }


def drawdown_floor_check(
    live_returns: Any, dd_floor: float
) -> dict[str, Any]:
    """Preregistered drawdown barrier: breached or not?

    The floor is a path rule from the prereg, monitored continuously:
    a breach kills immediately, without waiting for the decision
    point and without any additional statistics.

    Args:
        live_returns: per-period live returns since deployment.
        dd_floor: preregistered maximum-drawdown floor, negative
            (e.g. ``-0.25`` kills at a 25% drawdown).

    Returns:
        Dict with ``mdd`` (deepest live drawdown, negative),
        ``dd_floor``, ``breached`` and ``n_obs``.

    Raises:
        ValueError: if ``dd_floor`` is not strictly negative.

    Example::

        >>> drawdown_floor_check(                     # doctest: +SKIP
        ...     [-0.30], dd_floor=-0.25)["breached"]
        True
    """
    if not float(dd_floor) < 0.0:
        raise ValueError(f"dd_floor must be < 0, got {dd_floor}")
    rets = _clean(live_returns)
    mdd = _max_drawdown(rets)
    return {
        "mdd": mdd,
        "dd_floor": float(dd_floor),
        "breached": bool(mdd <= float(dd_floor)),
        "n_obs": int(rets.size),
    }


def live_verdict(
    live_returns: Any,
    expected_sharpe: float,
    *,
    n_backtest: int,
    periods: int = 252,
    min_observations: int = 252,
    dd_floor: float = -0.25,
    kill_z: float = 1.64,
) -> dict[str, Any]:
    """Production track-record verdict for a deployed strategy.

    LIVE_OK / LIVE_KILLED / INCONCLUSIVE.  Feeds the live PnL track of
    a deployed SURVIVED strategy against
    its preregistered expectations.  Two preregistered kill rules, in
    order of precedence:

    * drawdown barrier - any live drawdown at or below ``dd_floor``
      kills immediately (path rule, no minimum sample);
    * divergence test - at the decision point
      (``len(live) >= min_observations``) the live Sharpe must not sit
      ``kill_z`` sigmas below ``expected_sharpe`` (Lo 2002 SEs).

    Verdicts: ``LIVE_KILLED`` (a rule fired; the prereg tombstone gets
    its production entry), ``LIVE_OK`` (no rule fired - NOT a
    confirmation of the edge), ``INCONCLUSIVE`` (track too short,
    degenerate returns, or undefined statistics; re-check later).

    Args:
        live_returns: per-period live returns since deployment.
        expected_sharpe: annualized Sharpe promised by the prereg;
            must be positive (nobody preregisters a dead edge).
        n_backtest: observations behind the preregistered estimate.
        periods: periods per year (252 for daily data).
        min_observations: preregistered decision point length.
        dd_floor: preregistered drawdown barrier, negative.
        kill_z: one-sided kill threshold in sigmas (1.64 ~= 5%).

    Returns:
        Dict with ``verdict``, ``reason``, ``n_obs``,
        ``live_sharpe``, ``expected_sharpe``, ``z``, ``se_live``,
        ``se_expected``, ``mdd``, ``dd_breached``, ``kill_z``,
        ``min_observations``, ``periods``.

    Raises:
        ValueError: on invalid prereg parameters.

    Example::

        >>> rng = np.random.default_rng(3)                # +SKIP
        >>> live = 0.001 + 0.01 * rng.standard_normal(600) # +SKIP
        >>> out = live_verdict(live, 1.5, n_backtest=1500)  # +SKIP
        >>> out["verdict"] in {"LIVE_OK", "LIVE_KILLED",    # +SKIP
        ...                    "INCONCLUSIVE"}
        True
    """
    if not math.isfinite(float(expected_sharpe)) or (
        float(expected_sharpe) <= 0.0
    ):
        raise ValueError(
            f"expected_sharpe must be finite and > 0, got {expected_sharpe}"
        )
    if int(n_backtest) < 1:
        raise ValueError(f"n_backtest must be >= 1, got {n_backtest}")
    if int(periods) < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    if int(min_observations) < 8:
        raise ValueError(
            f"min_observations must be >= 8, got {min_observations}"
        )
    if not 0.0 < float(kill_z) <= 4.0:
        raise ValueError(f"kill_z must be in (0, 4], got {kill_z}")

    rets = _clean(live_returns)
    n_obs = int(rets.size)
    dd = drawdown_floor_check(rets, dd_floor)
    result: dict[str, Any] = {
        "verdict": "INCONCLUSIVE",
        "reason": "empty_track_record",
        "n_obs": n_obs,
        "live_sharpe": float("nan"),
        "expected_sharpe": float(expected_sharpe),
        "z": float("nan"),
        "se_live": float("nan"),
        "se_expected": float("nan"),
        "mdd": dd["mdd"],
        "dd_breached": dd["breached"],
        "kill_z": float(kill_z),
        "min_observations": int(min_observations),
        "periods": int(periods),
    }
    if dd["breached"]:
        result["verdict"] = "LIVE_KILLED"
        result["reason"] = "drawdown_floor_breached"
        return result
    if n_obs < int(min_observations):
        result["reason"] = (
            "empty_track_record" if n_obs == 0 else "track_too_short"
        )
        return result

    live_sharpe = _annualized_sharpe(rets, periods)
    result["live_sharpe"] = live_sharpe
    if not math.isfinite(live_sharpe):
        result["reason"] = "degenerate_returns"
        return result
    divergence = prereg_divergence(
        live_sharpe,
        n_obs,
        float(expected_sharpe),
        int(n_backtest),
        periods=periods,
        kill_z=kill_z,
    )
    result["z"] = divergence["z"]
    result["se_live"] = divergence["se_live"]
    result["se_expected"] = divergence["se_expected"]
    if not math.isfinite(float(divergence["z"])):
        result["reason"] = "divergence_undefined"
        return result
    if divergence["diverged"]:
        result["verdict"] = "LIVE_KILLED"
        result["reason"] = "live_below_prereg"
        return result
    result["verdict"] = "LIVE_OK"
    result["reason"] = "consistent_with_prereg"
    return result


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _fmt_or_dash(value: float, template: str, dash: str) -> str:
    """Format ``value`` with ``template``, or a dash placeholder for NaN."""
    if not math.isfinite(value):
        return dash
    return template.format(value)


def run_all_examples() -> None:
    """Demo the production track-record verdict on four live fates."""
    # Deterministic streams (sample SR is exact, no seed luck): a
    # [0, gain] tile has mean = std = gain/2, so SR_pp = 1.0 exactly.
    live_intact = np.tile([0.0, 0.016], 500)  # SR_ann ~ sqrt(252)
    live_inverted = np.tile([0.0, -0.016], 500)  # SR_ann ~ -sqrt(252)
    live_crash = np.concatenate(
        [np.tile([0.0, 0.016], 60), np.full(40, -0.01)]
    )
    expected = _annualized_sharpe(np.tile([0.0, 0.016], 1000), 252)

    print("=" * 72)
    print(
        "track_record demo: live PnL vs prereg "
        f"(expected SR={expected:.2f}, n_backtest=2000)"
    )
    print("=" * 72)
    cases = [
        ("edge intact", live_intact, 500, -0.40),
        ("edge inverted in prod", live_inverted, 500, -0.40),
        ("crash breach", live_crash, 150, -0.25),
        ("track too short", live_intact[:80], 500, -0.40),
    ]
    for name, stream, min_obs, floor in cases:
        out = live_verdict(
            stream,
            expected,
            n_backtest=2000,
            min_observations=min_obs,
            dd_floor=floor,
        )
        z_text = _fmt_or_dash(out["z"], "z={:+.2f}", "z= --")
        sr_text = _fmt_or_dash(
            out["live_sharpe"], "live_SR={:+.2f}", "live_SR= --"
        )
        print(
            f"{name:22s} -> {out['verdict']:13s} {sr_text} {z_text} "
            f"mdd={out['mdd']:+.1%} ({out['reason']})"
        )


if __name__ == "__main__":
    run_all_examples()
