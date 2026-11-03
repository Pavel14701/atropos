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
The kill machinery fails closed: a drawdown that cannot be computed
(an overflowing equity path, :class:`DrawdownNotComputable`) is
INCONCLUSIVE, never a silent pass.  LIVE_OK is NOT confirmation of
the edge - it only means reality has not killed it yet.

Every function is a pure mapping: data in -> result out.

Run ``python -m atropos.track_record`` for a synthetic demo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from atropos._utils import clean_1d

__all__ = [
    "DrawdownNotComputable",
    "drawdown_floor_check",
    "live_verdict",
    "prereg_divergence",
    "sharpe_standard_error",
]


# N818 (Error suffix) waived: the class name deliberately mirrors the
# public ``drawdown_not_computable`` INCONCLUSIVE reason it maps to.
class DrawdownNotComputable(ValueError):  # noqa: N818
    """Drawdown cannot be computed: the equity path is non-finite.

    The compounded equity overflowed to infinity, so ``equity / peak``
    is NaN and the deepest drawdown has no numeric value.  A kill
    barrier must fail closed on this input, never silently pass.
    Subclasses :class:`ValueError` (backward compatible: pre-existing
    ``except ValueError`` callers keep catching it), while
    :func:`live_verdict` catches exactly this class to degrade to an
    ``INCONCLUSIVE`` verdict with reason ``drawdown_not_computable``.
    """


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

    The initial unit of capital is the first peak, so a loss in the
    very first period counts as a drawdown.

    Args:
        values: cleaned return series.

    Returns:
        The deepest drawdown as a negative fraction (e.g. ``-0.30``);
        0.0 for an empty or never-drawing series.  Terminal equity
        ``<= 0`` yields the ``-1.0`` sentinel WITHOUT distinguishing
        "total loss" (equity exactly zero, a single return of
        ``-1.0``) from a NEGATIVE balance (any return below ``-1.0``,
        e.g. from leverage): a drawdown below -100% is not
        expressible as a fraction of the peak, so both cases collapse
        to the same value - a diagnostic limitation (behavior kept;
        callers needing the distinction must inspect the raw path).
        NaN when the compounded equity path overflows to infinity
        (gigantic growth): callers must classify that as "not
        computable", never as "no drawdown".
    """
    if values.size == 0:
        return 0.0
    # Overflow to inf is for the caller to classify, not a warning to
    # spam: inf/inf yields NaN drawdown entries here on purpose.
    with np.errstate(over="ignore", invalid="ignore"):
        equity = np.cumprod(np.concatenate(([1.0], 1.0 + values)))
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
    ``sqrt(periods)`` to annual units.  Asymptotically exact for
    i.i.d. Gaussian returns; real return streams only push the true
    SE wider than this formula.  For heavy-tailed / skewed returns
    the true SE is wider; the divergence test becomes
    anti-conservative (over-kills); consider bootstrap SE if that
    matters.  Autocorrelation widens the true SE in the same
    direction.

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

        >>> sharpe_standard_error(1.0, 100, periods=1)
        0.1224744871391589
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
            (the backtest estimate).  Any finite value is accepted
            here, including negative ones: this primitive is a pure
            z-test of two estimates, and the "nobody preregisters a
            dead edge" policy (``expected_sharpe > 0``) is enforced
            one level up, by :func:`live_verdict`.
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


def _drawdown_floor_from_clean(
    rets: np.ndarray, dd_floor: float
) -> dict[str, Any]:
    """Barrier verdict on an ALREADY-CLEANED return series.

    Private single-cleaning fast path shared by the public
    :func:`drawdown_floor_check` and :func:`live_verdict` (which
    cleans its input once and feeds both the barrier and the Sharpe
    leg from the same array).

    Args:
        rets: cleaned (finite, 1-D) live returns.
        dd_floor: already-validated strictly-negative drawdown floor.

    Returns:
        Dict with ``mdd``, ``dd_floor``, ``breached`` and ``n_obs``.

    Raises:
        DrawdownNotComputable: if the deepest drawdown is not finite
            - the compounded equity path overflowed.  A kill barrier
            must fail closed: "cannot compute" must never degrade to
            "not breached" (``nan <= floor`` is False).
    """
    mdd = _max_drawdown(rets)
    if not math.isfinite(mdd):
        raise DrawdownNotComputable(
            "drawdown not computable (non-finite equity path)"
        )
    return {
        "mdd": mdd,
        "dd_floor": float(dd_floor),
        "breached": bool(mdd <= float(dd_floor)),
        "n_obs": int(rets.size),
    }


def drawdown_floor_check(
    live_returns: Any, dd_floor: float
) -> dict[str, Any]:
    """Preregistered drawdown barrier: breached or not?

    The floor is a path rule from the prereg, monitored continuously:
    a breach kills immediately, without waiting for the decision
    point and without any additional statistics.

    An empty track reports ``mdd=0.0`` / ``breached=False``: with no
    observations the barrier is vacuously intact.  That answer is
    misleading when a truncated or empty feed is called directly, so
    prefer routing through :func:`live_verdict`, whose ``n_obs`` gate
    never lets a too-short track pass as LIVE_OK.

    Args:
        live_returns: per-period live returns since deployment.
        dd_floor: preregistered maximum-drawdown floor, negative
            (e.g. ``-0.25`` kills at a 25% drawdown).

    Returns:
        Dict with ``mdd`` (deepest live drawdown, negative),
        ``dd_floor``, ``breached`` and ``n_obs``.

    Raises:
        ValueError: if ``dd_floor`` is not strictly negative.
        DrawdownNotComputable: if the deepest drawdown is not finite
            (the compounded equity path overflowed - the barrier
            fails closed instead of silently passing); a
            ``ValueError`` subclass, so ``except ValueError``
            callers are unaffected.

    Example::

        >>> drawdown_floor_check(                     # doctest: +SKIP
        ...     [-0.30], dd_floor=-0.25)["breached"]
        True
    """
    if not float(dd_floor) < 0.0:
        raise ValueError(f"dd_floor must be < 0, got {dd_floor}")
    rets = clean_1d(live_returns)
    return _drawdown_floor_from_clean(rets, float(dd_floor))


def _validate_verdict_prereg(
    *,
    expected_sharpe: float,
    n_backtest: int,
    periods: int,
    min_observations: int,
    dd_floor: float,
    kill_z: float,
) -> None:
    """Raise ``ValueError`` unless every prereg parameter is sane.

    All prereg validation happens upfront, before any data is
    touched: ``dd_floor`` included, so a bad barrier cannot slip past
    as a late failure from deep inside the drawdown check.

    Args:
        expected_sharpe: preregistered annualized Sharpe promise.
        n_backtest: observations behind the prereg estimate.
        periods: periods per year.
        min_observations: preregistered decision point length.
        dd_floor: preregistered drawdown barrier.
        kill_z: one-sided kill threshold in sigmas.

    Raises:
        ValueError: on the first invalid parameter encountered.
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
    if not float(dd_floor) < 0.0:
        raise ValueError(f"dd_floor must be < 0, got {dd_floor}")
    if not 0.0 < float(kill_z) <= 4.0:
        raise ValueError(f"kill_z must be in (0, 4], got {kill_z}")


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
    degenerate returns, undefined statistics, or a drawdown that
    cannot be computed - an overflowing equity path raises
    :class:`DrawdownNotComputable`, caught here to fail closed;
    re-check later).

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
        ValueError: on invalid prereg parameters (``expected_sharpe``,
            ``n_backtest``, ``periods``, ``min_observations``,
            ``dd_floor`` or ``kill_z``).

    Example::

        >>> rng = np.random.default_rng(3)                # +SKIP
        >>> live = 0.001 + 0.01 * rng.standard_normal(600) # +SKIP
        >>> out = live_verdict(live, 1.5, n_backtest=1500)  # +SKIP
        >>> out["verdict"] in {"LIVE_OK", "LIVE_KILLED",    # +SKIP
        ...                    "INCONCLUSIVE"}
        True
    """
    _validate_verdict_prereg(
        expected_sharpe=expected_sharpe,
        n_backtest=n_backtest,
        periods=periods,
        min_observations=min_observations,
        dd_floor=dd_floor,
        kill_z=kill_z,
    )

    rets = clean_1d(live_returns)
    n_obs = int(rets.size)
    result: dict[str, Any] = {
        "verdict": "INCONCLUSIVE",
        "reason": "empty_track_record",
        "n_obs": n_obs,
        "live_sharpe": float("nan"),
        "expected_sharpe": float(expected_sharpe),
        "z": float("nan"),
        "se_live": float("nan"),
        "se_expected": float("nan"),
        "mdd": float("nan"),
        "dd_breached": False,
        "kill_z": float(kill_z),
        "min_observations": int(min_observations),
        "periods": int(periods),
    }
    try:
        dd = _drawdown_floor_from_clean(rets, float(dd_floor))
    except DrawdownNotComputable:
        # Fail closed: NaN mdd would make `nan <= floor` False and a
        # naive barrier check would wave the strategy through.  The
        # catch is deliberately NARROW: an ordinary ValueError raised
        # anywhere else (a validation bug, a bad parameter that slipped
        # past the upfront checks) must propagate, not quietly turn
        # into an INCONCLUSIVE verdict.
        result["reason"] = "drawdown_not_computable"
        return result
    result["mdd"] = dd["mdd"]
    result["dd_breached"] = dd["breached"]
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
    else:
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


def _tile_with_sharpe(
    target_sharpe: float, up_return: float, n_pairs: int
) -> np.ndarray:
    """Deterministic ``[up, down]`` tile of analytically known Sharpe.

    A two-point tile has ``mean = (up + down) / 2`` and population
    std ``(up - down) / 2``, so ``SR_pp = (up + down) / (up - down)``;
    choosing ``down = up * (r - 1) / (r + 1)`` with
    ``r = target_sharpe / sqrt(252)`` pins the per-period Sharpe at
    ``r`` exactly (the sample ddof=1 estimate sits below it by a
    ``sqrt((n - 1) / n)`` factor, e.g. 0.9995 at n=1000).  No seed,
    no luck: the annualized Sharpe of the stream is known in closed
    form, and the drawdown equals the down leg (every up move sets a
    fresh peak).

    Args:
        target_sharpe: annualized Sharpe the tile realizes.
        up_return: the up leg of the tile.
        n_pairs: number of up/down pairs (stream length
            ``2 * n_pairs``).

    Returns:
        1-D array of the tiled return stream.
    """
    ratio = float(target_sharpe) / float(np.sqrt(252.0))
    down_return = up_return * (ratio - 1.0) / (ratio + 1.0)
    return np.tile([up_return, down_return], int(n_pairs))


def run_all_examples() -> None:
    """Demo (and regression-test) the verdict on four live fates.

    Realistic prereg: the backtest promised ``SR=1.5`` on 2000 daily
    observations.  All four canonical outcomes are asserted, so the
    demo doubles as a regression test: a drift in the verdict logic
    fails loudly instead of printing a wrong line.
    """
    expected = 1.5  # preregistered promise (n_backtest=2000)
    live_intact = _tile_with_sharpe(1.5, 0.02, 500)  # n=1000, SR~1.50
    live_faded = _tile_with_sharpe(0.5, 0.02, 1000)  # n=2000, SR~0.50
    live_crash = np.concatenate(  # 280 obs, mdd ~ -34% < -25% floor
        [live_intact[:240], np.full(40, -0.01)]
    )

    print("=" * 72)
    print(
        "track_record demo: live PnL vs prereg "
        f"(expected SR={expected:.2f}, n_backtest=2000)"
    )
    print("=" * 72)
    cases = [
        ("edge intact", live_intact, "LIVE_OK", "consistent_with_prereg"),
        (
            "edge faded in prod",
            live_faded,
            "LIVE_KILLED",
            "live_below_prereg",
        ),
        (
            "crash breach",
            live_crash,
            "LIVE_KILLED",
            "drawdown_floor_breached",
        ),
        (
            "track too short",
            live_intact[:80],
            "INCONCLUSIVE",
            "track_too_short",
        ),
    ]
    for name, stream, want_verdict, want_reason in cases:
        out = live_verdict(
            stream,
            expected,
            n_backtest=2000,
            min_observations=252,
            dd_floor=-0.25,
        )
        assert out["verdict"] == want_verdict, (name, out)
        assert out["reason"] == want_reason, (name, out)
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
