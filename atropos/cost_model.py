"""Transaction-cost modeling: turnover, proportional costs, COST-DEATH.

The Atropos doctrine kills strategies that only survive in a
frictionless world (the COST-DEATH flag), but the package had no cost
arithmetic.  This module closes that gap:

* :func:`turnover_series` - one-sided turnover ``|pos[t] - pos[t-1]|``
* :func:`apply_proportional_costs` - net returns after bps costs
* :func:`annual_turnover` - annualized turnover and rebalance count
* :func:`breakeven_cost_bps` - one-way cost level that eats the edge
* :func:`cost_death_test` - contract test with a PASS / FAIL
  (COST-DEATH) / INCONCLUSIVE verdict from a block-bootstrap CI of
  the net mean

Every function is a pure mapping: data in -> result out.

Run ``python -m atropos.cost_model`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atropos._utils import clean_1d, validate_int, validate_number
from atropos.bootstrap_ci import block_bootstrap_ci

__all__ = [
    "annual_turnover",
    "apply_proportional_costs",
    "breakeven_cost_bps",
    "cost_death_test",
    "turnover_series",
]

# Mean turnover below this level is numerical dust -- positions
# re-derived each period that differ at the 1e-15 scale -- not trading.
# Breakeven against dust is infinite, not an astronomic finite number.
_TURNOVER_DUST_EPS = 1e-12


def _clean_pair(
    strategy_returns: Any, positions: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Clean gross returns and positions jointly, pair by pair.

    A joint finite mask ``isfinite(gross) & isfinite(pos)`` is applied
    to BOTH arrays at once. Dropping non-finite entries independently
    (the old behaviour) silently misaligns the pair whenever the NaNs
    sit at different periods: ``returns[t]`` would be charged against
    ``positions[s]`` from a completely different period.

    Inputs must be 1-D after ``asarray``: a 2-D ``(n, 1)`` column used
    to be flattened silently, and any wider array was strung out into a
    bogus longer series - both now raise instead.

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions or weights behind those
            returns.

    Returns:
        Tuple ``(gross, pos)`` of equal-length 1-D float64 arrays
        holding only the periods where both inputs are finite.

    Raises:
        ValueError: if the raw inputs differ in length or either input
            is not 1-D (``must be 1-D``; an ``(n, 1)`` column counts as
            2-D and is rejected, not flattened).
    """
    gross_arr = np.asarray(strategy_returns, dtype=float)
    pos_arr = np.asarray(positions, dtype=float)
    if gross_arr.ndim > 1:
        raise ValueError(
            "strategy_returns must be 1-D, got "
            f"{gross_arr.ndim}-D input of shape {gross_arr.shape}"
        )
    if pos_arr.ndim > 1:
        raise ValueError(
            "positions must be 1-D, got "
            f"{pos_arr.ndim}-D input of shape {pos_arr.shape}"
        )
    gross_raw = gross_arr.reshape(-1)
    pos_raw = pos_arr.reshape(-1)
    if gross_raw.size != pos_raw.size:
        raise ValueError(
            "strategy_returns and positions must have the same length "
            f"(got {int(gross_raw.size)} and {int(pos_raw.size)})"
        )
    both_finite = np.isfinite(gross_raw) & np.isfinite(pos_raw)
    gross = np.asarray(gross_raw[both_finite])
    pos = np.asarray(pos_raw[both_finite])
    return gross, pos


def turnover_series(positions: Any) -> np.ndarray:
    """One-sided turnover per period ``|pos[t] - pos[t-1]|``.

    The first element is 0 by definition (no previous position to
    rebalance from); every later element is the absolute position
    change entering period ``t``.

    Warning:
        The entry into the initial position at ``t=0`` is NOT counted
        as turnover (``tvr[0]`` is pinned to 0 even when
        ``positions[0] != 0``), so entry costs are systematically
        understated by up to one entry trade.

    Args:
        positions: 1-D positions/weights (non-finite entries dropped).

    Returns:
        Turnover series, same length as the cleaned positions.

    Example::

        >>> turnover_series([0.0, 1.0, 1.0, 0.0])
        array([0., 1., 0., 1.])
    """
    pos = clean_1d(positions)
    tvr = np.zeros(pos.size, dtype=float)
    if pos.size > 1:
        tvr[1:] = np.abs(np.diff(pos))
    return tvr


def _net_returns_clean(
    gross: np.ndarray, pos: np.ndarray, cost_bps: float
) -> np.ndarray:
    """Net returns for already-cleaned, pair-aligned inputs.

    No re-validation and no re-cleaning: callers must pass outputs of
    :func:`_clean_pair` (or equivalent finite, equal-length arrays) and
    a non-negative ``cost_bps``, so the pair cleaning runs exactly once
    per pipeline instead of once per helper.

    Args:
        gross: cleaned gross returns.
        pos: cleaned positions aligned with ``gross``.
        cost_bps: non-negative one-way cost in basis points.

    Returns:
        Net return series ``gross - (cost_bps / 1e4) * turnover``.
    """
    cost_rate = float(cost_bps) / 1e4
    tvr = turnover_series(pos)
    drag = cost_rate * tvr
    overflow = np.isinf(drag)
    if overflow.size:
        overflow[0] = False  # tvr[0] is pinned to 0, never overflows
    if overflow.any():
        # opposite-sign extreme positions: |a - b| > float64 max, but
        # there |a - b| == |a| + |b| exactly, so distributing the rate
        # before the sum keeps the TRUE drag representable
        drag = np.where(
            overflow,
            cost_rate * np.abs(pos[:-1]) + cost_rate * np.abs(pos[1:]),
            drag,
        )
    return gross - drag


def apply_proportional_costs(
    strategy_returns: Any, positions: Any, cost_bps: float
) -> np.ndarray:
    """Net returns after proportional one-way transaction costs.

    ``net[t] = gross[t] - (cost_bps / 1e4) * turnover[t]`` where
    ``cost_bps`` is the one-way proportional cost per unit of traded
    exposure, in basis points.

    Warning:
        The entry into the initial position at ``t=0`` is NOT counted
        as turnover, so entry costs are systematically understated by
        up to one entry trade (see :func:`turnover_series`).

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions/weights behind those returns.
        cost_bps: non-negative one-way cost in basis points.

    Returns:
        Net return series, same length as the jointly cleaned inputs.

    Raises:
        ValueError: if ``cost_bps`` is negative or the raw inputs
            differ in length.

    Example::

        >>> apply_proportional_costs(
        ...     [0.01, 0.02], [0.0, 1.0], cost_bps=10.0)
        array([0.01 , 0.019])
    """
    if validate_number("cost_bps", cost_bps) < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    gross, pos = _clean_pair(strategy_returns, positions)
    return _net_returns_clean(gross, pos, cost_bps)


def annual_turnover(positions: Any, periods: int = 252) -> dict[str, Any]:
    """Annualized turnover plus rebalance counts.

    Warning:
        The entry into the initial position at ``t=0`` is NOT counted
        as turnover, so entry costs are systematically understated by
        up to one entry trade (see :func:`turnover_series`).

    Args:
        positions: 1-D positions/weights (non-finite entries dropped).
        periods: periods per year (252 for daily data); must be >= 1.

    Returns:
        Dict with ``annual_turnover`` (mean turnover per period scaled
        by ``periods``), ``n_changes`` (count of nonzero turnover
        steps) and ``n_obs`` (number of cleaned observations).

    Raises:
        ValueError: if ``periods`` is not an integer >= 1; raised
            before the data is touched.

    Example::

        >>> annual_turnover([0.0, 1.0, 1.0, 0.0], periods=10)
        {'annual_turnover': 5.0, 'n_changes': 2, 'n_obs': 4}
    """
    periods = validate_int("periods", periods)
    tvr = turnover_series(positions)
    if tvr.size == 0:
        return {
            "annual_turnover": float("nan"),
            "n_changes": 0,
            "n_obs": 0,
        }
    return {
        "annual_turnover": float(np.mean(tvr) * float(periods)),
        "n_changes": int(np.count_nonzero(tvr[1:])),
        "n_obs": int(tvr.size),
    }


def breakeven_cost_bps(
    strategy_returns: Any, positions: Any
) -> dict[str, Any]:
    """One-way cost level (bps) that exactly eats the gross edge.

    ``breakeven = 1e4 * mean(gross) / mean(turnover)``: the signal
    stays profitable at any one-way cost below this level and dies
    above it.

    Zero trading is detected robustly against floating dust: a mean
    turnover of exactly 0, any turnover series with zero peak-to-peak
    range, or mean turnover below ``_TURNOVER_DUST_EPS`` (1e-12) all
    count as "never trades". A mean turnover of 1e-15 left over from
    re-derived weights therefore yields ``inf`` (or NaN without edge)
    instead of an astronomic finite breakeven such as 1e13.

    Warning:
        The entry into the initial position at ``t=0`` is NOT counted
        as turnover, so entry costs are systematically understated by
        up to one entry trade (see :func:`turnover_series`).

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions/weights behind those returns.

    Returns:
        Dict with ``breakeven_cost_bps``; ``inf`` when the gross mean
        is positive but the strategy never trades, NaN when there is
        neither edge nor trading.

    Raises:
        ValueError: if the raw inputs differ in length.

    Example::

        >>> breakeven_cost_bps(
        ...     [0.001] * 4, [0.0, 1.0, 1.0, 1.0])
        {'breakeven_cost_bps': 40.0}
    """
    gross, pos = _clean_pair(strategy_returns, positions)
    tvr = turnover_series(pos)
    mean_gross = float(np.mean(gross)) if gross.size else float("nan")
    mean_tvr = float(np.mean(tvr)) if tvr.size else float("nan")
    never_trades = bool(mean_tvr < _TURNOVER_DUST_EPS) or (
        tvr.size > 0 and bool(np.ptp(tvr) == 0.0)
    )
    if never_trades:
        value = float("inf") if mean_gross > 0.0 else float("nan")
    else:
        value = float(1e4 * mean_gross / mean_tvr)
    return {"breakeven_cost_bps": value}


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


def cost_death_test(
    strategy_returns: Any,
    positions: Any,
    cost_bps: float,
    *,
    periods: int = 252,
    n_boot: int = 1000,
    block: int = 20,
    ci_level: float = 0.95,
    min_net_sharpe: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """COST-DEATH contract test: does the edge survive ``cost_bps``?

    Cleans the return/position pair jointly once, computes net returns
    via :func:`apply_proportional_costs`, builds a circular moving-block
    bootstrap CI for the net mean
    (:func:`atropos.bootstrap_ci.block_bootstrap_ci`) and derives the
    annualized net/gross Sharpe ratios.

    Verdict:

    * ``PASS`` - CI lower bound > 0 AND net Sharpe >= min_net_sharpe
    * ``FAIL`` - otherwise; FAIL == COST-DEATH (costs killed the edge)
    * ``INCONCLUSIVE`` - the bootstrap could not run (fewer than 8
      valid observations) or the net series is degenerate (constant)

    Warning:
        The entry into the initial position at ``t=0`` is NOT counted
        as turnover, so entry costs are systematically understated by
        up to one entry trade (see :func:`turnover_series`).

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions or weights behind those returns.
        cost_bps: non-negative one-way cost in basis points.
        periods: periods per year (252 for daily data); must be >= 1.
        n_boot: number of bootstrap replicates for the net-mean CI;
            must be >= 1.
        block: block length of the moving-block bootstrap; must be >= 1.
        ci_level: confidence level of the CI, strictly inside (0, 1).
        min_net_sharpe: floor the net annualized Sharpe must clear;
            must be >= 0. A negative floor would let ANY Sharpe pass -
            the opposite of a kill criterion - so it is rejected here
            instead of silently disabling the check.
        seed: seed for the bootstrap; fixed seed -> reproducible CI.

    Returns:
        Dict with keys ``net_mean`` (net mean), ``p_value`` (always
        None: the percentile bootstrap yields no p-value),
        ``verdict``, ``net_sharpe``, ``gross_sharpe``,
        ``cost_drag_mean`` (mean return lost to costs), ``ci_low``,
        ``ci_high``, ``n_obs``, ``breakeven_cost_bps``,
        ``annual_turnover``, ``n_changes``.

    Raises:
        ValueError: if ``ci_level`` is outside (0, 1) or not a finite
            number, ``periods`` / ``n_boot`` / ``block`` is not an
            integer >= 1, ``cost_bps`` is negative, ``min_net_sharpe``
            < 0, either raw input is not 1-D, or the raw inputs differ
            in length. Every parameter check (including ``n_boot`` /
            ``block``, which used to surface late from
            ``block_bootstrap_ci``) runs before the data is touched.

    Example::

        >>> rng = np.random.default_rng(7)               # doctest: +SKIP
        >>> out = cost_death_test(rng.normal(0.001, 0.01, # doctest: +SKIP
        ...     750), np.ones(750), 10.0, seed=1)
        >>> expected = {"PASS", "FAIL", "INCONCLUSIVE"}
        >>> out["verdict"] in expected  # doctest: +SKIP
        True
    """
    if not 0.0 < validate_number("ci_level", ci_level) < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    periods = validate_int("periods", periods)
    if validate_number("cost_bps", cost_bps) < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    n_boot = validate_int("n_boot", n_boot)
    block = validate_int("block", block)
    if validate_number("min_net_sharpe", min_net_sharpe) < 0.0:
        raise ValueError(f"min_net_sharpe must be >= 0, got {min_net_sharpe}")
    gross, pos = _clean_pair(strategy_returns, positions)
    net = _net_returns_clean(gross, pos, cost_bps)
    turn = annual_turnover(pos, periods=periods)
    breakeven = breakeven_cost_bps(gross, pos)
    net_sharpe = _annualized_sharpe(net, periods)
    gross_sharpe = _annualized_sharpe(gross, periods)
    cost_drag = float(np.mean(gross - net)) if net.size else float("nan")
    boot = block_bootstrap_ci(
        net,
        n_boot=n_boot,
        block=block,
        stat=np.mean,
        ci_level=ci_level,
        seed=seed,
    )
    boot_ok = bool(boot["method"] != "insufficient_data")
    if not boot_ok or _is_constant(net):
        verdict = "INCONCLUSIVE"
    elif float(boot["ci_low"]) > 0.0 and net_sharpe >= float(min_net_sharpe):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "net_mean": float(boot["point_estimate"]),
        "p_value": None,
        "verdict": verdict,
        "net_sharpe": net_sharpe,
        "gross_sharpe": gross_sharpe,
        "cost_drag_mean": cost_drag,
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
        "n_obs": int(net.size),
        "breakeven_cost_bps": breakeven["breakeven_cost_bps"],
        "annual_turnover": turn["annual_turnover"],
        "n_changes": turn["n_changes"],
    }


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo the cost model on a synthetic strong long/flat strategy."""
    rng = np.random.default_rng(7)
    n_obs = 750
    shocks = rng.standard_normal(n_obs)
    asset = np.zeros(n_obs)
    for step in range(1, n_obs):
        asset[step] = 0.5 * asset[step - 1] + 0.01 * shocks[step]
    positions = np.zeros(n_obs)
    positions[1:] = (asset[:-1] > 0).astype(float)
    strategy = positions * asset

    print("=" * 72)
    print(
        "cost_model demo: long/flat sign strategy on AR(1) returns"
        f" (n={n_obs}, seed=7)"
    )
    print("=" * 72)
    turn = annual_turnover(positions)
    print(
        f"annual_turnover   : {turn['annual_turnover']:.1f} "
        f"({turn['n_changes']} changes, n_obs={turn['n_obs']})"
    )
    # Layer-0 verification: the demo doubles as a regression test
    assert turn["n_obs"] == n_obs
    assert 0 < turn["n_changes"] < n_obs
    assert 30.0 < turn["annual_turnover"] < 150.0
    breakeven = breakeven_cost_bps(strategy, positions)
    print(f"breakeven_cost_bps: {breakeven['breakeven_cost_bps']:.1f}")
    # the two demo costs straddle the breakeven: 10 bps must survive,
    # 200 bps must die
    assert 10.0 < breakeven["breakeven_cost_bps"] < 200.0
    death_by_cost: dict[float, dict[str, Any]] = {}
    for cost in (10.0, 200.0):
        out = cost_death_test(
            strategy, positions, cost, n_boot=1000, block=20, seed=11
        )
        death_by_cost[cost] = out
        print(
            f"cost_death_test({cost:5.1f} bps): verdict={out['verdict']}"
            f" net_mean={out['net_mean']:+.5f}"
            f" net_sharpe={out['net_sharpe']:+.2f}"
            f" CI=[{out['ci_low']:+.5f}, {out['ci_high']:+.5f}]"
            f" cost_drag={out['cost_drag_mean']:.5f}"
        )
    out_10 = death_by_cost[10.0]
    out_200 = death_by_cost[200.0]
    assert out_10["verdict"] == "PASS"
    assert out_10["ci_low"] > 0.0 and out_10["net_mean"] > 0.0
    assert out_10["net_sharpe"] > 0.0
    assert out_10["gross_sharpe"] > out_10["net_sharpe"]
    assert out_10["cost_drag_mean"] > 0.0 and out_10["n_obs"] == n_obs
    assert out_10["p_value"] is None
    # the contract test reuses the same turnover / breakeven arithmetic
    assert out_10["annual_turnover"] == turn["annual_turnover"]
    assert out_10["n_changes"] == turn["n_changes"]
    assert out_10["breakeven_cost_bps"] == breakeven["breakeven_cost_bps"]
    assert out_200["verdict"] == "FAIL"
    assert out_200["net_mean"] < 0.0 and out_200["net_sharpe"] < 0.0
    assert out_200["ci_high"] < 0.0
    assert out_200["cost_drag_mean"] > out_10["cost_drag_mean"]
    # drag is exactly linear in cost_bps on the identical turnover path
    drag_10 = float(out_10["cost_drag_mean"])
    assert abs(float(out_200["cost_drag_mean"]) - 20.0 * drag_10) <= 1e-12
    net_shift = float(out_10["net_mean"]) - 19.0 * drag_10
    assert abs(float(out_200["net_mean"]) - net_shift) <= 1e-12


if __name__ == "__main__":
    run_all_examples()
