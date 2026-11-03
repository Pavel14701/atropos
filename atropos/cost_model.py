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

from atropos.bootstrap_ci import block_bootstrap_ci

__all__ = [
    "annual_turnover",
    "apply_proportional_costs",
    "breakeven_cost_bps",
    "cost_death_test",
    "run_all_examples",
    "turnover_series",
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


def _clean_pair(
    strategy_returns: Any, positions: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Clean gross returns and positions and enforce equal length.

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions or weights behind those
            returns.

    Returns:
        Tuple ``(gross, pos)`` of equal-length 1-D float64 arrays.

    Raises:
        ValueError: if the cleaned arrays differ in length.
    """
    gross = _clean(strategy_returns)
    pos = _clean(positions)
    if gross.size != pos.size:
        raise ValueError(
            "strategy_returns and positions must have the same number "
            "of finite observations "
            f"(got {int(gross.size)} and {int(pos.size)})"
        )
    return gross, pos


def turnover_series(positions: Any) -> np.ndarray:
    """One-sided turnover per period ``|pos[t] - pos[t-1]|``.

    The first element is 0 by definition (no previous position to
    rebalance from); every later element is the absolute position
    change entering period ``t``.

    Args:
        positions: 1-D positions/weights (non-finite entries dropped).

    Returns:
        Turnover series, same length as the cleaned positions.

    Example::

        >>> turnover_series([0.0, 1.0, 1.0, 0.0])    # doctest: +SKIP
        array([0., 1., 0., 1.])
    """
    pos = _clean(positions)
    tvr = np.zeros(pos.size, dtype=float)
    if pos.size > 1:
        tvr[1:] = np.abs(np.diff(pos))
    return tvr


def apply_proportional_costs(
    strategy_returns: Any, positions: Any, cost_bps: float
) -> np.ndarray:
    """Net returns after proportional one-way transaction costs.

    ``net[t] = gross[t] - (cost_bps / 1e4) * turnover[t]`` where
    ``cost_bps`` is the one-way proportional cost per unit of traded
    exposure, in basis points.

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions/weights behind those returns.
        cost_bps: non-negative one-way cost in basis points.

    Returns:
        Net return series, same length as the cleaned inputs.

    Raises:
        ValueError: if ``cost_bps`` is negative or the cleaned inputs
            differ in length.

    Example::

        >>> apply_proportional_costs(                     # +SKIP
        ...     [0.01, 0.02], [0.0, 1.0], cost_bps=10.0)
        array([0.01 , 0.019])
    """
    if float(cost_bps) < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    gross, pos = _clean_pair(strategy_returns, positions)
    tvr = turnover_series(pos)
    return gross - (float(cost_bps) / 1e4) * tvr


def annual_turnover(positions: Any, periods: int = 252) -> dict[str, Any]:
    """Annualized turnover plus rebalance counts.

    Args:
        positions: 1-D positions/weights (non-finite entries dropped).
        periods: periods per year (252 for daily data).

    Returns:
        Dict with ``annual_turnover`` (mean turnover per period scaled
        by ``periods``), ``n_changes`` (count of nonzero turnover
        steps) and ``n_obs`` (number of cleaned observations).

    Example::

        >>> annual_turnover([0.0, 1.0, 1.0, 0.0], periods=10)  # +SKIP
        {'annual_turnover': 5.0, 'n_changes': 2, 'n_obs': 4}
    """
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

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions/weights behind those returns.

    Returns:
        Dict with ``breakeven_cost_bps``; ``inf`` when the gross mean
        is positive but the strategy never trades, NaN when there is
        neither edge nor trading.

    Raises:
        ValueError: if the cleaned inputs differ in length.

    Example::

        >>> breakeven_cost_bps(                           # doctest: +SKIP
        ...     [0.001] * 4, [0.0, 1.0, 1.0, 1.0])
        {'breakeven_cost_bps': 40.0}
    """
    gross, pos = _clean_pair(strategy_returns, positions)
    tvr = turnover_series(pos)
    mean_gross = float(np.mean(gross)) if gross.size else float("nan")
    mean_tvr = float(np.mean(tvr)) if tvr.size else float("nan")
    if mean_tvr == 0.0:
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

    Computes net returns via :func:`apply_proportional_costs`, builds
    a circular moving-block bootstrap CI for the net mean
    (:func:`atropos.bootstrap_ci.block_bootstrap_ci`) and derives the
    annualized net/gross Sharpe ratios.

    Verdict:

    * ``PASS`` - CI lower bound > 0 AND net Sharpe >= min_net_sharpe
    * ``FAIL`` - otherwise; FAIL == COST-DEATH (costs killed the edge)
    * ``INCONCLUSIVE`` - the bootstrap could not run (fewer than 8
      valid observations) or the net series is degenerate (constant)

    Args:
        strategy_returns: per-period gross strategy returns.
        positions: per-period positions/weights behind those returns.
        cost_bps: non-negative one-way cost in basis points.
        periods: periods per year (252 for daily data).
        n_boot: number of bootstrap replicates for the net-mean CI.
        block: block length of the moving-block bootstrap.
        ci_level: confidence level of the CI.
        min_net_sharpe: floor the net annualized Sharpe must clear.
        seed: seed for the bootstrap; fixed seed -> reproducible CI.

    Returns:
        Dict with keys ``stat`` (net mean), ``p_value`` (NaN: the
        percentile bootstrap yields no p-value), ``verdict``,
        ``net_sharpe``, ``gross_sharpe``, ``cost_drag_mean`` (mean
        return lost to costs), ``ci_low``, ``ci_high``, ``n_obs``,
        ``breakeven_cost_bps``, ``annual_turnover``, ``n_changes``.

    Raises:
        ValueError: if ``cost_bps`` is negative or the cleaned inputs
            differ in length.

    Example::

        >>> rng = np.random.default_rng(7)               # doctest: +SKIP
        >>> out = cost_death_test(rng.normal(0.001, 0.01, # doctest: +SKIP
        ...     750), np.ones(750), 10.0, seed=1)
        >>> out["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}  # +SKIP
        True
    """
    gross, pos = _clean_pair(strategy_returns, positions)
    net = apply_proportional_costs(gross, pos, cost_bps)
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
        "stat": float(boot["point_estimate"]),
        "p_value": float("nan"),
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
    breakeven = breakeven_cost_bps(strategy, positions)
    print(f"breakeven_cost_bps: {breakeven['breakeven_cost_bps']:.1f}")
    for cost in (10.0, 200.0):
        out = cost_death_test(
            strategy, positions, cost, n_boot=1000, block=20, seed=11
        )
        print(
            f"cost_death_test({cost:5.1f} bps): verdict={out['verdict']}"
            f" net_mean={out['stat']:+.5f}"
            f" net_sharpe={out['net_sharpe']:+.2f}"
            f" CI=[{out['ci_low']:+.5f}, {out['ci_high']:+.5f}]"
            f" cost_drag={out['cost_drag_mean']:.5f}"
        )


if __name__ == "__main__":
    run_all_examples()
