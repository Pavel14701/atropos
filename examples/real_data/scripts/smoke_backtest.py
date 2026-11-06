"""Smoke-check the backtest environment on a tiny synthetic world.

Builds a seeded 30-bar OHLC series, fires ONE long signal through
``atropos.backtesting.simulate_signal`` and prints the returned
``r_opt`` / ``r_pess`` / ``exit_idx``.  This is an ENVIRONMENT smoke
test, not a verdict: the only assertions are finiteness/type
contracts, never numeric expectations, so a green run means "the
engine imports and runs" — nothing about the signal's quality.  No
real data is touched; the whole run stays well under a second.

Run from the repository root:

    uv run python examples/real_data/scripts/smoke_backtest.py
"""

from __future__ import annotations

import math

import numpy as np
from atropos.backtesting import CostModel, simulate_signal

N_BARS = 30
SEED = 17
ENTRY_IDX = 5
HOLD_BARS = 10


def build_world(
    n_bars: int = N_BARS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded tiny OHLC world (drift + noise, small wicks)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(0.02 + 0.20 * rng.standard_normal(n_bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.standard_normal(n_bars)) * 0.10 + 0.02
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    return open_, high, low, close


def main() -> int:
    """Replay one signal and print the simulation contract fields."""
    opens, highs, lows, closes = build_world()
    sizes = {arr.size for arr in (opens, highs, lows, closes)}
    assert sizes == {N_BARS}, f"world must be {N_BARS} bars, got {sizes}"
    for name, arr in (
        ("opens", opens),
        ("highs", highs),
        ("lows", lows),
        ("closes", closes),
    ):
        assert np.all(np.isfinite(arr)), f"{name} must be finite"

    fill = float(opens[ENTRY_IDX])
    result = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        ENTRY_IDX,
        "long",
        fill - 1.5,
        fill + 3.0,
        HOLD_BARS,
        1.0,
        cost=CostModel(),
    )

    r_opt = result["r_opt"]
    r_pess = result["r_pess"]
    exit_idx = result["exit_idx"]
    assert isinstance(r_opt, float) and isinstance(r_pess, float), (
        "r_opt/r_pess must be floats"
    )
    assert isinstance(exit_idx, int), "exit_idx must be an int"
    assert -1 <= exit_idx < N_BARS, (
        f"exit_idx {exit_idx} outside [-1, {N_BARS})"
    )
    if exit_idx == -1:
        # no exit inside the hold window: both R values are NaN
        assert math.isnan(r_opt) and math.isnan(r_pess), (
            "exit_idx == -1 must carry NaN R values"
        )
    else:
        assert math.isfinite(r_opt) and math.isfinite(r_pess), (
            "a decided trade must carry finite r_opt/r_pess"
        )

    print(f"r_opt={r_opt:+.4f}R  r_pess={r_pess:+.4f}R  exit_idx={exit_idx}")
    print(
        "smoke_backtest: OK (environment only — no verdict, "
        "no numeric expectations)"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
