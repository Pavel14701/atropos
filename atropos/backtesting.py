"""Event-simulator backtest engine: signals to R, slot discipline, metrics.

The engine answers three questions a per-period return backtest cannot:

* what is ONE signal worth (``simulate_signal``: optimistic vs
  pessimistic net R under the taker cost model, plus the maker-entry
  variants ``market_sim`` / ``maker_sim`` / ``maker_entry_sim``);
* which signals become trades when ONE slot is occupied at a time
  (``run_position_state_machine``: fills, force-closes, cooldowns);
* which trades survive a concurrent-slot cap (``run_admission_policy``:
  FCFS vs REPLACE policies over rank-normalized scores).

Per-trade metrics (``trade_curve_stats``, ``per_trade_sharpe``,
``bucketed_sharpe``, ``pooled_stats``) close the loop; they are
DEFINED in :mod:`atropos.backtest_metrics` (single home) and
re-exported here.  They operate on the PER-TRADE R scale and are
deliberately distinct from
``atropos.evaluation_metrics.sharpe_ratio``, which annualizes a
PER-PERIOD return series with a block-bootstrap CI: a per-trade
t-stat of ~10+ corresponds to a per-trade Sharpe of only ~0.3-0.6, so
the two must never be read interchangeably.

The taker cost model defaults are the validated engine constants
(``GEN_SLIP=0.0005, COMM=0.001, GAP=0.25, E_MULT=2.0, X_MULT=2.0``),
parameterized as :class:`CostModel` instead of hardcoded.

Score contract for admission: candidates arriving from DIFFERENT model
groups are compared only on WITHIN-GROUP rank-normalized scores
(:func:`rank_normalize_group`); raw ``predict_proba`` outputs are not
comparable across models (KNN probabilities are noisy and biased,
calibrated GBM probabilities are not).  If a ``CandidateGroup`` frame
is used upstream, pass its ``normalized_scores`` column -- never the
raw ``scores``.

Run ``python -m atropos.backtesting`` for a synthetic demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from atropos._utils import validate_int, validate_number
from atropos.backtest_metrics import (
    bucketed_sharpe,
    per_trade_sharpe,
    pooled_stats,
    trade_curve_stats,
)

__all__ = [
    "DEFAULT_COST",
    "CostModel",
    "bucketed_sharpe",
    "maker_entry_sim",
    "maker_sim",
    "market_sim",
    "per_trade_sharpe",
    "pessimistic_r",
    "pooled_stats",
    "rank_normalize_group",
    "run_admission_policy",
    "run_position_state_machine",
    "simulate_signal",
    "trade_curve_stats",
]

MAKER_FRAC = 0.3
"""Maker entry fee as a fraction of the taker commission."""

HOLD_BARS = 48
"""Default maximum hold in bars (the validated taker-path window)."""

DEFAULT_DELTA_ATR = 0.25
"""Default limit distance into the zone, in ATR units."""

DEFAULT_WAIT_BARS = 4
"""Default number of bars to wait for a limit fill."""

MIN_ZONE_ATR_FRAC = 0.05
"""A limit closer than this fraction of ATR to the stop is unfillable."""

_ADMISSION_POLICIES = ("fcfs", "replace_low", "replace_high")
_TRADE_KEYS = ("ts_entry", "ts_exit", "s", "r_pess")
_SIDES = ("long", "short")
_EXIT_REASONS = ("sl", "tp", "time")
_SIGNAL_COLUMNS = ("decision_idx", "side", "exit_idx")
_OUTPUT_COLUMNS = ("status", "reason", "force_exit_idx", "force_close_r")
_NO_SLOT = -2
"""Busy-until sentinel that is never a real bar index."""


@dataclass(frozen=True)
class CostModel:
    """Taker-path cost constants of the event simulator (all finite).

    The defaults are the validated engine constants; every field can be
    overridden for sensitivity runs (e.g. ``comm=0.0`` for a
    frictionless world).

    Args:
        gen_slip: general slippage as a fraction of price, charged once
            at entry and once at exit.
        comm: taker commission per side as a fraction of price; the
            round trip charges ``2 * comm``.
        gap: adverse gap at a stop exit, in ATR units.
        e_mult: entry-slippage pessimism multiplier.
        x_mult: exit-slippage pessimism multiplier; the extra exit
            penalty is ``(x_mult - 1) * gen_slip``, so values below 1
            would flip that penalty negative.

    Raises:
        ValueError: if any field is not a finite number, ``gen_slip`` /
            ``comm`` / ``gap`` / ``e_mult`` is negative, or ``x_mult``
            is below 1.

    Example::

        >>> cost = CostModel(gen_slip=0.001)
        >>> (cost.gen_slip, cost.comm, cost.gap, cost.e_mult, cost.x_mult)
        (0.001, 0.001, 0.25, 2.0, 2.0)
        >>> CostModel(comm=0.0).comm
        0.0
        >>> CostModel(gen_slip=-1.0)
        Traceback (most recent call last):
            ...
        ValueError: gen_slip must be >= 0, got -1.0
    """

    gen_slip: float = 0.0005
    comm: float = 0.001
    gap: float = 0.25
    e_mult: float = 2.0
    x_mult: float = 2.0

    def __post_init__(self) -> None:
        """Reject non-finite or sign-flipped cost constants loudly."""
        for field_name in ("gen_slip", "comm", "gap", "e_mult"):
            value = validate_number(field_name, getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")
        x_mult = validate_number("x_mult", self.x_mult)
        if x_mult < 1.0:
            raise ValueError(f"x_mult must be >= 1, got {x_mult}")


DEFAULT_COST = CostModel()
"""The validated engine cost model (the default ``cost`` argument)."""


class SimResult(TypedDict):
    """Net R and exit bar of one simulated signal.

    Keys: ``r_opt`` (optimistic net R), ``r_pess`` (pessimistic net R)
    and ``exit_idx`` (exit bar index; ``-1`` = no exit inside the hold
    window, in which case both R values are NaN).
    """

    r_opt: float
    r_pess: float
    exit_idx: int


# ---------------------------------------------------------------------------
# score normalization (the admission seam)
# ---------------------------------------------------------------------------
def rank_normalize_group(scores: Any) -> np.ndarray:
    """Within-group rank normalization to ``[0, 1]`` with average ties.

    Admission compares candidates coming from DIFFERENT model groups,
    so each candidate must carry its WITHIN-GROUP position, not a raw
    cross-model probability (KNN and calibrated GBM probabilities live
    on incomparable scales).  The mapping is
    ``(average_rank - 1) / (count - 1)`` via ``scipy.stats.rankdata``
    with ``method="average"``: tied scores share the AVERAGE of their
    rank positions, so a tie broken for display purposes can never leak
    into the normalized score.

    Edge cases, pinned: a single-candidate group maps to exactly
    ``0.5`` (neutral - real groups have one candidate, and a neutral
    score neither wins nor loses a replacement contest), and an empty
    input maps to an empty output.  This is a different contract from
    ``atropos.ml_ensemble.rank_normalize`` (blend space, which maps
    ``count < 2`` to 0.0); here 0.5 is the admission-neutral value.

    Args:
        scores: 1-D array-like of finite within-group scores.

    Returns:
        1-D float64 array of normalized average ranks (higher score =
        higher value); monotone non-decreasing in ``scores``.

    Raises:
        ValueError: if ``scores`` is not 1-D or contains non-finite
            values.

    Example::

        >>> rank_normalize_group(np.array([3.0, 1.0, 2.0])).tolist()
        [1.0, 0.0, 0.5]
        >>> rank_normalize_group(np.array([5.0, 5.0, 1.0])).tolist()
        [0.75, 0.75, 0.0]
        >>> rank_normalize_group(np.array([7.0])).tolist()
        [0.5]
    """
    flat = np.asarray(scores, dtype=np.float64)
    if flat.ndim != 1:
        raise ValueError(
            f"scores must be 1-D, got ndim={flat.ndim} of shape {flat.shape}"
        )
    count = int(flat.size)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    if count == 1:
        return np.full(1, 0.5, dtype=np.float64)
    if not bool(np.all(np.isfinite(flat))):
        raise ValueError("scores must be finite, got NaN/inf entries")
    ranks = np.asarray(rankdata(flat, method="average"), dtype=np.float64)
    return (ranks - 1.0) / float(count - 1)


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------
def _validated_ohlc(
    opens: Any, highs: Any, lows: Any, closes: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce the four price series to equal-length finite 1-D arrays.

    Raises:
        ValueError: if any series is not 1-D, the lengths differ, or
            any entry is NaN/inf.
    """
    arrays: list[np.ndarray] = []
    for name, values in (
        ("opens", opens),
        ("highs", highs),
        ("lows", lows),
        ("closes", closes),
    ):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D, got ndim={arr.ndim} of shape"
                f" {arr.shape}"
            )
        if arr.size and not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{name} must be finite, got NaN/inf entries")
        arrays.append(arr)
    sizes = {arr.size for arr in arrays}
    if len(sizes) != 1:
        raise ValueError(
            "opens/highs/lows/closes must have the same length, got"
            f" {[arr.size for arr in arrays]}"
        )
    open_arr, high_arr, low_arr, close_arr = arrays
    return open_arr, high_arr, low_arr, close_arr


def _validated_direction(side: str) -> float:
    """Return +1.0 for ``"long"`` and -1.0 for ``"short"``.

    Raises:
        ValueError: if ``side`` is neither literal.
    """
    if side == "long":
        return 1.0
    if side == "short":
        return -1.0
    raise ValueError(f"side must be 'long' or 'short', got {side!r}")


def _validated_entry_index(entry_idx: int, n_bars: int) -> int:
    """Validate the entry bar index against the price series length.

    Raises:
        ValueError: if ``entry_idx`` is not an integer >= 0 or points
            past the data.
    """
    idx = validate_int("entry_idx", entry_idx, minimum=0)
    if idx >= n_bars:
        raise ValueError(
            f"entry_idx must be < len(opens)={n_bars}, got {idx}"
        )
    return idx


# ---------------------------------------------------------------------------
# one-signal simulation (the exit-scan core)
# ---------------------------------------------------------------------------
def _exit_scan(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_idx: int,
    sign: float,
    *,
    fill: float,
    stop_level: float,
    target_level: float,
    denom: float,
    atr_value: float,
    hold_bars: int,
    cost: CostModel,
    maker_frac: float | None,
) -> SimResult:
    """Scan bars from the entry for the stop / target / time exit.

    Single home of the engine's exit mechanics, shared by the taker
    sim and the maker sim:

    * SL-first pessimism inside a bar (a bar touching BOTH levels is
      resolved as a stop);
    * stop exits pessimize by the extra exit slippage
      ``(x_mult - 1) * gen_slip`` PLUS the adverse ``gap * atr`` term;
    * target exits pessimize by the entry slippage ``e_mult *
      gen_slip`` only;
    * a time exit (close of the final hold bar, slipped) pessimizes by
      entry + extra exit slippage;
    * an entry bar opening beyond the stop is an immediate market
      scratch (never a win in gap-distance units).

    ``maker_frac=None`` charges the taker entry fee ``2 * comm``; a
    fraction charges the reduced maker fee ``comm * (1 + frac)``.  On
    the maker path ``r_opt`` is undefined and reported as NaN.
    """
    if maker_frac is None:
        cost_r = (2 * cost.comm * fill + cost.gen_slip * fill) / denom
        r_opt_defined = True
    else:
        cost_r = (
            (cost.comm * (1 + maker_frac) + cost.gen_slip) * fill
        ) / denom
        r_opt_defined = False
    entry_slip_r = cost.e_mult * cost.gen_slip * fill / denom
    gap_through = (sign > 0 and fill < stop_level) or (
        sign < 0 and fill > stop_level
    )
    if gap_through:
        # the stop order fires immediately at market: a scratch
        xtr = (cost.x_mult - 1) * cost.gen_slip * abs(fill) / denom
        r_opt = float(-cost_r) if r_opt_defined else float("nan")
        return SimResult(
            r_opt=r_opt,
            r_pess=float(-cost_r - entry_slip_r - xtr),
            exit_idx=int(entry_idx),
        )
    n_bars = int(closes.size)
    last = min(n_bars - 1, entry_idx + hold_bars - 1)
    for bars_held, bar_idx in enumerate(range(entry_idx, last + 1)):
        stop_hit = (
            lows[bar_idx] <= stop_level
            if sign > 0
            else highs[bar_idx] >= stop_level
        )
        target_hit = (
            highs[bar_idx] >= target_level
            if sign > 0
            else lows[bar_idx] <= target_level
        )
        if stop_hit:
            net_r = (
                sign * (stop_level * (1 - sign * cost.gen_slip) - fill)
            ) / denom - cost_r
            gap_r = cost.gap * atr_value / denom
            xtr = (cost.x_mult - 1) * cost.gen_slip * abs(stop_level) / denom
            r_opt = float(net_r) if r_opt_defined else float("nan")
            return SimResult(
                r_opt=r_opt,
                r_pess=float(net_r - entry_slip_r - xtr - gap_r),
                exit_idx=int(bar_idx),
            )
        if target_hit:
            net_r = sign * (target_level - fill) / denom - cost_r
            r_opt = float(net_r) if r_opt_defined else float("nan")
            return SimResult(
                r_opt=r_opt,
                r_pess=float(net_r - entry_slip_r),
                exit_idx=int(bar_idx),
            )
        if bars_held == hold_bars - 1:
            exit_px = closes[bar_idx] * (1 - sign * cost.gen_slip)
            net_r = sign * (exit_px - fill) / denom - cost_r
            xtr = (cost.x_mult - 1) * cost.gen_slip * abs(exit_px) / denom
            r_opt = float(net_r) if r_opt_defined else float("nan")
            return SimResult(
                r_opt=r_opt,
                r_pess=float(net_r - entry_slip_r - xtr),
                exit_idx=int(bar_idx),
            )
    return SimResult(r_opt=float("nan"), r_pess=float("nan"), exit_idx=-1)


# ---------------------------------------------------------------------------
# public sims
# ---------------------------------------------------------------------------
def simulate_signal(  # noqa: PLR0917  (donor OHLC-world signature)
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    entry_idx: int,
    side: str,
    stop_price: float,
    target_price: float,
    hold_bars: int,
    atr_value: float,
    *,
    risk_ref: float | None = None,
    cost: CostModel = DEFAULT_COST,
) -> SimResult:
    """Simulate one signal from its entry bar under the taker model.

    The entry fills at ``opens[entry_idx]``; the scan then walks
    forward at most ``hold_bars`` bars and resolves the first stop /
    target / time exit.  Returns ``{"r_opt", "r_pess", "exit_idx"}``:
    optimistic and pessimistic net R and the exit bar.

    Contracts preserved from the engine:

    * SL-first pessimism inside a bar: a bar touching BOTH the stop
      and the target is resolved as a stop.
    * A stop exit pessimizes by the extra exit slippage
      ``(x_mult - 1) * gen_slip`` PLUS the adverse ``gap * atr`` term;
      a target exit pessimizes by the entry slippage ``e_mult *
      gen_slip`` only; a time exit (close of the final hold bar,
      slipped) pessimizes by entry + extra exit slippage.
    * Gap-through-stop at ENTRY: when the entry bar already OPENS
      beyond the stop, live the stop order fires immediately at
      market - a scratch, never a +1R win in gap-distance units.  Both
      R values are the scratch (the pessimistic one adds the
      entry/exit pessimism terms).
    * ``risk_ref`` is the intended risk distance used as the R
      denominator (a panel ``risk_unit``); it defaults to the actual
      ``|entry fill - stop|``.  A non-positive or absent ``risk_ref``
      falls back to that actual distance.
    * No exit inside the hold window, or ``risk <= 0``: both R values
      are NaN and ``exit_idx`` is ``-1``.

    Args:
        opens: 1-D finite open prices.
        highs: 1-D finite high prices, same length as ``opens``.
        lows: 1-D finite low prices, same length as ``opens``.
        closes: 1-D finite close prices, same length as ``opens``.
        entry_idx: entry bar index (the fill bar), ``0 <= idx < len``.
        side: ``"long"`` or ``"short"``.
        stop_price: stop-loss price level.
        target_price: take-profit price level.
        hold_bars: maximum bars to hold (time exit on the last one);
            must be >= 1.
        atr_value: ATR in price units at entry (scales the stop-gap
            pessimism term).
        risk_ref: optional intended risk distance for the R
            denominator; must be finite and positive when given.
        cost: cost model; defaults to :data:`DEFAULT_COST`.

    Returns:
        ``SimResult`` dict with ``r_opt``, ``r_pess`` and ``exit_idx``.

    Raises:
        ValueError: if the price series are not 1-D finite arrays of
            equal length, ``entry_idx`` / ``hold_bars`` are not valid
            integers in range, ``side`` is not ``"long"``/``"short"``,
            or a scalar parameter is not finite.

    Example::

        >>> opens = [100.0, 100.0, 100.0, 100.0]
        >>> highs = [100.0, 100.0, 102.4, 100.0]
        >>> lows = [100.0, 100.0, 100.0, 100.0]
        >>> closes = [100.0, 100.0, 101.0, 100.0]
        >>> out = simulate_signal(
        ...     opens, highs, lows, closes, 0, "long", 99.0, 102.0,
        ...     4, 1.0,
        ... )
        >>> (round(out["r_opt"], 6), round(out["r_pess"], 6))
        (1.75, 1.65)
        >>> # data ends one bar after entry: the hold window is cut
        >>> # short, so no exit happens at all (the time exit needs the
        >>> # full window; a lone final bar cannot time-exit)
        >>> missed = simulate_signal(
        ...     opens, highs, lows, closes, 3, "long", 90.0, 130.0,
        ...     2, 1.0,
        ... )
        >>> (math.isnan(missed["r_opt"]), missed["exit_idx"])
        (True, -1)
    """
    open_arr, high_arr, low_arr, close_arr = _validated_ohlc(
        opens, highs, lows, closes
    )
    sign = _validated_direction(side)
    idx = _validated_entry_index(entry_idx, int(open_arr.size))
    stop_level = validate_number("stop_price", stop_price)
    target_level = validate_number("target_price", target_price)
    bars = validate_int("hold_bars", hold_bars, minimum=1)
    atr = validate_number("atr_value", atr_value)
    fill = float(open_arr[idx])
    denom = abs(fill - stop_level)
    if denom <= 0.0:
        return SimResult(r_opt=float("nan"), r_pess=float("nan"), exit_idx=-1)
    if risk_ref is not None and validate_number("risk_ref", risk_ref) > 0.0:
        denom = float(risk_ref)
    return _exit_scan(
        high_arr,
        low_arr,
        close_arr,
        idx,
        sign,
        fill=fill,
        stop_level=stop_level,
        target_level=target_level,
        denom=denom,
        atr_value=atr,
        hold_bars=bars,
        cost=cost,
        maker_frac=None,
    )


def market_sim(  # noqa: PLR0917  (donor OHLC-world signature)
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    entry_idx: int,
    side: str,
    stop_price: float,
    target_price: float,
    atr_value: float,
    *,
    hold_bars: int = HOLD_BARS,
    cost: CostModel = DEFAULT_COST,
) -> float:
    """Baseline always-market entry: fill at ``opens[entry_idx]``.

    This is the pessimistic R of :func:`simulate_signal` under the
    default hold window - the engine's baseline every execution
    variant is measured against.  A next-bar open beyond the stop is
    an immediate market scratch, never a win (same contract as
    :func:`simulate_signal`).

    Args:
        opens: 1-D finite open prices.
        highs: 1-D finite high prices, same length as ``opens``.
        lows: 1-D finite low prices, same length as ``opens``.
        closes: 1-D finite close prices, same length as ``opens``.
        entry_idx: entry bar index (the fill bar).
        side: ``"long"`` or ``"short"``.
        stop_price: stop-loss price level.
        target_price: take-profit price level.
        atr_value: ATR in price units at entry.
        hold_bars: maximum bars to hold; defaults to
            :data:`HOLD_BARS` (the validated engine window).
        cost: cost model; defaults to :data:`DEFAULT_COST`.

    Returns:
        Pessimistic net R of the trade; NaN when there is no exit
        inside the hold window or the risk distance is non-positive.

    Raises:
        ValueError: on the same invalid inputs as
            :func:`simulate_signal`.

    Example::

        >>> opens = [100.0] * 12
        >>> highs = [100.0] * 12
        >>> highs[10] = 102.0
        >>> lows = [100.0] * 12
        >>> closes = [100.0] * 12
        >>> round(
        ...     market_sim(opens, highs, lows, closes, 0, "long",
        ...                99.0, 102.0, 1.0),
        ...     6,
        ... )
        1.65
    """
    result = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        entry_idx,
        side,
        stop_price,
        target_price,
        hold_bars,
        atr_value,
        cost=cost,
    )
    return result["r_pess"]


def maker_sim(  # noqa: PLR0917  (donor OHLC-world signature)
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    entry_idx: int,
    side: str,
    stop_price: float,
    target_price: float,
    limit_price: float,
    atr_value: float,
    maker_frac: float = MAKER_FRAC,
    *,
    hold_bars: int = HOLD_BARS,
    cost: CostModel = DEFAULT_COST,
) -> float:
    """Pessimistic R for a KNOWN limit fill at ``limit_price``.

    The position opens at bar ``entry_idx`` at the limit price (the
    caller established the fill, e.g. via :func:`maker_entry_sim`).
    Mechanics mirror the taker sim with two execution differences:

    * the entry fee is ``comm * (1 + maker_frac)`` instead of
      ``2 * comm`` (maker entry, taker exit);
    * the outcome is computed from the ACTUAL fill price, so a better
      entry WIDENS the stop distance: each win is smaller in R but
      stop-outs become rarer.

    Gap-through-stop guard: a limit filled beyond the stop fires the
    stop immediately at market - a scratch net of costs, never a
    ~+1R win measured in gap-distance units.

    Args:
        opens: 1-D finite open prices.
        highs: 1-D finite high prices, same length as ``opens``.
        lows: 1-D finite low prices, same length as ``opens``.
        closes: 1-D finite close prices, same length as ``opens``.
        entry_idx: bar index at which the limit filled.
        side: ``"long"`` or ``"short"``.
        stop_price: stop-loss price level.
        target_price: take-profit price level.
        limit_price: the actual limit fill price.
        atr_value: ATR in price units at entry.
        maker_frac: maker entry fee as a fraction of the taker
            commission; must be >= 0 (1.0 == full taker round trip).
        hold_bars: maximum bars to hold; defaults to
            :data:`HOLD_BARS`.
        cost: cost model; defaults to :data:`DEFAULT_COST`.

    Returns:
        Pessimistic net R from the actual fill price; NaN when the
        risk distance is non-positive or no exit happens inside the
        hold window.

    Raises:
        ValueError: on invalid inputs (non-finite prices, bad side,
            negative ``maker_frac``, out-of-range indices).

    Example::

        >>> opens = [100.0] * 12
        >>> highs = [100.0] * 12
        >>> lows = [100.0] * 12
        >>> lows[2] = 99.5
        >>> highs[10] = 102.0
        >>> closes = [100.0] * 12
        >>> round(
        ...     maker_sim(opens, highs, lows, closes, 2, "long",
        ...               99.0, 102.0, 99.5, 1.0),
        ...     6,
        ... )
        4.4428
    """
    open_arr, high_arr, low_arr, close_arr = _validated_ohlc(
        opens, highs, lows, closes
    )
    sign = _validated_direction(side)
    idx = _validated_entry_index(entry_idx, int(open_arr.size))
    stop_level = validate_number("stop_price", stop_price)
    target_level = validate_number("target_price", target_price)
    limit = validate_number("limit_price", limit_price)
    atr = validate_number("atr_value", atr_value)
    frac = validate_number("maker_frac", maker_frac)
    if frac < 0.0:
        raise ValueError(f"maker_frac must be >= 0, got {maker_frac}")
    bars = validate_int("hold_bars", hold_bars, minimum=1)
    denom = abs(limit - stop_level)
    if denom <= 0.0:
        return float("nan")
    result = _exit_scan(
        high_arr,
        low_arr,
        close_arr,
        idx,
        sign,
        fill=limit,
        stop_level=stop_level,
        target_level=target_level,
        denom=denom,
        atr_value=atr,
        hold_bars=bars,
        cost=cost,
        maker_frac=frac,
    )
    return result["r_pess"]


def maker_entry_sim(  # noqa: PLR0917  (donor OHLC-world signature)
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    entry_idx: int,
    side: str,
    stop_price: float,
    target_price: float,
    fill_price: float,
    atr_value: float,
    *,
    delta_atr: float = DEFAULT_DELTA_ATR,
    wait_bars: int = DEFAULT_WAIT_BARS,
    maker_frac: float = MAKER_FRAC,
    hold_bars: int = HOLD_BARS,
    cost: CostModel = DEFAULT_COST,
) -> float:
    """Maker-or-skip entry: place a limit into the zone, skip if missed.

    The limit is placed ``delta_atr * ATR`` INTO the zone from the
    market-fill reference price (``fill_price``, typically the
    next-bar open).  The fill is deterministic in the price path: the
    order fills on the FIRST bar (within ``wait_bars``) whose range
    trades through the limit; a missed fill means a missed trade and
    contributes exactly 0 (maker-or-skip).  A limit closer than
    ``MIN_ZONE_ATR_FRAC * ATR`` to the stop is unfillable (NaN) - the
    zone would put the entry at or beyond the stop.

    Determinism note: fills depend ONLY on the price path, so the
    same inputs reproduce the same outcome bit-for-bit (stronger than
    a seeded RNG draw).

    Args:
        opens: 1-D finite open prices.
        highs: 1-D finite high prices, same length as ``opens``.
        lows: 1-D finite low prices, same length as ``opens``.
        closes: 1-D finite close prices, same length as ``opens``.
        entry_idx: signal decision bar; the limit may fill from this
            bar onward.
        side: ``"long"`` or ``"short"``.
        stop_price: stop-loss price level.
        target_price: take-profit price level.
        fill_price: market-fill reference price (the next-bar open).
        atr_value: ATR in price units (scales the limit distance).
        delta_atr: limit distance into the zone, in ATR units; must
            be >= 0.
        wait_bars: bars to wait for a fill; must be >= 1.
        maker_frac: maker entry fee as a fraction of the taker
            commission; must be >= 0.
        hold_bars: maximum bars to hold after a fill; defaults to
            :data:`HOLD_BARS`.
        cost: cost model; defaults to :data:`DEFAULT_COST`.

    Returns:
        Pessimistic net R of the filled trade, exactly ``0.0`` for a
        missed fill (maker-or-skip), NaN when the zone is unfillable
        or the filled position has no exit inside the hold window.

    Raises:
        ValueError: on invalid inputs (same rules as
            :func:`maker_sim`).

    Example::

        >>> opens = [100.0] * 12
        >>> highs = [100.0] * 12
        >>> lows = [100.0] * 12
        >>> lows[2] = 99.5
        >>> highs[10] = 102.0
        >>> closes = [100.0] * 12
        >>> round(
        ...     maker_entry_sim(opens, highs, lows, closes, 0, "long",
        ...                     99.0, 102.0, 100.0, 1.0, delta_atr=0.5),
        ...     6,
        ... )
        4.4428
        >>> maker_entry_sim(opens, highs, lows, closes, 0, "long",
        ...                 99.0, 102.0, 100.0, 1.0, delta_atr=0.5,
        ...                 wait_bars=1)
        0.0
    """
    open_arr, high_arr, low_arr, close_arr = _validated_ohlc(
        opens, highs, lows, closes
    )
    sign = _validated_direction(side)
    idx = _validated_entry_index(entry_idx, int(open_arr.size))
    stop_level = validate_number("stop_price", stop_price)
    target_level = validate_number("target_price", target_price)
    reference = validate_number("fill_price", fill_price)
    atr = validate_number("atr_value", atr_value)
    delta = validate_number("delta_atr", delta_atr)
    if delta < 0.0:
        raise ValueError(f"delta_atr must be >= 0, got {delta_atr}")
    bars_wait = validate_int("wait_bars", wait_bars, minimum=1)
    limit = reference - sign * delta * atr
    zone_width = sign * (limit - stop_level)
    if zone_width <= MIN_ZONE_ATR_FRAC * atr:
        return float("nan")
    n_bars = int(open_arr.size)
    for bar_idx in range(idx, min(idx + bars_wait, n_bars)):
        touched = (
            low_arr[bar_idx] <= limit
            if sign > 0
            else high_arr[bar_idx] >= limit
        )
        if touched:
            return maker_sim(
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                bar_idx,
                side,
                stop_level,
                target_level,
                limit,
                atr,
                maker_frac=maker_frac,
                hold_bars=hold_bars,
                cost=cost,
            )
    return 0.0


# ---------------------------------------------------------------------------
# per-row pessimistic R (labeled panel rows)
# ---------------------------------------------------------------------------
def pessimistic_r(
    *,
    risk_unit: float,
    fill_price: float,
    sl_price: float,
    exit_reason: str,
    exit_price: float,
    atr_i: float,
    r_net: float,
    cost: CostModel = DEFAULT_COST,
) -> float:
    """Pessimistic net R of one labeled panel row (taker cost model).

    The pessimism stack mirrors :func:`simulate_signal`: the entry
    slippage ``e_mult * gen_slip`` is always charged; a stop exit
    adds the extra exit slippage ``(x_mult - 1) * gen_slip`` on the
    stop price plus the adverse ``gap * atr`` term; a time exit adds
    the extra exit slippage on the exit price; a target exit carries
    no penalty beyond the entry slippage.

    Args:
        risk_unit: intended risk distance (the R denominator); must
            be > 0.
        fill_price: entry fill price of the trade.
        sl_price: stop-loss price level (used by ``"sl"`` exits).
        exit_reason: ``"sl"``, ``"tp"`` or ``"time"``.
        exit_price: exit price (used by ``"time"`` exits).
        atr_i: ATR in price units at the trade's entry bar.
        r_net: isolated net R of the row (the optimistic bookkeeping).
        cost: cost model; defaults to :data:`DEFAULT_COST`.

    Returns:
        ``r_net`` minus the pessimism stack.

    Raises:
        ValueError: if ``risk_unit`` is not a finite positive number,
            any price input is not finite, or ``exit_reason`` is not
            one of ``"sl"`` / ``"tp"`` / ``"time"``.

    Example::

        >>> round(
        ...     pessimistic_r(
        ...         risk_unit=1.0, fill_price=100.0, sl_price=99.0,
        ...         exit_reason="sl", exit_price=99.0, atr_i=0.4,
        ...         r_net=2.0,
        ...     ),
        ...     6,
        ... )
        1.7505
    """
    risk = validate_number("risk_unit", risk_unit)
    if risk <= 0.0:
        raise ValueError(f"risk_unit must be > 0, got {risk_unit}")
    fill = validate_number("fill_price", fill_price)
    sl_level = validate_number("sl_price", sl_price)
    exit_px = validate_number("exit_price", exit_price)
    atr_value = validate_number("atr_i", atr_i)
    net = validate_number("r_net", r_net)
    if exit_reason not in _EXIT_REASONS:
        raise ValueError(
            f"exit_reason must be one of {_EXIT_REASONS}, got {exit_reason!r}"
        )
    penalty = cost.e_mult * cost.gen_slip * fill / risk
    if exit_reason == "sl":
        penalty += (cost.x_mult - 1) * cost.gen_slip * abs(sl_level) / risk
        penalty += cost.gap * atr_value / risk
    elif exit_reason == "time":
        penalty += (cost.x_mult - 1) * cost.gen_slip * abs(exit_px) / risk
    return float(net - penalty)


# ---------------------------------------------------------------------------
# position state machine
# ---------------------------------------------------------------------------
def _validated_signal_records(
    signals_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate the signal frame and normalize it to record dicts.

    Every row must carry ``decision_idx`` (integer >= 0), ``side``
    (``"long"``/``"short"``) and ``exit_idx`` (integer, ``-1`` for
    open-ended).  The generator rule (fill at ``decision_idx + 1``)
    requires ``exit_idx >= decision_idx + 1`` whenever it is not
    ``-1``.  An optional ``priority`` column must be finite.

    Raises:
        ValueError: on any contract violation, naming the row.
    """
    missing = [
        column
        for column in _SIGNAL_COLUMNS
        if column not in signals_df.columns
    ]
    if missing:
        raise ValueError(f"signals_df is missing required columns: {missing}")
    has_priority = "priority" in signals_df.columns
    records: list[dict[str, Any]] = []
    for position, (_, raw_row) in enumerate(signals_df.iterrows()):
        decision = validate_int(
            "decision_idx", raw_row["decision_idx"], minimum=0
        )
        side = str(raw_row["side"])
        if side not in _SIDES:
            raise ValueError(
                f"signals row {position}: side must be 'long' or 'short',"
                f" got {raw_row['side']!r}"
            )
        exit_idx = validate_int("exit_idx", raw_row["exit_idx"], minimum=-1)
        if exit_idx != -1 and exit_idx < decision + 1:
            raise ValueError(
                f"signals row {position}: exit_idx={exit_idx} violates the"
                " generator rule (fill at decision_idx + 1, so exit_idx"
                f" must be >= {decision + 1} or exactly -1)"
            )
        priority = 0.0
        if has_priority:
            priority = validate_number("priority", raw_row["priority"])
        records.append(
            {
                "position": position,
                "decision_idx": decision,
                "side": side,
                "exit_idx": exit_idx,
                "priority": priority,
                "row": raw_row.to_dict(),
            }
        )
    return records


def _reconstructed_force_close_r(
    open_record: dict[str, Any],
    close_bar: int,
    *,
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    cost: CostModel,
) -> float:
    """Recompute the open trade's pessimistic P&L up to ``close_bar``.

    Runs :func:`simulate_signal` from the ORIGINAL entry (fill at
    ``decision_idx + 1``, same stop/target) with the hold window
    truncated so the time exit lands on ``close_bar``: if the stop or
    target was hit earlier the scan returns that earlier exit, which
    is the truthful path.  The open row may carry ``atr_entry`` (gap
    penalty; defaults to 0.0 when absent) and ``risk_ref`` (R
    denominator; optional).

    A force-close on a bar BEFORE the fill (``close_bar < decision_idx
    + 1``) means the trade never existed at that point: exactly 0.0.

    Raises:
        ValueError: if the price series were not provided, or a
            stop/target/atr/risk value is invalid.
    """
    if opens is None or highs is None or lows is None or closes is None:
        raise ValueError(
            "allow_reverse=True needs the opens/highs/lows/closes price"
            " series to reconstruct the force-closed trade's P&L; pass"
            " them as keyword arguments"
        )
    fill_idx = open_record["decision_idx"] + 1
    if close_bar < fill_idx:
        return 0.0
    row = open_record["row"]
    stop_level = validate_number("stop_price", row["stop_price"])
    target_level = validate_number("target_price", row["target_price"])
    atr_raw = row.get("atr_entry")
    atr_value = (
        0.0 if atr_raw is None else validate_number("atr_entry", atr_raw)
    )
    risk_raw = row.get("risk_ref")
    risk_ref = (
        None if risk_raw is None else validate_number("risk_ref", risk_raw)
    )
    result = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        fill_idx,
        open_record["side"],
        stop_level,
        target_level,
        close_bar - fill_idx + 1,
        atr_value,
        risk_ref=risk_ref,
        cost=cost,
    )
    return result["r_pess"]


def run_position_state_machine(
    signals_df: pd.DataFrame,
    *,
    allow_reverse: bool = False,
    same_bar_reentry: bool = False,
    cooldown: int = 0,
    opens: Any = None,
    highs: Any = None,
    lows: Any = None,
    closes: Any = None,
    cost: CostModel = DEFAULT_COST,
) -> pd.DataFrame:
    """Decide which candidate signals become actual trades.

    The dataset builder simulates every candidate in isolation; live,
    one slot is occupied at a time.  This machine walks the signals
    chronologically and labels each one ``"taken"`` or ``"skipped"``.
    Transition specification (documented decisions):

    * a signal decided at bar ``d`` fills at ``d + 1`` (generator
      rule; validated on the input frame);
    * while a position is open (until its ``exit_idx``), ALL new
      signals are skipped (reason ``"in_position"``), except under
      ``allow_reverse``: an opposite-side signal at bar ``d < exit_idx``
      force-closes the open trade at bar ``d`` - its row is flagged
      with ``force_exit_idx = d`` AND ``force_close_r``, the P&L
      RECOMPUTED to bar ``d`` via :func:`simulate_signal` from the
      original entry with the same stop/target (the row's isolated
      ``r_pess``/``exit_idx`` stay STALE; read ``force_close_r``
      instead), then the new side opens at ``d + 1``;
    * a same-bar exit + signal (decision bar == exit bar) is skipped
      (``"same_bar_after_exit"``) unless ``same_bar_reentry`` - the
      SL-first pessimism rule means the position is still held during
      that bar; with the flag set the signal TAKES the slot (with
      ``cooldown > 0`` the cooldown still applies on the exit bar);
    * after an exit, ``cooldown`` further decision bars are blocked
      (``"cooldown"``; ``cooldown=0`` keeps equivalence with the
      isolated simulator);
    * several signals on one bar: highest ``priority`` wins, ties
      broken by side then stable input order (deterministic);
    * a trade whose ``exit_idx == -1`` (data ended while open) holds
      the slot until the end of data (later signals get
      ``"data_ended_open"``).

    Args:
        signals_df: candidate signals; required columns
            ``decision_idx`` (int >= 0), ``side`` ("long"/"short"),
            ``exit_idx`` (int >= decision_idx + 1, or exactly -1);
            optional ``priority`` (finite; default 0.0).  For
            force-close reconstruction the open rows also need
            ``stop_price``/``target_price`` (required under
            ``allow_reverse``) and optionally ``atr_entry``/``risk_ref``.
        allow_reverse: let an opposite-side signal force-close the
            open trade at its decision bar.
        same_bar_reentry: allow a signal decided on the exit bar to
            open the next position.
        cooldown: decision bars blocked after an exit; must be >= 0.
        opens: 1-D finite open prices; REQUIRED for reconstruction
            when a force-close actually fires.
        highs: 1-D finite high prices (force-close reconstruction).
        lows: 1-D finite low prices (force-close reconstruction).
        closes: 1-D finite close prices (force-close reconstruction).
        cost: cost model for the ``force_close_r`` reconstruction;
            defaults to :data:`DEFAULT_COST`.

    Returns:
        DataFrame of ALL signals in processed (chronological,
        priority-sorted) order: the input columns plus ``status``
        ("taken"/"skipped"), ``reason`` (empty string for taken rows;
        one of ``"in_position"``, ``"same_bar_after_exit"``,
        ``"cooldown"``, ``"data_ended_open"`` for skipped rows),
        ``force_exit_idx`` (NaN unless the row was force-closed) and
        ``force_close_r`` (NaN unless the row was force-closed).

    Raises:
        ValueError: if required columns are missing, a row violates
            the generator rule, ``side``/``priority`` values are
            invalid, ``allow_reverse`` is set without the
            ``stop_price``/``target_price`` columns, a force-close
            fires without the price series, or ``cooldown`` is not an
            integer >= 0.

    Example::

        >>> signals = pd.DataFrame(
        ...     {
        ...         "decision_idx": [3, 5, 7],
        ...         "side": ["long", "long", "long"],
        ...         "exit_idx": [10, 12, 14],
        ...     }
        ... )
        >>> replay = run_position_state_machine(signals)
        >>> replay["status"].tolist()
        ['taken', 'skipped', 'skipped']
        >>> replay["reason"].tolist()
        ['', 'in_position', 'in_position']
    """
    cooldown_bars = validate_int("cooldown", cooldown, minimum=0)
    if allow_reverse:
        for name in ("stop_price", "target_price"):
            if name not in signals_df.columns:
                raise ValueError(
                    f"allow_reverse=True requires the {name!r} column on"
                    " signals_df for the force-close reconstruction"
                )
    records = _validated_signal_records(signals_df)
    order = sorted(
        records,
        key=lambda rec: (
            rec["decision_idx"],
            -rec["priority"],
            rec["side"],
            rec["position"],
        ),
    )
    out_rows = _walk_signal_transitions(
        order,
        allow_reverse=allow_reverse,
        same_bar_reentry=same_bar_reentry,
        cooldown_bars=cooldown_bars,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        cost=cost,
    )
    if not out_rows:
        return pd.DataFrame(
            columns=list(signals_df.columns) + list(_OUTPUT_COLUMNS)
        )
    return pd.DataFrame(out_rows)


def _walk_signal_transitions(
    order: list[dict[str, Any]],
    *,
    allow_reverse: bool,
    same_bar_reentry: bool,
    cooldown_bars: int,
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any,
    cost: CostModel,
) -> list[dict[str, Any]]:
    """Walk the ordered records and label each one taken/skipped.

    One iteration of the documented transition specification; see
    :func:`run_position_state_machine` for the contract.  Returns the
    output row dicts in processed order.

    Raises:
        ValueError: when a force-close fires without the price series.
    """
    out_rows: list[dict[str, Any]] = []
    busy_until = _NO_SLOT
    busy_side: str | None = None
    open_ended = False
    open_record: dict[str, Any] | None = None
    open_row: dict[str, Any] | None = None
    for record in order:
        out_row: dict[str, Any] = dict(record["row"])
        out_row["status"] = "skipped"
        out_row["reason"] = ""
        out_row["force_exit_idx"] = float("nan")
        out_row["force_close_r"] = float("nan")
        decision = record["decision_idx"]
        if open_ended:
            out_row["reason"] = "data_ended_open"
            out_rows.append(out_row)
            continue
        same_bar_reentry_ok = decision == busy_until and same_bar_reentry
        if decision == busy_until and not same_bar_reentry:
            out_row["reason"] = "same_bar_after_exit"
            out_rows.append(out_row)
            continue
        if decision <= busy_until and not same_bar_reentry_ok:
            reverse = (
                busy_side is not None
                and allow_reverse
                and record["side"] != busy_side
            )
            if reverse and open_record is not None and open_row is not None:
                # force-close the open trade at bar d and RECOMPUTE its
                # P&L to d (its isolated r_pess/exit_idx are stale)
                open_row["force_exit_idx"] = float(decision)
                open_row["force_close_r"] = _reconstructed_force_close_r(
                    open_record,
                    decision,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    cost=cost,
                )
                out_row["status"] = "taken"
                out_rows.append(out_row)
                open_record = record
                open_row = out_row
                busy_until = record["exit_idx"]
                busy_side = record["side"]
                open_ended = busy_until < 0
                continue
            out_row["reason"] = "in_position"
            out_rows.append(out_row)
            continue
        if decision < busy_until + cooldown_bars:
            out_row["reason"] = "cooldown"
            out_rows.append(out_row)
            continue
        out_row["status"] = "taken"
        out_rows.append(out_row)
        open_record = record
        open_row = out_row
        busy_until = record["exit_idx"]
        busy_side = record["side"]
        open_ended = busy_until < 0
    return out_rows


# ---------------------------------------------------------------------------
# admission policies
# ---------------------------------------------------------------------------
def _validated_trade(trade: Any, position: int) -> dict[str, Any]:
    """Validate one trade dict of the admission stream.

    Required keys: ``ts_entry``, ``ts_exit``, ``s`` (score),
    ``r_pess``.  Optional: ``candidate_id`` (int >= 0), which becomes
    the score-tie secondary key.  Returns a normalized copy; the
    input dict is never mutated.

    Raises:
        ValueError: on any missing key or non-finite value.
    """
    if not isinstance(trade, dict):
        raise ValueError(
            f"trades[{position}] must be a dict, got {type(trade).__name__}"
        )
    for key in _TRADE_KEYS:
        if key not in trade:
            raise ValueError(
                f"trades[{position}] is missing required key {key!r}"
            )
    entry_ts = validate_number("ts_entry", trade["ts_entry"])
    exit_ts = validate_number("ts_exit", trade["ts_exit"])
    if exit_ts < entry_ts:
        raise ValueError(
            f"trades[{position}]: ts_exit must be >= ts_entry, got"
            f" {exit_ts} < {entry_ts}"
        )
    score = validate_number("s", trade["s"])
    r_pess_value = validate_number("r_pess", trade["r_pess"])
    raw_id = trade.get("candidate_id")
    candidate_id = (
        None
        if raw_id is None
        else validate_int("candidate_id", raw_id, minimum=0)
    )
    # score-tie secondary key: candidate_id (ascending) when present,
    # else the stream arrival position - both stable and deterministic
    tie_key = position if candidate_id is None else candidate_id
    return {
        "ts_entry": entry_ts,
        "ts_exit": exit_ts,
        "score": score,
        "r_pess": r_pess_value,
        "candidate_id": candidate_id,
        "tie_key": tie_key,
        "position": position,
        "row": dict(trade),
    }


def run_admission_policy(
    trades: list[dict[str, Any]],
    *,
    cap: int,
    policy: str = "fcfs",
) -> dict[str, Any]:
    """Replay one portfolio admission policy over a candidate stream.

    Engine finding (robustness checks): first-come-first-served
    capping rejects BETTER trades - on the donor's measured 1040-trade
    stream the rejected mean r_pess was +0.484R vs +0.441R for the
    accepted, a pure capacity loss whose reject rate drifted 16% -> 24%
    as candidate density grew.  The REPLACE policies exist because of
    this finding; the test suite carries the executable version of the
    proof (a stream where the best candidate arrives second loses it
    under FCFS but wins under REPLACE-low).

    Policies:

    * ``"fcfs"`` - slots full -> reject (reason ``"capacity"``).
    * ``"replace_low"`` - slots full -> if the new within-group score
      beats the worst open score, close the worst and take the new;
      the replaced trade contributes 0 (conservative: a mid-flight
      close is assumed to give back).  A candidate not beating the
      worst open score is rejected (reason ``"inferior_score"``).
    * ``"replace_high"`` - same selection, but the replaced trade
      keeps its full isolated ``r_pess`` (optimistic bound).

    Realized equity is EVENT-based (R credited at each trade's
    ``ts_exit``), so the drawdown is not flattered by daily
    aggregation.  Score ties are broken by the secondary key
    ``candidate_id`` (ascending; stream arrival position when absent),
    so equal-score candidates resolve deterministically.

    Args:
        trades: candidate trade stream, each a dict with keys
            ``ts_entry``, ``ts_exit`` (timestamps in one common unit),
            ``s`` (score) and ``r_pess`` (isolated pessimistic R);
            optional ``candidate_id`` (int >= 0).  Scores MUST be
            WITHIN-GROUP rank-normalized
            (:func:`rank_normalize_group`); raw cross-model
            probabilities are not comparable.  If a ``CandidateGroup``
            frame is used upstream, pass its ``normalized_scores``
            column as ``s`` - never the raw ``scores``.
        cap: maximum concurrent slots; must be >= 1.
        policy: ``"fcfs"``, ``"replace_low"`` or ``"replace_high"``.

    Returns:
        Dict with keys ``policy``, ``cap``, ``accepted`` (trade dicts),
        ``rejected`` (trade dicts + ``reason``), ``replaced`` (trade
        dicts + ``reason="replaced"``), ``equity_curve`` (event-based
        cumulative R at exits), ``max_drawdown_r``, ``final_equity_r``
        and ``summary`` (the compact ``{"n", "ev", "dd", "replaced"}``
        report row).

    Raises:
        ValueError: if ``cap`` is not an integer >= 1, ``policy`` is
            unknown, ``trades`` is empty or not a list of dicts, or a
            trade dict misses a required key or carries a non-finite
            value.

    Example::

        >>> trades = [
        ...     {"ts_entry": 0, "ts_exit": 10, "s": 0.8, "r_pess": 1.0},
        ...     {"ts_entry": 1, "ts_exit": 20, "s": 0.9, "r_pess": 0.5},
        ... ]
        >>> fcfs = run_admission_policy(trades, cap=1, policy="fcfs")
        >>> fcfs["summary"]
        {'n': 1, 'ev': 1.0, 'dd': 0.0, 'replaced': 0}
        >>> swap = run_admission_policy(trades, cap=1, policy="replace_low")
        >>> [trade["s"] for trade in swap["accepted"]]
        [0.8, 0.9]
        >>> [trade["s"] for trade in swap["replaced"]]
        [0.8]
        >>> swap["summary"]["ev"]
        0.5
    """
    slots = validate_int("cap", cap, minimum=1)
    if policy not in _ADMISSION_POLICIES:
        raise ValueError(
            f"policy must be one of {list(_ADMISSION_POLICIES)}, got"
            f" {policy!r}"
        )
    if not isinstance(trades, list) or not trades:
        raise ValueError("trades must be a non-empty list of trade dicts")
    normalized = [
        _validated_trade(trade, position)
        for position, trade in enumerate(trades)
    ]
    open_slots: list[dict[str, Any]] = []
    entered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    for item in normalized:
        admitted, rejected_out, displaced = _admit_item(
            item, open_slots, slots=slots, policy=policy
        )
        if admitted is not None:
            entered.append(admitted)
        if rejected_out is not None:
            rejected.append(rejected_out)
        if displaced is not None:
            replaced.append(displaced)
    events = _realized_events(entered, replaced, policy)
    return {
        "policy": policy,
        "cap": slots,
        "accepted": [dict(item["row"]) for item in entered],
        "rejected": rejected,
        "replaced": [
            dict(item["row"]) | {"reason": "replaced"} for item in replaced
        ],
        "equity_curve": events["equity"],
        "max_drawdown_r": events["drawdown"],
        "final_equity_r": events["final_equity"],
        "summary": {
            "n": len(entered),
            "ev": events["final_equity"],
            "dd": events["drawdown"],
            "replaced": len(replaced),
        },
    }


def _admit_item(
    item: dict[str, Any],
    open_slots: list[dict[str, Any]],
    *,
    slots: int,
    policy: str,
) -> tuple[
    dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None
]:
    """Admit one candidate against the currently open slots.

    Frees expired slots, then applies the policy.  Returns ``(admitted,
    rejected, displaced)``: exactly one of admitted/rejected is set;
    ``displaced`` is the worst open holder only when the candidate took
    its slot under a REPLACE policy.  Score ties never displace (a
    candidate must STRICTLY beat the worst open score); the worst
    holder itself is the lowest ``score``, ties broken by the stable
    ``tie_key`` (candidate_id ascending, else arrival position).
    """
    open_slots[:] = [
        held for held in open_slots if held["ts_exit"] > item["ts_entry"]
    ]
    if len(open_slots) < slots:
        open_slots.append(item)
        return item, None, None
    if policy == "fcfs":
        return None, dict(item["row"]) | {"reason": "capacity"}, None
    worst = min(open_slots, key=lambda held: (held["score"], held["tie_key"]))
    if item["score"] > worst["score"]:
        open_slots.remove(worst)
        open_slots.append(item)
        return item, None, worst
    return None, dict(item["row"]) | {"reason": "inferior_score"}, None


def _realized_events(
    entered: list[dict[str, Any]],
    replaced: list[dict[str, Any]],
    policy: str,
) -> dict[str, Any]:
    """Event-based equity of the accepted stream (R credited at exits).

    Replaced trades contribute 0 under ``replace_low`` (a mid-flight
    close is assumed to give back) and their full isolated ``r_pess``
    under ``replace_high`` (optimistic bound).

    Returns ``{"equity", "drawdown", "final_equity"}`` with an honest
    all-zero scaffold when nothing was entered.
    """
    replaced_ids = {id(item) for item in replaced}
    events: list[tuple[float, float]] = []
    for item in entered:
        if id(item) in replaced_ids:
            contribution = 0.0 if policy == "replace_low" else item["r_pess"]
        else:
            contribution = item["r_pess"]
        events.append((item["ts_exit"], contribution))
    events.sort(key=lambda pair: pair[0])
    if not events:
        return {
            "equity": np.zeros(0, dtype=np.float64),
            "drawdown": 0.0,
            "final_equity": 0.0,
        }
    equity = np.cumsum(
        np.asarray([pair[1] for pair in events], dtype=np.float64)
    )
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return {
        "equity": equity,
        "drawdown": float(np.max(peaks[1:] - equity)),
        "final_equity": float(equity[-1]),
    }


# ---------------------------------------------------------------------------
# per-trade metrics (single home: atropos.backtest_metrics, re-exported)
# ---------------------------------------------------------------------------
# The four per-trade statistics are DEFINED in
# atropos.backtest_metrics and re-exported here so the engine's
# facade stays one import away from its report row.  Do not copy the
# arithmetic into this module (house rule: helpers have one home).
# Scale warning: these are PER-TRADE statistics - deliberately distinct
# from atropos.evaluation_metrics.sharpe_ratio, which annualizes a
# PER-PERIOD return stream; a per-trade t-stat of ~10+ corresponds to
# a per-trade Sharpe of only ~0.3-0.6.


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _demo_world() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded synthetic OHLC world: chop, downtrend, uptrend, panic."""
    rng = np.random.default_rng(11)
    n_bars = 160
    steps = rng.normal(0.0, 0.6, n_bars)
    steps[20:32] -= 0.9  # downtrend: stop-out material
    steps[40:60] += 1.2  # uptrend: target material
    steps[90] -= 14.0  # panic bar: gap-through material
    closes = 100.0 + np.cumsum(steps)
    opens = np.empty(n_bars)
    opens[0] = 100.0
    opens[1:] = closes[:-1]
    wick_up = np.abs(rng.normal(0.0, 0.3, n_bars))
    wick_down = np.abs(rng.normal(0.0, 0.3, n_bars))
    highs = np.maximum(opens, closes) + wick_up
    lows = np.minimum(opens, closes) - wick_down
    return opens, highs, lows, closes


def _demo_case_a(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    """One-signal sims: TP win, stop loss, gap-through scratch."""
    cost_model = CostModel()
    # A1: long the uptrend -> target hit before the time limit
    entry = 41
    fill = float(opens[entry])
    win = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        fill - 2.0,
        fill + 6.0,
        HOLD_BARS,
        1.0,
    )
    print(
        f"A1 uptrend long: exit={win['exit_idx']}"
        f" r_opt={win['r_opt']:+.3f} r_pess={win['r_pess']:+.3f}"
    )
    assert win["exit_idx"] > entry
    assert win["exit_idx"] < entry + HOLD_BARS  # before the time limit
    assert win["r_opt"] > 0.0 and win["r_pess"] > 0.0
    # target-only pessimism: exactly the e_mult * gen_slip entry term
    entry_slip_r = cost_model.e_mult * cost_model.gen_slip * fill / 2.0
    assert math.isclose(
        win["r_opt"] - win["r_pess"], entry_slip_r, rel_tol=1e-12
    )
    # A2: long the downtrend -> stop hit, pessimism below optimism
    entry = 21
    fill = float(opens[entry])
    loss = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        fill - 1.0,
        fill + 40.0,
        HOLD_BARS,
        1.0,
    )
    print(
        f"A2 downtrend long: exit={loss['exit_idx']}"
        f" r_opt={loss['r_opt']:+.3f} r_pess={loss['r_pess']:+.3f}"
    )
    assert loss["exit_idx"] >= entry  # may stop out on the entry bar
    assert loss["exit_idx"] < entry + HOLD_BARS
    assert loss["r_opt"] < 0.0
    assert loss["r_pess"] < loss["r_opt"]  # SL pessimism stack applied
    # A3: panic gap through the stop at entry -> immediate scratch
    entry = 91
    fill = float(opens[entry])
    scratch = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        fill + 1.0,
        fill + 30.0,
        HOLD_BARS,
        1.0,
    )
    cost_r = (2 * cost_model.comm * fill + cost_model.gen_slip * fill) / 1.0
    print(
        f"A3 gap-through scratch: exit={scratch['exit_idx']}"
        f" r_opt={scratch['r_opt']:+.3f} r_pess={scratch['r_pess']:+.3f}"
    )
    assert scratch["exit_idx"] == entry
    assert math.isclose(scratch["r_opt"], -cost_r, rel_tol=1e-12)
    assert scratch["r_pess"] < scratch["r_opt"]


def _demo_case_b(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    """State machine: one slot, force-close with reconstructed P&L."""
    long_exit_idx = simulate_signal(
        opens,
        highs,
        lows,
        closes,
        41,
        "long",
        float(opens[41]) - 2.0,
        float(opens[41]) + 10.0,
        HOLD_BARS,
        1.0,
    )["exit_idx"]
    candidates = pd.DataFrame(
        {
            "decision_idx": [40, 44, 47],
            "side": ["long", "long", "long"],
            "exit_idx": [long_exit_idx, 60, 65],
            "priority": [0.5, 0.9, 0.7],
        }
    )
    replay = run_position_state_machine(candidates)
    taken = replay[replay["status"] == "taken"]
    print(
        "B1 overlapping signals:"
        f" taken={taken['decision_idx'].tolist()}"
        f" reasons={replay['reason'].tolist()}"
    )
    assert len(taken) == 1  # exactly one slot despite higher-priority peers
    assert taken["decision_idx"].tolist() == [40]
    assert replay["reason"].tolist()[1:] == ["in_position", "in_position"]
    # B2: opposite-signal force-close with reconstructed P&L
    with_reverse = pd.DataFrame(
        {
            "decision_idx": [40, 46],
            "side": ["long", "short"],
            "exit_idx": [80, 90],
            "stop_price": [float(opens[41]) - 2.0, float(opens[47]) + 2.0],
            "target_price": [float(opens[41]) + 10.0, float(opens[47]) - 6.0],
            "atr_entry": [1.0, 1.0],
        }
    )
    reversed_replay = run_position_state_machine(
        with_reverse,
        allow_reverse=True,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
    )
    closed = reversed_replay.iloc[0]
    print(
        f"B2 force-close: force_exit_idx={closed['force_exit_idx']:.0f}"
        f" force_close_r={closed['force_close_r']:+.3f}"
        " (isolated stale r stays stale)"
    )
    assert closed["force_exit_idx"] == 46.0
    assert math.isfinite(closed["force_close_r"])
    assert reversed_replay.iloc[1]["status"] == "taken"  # reverse opened


def _demo_case_c() -> None:
    """Admission: FCFS loses the best candidate; REPLACE policies win."""
    best_arrives_last = [
        {
            "ts_entry": 0,
            "ts_exit": 30,
            "s": 0.6,
            "r_pess": 0.2,
            "candidate_id": 1,
        },
        {
            "ts_entry": 5,
            "ts_exit": 40,
            "s": 0.4,
            "r_pess": -0.3,
            "candidate_id": 2,
        },
        {
            "ts_entry": 10,
            "ts_exit": 50,
            "s": 0.9,
            "r_pess": 1.2,
            "candidate_id": 3,
        },
    ]
    fcfs = run_admission_policy(best_arrives_last, cap=2, policy="fcfs")
    replace_low = run_admission_policy(
        best_arrives_last, cap=2, policy="replace_low"
    )
    replace_high = run_admission_policy(
        best_arrives_last, cap=2, policy="replace_high"
    )
    fcfs_scores = [trade["s"] for trade in fcfs["accepted"]]
    print(
        f"C1 FCFS accepted scores={fcfs_scores}"
        f" ev={fcfs['final_equity_r']:+.2f} dd={fcfs['max_drawdown_r']:.2f}"
        f" | REPLACE-low ev={replace_low['final_equity_r']:+.2f}"
        f" | REPLACE-high ev={replace_high['final_equity_r']:+.2f}"
    )
    # FCFS loses the BEST candidate (arrived last, slots full):
    assert fcfs_scores == [0.6, 0.4]
    assert [trade["s"] for trade in fcfs["rejected"]] == [0.9]
    # REPLACE-low takes it instead of the worst open holder (the
    # displaced trade stays in the ledger as ever-entered; its R
    # credit is policy-adjusted in the equity):
    assert [trade["s"] for trade in replace_low["accepted"]] == [
        0.6,
        0.4,
        0.9,
    ]
    assert [trade["s"] for trade in replace_low["replaced"]] == [0.4]
    replace_equity = replace_low["equity_curve"].tolist()
    assert len(replace_equity) == 3
    assert math.isclose(
        replace_equity[1], replace_equity[0], rel_tol=1e-12
    )  # the displaced trade credits exactly 0 under replace_low
    # FCFS rejected mean BEATS its accepted mean (the measured finding):
    assert sum(fcfs_scores) / len(fcfs_scores) < 0.9
    assert replace_low["final_equity_r"] > fcfs["final_equity_r"]
    # replaced contributes 0 under low, full r_pess under high:
    assert math.isclose(
        replace_low["final_equity_r"], 0.2 + 1.2, rel_tol=1e-12
    )
    assert math.isclose(
        replace_high["final_equity_r"], 0.2 - 0.3 + 1.2, rel_tol=1e-12
    )
    # event-based equity/drawdown on the FCFS run (exits 30, 40):
    equity_values = fcfs["equity_curve"].tolist()
    assert len(equity_values) == 2
    assert math.isclose(equity_values[0], 0.2, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(equity_values[1], -0.1, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(fcfs["max_drawdown_r"], 0.3, rel_tol=1e-12)
    # determinism: the same stream replays bit-for-bit
    fcfs_again = run_admission_policy(best_arrives_last, cap=2, policy="fcfs")
    assert fcfs_again["accepted"] == fcfs["accepted"]
    assert (
        fcfs_again["equity_curve"].tolist() == fcfs["equity_curve"].tolist()
    )
    print("C2 determinism: two FCFS replays agree bit-for-bit")


def _demo_case_d(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    """Maker vs market mechanics on one trend signal."""
    # find a trend bar whose near zone is touched within a few bars
    entry = None
    for candidate_bar in range(42, 58):
        limit_probe = float(opens[candidate_bar]) - 0.35
        if any(
            lows[bar] <= limit_probe
            for bar in range(candidate_bar, candidate_bar + 3)
        ):
            entry = candidate_bar
            break
    assert entry is not None
    fill = float(opens[entry])
    maker_r = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        fill - 1.0,
        fill + 5.0,
        fill,
        1.0,
        delta_atr=0.35,
        wait_bars=3,
    )
    market_r = market_sim(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        fill - 1.0,
        fill + 5.0,
        1.0,
    )
    print(
        f"D1 maker (delta=0.35 ATR): r={maker_r:+.3f}"
        f" vs market r={market_r:+.3f}"
    )
    assert maker_r != 0.0  # the zone was touched (fill happened)
    assert math.isfinite(maker_r) and math.isfinite(market_r)
    # maker fee monotonicity: cheaper entry fee -> strictly better R
    maker_free = maker_sim(
        opens,
        highs,
        lows,
        closes,
        entry + 1,
        "long",
        fill - 1.0,
        fill + 5.0,
        float(lows[entry + 1]),
        1.0,
        maker_frac=0.0,
    )
    maker_taker = maker_sim(
        opens,
        highs,
        lows,
        closes,
        entry + 1,
        "long",
        fill - 1.0,
        fill + 5.0,
        float(lows[entry + 1]),
        1.0,
        maker_frac=1.0,
    )
    assert maker_free > maker_taker
    # mechanics equivalence: delta=0 + full-taker maker fee == market_sim
    flat_maker = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        entry + 1,
        "long",
        fill - 1.0,
        fill + 5.0,
        float(opens[entry + 1]),
        1.0,
        delta_atr=0.0,
        maker_frac=1.0,
    )
    flat_market = market_sim(
        opens,
        highs,
        lows,
        closes,
        entry + 1,
        "long",
        fill - 1.0,
        fill + 5.0,
        1.0,
    )
    assert math.isclose(flat_maker, flat_market, rel_tol=1e-12)
    # missed zone -> maker-or-skip contributes exactly 0 (a deep but
    # valid zone that the path never trades through in wait_bars)
    calm_fill = float(opens[entry])
    missed = maker_entry_sim(
        opens,
        highs,
        lows,
        closes,
        entry,
        "long",
        calm_fill - 1.0,
        calm_fill + 5.0,
        calm_fill,
        1.0,
        delta_atr=0.9,
        wait_bars=2,
    )
    print(f"D2 missed fill r={missed:+.1f} (maker-or-skip)")
    assert missed == 0.0


def _demo_case_e() -> None:
    """Per-trade metrics and the rank-normalized score glue."""
    stream = [1.0, -0.5, 2.0, -1.0]
    stats_row = trade_curve_stats(stream)
    print(
        f"E1 stream {stream}: n={stats_row['n']}"
        f" t={stats_row['t_stat']:+.3f} dd={stats_row['max_dd_r']:.2f}"
        f" pf={stats_row['profit_factor']:.2f}"
        f" per_trade_sharpe={per_trade_sharpe(stream):+.3f}"
    )
    assert stats_row["n"] == 4
    assert math.isclose(stats_row["max_dd_r"], 1.0, rel_tol=1e-12)
    assert math.isclose(stats_row["profit_factor"], 2.0, rel_tol=1e-12)
    # identity: t-stat == per-trade Sharpe * sqrt(n)
    assert math.isclose(
        stats_row["t_stat"],
        per_trade_sharpe(stream) * math.sqrt(stats_row["n"]),
        rel_tol=1e-12,
    )
    assert 0.27 < per_trade_sharpe(stream) < 0.28  # healthy band sanity
    # zero dispersion: honest NaN (the house metric policy), never 0.0
    assert math.isnan(per_trade_sharpe([0.7] * 5))
    assert math.isnan(trade_curve_stats([])["t_stat"])
    scores = rank_normalize_group(np.array([0.1, 0.9, 0.5]))
    assert scores.tolist() == [0.0, 1.0, 0.5]
    assert rank_normalize_group(np.array([4.2])).tolist() == [0.5]
    print(
        "E2 rank_normalize_group: within-group ranks"
        f" {scores.tolist()} (singleton -> [0.5])"
    )


def run_all_examples() -> None:
    """Demo the engine on a seeded synthetic OHLC world (asserting)."""
    opens, highs, lows, closes = _demo_world()
    print("=" * 72)
    print("backtesting demo: seeded synthetic OHLC world (trend + panic)")
    print("=" * 72)
    _demo_case_a(opens, highs, lows, closes)
    _demo_case_b(opens, highs, lows, closes)
    _demo_case_c()
    _demo_case_d(opens, highs, lows, closes)
    _demo_case_e()
    print("all backtesting demo asserts passed")


if __name__ == "__main__":
    run_all_examples()
