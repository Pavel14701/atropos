"""The Book: the imperative shell around the pure leg engine.

FUNCTIONAL CORE / IMPERATIVE SHELL.  :mod:`atropos.legs` is the
functional core: a stateless, pure engine maps
``(leg, md, ts) -> LegDecision`` with no memory.  :class:`Book` is
the shell: it owns everything the engine must not -- the open
position (one slot per leg), the entry cooldown and the equity curve
in R -- walks the decision grid once per leg, consumes one decision
per bar and emits one :class:`TradeRecord` per completed trade, the
single input format for every Atropos analytic tool.

DELEGATION LAW.  Stop/target/time exits are resolved EXCLUSIVELY
through the PUBLIC :func:`atropos.backtesting.simulate_intents`
(never the private scan): at entry acceptance the Book delegates the
intent's full lifecycle and then replays the resolved hold window
bar by bar, so the equality pin "a Book trade == a simulated trade"
for legs without signal exits holds against the public batch API by
construction.  The Book's own exits reuse the same public pessimism
pieces (:func:`atropos.backtesting.pessimistic_r` + the bundled
cost formula).

EXIT REASONS ARE DIAGNOSTICS.  ``TradeRecord.exit_reason`` ("stop" /
"target" / "time" / "signal" / "forced") is the most valuable
post-hoc field the Book emits: a leg closing ~90% of its trades on
stops is being stop-hunted (or its stop is noise), not validated by
its signal -- read the reason histogram before the P&L.  The
simulator's raw "sl"/"tp" labels are translated in EXACTLY ONE
place, the module-level :data:`_EXIT_REASON_MAP`.

SNAPSHOT SEMANTICS (review item 5).  ``exit_ts`` is the CLOSE stamp
of the bar in which the exit filled (intrabar stop/target fills land
on that bar's close label); ``features_at_exit`` is the feature row
AS OF that exit bar; ``features_at_entry`` is the row AS OF the
entry DECISION bar (normally one bar before the fill).  Snapshots
pass NaN through: a snapshot, not a filter.

MVP PRICE PATH.  The materialized frame carries no separate
instrument OHLC world, so the Book requires the instrument's
open/high/low/close as FEATURE columns literally named "open",
"high", "low", "close" (the OHLCV adapter serves exactly these) and
prices its own signal exits with the instrument field's feature
column.  Everything else -- slippage, gaps, commissions -- is the
delegated simulator's validated mechanics, driven by the Book's
:class:`CostModel`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from atropos._utils import validate_number
from atropos.backtesting import (
    DEFAULT_COST,
    CostModel,
    SimIntentResult,
    TradeIntent,
    pessimistic_r,
    simulate_intents,
)
from atropos.data_contract import MaterializedData
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegEngine,
    LegSpec,
    StopSpec,
    validate_features_declared,
)

__all__ = ["Book", "TradeRecord", "run_book"]

_DEFERRED_NOTE = "deferred (KNOWN-ISSUES section 14)"
"""House phrase every deferred-feature ValueError must contain."""

_OHLC_COLUMNS = ("open", "high", "low", "close")
"""Feature-column names the MVP price path requires on md.features."""

# The SINGLE translation point sim-label -> TradeRecord exit_reason:
# every "sl"/"tp" -> "stop"/"target" mapping must go through this
# dict exactly once, so record consumers only ever see mapped labels.
_EXIT_REASON_MAP = {"sl": "stop", "tp": "target", "time": "time"}

# Carry-calendar table (W3-1): calendar -> (daily bars per year,
# session hours per day).  The year figure is the TRADING-day count,
# not 365 -- pricing an XNYS daily bar with 365 would understate
# every short's carry cost by roughly a third.
_YEAR_PERIODS_TABLE: dict[str, tuple[float, float]] = {
    "XNYS": (252.0, 6.5),
    "24/7": (365.0, 24.0),
}


class TradeRecord(TypedDict, total=True):
    """One completed trade: the universal Atropos analytic input.

    Emitted by :class:`Book` (and :func:`run_book`) in close order;
    every analytic tool consumes THIS shape and nothing else.  All
    feature snapshots pass NaN through (a snapshot, not a filter).

    Keys:
        leg_name: the owning leg's name.
        entry_ts: the fill bar's label (bar-close stamp convention).
        exit_ts: the CLOSE stamp of the exit bar (intrabar stop/
            target fills land on that bar's close label; review
            item 5).
        direction: ``+1`` long / ``-1`` short.
        instrument_id: the feature column that IS the instrument.
        entry_bar: fill-bar position on the decision grid.
        exit_bar: exit-bar position on the decision grid.
        entry_price: the entry fill (the entry bar's open).
        exit_price: slipped simulator price for stop/target/time;
            raw instrument value for "signal"; final close for
            "forced".
        size_r: position size in R, fixed at entry.
        pnl_gross_r: ``pnl_net_r + costs_r`` (costs live INSIDE the
            pessimistic R; gross is reconstructed, not measured).
        pnl_net_r: pessimistic net R scaled by ``size_r``.
        costs_r: bundled round-trip taker cost scaled by ``size_r``.
        exit_reason: "stop" | "target" | "time" | "signal" |
            "forced".
        features_at_entry: feature row AS OF the entry DECISION bar
            (one bar before the fill, except the final-bar clamp).
        features_at_exit: feature row AS OF the exit bar.
        diagnostics: free-form dict (positions, raw sim labels,
            per-1R values, equity after; ``r_opt`` is the sim's
            value, or the optimistic net for Book-managed exits).
    """

    leg_name: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: int
    instrument_id: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    size_r: float
    pnl_gross_r: float
    pnl_net_r: float
    costs_r: float
    exit_reason: str
    features_at_entry: dict[str, float]
    features_at_exit: dict[str, float]
    diagnostics: dict[str, Any]


def _feature_snapshot(
    md: MaterializedData, names: tuple[str, ...], ts: pd.Timestamp
) -> dict[str, float]:
    """Float row of the declared features at ``ts`` (snapshot).

    Mirrors the engine's decision-row lookup (``.loc`` on the exact
    timestamp, float ``to_numpy``).  A missing ``ts`` raises
    KeyError (a caller bug, loud); NaN passes through.
    """
    columns = md.features[[*names]]
    row_values = columns.loc[columns.index == ts].to_numpy(dtype=float)
    if row_values.size == 0:
        raise KeyError(ts)
    return {
        name: float(value)
        for name, value in zip(names, row_values[0], strict=True)
    }


def _year_periods(calendar: str, bar_delta: pd.Timedelta) -> float:
    """Bars per year the carry accrual divides annual rates by.

    The calendar-aware alternative to a blind 365: an XNYS year has
    252 TRADING-day bars, so pricing a daily equity bar with 365
    understates the per-bar carry by ~31% -- exactly the short-cost
    self-deception this framework kills.  Derivation:

    * ``bar_delta > 1 day``: the daily figure spread over the bar's
      CALENDAR day length (a weekly XNYS bar -> 252 / 7 = 36.0).
      Kept simple and honest -- it never pretends non-trading days
      were sessions.
    * ``bar_delta == 1 day``: the daily figure (XNYS 252.0,
      "24/7" 365.0).
    * ``bar_delta < 1 day``: ``daily * session_hours /
      hours_per_bar`` (XNYS 1h -> 252 * 6.5 = 1638.0; "24/7" 1h ->
      365 * 24 = 8760.0) -- bars accrue only while sessions run.

    Args:
        calendar: the manifest's calendar key
            (``DataManifest["calendar"]``).
        bar_delta: the grid's median bar spacing (a positive
            ``pd.Timedelta``).

    Returns:
        Positive bars-per-year for the per-bar carry division.

    Raises:
        ValueError: on an unknown calendar or a non-positive
            ``bar_delta``.

    Example::

        >>> _year_periods("XNYS", pd.Timedelta(days=1))
        252.0
        >>> _year_periods("24/7", pd.Timedelta(days=1))
        365.0
        >>> _year_periods("XNYS", pd.Timedelta(hours=1))
        1638.0
        >>> _year_periods("24/7", pd.Timedelta(hours=1))
        8760.0
        >>> _year_periods("XNYS", pd.Timedelta(days=7))
        36.0
    """
    table_row = _YEAR_PERIODS_TABLE.get(calendar)
    if table_row is None:
        raise ValueError(
            f"unknown calendar {calendar!r}; pass periods_override"
            " explicitly (supported: XNYS, 24/7)"
        )
    daily_bars, session_hours = table_row
    one_day = pd.Timedelta(days=1)
    if not bar_delta > pd.Timedelta(0):
        raise ValueError(
            f"bar_delta must be a positive pd.Timedelta, got {bar_delta!r}"
        )
    if bar_delta > one_day:
        return daily_bars / (bar_delta / one_day)
    if bar_delta == one_day:
        return daily_bars
    hours_per_bar = bar_delta / pd.Timedelta(hours=1)
    return daily_bars * session_hours / hours_per_bar


@dataclass(frozen=True)
class _OpenPosition:
    """One admitted entry's immutable tracking snapshot (Book state)."""

    intent: TradeIntent
    sim: SimIntentResult
    entry_decision_pos: int
    forced_last_bar: bool
    size_r: float
    fill: float
    risk_ref: float


def _demo_md() -> MaterializedData:
    """Synthetic 8-bar decision grid carrying the OHLC feature columns.

    Prices walk 100 -> 102 -> 98, dipping through the demo stop at
    bar 4; ``mom`` fires entry decisions at bars 1-3 only.
    """
    bars = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0],
            "high": [101.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0],
            "low": [99.0, 99.0, 100.0, 101.0, 99.0, 98.0, 97.0, 96.0],
            "close": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0],
            "mom": [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            "atr": [0.5] * 8,
        },
        index=bars,
    )
    return MaterializedData(
        outcome=pd.Series(0.0, index=bars),
        features=features,
        controls=pd.DataFrame(index=bars),
        index=pd.DataFrame(
            {"ts": bars, "entity": "", "split": "train"}, index=bars
        ),
    )


def _demo_leg() -> LegSpec:
    """Long demo leg: momentum gate, 2-ATR stop, no target/time/signal."""
    return LegSpec(
        name="demo_long",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 2.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="demo fixture: momentum entry with a tight ATR stop",
    )


class Book:
    """Stateful imperative shell: decisions in, TradeRecords out.

    The Book owns what the pure engine must not: the open position
    (one slot per leg), the entry cooldown, and the equity curve in
    R (starts at 0.0; every close adds ``pnl_net_r``).  Delegated
    exits go through the public
    :func:`atropos.backtesting.simulate_intents`; the Book-managed
    exits ("signal", "forced") are priced with the same public
    pessimism pieces under the ``"time"`` label -- entry slippage
    plus extra exit slippage, NO stop-gap term (the adverse gap is a
    stop phenomenon; these exits are not stops).

    Sizing (MVP): ``mode="fixed"`` trades ``leg.sizing.param`` R per
    trade; ``mode="risk_per_trade"`` sizes at ``equity_r * param`` R
    -- equity is tracked in R, so the fraction acts on accumulated R
    directly; at ``equity_r <= 0`` the size is 0 (no re-risking after
    losses).  A zero-size trade is still recorded with zero P&L so
    the record stream stays the full decision audit.

    P&L decomposition (honest): costs are booked INSIDE the
    pessimistic R, so ``pnl_net_r`` is pessimistic R scaled by
    ``size_r`` and ``pnl_gross_r`` is reconstructed as ``pnl_net_r +
    costs_r``; gross is never measured independently in this stack.

    Carry costs (W3-1): while a position is open the walk counts
    closed bars and the close books their drag as extra cost.  Per
    closed bar, on the notional ``size_r * risk_ref * fill /
    |fill - stop|``: shorts pay ``borrow_bps_annual / 1e4 / periods
    * notional`` (longs pay no borrow) and funding follows the perp
    convention with the sign carried by ``funding_bps_annual`` --
    positive funding is PAID by longs and RECEIVED by shorts
    (negative reverses the sides).  Exactly ``k = exit_bar -
    entry_bar`` bars accrue: same-bar closes accrue zero, never
    ``k + 1``.  ``periods`` (bars per year) is ``periods_override``
    or is derived from the manifest's calendar and the median ts
    spacing (XNYS: 252 daily bars x 6.5h session; "24/7": 365 x 24h)
    -- never a blind 365, which would understate equity short costs
    by about a third.  The drag moves money only AFTER the delegated
    sim has chosen the exit: exit selection and ``r_pess`` stay
    pre-carry (the delegation pin depends on that ordering).

    One instance may walk several legs (``run`` accumulates on
    ``closed`` / ``equity_r``); :func:`run_book` gives every leg a
    fresh Book.

    Args:
        cost: the taker cost model handed to every delegated
            simulation and every Book-managed R computation.
        seed: reserved for future stochastic draws; the MVP walk is
            deterministic and the value is stored unused.

    Example::

        >>> md = _demo_md()
        >>> leg = _demo_leg()
        >>> book = Book(
        ...     borrow_bps_annual=500.0,
        ...     funding_bps_annual=100.0,
        ...     periods_override=252.0,
        ... )
        >>> records = book.run(leg, md)
        >>> len(records)  # one slot: the first firing entry wins
        1
        >>> records[0]["exit_reason"]
        'stop'
        >>> records[0]["entry_bar"], records[0]["exit_bar"]
        (2, 4)
        >>> records[0]["features_at_entry"]
        {'mom': 1.0, 'atr': 0.5, 'close': 100.0}
        >>> records[0]["features_at_exit"]
        {'mom': -1.0, 'atr': 0.5, 'close': 101.0}
        >>> records[0]["size_r"]
        1.0
        >>> book.equity_r == records[0]["pnl_net_r"]
        True
    """

    def __init__(
        self,
        *,
        borrow_bps_annual: float,
        funding_bps_annual: float,
        periods_override: float | None = None,
        cost: CostModel = DEFAULT_COST,
        seed: int = 42,
    ) -> None:
        """Store the cost model, the carry policy, reset the ledger.

        The two carry rates are REQUIRED (no default): a silent zero
        would make every short strategy look better than it is --
        exactly the self-deception this framework kills.  Zero is
        ALLOWED but warns loudly.
        """
        validate_number("borrow_bps_annual", borrow_bps_annual)
        if borrow_bps_annual < 0.0:
            raise ValueError(
                "borrow_bps_annual must be >= 0 (a short-borrow cost),"
                f" got {borrow_bps_annual!r}"
            )
        validate_number("funding_bps_annual", funding_bps_annual)
        if periods_override is not None and not (
            np.isfinite(periods_override) and periods_override > 0.0
        ):
            raise ValueError(
                "periods_override must be a positive finite float when"
                f" given, got {periods_override!r}"
            )
        if borrow_bps_annual == 0.0:
            warnings.warn(
                "borrow_bps_annual=0 disables short-cost modeling;"
                " COST_DEATH for short strategies will be understated",
                UserWarning,
                stacklevel=2,
            )
        if funding_bps_annual == 0.0:
            warnings.warn(
                "funding_bps_annual=0 disables funding modeling;"
                " carry costs will be understated",
                UserWarning,
                stacklevel=2,
            )
        self.borrow_bps_annual = float(borrow_bps_annual)
        self.funding_bps_annual = float(funding_bps_annual)
        self.periods_override = periods_override
        self.cost = cost
        self.seed = seed
        self.equity_r: float = 0.0
        self.closed: list[TradeRecord] = []

    def run(self, leg: LegSpec, md: MaterializedData) -> list[TradeRecord]:
        """Walk one leg across the decision grid; return its trades.

        The single-slot walk: every decision bar is evaluated ONCE
        by the stateless engine; an accepted entry delegates its full
        lifecycle to :func:`atropos.backtesting.simulate_intents`
        immediately, then the Book replays that resolved hold window
        bar by bar, closing early only on the leg's own signal exit
        (at that bar's instrument value, reason "signal").  Cooldown:
        after a close at bar ``b``, entry decisions are ignored while
        ``decision_pos <= b + cooldown_bars`` (median grid spacing,
        min 0).  An unresolved delegation (data ended inside the
        hold) is force-closed at the last bar with reason "forced".
        With nonzero carry rates the walk also counts the held bars
        and the close books their per-bar drag (see the class
        docstring); the drag books only after the exit is chosen and
        so never influences exit selection.

        Args:
            leg: the leg spec (feature contract validated loudly).
            md: the materialized bundle; its features frame must
                carry the leg's declared columns plus the OHLC
                columns (see the MVP price path above).

        Returns:
            This leg's closed trades, in close order; the same
            records are appended to ``self.closed`` and each
            ``pnl_net_r`` is added to ``self.equity_r``.

        Raises:
            ValueError: on a violated feature contract,
                ``entry.max_concurrent > 1`` (deferred), missing or
                misaligned OHLC feature columns, a non-finite
                instrument value at a signal exit, a degenerate risk
                distance on a Book-managed exit, an undecidable
                carry calendar (no manifest and no
                ``periods_override``, or an unknown calendar), a
                degenerate fill-equals-stop geometry while carry
                accrues, or an unexpected simulator exit label.
        """
        validate_features_declared(leg)
        if leg.entry.max_concurrent > 1:
            raise ValueError(
                f"Book: LegSpec {leg.name!r} entry.max_concurrent="
                f"{leg.entry.max_concurrent} is {_DEFERRED_NOTE}"
                " (MVP runs one slot per leg)"
            )
        opens, highs, lows, closes, instrument_px = self._price_arrays(
            leg, md
        )
        ts_series = pd.DatetimeIndex(md.index["ts"])
        n_bars = int(ts_series.size)
        cooldown_bars = self._cooldown_bars(leg, ts_series)
        # Carry setup (W3-1), ONCE per run, before the walk.  The
        # drag never touches exit selection: the delegated sim below
        # has already resolved exit_idx / r_pess on pre-carry R; the
        # walk only counts held bars and the close shifts money.
        # ORDERING PIN (see the class docstring).  Fast path: with
        # both rates zero there is no accrual at all (periods stays
        # an undivided sentinel).  On a <2-bar grid every trade is
        # same-bar (k = exit_bar - entry_bar = 0), so the calendar
        # stays unresolved without changing any record.
        carry_active, periods = self._carry_setup(md, n_bars)
        instrument_id = str(leg.instrument.spec["field"])
        engine = LegEngine()
        records: list[TradeRecord] = []
        open_pos: _OpenPosition | None = None
        cooldown_until = -1
        carry_bars = 0
        for decision_pos in range(n_bars):
            ts = ts_series[decision_pos]
            decision = engine.evaluate(leg, md, ts)
            if open_pos is None:
                intent = decision.entry_intent
                if intent is None or decision_pos <= cooldown_until:
                    continue
                sim = simulate_intents(
                    [intent], opens, highs, lows, closes, cost=self.cost
                )[0]
                open_pos = _OpenPosition(
                    intent=intent,
                    sim=sim,
                    entry_decision_pos=decision_pos,
                    forced_last_bar=bool(
                        decision.diagnostics["forced_last_bar"]
                    ),
                    size_r=self._size_r(leg),
                    fill=float(opens[intent.entry_idx]),
                    risk_ref=float(sim["risk_ref"]),
                )
                carry_bars = 0
                if 0 <= sim["exit_idx"] <= intent.entry_idx:
                    # resolved ON the entry bar (gap-through scratch
                    # or same-bar stop/target): no signal window, no
                    # accrual (k = 0 by the discrete-bar rule).
                    record = self._close_with_sim(
                        leg,
                        md,
                        ts_series,
                        open_pos,
                        instrument_id,
                        periods=periods,
                        carry_bars=carry_bars,
                    )
                    records.append(record)
                    cooldown_until = record["exit_bar"] + cooldown_bars
                    open_pos = None
                    carry_bars = 0
                continue
            if decision.exit_requested:
                exit_price = float(instrument_px[decision_pos])
                if not math.isfinite(exit_price):
                    raise ValueError(
                        f"Book: instrument field {instrument_id!r} is"
                        f" not finite at signal-exit bar"
                        f" {decision_pos}; cannot anchor the close"
                    )
                record = self._close_market(
                    leg,
                    md,
                    ts_series,
                    open_pos,
                    instrument_id,
                    exit_bar=decision_pos,
                    exit_price=exit_price,
                    exit_reason="signal",
                    periods=periods,
                    carry_bars=carry_bars,
                )
            elif decision_pos == open_pos.sim["exit_idx"]:
                record = self._close_with_sim(
                    leg,
                    md,
                    ts_series,
                    open_pos,
                    instrument_id,
                    periods=periods,
                    carry_bars=carry_bars,
                )
            elif open_pos.sim["exit_idx"] < 0 and decision_pos == n_bars - 1:
                record = self._close_market(
                    leg,
                    md,
                    ts_series,
                    open_pos,
                    instrument_id,
                    exit_bar=n_bars - 1,
                    exit_price=float(closes[n_bars - 1]),
                    exit_reason="forced",
                    periods=periods,
                    carry_bars=carry_bars,
                )
            else:
                # The position survives decision_pos: one more closed
                # bar of carry.  Discrete-bar rule: the bar that
                # CLOSES the trade accrues nothing, so the count at
                # close is exactly exit_bar - entry_bar.
                carry_bars += 1
                continue
            records.append(record)
            cooldown_until = record["exit_bar"] + cooldown_bars
            open_pos = None
            carry_bars = 0
        if open_pos is not None:
            # An entry clamped onto the final bar (forced_last_bar)
            # never gets a second visit from the walk; force-close it.
            record = self._close_market(
                leg,
                md,
                ts_series,
                open_pos,
                instrument_id,
                exit_bar=n_bars - 1,
                exit_price=float(closes[n_bars - 1]),
                exit_reason="forced",
                periods=periods,
                carry_bars=carry_bars,
            )
            records.append(record)
        return records

    def _price_arrays(
        self, leg: LegSpec, md: MaterializedData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """OHLC + instrument price arrays aligned to the decision grid.

        The MVP price path: the instrument's OHLC ride in
        ``md.features`` under the fixed OHLC names (the OHLCV adapter
        serves exactly these); the instrument field is itself a
        feature column, so its prices read from the same frame.
        OHLC finiteness is delegated to :func:`simulate_intents`.
        """
        missing = [
            name for name in _OHLC_COLUMNS if name not in md.features.columns
        ]
        if missing:
            raise ValueError(
                "Book MVP needs open/high/low/close feature columns"
                " (serve them via the OHLCV adapter); missing:"
                f" {missing}"
            )
        n_bars = int(md.index.shape[0])
        if md.features.shape[0] != n_bars:
            raise ValueError(
                f"Book: md.features has {md.features.shape[0]} rows vs"
                f" {n_bars} decision bars; the OHLC columns must"
                " align to the decision grid"
            )
        instrument_id = str(leg.instrument.spec["field"])
        opens = md.features["open"].to_numpy(dtype=float)
        highs = md.features["high"].to_numpy(dtype=float)
        lows = md.features["low"].to_numpy(dtype=float)
        closes = md.features["close"].to_numpy(dtype=float)
        instrument_px = md.features[instrument_id].to_numpy(dtype=float)
        return opens, highs, lows, closes, instrument_px

    def _cooldown_bars(
        self, leg: LegSpec, ts_series: pd.DatetimeIndex
    ) -> int:
        """Entry cooldown in decision bars (median grid spacing, min 0)."""
        cooldown = leg.entry.cooldown
        if ts_series.size < 2 or cooldown == pd.Timedelta(0):
            return 0
        bar_delta = pd.Timedelta(np.median(np.diff(ts_series.to_numpy())))
        return max(0, int(cooldown / bar_delta))

    def _resolve_periods(self, md: MaterializedData) -> float:
        """Bars per year the carry accrual divides annual rates by.

        ``periods_override`` wins verbatim (an explicit number is
        the loudest statement a researcher can make).  Otherwise the
        bundle must speak for itself: the manifest's ``calendar``
        key names the trading calendar and the median
        ``md.index["ts"]`` spacing names the bar;
        :func:`_year_periods` turns the pair into a positive
        bars-per-year figure (never a blind 365).

        Args:
            md: the materialized bundle being walked.

        Returns:
            Positive bars-per-year for the per-bar carry division.

        Raises:
            ValueError: when there is no manifest and no
                ``periods_override`` (the accrual refuses to guess
                the calendar), when the ts grid is too short to
                measure a bar spacing, or through
                :func:`_year_periods` on an unknown calendar or a
                non-positive spacing.
        """
        if self.periods_override is not None:
            return float(self.periods_override)
        manifest = md.manifest
        if manifest is None:
            raise ValueError(
                "no manifest and no periods_override: the carry"
                " accrual refuses to guess the calendar (pass"
                " periods_override or materialize with a manifest)"
            )
        calendar = manifest["calendar"]
        ts_series = pd.DatetimeIndex(md.index["ts"])
        if ts_series.size < 2:
            raise ValueError(
                "cannot measure a carry bar spacing on a"
                f" {ts_series.size}-bar ts grid; pass periods_override"
            )
        bar_delta = pd.Timedelta(np.median(np.diff(ts_series.to_numpy())))
        return _year_periods(calendar, bar_delta)

    def _size_r(self, leg: LegSpec) -> float:
        """Position size in R at entry (fixed param or equity fraction).

        MVP ``risk_per_trade`` rule: equity is tracked in R, so the
        fraction acts on accumulated R directly (``size_r =
        equity_r * param``, floored at 0); at ``equity_r <= 0`` the
        size is 0 -- no re-risking after losses.
        """
        if leg.sizing.mode == "fixed":
            return float(leg.sizing.param)
        if self.equity_r <= 0.0:
            return 0.0
        return max(0.0, self.equity_r * float(leg.sizing.param))

    def _carry_setup(
        self, md: MaterializedData, n_bars: int
    ) -> tuple[bool, float]:
        """Resolve the carry switch and the bars-per-year for one run.

        The drag never touches exit selection: the delegated sim has
        already resolved ``exit_idx`` / ``r_pess`` on pre-carry R;
        the walk only counts held bars and the close shifts money
        (ORDERING PIN, see the class docstring).  Inactive when both
        rates are zero (no accrual at all; ``periods`` stays an
        undivided sentinel).  On a <2-bar grid every trade is
        same-bar (``k = exit_bar - entry_bar = 0``), so the calendar
        stays unresolved without changing any record.
        """
        active = self.borrow_bps_annual != 0.0 or (
            self.funding_bps_annual != 0.0
        )
        periods = 0.0
        if active and n_bars >= 2:
            periods = self._resolve_periods(md)
        return active, periods

    def _carry_drag_r(
        self,
        leg: LegSpec,
        open_pos: _OpenPosition,
        bars_held: int,
        periods: float,
    ) -> tuple[float, float]:
        """(borrow_r, funding_r) accrued over ``bars_held`` closed bars.

        Notional (the entry risk geometry, R-flat): longs
        ``size_r * risk_ref * fill / (fill - stop_price)``, shorts
        ``size_r * risk_ref * fill / (stop_price - fill)``.  Per
        bar: shorts pay ``borrow_bps_annual / 1e4 / periods *
        notional`` (longs accrue zero borrow), and funding carries
        the SIGN of ``funding_bps_annual`` -- a positive rate is
        paid by longs (recorded ``+rate * notional``, a cost) and
        received by shorts (recorded ``-rate * notional``, a
        credit); a negative rate reverses the two sides.  Recorded
        values are PAID magnitudes: positive = you paid, negative =
        you received.  The total is exactly ``bars_held * per-bar drag``:
        the per-bar drag is computed once, then multiplied.  Zero
        rates or a zero hold take the fast path, ``(0.0, 0.0)``,
        without touching the notional math.

        Args:
            leg: the owning leg (names the degenerate-geometry
                error).
            open_pos: the open position's tracking snapshot (its
                entry geometry drives the notional).
            bars_held: closed hold bars, ``exit_bar - entry_bar``
                (same-bar closes pass 0).
            periods: bars per year from :meth:`_resolve_periods`.

        Returns:
            ``(borrow_r, funding_r)`` in R; both ``0.0`` on the
            fast path.

        Raises:
            ValueError: on a degenerate fill-equals-stop geometry
                (the notional division would be undefined; loud,
                never NaN), or on non-positive ``periods`` while
                rates are nonzero (a caller invariant).
        """
        borrow_bps = self.borrow_bps_annual
        funding_bps = self.funding_bps_annual
        if bars_held <= 0 or (borrow_bps == 0.0 and funding_bps == 0.0):
            return 0.0, 0.0
        if not periods > 0.0:
            raise ValueError(
                "carry periods must be positive while rates are"
                f" nonzero, got {periods!r}"
            )
        side = int(open_pos.intent.side)
        fill = open_pos.fill
        stop_price = float(open_pos.intent.stop_price)
        denom = fill - stop_price if side == 1 else stop_price - fill
        if denom == 0.0:
            raise ValueError(
                "degenerate risk geometry: fill equals stop"
                f" (leg {leg.name!r}, side {side}, fill {fill},"
                f" stop {stop_price})"
            )
        notional_r = open_pos.size_r * open_pos.risk_ref * fill / denom
        borrow_per_bar_r = borrow_bps / 1e4 / periods * notional_r
        funding_per_bar_r = funding_bps / 1e4 / periods * notional_r
        if side == 1:
            # Long: PAYS positive funding (perp convention) -- the
            # recorded value is the PAID magnitude (positive = cost);
            # the borrow book is shorts-only.
            return 0.0, funding_per_bar_r * bars_held
        # Short: pays borrow (positive cost), RECEIVES positive
        # funding (recorded negative -- a credit).
        return borrow_per_bar_r * bars_held, -funding_per_bar_r * bars_held

    def _market_close_r(
        self, leg: LegSpec, open_pos: _OpenPosition, *, exit_price: float
    ) -> tuple[float, float, float]:
        """(r_pess_1r, costs_r_1r, r_net_1r) of a Book-managed exit.

        Signal and forced exits fill at a raw market price, so their
        pessimistic R reuses the PUBLIC pessimism pieces: the bundled
        round-trip cost formula ``(2 * comm + gen_slip) * fill /
        risk_ref``, then :func:`atropos.backtesting.pessimistic_r`
        under ``"time"`` -- entry slippage plus extra exit slippage,
        NO stop-gap term (the adverse gap is a stop phenomenon).
        """
        risk_ref = open_pos.risk_ref
        if not risk_ref > 0.0:
            raise ValueError(
                f"LegSpec {leg.name!r}: degenerate risk_ref"
                f" {risk_ref}; fill equals stop, R is undefined"
            )
        fill = open_pos.fill
        bundled_cost = 2.0 * self.cost.comm + self.cost.gen_slip
        costs_r_1r = bundled_cost * fill / risk_ref
        sign = 1.0 if int(open_pos.intent.side) == 1 else -1.0
        raw_r = sign * (exit_price - fill) / risk_ref
        r_net_1r = raw_r - costs_r_1r
        r_pess_1r = pessimistic_r(
            risk_unit=risk_ref,
            fill_price=fill,
            sl_price=float(open_pos.intent.stop_price),
            exit_reason="time",
            exit_price=exit_price,
            atr_i=float(open_pos.intent.atr_value),
            r_net=r_net_1r,
            cost=self.cost,
        )
        return r_pess_1r, costs_r_1r, r_net_1r

    def _close_with_sim(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts_series: pd.DatetimeIndex,
        open_pos: _OpenPosition,
        instrument_id: str,
        *,
        periods: float,
        carry_bars: int,
    ) -> TradeRecord:
        """Close exactly as :func:`simulate_intents` resolved it.

        The sim's exit numbers pass through untouched; the carry
        window only rides along to the record builder.
        """
        sim = open_pos.sim
        raw_reason = str(sim["exit_reason"])
        if raw_reason not in _EXIT_REASON_MAP:
            raise ValueError(
                "Book: simulate_intents returned exit_reason"
                f" {raw_reason!r} for a resolved trade"
            )
        return self._build_record(
            leg,
            md,
            ts_series,
            open_pos,
            instrument_id,
            exit_bar=int(sim["exit_idx"]),
            exit_reason=_EXIT_REASON_MAP[raw_reason],
            exit_price=float(sim["exit_price"]),
            r_pess_1r=float(sim["r_pess"]),
            costs_r_1r=float(sim["costs_r"]),
            r_opt=float(sim["r_opt"]),
            periods=periods,
            carry_bars=carry_bars,
        )

    def _close_market(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts_series: pd.DatetimeIndex,
        open_pos: _OpenPosition,
        instrument_id: str,
        *,
        exit_bar: int,
        exit_price: float,
        exit_reason: str,
        periods: float,
        carry_bars: int,
    ) -> TradeRecord:
        """Close at a raw market price (the "signal"/"forced" flavors).

        The pessimism pieces decide the exit economics; the carry
        window only rides along to the record builder.
        """
        r_pess_1r, costs_r_1r, r_net_1r = self._market_close_r(
            leg, open_pos, exit_price=exit_price
        )
        return self._build_record(
            leg,
            md,
            ts_series,
            open_pos,
            instrument_id,
            exit_bar=exit_bar,
            exit_reason=exit_reason,
            exit_price=exit_price,
            r_pess_1r=r_pess_1r,
            costs_r_1r=costs_r_1r,
            r_opt=r_net_1r,
            periods=periods,
            carry_bars=carry_bars,
        )

    def _build_record(
        self,
        leg: LegSpec,
        md: MaterializedData,
        ts_series: pd.DatetimeIndex,
        open_pos: _OpenPosition,
        instrument_id: str,
        *,
        exit_bar: int,
        exit_reason: str,
        exit_price: float,
        r_pess_1r: float,
        costs_r_1r: float,
        r_opt: float,
        periods: float,
        carry_bars: int,
    ) -> TradeRecord:
        """Materialize one TradeRecord and book it (ledger + equity).

        Ordering pin: ``r_pess_1r`` / ``costs_r_1r`` arrive already
        decided by the close path (the sim's own numbers, untouched
        by carry); the drag is applied here, AFTER that decision, as
        pure money movement -- ``costs_r = taker_r + borrow_r +
        funding_r`` and ``pnl_gross_r == pnl_net_r + costs_r`` stay
        exact by construction (gross absorbs the drag, net shrinks
        by it, equity follows net).
        """
        size_r = open_pos.size_r
        taker_r = costs_r_1r * size_r
        borrow_r, funding_r = self._carry_drag_r(
            leg, open_pos, carry_bars, periods
        )
        drag_r = borrow_r + funding_r
        costs_r = taker_r + drag_r
        pnl_net_r = r_pess_1r * size_r - drag_r
        entry_ts = ts_series[int(open_pos.intent.entry_idx)]
        exit_ts = ts_series[exit_bar]
        entry_decision_ts = ts_series[open_pos.entry_decision_pos]
        self.equity_r += pnl_net_r
        diagnostics: dict[str, Any] = {
            "entry_decision_pos": open_pos.entry_decision_pos,
            "forced_last_bar": open_pos.forced_last_bar,
            "sim_exit_idx": int(open_pos.sim["exit_idx"]),
            "sim_exit_reason": str(open_pos.sim["exit_reason"]),
            "risk_ref": open_pos.risk_ref,
            "r_opt": r_opt,
            "r_pess_1r": r_pess_1r,
            "costs_r_1r": costs_r_1r,
            "taker_r": taker_r,
            "borrow_r": borrow_r,
            "funding_r": funding_r,
            "equity_r_after": self.equity_r,
        }
        record = TradeRecord(
            leg_name=leg.name,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction=int(open_pos.intent.side),
            instrument_id=instrument_id,
            entry_bar=int(open_pos.intent.entry_idx),
            exit_bar=int(exit_bar),
            entry_price=open_pos.fill,
            exit_price=exit_price,
            size_r=size_r,
            pnl_gross_r=pnl_net_r + costs_r,
            pnl_net_r=pnl_net_r,
            costs_r=costs_r,
            exit_reason=exit_reason,
            features_at_entry=_feature_snapshot(
                md, leg.features_used, entry_decision_ts
            ),
            features_at_exit=_feature_snapshot(
                md, leg.features_used, exit_ts
            ),
            diagnostics=diagnostics,
        )
        self.closed.append(record)
        return record


def run_book(
    legs: tuple[LegSpec, ...] | list[LegSpec],
    md: MaterializedData,
    *,
    cost: CostModel = DEFAULT_COST,
    borrow_bps_annual: float,
    funding_bps_annual: float,
    periods_override: float | None = None,
) -> dict[str, Any]:
    """Run every leg through a FRESH Book (leg isolation by design).

    One :class:`Book` per leg: no shared equity, no shared cooldowns,
    no cross-leg state -- legs are isolated BY CONSTRUCTION, so one
    leg's trades cannot contaminate another's.

    Args:
        legs: the legs to run (tuple or list; non-empty, with
            distinct names -- records are keyed by ``leg.name``).
        md: the materialized bundle every leg walks.
        cost: the cost model handed to every leg's Book; defaults to
            :data:`DEFAULT_COST`.
        borrow_bps_annual: REQUIRED annual short-borrow rate in bps;
            ``0`` is allowed but warns (short-cost modeling off).
        funding_bps_annual: REQUIRED annual funding rate in bps
            (sign-carried); ``0`` is allowed but warns.
        periods_override: optional explicit bars-per-year; ``None``
            derives from the bundle's calendar (checkpoint W3-1).

    Returns:
        ``{"records": {leg_name: [TradeRecord, ...]}, "equity_r":
        sum of per-leg net R, "per_leg": {leg_name: {"n_trades":
        int, "pnl_net_r": float}}}``.

    Raises:
        ValueError: when ``legs`` is empty or two legs share a name.

    Example::

        >>> out = run_book(
        ...     (_demo_leg(),), _demo_md(),
        ...     borrow_bps_annual=500.0,
        ...     funding_bps_annual=100.0,
        ...     periods_override=252.0,
        ... )
        >>> out["per_leg"]["demo_long"]["n_trades"]
        1
        >>> out["records"]["demo_long"][0]["exit_reason"] in {
        ...     "stop", "target", "time", "forced",
        ... }
        True
        >>> out["equity_r"] == out["records"]["demo_long"][0]["pnl_net_r"]
        True
        >>> run_book(
        ...     (), _demo_md(), borrow_bps_annual=1.0, funding_bps_annual=1.0
        ... )
        Traceback (most recent call last):
            ...
        ValueError: run_book needs at least one leg, got 0
    """
    leg_list = list(legs)
    if not leg_list:
        raise ValueError(
            f"run_book needs at least one leg, got {len(leg_list)}"
        )
    names_seen: set[str] = set()
    for spec in leg_list:
        if spec.name in names_seen:
            raise ValueError(
                f"run_book: duplicate leg name {spec.name!r}; per-leg"
                " records are keyed by name"
            )
        names_seen.add(spec.name)
    records: dict[str, list[TradeRecord]] = {}
    per_leg: dict[str, dict[str, Any]] = {}
    equity_total = 0.0
    for spec in leg_list:
        book = Book(
            cost=cost,
            borrow_bps_annual=borrow_bps_annual,
            funding_bps_annual=funding_bps_annual,
            periods_override=periods_override,
        )
        leg_records = book.run(spec, md)
        records[spec.name] = leg_records
        leg_pnl = math.fsum(rec["pnl_net_r"] for rec in leg_records)
        per_leg[spec.name] = {
            "n_trades": len(leg_records),
            "pnl_net_r": leg_pnl,
        }
        equity_total += leg_pnl
    return {"records": records, "equity_r": equity_total, "per_leg": per_leg}


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: delegation equality, carry accrual, exit diagnostics."""
    md = _demo_md()
    leg = _demo_leg()
    print("=" * 72)
    print("book demo: delegation equality + carry + exit diagnostics")
    print("=" * 72)
    records = Book(
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    ).run(leg, md)
    opens = md.features["open"].to_numpy(dtype=float)
    highs = md.features["high"].to_numpy(dtype=float)
    lows = md.features["low"].to_numpy(dtype=float)
    closes = md.features["close"].to_numpy(dtype=float)
    stop_price = 100.0 - 2.0 * 0.5
    reference = simulate_intents(
        [
            TradeIntent(
                entry_idx=2,
                side=1,
                stop_price=stop_price,
                target_price=100.0 + 1e9 * 0.5,
                hold_bars=10_000_000,
                atr_value=0.5,
                risk_ref=1.0,
            )
        ],
        opens,
        highs,
        lows,
        closes,
    )[0]
    record = records[0]
    print(
        f"trades={len(records)} exit_bar={record['exit_bar']}"
        f" reason={record['exit_reason']}"
        f" pnl_net_r={record['pnl_net_r']:.4f}"
    )
    assert record["exit_bar"] == reference["exit_idx"]
    diagnostics = record["diagnostics"]
    drag_r = diagnostics["borrow_r"] + diagnostics["funding_r"]
    # the sim decided the exit on pre-carry R; the drag shifted net:
    assert record["pnl_net_r"] == reference["r_pess"] - drag_r
    assert record["exit_reason"] == "stop"
    assert record["features_at_entry"]["mom"] == 1.0
    assert diagnostics["borrow_r"] == 0.0  # a long never pays borrow
    assert diagnostics["funding_r"] > 0.0  # a long PAYS positive funding
    assert diagnostics["taker_r"] + drag_r == record["costs_r"]
    assert record["pnl_gross_r"] == record["pnl_net_r"] + record["costs_r"]


if __name__ == "__main__":
    run_all_examples()
