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
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd

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
        >>> book = Book()
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
        self, *, cost: CostModel = DEFAULT_COST, seed: int = 42
    ) -> None:
        """Store the cost model and reset the R ledger."""
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
                distance on a Book-managed exit, or an unexpected
                simulator exit label.
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
        instrument_id = str(leg.instrument.spec["field"])
        engine = LegEngine()
        records: list[TradeRecord] = []
        open_pos: _OpenPosition | None = None
        cooldown_until = -1
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
                if 0 <= sim["exit_idx"] <= intent.entry_idx:
                    # resolved ON the entry bar (gap-through scratch
                    # or same-bar stop/target): no signal window.
                    record = self._close_with_sim(
                        leg, md, ts_series, open_pos, instrument_id
                    )
                    records.append(record)
                    cooldown_until = record["exit_bar"] + cooldown_bars
                    open_pos = None
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
                )
            elif decision_pos == open_pos.sim["exit_idx"]:
                record = self._close_with_sim(
                    leg, md, ts_series, open_pos, instrument_id
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
                )
            else:
                continue
            records.append(record)
            cooldown_until = record["exit_bar"] + cooldown_bars
            open_pos = None
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
    ) -> TradeRecord:
        """Close exactly as :func:`simulate_intents` resolved it."""
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
    ) -> TradeRecord:
        """Close at a raw market price (the "signal"/"forced" flavors)."""
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
    ) -> TradeRecord:
        """Materialize one TradeRecord and book it (ledger + equity)."""
        size_r = open_pos.size_r
        costs_r = costs_r_1r * size_r
        pnl_net_r = r_pess_1r * size_r
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

    Returns:
        ``{"records": {leg_name: [TradeRecord, ...]}, "equity_r":
        sum of per-leg net R, "per_leg": {leg_name: {"n_trades":
        int, "pnl_net_r": float}}}``.

    Raises:
        ValueError: when ``legs`` is empty or two legs share a name.

    Example::

        >>> out = run_book((_demo_leg(),), _demo_md())
        >>> out["per_leg"]["demo_long"]["n_trades"]
        1
        >>> out["records"]["demo_long"][0]["exit_reason"] in {
        ...     "stop", "target", "time", "forced",
        ... }
        True
        >>> out["equity_r"] == out["records"]["demo_long"][0]["pnl_net_r"]
        True
        >>> run_book((), _demo_md())
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
        book = Book(cost=cost)
        leg_records = book.run(spec, md)
        records[spec.name] = leg_records
        leg_pnl = math.fsum(rec["pnl_net_r"] for rec in leg_records)
        per_leg[spec.name] = {
            "n_trades": len(leg_records),
            "pnl_net_r": leg_pnl,
        }
        equity_total += leg_pnl
    return {"records": records, "equity_r": equity_total, "per_leg": per_leg}
