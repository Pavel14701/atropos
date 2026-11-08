"""Strategy-level composition: legs + portfolio policy, one shape.

A strategy is ``tuple[LegSpec, ...] + PortfolioPolicy`` -- ONE
abstraction so momentum, pairs and multi-leg options are the same
object for feature analysis.  An iron condor is 4 legs + an
independent policy + caps; a pairs trade is 2 legs (netting later);
a momentum system is 1 leg.  The object shape does not change with
the leg count, so every strategy's trades flatten into the same
analysis-ready frame (:func:`records_to_features`, the bridge from
"what the strategy did" to "what Atropos analyzes").

Anti-patterns this module forbids:

* **strategy-as-backtest** -- a :class:`StrategySpec` is a pure
  DECLARATION; the Book (:func:`atropos.book.run_book`) is the only
  simulator.  Positions, cooldowns, concurrency and costs live
  there, never here.
* **multi-leg-as-special-case** -- composition is homogeneous:
  ``run_book`` runs each leg in a fresh Book so legs cannot
  interact, and the portfolio layer is policy metadata plus
  diagnostics, not a second simulator.

MVP boundary: ``combination`` accepts ``"independent"`` only.  The
deferred combination families (netting / sequential / priority),
exposure ENFORCEMENT (the caps are diagnostics, not admission
control) and universe enforcement (survivorship filters, calendar
gating) are deferred (KNOWN-ISSUES section 14).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from atropos._utils import validate_int, validate_number
from atropos.backtesting import DEFAULT_COST, CostModel
from atropos.book import run_book
from atropos.data_contract import MaterializedData
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    StopSpec,
    TargetSpec,
    validate_features_declared,
)

__all__ = [
    "PortfolioPolicy",
    "StrategySpec",
    "StrategySummary",
    "UniverseSpec",
    "records_to_features",
    "run_strategy",
]

_DEFERRED_NOTE = "deferred (KNOWN-ISSUES section 14)"
"""House phrase every deferred-feature ValueError must contain."""

_DEFERRED_COMBINATIONS = ("netting", "sequential", "priority")
"""Portfolio combination families rejected loudly in the MVP."""


# ---------------------------------------------------------------------------
# policy + universe + strategy specs (frozen, validated at construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioPolicy:
    """How a strategy's legs share one book (MVP: independent only).

    ``combination="independent"`` means positions from different
    legs COEXIST -- a long-X and a short-X leg both hold at the same
    time and each accrues its own PnL; this is NOT netting (no
    offsetting, no margin netting, no position merging).  The other
    combination families are loud deferred rejections, and the caps
    are validated here but only REPORTED as diagnostics by
    :func:`run_strategy` -- enforcement is deferred.

    Args:
        borrow_bps_annual: REQUIRED annual short-borrow rate in bps;
            ``>= 0`` (a cost, never a credit).  ``0`` is allowed --
            it disables short-cost modeling and the Book warns.
        funding_bps_annual: REQUIRED annual funding rate in bps;
            ANY finite float is legal (sign-carried: ``> 0`` means
            longs pay, ``< 0`` means shorts pay).
        periods_override: optional explicit bars-per-year for the
            carry accrual; ``None`` derives from the bundle's
            calendar (checkpoint W3-1).
        combination: how legs combine; the MVP accepts
            ``"independent"`` only.  ``"netting"`` /
            ``"sequential"`` / ``"priority"`` raise ``ValueError``
            naming KNOWN-ISSUES section 14 (the runtime check exists
            despite the ``Literal`` so dynamic callers die loudly
            too).
        max_concurrent_positions: soft cap on simultaneous open
            positions; must be >= 1.  Exceeding it is reported as a
            diagnostic note, never enforced in the MVP.
        max_gross_exposure: soft cap on gross open exposure in R
            units; must be finite > 0.  Not enforced in the MVP.
        max_net_exposure: soft cap on net open exposure in R units;
            must be finite > 0.  Not enforced in the MVP.

    Raises:
        ValueError: on a deferred or unknown ``combination``, a
            ``max_concurrent_positions`` below 1, or a non-finite /
            non-positive exposure cap.

    Example::

        >>> policy = PortfolioPolicy(
        ...     borrow_bps_annual=500.0, funding_bps_annual=100.0
        ... )
        >>> (policy.combination, policy.max_concurrent_positions,
        ...  policy.max_gross_exposure, policy.max_net_exposure)
        ('independent', 10, 2.0, 1.0)
        >>> try:
        ...     PortfolioPolicy(
        ...         borrow_bps_annual=500.0,
        ...         funding_bps_annual=100.0,
        ...         combination="netting",
        ...     )
        ... except ValueError as error:
        ...     "deferred (KNOWN-ISSUES section 14)" in str(error)
        True
        >>> try:
        ...     PortfolioPolicy(
        ...         borrow_bps_annual=500.0,
        ...         funding_bps_annual=100.0,
        ...         combination="random",
        ...     )
        ... except ValueError as error:
        ...     "combination must be 'independent'" in str(error)
        True
    """

    borrow_bps_annual: float
    funding_bps_annual: float
    periods_override: float | None = None
    combination: str = "independent"
    max_concurrent_positions: int = 10
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0

    def __post_init__(self) -> None:
        """Reject deferred families and validate the soft caps."""
        validate_number("borrow_bps_annual", self.borrow_bps_annual)
        if self.borrow_bps_annual < 0.0:
            raise ValueError(
                "borrow_bps_annual must be >= 0 (a short-borrow cost),"
                f" got {self.borrow_bps_annual!r}"
            )
        validate_number("funding_bps_annual", self.funding_bps_annual)
        if self.periods_override is not None and not (
            np.isfinite(self.periods_override) and self.periods_override > 0.0
        ):
            raise ValueError(
                "periods_override must be a positive finite float when"
                f" given, got {self.periods_override!r}"
            )
        if self.combination != "independent":
            if self.combination in _DEFERRED_COMBINATIONS:
                raise ValueError(
                    f"PortfolioPolicy: combination"
                    f" {self.combination!r} is {_DEFERRED_NOTE}"
                    " (MVP composes independent legs)"
                )
            raise ValueError(
                "PortfolioPolicy: combination must be 'independent'"
                " (or a deferred family), got"
                f" {self.combination!r}"
            )
        validate_int(
            "PortfolioPolicy.max_concurrent_positions",
            self.max_concurrent_positions,
            minimum=1,
        )
        gross = validate_number(
            "PortfolioPolicy.max_gross_exposure", self.max_gross_exposure
        )
        if gross <= 0.0:
            raise ValueError(
                "PortfolioPolicy.max_gross_exposure must be finite > 0,"
                f" got {gross}"
            )
        net = validate_number(
            "PortfolioPolicy.max_net_exposure", self.max_net_exposure
        )
        if net <= 0.0:
            raise ValueError(
                "PortfolioPolicy.max_net_exposure must be finite > 0,"
                f" got {net}"
            )


@dataclass(frozen=True)
class UniverseSpec:
    """Declared trading universe (MVP: informational metadata only).

    Validated and echoed into the strategy summary so runs are
    self-describing.  ENFORCEMENT is deferred: no survivorship
    filters, no calendar gating, no entity resolution -- the symbols
    and the calendar name do NOT restrict what a run trades
    (KNOWN-ISSUES section 14).

    Args:
        symbols: non-empty tuple of unique, non-empty symbol strs.
        calendar: session-calendar name (informational label only).

    Raises:
        ValueError: on an empty symbol tuple, a non-str or empty
            symbol, repeated symbols, or an empty calendar name.

    Example::

        >>> universe = UniverseSpec(symbols=("ES", "ZN"), calendar="CME")
        >>> (universe.symbols, universe.calendar)
        (('ES', 'ZN'), 'CME')
        >>> try:
        ...     UniverseSpec(symbols=("ES", "ES"))
        ... except ValueError as error:
        ...     "symbols must be unique" in str(error)
        True
    """

    symbols: tuple[str, ...]
    calendar: str = "24/7"

    def __post_init__(self) -> None:
        """Validate the symbol tuple and the calendar label."""
        symbols = tuple(self.symbols)
        if not symbols:
            raise ValueError(
                "UniverseSpec.symbols must be a non-empty tuple of"
                " non-empty strs"
            )
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(
                    "UniverseSpec.symbols entries must be non-empty"
                    f" strs, got {symbol!r}"
                )
        if len(set(symbols)) != len(symbols):
            repeated = sorted(
                symbol for symbol in symbols if symbols.count(symbol) > 1
            )
            raise ValueError(
                f"UniverseSpec: symbols must be unique, repeated:"
                f" {sorted(set(repeated))}"
            )
        object.__setattr__(self, "symbols", symbols)
        if not isinstance(self.calendar, str) or not self.calendar:
            raise ValueError(
                "UniverseSpec.calendar must be a non-empty str, got"
                f" {self.calendar!r}"
            )


@dataclass(frozen=True)
class StrategySpec:
    """A strategy: named legs plus a portfolio policy and a universe.

    ONE shape for every composition: ``legs`` must be a non-empty
    tuple of :class:`atropos.legs.LegSpec`, and EACH leg is audited
    with :func:`atropos.legs.validate_features_declared` at
    construction -- a strategy with an inconsistent leg is dead on
    arrival, not at run time.  Leg names must be UNIQUE across the
    strategy: ``run_book`` keys its records by leg name, so a
    duplicate would silently collide.  The MVP policy is
    an explicit REQUIRED ``PortfolioPolicy`` (the carry policy is
    declared, never defaulted) and the MVP universe is an unset
    placeholder.

    Args:
        name: non-empty strategy name (echoed into the summary).
        legs: one or more legs; each is feature-contract audited.
        portfolio: how legs share the book (REQUIRED -- the carry
            policy is declared, never defaulted).
        universe: informational universe metadata (enforcement
            deferred).

    Raises:
        ValueError: on an empty or non-str name, a non-LegSpec entry,
            a leg failing :func:`atropos.legs.validate_features_
            declared`, an empty legs tuple, or duplicate leg names.

    Example::

        >>> from atropos.legs import (
        ...     Compare, EntrySpec, ExitSpec, InstrumentRef, LegSpec,
        ... )
        >>> leg = LegSpec(
        ...     name="momo_long", direction="long",
        ...     entry=EntrySpec(Compare("mom", ">", 0.0)),
        ...     exit=ExitSpec(signal=Compare("mom", "<", 0.0)),
        ...     instrument=InstrumentRef("field", {"field": "mom"}),
        ...     features_used=("mom",),
        ...     rationale="momentum persistence",
        ... )
        >>> strategy = StrategySpec(
        ...     name="demo", legs=(leg,),
        ...     portfolio=PortfolioPolicy(
        ...         borrow_bps_annual=500.0, funding_bps_annual=100.0
        ...     ),
        ... )
        >>> (strategy.portfolio.combination, strategy.universe.symbols)
        ('independent', ('unset',))
        >>> try:
        ...     StrategySpec(
        ...         name="demo", legs=(leg, leg),
        ...         portfolio=PortfolioPolicy(
        ...             borrow_bps_annual=500.0,
        ...             funding_bps_annual=100.0,
        ...         ),
        ...     )
        ... except ValueError as error:
        ...     "leg names must be unique" in str(error)
        True
    """

    name: str
    legs: tuple[LegSpec, ...]
    portfolio: PortfolioPolicy
    universe: UniverseSpec = UniverseSpec(symbols=("unset",))

    def __post_init__(self) -> None:
        """Validate the name, audit every leg, pin name uniqueness."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"StrategySpec.name must be a non-empty str,"
                f" got {self.name!r}"
            )
        legs = tuple(self.legs)
        if not legs:
            raise ValueError(
                f"StrategySpec {self.name!r}: legs needs at least one LegSpec"
            )
        for leg in legs:
            if not isinstance(leg, LegSpec):
                raise ValueError(
                    f"StrategySpec {self.name!r}: legs entries must be"
                    f" LegSpec, got {type(leg).__name__}"
                )
            validate_features_declared(leg)
        leg_names = [leg.name for leg in legs]
        if len(set(leg_names)) != len(leg_names):
            repeated = sorted(
                name for name in leg_names if leg_names.count(name) > 1
            )
            raise ValueError(
                f"StrategySpec {self.name!r}: leg names must be unique"
                " across the strategy (run_book records are keyed by"
                f" leg name), repeated: {sorted(set(repeated))}"
            )
        object.__setattr__(self, "legs", legs)


class StrategySummary(TypedDict):
    """One-glance facts of a :func:`run_strategy` call (all present).

    ``per_leg`` maps each leg name to ``{"n_trades": float,
    "pnl_net_r": float}`` (counts are floats so the summary stays
    frame-friendly); ``policy_notes`` carries the honest MVP-boundary
    strings; ``leg_isolation_check`` is True only after the record
    attribution and per-leg recount cross-checks passed -- a failure
    raises instead of flipping the flag.
    """

    strategy_name: str
    n_legs: int
    n_trades: int
    equity_r: float
    per_leg: dict[str, dict[str, float]]
    combination: str
    universe_symbols: list[str]
    policy_notes: list[str]
    leg_isolation_check: bool


# ---------------------------------------------------------------------------
# run + flatten (the strategy -> book -> analysis path)
# ---------------------------------------------------------------------------
def run_strategy(
    strategy: StrategySpec,
    md: MaterializedData,
    *,
    cost: CostModel = DEFAULT_COST,
) -> dict[str, Any]:
    """Run one strategy spec through the Book and summarize it.

    The strategy's legs are handed to :func:`atropos.book.run_book`,
    which runs each leg in a fresh Book so legs cannot interact
    (independent composition).  The policy's caps are NOT enforced:
    a leg count above ``max_concurrent_positions`` becomes a
    diagnostic note, and net/gross exposure enforcement is skipped
    with an honest note (records describe CLOSED trades, not open
    exposure) -- both land in ``summary["policy_notes"]``.  Before
    summarizing, the records' leg attribution is verified (every
    record must belong to a declared leg) and the per-leg trade
    counts are recounted from the records against the Book's
    ``per_leg`` map; the REAL isolation pin lives in the tests.

    Args:
        strategy: the strategy spec (legs audited at construction).
        md: the materialized bundle every leg consumes.
        cost: the cost model handed unchanged to the Book.

    Returns:
        A dict with keys ``"records"`` (per-leg TradeRecord lists,
        echoed from ``run_book``), ``"per_leg"`` (the Book's per-leg
        counts and net R, echoed), ``"equity_r"`` (total net R) and
        ``"summary"`` (a :class:`StrategySummary`).

    Raises:
        ValueError: when leg attribution is lost (a record keyed by,
            or carrying a ``leg_name`` different from, a name outside
            ``strategy.legs``) or the Book's per-leg trade counts
            disagree with the recount from the records.

    Example::

        The entry gates sit at +/- 1.0 because ``StopSpec("price",
        0.9)`` is an ABSOLUTE price: with the N(0,1)-scaled
        ``mom_signal`` as instrument, a long gate at ``> 0`` could
        fire on instrument values in ``(0, 0.9]`` where that stop is
        degenerate (LegEngine raises).  The example asserts on
        structure (keys, attribution, count consistency), not on
        exact trade counts.

        >>> from atropos.data_contract import MaterializedData
        >>> from atropos.legs import (
        ...     Compare, EntrySpec, ExitSpec, InstrumentRef, LegSpec,
        ...     StopSpec, TargetSpec,
        ... )
        >>> bars = pd.date_range("2024-01-01", periods=6, freq="D",
        ...                      tz="UTC")
        >>> close = pd.Series([1.0, 1.2, 0.8, 1.1, 1.3, 0.9], index=bars)
        >>> md = MaterializedData(
        ...     outcome=pd.Series(0.0, index=bars),
        ...     features=pd.DataFrame(
        ...         {
        ...             "mom_signal": [2.0, -2.0, 2.5, -2.5, 2.2, -2.2],
        ...             "open": [0.99, 1.19, 0.79, 1.09, 1.29, 0.89],
        ...             "high": [1.21, 1.21, 0.95, 1.21, 1.31, 1.05],
        ...             "low": [0.97, 0.99, 0.75, 0.95, 1.09, 0.85],
        ...             "close": close,
        ...         },
        ...         index=bars,
        ...     ),
        ...     controls=pd.DataFrame(index=bars),
        ...     index=pd.DataFrame(
        ...         {"ts": bars, "entity": "", "split": "train"}, index=bars
        ...     ),
        ... )
        >>> long_leg = LegSpec(
        ...     name="momo_long", direction="long",
        ...     entry=EntrySpec(Compare("mom_signal", ">", 1.0)),
        ...     exit=ExitSpec(
        ...         stop=StopSpec("price", 0.9),
        ...         target=TargetSpec("price", 1.1),
        ...         signal=Compare("mom_signal", "<", 0.0),
        ...     ),
        ...     instrument=InstrumentRef("field", {"field": "mom_signal"}),
        ...     features_used=("mom_signal",),
        ...     rationale="momentum persistence on the daily grid",
        ... )
        >>> short_leg = LegSpec(
        ...     name="momo_short", direction="short",
        ...     entry=EntrySpec(Compare("mom_signal", "<", -1.0)),
        ...     exit=ExitSpec(
        ...         stop=StopSpec("price", 0.9),
        ...         target=TargetSpec("price", 1.1),
        ...         signal=Compare("mom_signal", ">", 0.0),
        ...     ),
        ...     instrument=InstrumentRef("field", {"field": "mom_signal"}),
        ...     features_used=("mom_signal",),
        ...     rationale="momentum reversal on the daily grid",
        ... )
        >>> strategy = StrategySpec(
        ...     name="momo_pair",
        ...     legs=(long_leg, short_leg),
        ...     portfolio=PortfolioPolicy(
        ...         borrow_bps_annual=500.0, funding_bps_annual=100.0
        ...     ),
        ...     universe=UniverseSpec(symbols=("synthetic",)),
        ... )
        >>> result = run_strategy(strategy, md)
        >>> summary = result["summary"]
        >>> (summary["strategy_name"], summary["n_legs"])
        ('momo_pair', 2)
        >>> (summary["combination"], summary["universe_symbols"])
        ('independent', ['synthetic'])
        >>> required = {
        ...     "combination", "equity_r", "leg_isolation_check",
        ...     "n_legs", "n_trades", "per_leg", "policy_notes",
        ...     "strategy_name", "universe_symbols",
        ... }
        >>> required <= set(summary)
        True
        >>> sorted(result["records"])
        ['momo_long', 'momo_short']
        >>> sorted(summary["per_leg"])
        ['momo_long', 'momo_short']
        >>> leg_names = {"momo_long", "momo_short"}
        >>> all(record["leg_name"] in leg_names
        ...     for leg_records in result["records"].values()
        ...     for record in leg_records)
        True
        >>> summary["n_trades"] == sum(
        ...     len(leg_records)
        ...     for leg_records in result["records"].values()
        ... )
        True
        >>> summary["per_leg"]["momo_long"]["n_trades"] == (
        ...     len(result["records"]["momo_long"])
        ... )
        True
        >>> result["equity_r"] == summary["equity_r"]
        True
        >>> summary["leg_isolation_check"]
        True
        >>> notes = summary["policy_notes"]
        >>> isinstance(notes, list) and len(notes) >= 1
        True
    """
    book_result = run_book(
        list(strategy.legs),
        md,
        cost=cost,
        borrow_bps_annual=strategy.portfolio.borrow_bps_annual,
        funding_bps_annual=strategy.portfolio.funding_bps_annual,
        periods_override=strategy.portfolio.periods_override,
    )
    records = book_result["records"]
    per_leg = book_result["per_leg"]
    counts = _recount_and_verify(strategy, records, per_leg)
    equity_r = float(book_result["equity_r"])
    summary = _build_summary(strategy, counts, per_leg, equity_r)
    return {
        "records": records,
        "per_leg": per_leg,
        "equity_r": equity_r,
        "summary": summary,
    }


def records_to_features(
    records: list[Any],
    feature_keys: tuple[str, ...] = (
        "leg_name",
        "entry_ts",
        "direction",
        "size_r",
        "pnl_net_r",
        "exit_reason",
    ),
) -> pd.DataFrame:
    """Flatten TradeRecords into one analysis-ready frame.

    This is the BRIDGE from "what the strategy did" to "what Atropos
    analyzes": one row per trade, the record's scalar fields
    selected by ``feature_keys`` as columns (dict-valued keys --
    ``features_at_entry`` / ``features_at_exit`` / ``diagnostics``
    -- are never raw columns), PLUS the entry feature snapshot
    expanded with an ``entry::`` prefix so ``orthogonal_check`` /
    ``bucket_scan`` / ``feature_importance`` can consume the trades
    directly.  Causal importance is NOT served by this frame:
    causal tools need features re-generated from the data, not a
    snapshot copied at entry time.

    Args:
        records: TradeRecord dicts (as returned under ``"records"``
            by :func:`run_strategy`).
        feature_keys: which scalar record fields become columns; the
            ``entry::`` expansion is always added on top.

    Returns:
        The analysis-ready frame.  Empty records give an empty frame
        with exactly the ``feature_keys`` columns (the ``entry::``
        columns are data-dependent and therefore absent).

    Raises:
        ValueError: when a record is not a TradeRecord-like mapping.

    Example::

        >>> records = [
        ...     {
        ...         "leg_name": "momo_long", "entry_ts": "2024-01-02",
        ...         "direction": "long", "size_r": 1.0,
        ...         "pnl_net_r": 0.4, "exit_reason": "target",
        ...         "features_at_entry": {"mom_signal": 1.5, "atr": 0.5},
        ...         "features_at_exit": {"mom_signal": 2.1},
        ...         "diagnostics": {"predicate_row": {"mom_signal": 1.5}},
        ...     },
        ...     {
        ...         "leg_name": "momo_short", "entry_ts": "2024-01-03",
        ...         "direction": "short", "size_r": 1.0,
        ...         "pnl_net_r": -0.2, "exit_reason": "stop",
        ...         "features_at_entry": {"mom_signal": -1.2},
        ...         "features_at_exit": {"mom_signal": -0.4},
        ...         "diagnostics": {},
        ...     },
        ... ]
        >>> frame = records_to_features(records)
        >>> len(frame)
        2
        >>> "entry::mom_signal" in frame.columns
        True
        >>> frame["entry::mom_signal"].iloc[0]
        np.float64(1.5)
        >>> empty = records_to_features([])
        >>> len(empty)
        0
        >>> list(empty.columns)[:3]
        ['leg_name', 'entry_ts', 'direction']
    """
    rows: list[dict[str, Any]] = []
    entry_columns: list[str] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"records[{position}] must be a TradeRecord-like"
                f" mapping, got {type(record).__name__}"
            )
        row: dict[str, Any] = {}
        for key in feature_keys:
            value = record.get(key)
            if isinstance(value, Mapping):
                continue  # dict-valued keys never become raw columns
            row[key] = value
        snapshot = record.get("features_at_entry")
        if isinstance(snapshot, Mapping):
            for feature_name, feature_value in snapshot.items():
                if isinstance(feature_value, Mapping):
                    continue
                column = f"entry::{feature_name}"
                row[column] = feature_value
                if column not in entry_columns:
                    entry_columns.append(column)
        rows.append(row)
    if not rows:
        return pd.DataFrame([], columns=list(feature_keys))
    return pd.DataFrame(rows, columns=list(feature_keys) + entry_columns)


# ---------------------------------------------------------------------------
# private helpers (verification + summary assembly)
# ---------------------------------------------------------------------------
def _recount_and_verify(
    strategy: StrategySpec,
    records: Mapping[str, list[Mapping[str, Any]]],
    per_leg: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Verify leg attribution and recount trades per strategy leg.

    ``run_book`` gives each leg a fresh Book (so legs cannot
    interact); here the records must still carry honest attribution:
    a record stored under one leg's name must name exactly that leg,
    and the Book's ``per_leg`` counts must match the recount.

    Returns:
        Trade count per declared leg name (0 for a leg with no
        records), in ``strategy.legs`` order.

    Raises:
        ValueError: with ``"leg attribution lost"`` when a record is
            keyed by an unknown leg name or its ``leg_name``
            disagrees with its key, and on any per-leg count
            disagreement between ``per_leg`` and the recount.
    """
    leg_names = {leg.name for leg in strategy.legs}
    counts: dict[str, int] = {leg.name: 0 for leg in strategy.legs}
    for leg_name, leg_records in records.items():
        if leg_name not in leg_names:
            raise ValueError(
                "leg attribution lost: records are keyed by"
                f" {leg_name!r}, which no leg of strategy"
                f" {strategy.name!r} declares"
            )
        for record in leg_records:
            attributed = record.get("leg_name")
            if attributed != leg_name:
                raise ValueError(
                    "leg attribution lost: a record stored under"
                    f" {leg_name!r} carries leg_name={attributed!r}"
                    f" (strategy {strategy.name!r})"
                )
        counts[leg_name] = len(leg_records)
    for leg_name, count in counts.items():
        book_entry = per_leg.get(leg_name)
        book_count = (
            0 if book_entry is None else int(book_entry.get("n_trades", 0))
        )
        if book_count != count:
            raise ValueError(
                f"strategy {strategy.name!r}: leg {leg_name!r}"
                f" n_trades {book_count} from run_book per_leg !="
                f" recount {count} from records"
            )
    return counts


def _build_summary(
    strategy: StrategySpec,
    counts: dict[str, int],
    per_leg: Mapping[str, Mapping[str, Any]],
    equity_r: float,
) -> StrategySummary:
    """Assemble the :class:`StrategySummary` from verified parts."""
    summary_per_leg: dict[str, dict[str, float]] = {}
    for leg in strategy.legs:
        book_entry = per_leg.get(leg.name)
        pnl_net_r = (
            0.0
            if book_entry is None
            else float(book_entry.get("pnl_net_r", 0.0))
        )
        summary_per_leg[leg.name] = {
            "n_trades": float(counts[leg.name]),
            "pnl_net_r": pnl_net_r,
        }
    return StrategySummary(
        strategy_name=strategy.name,
        n_legs=len(strategy.legs),
        n_trades=sum(counts.values()),
        equity_r=equity_r,
        per_leg=summary_per_leg,
        combination=strategy.portfolio.combination,
        universe_symbols=list(strategy.universe.symbols),
        policy_notes=_policy_notes(strategy),
        leg_isolation_check=True,
    )


def _policy_notes(strategy: StrategySpec) -> list[str]:
    """The honest MVP-boundary notes carried into the summary."""
    notes: list[str] = []
    policy = strategy.portfolio
    n_legs = len(strategy.legs)
    if n_legs > policy.max_concurrent_positions:
        notes.append(
            f"{n_legs} legs exceed max_concurrent_positions="
            f"{policy.max_concurrent_positions}: the cap is DIAGNOSTIC"
            " ONLY in the MVP (no admission control, no replacement)"
            f" -- {_DEFERRED_NOTE}"
        )
    notes.append(
        "max_gross_exposure / max_net_exposure are NOT enforced in the"
        " MVP: records describe CLOSED trades (historical PnL), open"
        " exposure would need the Book's position ledger --"
        f" {_DEFERRED_NOTE}"
    )
    return notes


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: two-leg composition + the records bridge, asserted."""
    bars = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    md = MaterializedData(
        outcome=pd.Series(0.0, index=bars),
        features=pd.DataFrame(
            {
                "mom_signal": [2.0, -2.0, 2.5, -2.5, 2.2, -2.2],
                "open": [0.99, 1.19, 0.79, 1.09, 1.29, 0.89],
                "high": [1.21, 1.21, 0.95, 1.21, 1.31, 1.05],
                "low": [0.97, 0.99, 0.75, 0.95, 1.09, 0.85],
                "close": [1.0, 1.2, 0.8, 1.1, 1.3, 0.9],
            },
            index=bars,
        ),
        controls=pd.DataFrame(index=bars),
        index=pd.DataFrame(
            {"ts": bars, "entity": "", "split": "train"}, index=bars
        ),
    )

    def gated(name: str, direction: Literal["long", "short"]) -> LegSpec:
        gate = (
            Compare("mom_signal", ">", 1.0)
            if direction == "long"
            else Compare("mom_signal", "<", -1.0)
        )
        return LegSpec(
            name=name,
            direction=direction,
            entry=EntrySpec(gate),
            exit=ExitSpec(
                stop=StopSpec("price", 0.9), target=TargetSpec("price", 1.1)
            ),
            instrument=InstrumentRef("field", {"field": "mom_signal"}),
            features_used=("mom_signal",),
            rationale="demo leg",
        )

    strategy = StrategySpec(
        name="mirror",
        legs=(
            gated("long_gate", "long"),
            gated("short_gate", "short"),
        ),
        portfolio=PortfolioPolicy(
            borrow_bps_annual=500.0, funding_bps_annual=100.0
        ),
        universe=UniverseSpec(symbols=("synthetic",)),
    )
    print("=" * 72)
    print("strategy demo: independent two-leg composition + bridge")
    print("=" * 72)
    result = run_strategy(strategy, md)
    summary = result["summary"]
    frame = records_to_features(
        [
            record
            for records in result["records"].values()
            for record in records
        ]
    )
    print(
        f"legs={summary['n_legs']} trades={summary['n_trades']}"
        f" isolation={summary['leg_isolation_check']}"
        f" bridge_rows={len(frame)}"
    )
    assert summary["n_legs"] == 2
    assert summary["leg_isolation_check"] is True
    assert sorted(result["records"]) == ["long_gate", "short_gate"]
    assert all(
        record["leg_name"] == leg_name
        for leg_name, records in result["records"].items()
        for record in records
    )
    assert "entry::mom_signal" in frame.columns
    try:
        PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            combination="netting",
        )
        raised = False
    except ValueError as error:
        raised = "section 14" in str(error)
    assert raised


if __name__ == "__main__":
    run_all_examples()
