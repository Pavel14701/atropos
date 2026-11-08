"""Strategy-level composition: legs + portfolio policy, one shape.

A strategy is ``tuple[LegSpec, ...] + PortfolioPolicy`` -- ONE
abstraction so momentum, pairs and multi-leg options are the same
object for feature analysis.  An iron condor is 4 legs + an
independent policy + caps; a pairs trade is 2 legs (independent or
netting); a momentum system is 1 leg.  The object shape does not change with
the leg count, so every strategy's trades flatten into the same
analysis-ready frame (:func:`records_to_features`, the bridge from
"what the strategy did" to "what Atropos analyzes").

Anti-patterns this module forbids:

* **strategy-as-backtest** -- a :class:`StrategySpec` is a pure
  DECLARATION; the Book (:mod:`atropos.book`) is the only
  simulator.  Positions, cooldowns, concurrency and costs live
  there, never here.
* **multi-leg-as-special-case** -- composition is homogeneous:
  independent legs run one fresh Book each (``run_book``) so they
  cannot interact; the shared families delegate to the Book's own
  portfolio walk (``Book.walk_portfolio``).  This module adds
  policy and diagnostics, never a second simulator.

Composition paths: ``"independent"`` isolates the legs;
``"netting"`` / ``"sequential"`` / ``"priority"`` share ONE
position pool and equity ledger inside a single walk (sequential
and priority have NO offsetting -- opposites coexist; netting is
its own combination).  Exposure ENFORCEMENT (the caps are
diagnostics, not admission control) and universe enforcement
(survivorship filters, calendar gating) are deferred (KNOWN-ISSUES
section 14).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from atropos._utils import validate_int, validate_number
from atropos.backtesting import DEFAULT_COST, CostModel
from atropos.book import Book, run_book
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
from atropos.portfolio import GateReport, portfolio_gates

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

_SHARED_COMBINATIONS = ("sequential", "priority", "netting")
"""Combinations routed through Book.walk_portfolio's shared state."""

_VALID_COMBINATIONS = ("independent", *_SHARED_COMBINATIONS)
"""Every combination PortfolioPolicy accepts (validated loudly)."""


# ---------------------------------------------------------------------------
# policy + universe + strategy specs (frozen, validated at construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioPolicy:
    """How a strategy's legs share one book.

    ``combination="independent"`` runs each leg in a FRESH Book
    (:func:`atropos.book.run_book`) so legs cannot interact:
    positions from different legs COEXIST -- a long-X and a short-X
    leg both hold at the same time and each accrues its own PnL
    (this is NOT netting: no offsetting, no margin netting, no
    position merging).  ``combination="sequential"`` /
    ``"priority"`` / ``"netting"`` delegate to the ONE shared-state
    walk (:meth:`atropos.book.Book.walk_portfolio`): a single
    position pool and equity ledger -- sequential and priority have
    NO offsetting (opposites coexist), netting closes opposite
    intents.  The caps are validated here but only REPORTED as
    diagnostics by :func:`run_strategy` -- enforcement is deferred.

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
        combination: how legs combine -- ``"independent"`` (fresh
            Book per leg), ``"sequential"`` (shared walk, entry
            competition in declaration order), ``"priority"``
            (shared walk, highest ``LegSpec.priority`` first) or
            ``"netting"`` (shared walk; an entry intent opposite in
            direction to an open position on the same instrument
            closes it and is itself consumed).  Unknown values
            raise ``ValueError`` (the runtime check exists so
            dynamic callers die loudly too).
        max_concurrent_positions: soft cap on simultaneous open
            positions; must be >= 1.  Exceeding it is reported as a
            diagnostic note, never enforced in the MVP.
        max_gross_exposure: soft cap on gross open exposure in R
            units; must be finite > 0.  Not enforced in the MVP.
        max_net_exposure: soft cap on net open exposure in R units;
            must be finite > 0.  Not enforced in the MVP.

    Raises:
        ValueError: on an unknown ``combination``, a
            ``max_concurrent_positions`` below 1, or a non-finite /
            non-positive exposure cap.

    Example::

        >>> policy = PortfolioPolicy(
        ...     borrow_bps_annual=500.0, funding_bps_annual=100.0
        ... )
        >>> (policy.combination, policy.max_concurrent_positions,
        ...  policy.max_gross_exposure, policy.max_net_exposure)
        ('independent', 10, 2.0, 1.0)
        >>> netted = PortfolioPolicy(
        ...     borrow_bps_annual=500.0,
        ...     funding_bps_annual=100.0,
        ...     combination="netting",
        ... )
        >>> netted.combination
        'netting'
        >>> try:
        ...     PortfolioPolicy(
        ...         borrow_bps_annual=500.0,
        ...         funding_bps_annual=100.0,
        ...         combination="random",
        ...     )
        ... except ValueError as error:
        ...     "combination must be one of" in str(error)
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
        """Reject unknown combinations and validate the soft caps."""
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
        if self.combination not in _VALID_COMBINATIONS:
            valid = ", ".join(repr(name) for name in _VALID_COMBINATIONS)
            raise ValueError(
                "PortfolioPolicy: combination must be one of"
                f" {valid}, got {self.combination!r}"
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
    gates: bool = False,
) -> dict[str, Any]:
    """Run one strategy spec through the Book and summarize it.

    Dispatch on ``strategy.portfolio.combination``:
    ``"independent"`` hands the legs to :func:`atropos.book.run_book`,
    which runs each leg in a FRESH Book so legs cannot interact;
    ``"sequential"`` / ``"priority"`` / ``"netting"`` delegate to
    :meth:`atropos.book.Book.walk_portfolio`, the ONE shared-state
    walk (single position pool, shared equity ledger).
    ``"sequential"`` and ``"priority"`` have NO offsetting -- a long
    and a short on the same instrument both hold and each accrues
    its own PnL; ``"netting"`` is its own combination (an entry
    intent opposite to an open position on the same instrument
    closes it and is itself consumed).  The policy's caps are NOT
    enforced on either path: the leg-count cap and the per-bar
    gross/net cap breaches land in ``result["policy_notes"]`` as
    diagnostics only -- enforcement is deferred (KNOWN-ISSUES
    section 14).  Before summarizing, the records' leg attribution
    is verified (every record must belong to a declared leg) and the
    per-leg trade counts are recounted from the records against the
    run's ``per_leg`` map; the REAL isolation pin lives in the
    tests.

    Args:
        strategy: the strategy spec (legs audited at construction).
        md: the materialized bundle every leg consumes.
        cost: the cost model handed unchanged to the Book (each
            fresh Book, or the shared walk's Book).
        gates: when True, run the portfolio kill gates
            (:func:`atropos.portfolio.portfolio_gates`) with
            ``min_trades=20``, ``n_trials=5`` and
            ``prereg_hash=None``.  The gates' ``daily_ev`` input is
            CALENDAR-DAY, NOT per-bar: each trade's ``pnl_net_r`` is
            bucketed by its exit day (``exit_ts.normalize()``) and
            summed per day.  The report's ``prereg_hash`` is
            therefore None and the ``"gates UNVERIFIED (no
            prereg_hash)"`` note fires; no hash is ever fabricated.

    Returns:
        The UNIFIED result -- the same keys for every combination:
        ``"records"`` (per-leg TradeRecord lists), ``"equity_r"``
        (total net R), ``"per_leg"`` (per-leg counts and net R),
        ``"exposure"`` (``{"concurrent": [...], "net": [...]}``, one
        entry per decision bar; for the shared modes the walk's own
        authoritative series, for ``"independent"`` reconstructed
        from the closed-trade ledger where a trade occupies
        ``[entry_bar, exit_bar]`` inclusive), ``"equity_curve"``
        (per-bar cumulative net R, credited at exit bars, ``0.0``
        before the first close), ``"policy_notes"`` (the summary's
        boundary notes extended with the shared walk's warnings
        verbatim, the cap-breach diagnostics and, when ``gates``
        ran, the UNVERIFIED note; ``summary["policy_notes"]`` keeps
        its stable base list), ``"gates"`` (a
        :class:`atropos.portfolio.GateReport` when ``gates=True``,
        else ``None``) and ``"summary"`` (a
        :class:`StrategySummary`).

    Raises:
        ValueError: when leg attribution is lost (a record keyed by,
            or carrying a ``leg_name`` different from, a name outside
            ``strategy.legs``) or the run's per-leg trade counts
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
        ...         borrow_bps_annual=500.0,
        ...         funding_bps_annual=100.0,
        ...         periods_override=252.0,
        ...     ),
        ...     universe=UniverseSpec(symbols=("synthetic",)),
        ... )
        >>> result = run_strategy(strategy, md)
        >>> summary = result["summary"]
        >>> (summary["strategy_name"], summary["n_legs"])
        ('momo_pair', 2)
        >>> (summary["combination"], summary["universe_symbols"])
        ('independent', ['synthetic'])
        >>> required_summary = {
        ...     "combination", "equity_r", "leg_isolation_check",
        ...     "n_legs", "n_trades", "per_leg", "policy_notes",
        ...     "strategy_name", "universe_symbols",
        ... }
        >>> required_summary <= set(summary)
        True
        >>> required_result = {
        ...     "equity_curve", "equity_r", "exposure", "gates",
        ...     "per_leg", "policy_notes", "records", "summary",
        ... }
        >>> required_result <= set(result)
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
        >>> summary_notes = summary["policy_notes"]
        >>> isinstance(summary_notes, list) and len(summary_notes) >= 1
        True
        >>> result["gates"] is None
        True
        >>> n_bars = len(bars)
        >>> lengths = (
        ...     len(result["exposure"]["concurrent"]),
        ...     len(result["exposure"]["net"]),
        ...     len(result["equity_curve"]),
        ... )
        >>> lengths == (n_bars, n_bars, n_bars)
        True
        >>> abs(result["equity_curve"][-1] - result["equity_r"]) < 1e-9
        True
        >>> gated = run_strategy(strategy, md, gates=True)
        >>> gated["gates"]["portfolio_verdict"]
        'UNDERPOWERED'
        >>> gated["gates"]["prereg_hash"] is None
        True
        >>> "gates UNVERIFIED (no prereg_hash)" in gated["policy_notes"]
        True
    """
    policy = strategy.portfolio
    legs = list(strategy.legs)
    n_bars = len(md.index)
    walk_warnings: list[str] = []
    if policy.combination == "independent":
        book_result = run_book(
            legs,
            md,
            cost=cost,
            borrow_bps_annual=policy.borrow_bps_annual,
            funding_bps_annual=policy.funding_bps_annual,
            periods_override=policy.periods_override,
        )
        records = book_result["records"]
        per_leg = book_result["per_leg"]
        equity_r = float(book_result["equity_r"])
        concurrent, net_exposure, equity_curve = _exposure_from_records(
            records, n_bars
        )
    else:
        book = Book(
            borrow_bps_annual=policy.borrow_bps_annual,
            funding_bps_annual=policy.funding_bps_annual,
            periods_override=policy.periods_override,
            cost=cost,
        )
        book_result = book.walk_portfolio(
            legs,
            md,
            combination=policy.combination,
            max_concurrent_positions=policy.max_concurrent_positions,
            periods_override=policy.periods_override,
        )
        records = book_result["records"]
        per_leg = book_result["per_leg"]
        equity_r = float(book_result["equity_r"])
        walk_exposure = book_result["exposure"]
        concurrent = [int(count) for count in walk_exposure["concurrent"]]
        net_exposure = [float(value) for value in walk_exposure["net"]]
        equity_curve = [float(value) for value in book_result["equity_curve"]]
        walk_warnings = [str(warning) for warning in book_result["warnings"]]
    counts = _recount_and_verify(strategy, records, per_leg)
    # Gross open exposure is reconstructed from the closed-trade
    # ledger in BOTH branches: the shared walk's exposure carries
    # concurrent/net only, and those two stay authoritative there.
    gross = _open_series(records, n_bars)[2]
    gate_report: GateReport | None = None
    if gates:
        gate_report = portfolio_gates(
            equity_r=equity_curve,
            daily_ev=_daily_ev(records).to_numpy(dtype=np.float64),
            concurrent=concurrent,
            net_exposure=net_exposure,
            min_trades=20,
            n_trials=5,
            prereg_hash=None,
        )
    summary = _build_summary(strategy, counts, per_leg, equity_r)
    policy_notes = _policy_notes(strategy)
    policy_notes.extend(walk_warnings)
    policy_notes.extend(_cap_breach_notes(policy, gross, net_exposure))
    if gate_report is not None:
        policy_notes.append("gates UNVERIFIED (no prereg_hash)")
    return {
        "records": records,
        "equity_r": equity_r,
        "per_leg": per_leg,
        "exposure": {"concurrent": concurrent, "net": net_exposure},
        "equity_curve": equity_curve,
        "policy_notes": policy_notes,
        "gates": gate_report,
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

    Both dispatch paths key records by leg name (a fresh Book per
    leg, or the shared walk's one pool); here the records must still
    carry honest attribution: a record stored under one leg's name
    must name exactly that leg, and the run's ``per_leg`` counts
    must match the recount.

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
# private helpers (exposure + gates inputs from the closed-trade ledger)
# ---------------------------------------------------------------------------
def _open_series(
    records_by_leg: dict[str, list[dict[str, Any]]],
    n_bars: int,
) -> tuple[list[int], list[float], list[float]]:
    """Per-bar open-position series rebuilt from CLOSED-trade records.

    A trade occupies ``[entry_bar, exit_bar]`` inclusive (its
    ``entry_bar`` / ``exit_bar`` keys, positions on the decision
    grid).  ``concurrent`` counts the open positions at each bar,
    ``net`` sums ``direction * size_r`` over them (``direction`` is
    ``+1`` long / ``-1`` short) and ``gross`` sums ``abs(size_r)``
    over them.  This is a RECONSTRUCTION from the closed-trade
    ledger, not the walk's live ledger -- the shared modes keep
    their own walk's concurrent/net series as authoritative.

    Returns:
        ``(concurrent, net, gross)`` -- one entry per bar, zeros
        where nothing is open.
    """
    concurrent = [0] * n_bars
    net = [0.0] * n_bars
    gross = [0.0] * n_bars
    for leg_records in records_by_leg.values():
        for record in leg_records:
            entry_bar = int(record["entry_bar"])
            exit_bar = int(record["exit_bar"])
            signed = int(record["direction"]) * float(record["size_r"])
            for bar in range(entry_bar, exit_bar + 1):
                concurrent[bar] += 1
                net[bar] += signed
                gross[bar] += abs(signed)
    return concurrent, net, gross


def _equity_curve_from_records(
    records_by_leg: dict[str, list[dict[str, Any]]],
    n_bars: int,
) -> list[float]:
    """Per-bar cumulative net R, credited at each trade's exit bar.

    Matches the shared walk's equity-curve shape: ``curve[i]`` is
    the cumulative ``pnl_net_r`` AFTER bar ``i`` (a close credits
    its bar), ``0.0`` before the first close.
    """
    credited = [0.0] * n_bars
    for leg_records in records_by_leg.values():
        for record in leg_records:
            credited[int(record["exit_bar"])] += float(record["pnl_net_r"])
    curve: list[float] = []
    running = 0.0
    for pnl in credited:
        running += pnl
        curve.append(running)
    return curve


def _exposure_from_records(
    records_by_leg: dict[str, list[dict[str, Any]]],
    n_bars: int,
) -> tuple[list[int], list[float], list[float]]:
    """The independent-mode per-bar exposure triple from records.

    Composes :func:`_open_series` and
    :func:`_equity_curve_from_records` into
    ``(concurrent, net, equity_curve)`` -- the same series shape the
    shared walk reports, so :func:`run_strategy` can hand one
    unified result contract to consumers of either composition
    path.
    """
    concurrent, net, _gross = _open_series(records_by_leg, n_bars)
    equity_curve = _equity_curve_from_records(records_by_leg, n_bars)
    return concurrent, net, equity_curve


def _daily_ev(
    records_by_leg: dict[str, list[dict[str, Any]]],
) -> pd.Series:
    """Calendar-day aggregation of closed-trade net R.

    Each trade's ``pnl_net_r`` is bucketed by its ``exit_ts``
    normalized to midnight (``ts.normalize()``), summed per day and
    ordered by day -- the ``daily_ev`` stream the portfolio gates
    consume is calendar-day, NOT per-bar.

    Returns:
        A ``pd.Series`` indexed by normalized day Timestamps whose
        values are the summed ``pnl_net_r``; empty records give an
        empty float series.
    """
    day_totals: dict[pd.Timestamp, float] = {}
    for leg_records in records_by_leg.values():
        for record in leg_records:
            day = pd.Timestamp(record["exit_ts"]).normalize()
            day_totals[day] = day_totals.get(day, 0.0) + float(
                record["pnl_net_r"]
            )
    if not day_totals:
        return pd.Series(dtype=np.float64)
    ordered_days = sorted(day_totals)
    return pd.Series(
        [day_totals[day] for day in ordered_days],
        index=pd.DatetimeIndex(ordered_days),
        dtype=np.float64,
    )


def _cap_breach_notes(
    policy: PortfolioPolicy,
    gross: list[float],
    net: list[float],
) -> list[str]:
    """Soft-cap breach diagnostics (reporting only, no enforcement).

    A note fires only when the series' per-bar peak EXCEEDS the
    policy cap (strictly); the bar count is the number of bars
    strictly above the cap.  Enforcement stays deferred.
    """
    notes: list[str] = []
    gross_peak = max(gross, default=0.0)
    if gross_peak > policy.max_gross_exposure:
        gross_bars = sum(
            1 for value in gross if value > policy.max_gross_exposure
        )
        notes.append(
            f"cap breach: gross {gross_peak:.2f}"
            f" > {policy.max_gross_exposure:.2f}"
            f" on {gross_bars} bars"
        )
    net_peak = max(net, default=0.0)
    if net_peak > policy.max_net_exposure:
        net_bars = sum(1 for value in net if value > policy.max_net_exposure)
        notes.append(
            f"cap breach: net {net_peak:.2f}"
            f" > {policy.max_net_exposure:.2f}"
            f" on {net_bars} bars"
        )
    return notes


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: independent + netting two-leg composition, asserted."""
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
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
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
    n_bars = len(md.index)
    assert len(result["equity_curve"]) == n_bars
    assert len(result["exposure"]["concurrent"]) == n_bars
    assert len(result["exposure"]["net"]) == n_bars
    assert result["gates"] is None
    netted_strategy = StrategySpec(
        name="mirror_netted",
        legs=(
            gated("long_gate", "long"),
            gated("short_gate", "short"),
        ),
        portfolio=PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
            combination="netting",
        ),
        universe=UniverseSpec(symbols=("synthetic",)),
    )
    print("strategy demo: the same mirror world under 'netting'")
    netted = run_strategy(netted_strategy, md)
    netted_summary = netted["summary"]
    print(
        f"legs={netted_summary['n_legs']}"
        f" trades={netted_summary['n_trades']}"
        f" combination={netted_summary['combination']}"
        f" equity_r={netted_summary['equity_r']:.4f}"
    )
    assert netted_summary["combination"] == "netting"
    assert len(netted["exposure"]["concurrent"]) == n_bars
    assert len(netted["exposure"]["net"]) == n_bars
    assert len(netted["equity_curve"]) == n_bars
    assert netted["gates"] is None


if __name__ == "__main__":
    run_all_examples()
