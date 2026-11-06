"""Multi-slot portfolio layer: replay, correlation admission, gates.

Where :mod:`atropos.backtesting` answers "what is ONE signal worth"
and "which signals survive ONE occupied slot", this module generalizes
the slot discipline to N concurrent positions and adds the two
portfolio-level controls the single-slot engine cannot express:

* admission under a slot cap with replacement -- signals arrive in
  deterministic ``(decision_idx, priority desc, side, input position)``
  order; a candidate takes a free slot, or (under the ``replace_low``
  / ``replace_high`` policies) displaces the WORST open position when
  its rank-normalized score strictly beats it.  The displaced trade is
  closed at the replacement bar and its ledger row carries
  ``force_close_r`` -- the pessimistic P&L RECOMPUTED to that bar via
  :func:`atropos.backtesting.simulate_signal` (a partial hold; the
  row's isolated ``r_pess``/``exit_idx`` stay stale, mirroring the
  engine's force-close contract).  The equity CREDIT follows the
  policy: ``replace_low`` credits the displaced trade exactly 0 (a
  mid-flight close is assumed to give back), ``replace_high`` keeps
  its full isolated ``r_pess`` (optimistic bound) -- both bounds
  motivated by the donor finding that FCFS capping rejects BETTER
  trades.
* correlation-aware admission -- a candidate whose recent R proxy is
  |Spearman|-correlated above ``corr_cap`` with ANY open position's
  proxy is rejected (reason ``"correlation_cap"``); insufficient
  overlap history is inconclusive and never rejects by itself
  (not-enough-data is not correlation).

Realized equity is EVENT-based (R credited at each trade's credit
bar), so drawdown is not flattered by aggregation.  Portfolio gates
close the loop: the deflated Sharpe of the per-period portfolio EV
stream -- DELEGATED to
:func:`atropos.evaluation_metrics.deflated_sharpe_from_moments`, the
single home of the formula -- plus the drawdown floor and the
same-side concentration share, with the verdict ``SURVIVES`` /
``DEAD(<gate>)`` / ``UNDERPOWERED``.

Score contract: candidates from different model groups are compared
only on WITHIN-GROUP rank-normalized scores
(:func:`atropos.ranking.rank_normalize_group`); pass a
``CandidateGroup`` frame's ``normalized_scores`` column as ``score``
-- never raw cross-model probabilities.

Run ``python -m atropos.portfolio`` for a synthetic demo.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr

from atropos._utils import validate_int, validate_number
from atropos.backtesting import (
    DEFAULT_COST,
    HOLD_BARS,
    CostModel,
    simulate_signal,
)
from atropos.evaluation_metrics import deflated_sharpe_from_moments

__all__ = [
    "ExposureSeries",
    "GateReport",
    "GateVerdict",
    "PortfolioResult",
    "PortfolioSummary",
    "SlotAttribution",
    "filter_by_correlation",
    "portfolio_gates",
    "run_portfolio",
    "run_portfolio_replay",
]

_PORTFOLIO_POLICIES = ("fcfs", "replace_low", "replace_high")
"""Admission policies: capacity rejection vs score-based replacement."""

_PRICE_FIELDS = ("open", "high", "low", "close")
"""OHLC fields every panel instrument must provide (level 0 of the
price column MultiIndex)."""

_SIGNAL_COLUMNS = ("decision_idx", "side", "instrument", "sl", "tp", "score")
"""Required signal columns; ``priority``/``atr_entry``/``risk_ref``/
``r_proxy`` are optional."""

_MIN_SKEW_OBS = 8
"""Minimum observations for honest skew/kurtosis (the
``deflated_sharpe`` convention; below it the moments are NaN)."""

_NO_CREDIT = 0.0
"""Credit of an unresolved trade (no exit inside the hold window) and
of a displaced trade under ``replace_low``."""


# ---------------------------------------------------------------------------
# typed result payloads
# ---------------------------------------------------------------------------
class ExposureSeries(TypedDict):
    """Per-bar exposure of a replay.

    Keys: ``gross`` (held unit positions per bar), ``net`` (signed
    sum: longs minus shorts) and ``concurrent`` (identical to
    ``gross`` -- each position enters at exactly one risk unit).
    All three are float64 arrays as long as the price panel.
    """

    gross: np.ndarray
    net: np.ndarray
    concurrent: np.ndarray


class SlotAttribution(TypedDict):
    """Per-slot attribution row of a replay.

    Keys: ``slot`` (slot index), ``trades`` (trades that occupied the
    slot, displaced ones included) and ``ev_r`` (sum of their equity
    credits).
    """

    slot: int
    trades: int
    ev_r: float


class PortfolioSummary(TypedDict):
    """Compact report row of a replay (the ``summary`` key).

    ``ev`` is the final event equity, ``dd`` the event-basis max
    drawdown (both in R), ``concurrent_max`` the busiest bar,
    ``n_correlation_rejects`` the ``"correlation_cap"`` rejects and
    ``n_unresolved`` the accepted trades with no exit inside the data.
    """

    slots: int
    policy: str
    n_accepted: int
    n_rejected: int
    n_replaced: int
    n_correlation_rejects: int
    n_unresolved: int
    ev: float
    dd: float
    concurrent_max: int


class GateVerdict(TypedDict):
    """One portfolio gate outcome.

    Keys: ``gate`` (name), ``triggered``, ``value`` (the observed
    statistic; NaN when the gate had no data), ``floor`` (the
    preregistered bound) and ``reason`` (human-readable sentence).
    """

    gate: str
    triggered: bool
    value: float
    floor: float
    reason: str


class GateReport(TypedDict):
    """Result of :func:`portfolio_gates`.

    ``gates`` holds the gate verdicts in fixed order (deflated Sharpe,
    drawdown floor, side concentration); ``portfolio_verdict`` is
    ``"SURVIVES"``, ``"DEAD(<gate>)"`` for the first triggered gate or
    ``"UNDERPOWERED"`` below ``min_trades`` credited trades.
    """

    gates: list[GateVerdict]
    portfolio_verdict: str
    n_trades: int
    min_trades: int


class PortfolioResult(TypedDict):
    """Full result of a replay (and of :func:`run_portfolio`).

    Keys: ``slots`` / ``policy`` (echoed config), ``accepted`` (ledger
    rows: input columns + ``slot``/``fill_idx``/``exit_idx``/
    ``r_pess``/``credit_bar``/``credit_r``/``replaced``/
    ``force_exit_idx``/``force_close_r``), ``rejected`` (input columns
    + ``reason``), ``replaced`` (ledger rows + ``reason="replaced"``),
    ``per_slot`` attribution, ``equity_curve`` (event-based cumulative
    R), ``max_drawdown_r`` / ``final_equity_r``, ``exposure``
    (:class:`ExposureSeries`), ``summary``
    (:class:`PortfolioSummary`), ``gates`` (empty list from the raw
    replay; populated by :func:`run_portfolio`) and
    ``portfolio_verdict`` (``"not_evaluated"`` from the raw replay).
    """

    slots: int
    policy: str
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    replaced: list[dict[str, Any]]
    per_slot: list[SlotAttribution]
    equity_curve: np.ndarray
    max_drawdown_r: float
    final_equity_r: float
    exposure: ExposureSeries
    summary: PortfolioSummary
    gates: list[GateVerdict]
    portfolio_verdict: str


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------
def _validated_price_panel(
    prices: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Validate the price panel into ``{instrument: {field: array}}``.

    ``prices`` must be a DataFrame whose columns form a 2-level
    MultiIndex ``(field, instrument)`` with ``field`` one of
    ``open``/``high``/``low``/``close``: one OHLC quadruple per
    instrument on ONE shared positional bar grid.  All arrays are
    finite float64 and share one length >= 2 (a signal needs a
    decision bar and a fill bar).

    Returns:
        ``(panel, n_bars)`` with instruments keyed by their string
        level-1 values.

    Raises:
        ValueError: on a non-DataFrame input, missing/wrong MultiIndex,
            unknown or duplicate fields, a non-finite array, unequal
            bar-grid lengths, a missing OHLC field, or fewer than two
            bars.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    columns = prices.columns
    if not isinstance(columns, pd.MultiIndex) or columns.nlevels != 2:
        raise ValueError(
            "prices columns must be a 2-level (field, instrument)"
            " MultiIndex; build it e.g. with"
            " pd.MultiIndex.from_product([['open', 'high', 'low',"
            " 'close'], instruments])"
        )
    if len(set(map(tuple, columns))) != len(columns):
        raise ValueError(
            "prices columns must not repeat a (field, instrument) pair"
        )
    panel: dict[str, dict[str, np.ndarray]] = {}
    sizes: set[int] = set()
    for column in columns:
        field = str(column[0])
        instrument = str(column[1])
        if field not in _PRICE_FIELDS:
            raise ValueError(
                f"prices field must be one of {_PRICE_FIELDS}, got {field!r}"
            )
        values = np.asarray(prices[column], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"prices[{field!r}, {instrument!r}] must be 1-D, got"
                f" ndim={values.ndim}"
            )
        if values.size and not bool(np.all(np.isfinite(values))):
            raise ValueError(
                f"prices[{field!r}, {instrument!r}] must be finite, got"
                " NaN/inf entries"
            )
        panel.setdefault(instrument, {})[field] = values
        sizes.add(int(values.size))

    if not panel:
        raise ValueError("prices panel has no instruments")
    for instrument, fields in panel.items():
        missing = [name for name in _PRICE_FIELDS if name not in fields]
        if missing:
            raise ValueError(
                f"instrument {instrument!r} is missing price fields {missing}"
            )
    if len(sizes) != 1:
        raise ValueError(
            f"all instruments must share one bar grid, got lengths"
            f" {sorted(sizes)}"
        )
    n_bars = sizes.pop()
    if n_bars < 2:
        raise ValueError(
            f"prices need at least 2 bars (decision + fill), got {n_bars}"
        )
    return panel, n_bars


def _validated_proxy_series(values: Any, name: str) -> np.ndarray:
    """Coerce one R-proxy cell to a finite 1-D float array.

    Raises:
        ValueError: if the value cannot be coerced or is non-finite.
    """
    proxy = np.asarray(values, dtype=np.float64).ravel()
    if proxy.size and not bool(np.all(np.isfinite(proxy))):
        raise ValueError(f"{name} must be finite, got NaN/inf entries")
    return proxy


def _validated_portfolio_signals(
    signals_df: pd.DataFrame,
    panel: dict[str, dict[str, np.ndarray]],
    n_bars: int,
    *,
    corr_enabled: bool,
) -> list[dict[str, Any]]:
    """Validate the signal frame into normalized record dicts.

    Required columns: ``decision_idx`` (int >= 0), ``side``
    (``"long"``/``"short"``), ``instrument`` (panel key), ``sl`` /
    ``tp`` (finite levels) and ``score`` (finite, within-group
    rank-normalized upstream).  Optional: ``priority`` (default 0.0),
    ``atr_entry`` (default 0.0), ``risk_ref`` (default None) and
    ``r_proxy`` (the instrument's GROWING R-proxy series from one
    common anchor through ``decision_idx`` -- e.g. its returns since
    bar 0 -- consumed by the correlation filter, which extends open
    positions' snapshots to the candidate's bar from the candidate's
    own cell; required when ``corr_enabled``).

    The generator rule (fill at ``decision_idx + 1``) requires
    ``decision_idx + 1 <= n_bars - 1``.

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
    if corr_enabled and "r_proxy" not in signals_df.columns:
        raise ValueError(
            "corr_cap requires an 'r_proxy' column on signals_df: the"
            " candidate's recent R-proxy series (e.g. trailing"
            " instrument returns) compared against open positions"
        )
    has_priority = "priority" in signals_df.columns
    has_atr = "atr_entry" in signals_df.columns
    has_risk_ref = "risk_ref" in signals_df.columns
    has_proxy = "r_proxy" in signals_df.columns
    records: list[dict[str, Any]] = []
    for position, (_, raw_row) in enumerate(signals_df.iterrows()):
        decision = validate_int(
            "decision_idx", raw_row["decision_idx"], minimum=0
        )
        fill_idx = decision + 1
        if fill_idx > n_bars - 1:
            raise ValueError(
                f"signals row {position}: decision_idx={decision} fills"
                f" at bar {fill_idx} past the data (n_bars={n_bars})"
            )
        side = str(raw_row["side"])
        if side not in ("long", "short"):
            raise ValueError(
                f"signals row {position}: side must be 'long' or"
                f" 'short', got {raw_row['side']!r}"
            )
        instrument = str(raw_row["instrument"])
        if instrument not in panel:
            raise ValueError(
                f"signals row {position}: instrument {instrument!r} is"
                " not in the price panel"
            )
        record: dict[str, Any] = {
            "position": position,
            "decision_idx": decision,
            "fill_idx": fill_idx,
            "side": side,
            "instrument": instrument,
            "sl": validate_number("sl", raw_row["sl"]),
            "tp": validate_number("tp", raw_row["tp"]),
            "score": validate_number("score", raw_row["score"]),
            "priority": (
                validate_number("priority", raw_row["priority"])
                if has_priority
                else 0.0
            ),
            "atr_entry": (
                validate_number("atr_entry", raw_row["atr_entry"])
                if has_atr
                else 0.0
            ),
            "risk_ref": None,
            "r_proxy": None,
            "row": raw_row.to_dict(),
        }
        if has_risk_ref and raw_row["risk_ref"] is not None:
            record["risk_ref"] = validate_number(
                "risk_ref", raw_row["risk_ref"]
            )
        if has_proxy:
            record["r_proxy"] = _validated_proxy_series(
                raw_row["r_proxy"], f"signals row {position} r_proxy"
            )
            if record["r_proxy"].size == 0:
                raise ValueError(
                    f"signals row {position}: r_proxy must be non-empty"
                )
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# correlation-aware admission
# ---------------------------------------------------------------------------
def _stream_correlated(
    candidate_tail: np.ndarray,
    open_proxy: np.ndarray,
    *,
    window_bars: int,
    overlap_min: int,
    cap: float,
) -> bool | None:
    """|Spearman| verdict for ONE candidate-vs-open-position pair.

    ``True`` = correlated above the cap, ``False`` = within the cap,
    ``None`` = inconclusive (fewer than ``overlap_min`` shared bars, a
    constant degenerate proxy, or a non-finite correlation).  Both
    series are truncated to their most recent ``window_bars`` before
    the tail-aligned overlap is formed.
    """
    open_tail = open_proxy[-window_bars:]
    overlap = min(candidate_tail.size, open_tail.size)
    if overlap < overlap_min:
        return None
    candidate_view = candidate_tail[-overlap:]
    open_view = open_tail[-overlap:]
    if (
        float(np.std(candidate_view)) <= 0.0
        or float(np.std(open_view)) <= 0.0
    ):
        return None
    correlation = float(spearmanr(candidate_view, open_view).statistic)
    if not math.isfinite(correlation):
        return None
    return bool(abs(correlation) > cap)


def filter_by_correlation(
    signals_df: pd.DataFrame,
    candidate_idx: int,
    open_positions_r_streams: Mapping[Any, Any],
    *,
    corr_cap: float = 0.7,
    min_overlap: int = 20,
    window: int = 60,
) -> bool:
    """Admit or reject one candidate against the OPEN positions' proxies.

    Computes ``|Spearman(candidate recent R proxy, open position R
    proxy)|`` over the overlapping recent window for EVERY open
    position stream; a magnitude STRICTLY above ``corr_cap`` for any
    one of them rejects the candidate (the caller records reason
    ``"correlation_cap"``).  Returns ``True`` = admit (no open
    position correlates above the cap), ``False`` = reject.

    Inconclusive is never a reject: a pair with fewer than
    ``min_overlap`` overlapping bars, a constant (degenerate) proxy,
    or a non-finite correlation is skipped -- not-enough-data is not
    correlation.  An empty stream mapping (no open positions) admits.

    The R proxies are caller-built per-bar series on ONE common anchor
    and bar grid (typically the instrument's trailing returns): the
    candidate's series lives in its ``signals_df`` row under
    ``r_proxy``, the open positions' series are the mapping values
    (keys are opaque position labels, kept for caller bookkeeping
    only).  Spearman (rank) correlation is deliberate: it is scale
    free, so proxies from differently priced instruments compare
    fairly.  Alignment is by TAIL (the most recent
    ``min(len_candidate, len_stream)`` bars of each): streams that END
    at the same bar pair bar-for-bar, while stale snapshots ending
    earlier pair different days and dilute the verdict toward
    inconclusive -- they never fabricate a reject.
    :func:`run_portfolio_replay` keeps its comparisons bar-for-bar by
    extending every open position's snapshot to the candidate's
    decision bar from the candidate's own cell (shared anchor), so
    under the replay contract ("growing series from a common anchor
    through the row's decision bar") the verdicts are time-aligned by
    construction.

    Args:
        signals_df: signal frame; needs the ``instrument`` and
            ``r_proxy`` columns (the candidate row is
            ``signals_df.iloc[candidate_idx]``).
        candidate_idx: positional index of the candidate row.
        open_positions_r_streams: mapping position-label -> R-proxy
            series of each OPEN position.
        corr_cap: ``|Spearman|`` rejection threshold in ``[0, 1]``.
        min_overlap: minimum overlapping bars for a verdict, >= 2.
        window: recent bars each proxy is truncated to, >= 1.

    Returns:
        ``True`` = admit, ``False`` = reject (some open stream
        correlates above the cap).

    Raises:
        ValueError: on a non-DataFrame ``signals_df``, a missing
            ``instrument``/``r_proxy`` column, an out-of-range
            ``candidate_idx``, a non-finite proxy series, or an
            invalid ``corr_cap`` / ``min_overlap`` / ``window``.

    Example::

        >>> proxy = np.array([0.01, -0.02, 0.03, -0.01, 0.02,
        ...                   -0.03, 0.01, 0.02])
        >>> other = np.array([0.02, 0.01, -0.01, 0.03, -0.02, 0.01,
        ...                   -0.03, 0.01])
        >>> frame = pd.DataFrame(
        ...     {"instrument": ["btc", "eth", "gold"],
        ...      "r_proxy": [proxy, proxy * 1.1, other]}
        ... )
        >>> streams = {"open_1": proxy}
        >>> filter_by_correlation(frame, 1, streams, min_overlap=4)
        False
        >>> filter_by_correlation(frame, 2, streams, min_overlap=4)
        True
        >>> filter_by_correlation(frame, 2, streams, min_overlap=50)
        True
    """
    cap = validate_number("corr_cap", corr_cap)
    if not 0.0 <= cap <= 1.0:
        raise ValueError(f"corr_cap must lie in [0, 1], got {corr_cap}")
    overlap_min = validate_int("min_overlap", min_overlap, minimum=2)
    window_bars = validate_int("window", window, minimum=1)
    candidate_position = validate_int(
        "candidate_idx", candidate_idx, minimum=0
    )
    if not isinstance(signals_df, pd.DataFrame):
        raise ValueError("signals_df must be a pandas DataFrame")
    for column in ("instrument", "r_proxy"):
        if column not in signals_df.columns:
            raise ValueError(
                f"signals_df is missing the required {column!r} column"
            )
    if candidate_position >= len(signals_df):
        raise ValueError(
            f"candidate_idx={candidate_position} is past the frame"
            f" (len={len(signals_df)})"
        )
    streams = list(open_positions_r_streams.values())
    if not streams:
        return True
    candidate_proxy = _validated_proxy_series(
        signals_df.iloc[candidate_position]["r_proxy"],
        "candidate r_proxy",
    )
    if candidate_proxy.size == 0:
        return True
    candidate_tail = candidate_proxy[-window_bars:]
    for stream in streams:
        open_proxy = _validated_proxy_series(stream, "open r_proxy")
        if open_proxy.size == 0:
            continue
        verdict = _stream_correlated(
            candidate_tail,
            open_proxy,
            window_bars=window_bars,
            overlap_min=overlap_min,
            cap=cap,
        )
        if verdict is True:
            return False
    return True


# ---------------------------------------------------------------------------
# replay mechanics
# ---------------------------------------------------------------------------
def _slot_status(
    holder: dict[str, Any], decision: int, cooldown_bars: int
) -> str:
    """Classify one occupied slot at the candidate's decision bar.

    ``"occupied"`` while the position is held (its exit bar included,
    SL-first pessimism), ``"cooldown"`` for the blocked decision bars
    after the exit, ``"free"`` once the slot can take a new fill.  An
    open-ended trade (no exit inside the data) occupies forever.
    """
    if holder["open_ended"] or decision <= holder["exit_idx"]:
        return "occupied"
    if decision < holder["exit_idx"] + cooldown_bars:
        return "cooldown"
    return "free"


def _simulate_admission(
    record: dict[str, Any],
    panel: dict[str, dict[str, np.ndarray]],
    *,
    slot: int,
    hold_bars: int,
    cost: CostModel,
    n_bars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simulate one admitted candidate; return ``(holder, ledger_row)``.

    The ledger row extends the input row with the portfolio fields;
    ``exit_idx == -1`` (no exit inside the hold window) means the
    trade occupies its slot until the end of the data and will credit
    :data:`_NO_CREDIT` (no exit observed -> nothing realized).
    ``credit_bar`` is the original exit bar (donor parity: even a
    later-displaced trade credits at its own exit bar).
    """
    ohlc = panel[record["instrument"]]
    sim = simulate_signal(
        ohlc["open"],
        ohlc["high"],
        ohlc["low"],
        ohlc["close"],
        record["fill_idx"],
        record["side"],
        record["sl"],
        record["tp"],
        hold_bars,
        record["atr_entry"],
        risk_ref=record["risk_ref"],
        cost=cost,
    )
    exit_idx = int(sim["exit_idx"])
    open_ended = exit_idx == -1
    ledger_row: dict[str, Any] = dict(record["row"])
    ledger_row.update(
        {
            "slot": int(slot),
            "fill_idx": int(record["fill_idx"]),
            "exit_idx": exit_idx,
            "r_pess": float(sim["r_pess"]),
            "credit_bar": n_bars - 1 if open_ended else exit_idx,
            "credit_r": float("nan"),
            "replaced": False,
            "force_exit_idx": float("nan"),
            "force_close_r": float("nan"),
        }
    )
    holder: dict[str, Any] = {
        "slot": int(slot),
        "rec": record,
        "row": ledger_row,
        "exit_idx": exit_idx,
        "open_ended": open_ended,
    }
    return holder, ledger_row


def _reconstructed_close_r(
    holder: dict[str, Any],
    close_bar: int,
    panel: dict[str, dict[str, np.ndarray]],
    *,
    cost: CostModel,
) -> float:
    """Recompute an open trade's pessimistic P&L up to ``close_bar``.

    Runs :func:`simulate_signal` from the ORIGINAL fill bar with the
    same stop/target and the hold window truncated so the time exit
    lands on ``close_bar``; if the stop or target fired earlier the
    scan returns that earlier exit (the truthful path).  A close bar
    before the fill means the trade never existed yet: exactly 0.0.
    """
    record = holder["rec"]
    fill_idx = record["fill_idx"]
    if close_bar < fill_idx:
        return 0.0
    ohlc = panel[record["instrument"]]
    result = simulate_signal(
        ohlc["open"],
        ohlc["high"],
        ohlc["low"],
        ohlc["close"],
        fill_idx,
        record["side"],
        record["sl"],
        record["tp"],
        close_bar - fill_idx + 1,
        record["atr_entry"],
        risk_ref=record["risk_ref"],
        cost=cost,
    )
    return float(result["r_pess"])


def _position_streams(
    record: dict[str, Any], occupied_holders: list[dict[str, Any]]
) -> dict[int, np.ndarray]:
    """Open-position R proxies, each extended to the candidate's bar.

    Under the replay contract every ``r_proxy`` cell is a growing
    per-bar series from one common anchor, ending at ITS row's
    decision bar.  An occupied position's snapshot ends at its own
    admission decision, so the candidate's cell supplies the bars
    since then (same instrument, same anchor) and both streams end at
    the candidate's decision bar -- the filter's tail alignment then
    pairs identical bars instead of different days.
    """
    candidate_proxy: np.ndarray = record["r_proxy"]
    streams: dict[int, np.ndarray] = {}
    for holder in occupied_holders:
        rec = holder["rec"]
        snapshot: np.ndarray = rec["r_proxy"]
        stale = candidate_proxy.size - snapshot.size
        if stale > 0:
            snapshot = np.concatenate([snapshot, candidate_proxy[-stale:]])
        elif stale < 0:
            # cannot happen under the sorted replay (positions are
            # admitted at earlier-or-equal decisions); truncate rather
            # than pair different days
            snapshot = snapshot[: candidate_proxy.size]
        streams[rec["position"]] = snapshot
    return streams


def _admit_free_slot(
    record: dict[str, Any],
    holders: list[dict[str, Any] | None],
    free_idx: list[int],
    occupied_holders: list[dict[str, Any]],
    *,
    signals_df: pd.DataFrame,
    panel: dict[str, dict[str, np.ndarray]],
    corr_cap: float | None,
    corr_min_overlap: int,
    corr_window: int,
    hold_bars: int,
    cost: CostModel,
    n_bars: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Free-slot path: correlation filter, then admit.

    Returns ``(ledger_row, rejected_row, correlation_rejects)`` --
    exactly one of the first two is set.  The correlation filter (when
    enabled) consults every OCCUPIED slot's R proxy; the correlation
    verdict runs BEFORE any policy decision, so a correlated candidate
    never enters, free slot or not.  The lowest free slot index wins
    (deterministic slot assignment).
    """
    if corr_cap is not None and occupied_holders:
        admitted = filter_by_correlation(
            signals_df,
            record["position"],
            _position_streams(record, occupied_holders),
            corr_cap=corr_cap,
            min_overlap=corr_min_overlap,
            window=corr_window,
        )
        if not admitted:
            return (
                None,
                dict(record["row"]) | {"reason": "correlation_cap"},
                1,
            )
    target = free_idx[0]
    holder, ledger_row = _simulate_admission(
        record,
        panel,
        slot=target,
        hold_bars=hold_bars,
        cost=cost,
        n_bars=n_bars,
    )
    holders[target] = holder
    return ledger_row, None, 0


def _admit_full_policy(
    record: dict[str, Any],
    holders: list[dict[str, Any] | None],
    occupied_holders: list[dict[str, Any]],
    *,
    signals_df: pd.DataFrame,
    panel: dict[str, dict[str, np.ndarray]],
    policy: str,
    corr_cap: float | None,
    corr_min_overlap: int,
    corr_window: int,
    hold_bars: int,
    cost: CostModel,
    n_bars: int,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    int,
]:
    """No free slot: FCFS rejection or score-based replacement.

    Returns ``(ledger_row, rejected_row, displaced_row,
    correlation_rejects)``.  The correlation filter runs first (against
    the occupied slots); only a correlation-clean candidate may
    contest the worst open holder.  Replacement requires a STRICTLY
    higher score (ties never displace); the worst holder is the lowest
    ``score``, ties broken by input position.  The displaced trade's
    ledger row is flagged in place (``replaced``/``force_exit_idx``/
    ``force_close_r``) and returned so the replay can list it under
    ``replaced``.
    """
    if corr_cap is not None:
        admitted = filter_by_correlation(
            signals_df,
            record["position"],
            _position_streams(record, occupied_holders),
            corr_cap=corr_cap,
            min_overlap=corr_min_overlap,
            window=corr_window,
        )
        if not admitted:
            return (
                None,
                dict(record["row"]) | {"reason": "correlation_cap"},
                None,
                1,
            )
    if policy == "fcfs":
        return None, dict(record["row"]) | {"reason": "capacity"}, None, 0
    worst = min(
        occupied_holders,
        key=lambda holder: (
            holder["rec"]["score"],
            holder["rec"]["position"],
        ),
    )
    if record["score"] <= worst["rec"]["score"]:
        return (
            None,
            dict(record["row"]) | {"reason": "inferior_score"},
            None,
            0,
        )
    decision = record["decision_idx"]
    worst["row"]["replaced"] = True
    worst["row"]["force_exit_idx"] = float(decision)
    worst["row"]["force_close_r"] = _reconstructed_close_r(
        worst, decision, panel, cost=cost
    )
    holder, ledger_row = _simulate_admission(
        record,
        panel,
        slot=worst["slot"],
        hold_bars=hold_bars,
        cost=cost,
        n_bars=n_bars,
    )
    holders[worst["slot"]] = holder
    return ledger_row, None, worst["row"], 0


def _policy_credit(row: dict[str, Any], policy: str) -> float:
    """Equity credit of one accepted ledger row under ``policy``.

    Ordinary trades credit their isolated pessimistic R at their exit
    bar.  A displaced trade credits :data:`_NO_CREDIT` under
    ``replace_low`` (mid-flight close assumed to give back) and keeps
    its full isolated R under ``replace_high`` -- in both cases the
    honest partial-hold P&L it actually realized sits on its
    ``force_close_r`` field, never overwriting the policy credit.  A
    trade with no exit inside the data credits nothing (unresolved).
    """
    if row["replaced"]:
        return _NO_CREDIT if policy == "replace_low" else float(row["r_pess"])
    if int(row["exit_idx"]) == -1:
        return _NO_CREDIT
    return float(row["r_pess"])


def _exposure_series(
    accepted: list[dict[str, Any]], n_bars: int
) -> ExposureSeries:
    """Per-bar gross / net / concurrent exposure of the accepted trades.

    A trade occupies ``[fill_idx, effective_exit]`` inclusive, where a
    displaced trade ends at its ``force_exit_idx`` and an unresolved
    trade at the last bar; the successor of a displaced trade fills on
    the very next bar, so the two never overlap.  Both series are
    bounded by the slot count (one risk unit per position).
    """
    gross_delta = np.zeros(n_bars, dtype=np.float64)
    net_delta = np.zeros(n_bars, dtype=np.float64)
    for row in accepted:
        sign = 1.0 if str(row["side"]) == "long" else -1.0
        if row["replaced"]:
            end_bar = int(row["force_exit_idx"])
        elif int(row["exit_idx"]) == -1:
            end_bar = n_bars - 1
        else:
            end_bar = int(row["exit_idx"])
        gross_delta[int(row["fill_idx"])] += 1.0
        net_delta[int(row["fill_idx"])] += sign
        if end_bar + 1 < n_bars:
            gross_delta[end_bar + 1] -= 1.0
            net_delta[end_bar + 1] -= sign
    gross = np.cumsum(gross_delta)
    net = np.cumsum(net_delta)
    return ExposureSeries(gross=gross, net=net, concurrent=gross.copy())


def _realized_equity(
    accepted: list[dict[str, Any]], policy: str
) -> tuple[np.ndarray, float, float]:
    """Event-based equity: R credited at each trade's credit bar.

    Returns ``(equity_curve, max_drawdown_r, final_equity_r)``; events
    sort by ``(credit_bar, admission order)`` so ties resolve
    deterministically, and the drawdown runs against a leading-zero
    peak (the donor convention).
    """
    events = [
        (int(row["credit_bar"]), seq, _policy_credit(row, policy))
        for seq, row in enumerate(accepted)
    ]
    events.sort(key=lambda event: (event[0], event[1]))
    if not events:
        return np.zeros(0, dtype=np.float64), 0.0, 0.0
    credited = np.asarray([event[2] for event in events], dtype=np.float64)
    equity = np.cumsum(credited)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    drawdown = float(np.max(peaks[1:] - equity))
    return equity, drawdown, float(equity[-1])


def _replay_signals(
    order: list[dict[str, Any]],
    holders: list[dict[str, Any] | None],
    slot_rows: list[list[dict[str, Any]]],
    *,
    signals_df: pd.DataFrame,
    panel: dict[str, dict[str, np.ndarray]],
    policy: str,
    cap: float | None,
    overlap_min: int,
    window_bars: int,
    hold_window: int,
    cost: CostModel,
    n_bars: int,
    cooldown_bars: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    """Walk the ordered records over the slot holders.

    Returns ``(accepted, rejected, displaced, correlation_rejects)``.
    The per-signal decision tree: a FREE slot (correlation filter
    first), else the admission policy over the OCCUPIED slots, else
    every slot is cooling down.  Ledger rows are appended to
    ``slot_rows`` here (attribution), the three ledger lists are
    returned in admission order.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    displaced: list[dict[str, Any]] = []
    correlation_rejects = 0
    for record in order:
        decision = record["decision_idx"]
        free_idx = [
            idx
            for idx, holder in enumerate(holders)
            if holder is None
            or _slot_status(holder, decision, cooldown_bars) == "free"
        ]
        occupied_holders = [
            holder
            for holder in holders
            if holder is not None
            and _slot_status(holder, decision, cooldown_bars) == "occupied"
        ]
        ledger_row = None
        rejected_row = None
        displaced_row = None
        if free_idx:
            ledger_row, rejected_row, rejects = _admit_free_slot(
                record,
                holders,
                free_idx,
                occupied_holders,
                signals_df=signals_df,
                panel=panel,
                corr_cap=cap,
                corr_min_overlap=overlap_min,
                corr_window=window_bars,
                hold_bars=hold_window,
                cost=cost,
                n_bars=n_bars,
            )
        elif occupied_holders:
            (
                ledger_row,
                rejected_row,
                displaced_row,
                rejects,
            ) = _admit_full_policy(
                record,
                holders,
                occupied_holders,
                signals_df=signals_df,
                panel=panel,
                policy=policy,
                corr_cap=cap,
                corr_min_overlap=overlap_min,
                corr_window=window_bars,
                hold_bars=hold_window,
                cost=cost,
                n_bars=n_bars,
            )
        else:
            rejected_row = dict(record["row"]) | {"reason": "cooldown"}
            rejects = 0
        correlation_rejects += rejects
        if ledger_row is not None:
            accepted.append(ledger_row)
            slot_rows[ledger_row["slot"]].append(ledger_row)
        if rejected_row is not None:
            rejected.append(rejected_row)
        if displaced_row is not None:
            displaced.append(displaced_row)
    return accepted, rejected, displaced, correlation_rejects


def run_portfolio_replay(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    slots: int = 2,
    cooldown: int = 0,
    policy: str = "fcfs",
    corr_cap: float | None = None,
    corr_min_overlap: int = 20,
    corr_window: int = 60,
    hold_bars: int = HOLD_BARS,
    cost: CostModel = DEFAULT_COST,
) -> PortfolioResult:
    """Replay candidate signals over N concurrent portfolio slots.

    Generalizes the single-slot engine machine: signals are processed
    in deterministic ``(decision_idx, priority desc, side, input
    position)`` order; a signal decided at bar ``d`` fills at
    ``d + 1`` (generator rule) and takes a slot when

    * a slot is FREE -- exited before ``d`` (the exit bar itself is
      still held; SL-first pessimism), past its per-slot ``cooldown``
      and not open-ended (a trade with no exit inside the data holds
      its slot to the end);
    * the correlation filter passes (when ``corr_cap`` is set): the
      candidate's ``r_proxy`` must not be |Spearman|-correlated above
      the cap with any OCCUPIED slot's proxy -- inconclusive pairs
      (too little overlap, degenerate series) admit.  Same-instrument
      stacking is the extreme case: a second signal on an instrument
      with an OPEN position carries |rho| = 1 against it and is
      rejected under any cap below 1;
    * otherwise the admission ``policy`` decides: ``"fcfs"`` rejects
      (reason ``"capacity"``); ``"replace_low"`` / ``"replace_high"``
      let a candidate with a STRICTLY higher score displace the worst
      open holder (reason ``"inferior_score"`` otherwise).  The
      displaced trade is closed at the decision bar and its row
      carries the RECONSTRUCTED partial-hold P&L (``force_close_r``,
      via :func:`atropos.backtesting.simulate_signal`) -- its equity
      credit is 0 under ``replace_low``, its full isolated R under
      ``replace_high``.

    Equity is EVENT-based: every accepted trade credits exactly once
    (``credit_r`` at ``credit_bar``; an unresolved trade credits 0 at
    the last bar), so the drawdown is not flattered by aggregation.

    Args:
        signals_df: candidate signals; required columns
            ``decision_idx`` (int >= 0, fill at ``decision_idx + 1``
            must exist in the panel), ``side`` ("long"/"short"),
            ``instrument`` (panel key), ``sl`` / ``tp`` (finite
            levels), ``score`` (finite, within-group rank-normalized
            upstream); optional ``priority`` (default 0.0),
            ``atr_entry`` (default 0.0), ``risk_ref`` and ``r_proxy``
            (required when ``corr_cap`` is set: the instrument's
            GROWING R-proxy series from one common anchor through
            ``decision_idx``, so the filter can extend open
            positions' snapshots bar-for-bar).
        prices: OHLC panel -- DataFrame with a 2-level
            ``(field, instrument)`` column MultiIndex, one finite bar
            grid shared by all instruments, at least 2 bars.
        slots: maximum concurrent positions, integer >= 1.
        cooldown: decision bars blocked after an exit, integer >= 0.
        policy: ``"fcfs"``, ``"replace_low"`` or ``"replace_high"``.
        corr_cap: correlation admission threshold in ``[0, 1]``, or
            None to disable the filter.
        corr_min_overlap: minimum overlapping proxy bars for a
            correlation verdict (below it: inconclusive -> admit).
        corr_window: recent bars each proxy is truncated to.
        hold_bars: maximum hold per trade (the isolated engine
            window); integer >= 1.
        cost: cost model for every simulation; defaults to
            :data:`atropos.backtesting.DEFAULT_COST`.

    Returns:
        :class:`PortfolioResult`; ``gates`` is empty and
        ``portfolio_verdict`` is ``"not_evaluated"`` -- run
        :func:`run_portfolio` for the gated pipeline.

    Raises:
        ValueError: on any invalid constant (``slots``/``cooldown``/
            ``hold_bars``/``policy``/``corr_*``/``cost``), an invalid
            price panel or signal frame, a fill bar past the data, an
            instrument missing from the panel, or a non-finite
            ``r_proxy``.

    Example::

        >>> opens = np.full(8, 100.0)
        >>> highs = np.full(8, 100.5)
        >>> lows = np.full(8, 99.5)
        >>> highs[4] = 104.0
        >>> prices = pd.DataFrame(
        ...     np.column_stack([opens, highs, lows, opens]),
        ...     columns=pd.MultiIndex.from_product(
        ...         [["open", "high", "low", "close"], ["demo"]]
        ...     ),
        ... )
        >>> signals = pd.DataFrame({
        ...     "decision_idx": [0, 3, 4],
        ...     "side": ["long", "long", "long"],
        ...     "instrument": ["demo", "demo", "demo"],
        ...     "sl": [99.0, 99.0, 99.0],
        ...     "tp": [104.0, 106.0, 104.0],
        ...     "score": [0.4, 0.6, 0.9],
        ... })
        >>> replay = run_portfolio_replay(signals, prices, slots=2,
        ...                               hold_bars=4)
        >>> [row["decision_idx"] for row in replay["accepted"]]
        [0, 3]
        >>> [row["decision_idx"] for row in replay["rejected"]]
        [4]
        >>> [row["reason"] for row in replay["rejected"]]
        ['capacity']
        >>> replay["exposure"]["gross"].tolist()
        [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
        >>> replay["portfolio_verdict"]
        'not_evaluated'
    """
    slots_total = validate_int("slots", slots, minimum=1)
    cooldown_bars = validate_int("cooldown", cooldown, minimum=0)
    hold_window = validate_int("hold_bars", hold_bars, minimum=1)
    if policy not in _PORTFOLIO_POLICIES:
        raise ValueError(
            f"policy must be one of {list(_PORTFOLIO_POLICIES)}, got"
            f" {policy!r}"
        )
    if not isinstance(cost, CostModel):
        raise ValueError(
            f"cost must be a CostModel, got {type(cost).__name__}"
        )
    cap: float | None = None
    if corr_cap is not None:
        cap = validate_number("corr_cap", corr_cap)
        if not 0.0 <= cap <= 1.0:
            raise ValueError(f"corr_cap must lie in [0, 1], got {corr_cap}")
    overlap_min = validate_int(
        "corr_min_overlap", corr_min_overlap, minimum=2
    )
    window_bars = validate_int("corr_window", corr_window, minimum=1)
    panel, n_bars = _validated_price_panel(prices)
    records = _validated_portfolio_signals(
        signals_df, panel, n_bars, corr_enabled=cap is not None
    )
    order = sorted(
        records,
        key=lambda rec: (
            rec["decision_idx"],
            -rec["priority"],
            rec["side"],
            rec["position"],
        ),
    )
    holders: list[dict[str, Any] | None] = [None] * slots_total
    slot_rows: list[list[dict[str, Any]]] = [[] for _ in range(slots_total)]
    accepted, rejected, displaced, correlation_rejects = _replay_signals(
        order,
        holders,
        slot_rows,
        signals_df=signals_df,
        panel=panel,
        policy=policy,
        cap=cap,
        overlap_min=overlap_min,
        window_bars=window_bars,
        hold_window=hold_window,
        cost=cost,
        n_bars=n_bars,
        cooldown_bars=cooldown_bars,
    )
    for row in accepted:
        row["credit_r"] = _policy_credit(row, policy)
    equity, drawdown, final_equity = _realized_equity(accepted, policy)
    exposure = _exposure_series(accepted, n_bars)
    per_slot: list[SlotAttribution] = [
        SlotAttribution(
            slot=slot,
            trades=len(slot_rows[slot]),
            ev_r=math.fsum(float(row["credit_r"]) for row in slot_rows[slot]),
        )
        for slot in range(slots_total)
    ]
    unresolved = sum(1 for row in accepted if int(row["exit_idx"]) == -1)
    gross = exposure["gross"]
    summary = PortfolioSummary(
        slots=slots_total,
        policy=policy,
        n_accepted=len(accepted),
        n_rejected=len(rejected),
        n_replaced=len(displaced),
        n_correlation_rejects=correlation_rejects,
        n_unresolved=unresolved,
        ev=final_equity,
        dd=drawdown,
        concurrent_max=int(gross.max()) if gross.size else 0,
    )
    return PortfolioResult(
        slots=slots_total,
        policy=policy,
        accepted=accepted,
        rejected=rejected,
        replaced=[dict(row) | {"reason": "replaced"} for row in displaced],
        per_slot=per_slot,
        equity_curve=equity,
        max_drawdown_r=drawdown,
        final_equity_r=final_equity,
        exposure=exposure,
        summary=summary,
        gates=[],
        portfolio_verdict="not_evaluated",
    )


# ---------------------------------------------------------------------------
# portfolio-level gates
# ---------------------------------------------------------------------------
def _validated_1d_finite(values: Any, name: str) -> np.ndarray:
    """Coerce an input to a finite 1-D float array (empty allowed).

    Raises:
        ValueError: if the input is not 1-D or carries NaN/inf.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got ndim={arr.ndim} of shape {arr.shape}"
        )
    if arr.size and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite, got NaN/inf entries")
    return arr


def _drawdown_magnitude(equity: np.ndarray) -> float:
    """Max drawdown (positive magnitude) of an event equity curve.

    The peak runs against a leading zero, so a curve that never leaves
    the red still reports an honest positive drawdown.
    """
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return float(np.max(peaks[1:] - equity))


def _deflated_sharpe_gate(
    ev_stream: np.ndarray, *, n_trials: int, dsr_min: float
) -> GateVerdict:
    """Deflated Sharpe gate over the per-period portfolio EV stream.

    Delegates the formula to
    :func:`atropos.evaluation_metrics.deflated_sharpe_from_moments`
    (its single home): the per-period SR, sample skew / Pearson
    kurtosis (NaN below ``_MIN_SKEW_OBS`` observations -- the
    ``deflated_sharpe`` convention) and a unit trial variance
    (unit-variance sibling trials in per-period SR units) go in.  A
    non-finite DSR triggers FAIL-CLOSED: an unverifiable stream can
    not count as verified.
    """
    n_obs = int(ev_stream.size)
    if n_obs >= _MIN_SKEW_OBS:
        spread = float(np.std(ev_stream, ddof=1))
        sr = float(np.mean(ev_stream) / spread) if spread > 0.0 else math.nan
        skew_val = float(skew(ev_stream))
        kurt_val = float(kurtosis(ev_stream, fisher=False))
    else:
        sr = math.nan
        skew_val = math.nan
        kurt_val = math.nan
    dsr = deflated_sharpe_from_moments(
        sr, n_obs, skew_val, kurt_val, 1.0, n_trials=n_trials
    )
    if not math.isfinite(dsr):
        triggered = True
        reason = (
            "dsr non-finite: fail-closed (too few observations or"
            " degenerate stream)"
        )
    elif dsr < dsr_min:
        triggered = True
        reason = f"dsr {dsr:.3f} below floor {dsr_min:.3f}"
    else:
        triggered = False
        reason = "ok"
    return GateVerdict(
        gate="deflated_sharpe",
        triggered=triggered,
        value=float(dsr),
        floor=dsr_min,
        reason=reason,
    )


def _drawdown_gate(equity: np.ndarray, dd_floor: float) -> GateVerdict:
    """Drawdown-floor gate over the event equity curve.

    ``value`` is the max drawdown as a NEGATIVE number in the equity
    curve's units (R by default); REACHING the floor breaches it
    (``value <= floor``), mirroring the preregistration barrier
    semantics.
    """
    value = -_drawdown_magnitude(equity)
    triggered = bool(value <= dd_floor)
    if triggered:
        reason = f"drawdown {value:.3f} reached the floor {dd_floor:.3f}"
    else:
        reason = "ok"
    return GateVerdict(
        gate="drawdown_floor",
        triggered=triggered,
        value=float(value),
        floor=dd_floor,
        reason=reason,
    )


def _concentration_gate(
    concurrent: np.ndarray | None,
    net_exposure: np.ndarray | None,
    share_cap: float,
) -> GateVerdict:
    """Same-side concentration gate over the exposure series.

    Per bar with positions held, the same-side share is
    ``0.5 + |net| / (2 * concurrent)`` (the larger of the long /
    short fractions); ``value`` is the maximum over bars.  Without
    exposure data the gate is skipped honestly: value NaN, triggered
    False, the reason says so.
    """
    if concurrent is None or net_exposure is None:
        return GateVerdict(
            gate="side_concentration",
            triggered=False,
            value=float("nan"),
            floor=share_cap,
            reason="no exposure series: gate skipped",
        )
    if concurrent.size != net_exposure.size:
        raise ValueError(
            "concurrent and net_exposure must have equal length, got"
            f" {concurrent.size} and {net_exposure.size}"
        )
    held = concurrent > 0
    if not bool(np.any(held)):
        return GateVerdict(
            gate="side_concentration",
            triggered=False,
            value=0.0,
            floor=share_cap,
            reason="ok (no positions held)",
        )
    shares = 0.5 + np.abs(net_exposure[held]) / (2.0 * concurrent[held])
    value = float(np.max(shares))
    triggered = bool(value > share_cap)
    if triggered:
        reason = f"same-side share {value:.3f} above cap {share_cap:.3f}"
    else:
        reason = "ok"
    return GateVerdict(
        gate="side_concentration",
        triggered=triggered,
        value=value,
        floor=share_cap,
        reason=reason,
    )


def portfolio_gates(
    equity_r: Any,
    daily_ev: Any,
    *,
    dsr_min: float = 0.95,
    dd_floor: float = -0.20,
    max_same_side_share: float = 0.80,
    n_trials: int = 5,
    min_trades: int = 30,
    concurrent: Any | None = None,
    net_exposure: Any | None = None,
) -> GateReport:
    """Portfolio-level kill gates over the realized portfolio streams.

    Three gates, fixed order, each a :class:`GateVerdict`:

    * ``deflated_sharpe`` -- the per-period portfolio EV stream
      (``daily_ev``: R credited per aggregation period, "daily" in
      the caller's grid) run through
      :func:`atropos.evaluation_metrics.deflated_sharpe_from_moments`
      (delegation, never a copy of the arithmetic); triggered below
      ``dsr_min`` or when the DSR is non-finite (fail-closed).
    * ``drawdown_floor`` -- the event-basis max drawdown of
      ``equity_r`` (the cumulative event equity) against ``dd_floor``
      in the curve's units; REACHING the floor breaches.
    * ``side_concentration`` -- the max same-side (long/short) share
      of concurrent positions against ``max_same_side_share``; needs
      the ``concurrent`` / ``net_exposure`` series (skipped honestly
      when either is missing).

    Verdict: ``"UNDERPOWERED"`` when fewer than ``min_trades`` trades
    are credited (the equity curve carries one point per credited
    trade), else ``"DEAD(<gate>)"`` naming the first triggered gate,
    else ``"SURVIVES"``.

    Args:
        equity_r: 1-D finite event equity curve (cumulative credited
            R; its length IS the credited-trade count).
        daily_ev: 1-D finite per-period portfolio EV stream.
        dsr_min: minimum deflated Sharpe in ``[0, 1]``.
        dd_floor: drawdown floor, strictly negative, in equity units.
        max_same_side_share: same-side share cap in ``(0, 1]``.
        n_trials: declared number of trials for the deflation,
            integer >= 1.
        min_trades: minimum credited trades for a verdict other than
            ``"UNDERPOWERED"``, integer >= 0.
        concurrent: optional per-bar concurrent-position series (the
            replay's ``exposure["concurrent"]``).
        net_exposure: optional per-bar signed exposure series (the
            replay's ``exposure["net"]``).

    Returns:
        :class:`GateReport` with the three gate verdicts and the
        portfolio verdict.

    Raises:
        ValueError: on any invalid constant, a non-1-D or non-finite
            stream, or unequal ``concurrent`` / ``net_exposure``
            lengths.

    Example::

        >>> drift = np.repeat([0.10, 0.05], 5).astype(np.float64)
        >>> equity = np.cumsum(drift)
        >>> concurrent = np.full(10, 2.0)
        >>> net = np.tile([1.0, -1.0], 5)
        >>> report = portfolio_gates(
        ...     equity, drift, min_trades=10, n_trials=1,
        ...     concurrent=concurrent, net_exposure=net,
        ... )
        >>> report["portfolio_verdict"]
        'SURVIVES'
        >>> [gate["gate"] for gate in report["gates"]]
        ['deflated_sharpe', 'drawdown_floor', 'side_concentration']
        >>> noise = np.tile([1.0, -1.0], 5)
        >>> portfolio_gates(
        ...     np.cumsum(noise), noise, min_trades=10, n_trials=1,
        ... )["portfolio_verdict"]
        'DEAD(deflated_sharpe)'
    """
    dsr_floor = validate_number("dsr_min", dsr_min)
    if not 0.0 <= dsr_floor <= 1.0:
        raise ValueError(f"dsr_min must lie in [0, 1], got {dsr_min}")
    floor_dd = validate_number("dd_floor", dd_floor)
    if floor_dd >= 0.0:
        raise ValueError(f"dd_floor must be < 0, got {dd_floor}")
    share_cap = validate_number("max_same_side_share", max_same_side_share)
    if not 0.0 < share_cap <= 1.0:
        raise ValueError(
            f"max_same_side_share must lie in (0, 1], got"
            f" {max_same_side_share}"
        )
    trials = validate_int("n_trials", n_trials, minimum=1)
    trades_floor = validate_int("min_trades", min_trades, minimum=0)
    equity = _validated_1d_finite(equity_r, "equity_r")
    ev_stream = _validated_1d_finite(daily_ev, "daily_ev")
    conc_series: np.ndarray | None = None
    net_series: np.ndarray | None = None
    if concurrent is not None:
        conc_series = _validated_1d_finite(concurrent, "concurrent")
    if net_exposure is not None:
        net_series = _validated_1d_finite(net_exposure, "net_exposure")
    gates = [
        _deflated_sharpe_gate(ev_stream, n_trials=trials, dsr_min=dsr_floor),
        _drawdown_gate(equity, floor_dd),
        _concentration_gate(conc_series, net_series, share_cap),
    ]
    n_trades = int(equity.size)
    if n_trades < trades_floor:
        verdict = "UNDERPOWERED"
    else:
        first_kill = next((gate for gate in gates if gate["triggered"]), None)
        verdict = (
            f"DEAD({first_kill['gate']})"
            if first_kill is not None
            else "SURVIVES"
        )
    return GateReport(
        gates=gates,
        portfolio_verdict=verdict,
        n_trades=n_trades,
        min_trades=trades_floor,
    )


# ---------------------------------------------------------------------------
# portfolio run wrapper
# ---------------------------------------------------------------------------
def run_portfolio(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    slots: int = 2,
    policy: str = "fcfs",
    cooldown: int = 0,
    corr_cap: float | None = 0.7,
    corr_min_overlap: int = 20,
    corr_window: int = 60,
    hold_bars: int = HOLD_BARS,
    cost: CostModel = DEFAULT_COST,
    dsr_min: float = 0.95,
    dd_floor: float = -0.20,
    max_same_side_share: float = 0.80,
    n_trials: int = 5,
    min_trades: int = 30,
    ev_period_bars: int = 1,
) -> PortfolioResult:
    """Full portfolio pipeline: replay -> gates -> verdict.

    Runs :func:`run_portfolio_replay` with the admission configuration
    (correlation filter INCLUDED -- pass ``corr_cap=None`` to disable),
    buckets the event credits into ``ev_period_bars`` bar buckets (the
    per-period "daily" EV stream), evaluates
    :func:`portfolio_gates` on the equity curve, the EV stream and the
    replay's exposure series, and attaches ``gates`` /
    ``portfolio_verdict`` to the replay result.  This is the function
    a user calls.

    Args:
        signals_df: candidate signals (see
            :func:`run_portfolio_replay`; with the default
            ``corr_cap`` an ``r_proxy`` column is required).
        prices: OHLC panel (see :func:`run_portfolio_replay`).
        slots: maximum concurrent positions, integer >= 1.
        policy: ``"fcfs"``, ``"replace_low"`` or ``"replace_high"``.
        cooldown: decision bars blocked after an exit, integer >= 0.
        corr_cap: correlation admission threshold in ``[0, 1]``, or
            None to disable the filter (0.7 by default here).
        corr_min_overlap: minimum overlapping proxy bars for a
            correlation verdict.
        corr_window: recent bars each proxy is truncated to.
        hold_bars: maximum hold per trade; integer >= 1.
        cost: cost model for every simulation.
        dsr_min: minimum deflated Sharpe in ``[0, 1]``.
        dd_floor: drawdown floor, strictly negative, in equity units.
        max_same_side_share: same-side share cap in ``(0, 1]``.
        n_trials: declared number of trials for the deflation,
            integer >= 1.
        min_trades: minimum credited trades for a verdict other than
            ``"UNDERPOWERED"``, integer >= 0.
        ev_period_bars: bar width of the EV buckets fed to the DSR
            gate (1 = the per-bar grid), integer >= 1.

    Returns:
        :class:`PortfolioResult` with ``gates`` and
        ``portfolio_verdict`` populated.

    Raises:
        ValueError: on any invalid input (the replay's and the gates'
            contracts apply).

    Example::

        >>> opens = np.full(8, 100.0)
        >>> highs = np.full(8, 100.5)
        >>> lows = np.full(8, 99.5)
        >>> highs[4] = 104.0
        >>> prices = pd.DataFrame(
        ...     np.column_stack([opens, highs, lows, opens]),
        ...     columns=pd.MultiIndex.from_product(
        ...         [["open", "high", "low", "close"], ["demo"]]
        ...     ),
        ... )
        >>> signals = pd.DataFrame({
        ...     "decision_idx": [0, 3],
        ...     "side": ["long", "long"],
        ...     "instrument": ["demo", "demo"],
        ...     "sl": [99.0, 99.0],
        ...     "tp": [104.0, 106.0],
        ...     "score": [0.4, 0.6],
        ... })
        >>> report = run_portfolio(
        ...     signals, prices, slots=2, hold_bars=4, corr_cap=None
        ... )
        >>> report["portfolio_verdict"]
        'UNDERPOWERED'
        >>> report["summary"]["n_accepted"]
        2
        >>> len(report["gates"])
        3
    """
    period = validate_int("ev_period_bars", ev_period_bars, minimum=1)
    result = run_portfolio_replay(
        signals_df,
        prices,
        slots=slots,
        cooldown=cooldown,
        policy=policy,
        corr_cap=corr_cap,
        corr_min_overlap=corr_min_overlap,
        corr_window=corr_window,
        hold_bars=hold_bars,
        cost=cost,
    )
    n_bars = int(result["exposure"]["gross"].size)
    n_buckets = max(1, -(-n_bars // period))
    daily_ev = np.zeros(n_buckets, dtype=np.float64)
    for row in result["accepted"]:
        bucket = int(row["credit_bar"]) // period
        daily_ev[bucket] += float(row["credit_r"])
    report = portfolio_gates(
        result["equity_curve"],
        daily_ev,
        dsr_min=dsr_min,
        dd_floor=dd_floor,
        max_same_side_share=max_same_side_share,
        n_trials=n_trials,
        min_trades=min_trades,
        concurrent=result["exposure"]["concurrent"],
        net_exposure=result["exposure"]["net"],
    )
    result["gates"] = report["gates"]
    result["portfolio_verdict"] = report["portfolio_verdict"]
    return result


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _demo_panel() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Seeded 3-instrument OHLC world (500 bars): cluster + maverick.

    Instruments ``"AAA"`` and ``"BBB"`` share one random walk (BBB is
    a near-clone of AAA -- correlation-cap material), ``"CCC"`` runs
    its own independent walk.  Returns the panel frame and the close
    series per instrument (proxy building).
    """
    rng = np.random.default_rng(11)
    n_bars = 500
    factor = rng.normal(0.0, 0.45, n_bars)
    idio_a = rng.normal(0.0, 0.25, n_bars)
    idio_b = rng.normal(0.0, 0.05, n_bars)
    idio_c = rng.normal(0.0, 0.45, n_bars)
    closes = {
        "AAA": 100.0 + np.cumsum(factor + idio_a),
        "BBB": 100.0 + np.cumsum(0.97 * (factor + idio_a) + idio_b),
        "CCC": 100.0 + np.cumsum(idio_c),
    }
    columns = pd.MultiIndex.from_product(
        [list(_PRICE_FIELDS), ["AAA", "BBB", "CCC"]]
    )
    data_columns: list[np.ndarray] = []
    for field in _PRICE_FIELDS:
        for instrument in ("AAA", "BBB", "CCC"):
            close_series = closes[instrument]
            if field == "close":
                data_columns.append(close_series)
                continue
            opens = np.empty(n_bars)
            opens[0] = 100.0
            opens[1:] = close_series[:-1]
            if field == "open":
                data_columns.append(opens)
                continue
            wick = np.abs(rng.normal(0.0, 0.2, n_bars))
            if field == "high":
                data_columns.append(np.maximum(opens, close_series) + wick)
            else:
                data_columns.append(np.minimum(opens, close_series) - wick)
    panel = pd.DataFrame(np.column_stack(data_columns), columns=columns)
    return panel, closes


def _demo_signals(closes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Schedule with an overlap window, a clone signal and a big score.

    Two AAA longs (bars 240 / 245) hold both slots from bar ~246; the
    BBB clone signal (bar 246, one bar behind the freshest AAA
    snapshot so the tail-aligned windows nearly coincide) is the
    correlation-cap bait, the CCC signal on the same bar is its
    independent control; the CCC candidate with score 0.95 (bar 250)
    is the replacement bait; a short CCC (bar 320) gives the exposure
    series both sides.
    """
    schedule = [
        (240, "AAA", "long", 0.50),
        (245, "AAA", "long", 0.60),
        (246, "BBB", "long", 0.80),
        (246, "CCC", "long", 0.70),
        (250, "CCC", "long", 0.95),
        (300, "CCC", "long", 0.55),
        (320, "CCC", "short", 0.65),
        (400, "AAA", "long", 0.75),
    ]
    rows: list[dict[str, Any]] = []
    for decision, instrument, side, score in schedule:
        fill = float(closes[instrument][decision + 1])
        if side == "long":
            stop = fill * 0.97
            target = fill * 1.06
        else:
            stop = fill * 1.03
            target = fill * 0.94
        history = closes[instrument][: decision + 1]
        # growing common-anchor series (returns since bar 0): the
        # replay extends position snapshots to the candidate's bar
        # growing common-anchor series (returns since bar 0): the
        # replay extends position snapshots to the candidate's bar
        proxy = np.diff(history) / history[:-1]
        rows.append(
            {
                "decision_idx": decision,
                "side": side,
                "instrument": instrument,
                "sl": stop,
                "tp": target,
                "score": score,
                "atr_entry": 1.0,
                "r_proxy": proxy,
            }
        )
    return pd.DataFrame(rows)


def _fingerprint(result: PortfolioResult) -> tuple[Any, ...]:
    """Array-free comparable fingerprint of a replay result."""
    return (
        result["equity_curve"].tolist(),
        result["exposure"]["gross"].tolist(),
        result["exposure"]["net"].tolist(),
        [
            (row["decision_idx"], row["slot"], row["credit_r"])
            for row in result["accepted"]
        ],
        [row["reason"] for row in result["rejected"]],
        list(result["summary"].items()),
    )


def _demo_case_replay() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Multi-slot discipline: cap, cooldown of exposure, equity identity."""
    panel, closes = _demo_panel()
    signals = _demo_signals(closes)
    replay = run_portfolio_replay(
        signals, panel, slots=2, policy="fcfs", hold_bars=24
    )
    gross = replay["exposure"]["gross"]
    net = replay["exposure"]["net"]
    print(
        f"A1 fcfs replay: accepted={replay['summary']['n_accepted']}"
        f" rejected={replay['summary']['n_rejected']}"
        f" reasons={[row['reason'] for row in replay['rejected']]}"
    )
    # (a) the slot cap holds at every bar
    assert int(gross.max()) == 2
    assert bool(np.all(gross <= 2.0)) and bool(np.all(net <= 2.0))
    assert bool(np.all(net >= -2.0))
    # (b) equity is EXACTLY the cumsum of the event credits
    ordered = sorted(replay["accepted"], key=lambda row: row["credit_bar"])
    expected = np.cumsum(
        np.asarray([row["credit_r"] for row in ordered], dtype=np.float64)
    )
    assert replay["equity_curve"].tolist() == expected.tolist()
    # both slots carry trades; reasons are the fcfs vocabulary
    assert sum(entry["trades"] for entry in replay["per_slot"]) == len(
        replay["accepted"]
    )
    assert {row["reason"] for row in replay["rejected"]} <= {
        "capacity",
        "cooldown",
    }
    print(
        f"A2 exposure peak={int(gross.max())}"
        f" dd={replay['max_drawdown_r']:.3f}"
        f" ev={replay['final_equity_r']:+.3f}"
    )
    return panel, closes


def _demo_case_replace(
    panel: pd.DataFrame, closes: dict[str, np.ndarray]
) -> None:
    """REPLACE bounds: 0 credit vs full isolated R, reconstructed P&L."""
    signals = _demo_signals(closes)
    low = run_portfolio_replay(
        signals, panel, slots=2, policy="replace_low", hold_bars=24
    )
    high = run_portfolio_replay(
        signals, panel, slots=2, policy="replace_high", hold_bars=24
    )
    print(
        f"B1 replace_low ev={low['final_equity_r']:+.3f}"
        f" replaced={low['summary']['n_replaced']}"
        f" | replace_high ev={high['final_equity_r']:+.3f}"
    )
    assert low["summary"]["n_replaced"] >= 1
    displaced_scores = [row["score"] for row in low["replaced"]]
    assert 0.50 in displaced_scores  # the worst open holder was displaced
    for row in low["replaced"]:
        # the reconstruction contract: recomputed to the replacement
        # bar (>= the fill bar: a same-day entry may be displaced the
        # bar it fills), never stale
        assert math.isfinite(row["force_close_r"])
        assert row["force_exit_idx"] >= row["fill_idx"]
    # the checked EV formula: low credits the displaced trade 0,
    # high keeps its full isolated R
    replaced_r = sum(float(row["r_pess"]) for row in low["replaced"])
    assert math.isclose(
        high["final_equity_r"] - low["final_equity_r"],
        replaced_r,
        rel_tol=1e-12,
    )
    assert low["accepted"] != high["accepted"]  # credits differ
    print("B2 reconstruction + EV bounds verified")


def _demo_case_portfolio_gates(
    panel: pd.DataFrame, closes: dict[str, np.ndarray]
) -> None:
    """Correlation admission + gates on the full pipeline."""
    signals = _demo_signals(closes)
    report = run_portfolio(
        signals,
        panel,
        slots=2,
        policy="fcfs",
        corr_cap=0.7,
        hold_bars=24,
        min_trades=5,
    )
    reasons = [row["reason"] for row in report["rejected"]]
    print(
        f"C1 corr pipeline: reasons={reasons}"
        f" corr_rejects={report['summary']['n_correlation_rejects']}"
        f" verdict={report['portfolio_verdict']}"
    )
    # (c) the correlation policy really filtered candidates: the BBB
    # clone of an open AAA position, plus same-instrument stacking
    # (|rho| = 1 against an open CCC/AAA position)
    corr_rejected = [
        row
        for row in report["rejected"]
        if row["reason"] == "correlation_cap"
    ]
    assert len(corr_rejected) >= 1
    assert any(
        row["instrument"] == "BBB" and row["decision_idx"] == 246
        for row in corr_rejected
    )
    # the INDEPENDENT CCC control on the same bar was admitted: the
    # layer rejects correlation, not co-timing
    accepted_keys = {
        (str(row["instrument"]), int(row["decision_idx"]))
        for row in report["accepted"]
    }
    assert ("CCC", 246) in accepted_keys
    # ... while stacking a second CCC long on the open CCC position
    # is correlation-killed (|Spearman| = 1 against itself)
    assert any(row["instrument"] == "CCC" for row in corr_rejected)
    # (d) the gates returned the full structure
    assert len(report["gates"]) == 3
    for gate in report["gates"]:
        assert set(gate) == {"gate", "triggered", "value", "floor", "reason"}
    assert report["portfolio_verdict"].startswith(
        ("SURVIVES", "DEAD", "UNDERPOWERED")
    )
    # (e) determinism: same seeded world twice -> bit-for-bit
    panel_again, closes_again = _demo_panel()
    signals_again = _demo_signals(closes_again)
    report_again = run_portfolio(
        signals_again,
        panel_again,
        slots=2,
        policy="fcfs",
        corr_cap=0.7,
        hold_bars=24,
        min_trades=5,
    )
    assert _fingerprint(report) == _fingerprint(report_again)
    assert report["portfolio_verdict"] == report_again["portfolio_verdict"]
    print("C2 determinism: two seeded portfolio runs agree bit-for-bit")


def run_all_examples() -> None:
    """Demo the portfolio layer on a seeded 3-instrument world."""
    print("=" * 72)
    print("portfolio demo: 3 instruments x 500 bars, slots=2, cap=0.7")
    print("=" * 72)
    panel, closes = _demo_case_replay()
    _demo_case_replace(panel, closes)
    _demo_case_portfolio_gates(panel, closes)
    print("all portfolio demo asserts passed")


if __name__ == "__main__":
    run_all_examples()
