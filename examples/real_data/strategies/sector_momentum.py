"""10-instrument cross-sectional momentum portfolio dataset (sector SPDRs).

Dataset construction for the PORTFOLIO-LEVEL controls
(``run_portfolio_controls.py``): the positive control is the classic
cross-sectional (XS) 12-1 momentum book -- long the top-3 sector ETFs
by trailing momentum, short the bottom-3, re-selected at every
month-end -- replayed through the multi-slot portfolio engine
(:func:`atropos.portfolio.run_portfolio`); the negative controls are
random books built from the SAME bars, cadence, sizing and stop
geometry with the cross-sectional SELECTION randomized.

Strategy (XS momentum, the Moskowitz-Ooi-Pedersen 12-1 signal applied
cross-sectionally): at each month-end compute every instrument's
12-month total-return momentum SKIPPING the most recent month,
``mom_12_1 = M[m-1] / M[m-13] - 1`` on month-end adjusted closes;
z-score the ten values cross-sectionally; next month hold long the
top-3 and short the bottom-3.  The skip-1 construction puts at least
three weeks between the newest price in the formation window and the
entry -- the no-lookahead lag is built into the signal, not bolted on.

Engine-facing dataset:

- one row per ENTRY: ``decision_idx`` = the month-end's last trading
  bar (all signal inputs -- momentum, ATR, stop levels -- use data up
  to and including that close), fill at ``decision_idx + 1`` (the
  engine's generator rule: the next bar's open, strictly AFTER every
  input), 6 rows per monthly cohort (3 long / 3 short);
- ``sl`` / ``tp``: 2xATR(14, Wilder) around the DECISION close
  (levels known at decision time; the engine resolves fills and
  intrabar touches pessimistically, SL first);
- ``score``: the cohort's momentum z-scores pushed through
  :func:`atropos.ranking.rank_normalize_group` -- the engine's
  admission contract (within-group rank-normalized scores, never raw
  values from incomparable sources);
- ``priority``: conviction (|score - 0.5| * 2), so same-bar candidates
  are processed strongest-first;
- ``r_proxy``: the instrument's daily adjusted returns from one common
  anchor (the panel's first bar) through ``decision_idx`` -- the
  growing series the correlation admission filter consumes;
- cohorts whose 21-bar hold window would run past the last bar are
  DROPPED (``decision_idx <= n_bars - hold_bars - 1``), so every
  accepted trade resolves inside the data -- no open-ended credits.

Data provenance
---------------
``data/<TICKER>.csv`` are COMMITTED snapshots of the ten Select Sector
SPDR ETFs (XLF XLE XLK XLI XLV XLY XLP XLRE XLU XLB), daily bars in
the same layout as ``data/SPY.csv`` (``Date,Open,High,Low,Close,
AdjClose,Volume``; ``AdjClose`` split- AND dividend-adjusted), request
window 2005-01-01..2025-01-01, source Yahoo Finance chart API,
retrieved 2026-11-06, each vintage pinned by a
``data/<TICKER>.csv.sha256`` sidecar (see ``download_data.py``).
One documented data fact: the real-estate Select Sector fund is
``XLRE`` (there is no ``XRE`` ticker) and it INCEPTED 2015-10-08, so
its snapshot holds 2323 bars where the other nine hold 5033.  The
panel aligns all ten instruments on their SHARED finite bar grid
(inner join), which therefore starts 2015-10-08: the 10-name
cross-section only exists from the real-estate fund's inception.
Not for redistribution; source data belongs to the provider.

All functions here are pure and deterministic: same CSVs and seed in,
same frames out.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from atropos.backtesting import CostModel
from atropos.ranking import rank_normalize_group

SECTOR_TICKERS: tuple[str, ...] = (
    "XLF",
    "XLE",
    "XLK",
    "XLI",
    "XLV",
    "XLY",
    "XLP",
    "XLRE",
    "XLU",
    "XLB",
)
"""The ten Select Sector SPDR ETFs (real estate is XLRE; see module
docstring)."""

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
"""Committed snapshots directory (``data/<TICKER>.csv``)."""

PANEL_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
"""Level-0 keys of the price panel (the engine's OHLC contract)."""

MOMENTUM_MONTHS = 12
"""Formation window of the 12-1 signal, in month-end bars."""

SKIP_MONTHS = 1
"""Months skipped between the window's newest price and the signal."""

ATR_PERIOD = 14
"""Wilder ATR period for the stop geometry."""

SL_TP_ATR = 2.0
"""Stop-loss / take-profit distance in ATR multiples."""

HOLD_BARS = 21
"""Maximum hold per trade (~one calendar month; time exit at the cap)."""

COST_BPS = 2
"""Preregistered round-trip friction, in basis points of notional."""

TOP_S = 3
"""Long leg size per monthly cohort."""

BOTTOM_S = 3
"""Short leg size per monthly cohort."""

_REQUIRED_COLUMNS = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "AdjClose",
    "Volume",
)


def load_prices(
    data_dir: str | Path = DATA_DIR,
    tickers: Sequence[str] = SECTOR_TICKERS,
) -> pd.DataFrame:
    """Load the committed snapshots into an adjusted OHLC panel.

    Each snapshot's intraday levels are mapped into the adjusted-close
    price space with the per-bar factor ``AdjClose / Close`` (the
    ``spy_rsi`` convention) so momentum, entries and intrabar stop /
    target fills share one consistent series per instrument.  The ten
    instruments are aligned on their SHARED finite bar grid (inner
    join on dates), which starts at the latest inception (XLRE,
    2015-10-08) -- see the module docstring.

    Args:
        data_dir: directory holding ``<TICKER>.csv`` snapshots.
        tickers: instruments to load; all must have snapshots.

    Returns:
        DataFrame indexed by naive midnight timestamps, sorted, with a
        ``(field, instrument)`` column MultiIndex over
        :data:`PANEL_FIELDS`; no missing values.

    Raises:
        ValueError: if a snapshot lacks the expected columns or the
            aligned grid still carries gaps (NaNs).
    """
    directory = Path(data_dir)
    adjusted: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = directory / f"{ticker}.csv"
        prices = pd.read_csv(path, parse_dates=["Date"])
        missing = [
            name for name in _REQUIRED_COLUMNS if name not in prices.columns
        ]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        prices = prices.set_index("Date").sort_index()
        prices = prices.loc[~prices.index.duplicated(keep="first")]
        factor = prices["AdjClose"] / prices["Close"]
        adjusted[ticker] = pd.DataFrame(
            {
                "open": prices["Open"] * factor,
                "high": prices["High"] * factor,
                "low": prices["Low"] * factor,
                "close": prices["Close"] * factor,
            }
        )
    combined = pd.concat(adjusted, axis=1, join="inner").dropna()
    if combined.empty:
        raise ValueError("no shared bar grid across the given tickers")
    panel = combined.swaplevel(axis=1).reindex(
        columns=pd.MultiIndex.from_product(
            [PANEL_FIELDS, list(adjusted)], names=["field", "instrument"]
        )
    )
    if not bool(panel.notna().all().all()):
        raise ValueError("aligned panel carries NaNs; snapshots disagree")
    return panel.astype("float64")


def wilder_atr(panel: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Per-instrument Wilder ATR over the panel's adjusted OHLC.

    True range is the largest of high-low, |high - prev close| and
    |low - prev close| (per instrument); smoothing is the Wilder
    recursion (``alpha = 1 / period``).  The head is NaN until the
    window fills.
    """
    parts: dict[str, pd.Series] = {}
    for ticker in panel.columns.get_level_values(1):
        high = panel[("high", ticker)]
        low = panel[("low", ticker)]
        close = panel[("close", ticker)]
        prev_close = close.shift(1)
        true_range = pd.concat(
            (
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ),
            axis=1,
        ).max(axis=1)
        parts[ticker] = true_range.ewm(
            alpha=1.0 / period, adjust=False, min_periods=period
        ).mean()
    return pd.DataFrame(parts)


def cross_sectional_momentum_12_1(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-month cross-sectionally z-scored 12-1 momentum scores.

    ``mom_12_1[m] = M[m-1] / M[m-13] - 1`` on month-end adjusted
    closes (12-month total-return momentum skipping the newest month),
    then z-scored ACROSS the ten instruments within each month-end
    row (within-group normalization: the cohort's scores live on one
    comparable scale).  The first 13 month-end rows are NaN.

    Returns:
        DataFrame indexed by month-end timestamps, one column per
        instrument; rows keep NaN until the formation window is
        available for every instrument.
    """
    # pandas-stubs type a MultiIndex frame's level slice as Series; at
    # runtime it is the (bars x instruments) DataFrame of closes.
    closes = cast(pd.DataFrame, panel["close"])
    month_end_close = closes.resample("ME").last()
    mom = (
        month_end_close.shift(SKIP_MONTHS)
        / month_end_close.shift(SKIP_MONTHS + MOMENTUM_MONTHS)
        - 1.0
    )
    cross_mean = mom.mean(axis=1)
    cross_sd = mom.std(axis=1, ddof=1)
    return mom.sub(cross_mean, axis=0).div(
        cross_sd.where(cross_sd > 0.0), axis=0
    )


def _month_end_positions(index: pd.DatetimeIndex) -> dict[pd.Period, int]:
    """Last bar position of every calendar month in the panel."""
    return {
        period: position
        for position, period in enumerate(index.to_period("M"))
    }


def _admissible_decisions(
    panel: pd.DataFrame,
    hold_bars: int,
    *,
    warmup_months: int = SKIP_MONTHS + MOMENTUM_MONTHS,
) -> dict[pd.Period, int]:
    """Decision-bar position per month, matched to the momentum span.

    Two structural guards, both mirroring the momentum book:

    - hold window inside the data: a cohort decided at
      ``decision_idx`` fills at ``decision_idx + 1`` and time-exits at
      ``fill + hold_bars - 1``; requiring
      ``decision_idx <= n_bars - hold_bars - 1`` keeps every trade's
      hold window inside the panel, so no accepted trade stays
      open-ended (the engine would credit it 0 at the last bar);
    - formation warm-up: the first ``warmup_months`` month-ends are
      skipped (the momentum signal is NaN there), so the null books
      span EXACTLY the same months as the momentum book.
    """
    month_positions = _month_end_positions(pd.DatetimeIndex(panel.index))
    if not month_positions:
        return {}
    first_month = min(month_positions)
    last_ok = len(panel) - hold_bars - 1
    return {
        period: position
        for period, position in month_positions.items()
        if position <= last_ok
        and (period.year * 12 + period.month)
        - (first_month.year * 12 + first_month.month)
        >= warmup_months
    }


def _cohort_rows(
    panel: pd.DataFrame,
    decisions: dict[pd.Period, int],
    atr_frame: pd.DataFrame,
    *,
    selected: dict[pd.Period, list[tuple[str, str]]],
    convictions: dict[pd.Period, np.ndarray],
    include_r_proxy: bool,
) -> list[dict[str, Any]]:
    """Materialize signal rows for the pre-selected monthly cohorts.

    ``selected`` maps each month to its ``(instrument, side)`` picks;
    ``convictions`` carries the cohort's raw per-pick values (aligned
    with the pick order), rank-normalized into the engine's admission
    score.  Stop geometry: 2xATR(14) around the DECISION close, levels
    known at decision time; ``r_proxy`` is the instrument's daily
    adjusted returns from the common anchor through the decision bar.
    """
    daily_returns = cast(pd.DataFrame, panel["close"]).pct_change()
    rows: list[dict[str, Any]] = []
    for period, decision in decisions.items():
        picks = selected.get(period)
        raw = convictions.get(period)
        if picks is None or raw is None:
            continue
        raw_array = np.asarray(raw, dtype=np.float64)
        if not bool(np.isfinite(raw_array).all()):
            continue
        atr_row = atr_frame.iloc[decision]
        if not bool(np.isfinite(atr_row.to_numpy(dtype=np.float64)).all()):
            continue
        cohort: list[dict[str, Any]] = []
        for (instrument, side), conviction in zip(
            picks, raw_array, strict=True
        ):
            ref = float(panel[("close", instrument)].iloc[decision])
            atr = float(atr_row[instrument])
            offset = SL_TP_ATR * atr
            if side == "long":
                stop, target = ref - offset, ref + offset
            else:
                stop, target = ref + offset, ref - offset
            entry: dict[str, Any] = {
                "decision_idx": decision,
                "side": side,
                "instrument": instrument,
                "sl": stop,
                "tp": target,
                "atr_entry": atr,
                "mom_z": float(conviction),
            }
            if include_r_proxy:
                proxy = daily_returns[instrument].to_numpy(dtype=np.float64)[
                    1 : decision + 1
                ]
                entry["r_proxy"] = proxy
            cohort.append(entry)
        normalized = rank_normalize_group(
            np.asarray([item["mom_z"] for item in cohort])
        )
        for item, rank in zip(cohort, normalized, strict=True):
            item["score"] = float(rank)
            item["priority"] = float(abs(rank - 0.5) * 2.0)
            rows.append(item)
    return rows


_SIGNAL_COLUMNS = (
    "decision_idx",
    "side",
    "instrument",
    "sl",
    "tp",
    "score",
    "priority",
    "atr_entry",
    "mom_z",
)


def _signal_frame(
    rows: list[dict[str, Any]], include_r_proxy: bool
) -> pd.DataFrame:
    """Column-ordered signal frame (adds ``r_proxy`` when requested)."""
    if not rows:
        return pd.DataFrame(columns=[*_SIGNAL_COLUMNS])
    columns = [*_SIGNAL_COLUMNS, *(["r_proxy"] if include_r_proxy else [])]
    return pd.DataFrame(rows)[columns]


def build_portfolio_dataset(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    top_s: int = TOP_S,
    bottom_s: int = BOTTOM_S,
    cost_bps: int = COST_BPS,
    hold_bars: int = HOLD_BARS,
    include_r_proxy: bool = True,
) -> dict[str, Any]:
    """Assemble the momentum portfolio's engine-ready dataset.

    Long the ``top_s`` instruments by cross-sectional 12-1 momentum,
    short the ``bottom_s``, one monthly cohort per scored month-end;
    fills at the next bar's open (the engine's generator rule), hold
    at most ``hold_bars`` bars.  The admission ``score`` column is the
    cohort's momentum z-scores rank-normalized within the cohort (the
    engine's within-group contract); the raw z-score rides along in
    ``mom_z`` for transparency.

    Args:
        panel: adjusted OHLC panel from :func:`load_prices`.
        scores: momentum scores from
            :func:`cross_sectional_momentum_12_1` (index = month-end
            timestamps).
        top_s: long leg size.
        bottom_s: short leg size.
        cost_bps: round-trip friction baked into the returned
            :class:`~atropos.backtesting.CostModel` (flat RT cost, no
            extra slippage pessimism; the engine's SL-first fill logic
            is structural and stays).
        hold_bars: maximum hold per trade.
        include_r_proxy: attach the growing per-instrument return
            series the correlation admission filter consumes (skip it
            for corr-cap-less replays to save memory).

    Returns:
        ``{"signals", "prices", "cost_model", "cost_bps", "hold_bars",
        "n_cohorts"}`` where ``signals`` is the engine-ready frame and
        ``prices`` echoes the panel.
    """
    decisions = _admissible_decisions(panel, hold_bars)
    atr_frame = wilder_atr(panel)
    selected: dict[pd.Period, list[tuple[str, str]]] = {}
    convictions: dict[pd.Period, np.ndarray] = {}
    for stamp, row in scores.iterrows():
        # scores' index is a DatetimeIndex (same resample as the
        # panel's closes); iterrows types the label Hashable.
        period = cast(pd.Timestamp, stamp).to_period("M")
        if period not in decisions:
            continue
        valid = row.dropna()
        if len(valid) < top_s + bottom_s:
            continue
        ranked = valid.sort_values(ascending=False)
        picks = [(str(name), "long") for name in ranked.index[:top_s]]
        picks.extend(
            (str(name), "short") for name in ranked.index[-bottom_s:]
        )
        selected[period] = picks
        convictions[period] = ranked.reindex(
            [instrument for instrument, _ in picks]
        ).to_numpy(dtype=np.float64)
    rows = _cohort_rows(
        panel,
        decisions,
        atr_frame,
        selected=selected,
        convictions=convictions,
        include_r_proxy=include_r_proxy,
    )
    return {
        "signals": _signal_frame(rows, include_r_proxy),
        "prices": panel,
        "cost_model": _flat_cost_model(cost_bps),
        "cost_bps": cost_bps,
        "hold_bars": hold_bars,
        "n_cohorts": len(selected),
    }


def build_null_signals(
    panel: pd.DataFrame,
    seed: int,
    *,
    top_s: int = TOP_S,
    bottom_s: int = BOTTOM_S,
    cost_bps: int = COST_BPS,
    hold_bars: int = HOLD_BARS,
    include_r_proxy: bool = True,
) -> dict[str, Any]:
    """Random admission control: noise books with the SAME structure.

    Identical bars, cadence, cohort sizes, stop geometry (2xATR around
    the decision close) and hold cap as the momentum book -- but the
    cross-sectional SELECTION is random: each cohort picks
    ``top_s + bottom_s`` instruments uniformly without replacement,
    assigns the first ``top_s`` to the long leg and the last
    ``bottom_s`` to the short leg, and scores every pick ``U(0, 1)``
    (random admission).  Whatever the gates say about these books is
    what they say about pure noise wearing the momentum book's
    clothes.
    """
    rng = np.random.default_rng(seed)
    decisions = _admissible_decisions(panel, hold_bars)
    atr_frame = wilder_atr(panel)
    tickers = list(panel.columns.get_level_values(1))
    selected: dict[pd.Period, list[tuple[str, str]]] = {}
    draws: dict[pd.Period, np.ndarray] = {}
    for period in sorted(decisions):
        picked = rng.choice(
            np.asarray(tickers), size=top_s + bottom_s, replace=False
        )
        picks = [(str(name), "long") for name in picked[:top_s]]
        picks.extend((str(name), "short") for name in picked[-bottom_s:])
        selected[period] = picks
        draws[period] = rng.uniform(size=top_s + bottom_s)
    rows = _cohort_rows(
        panel,
        decisions,
        atr_frame,
        selected=selected,
        convictions=draws,
        include_r_proxy=include_r_proxy,
    )
    for row in rows:
        row["mom_z"] = float("nan")
    return {
        "signals": _signal_frame(rows, include_r_proxy),
        "prices": panel,
        "cost_model": _flat_cost_model(cost_bps),
        "cost_bps": cost_bps,
        "hold_bars": hold_bars,
        "n_cohorts": len(selected),
    }


def _flat_cost_model(cost_bps: int) -> CostModel:
    """Flat round-trip cost model at ``cost_bps`` bps of notional.

    ``comm`` is charged per side (``2 * comm`` = the round trip);
    slippage and stop-gap pessimism are zeroed so the stated friction
    is the exact friction.  The engine's SL-first intrabar resolution
    (a structural pessimism, not a cost) is unaffected.
    """
    per_side = cost_bps * 0.5 * 1e-4
    return CostModel(gen_slip=0.0, comm=per_side, gap=0.0)
