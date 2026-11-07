"""SPY 12-1 time-series momentum: dataset construction for the ML chain.

Strategy (Moskowitz, Ooi & Pedersen 2012, adapted to a single ETF):
at each month-end compute the 12-month total-return momentum SKIPPING
the most recent month, ``mom_12_1 = P[m-1] / P[m-13] - 1`` on monthly
bars; go long the NEXT month when ``mom_12_1 > 0`` and stay flat
otherwise.  Each entry holds a fixed 21-trading-day horizon (one
month), so consecutive events OVERLAP slightly (~1-2 bars, measured
on the committed sample): the monthly entry cadence (~21 trading
days) does not clear the 21-bar hold before the next entry fires.
The event overlap is handled honestly downstream -- the protocol's
horizon-aware purge/embargo (not event independence) is what keeps
the evaluation leakage-free.  The skip-1-month
construction puts at least three weeks between the newest price in
the formation window and the entry -- the no-lookahead lag is built
into the signal, not bolted on.

The event dataset handed to ``run_ml_kill_protocol``:

- one row per ENTRY day (first trading day of the month, entry at the
  close that day; the decision to hold 21 days uses only information
  up to and including that close, which is also the execution price);
- features at the entry close: ``vol_20d`` (trailing 20-day log-return
  sd), ``ma_gap_50`` / ``ma_gap_200`` (close / trailing mean - 1) and
  the raw ``mom_12_1`` signal.  The runner passes ``mom_12_1`` as a
  battery control, so the classifiers must earn their keep on the
  conditioning features against a linear battery that holds the
  classic momentum signal;
- ``exec_return`` = the 21-day close-to-close forward return minus a
  2 bps round-trip cost (caller-side netting -- the engine never
  applies costs itself);
- ``target`` = ``(exec_return > 0)``;
- ``times_us`` = entry dates as int64 microseconds (UTC midnight),
  strictly monotone.

Data provenance
---------------
``data/SPY.csv`` is a COMMITTED snapshot of SPY (SPDR S&P 500 ETF)
daily bars, 2005-01-03 .. 2024-12-31, 5033 rows, column ``AdjClose``
being the split- AND dividend-adjusted close (the total-return series
the momentum literature uses; Stooq's CSV serves the same adjusted
series).  Source: Yahoo Finance chart API, retrieved 2026-11-05; the
adjusted close is vintage-dependent (each new dividend re-bases the
back-adjusted history), so the snapshot freezes ONE adjustment
vintage and ``SPY.csv.sha256`` pins it.  See ``download_data.py`` and
the README.  Not for redistribution; source data belongs to the
provider.

All functions here are pure and deterministic: same CSV in, same
frames out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COST_RT = 0.0002
"""Round-trip transaction cost netted out of every event's exec return."""

HORIZON_DAYS = 21
"""Entry-to-exit holding horizon in trading days (~one calendar month)."""

FEATURES: list[str] = ["vol_20d", "ma_gap_50", "ma_gap_200"]
"""Conditioning features the classifiers are judged on."""

BATTERY_CONTROL = "mom_12_1"
"""The raw signal: a battery-only column (excluded from the models)."""

_REQUIRED_COLUMNS = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "AdjClose",
    "Volume",
)

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "SPY.csv"


def load_spy(csv_path: str | Path = DEFAULT_SNAPSHOT) -> pd.DataFrame:
    """Load the committed SPY snapshot as a date-indexed OHLCV frame.

    Args:
        csv_path: path to the snapshot CSV (defaults to the committed
            ``data/SPY.csv`` next to this package).

    Returns:
        DataFrame indexed by naive midnight timestamps, sorted, with
        float price columns and int volume.

    Raises:
        ValueError: if the file lacks the expected columns.
    """
    path = Path(csv_path)
    prices = pd.read_csv(path, parse_dates=["Date"])
    missing = [
        name for name in _REQUIRED_COLUMNS if name not in prices.columns
    ]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    prices = prices.set_index("Date").sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    prices = prices.astype(
        dict.fromkeys(("Open", "High", "Low", "Close", "AdjClose"), "float64")
    )
    return prices.astype({"Volume": "int64"})


def momentum_12_1(prices: pd.DataFrame) -> pd.Series:
    """Monthly 12-1 momentum on month-end bars (skip the newest month).

    ``mom_12_1[k] = M[k-1] / M[k-13] - 1`` where ``M`` are month-end
    adjusted closes: the 12-month return ending ONE month before the
    signal date.  The first 13 monthly bars are NaN.
    """
    monthly = prices["AdjClose"].resample("ME").last()
    return monthly.shift(1) / monthly.shift(13) - 1.0


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily conditioning features, causal (trailing windows only).

    Returns a frame aligned with ``prices`` holding ``vol_20d``,
    ``ma_gap_50`` and ``ma_gap_200`` (see module docstring); the head
    is NaN until each window fills.
    """
    adj = prices["AdjClose"]
    log_ret = np.log(adj).diff()
    return pd.DataFrame(
        {
            "vol_20d": log_ret.rolling(20).std(),
            "ma_gap_50": adj / adj.rolling(50).mean() - 1.0,
            "ma_gap_200": adj / adj.rolling(200).mean() - 1.0,
        },
        index=prices.index,
    )


def make_events(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per long entry: features, exec return, timestamps.

    Entry rule: the first trading day of the month AFTER a month-end
    with ``mom_12_1 > 0`` (long/flat book, entry/exit flips of the
    signal).  Exit is ``HORIZON_DAYS`` trading days later at the
    close; the 21-day forward return is fully inside the snapshot for
    every kept event.  Events whose features are not yet warm (NaN
    windows) or whose exit falls past the last bar are dropped -- no
    imputation, no lookahead.

    Returns:
        DataFrame with columns ``entry_date`` / ``exit_date`` /
        ``signal_month_end``, the feature columns, ``mom_12_1`` and
        ``exec_return`` (forward return net of :data:`COST_RT`),
        sorted by entry date.
    """
    adj = prices["AdjClose"]
    mom = momentum_12_1(prices)
    first_trading_day: dict[pd.Period, pd.Timestamp] = {}
    for stamp in adj.index:
        first_trading_day.setdefault(stamp.to_period("M"), stamp)
    rows: list[dict[str, Any]] = []
    for k in range(13, len(mom)):
        signal = mom.iloc[k]
        if not np.isfinite(signal) or signal <= 0.0:
            continue
        entry_day = first_trading_day.get(mom.index[k].to_period("M") + 1)
        if entry_day is None:
            continue
        i_entry = adj.index.get_loc(entry_day)
        i_exit = i_entry + HORIZON_DAYS
        if i_exit >= adj.size:
            continue
        feat_row = features.loc[entry_day]
        values = feat_row.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        gross = float(adj.iloc[i_exit] / adj.iloc[i_entry] - 1.0)
        rows.append(
            {
                "entry_date": entry_day,
                "exit_date": adj.index[i_exit],
                "signal_month_end": mom.index[k],
                "mom_12_1": float(signal),
                "vol_20d": float(values[0]),
                "ma_gap_50": float(values[1]),
                "ma_gap_200": float(values[2]),
                "exec_return": gross - COST_RT,
            }
        )
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def build_dataset(csv_path: str | Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    """Assemble the engine-ready dataset from the snapshot.

    Returns:
        ``{"prices", "features", "events", "feature_frame", "target",
        "exec_returns", "times_us"}`` where ``feature_frame`` is the
        event-level feature DataFrame (models' columns plus the
        battery control, in that order), ``target`` / ``exec_returns``
        / ``times_us`` are the aligned 1-D arrays
        ``run_ml_kill_protocol`` expects, and ``events`` keeps the
        human-readable event table.
    """
    prices = load_spy(csv_path)
    features = make_features(prices)
    events = make_events(prices, features)
    if events.empty:
        raise ValueError("no events built: check the snapshot coverage")
    feature_frame = events[[*FEATURES, BATTERY_CONTROL]]
    stamps_ns = (
        events["entry_date"]
        .dt.tz_localize("UTC")
        .astype("datetime64[ns, UTC]")
        .astype("int64")
    )
    return {
        "prices": prices,
        "features": features,
        "events": events,
        "feature_frame": feature_frame,
        "target": (events["exec_return"] > 0).astype(np.int64).to_numpy(),
        "exec_returns": events["exec_return"].to_numpy(dtype=np.float64),
        "times_us": (stamps_ns.to_numpy() // 1000).astype(np.int64),
    }
