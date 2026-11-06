"""SPY RSI(2) mean-reversion: dataset construction for the ML chain.

Strategy (Connors & Alvarez, "Short Term Trading Strategies That
Work", adapted to a single ETF): at each close compute the classic
2-period RSI, ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)`` with
Wilder smoothing; SHORT the NEXT bar's open when ``RSI(2) > 70``
(mean-reversion fade of strength) and go LONG the next bar's open
when ``RSI(2) < 30`` (fade of weakness).  Exit whichever side on the
FIRST of: RSI(2) crossing back through 50 (evaluated at a close),
a 2xATR(14) stop-loss or 2xATR(14) take-profit touch (intrabar), or
5 trading bars held.  Each entry therefore holds at most 5 bars and
consecutive events never overlap.

This is the framework's NEGATIVE CONTROL on real data.  The gross
short-term reversal edge on SPY is real but thin: a few basis points
per trade against a ~1.7% per-trade sd, on a book turning over every
couple of days.  The preregistered expectation for this book is
``DEAD`` -- the engine should kill a strategy that looks tradable
gross but is too weak to survive walk-forward scrutiny.  The honest
anatomy of the recorded kill (expected_rsi_output.md): the taken
book actually clears the 2 bps cost line once thresholds select its
trades, so the deaths come from the harder gates -- unstable
per-fold feature rankings (gbm) and a deflated Sharpe below the
multiple-testing bar (rf / logit).  The companion example is
``spy_tsmom`` (same snapshot, a book with an order-of-magnitude
stronger cost-line edge).

The event dataset handed to ``run_ml_kill_protocol``:

- one row per ENTRY (next bar's open after a signal close; the
  decision uses only information up to the signal close, which
  strictly precedes the execution price -- the no-lookahead lag is
  built into the signal, not bolted on);
- features frozen at the SIGNAL close: ``rsi_2`` (the raw signal
  state), ``rsi_14`` (slower momentum context) and ``vol_20d``
  (trailing 20-day log-return sd).  ``battery_controls`` is empty:
  the RSI signal itself is passed to the models, so the incremental
  gate reports inconclusive and the book must survive the remaining
  gates on its own;
- ``exec_return`` = the signed return from entry open to the rule
  exit (RSI-50 cross / SL / TP / 5-bar cap), net of a 2 bps
  round-trip cost (caller-side netting -- the engine never applies
  costs itself).  Long and short events share one frame with returns
  signed by side, so one engine run evaluates the two-sided book;
- ``target`` = ``(exec_return > 0)``;
- ``times_us`` = entry dates as int64 microseconds (UTC midnight),
  strictly monotone.

Data provenance
---------------
``data/SPY.csv`` is the same COMMITTED snapshot ``spy_tsmom`` uses
(SPY daily bars, 2005-01-03 .. 2024-12-31, 5033 rows, ``AdjClose``
the split- AND dividend-adjusted close; source Yahoo Finance chart
API, one frozen adjustment vintage pinned by ``SPY.csv.sha256``).
Intraday levels (Open/High/Low) are mapped into adjusted space with
the per-bar factor ``AdjClose / Close`` so entries, stops and
targets are quoted in one consistent series.  See ``download_data.py``
and the README.  Not for redistribution; source data belongs to the
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

HORIZON_BARS = 5
"""Maximum hold in trading bars including the entry bar (the cap)."""

ENTRY_HI = 70.0
"""Short threshold: RSI(2) strictly above this fades strength."""

ENTRY_LO = 30.0
"""Long threshold: RSI(2) strictly below this fades weakness."""

EXIT_LEVEL = 50.0
"""RSI(2) crossing back through this level (at a close) exits."""

SL_TP_ATR = 2.0
"""Stop-loss / take-profit distance in ATR(14) multiples."""

FEATURES: list[str] = ["rsi_2", "rsi_14", "vol_20d"]
"""Conditioning features the classifiers are judged on."""

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


def wilder_rsi(adj_close: pd.Series, period: int) -> pd.Series:
    """Classic RSI with Wilder smoothing over ``period`` bars.

    ``avg_gain`` / ``avg_loss`` are Wilder-exponentially smoothed
    mean gains/losses (``alpha = 1 / period``) over close-to-close
    changes; ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)``.  A
    zero average loss pins RSI to 100; the head is NaN until the
    window fills.
    """
    ret = adj_close.diff()
    gain = ret.clip(lower=0.0)
    loss = (-ret).clip(lower=0.0)
    avg_gain = gain.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi.where(avg_loss > 0.0, 100.0).where(ret.notna(), np.nan)


def wilder_atr(prices: pd.DataFrame, period: int) -> pd.Series:
    """Average true range with Wilder smoothing over ``period`` bars.

    True range is the largest of high-low, |high - prev close| and
    |low - prev close|; smoothing is the same Wilder recursion as
    :func:`wilder_rsi`.  The head is NaN until the window fills.
    """
    prev_close = prices["Close"].shift(1)
    true_range = pd.concat(
        (
            prices["High"] - prices["Low"],
            (prices["High"] - prev_close).abs(),
            (prices["Low"] - prev_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily conditioning features, causal (trailing windows only).

    Returns a frame aligned with ``prices`` holding ``rsi_2``,
    ``rsi_14`` and ``vol_20d`` (see module docstring); the head is
    NaN until each window fills.
    """
    adj = prices["AdjClose"]
    log_ret = np.log(adj).diff()
    return pd.DataFrame(
        {
            "rsi_2": wilder_rsi(adj, 2),
            "rsi_14": wilder_rsi(adj, 14),
            "vol_20d": log_ret.rolling(20).std(),
        },
        index=prices.index,
    )


def _adjusted_levels(prices: pd.DataFrame) -> pd.DataFrame:
    """Open/High/Low mapped into the adjusted-close price space.

    Each bar's levels are scaled by the per-bar adjustment factor
    ``AdjClose / Close`` so entries and intrabar stop/target fills
    share one consistent series with the adjusted closes.
    """
    factor = prices["AdjClose"] / prices["Close"]
    return pd.DataFrame(
        {
            "open": prices["Open"] * factor,
            "high": prices["High"] * factor,
            "low": prices["Low"] * factor,
        },
        index=prices.index,
    )


def _first_exit(
    side: float,
    stop: float,
    target: float,
    i_entry: int,
    *,
    levels: pd.DataFrame,
    adj: pd.Series,
    rsi_2: pd.Series,
) -> tuple[int, float, str]:
    """Simulate one position's exit scan; return (bar, price, reason).

    Bars are scanned from the entry bar to the :data:`HORIZON_BARS`
    cap.  An opening gap through the stop fills at that open; an
    intrabar touch fills at the stop / target level, stops first
    (worst case); an RSI(2) cross of 50 at a close fills at that
    close; the cap fills at its close.
    """
    last_bar = i_entry + HORIZON_BARS - 1
    for j in range(i_entry, last_bar + 1):
        open_px = float(levels["open"].iloc[j])
        high_px = float(levels["high"].iloc[j])
        low_px = float(levels["low"].iloc[j])
        # a stop gapped through at the open fills at that open
        if j > i_entry:
            gapped = (open_px <= stop) if side > 0 else (open_px >= stop)
            if gapped:
                return j, open_px, "sl"
        # intrabar touch: stops first (worst case), then targets
        if (low_px <= stop) if side > 0 else (high_px >= stop):
            return j, stop, "sl"
        if (high_px >= target) if side > 0 else (low_px <= target):
            return j, target, "tp"
        # RSI(2) crossed back through 50 at this bar's close
        rsi_now = rsi_2.iloc[j]
        if np.isfinite(rsi_now) and (
            (rsi_now >= EXIT_LEVEL) if side > 0 else (rsi_now <= EXIT_LEVEL)
        ):
            return j, float(adj.iloc[j]), "rsi50"
    return last_bar, float(adj.iloc[last_bar]), "hold"


def make_events(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per mean-reversion entry: features, exec return, stamps.

    Entry rule: the next bar AFTER a close with ``rsi_2 > 70`` (short)
    or ``rsi_2 < 30`` (long), executed at that bar's OPEN -- the
    signal close strictly precedes the execution price.  Exit is the
    first of: intrabar 2xATR(14) stop / target touch (a stop gapped
    through at the open fills at the open; stops take priority over
    targets), an RSI(2) cross of 50 at a close, or the 5-bar cap at
    that close (see :func:`_first_exit`).  The book is
    non-overlapping by construction: the scan resumes after each exit
    bar.  Events whose features are not yet warm (NaN windows) or
    whose exit would fall past the last bar are dropped -- no
    imputation, no lookahead.

    Returns:
        DataFrame with columns ``entry_date`` / ``exit_date`` /
        ``signal_date`` / ``side``, the feature columns, ``hold_bars``
        / ``exit_reason`` and ``exec_return`` (signed trade return net
        of :data:`COST_RT`), sorted by entry date.
    """
    adj = prices["AdjClose"]
    levels = _adjusted_levels(prices)
    atr14 = wilder_atr(prices, 14)
    rsi_2 = features["rsi_2"]
    rows: list[dict[str, Any]] = []
    n = len(adj)
    i = 0
    while i < n - 1 - HORIZON_BARS:
        signal = rsi_2.iloc[i]
        if not np.isfinite(signal):
            i += 1
            continue
        if signal > ENTRY_HI:
            side = -1.0
        elif signal < ENTRY_LO:
            side = 1.0
        else:
            i += 1
            continue
        values = features[FEATURES].iloc[i].to_numpy(dtype=np.float64)
        atr = atr14.iloc[i]
        if not np.isfinite(values).all() or not np.isfinite(atr):
            i += 1
            continue
        i_entry = i + 1
        entry_px = float(levels["open"].iloc[i_entry])
        stop = entry_px - side * SL_TP_ATR * float(atr)
        target = entry_px + side * SL_TP_ATR * float(atr)
        exit_i, exit_px, reason = _first_exit(
            side,
            stop,
            target,
            i_entry,
            levels=levels,
            adj=adj,
            rsi_2=rsi_2,
        )
        gross = side * (exit_px / entry_px - 1.0)
        rows.append(
            {
                "entry_date": adj.index[i_entry],
                "exit_date": adj.index[exit_i],
                "signal_date": adj.index[i],
                "side": "short" if side < 0 else "long",
                "rsi_2": float(values[0]),
                "rsi_14": float(values[1]),
                "vol_20d": float(values[2]),
                "hold_bars": exit_i - i_entry + 1,
                "exit_reason": reason,
                "exec_return": gross - COST_RT,
            }
        )
        i = exit_i + 1
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def build_dataset(csv_path: str | Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    """Assemble the engine-ready dataset from the snapshot.

    Returns:
        ``{"prices", "features", "events", "feature_frame", "target",
        "exec_returns", "times_us"}`` where ``feature_frame`` is the
        event-level feature DataFrame (exactly the model columns --
        ``battery_controls`` is empty for this book: the RSI signal is
        the feature), ``target`` / ``exec_returns`` / ``times_us`` are
        the aligned 1-D arrays ``run_ml_kill_protocol`` expects, and
        ``events`` keeps the human-readable event table.
    """
    prices = load_spy(csv_path)
    features = make_features(prices)
    events = make_events(prices, features)
    if events.empty:
        raise ValueError("no events built: check the snapshot coverage")
    feature_frame = events[FEATURES]
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
