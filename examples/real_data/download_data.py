"""Refresh/verify the committed data snapshots from public sources.

The COMMITTED snapshots in ``data/`` (each hash-pinned by a
``data/<TICKER>.csv.sha256`` sidecar) are the reproducibility anchors
of the real-data examples: the runners read them offline and their
verdicts are only reproducible against exactly those bytes.  This
utility documents and automates the refresh path PER TICKER:

1. try Stooq (``https://stooq.com/q/d/l/?s=<sym>.us&d1=20050101
   &d2=20250101&i=d``) -- the endpoint currently serves a JavaScript
   proof-of-work challenge to plain HTTP clients and, after clearing
   it, may still answer ``Access denied`` for the historical CSV;
2. fall back to the Yahoo Finance chart API (same window, adjusted
   close included) and normalize it to the snapshots' column layout;
3. compare the fetched bytes against the committed hash.

The comparison is intentionally strict: adjusted closes are
vintage-dependent (every new dividend re-bases the whole
back-adjusted history), and the provider recomputes them per request
-- re-fetches differ from the committed vintage in the last digits of
``AdjClose`` even within one day, while raw OHLCV is stable.  On a
mismatch for an ALREADY-PINNED ticker this utility records a failure
and writes NOTHING for it -- keep using the committed snapshot, or
consciously re-pin a new vintage by replacing both
``data/<TICKER>.csv`` and its sidecar in a commit of their own.  A
ticker with NO sidecar yet is pinned fresh: the snapshot and its
sidecar are written together in one step.

Failures are collected across tickers and raised together at the end,
so one drifted vintage (e.g. SPY after a new dividend) never blocks
pinning the remaining tickers.

Coverage: ``SPY`` (the single-asset examples) plus the ten Select
Sector SPDR ETFs of the portfolio controls.  The real-estate sleeve
is ``XLRE`` (incepted 2015-10-08 -- there is no ``XRE`` ticker; its
shorter history is a documented fact of the panel, see
``strategies/sector_momentum.py``).

Run:

    uv run python examples/real_data/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

SPY = "SPY"
SECTOR_ETFS: tuple[str, ...] = (
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
ALL_TICKERS: tuple[str, ...] = (SPY, *SECTOR_ETFS)

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&d1=20050101&d2=20250101&i=d"
YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1=1104537600&period2=1735689600&interval=1d"
    "&events=div%2Csplits&includeAdjustedClose=true"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
COLUMNS = ("Date", "Open", "High", "Low", "Close", "AdjClose", "Volume")


def snapshot_path(ticker: str) -> Path:
    """Committed snapshot path for ``ticker``."""
    return DATA_DIR / f"{ticker}.csv"


def sidecar_path(ticker: str) -> Path:
    """Committed sha256 sidecar path for ``ticker``."""
    return DATA_DIR / f"{ticker}.csv.sha256"


def _fetch(url: str) -> str:
    """GET ``url`` with a browser UA; return the body as text."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: str = response.read().decode("utf-8", errors="replace")
        return body


def _fetch_stooq(ticker: str) -> str | None:
    """Stooq daily CSV, or None when the endpoint refuses plain clients."""
    body = _fetch(STOOQ_URL.format(symbol=f"{ticker.lower()}.us"))
    if body.startswith("Date,"):
        return body
    print(
        f"stooq {ticker}: endpoint served a challenge/access denial "
        "instead of CSV; falling back to Yahoo Finance",
        file=sys.stderr,
    )
    return None


def _fetch_yahoo(ticker: str) -> str:
    """Yahoo chart API -> normalized CSV in the snapshots' layout."""
    payload = json.loads(_fetch(YAHOO_URL.format(ticker=ticker)))
    error = payload["chart"].get("error")
    if error is not None:
        raise RuntimeError(
            f"yahoo {ticker}: chart API error: {error.get('description')}"
        )
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    lines = [",".join(COLUMNS)]
    for i, stamp in enumerate(result["timestamp"]):
        row = [
            quote["open"][i],
            quote["high"][i],
            quote["low"][i],
            quote["close"][i],
            adjclose[i],
            quote["volume"][i],
        ]
        if any(value is None for value in row):
            continue
        day = datetime.fromtimestamp(stamp, tz=UTC).date()
        cells = [
            str(day),
            *(str(round(value, 4)) for value in row[:4]),
            str(round(row[4], 6)),
            str(int(row[5])),
        ]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def fetch_body(ticker: str) -> str:
    """Fetch one ticker's daily history: Stooq first, Yahoo fallback."""
    body: str | None = None
    try:
        body = _fetch_stooq(ticker)
    except (OSError, urllib.error.URLError) as error:
        print(
            f"stooq {ticker} unreachable ({error}); trying Yahoo",
            file=sys.stderr,
        )
    if body is None:
        body = _fetch_yahoo(ticker)
    return body


def committed_hash(ticker: str) -> str:
    """Hash pinned by the ticker's sidecar (first whitespace token)."""
    return sidecar_path(ticker).read_text(encoding="ascii").split()[0]


def _write_snapshot(ticker: str, body: str, digest: str) -> None:
    """Commit one snapshot vintage: bytes and sidecar together."""
    snapshot_path(ticker).write_text(body, encoding="utf-8", newline="\n")
    sidecar_path(ticker).write_text(
        f"{digest}  {ticker}.csv\n", encoding="ascii", newline="\n"
    )


def process_ticker(ticker: str) -> str:
    """Fetch, normalize, compare against the pin; pin or verify.

    Returns:
        One human-readable status line.

    Raises:
        RuntimeError: if the fetched data does not match an existing
            pin (nothing is written in that case).
    """
    body = fetch_body(ticker)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if not sidecar_path(ticker).is_file():
        _write_snapshot(ticker, body, digest)
        return f"{ticker}: pinned fresh vintage ({digest})"
    pinned = committed_hash(ticker)
    if digest != pinned:
        raise RuntimeError(
            f"{ticker}: fetched data hash does not match the committed "
            f"snapshot ({digest} != {pinned}).  Adjusted closes are "
            "vintage-dependent, so upstream drift is expected: keep "
            f"using the committed data/{ticker}.csv for reproducible "
            "runs, or consciously re-pin a new vintage by committing "
            f"data/{ticker}.csv and data/{ticker}.csv.sha256 together."
        )
    snapshot_path(ticker).write_text(body, encoding="utf-8", newline="\n")
    return f"{ticker}: verified and rewritten in place ({digest})"


def main() -> None:
    """Pin/verify every ticker; raise together at the end on failures."""
    failures: list[tuple[str, RuntimeError]] = []
    for ticker in ALL_TICKERS:
        try:
            print(process_ticker(ticker))
        except RuntimeError as error:
            failures.append((ticker, error))
            print(f"FAILED: {error}", file=sys.stderr)
    if failures:
        detail = "\n  - ".join(
            f"{ticker}: {error}" for ticker, error in failures
        )
        raise RuntimeError(
            "some snapshots did not verify against their committed "
            f"pins (nothing was written for them):\n  - {detail}"
        )


if __name__ == "__main__":
    main()
