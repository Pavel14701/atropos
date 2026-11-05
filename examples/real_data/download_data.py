"""Refresh/verify the SPY snapshot from public sources (best effort).

The COMMITTED snapshot ``data/SPY.csv`` (hash-pinned by
``data/SPY.csv.sha256``) is the reproducibility anchor of this
example: the runner reads it offline and its verdicts are only
reproducible against exactly those bytes.  This utility documents and
automates the refresh path:

1. try Stooq (``https://stooq.com/q/d/l/?s=spy.us&d1=20050101
   &d2=20250101&i=d``) -- the endpoint currently serves a JavaScript
   proof-of-work challenge to plain HTTP clients and, after clearing
   it, may still answer ``Access denied`` for the historical CSV;
2. fall back to the Yahoo Finance chart API (same window, adjusted
   close included) and normalize it to the snapshot's column layout;
3. compare the fetched bytes against the committed hash.

The comparison is intentionally strict: adjusted closes are
vintage-dependent (every new SPY dividend re-bases the whole
back-adjusted history), and the provider recomputes them per request
-- re-fetches differ from the committed vintage in the last digits of
``AdjClose`` even within one day, while raw OHLCV is stable.  On ANY
mismatch this script RAISES and writes nothing -- use the committed
snapshot for the example, or consciously re-pin a new vintage by
replacing both ``data/SPY.csv`` and ``data/SPY.csv.sha256`` in a
commit of its own.

Run:

    uv run python examples/real_data/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "data" / "SPY.csv"
SIDECAR = HERE / "data" / "SPY.csv.sha256"

STOOQ_URL = "https://stooq.com/q/d/l/?s=spy.us&d1=20050101&d2=20250101&i=d"
YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?period1=1104537600&period2=1735689600&interval=1d"
    "&events=div%2Csplits&includeAdjustedClose=true"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
COLUMNS = ("Date", "Open", "High", "Low", "Close", "AdjClose", "Volume")


def _fetch(url: str) -> str:
    """GET ``url`` with a browser UA; return the body as text."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_stooq() -> str | None:
    """Stooq daily CSV, or None when the endpoint refuses plain clients."""
    body = _fetch(STOOQ_URL)
    if body.startswith("Date,"):
        return body
    print(
        "stooq: endpoint served a challenge/access denial instead of "
        "CSV; falling back to Yahoo Finance",
        file=sys.stderr,
    )
    return None


def _fetch_yahoo() -> str:
    """Yahoo chart API -> normalized CSV in the snapshot's layout."""
    payload = json.loads(_fetch(YAHOO_URL))
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


def committed_hash() -> str:
    """Hash pinned by the sidecar (first whitespace token)."""
    return SIDECAR.read_text(encoding="ascii").split()[0]


def main() -> None:
    """Fetch, normalize, compare against the committed hash, decide."""
    body: str | None = None
    try:
        body = _fetch_stooq()
    except OSError as error:
        print(f"stooq unreachable ({error}); trying Yahoo", file=sys.stderr)
    if body is None:
        body = _fetch_yahoo()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    pinned = committed_hash()
    if digest != pinned:
        raise RuntimeError(
            "fetched data hash does not match the committed snapshot "
            f"({digest} != {pinned}).  Adjusted closes are "
            "vintage-dependent, so upstream drift is expected: keep "
            "using the committed data/SPY.csv for reproducible runs, "
            "or consciously re-pin a new vintage by committing "
            "data/SPY.csv and data/SPY.csv.sha256 together."
        )
    SNAPSHOT.write_text(body, encoding="utf-8", newline="\n")
    print(f"snapshot verified and rewritten in place ({digest})")


if __name__ == "__main__":
    main()
