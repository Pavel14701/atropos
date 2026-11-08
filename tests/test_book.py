"""Book pins: delegation equality, SL-first semantics, snapshots.

The core pin (review item 1): for legs WITHOUT signal exits, a Book
trade equals the PUBLIC simulate_intents result on the same intent —
equality against the public API, never the private scan.  Each test
names its mutation (applied -> fail -> restored before commit).

Known MVP wart (recorded in KNOWN-ISSUES section 14): the engine's
no-target sentinel price is negative for SHORT legs, which
TradeIntent's positive-price validation rejects -- tests here use
valid geometry (explicit targets on shorts).
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Literal

import pandas as pd
import pytest
from atropos.backtesting import (
    DEFAULT_COST,
    CostModel,
    TradeIntent,
    simulate_intents,
)
from atropos.book import Book, _demo_leg, _demo_md, run_book
from atropos.data_contract import (
    DataManifest,
    MaterializedData,
    ValidationReport,
)
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    StopSpec,
    TargetSpec,
)


def _make_book(**overrides: Any) -> Book:
    """Construct the demo Book with the declared carry policy.

    ``periods_override=252.0`` pins the explicit path: the demo
    bundle is manifest-less, and the accrual refuses to guess a
    calendar (loud, by design).
    """
    kwargs: dict[str, Any] = {
        "borrow_bps_annual": 500.0,
        "funding_bps_annual": 100.0,
        "periods_override": 252.0,
    }
    kwargs.update(overrides)
    return Book(**kwargs)


_BARS = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")


def _ohlc_md(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    mom: list[float],
    atr: list[float] | None = None,
) -> MaterializedData:
    """An OHLC decision grid with a mom gate column."""
    features = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "mom": mom,
            "atr": atr if atr is not None else [0.5] * len(opens),
        },
        index=_BARS[: len(opens)],
    )
    return MaterializedData(
        outcome=pd.Series(0.0, index=features.index),
        features=features,
        controls=pd.DataFrame(index=features.index),
        index=pd.DataFrame(
            {
                "ts": features.index,
                "entity": "",
                "split": "train",
            },
            index=features.index,
        ),
    )


def _leg(
    name: str = "demo_long",
    direction: str = "long",
    *,
    entry_gate: str = "mom",
    op: str = ">",
    with_target: bool = False,
    cooldown_bars: int = 0,
    signal: str | None = None,
    stop_multiple: float = 2.0,
) -> LegSpec:
    """Long/short demo geometry: 2-ATR stop, optional 3-ATR target."""
    exits = ExitSpec(stop=StopSpec("atr", stop_multiple))
    if with_target:
        exits = ExitSpec(
            stop=StopSpec("atr", stop_multiple),
            target=TargetSpec("atr", 3.0),
        )
    if signal is not None:
        exits = ExitSpec(
            stop=StopSpec("atr", stop_multiple),
            signal=Compare("mom", "<", 0.0)
            if signal == "flip"
            else Compare("mom", ">", 0.0),
        )
    gate_op: Literal[">", "<"] = ">" if op == ">" else "<"
    gate_value = 0.0
    leg_direction: Literal["long", "short"] = (
        "long" if direction == "long" else "short"
    )
    return LegSpec(
        name=name,
        direction=leg_direction,
        entry=EntrySpec(
            Compare(entry_gate, gate_op, gate_value),
            cooldown=pd.Timedelta(days=cooldown_bars),
        ),
        exit=exits,
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="book pin fixture",
    )


def test_delegation_equality_no_signal() -> None:
    """Book-trade == simulate_intents-trade for a stop-only leg.

    The review's core demand: the pin runs against the PUBLIC batch
    API (a private _exit_scan coupling would break on any refactor).
    Mutation fact: swapping the delegation for a naive target-first
    scanner (or dropping the intent translation) breaks the r/exit
    equality.
    """
    md = _demo_md()
    leg = _demo_leg()
    records = _make_book().run(leg, md)
    assert len(records) >= 1
    record = records[0]
    opens = md.features["open"].to_numpy(dtype=float)
    highs = md.features["high"].to_numpy(dtype=float)
    lows = md.features["low"].to_numpy(dtype=float)
    closes = md.features["close"].to_numpy(dtype=float)
    entry_ref = 100.0  # decision bar 1 close
    stop_price = entry_ref - 2.0 * 0.5
    reference = simulate_intents(
        [
            TradeIntent(
                entry_idx=2,
                side=1,
                stop_price=stop_price,
                target_price=entry_ref + 1e9 * 0.5,
                hold_bars=10_000_000,
                atr_value=0.5,
                risk_ref=abs(entry_ref - stop_price),
            )
        ],
        opens,
        highs,
        lows,
        closes,
    )[0]
    assert record["exit_bar"] == reference["exit_idx"]
    assert (
        record["exit_reason"]
        == {
            "sl": "stop",
            "tp": "target",
            "time": "time",
        }[reference["exit_reason"]]
    )
    # W3-1 semantic: the SIM still decides exits and r_pess; the
    # recorded NET is r_pess minus the carry drag (long: no borrow,
    # funding paid when the rate is positive)
    drag = (
        record["diagnostics"]["borrow_r"] + record["diagnostics"]["funding_r"]
    )
    assert record["pnl_net_r"] == pytest.approx(
        reference["r_pess"] - drag, abs=1e-12
    )
    assert record["diagnostics"]["borrow_r"] == 0.0  # long: no borrow
    assert drag > 0.0


def test_sl_first_race_and_gap_through() -> None:
    """Same-bar stop+target resolves as STOP; entry gap scratches.

    World: fill at 101 (stop 99, target 102.5); the bar at index 4
    touches BOTH (low 99, high 103) -> SL-first books the stop.
    Mutation fact: a target-first intrabar rule flips that race to
    "target"; dropping the gap scratch un-scratches the entry gap.
    """
    opens = [100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0]
    highs = [101.0, 101.0, 102.0, 102.0, 103.0, 101.0, 101.0, 101.0]
    lows = [99.0, 99.0, 100.0, 100.0, 97.0, 100.0, 100.0, 100.0]
    closes = [100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0]
    mom = [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    md = _ohlc_md(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        mom=mom,
        atr=[1.0] * 8,
    )
    records = _make_book().run(_leg(name="race", with_target=True), md)
    assert records != []
    # entry_ref = close[1] = 100 -> stop 98, target 103; bar 4 touches
    # BOTH (high 103, low 97) -> SL-first books the STOP
    assert records[0]["exit_reason"] == "stop"
    assert records[0]["exit_bar"] == 4
    assert records[0]["pnl_net_r"] < 0.0
    # gap-through: the entry bar opens THROUGH the stop -> scratch
    gap_opens = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    gap_lows = [99.0, 99.0, 89.0, 89.0, 89.0, 89.0, 89.0, 89.0]
    gap_md = _ohlc_md(
        opens=gap_opens,
        highs=highs,
        lows=gap_lows,
        closes=closes,
        mom=mom,
    )
    gap_records = _make_book().run(_leg(name="gap"), gap_md)
    assert gap_records != []
    scratch = gap_records[0]
    assert scratch["exit_reason"] == "stop"
    assert scratch["entry_bar"] == scratch["exit_bar"]
    assert scratch["pnl_net_r"] < 0.0


def test_signal_exit_and_forced() -> None:
    """Signal exits close at the engine's say-so; data-end forces.

    Mutation fact: ignoring exit_requested (or skipping the post-walk
    force-close) leaves the position open and the count/reason
    asserts fail.
    """
    md = _demo_md()
    signal_leg = _leg(name="signal_long", signal="flip")
    records = _make_book().run(signal_leg, md)
    assert records != []
    assert any(record["exit_reason"] == "signal" for record in records)
    evergreen = LegSpec(
        name="evergreen",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 100.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="no exit inside the window -> forced at data end",
    )
    forced_records = _make_book().run(evergreen, md)
    assert forced_records != []
    assert forced_records[-1]["exit_reason"] == "forced"
    assert forced_records[-1]["exit_bar"] == 7


def test_costs_and_snapshots() -> None:
    """Costs decompose honestly; snapshots come from the right bars.

    Pins: costs_r >= 0 and pnl_gross_r == pnl_net_r + costs_r on every
    resolved trade; features_at_entry carries the DECISION bar's row
    (the demo leg fires at bar 1, so mom 1.0).  Mutation fact: a
    snapshot taken at entry_bar+1 breaks the mom equality.
    """
    md = _demo_md()
    records = _make_book().run(_demo_leg(), md)
    for record in records:
        assert record["costs_r"] >= 0.0
        assert record["pnl_gross_r"] == pytest.approx(
            record["pnl_net_r"] + record["costs_r"], abs=1e-12
        )
    first = records[0]
    assert first["features_at_entry"]["mom"] == 1.0
    assert first["features_at_entry"]["atr"] == 0.5
    assert first["features_at_exit"]["mom"] == -1.0  # exit bar 4 row


def test_determinism_and_cost_model() -> None:
    """Two runs are identical; a zero-cost model changes only costs.

    Mutation fact: any rng or dict-ordering dependence makes the two
    runs diverge; dropping costs from the net P&L breaks the
    zero-cost comparison.
    """
    md = _demo_md()
    leg = _demo_leg()
    first = _make_book().run(leg, md)
    second = _make_book().run(leg, md)
    assert first == second
    free = _make_book(cost=CostModel(gen_slip=0.0, comm=0.0, gap=0.0)).run(
        leg, md
    )
    priced = _make_book(cost=DEFAULT_COST).run(leg, md)
    assert free[0]["exit_bar"] == priced[0]["exit_bar"]
    assert free[0]["pnl_net_r"] > priced[0]["pnl_net_r"]
    assert priced[0]["costs_r"] > 0.0


def test_cooldown_and_max_concurrent() -> None:
    """Cooldown suppresses re-entry; >1 slot is a loud deferred.

    Mutation fact: removing the cooldown gate produces a second
    entry inside the cooldown window and the no-reentry assert
    fails; silencing the max_concurrent rejection accepts garbage.
    """
    md = _demo_md()
    leg = _leg(name="cooled", with_target=True, cooldown_bars=10)
    records = _make_book().run(leg, md)
    assert records != []
    for earlier, later in itertools.pairwise(records):
        assert later["entry_bar"] > earlier["exit_bar"] + 10
    crowded = LegSpec(
        name="crowded",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0), max_concurrent=2),
        exit=ExitSpec(stop=StopSpec("atr", 2.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="multi-slot is deferred",
    )
    with pytest.raises(ValueError, match="deferred"):
        _make_book().run(crowded, md)


def test_run_book_isolation() -> None:
    """run_book runs each leg in a fresh Book: legs cannot interact.

    Mutation fact: a shared position state across legs (or lost leg
    attribution in the records map) breaks the isolation or
    attribution asserts.
    """
    md = _demo_md()
    long_leg = _demo_leg()
    short_leg = _leg(
        name="demo_short",
        direction="short",
        entry_gate="mom",
        op="<",
        with_target=True,
    )
    result = run_book(
        (long_leg, short_leg),
        md,
        borrow_bps_annual=500.0,
        funding_bps_annual=100.0,
        periods_override=252.0,
    )
    assert sorted(result["records"]) == ["demo_long", "demo_short"]
    for leg_name, records in result["records"].items():
        for record in records:
            assert record["leg_name"] == leg_name
    long_entries = {
        record["entry_bar"] for record in result["records"]["demo_long"]
    }
    short_entries = {
        record["entry_bar"] for record in result["records"]["demo_short"]
    }
    assert not long_entries & short_entries
    assert result["equity_r"] == pytest.approx(
        sum(
            record["pnl_net_r"]
            for records in result["records"].values()
            for record in records
        )
    )


# ---------------------------------------------------------------------------
# W3-1 -- carry accrual: calendar periods, formulas, signs
# ---------------------------------------------------------------------------
def test_periods_from_calendar_xnys_daily() -> None:
    """XNYS + daily grid resolves 252, never 365 (the footgun pin).

    The manifest carries the calendar; the derivation reads it.
    Mutation fact: deriving periods as year/bar_delta (365) instead
    of the calendar table breaks the 252 equality.
    """
    book = _make_book(periods_override=None)  # derivation must run
    periods = book._resolve_periods(_manifest_md(_demo_md(), calendar="XNYS"))
    assert periods == 252.0


def test_periods_hourly_xnys_and_override() -> None:
    """XNYS + hourly -> 1638; an explicit override wins.

    Mutation fact: dropping the session-hours term (or the override
    branch) breaks either equality.
    """
    book = _make_book(periods_override=None)
    hourly = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    md = _manifest_md(
        MaterializedData(
            outcome=pd.Series(0.0, index=hourly),
            features=pd.DataFrame(
                {
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "mom": 1.0,
                    "atr": 1.0,
                },
                index=hourly,
            ),
            controls=pd.DataFrame(index=hourly),
            index=pd.DataFrame(
                {"ts": hourly, "entity": "", "split": "train"},
                index=hourly,
            ),
        ),
        calendar="XNYS",
    )
    assert book._resolve_periods(md) == 1638.0
    # an explicit override wins over the calendar table
    override_book = _make_book(periods_override=8760.0)
    assert override_book._resolve_periods(md) == 8760.0


def test_periods_no_manifest_no_override_fatal() -> None:
    """Manifest-less + no override refuses loudly (never guesses).

    Mutation fact: a silent default calendar (or a guessed 252)
    makes the pytest.raises fail.
    """
    book = _make_book(periods_override=None)
    with pytest.raises(ValueError, match="refuses to guess"):
        book._resolve_periods(_demo_md())
    unknown = _manifest_md(_demo_md(), calendar="JPX")
    with pytest.raises(ValueError, match="unknown calendar"):
        book._resolve_periods(unknown)


def _manifest_md(md: MaterializedData, *, calendar: str) -> MaterializedData:
    """Return ``md`` with a minimal manifest carrying ``calendar``."""
    manifest: DataManifest = {
        "bundle_hash": "0" * 64,
        "source_versions": {},
        "calendar": calendar,
        "row_count": len(md.index),
        "train_rows": 0,
        "test_rows": 0,
        "dropped_embargo_gap": 0,
        "date_range": ("", ""),
        "nan_fraction": {},
        "dropped": {},
        "warnings": [],
        "feature_hashes": {},
        "lookahead_checks": {},
        "validation": ValidationReport(
            passed=True,
            fatal=[],
            warnings=[],
        ),
    }
    return MaterializedData(
        outcome=md.outcome,
        features=md.features,
        controls=md.controls,
        index=md.index,
        manifest=manifest,
    )


def test_borrow_short_only_funding_sign() -> None:
    """Borrow is shorts-only; funding follows the rate's sign.

    A paying short (borrow > 0) loses MORE than its sim R; a paying
    long (funding > 0) loses; a receiving long (funding < 0) GAINS
    over the sim R.  Mutation facts: charging borrow on longs breaks
    the zero-borrow long; rejecting negative funding breaks the
    receiving-long world construction; flipping the funding sign
    breaks the loses/gains ordering.
    """
    md = _demo_md()
    paying_long = _make_book().run(_demo_leg(), md)
    long_diag = paying_long[0]["diagnostics"]
    assert long_diag["borrow_r"] == 0.0  # the borrow book is shorts-only
    assert long_diag["funding_r"] > 0.0  # pays positive funding
    paying = paying_long[0]["pnl_net_r"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free_long = _make_book(
            borrow_bps_annual=0.0, funding_bps_annual=0.0
        ).run(_demo_leg(), md)
        receiving_long = _make_book(funding_bps_annual=-100.0).run(
            _demo_leg(), md
        )
    # paying costs money; receiving gains it (vs the zero-carry R)
    assert free_long[0]["pnl_net_r"] > paying
    assert receiving_long[0]["pnl_net_r"] > paying

    # a HOLDING short pays borrow; a negative funding rate makes the
    # short pay too (sign-carried)
    short_leg = _leg(
        name="short_hold", direction="short", op="<", stop_multiple=20.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short_records = _make_book(
            borrow_bps_annual=2000.0, funding_bps_annual=-100.0
        ).run(short_leg, md)
    assert short_records != []
    short_diag = short_records[0]["diagnostics"]
    assert short_diag["borrow_r"] > 0.0  # shorts pay borrow
    assert short_diag["funding_r"] > 0.0  # negative rate: shorts pay


def test_zero_carry_rates_warn() -> None:
    """Zero rates are legal but warn (the no-silent-zero pin).

    Mutation fact: removing either zero-warning makes the
    pytest.warns fail.
    """
    with pytest.warns(UserWarning, match="borrow_bps_annual=0"):
        Book(borrow_bps_annual=0.0, funding_bps_annual=100.0)
    with pytest.warns(UserWarning, match="funding_bps_annual=0"):
        Book(borrow_bps_annual=500.0, funding_bps_annual=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        Book(borrow_bps_annual=-50.0, funding_bps_annual=100.0)
    # negative funding is LEGAL (sign-carried)
    neg = Book(borrow_bps_annual=500.0, funding_bps_annual=-100.0)
    assert neg.funding_bps_annual == -100.0


def test_same_bar_close_zero_carry() -> None:
    """A same-bar close accrues zero carry (discrete-bar rule).

    Mutation fact: counting the entry bar (k+1 semantics) makes the
    zero assert fail.
    """
    opens = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    highs = [101.0, 101.0, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0]
    lows = [99.0, 99.0, 89.0, 89.0, 89.0, 89.0, 89.0, 89.0]
    closes = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    mom = [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    md = _ohlc_md(opens=opens, highs=highs, lows=lows, closes=closes, mom=mom)
    records = _make_book().run(_leg(name="gap_carry"), md)
    assert records != []
    scratch = records[0]
    assert scratch["entry_bar"] == scratch["exit_bar"]
    assert scratch["diagnostics"]["borrow_r"] == 0.0
    assert scratch["diagnostics"]["funding_r"] == 0.0


def test_carry_split_attribution() -> None:
    """Taker + borrow + funding == costs_r, exactly (the split pin).

    Mutation fact: collapsing the split into taker (or dropping any
    component) breaks the identity and the per-component asserts.
    """
    md = _demo_md()
    records = _make_book().run(_demo_leg(), md)
    diag = records[0]["diagnostics"]
    assert diag["taker_r"] > 0.0
    assert diag["funding_r"] > 0.0
    # k is pinned: the demo leg enters at bar 2, exits at bar 4
    # (two CLOSED hold bars) -- a k+1 mutation shifts funding_r by
    # exactly one per-bar unit and fails this exact-value pin.
    assert records[0]["exit_bar"] - records[0]["entry_bar"] == 2
    per_bar_funding = (
        100.0 / 1e4 / 252.0 * (1.0 * 1.0 * 101.0 / (101.0 - 99.0))
    )
    assert diag["funding_r"] == pytest.approx(
        2.0 * per_bar_funding, abs=1e-12
    )
    total = diag["taker_r"] + diag["borrow_r"] + diag["funding_r"]
    assert records[0]["costs_r"] == pytest.approx(total, abs=1e-15)
    assert records[0]["pnl_gross_r"] == pytest.approx(
        records[0]["pnl_net_r"] + records[0]["costs_r"], abs=1e-15
    )
