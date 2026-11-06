"""The packaged backtest demo: simulate -> rank -> normalize -> admit.

One callable wiring the three backtest layers together the way a
research pipeline would - the golden path behind ``atropos demo``:

1. ``simulate_signal`` - replay synthetic signals on an OHLC world with
   the taker cost model (SL/TP/time exits, gap-through-stop scratch);
2. ``run_position_state_machine`` - decide which signals actually trade
   (one slot, force-close with reconstructed P&L);
3. ``rank_within_groups`` - rank candidate signals within each day by
   a registry model trained past-only;
4. ``RANKING_METRICS`` - gbm arm vs a random arm on the same groups
   (raw within-group score order IS the metric order);
5. ``maker_entry_sim`` vs ``simulate_signal`` - maker-entry vs market
   entry on the first candidate.

Data: SYNTHETIC (default seed=41), a trend + panic OHLC world - same
honest marking as the VRP example: this demonstrates the PIPELINE, not
a market claim.

Run it (``--mode quick`` is a reduced model budget, seconds; ``--mode
full`` reproduces the docs transcript)::

    atropos demo --mode quick
    atropos demo --mode full

or programmatically, silencing the stage report::

    >>> summary = run_demo_pipeline(seed=41, quick=True, verbose=False)
    >>> summary["pipeline_completed"]
    True

Every printed value is deterministic for a fixed seed: the world, the
signals, the past-only gbm fits and the random comparison arm all draw
from seeds pinned inside this module, and no wall-clock value is ever
printed.  The ranking gate (gbm must not lose to random by more than
0.05) is a REPORTED verdict, not a crash: the synthetic labels are
near-degenerate (a handful of winning candidates), so on seeds where
no past-only window ever holds both classes the stage is skipped
honestly and on seeds where gbm loses the verdict says FAIL - only
structural pipeline failures raise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atropos.backtesting import (
    CostModel,
    maker_entry_sim,
    run_position_state_machine,
    simulate_signal,
)
from atropos.ml_training import rolling_z
from atropos.ranking import (
    RANKING_METRICS,
    rank_within_groups,
)

__all__ = [
    "build_ohlc_world",
    "make_signals",
    "replay_signals",
    "run_demo_pipeline",
]

HOUR_US = 3_600_000_000
"""One hour in microseconds (the synthetic world's bar spacing)."""

_N_BARS = 2600
"""Hourly bars of the demo world (both modes share the same world)."""

_QUICK_GBM_TREES = 60
"""gbm ``n_estimators`` override for quick mode (default: 400)."""

_HOLD_BARS = 48
"""Isolated-replay hold window shared by both modes."""

_GATE_FLOOR = -0.05
"""Ranking-gate floor: gbm may trail random by at most this much."""


def build_ohlc_world(n_bars: int = _N_BARS, seed: int = 41) -> pd.DataFrame:
    """Trend + panic OHLC world (deterministic, seed=41 default).

    A drifting hourly series with two panic bursts (fat negative gaps
    and widened ranges) so stops, time exits and gap-through-stop
    contracts are all exercised by the replay.

    Args:
        n_bars: number of hourly bars to generate; use >= 144 so both
            panic bursts land inside the world (smaller values crash
            the second burst slice), and >= 49 for the signal
            generator's 48-bar warm-up window.
        seed: RNG seed of the noise/gap draws; the same seed always
            produces the same world.

    Returns:
        OHLC frame with columns ``open`` / ``high`` / ``low`` /
        ``close`` / ``atr`` (20-bar rolling range mean) / ``ts_us``
        (hourly microsecond stamps from 2024-01-01).

    Example::

        >>> world = build_ohlc_world(n_bars=150, seed=41)
        >>> len(world)
        150
        >>> tuple(world.columns)
        ('open', 'high', 'low', 'close', 'atr', 'ts_us')
    """
    rng = np.random.default_rng(seed)
    drift = np.concatenate(
        [
            np.full(n_bars // 3, 0.0004),
            np.full(n_bars // 3, -0.0006),
            np.full(n_bars - 2 * (n_bars // 3), 0.0004),
        ]
    )
    close = 100.0 + np.cumsum(drift + 0.004 * rng.standard_normal(n_bars))
    # two panic bursts: gaps down + widened ranges
    for start in (n_bars // 3 + 20, 2 * (n_bars // 3) + 40):
        close[start : start + 8] -= np.linspace(0.8, 0.1, 8)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.standard_normal(n_bars)) * 0.3 + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    atr = pd.Series(high - low).rolling(20).mean().bfill().to_numpy()
    stamps = (
        int(np.datetime64("2024-01-01T00:00:00", "us").astype(np.int64))
        + np.arange(n_bars, dtype=np.int64) * HOUR_US
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "ts_us": stamps,
        }
    )


def make_signals(world: pd.DataFrame, seed: int = 41) -> pd.DataFrame:
    """Candidate signals: z-scored momentum crossings, both sides.

    The momentum z is computed causally (``rolling_z``); a signal fires
    when the z crosses the +/-1.5 band, with a side matching the sign.
    Priority encodes |z| so the ranking layer has something real to
    order by.

    Args:
        world: OHLC frame from :func:`build_ohlc_world` (needs the
            ``close`` / ``atr`` / ``ts_us`` columns and at least 49
            bars).
        seed: RNG seed of the per-signal noise feature (a filler
            column so the ranker sees an uninformative feature too).

    Returns:
        One row per crossing with columns ``ts_us`` / ``entry_pos``
        (fill at the NEXT bar open, the generator rule) / ``side`` /
        ``priority`` / ``atr`` / ``sl`` / ``tp`` / ``noise``.

    Example::

        >>> world = build_ohlc_world(n_bars=900, seed=41)
        >>> signals = make_signals(world, seed=41)
        >>> len(signals) > 0
        True
        >>> tuple(sorted(signals.columns))
        ('atr', 'entry_pos', 'noise', 'priority', 'side', 'sl', 'tp', 'ts_us')
    """
    momentum_z = rolling_z(world["close"].to_numpy(), 48)
    rng = np.random.default_rng(seed)
    signal_rows = []
    last_band = 0.0
    for pos in range(48, len(world)):
        crossed = (momentum_z[pos] > 1.5 and last_band <= 1.5) or (
            momentum_z[pos] < -1.5 and last_band >= -1.5
        )
        last_band = momentum_z[pos]
        if not crossed or not np.isfinite(momentum_z[pos]):
            continue
        signal_rows.append(
            {
                "ts_us": int(world["ts_us"].iloc[pos]),
                "entry_pos": pos
                + 1,  # fill at next bar open (generator rule)
                "side": "long" if momentum_z[pos] > 0 else "short",
                "priority": round(float(abs(momentum_z[pos])) * 10),
                "atr": float(world["atr"].iloc[pos + 1]),
                "sl": float(
                    world["close"].iloc[pos]
                    - np.sign(momentum_z[pos])
                    * 1.2
                    * world["atr"].iloc[pos + 1]
                ),
                "tp": float(
                    world["close"].iloc[pos]
                    + np.sign(momentum_z[pos])
                    * 2.0
                    * world["atr"].iloc[pos + 1]
                ),
                "noise": rng.standard_normal(),
            }
        )
    return pd.DataFrame(signal_rows)


def replay_signals(
    world: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """Replay every candidate in isolation through the cost model.

    Args:
        world: OHLC frame from :func:`build_ohlc_world`.
        signals: candidate frame from :func:`make_signals`.

    Returns:
        The signal columns joined with the isolated replay outcome:
        ``r_opt`` / ``r_pess`` (optimistic vs pessimistic net R) and
        ``exit_idx`` (-1 when the data ended with the trade still
        open).  Candidates whose ``entry_pos`` falls past the world
        are dropped (nothing to replay); the numeric outcome columns
        are coerced back to numeric because ``row.to_dict()`` mixes
        the signal dtypes.

    Example::

        >>> world = build_ohlc_world(n_bars=900, seed=41)
        >>> signals = make_signals(world, seed=41)
        >>> replayed = replay_signals(world, signals)
        >>> {"r_opt", "r_pess", "exit_idx"} <= set(replayed.columns)
        True
    """
    records = []
    for _, row in signals.iterrows():
        entry = int(row["entry_pos"])
        if entry >= len(world):
            continue
        sim = simulate_signal(
            world["open"].to_numpy(),
            world["high"].to_numpy(),
            world["low"].to_numpy(),
            world["close"].to_numpy(),
            entry,
            str(row["side"]),
            float(row["sl"]),
            float(row["tp"]),
            hold_bars=_HOLD_BARS,
            atr_value=float(row["atr"]),
            cost=CostModel(),
        )
        records.append(
            {
                **row.to_dict(),
                "r_opt": sim["r_opt"],
                "r_pess": sim["r_pess"],
                "exit_idx": sim["exit_idx"],
            }
        )
    frame = pd.DataFrame(records)
    # row.to_dict() mixes the signal dtypes; restore the numeric columns
    # the downstream filters rely on
    for numeric_column in ("r_opt", "r_pess", "exit_idx", "priority", "atr"):
        frame[numeric_column] = pd.to_numeric(
            frame[numeric_column], errors="coerce"
        )
    return frame


def run_demo_pipeline(  # noqa: PLR0912, PLR0915  (one block per stage)
    seed: int = 41, quick: bool = False, *, verbose: bool = True
) -> dict[str, Any]:
    """Run the five-stage demo pipeline and print the stage report.

    The stages (isolated replay -> slot machine -> past-only gbm
    ranking -> RANKING_METRICS gbm-vs-random -> maker-vs-market) run
    on a synthetic world; ``quick`` shrinks the model budget (60 gbm
    trees instead of 400) to a seconds-scale run while ``full``
    reproduces the committed docs transcript at ``seed=41``.  Every
    printed value is deterministic for a fixed seed.

    The ranking gate (gbm must not trail random by more than 0.05 on
    any RANKING_METRICS mean) is REPORTED, not raised: the synthetic
    labels are near-degenerate, so on seeds whose world produces no
    winning candidate the ranking stage is skipped honestly (gate
    ``"skipped"``, pipeline completes) and on seeds where gbm loses
    the gate fails (``pipeline_completed=False``).  Structural
    failures - a world without signals, a machine that trades
    nothing - still raise AssertionError.

    Args:
        seed: RNG seed of the synthetic world and its signals; the
            same seed prints the same report bit-for-bit (the ranker
            seed 42 and the random-arm seed 7 are pinned inside).
        quick: ``True`` for the reduced model budget (``--mode
            quick``), ``False`` for the full docs transcript.
        verbose: keyword-only; ``False`` silences the STAGE/VERDICT
            printing (used by tests and doctests) without changing
            the computed results.

    Returns:
        Summary dict with ``seed``, ``quick``, ``n_bars``,
        ``n_signals``, ``n_decided``, ``mean_r_pess``, ``mean_r_opt``,
        ``slots_filled``, ``n_scored_groups``, ``n_selections``,
        ``metrics`` (metric name -> ``(gbm, random)`` means, empty
        when ranking was skipped), ``ranking_gate`` (``"pass"`` /
        ``"fail"`` / ``"skipped"``), ``maker_r_pess``,
        ``market_r_pess`` and ``pipeline_completed``.

    Example::

        >>> summary = run_demo_pipeline(seed=41, quick=True,
        ...                             verbose=False)
        >>> summary["pipeline_completed"]
        True
        >>> summary["slots_filled"] > 0
        True
    """
    rank_config: dict[str, Any] | None = (
        {"model_params": {"gbm": {"n_estimators": _QUICK_GBM_TREES}}}
        if quick
        else None
    )

    def say(message: str) -> None:
        """Print one report line unless the run is silenced."""
        if verbose:
            print(message)

    world = build_ohlc_world(n_bars=_N_BARS, seed=seed)
    signals = make_signals(world, seed=seed)
    assert len(signals) > 0, "demo world produced no candidate signals"

    say("=" * 78)
    say(f"STAGE 1: isolated replay of {len(signals)} candidate signals")
    say("=" * 78)
    replayed = replay_signals(world, signals)
    decided = replayed[replayed["exit_idx"] >= 0]
    say(
        f"decided trades: {len(decided)}/{len(replayed)} "
        f"(mean r_pess {decided['r_pess'].mean():+.3f}R, "
        f"mean r_opt {decided['r_opt'].mean():+.3f}R)"
    )

    say("")
    say("=" * 78)
    say("STAGE 2: position state machine (one slot, no reverse)")
    say("=" * 78)
    # contract: decision_idx is the DECISION bar (the fill lands at
    # decision_idx + 1 by the generator rule), exit_idx comes from the
    # isolated replay (-1 when data ended with the trade still open).
    # A replayed exit must be >= fill (= decision_idx + 1); one bar can
    # legitimately exit at its own decision bar + 1, so clamp the
    # pathological lower edge instead of asserting it away.  A candidate
    # whose entry lands past the world was not replayed (nothing to
    # exit); it is dropped here and in the rank stage below.
    slot_input = signals.rename(columns={"entry_pos": "decision_idx"})
    slot_input = slot_input[
        slot_input["ts_us"].isin(replayed["ts_us"])
    ].copy()
    exit_by_stamp = replayed.set_index("ts_us")["exit_idx"]
    exits = exit_by_stamp.loc[slot_input["ts_us"]].to_numpy().astype(int)
    fill_idx = slot_input["decision_idx"].to_numpy() + 1
    exits = np.where(exits == -1, -1, np.maximum(exits, fill_idx))
    slot_input["exit_idx"] = exits
    slot_df = run_position_state_machine(
        slot_input, allow_reverse=False, cooldown=2
    )
    taken = slot_df[slot_df["status"] == "taken"]
    slots_filled = len(taken)
    assert slots_filled >= 1, "demo slot machine traded nothing"
    say(f"slots filled: {len(taken)} of {len(slot_input)} signals")

    say("")
    say("=" * 78)
    say("STAGE 3: rank candidates within days (gbm, past-only)")
    say("=" * 78)
    # group candidates by calendar day of the decision stamp; the ranker
    # learns P(win) past-only and picks the top-2 of each day
    ranked = signals.sort_values("ts_us").reset_index(drop=True)
    ranked = ranked[ranked["ts_us"].isin(replayed["ts_us"])]
    ranked = ranked.reset_index(drop=True)
    day_id = (ranked["ts_us"] // (24 * HOUR_US)).to_numpy()
    # synthetic labels: a candidate "wins" if its realized r_pess > 0
    labels = (
        (replayed.set_index("ts_us").loc[ranked["ts_us"], "r_pess"] > 0)
        .astype(np.int64)
        .to_numpy()
    )
    feature_frame = ranked[["priority", "noise", "atr"]].rename(
        columns={"atr": "atr_at_signal"}
    )
    report: dict[str, Any] | None
    try:
        report = rank_within_groups(
            feature_frame,
            labels,
            day_id,
            model="gbm",
            top_s=2,
            embargo_groups=1,
            config=rank_config,
            seed=42,
        )
    except ValueError:
        # honest skip, never a silent one: this seed's world produced
        # no winning candidate, so every past-only training window was
        # single-class and the ranker has nothing to learn; the verdict
        # below reports the skip explicitly
        report = None
        say(
            "groups scored: 0 (single-class windows: no winning "
            "candidate to learn from)"
        )
    if report is not None:
        selections = sum(len(group.candidates) for group in report["groups"])
        say(f"groups scored: {report['n_scored']}  selections: {selections}")
    else:
        selections = 0

    say("")
    say("=" * 78)
    say("STAGE 4: RANKING_METRICS — gbm vs random on the same groups")
    say("=" * 78)
    metric_values: dict[str, tuple[float, float]] = {}
    worst_diff = float("nan")
    if report is None:
        say("  skipped: the ranking stage scored no groups")
        ranking_gate = "skipped"
    else:
        rng = np.random.default_rng(7)
        for metric_name, metric in RANKING_METRICS.items():
            gbm_values: list[float] = []
            random_values: list[float] = []
            for group in report["groups"]:
                # candidate-id strings are the metric contract (see
                # _ordering_contract in atropos.ranking)
                ideal = [
                    candidate
                    for candidate, _ in sorted(
                        zip(group.candidates, group.labels, strict=True),
                        key=lambda pair: -pair[1],
                    )
                ]
                if len(ideal) < 2:
                    continue
                scored_gbm = [
                    candidate
                    for candidate, _ in sorted(
                        zip(group.candidates, group.scores, strict=True),
                        key=lambda pair: -pair[1],
                    )
                ]
                scored_random = rng.permutation(ideal).tolist()
                gbm_values.append(metric(ideal, scored_gbm, 2))
                random_values.append(metric(ideal, scored_random, 2))
            values = (
                float(np.mean(gbm_values)),
                float(np.mean(random_values)),
            )
            metric_values[metric_name] = values
            say(
                f"  {metric_name:<12} gbm {values[0]:.3f}  "
                f"random {values[1]:.3f}"
            )
            diff = values[0] - values[1]
            worst_diff = (
                diff if np.isnan(worst_diff) else min(worst_diff, diff)
            )
        ranking_gate = "pass" if worst_diff >= _GATE_FLOOR else "fail"

    say("")
    say("=" * 78)
    say(
        "STAGE 5: maker-entry vs market entry (limit at fill - 0.25 ATR,"
        " wait 4 bars)"
    )
    say("=" * 78)
    maker_r_pess = float("nan")
    market_r_pess = float("nan")
    if not signals.empty:
        first = signals.iloc[0]
        entry = int(first["entry_pos"])
        fill_price = float(world["open"].iloc[entry])
        sign = 1.0 if first["side"] == "long" else -1.0
        limit_price = fill_price - sign * 0.25 * float(first["atr"])
        maker_r_pess = float(
            maker_entry_sim(
                world["open"].to_numpy(),
                world["high"].to_numpy(),
                world["low"].to_numpy(),
                world["close"].to_numpy(),
                entry,
                str(first["side"]),
                float(first["sl"]),
                float(first["tp"]),
                fill_price=limit_price,
                atr_value=float(first["atr"]),
            )
        )
        market_r_pess = float(
            simulate_signal(
                world["open"].to_numpy(),
                world["high"].to_numpy(),
                world["low"].to_numpy(),
                world["close"].to_numpy(),
                entry,
                str(first["side"]),
                float(first["sl"]),
                float(first["tp"]),
                hold_bars=_HOLD_BARS,
                atr_value=float(first["atr"]),
                cost=CostModel(),
            )["r_pess"]
        )
        say(
            f"maker r_pess={maker_r_pess:+.3f}R  "
            f"market r_pess={market_r_pess:+.3f}R  "
            "(maker fills are path-deterministic; unfilled zone -> NaN)"
        )
    else:
        say("maker-vs-market: skipped (no candidate signals)")

    pipeline_completed = ranking_gate != "fail"
    say("")
    if pipeline_completed:
        say("backtest pipeline: all stage checks passed")
    else:
        say("backtest pipeline: ranking gate FAILED (see verdict)")
    mrr_gbm, mrr_random = metric_values.get(
        "mrr", (float("nan"), float("nan"))
    )
    say("")
    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    if ranking_gate == "skipped":
        say(
            f"pipeline completed: mode={'quick' if quick else 'full'},"
            f" seed={seed}, ranking gate SKIPPED (single-class"
            " windows)"
        )
        say(
            "  gbm-vs-random mrr: n/a (the synthetic world produced no"
            " winning candidate for this seed)"
        )
    elif ranking_gate == "fail":
        say(
            f"pipeline verdict: ranking gate FAIL on seed={seed}"
            f" (worst gbm-random {worst_diff:+.3f} < {_GATE_FLOOR:+.3f})"
        )
        say(
            f"  gbm-vs-random mrr: gbm {mrr_gbm:.3f}  random {mrr_random:.3f}"
        )
    else:
        say(
            f"pipeline completed: mode={'quick' if quick else 'full'},"
            f" seed={seed}, ranking gate PASS"
        )
        say(
            f"  gbm-vs-random mrr: gbm {mrr_gbm:.3f}  random {mrr_random:.3f}"
        )
    say(
        f"  candidates: {len(signals)} signals, {len(decided)} decided,"
        f" {slots_filled} slots filled"
    )
    say(
        f"  maker-vs-market r_pess: maker {maker_r_pess:+.3f}R  "
        f"market {market_r_pess:+.3f}R"
    )
    return {
        "seed": int(seed),
        "quick": bool(quick),
        "n_bars": int(_N_BARS),
        "n_signals": len(signals),
        "n_decided": len(decided),
        "mean_r_pess": float(decided["r_pess"].mean()),
        "mean_r_opt": float(decided["r_opt"].mean()),
        "slots_filled": slots_filled,
        "n_scored_groups": int(report["n_scored"])
        if report is not None
        else 0,
        "n_selections": int(selections),
        "metrics": dict(metric_values),
        "ranking_gate": ranking_gate,
        "maker_r_pess": maker_r_pess,
        "market_r_pess": market_r_pess,
        "pipeline_completed": bool(pipeline_completed),
    }
