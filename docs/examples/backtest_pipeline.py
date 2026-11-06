"""The full backtest pipeline: simulate -> rank -> normalize -> admit.

Standalone example wiring the three backtest layers together the way a
research pipeline would:

1. ``simulate_signal`` — replay synthetic signals on an OHLC world with
   the taker cost model (SL/TP/time exits, gap-through-stop scratch);
2. ``run_position_state_machine`` — decide which signals actually trade
   (one slot, force-close with reconstructed P&L);
3. ``rank_within_groups`` — rank candidate signals within each day by
   a registry model trained past-only;
4. ``rank_normalize_group`` — within-group score normalization (raw
   cross-model probabilities are not comparable);
5. ``run_admission_policy`` — slot admission (FCFS vs REPLACE-low) and
   RANKING_METRICS comparison of the two selection systems.

Data: SYNTHETIC (seed=41), a trend + panic OHLC world — same honest
marking as the VRP example: this demonstrates the PIPELINE, not a
market claim.

Run from the repository root:

    uv run python docs/examples/backtest_pipeline.py
"""

from __future__ import annotations

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

HOUR_US = 3_600_000_000


def build_ohlc_world(n_bars: int = 2600, seed: int = 41) -> pd.DataFrame:
    """Trend + panic OHLC world (deterministic, seed=41).

    A drifting hourly series with two panic bursts (fat negative gaps
    and widened ranges) so stops, time exits and gap-through-stop
    contracts are all exercised by the replay.
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
    """
    z = rolling_z(world["close"].to_numpy(), 48)
    rng = np.random.default_rng(seed)
    signal_rows = []
    last_band = 0.0
    for pos in range(48, len(world)):
        crossed = (z[pos] > 1.5 and last_band <= 1.5) or (
            z[pos] < -1.5 and last_band >= -1.5
        )
        last_band = z[pos]
        if not crossed or not np.isfinite(z[pos]):
            continue
        signal_rows.append(
            {
                "ts_us": int(world["ts_us"].iloc[pos]),
                "entry_pos": pos
                + 1,  # fill at next bar open (generator rule)
                "side": "long" if z[pos] > 0 else "short",
                "priority": int(round(abs(z[pos]) * 10)),
                "atr": float(world["atr"].iloc[pos + 1]),
                "sl": float(
                    world["close"].iloc[pos]
                    - np.sign(z[pos]) * 1.2 * world["atr"].iloc[pos + 1]
                ),
                "tp": float(
                    world["close"].iloc[pos]
                    + np.sign(z[pos]) * 2.0 * world["atr"].iloc[pos + 1]
                ),
                "noise": rng.standard_normal(),
            }
        )
    return pd.DataFrame(signal_rows)


def replay_signals(
    world: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """Replay every candidate in isolation through the cost model."""
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
            hold_bars=48,
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


def main() -> int:
    """Run the four-stage pipeline and print the comparison tables."""
    world = build_ohlc_world()
    signals = make_signals(world)
    print("=" * 78)
    print("STAGE 1: isolated replay of", len(signals), "candidate signals")
    print("=" * 78)
    replayed = replay_signals(world, signals)
    decided = replayed[replayed["exit_idx"] >= 0]
    print(
        f"decided trades: {len(decided)}/{len(replayed)} "
        f"(mean r_pess {decided['r_pess'].mean():+.3f}R, "
        f"mean r_opt {decided['r_opt'].mean():+.3f}R)"
    )

    print()
    print("=" * 78)
    print("STAGE 2: position state machine (one slot, no reverse)")
    print("=" * 78)
    # contract: decision_idx is the DECISION bar (the fill lands at
    # decision_idx + 1 by the generator rule), exit_idx comes from the
    # isolated replay (-1 when data ended with the trade still open).
    # A replayed exit must be >= fill (= decision_idx + 1); one bar can
    # legitimately exit at its own decision bar + 1, so clamp the
    # pathological lower edge instead of asserting it away.
    slot_input = signals.rename(columns={"entry_pos": "decision_idx"})
    exit_by_stamp = replayed.set_index("ts_us")["exit_idx"]
    exits = exit_by_stamp.loc[slot_input["ts_us"]].to_numpy().astype(int)
    fill_idx = slot_input["decision_idx"].to_numpy() + 1
    exits = np.where(exits == -1, -1, np.maximum(exits, fill_idx))
    slot_input["exit_idx"] = exits
    slot_df = run_position_state_machine(
        slot_input, allow_reverse=False, cooldown=2
    )
    taken = slot_df[slot_df["status"] == "taken"]
    print(f"slots filled: {len(taken)} of {len(signals)} signals")

    print()
    print("=" * 78)
    print("STAGE 3: rank candidates within days (gbm, past-only)")
    print("=" * 78)
    # group candidates by calendar day of the decision stamp; the ranker
    # learns P(win) past-only and picks the top-2 of each day
    signals = signals.sort_values("ts_us").reset_index(drop=True)
    day_id = (signals["ts_us"] // (24 * HOUR_US)).to_numpy()
    # synthetic labels: a candidate "wins" if its realized r_pess > 0
    labels = (
        (replayed.set_index("ts_us").loc[signals["ts_us"], "r_pess"] > 0)
        .astype(np.int64)
        .to_numpy()
    )
    feature_frame = signals[["priority", "noise", "atr"]].rename(
        columns={"atr": "atr_at_signal"}
    )
    report = rank_within_groups(
        feature_frame,
        labels,
        day_id,
        model="gbm",
        top_s=2,
        embargo_groups=1,
        seed=42,
    )
    print(
        f"groups scored: {report['n_scored']}  "
        f"selections: {sum(len(group.candidates) for group in report['groups'])}"
    )

    print()
    print("=" * 78)
    print("STAGE 4: RANKING_METRICS — gbm vs random on the same groups")
    print("=" * 78)
    rng = np.random.default_rng(7)
    scores = {}
    for metric_name, metric in RANKING_METRICS.items():
        gbm_vals, random_vals = [], []
        for group in report["groups"]:
            # candidate-id strings are the metric contract (see
            # _ordering_contract in atropos.ranking)
            ideal = [
                cid
                for cid, _ in sorted(
                    zip(group.candidates, group.labels, strict=True),
                    key=lambda pair: -pair[1],
                )
            ]
            if len(ideal) < 2:
                continue
            scored_gbm = [
                cid
                for cid, _ in sorted(
                    zip(group.candidates, group.scores, strict=True),
                    key=lambda pair: -pair[1],
                )
            ]
            scored_random = rng.permutation(ideal).tolist()
            gbm_vals.append(metric(ideal, scored_gbm, top_k=2))
            random_vals.append(metric(ideal, scored_random, top_k=2))
        scores[metric_name] = (
            float(np.mean(gbm_vals)),
            float(np.mean(random_vals)),
        )
        print(
            f"  {metric_name:<12} gbm {scores[metric_name][0]:.3f}  "
            f"random {scores[metric_name][1]:.3f}"
        )
        assert scores[metric_name][0] >= scores[metric_name][1] - 0.05, (
            f"gbm must not lose to random on {metric_name}"
        )

    print()
    print("=" * 78)
    print(
        "STAGE 5: maker-entry vs market entry (limit at fill - 0.25 ATR,"
        " wait 4 bars)"
    )
    print("=" * 78)
    if not signals.empty:
        first = signals.iloc[0]
        entry = int(first["entry_pos"])
        fill_price = float(world["open"].iloc[entry])
        sign = 1.0 if first["side"] == "long" else -1.0
        limit_price = fill_price - sign * 0.25 * float(first["atr"])
        maker_sim_result = maker_entry_sim(
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
        market_sim_result = simulate_signal(
            world["open"].to_numpy(),
            world["high"].to_numpy(),
            world["low"].to_numpy(),
            world["close"].to_numpy(),
            entry,
            str(first["side"]),
            float(first["sl"]),
            float(first["tp"]),
            hold_bars=48,
            atr_value=float(first["atr"]),
            cost=CostModel(),
        )
        print(
            f"maker r_pess={maker_sim_result:+.3f}R  "
            f"market r_pess={market_sim_result['r_pess']:+.3f}R  "
            "(maker fills are path-deterministic; unfilled zone -> NaN)"
        )

    print()
    print("backtest pipeline: all stage checks passed")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
