"""Robustness battery: seven adversarial tests for any signal -> outcome rule.

The battery trades the signal mechanically (long when ``signal >
threshold``, flat otherwise, position lagged one period) and then attacks
the resulting daily strategy-return stream with seven tests:

1. threshold sensitivity - threshold scaled by 0.5x / 1.0x / 1.5x
2. window sensitivity    - signal smoothed over 0.5x / 1.0x / 1.5x lookback
3. subperiod thirds      - effect present in each sample third?
4. leave-one-out (LOO)   - no single event day carries the effect?
5. placebo lag           - shifting the signal by ``window`` periods kills
                           it?
6. declustering          - effect survives keeping only the first day of
                           each cluster of consecutive active days?
7. sign monotonicity     - terciles of the signal are monotonically
                           ordered in the next-period outcome?

All confidence intervals use a circular moving-block bootstrap
(n_boot=1000, block=20) to handle overlapping returns and
autocorrelation.

Overall verdict:
* ``DEAD``    - any test is catastrophic (CI excludes zero with the
  OPPOSITE sign to the baseline effect),
* ``ROBUST``  - at least 5 of 7 tests pass,
* ``FRAGILE`` - otherwise.

Usage::

    from robustness_battery import run_battery, battery_overall

    table = run_battery(signal_series, outcome_series,
                        {"threshold": 0.5, "window": 20})
    print(battery_overall(table))

Run ``python -m atropos.robustness_battery`` for a synthetic demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["battery_overall", "run_battery"]

_REQUIRED = ("threshold", "window")
_MIN_PASS_ROBUST = 5


# ----------------------------------------------------------------------------
# helpers (self-contained: duplicated so the module imports alone)
# ----------------------------------------------------------------------------
def _frameify(source: Any, tag: str) -> pd.DataFrame:
    if isinstance(source, pd.Series):
        return source.to_frame(name=tag)
    if isinstance(source, pd.DataFrame):
        return source.add_prefix(tag + "::")
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 1:
        return pd.DataFrame({tag: arr})
    if arr.ndim != 2:
        raise ValueError(
            f"input {tag!r} must be 1-D or 2-D, got ndim={arr.ndim}"
        )
    return pd.DataFrame(
        arr, columns=[f"{tag}::{col}" for col in range(arr.shape[1])]
    )


def _align(*arrays: Any) -> list[np.ndarray]:
    """Align pandas inputs by index, numpy by position; drop NaN rows.

    Mixing pandas and numpy inputs is allowed: numpy arrays inherit the
    index of the first pandas input when lengths match (positional
    alignment).
    """
    frames = [_frameify(item, f"v{slot}") for slot, item in enumerate(arrays)]
    ref_index = None
    for item in arrays:
        if isinstance(item, (pd.Series, pd.DataFrame)) and not isinstance(
            item.index, pd.RangeIndex
        ):
            ref_index = item.index
            break
    if ref_index is not None:
        # frames is built 1:1 from arrays, so the strict zip is safe
        for slot, (item, frame) in enumerate(
            zip(arrays, frames, strict=True)
        ):
            if not isinstance(item, (pd.Series, pd.DataFrame)):
                if len(frame) == len(ref_index):
                    frame.index = ref_index
                else:
                    raise ValueError(
                        f"input {slot} (numpy, len={len(frame)}) cannot"
                        " be aligned with the pandas index"
                        f" (len={len(ref_index)}); pass pandas inputs or"
                        " equal-length arrays"
                    )
    comb = pd.concat(frames, axis=1, join="inner").dropna()
    out: list[np.ndarray] = []
    pos = 0
    for frame in frames:
        width = frame.shape[1]
        block = comb.iloc[:, pos : pos + width].to_numpy(dtype=float)
        out.append(block.reshape(-1) if width == 1 else block)
        pos += width
    return out


def _pos_from_signal(signal: np.ndarray, threshold: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (signal > threshold).astype(float)  # NaN -> False -> flat


def _strat_returns(
    signal: np.ndarray, outcome: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Daily strategy returns, position lagged one period (no lookahead)."""
    pos = _pos_from_signal(signal, threshold)
    strat = np.zeros_like(outcome)
    strat[1:] = pos[:-1] * outcome[1:]
    return strat, pos


def _boot_ci_mean(
    ret: np.ndarray,
    n_boot: int,
    block: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Block-bootstrap CI and SE for the mean of ``ret``."""
    vals = ret[np.isfinite(ret)]
    n_obs = int(vals.size)
    if n_obs < 10:
        return float("nan"), float("nan"), float("nan")
    block = int(min(max(block, 1), n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs, size=(int(n_boot), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    idx = idx.reshape(int(n_boot), n_blocks * block)[:, :n_obs]
    means = vals[idx].mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(means, tail))
    hi = float(np.quantile(means, 1.0 - tail))
    se = float(np.std(means, ddof=1))
    return lo, hi, se


def _effect(
    signal: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    n_boot: int,
    block: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    strat, pos = _strat_returns(signal, target, threshold)
    mu = float(np.mean(strat)) if strat.size else float("nan")
    lo, hi, se = _boot_ci_mean(strat, n_boot, block, ci_level, seed)
    t_stat = mu / se if (np.isfinite(se) and se > 0) else float("nan")
    return {
        "mu": mu,
        "t": t_stat,
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "n_active": int(pos.sum()),
        "pos": pos,
        "r": strat,
    }


def _excl0(eff: dict[str, Any]) -> bool:
    return bool(
        np.isfinite(eff["ci_low"])
        and np.isfinite(eff["ci_high"])
        and (eff["ci_high"] < 0 or eff["ci_low"] > 0)
    )


def _ci_sign(eff: dict[str, Any]) -> int:
    """+1 / -1 if the CI excludes zero, else 0."""
    if not _excl0(eff):
        return 0
    return 1 if eff["ci_low"] > 0 else -1


def _roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    csum = np.cumsum(np.concatenate([[0.0], values]))
    head = csum[1:window] / np.arange(1, window)  # expanding means first
    tail = (csum[window:] - csum[:-window]) / window  # trailing windows
    return np.concatenate([head, tail])


# ----------------------------------------------------------------------------
# the battery
# ----------------------------------------------------------------------------
# 7-stage battery orchestrator: one branch per test; decomposing the
# sequential battery would hurt readability, not reduce real complexity.
def run_battery(  # noqa: PLR0912, PLR0915  # one branch per test
    signal_series: Any, outcome_series: Any, base_params: dict[str, Any]
) -> pd.DataFrame:
    """Run the 7-test robustness battery; returns a results DataFrame.

    ``base_params`` must contain the frozen baseline parameters:

    * ``threshold`` : float - baseline entry threshold (signal >
      threshold)
    * ``window``    : int   - baseline lookback, also used as the placebo
      lag
    optional: ``n_boot`` (1000), ``block`` (20), ``ci_level`` (0.95),
    ``seed`` (0).

    Each row: ``{test, stat, t_stat, ci_low, ci_high, verdict,
    catastrophic, note}``.  ``stat`` is the mean daily strategy return.
    The final row ``OVERALL`` carries the ROBUST / FRAGILE / DEAD
    verdict in ``verdict``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> s = rng.standard_normal(500)  # doctest: +SKIP
        >>> y = 0.5 * s + rng.standard_normal(500)  # doctest: +SKIP
        >>> table = run_battery(  # doctest: +SKIP
        ...     s, y, {"threshold": 0.0, "window": 10})
        >>> battery_overall(table) in  # doctest: +SKIP
        ...     {"ROBUST", "FRAGILE", "DEAD"}
        True
    """
    for key in _REQUIRED:
        if key not in base_params:
            raise ValueError(f"base_params missing required key: {key!r}")
    threshold = float(base_params["threshold"])
    window = int(base_params["window"])
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    if window < 1:
        raise ValueError("window must be >= 1")
    n_boot = int(base_params.get("n_boot", 1000))
    block = int(base_params.get("block", 20))
    ci_level = float(base_params.get("ci_level", 0.95))
    seed0 = int(base_params.get("seed", 0))

    sig, outcome = _align(signal_series, outcome_series)
    n_obs = int(sig.size)
    if n_obs < 60:
        raise ValueError(
            f"need at least 60 aligned observations, got {n_obs}"
        )

    rows: list[dict[str, Any]] = []

    def add(
        test: str,
        eff: dict[str, Any],
        verdict: str,
        catastrophic: bool,
        note: str,
    ) -> None:
        rows.append(
            {
                "test": test,
                "stat": eff["mu"],
                "t_stat": eff["t"],
                "ci_low": eff["ci_low"],
                "ci_high": eff["ci_high"],
                "verdict": verdict,
                "catastrophic": bool(catastrophic),
                "note": note,
            }
        )

    base = _effect(
        sig,
        outcome,
        threshold=threshold,
        n_boot=n_boot,
        block=block,
        ci_level=ci_level,
        seed=seed0,
    )
    if not np.isfinite(base["mu"]):
        raise ValueError("baseline strategy return is not computable")
    bs = np.sign(base["mu"]) if np.sign(base["mu"]) != 0 else 1.0
    add(
        "baseline",
        base,
        "PASS" if _excl0(base) else "FAIL",
        False,
        "long/flat on signal > threshold, 1-period lag",
    )

    # --- 1. threshold sensitivity -------------------------------------
    sign_ok, excl_cnt, cat, parts = True, 0, False, []
    for mult in (0.5, 1.0, 1.5):
        eff = (
            base
            if mult == 1.0
            else _effect(
                sig,
                outcome,
                threshold=threshold * mult,
                n_boot=n_boot,
                block=block,
                ci_level=ci_level,
                seed=seed0 + int(mult * 100),
            )
        )
        parts.append(f"x{mult:g}: mu={eff['mu']:+.5f}")
        if np.sign(eff["mu"]) != bs:
            sign_ok = False
        if _excl0(eff):
            excl_cnt += 1
            if _ci_sign(eff) == -bs:
                cat = True
    add(
        "threshold_sensitivity",
        base,
        "PASS" if (sign_ok and excl_cnt >= 2) else "FAIL",
        cat,
        "; ".join(parts),
    )

    # --- 2. window sensitivity ----------------------------------------
    windows = sorted(
        {max(1, round(window * mult)) for mult in (0.5, 1.0, 1.5)}
    )
    sign_ok, excl_cnt, cat, parts = True, 0, False, []
    for win_len in windows:
        sig_smooth = _roll_mean(sig, win_len)
        eff = _effect(
            sig_smooth,
            outcome,
            threshold=threshold,
            n_boot=n_boot,
            block=block,
            ci_level=ci_level,
            seed=seed0 + win_len,
        )
        parts.append(f"w={win_len}: mu={eff['mu']:+.5f}")
        if np.sign(eff["mu"]) != bs:
            sign_ok = False
        if _excl0(eff):
            excl_cnt += 1
            if _ci_sign(eff) == -bs:
                cat = True
    add(
        "window_sensitivity",
        base,
        "PASS" if (sign_ok and excl_cnt >= 2) else "FAIL",
        cat,
        "; ".join(parts),
    )

    # --- 3. subperiod thirds ------------------------------------------
    splits = np.array_split(np.arange(n_obs), 3)
    sign_ok, excl_cnt, cat, parts = True, 0, False, []
    for third, chunk in enumerate(splits, start=1):
        eff = _effect(
            sig[chunk],
            outcome[chunk],
            threshold=threshold,
            n_boot=n_boot,
            block=block,
            ci_level=ci_level,
            seed=seed0 + third * 7,
        )
        parts.append(f"third{third}: mu={eff['mu']:+.5f}")
        if np.sign(eff["mu"]) != bs:
            sign_ok = False
        if _excl0(eff):
            excl_cnt += 1
            if _ci_sign(eff) == -bs:
                cat = True
    add(
        "subperiod_thirds",
        base,
        "PASS" if (sign_ok and excl_cnt >= 2) else "FAIL",
        cat,
        "; ".join(parts),
    )

    # --- 4. leave-one-out ---------------------------------------------
    active = np.where(base["pos"] > 0)[0]
    base_se = (
        abs(base["mu"] / base["t"])
        if np.isfinite(base["t"]) and base["t"] != 0
        else float("nan")
    )
    if active.size < 10:
        add(
            "leave_one_out",
            base,
            "FAIL",
            False,
            f"only {active.size} active days",
        )
    else:
        stride = int(np.ceil(active.size / 200.0))
        idx = active[::stride]
        total = float(np.sum(base["r"]))
        mus = (total - base["r"][idx]) / (n_obs - 1)
        flips = np.sign(mus) == -bs
        cat = bool(
            base_se > 0 and np.any(flips & (np.abs(mus) / base_se > 2.0))
        )
        worst = float(np.min(np.abs(mus)))
        add(
            "leave_one_out",
            {
                "mu": worst,
                "t": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            },
            "PASS" if not np.any(flips) else "FAIL",
            cat,
            f"{idx.size} leave-one-out draws (stride {stride}),"
            f" min |mu|={worst:.5f}",
        )

    # --- 5. placebo lag -----------------------------------------------
    sp = np.full(n_obs, np.nan)
    if window < n_obs:
        sp[window:] = sig[: n_obs - window]
    eff_p = _effect(
        sp,
        outcome,
        threshold=threshold,
        n_boot=n_boot,
        block=block,
        ci_level=ci_level,
        seed=seed0 + 999,
    )
    excl = _excl0(eff_p)
    cat = bool(
        excl and _ci_sign(eff_p) == bs and abs(eff_p["mu"]) >= abs(base["mu"])
    )
    verdict = "FAIL" if excl else "PASS"
    note = f"signal lagged by {window}; placebo mu={eff_p['mu']:+.5f} " + (
        "CI excludes 0" if excl else "CI includes 0"
    )
    if not np.isfinite(eff_p["ci_low"]):
        verdict, note = "PASS", note + " (degenerate CI)"
    add(
        "placebo_lag",
        {
            "mu": eff_p["mu"],
            "t": eff_p["t"],
            "ci_low": eff_p["ci_low"],
            "ci_high": eff_p["ci_high"],
        },
        verdict,
        cat,
        note,
    )

    # --- 6. declustering ----------------------------------------------
    first = base["pos"] > 0
    keep = first & np.concatenate([[True], ~first[:-1]])
    n_kept = int(keep.sum())
    if n_kept < 5:
        add(
            "declustering",
            base,
            "FAIL",
            False,
            f"only {n_kept} cluster-start days",
        )
    else:
        r_d = np.zeros_like(outcome)
        r_d[1:] = keep[:-1].astype(float) * outcome[1:]
        mu_d = float(np.mean(r_d))
        lo, hi, se_d = _boot_ci_mean(r_d, n_boot, block, ci_level, seed0 + 55)
        eff_d = {"mu": mu_d, "ci_low": lo, "ci_high": hi}
        excl = _excl0(eff_d)
        cat = bool(excl and _ci_sign(eff_d) == -bs)
        add(
            "declustering",
            {
                "mu": mu_d,
                "t": mu_d / se_d if se_d > 0 else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
            },
            "PASS" if (np.sign(mu_d) == bs and excl) else "FAIL",
            cat,
            f"{n_kept} cluster-start days kept of {int(first.sum())} active",
        )

    # --- 7. sign monotonicity -----------------------------------------
    mask = (
        np.isfinite(sig[:-1])
        & np.isfinite(outcome[1:])
        & (sig[:-1] > threshold)
    )
    xs, ys = sig[:-1][mask], outcome[1:][mask]
    used = "active days"
    if xs.size < 30:
        mask_all = np.isfinite(sig[:-1]) & np.isfinite(outcome[1:])
        xs, ys = sig[:-1][mask_all], outcome[1:][mask_all]
        used = "all days (few active)"
    if xs.size < 9:
        add(
            "sign_monotonicity",
            {
                "mu": float("nan"),
                "t": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            },
            "FAIL",
            False,
            f"too few pairs ({xs.size})",
        )
    else:
        ranks = stats.rankdata(xs) / xs.size
        bucket = np.clip(np.ceil(ranks * 3.0), 1, 3).astype(int)
        means = []
        for bucket_id in (1, 2, 3):
            sel = ys[bucket == bucket_id]
            means.append(float(np.mean(sel)) if sel.size else float("nan"))
        if any(not np.isfinite(mean_val) for mean_val in means):
            add(
                "sign_monotonicity",
                {
                    "mu": float("nan"),
                    "t": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
                "FAIL",
                False,
                "empty signal tercile",
            )
        else:
            rho = float(stats.spearmanr([1, 2, 3], means).statistic)
            # catastrophic requires a SIGNIFICANT inverse relation on the
            # raw pairs - a 3-point bucket Spearman flips sign ~1/6 of
            # the time under noise and must never declare DEAD on its own
            day_rho = float(stats.spearmanr(xs, ys).statistic)
            day_p = float(stats.spearmanr(xs, ys).pvalue)
            cat = bool(
                np.isfinite(day_rho)
                and day_rho <= -0.15
                and day_p < 0.01
                and xs.size >= 100
            )
            add(
                "sign_monotonicity",
                {
                    "mu": rho,
                    "t": day_rho,
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
                "PASS" if (np.isfinite(rho) and rho >= 0.5) else "FAIL",
                cat,
                f"tercile means=[{means[0]:+.4f}, {means[1]:+.4f}, "
                f"{means[2]:+.4f}] spearman={rho:+.2f} ({used}); "
                f"per-pair rho={day_rho:+.3f} p={day_p:.3g}",
            )

    # --- overall ------------------------------------------------------
    test_rows = [row for row in rows if row["test"] != "baseline"]
    n_pass = int(sum(row["verdict"] == "PASS" for row in test_rows))
    cat_any = bool(any(row["catastrophic"] for row in test_rows))
    verdict = (
        "DEAD"
        if cat_any
        else ("ROBUST" if n_pass >= _MIN_PASS_ROBUST else "FRAGILE")
    )
    rows.append(
        {
            "test": "OVERALL",
            "stat": float(n_pass),
            "t_stat": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "verdict": verdict,
            "catastrophic": cat_any,
            "note": (
                f"{n_pass}/7 tests passed; catastrophic={cat_any}; "
                f"rules: DEAD if any catastrophic, "
                f"ROBUST if >= {_MIN_PASS_ROBUST}/7"
            ),
        }
    )
    return pd.DataFrame(rows)


def battery_overall(table: pd.DataFrame) -> str:
    """Extract the OVERALL verdict (ROBUST / FRAGILE / DEAD) from the table.

    Example::

        >>> battery_overall(run_battery(s, y, params))     # doctest: +SKIP
        'ROBUST'
    """
    overall = table.loc[table["test"] == "OVERALL", "verdict"]
    if overall.empty:
        raise ValueError(
            "no OVERALL row found; pass the DataFrame from run_battery"
        )
    return str(overall.iloc[0])


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo the battery on synthetic data with a planted, persistent edge."""
    print("=" * 72)
    print("robustness_battery demo: planted AR(1) edge")
    print("=" * 72)
    rng = np.random.default_rng(11)
    n_days = 750
    sig = np.zeros(n_days)
    for day in range(1, n_days):
        sig[day] = 0.5 * sig[day - 1] + rng.standard_normal()
    sig = sig / float(np.std(sig))
    outcome = 0.8 * sig + rng.standard_normal(n_days)  # outcome on signal
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    table = run_battery(
        pd.Series(sig, index=idx),
        pd.Series(outcome, index=idx),
        {
            "threshold": float(np.median(sig)),
            "window": 20,
            "n_boot": 1000,
            "block": 20,
            "seed": 0,
        },
    )
    with pd.option_context("display.width", 140, "display.max_columns", 12):
        print(
            table.to_string(
                index=False, float_format=lambda val: f"{val:.4f}"
            )
        )
    print("\nOVERALL:", battery_overall(table))


if __name__ == "__main__":
    run_all_examples()
