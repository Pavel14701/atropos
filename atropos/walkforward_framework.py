"""Walk-forward evaluation with anti-peeking enforcement.

The only honest way to measure a rule's out-of-sample t-statistic.
:func:`walkforward` slices the data into folds, fits on the past,
predicts the next slice, and stacks the out-of-sample PnL:

* **Expanding, not rolling** - each fold trains on ALL history before the
  test slice (``data.iloc[:t0]``); the framework offers no rolling mode
  at all.
* **Purge / embargo** - ``purge_gap`` rows directly before each test
  slice are REMOVED from the train slice.  With multi-period forward
  returns the label of row ``t`` covers ``[t, t + horizon)``; without a
  purge the last training labels overlap the test period and leak
  (Lopez de Prado's purging).  ``purge_gap = label horizon`` is the
  correct setting; 0 keeps the historical behaviour.
* **No peeking** - ``fit_fn`` receives the train slice only,
  structurally; the helper :func:`check_no_peeking` re-verifies
  ``train_end + purge_gap <= test_start`` for every recorded fold.
* **Calendar terms forbidden** - any column whose name looks like a
  calendar feature (year / month / dow / date / month-day names) raises
  ``ValueError``, because calendar dummies let a "signal" memorize
  regimes.
* **Threshold frozen on train** - use :func:`frozen_threshold` inside
  a custom ``fit_fn``: the threshold is selected on the train slice and
  stored in the model; the test slice is scored against it untouched
  (the default OLS fit is threshold-free).

Returned dict: ``{wf_t, wf_direction_consistency, wf_shrink_ratio,
verdict, n_folds, in_sample_t, folds, warnings, mode}``.

Usage::

    from walkforward_framework import walkforward

    result = walkforward(data, train_window=180, test_window=30, step=30,
                         fit_fn=my_fit, predict_fn=my_predict)
    print(result["wf_t"], result["verdict"])

Run ``python -m atropos.walkforward_framework`` for a synthetic demo.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["check_no_peeking", "frozen_threshold", "walkforward"]

# Calendar terms are matched as whole tokens (with optional plural), so
# ordinary feature names containing them as substrings - "window" (dow),
# "update" (date) - do not trigger false positives.
_ALWAYS_BAD_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:year|month|week|date|quarter|holiday|day|dow|doy)"
    r"s?(?:[^a-z]|$)"
)
_MONTH_DOW_RE = re.compile(
    r"(?i)(?:^|[^a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"mon|tue|tues|wed|thu|thur|fri|sat|sun)(?:[^a-z]|$)"
)


def _forbidden_columns(columns: Any) -> list[str]:
    """Column names that smell like calendar features."""
    bad: list[str] = []
    for col in columns:
        col_name = str(col)
        if _ALWAYS_BAD_RE.search(col_name) or _MONTH_DOW_RE.search(col_name):
            bad.append(col_name)
    return bad


def frozen_threshold(
    train_df: pd.DataFrame,
    signal_col: str,
    outcome_col: str,
    grid: Any = None,
) -> dict[str, float]:
    """Select an entry threshold on the TRAIN slice; freeze it in the model.

    Scans the 30/40/50/60/70-percentile grid (or a user grid) of the
    signal, picks the threshold with the highest in-train t-stat of the
    lagged long/flat rule, and returns ``{"threshold": float,
    "train_t": float}``.  The threshold must then be reused verbatim on
    every test slice.

    Example::

        >>> model = {"thr":  # doctest: +SKIP
        ...     frozen_threshold(train_df, "signal", "outcome")}
        >>> positions_test = (  # doctest: +SKIP
        ...     test_df["signal"] > model["thr"]["threshold"])
    """
    signal = train_df[signal_col].to_numpy(dtype=float)
    outcome = train_df[outcome_col].to_numpy(dtype=float)
    ok = np.isfinite(signal) & np.isfinite(outcome)
    signal, outcome = signal[ok], outcome[ok]
    if grid is None:
        qs = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        grid = np.quantile(signal, qs)
    thresholds = np.atleast_1d(np.asarray(grid, dtype=float))
    best_thr, best_t = float("nan"), -np.inf
    for thr in thresholds:
        pos = (signal > thr).astype(float)
        pnl = np.zeros_like(outcome)
        pnl[1:] = pos[:-1] * outcome[1:]
        sd = float(np.std(pnl))
        t_stat = (
            float(np.mean(pnl) / sd * np.sqrt(pnl.size))
            if sd > 0
            else -np.inf
        )
        if t_stat > best_t:
            best_thr, best_t = float(thr), t_stat
    return {"threshold": best_thr, "train_t": best_t}


def _default_fit(
    train_df: pd.DataFrame, outcome_col: str = "outcome"
) -> dict[str, Any]:
    """Default fit: OLS on every numeric non-outcome column.

    The default model is a plain linear score consumed by
    :func:`_default_predict`.  The long/flat entry-threshold machinery
    (:func:`frozen_threshold`) belongs to custom ``fit_fn`` /
    ``predict_fn`` pairs, not to the default path - the walk-forward PnL
    contract is ``sign(forecast) * realized``, a long/short rule.
    """
    cols = [
        col
        for col in train_df.columns
        if col != outcome_col and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    if not cols:
        raise ValueError(
            "no numeric feature columns found for default fit_fn"
        )
    features = train_df[cols].to_numpy(dtype=float)
    outcome = train_df[outcome_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(outcome)), features])
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return {"cols": list(cols), "beta": beta}


def _default_predict(
    model: dict[str, Any], test_df: pd.DataFrame
) -> np.ndarray:
    """Default predict: linear model score from the frozen coefficients."""
    features = test_df[model["cols"]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(features)), features])
    scores: np.ndarray = design @ model["beta"]
    return scores


# Walk-forward orchestrator: one branch per pipeline stage; the stages are
# sequential by design, so the complexity limits are waived on walkforward.
def walkforward(  # noqa: PLR0912, PLR0915  # one branch per stage
    data: pd.DataFrame,
    train_window: int = 180,
    test_window: int = 30,
    step: int = 30,
    fit_fn: Callable[[pd.DataFrame], Any] | None = None,
    *,
    predict_fn: Callable[[Any, pd.DataFrame], Any] | None = None,
    outcome_col: str = "outcome",
    forbid_calendar: bool = True,
    purge_gap: int = 0,
) -> dict[str, Any]:
    """Expanding-window walk-forward evaluation with anti-peeking checks.

    Parameters
    ----------
    data : DataFrame with feature column(s) and ``outcome_col`` (forward
        return).
    train_window : initial training length; the training set then EXPANDS
        with every fold (rolling mode is deliberately not implemented).
    test_window, step : test-slice length and fold stride.
    fit_fn : ``fit_fn(train_df) -> model``.  Receives ONLY past data.
        None -> default OLS fit with a frozen entry threshold.
    predict_fn : (keyword-only) ``predict_fn(model, test_df) -> array of
        return forecasts``.  None -> default linear scorer.
    outcome_col : (keyword-only) name of the forward-return column.
    forbid_calendar : (keyword-only) raise on calendar-looking column
        names (recommended).
    purge_gap : (keyword-only) rows removed from the END of each train
        slice so overlapping forward-return labels cannot leak into the
        test period; set it to the label horizon (e.g. 1 for one-period
        forward returns).

    Verdict: PASS if ``wf_t >= 2`` and direction consistency >= 0.5;
    DEGRADED if ``wf_t >= 1``; FAIL otherwise.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> n = 400  # doctest: +SKIP
        >>> s = rng.standard_normal(n)  # doctest: +SKIP
        >>> df = pd.DataFrame(  # doctest: +SKIP
        ...     {"signal": s,
        ...      "outcome": 0.4 * s + rng.standard_normal(n)})
        >>> out = walkforward(  # doctest: +SKIP
        ...     df, train_window=200, test_window=50, step=50)
        >>> out["verdict"] in {"PASS", "DEGRADED", "FAIL"}  # doctest: +SKIP
        True
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if outcome_col not in data.columns:
        raise ValueError(f"outcome_col {outcome_col!r} not in data columns")
    n_rows = len(data)
    train_window, test_window, step = (
        int(train_window),
        int(test_window),
        int(step),
    )
    purge_gap = int(purge_gap)
    if train_window < 10 or test_window < 1 or step < 1:
        raise ValueError(
            "need train_window >= 10, test_window >= 1, step >= 1"
        )
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")
    if train_window + purge_gap + test_window > n_rows:
        raise ValueError(
            f"n={n_rows} too small for train_window={train_window}"
            f" + purge_gap={purge_gap} + test_window={test_window}"
        )

    if forbid_calendar:
        bad = _forbidden_columns(data.columns)
        if bad:
            raise ValueError(
                "calendar terms forbidden in walkforward features: "
                f"{bad}. A signal must not memorize calendar regimes."
            )

    if fit_fn is None:

        def _auto_fit(train_slice: pd.DataFrame) -> dict[str, Any]:
            return _default_fit(train_slice, outcome_col=outcome_col)

        fit_fn = _auto_fit
    if predict_fn is None:
        predict_fn = _default_predict

    fold_rows: list[dict[str, Any]] = []
    pnls: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    real_all: list[np.ndarray] = []

    t0 = train_window + purge_gap
    while t0 + test_window <= n_rows:
        # EXPANDING: all history up to t0, minus the purge window whose
        # forward-return labels overlap the test period
        train = data.iloc[: t0 - purge_gap]
        test = data.iloc[t0 : t0 + test_window]  # strictly after train
        model = fit_fn(train)  # fit_fn NEVER sees test data
        preds = np.asarray(predict_fn(model, test), dtype=float).reshape(-1)
        if preds.size != len(test):
            raise ValueError(
                f"predict_fn returned {preds.size} values"
                f" for {len(test)} rows"
            )
        realized = test[outcome_col].to_numpy(dtype=float)
        pnl = np.sign(preds) * realized
        pnls.append(pnl)
        preds_all.append(preds)
        real_all.append(realized)
        fold_rows.append(
            {
                "fold": len(fold_rows),
                "train_end": int(t0 - purge_gap),
                "test_start": int(t0),
                "test_end": int(t0 + test_window),
                "n_train": len(train),
                "purge_gap": int(purge_gap),
                "hit_rate": float(
                    np.mean(np.sign(preds) == np.sign(realized))
                ),
                "fold_pnl_mean": float(np.mean(pnl)),
            }
        )
        t0 += step

    if not fold_rows:
        raise ValueError("no walk-forward folds produced")

    pnl_all = np.concatenate(pnls)
    preds_stacked = np.concatenate(preds_all)
    real_stacked = np.concatenate(real_all)
    n_pnl = int(pnl_all.size)
    sd = float(np.std(pnl_all, ddof=1))
    wf_t = (
        float(np.mean(pnl_all) / sd * np.sqrt(n_pnl))
        if sd > 0
        else float("nan")
    )
    direction = float(
        np.mean(np.sign(preds_stacked) == np.sign(real_stacked))
    )

    # in-sample reference (diagnostic only - this is the shrink denominator)
    model_all = fit_fn(data)
    preds_is = np.asarray(predict_fn(model_all, data), dtype=float).reshape(
        -1
    )
    pnl_is = np.sign(preds_is) * data[outcome_col].to_numpy(dtype=float)
    sd_is = float(np.std(pnl_is, ddof=1))
    is_t = (
        float(np.mean(pnl_is) / sd_is * np.sqrt(pnl_is.size))
        if sd_is > 0
        else float("nan")
    )
    shrink = (
        float(wf_t / is_t)
        if (np.isfinite(is_t) and is_t != 0)
        else float("nan")
    )

    if np.isfinite(wf_t) and wf_t >= 2.0 and direction >= 0.5:
        verdict = "PASS"
    elif np.isfinite(wf_t) and wf_t >= 1.0:
        verdict = "DEGRADED"
    else:
        verdict = "FAIL"

    warnings: list[str] = []
    if step > test_window:
        warnings.append(
            "step > test_window: folds skip periods between tests"
        )
    if step < test_window:
        warnings.append(
            "step < test_window: test slices overlap - the naive wf_t "
            "counts some periods more than once and overstates "
            "significance"
        )

    folds_df = pd.DataFrame(fold_rows)
    return {
        "wf_t": wf_t,
        "wf_direction_consistency": direction,
        "wf_shrink_ratio": shrink,
        "verdict": verdict,
        "n_folds": int(folds_df.shape[0]),
        "in_sample_t": is_t,
        "folds": folds_df,
        "warnings": warnings,
        "mode": "expanding",
        "purge_gap": int(purge_gap),
    }


def check_no_peeking(result: dict[str, Any]) -> bool:
    """Verify each fold's train slice ends before its test slice starts.

    Honours the recorded ``purge_gap``: the check is
    ``train_end + purge_gap <= test_start`` per fold, so a purged
    walk-forward is verified against its own, stricter boundary.

    Example::

        >>> res = walkforward(df, fit_fn=f, predict_fn=p)   # doctest: +SKIP
        >>> check_no_peeking(res)                           # doctest: +SKIP
        True
    """
    folds = result.get("folds")
    if folds is None or len(folds) == 0:
        return False
    gaps = (
        folds["purge_gap"]
        if "purge_gap" in folds.columns
        else pd.Series(0, index=folds.index)
    )
    return bool(((folds["train_end"] + gaps) <= folds["test_start"]).all())


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: walk-forward on a genuine edge, plus the calendar guard."""
    print("=" * 72)
    print("walkforward_framework demo: genuine edge, expanding window")
    print("=" * 72)
    rng = np.random.default_rng(5)
    n_days = 600
    sig = np.zeros(n_days)
    for day in range(1, n_days):
        sig[day] = 0.4 * sig[day - 1] + rng.standard_normal()
    data = pd.DataFrame(
        {
            "signal": sig,
            "outcome": 0.5 * sig + rng.standard_normal(n_days),
        }
    )
    out = walkforward(data, train_window=180, test_window=30, step=30)
    print(
        f"wf_t={out['wf_t']:.2f}"
        f"  direction={out['wf_direction_consistency']:.3f}"
        f"  shrink={out['wf_shrink_ratio']:.2f}"
        f"  verdict={out['verdict']}"
        f"  n_folds={out['n_folds']}"
    )
    print(
        f"in_sample_t={out['in_sample_t']:.2f}  "
        f"no_peeking={check_no_peeking(out)}  mode={out['mode']}"
    )
    print(out["folds"].head(3).to_string(index=False))

    print()
    print("same walk-forward with a 5-row purge (overlapping-label guard):")
    purged = walkforward(
        data, train_window=180, test_window=30, step=30, purge_gap=5
    )
    print(
        f"wf_t={purged['wf_t']:.2f}  n_folds={purged['n_folds']}"
        f"  no_peeking={check_no_peeking(purged)}"
        f"  first fold train_end={int(purged['folds']['train_end'].iloc[0])}"
        f" < test_start={int(purged['folds']['test_start'].iloc[0])}"
    )

    print()
    print("=" * 72)
    print("walkforward_framework demo: calendar column guard")
    print("=" * 72)
    bad = data.copy()
    bad["year"] = np.arange(n_days) // 252
    try:
        walkforward(bad, train_window=180, test_window=30, step=30)
    except ValueError as exc:
        print("ValueError raised as expected:")
        print(" ", exc)


if __name__ == "__main__":
    run_all_examples()
