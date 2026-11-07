"""Walk-forward evaluation with anti-peeking enforcement.

The only honest way to measure a rule's out-of-sample t-statistic.
:func:`walkforward` slices the data into folds, fits on the past,
predicts the next slice, and stacks the out-of-sample PnL:

* **Expanding, not rolling** - each fold trains on ALL history before the
  test slice (``data.iloc[:t0]``); the framework offers no rolling mode
  at all.  This is the correct shape for a FALSIFICATION stage: the
  question is "does the edge hold out of sample as evidence accumulates",
  and more accumulated history means more held-out evidence and a
  sharper verdict - discarding it (a rolling window) would make the
  test WEAKER exactly when confidence should grow.  A rolling window
  would also smuggle in a free hyperparameter - the window length -
  which is itself tunable and therefore another overfitting surface;
  expanding has no such knob (``train_window`` is only the initial
  seed, every later fold trains on strictly more).
* **Purge / embargo** - ``purge_gap`` rows directly before each test
  slice are REMOVED from the train slice.  With multi-period forward
  returns the label of row ``t`` covers ``[t, t + horizon)``; without a
  purge the last training labels overlap the test period and leak
  (Lopez de Prado's purging).  Set ``purge_gap`` to the label horizon
  for multi-period labels; 0 keeps the historical behaviour.
  ``purge_gap`` (a leakage guard between train and test) is SEPARATE
  from ``label_horizon`` (the non-overlap assumption the ``wf_t``
  statistic makes about the test union); see the parameter docstring.
* **Overfit gate** - the verdict first consults the shrink ratio
  (``wf_t`` over the full-sample in-sample reference ``t``): a finite
  ratio STRICTLY above ``shrink_cap`` (default 2.0) is OVERFIT-DEATH -
  FAIL with a ``verdict_reason`` marked ``OVERFIT_DEATH`` (the
  FRAMEWORK.md stage-6 kill criterion "shrinkage > 2x -> OVERFIT-DEATH");
  a NaN ratio (degenerate baseline) is INCONCLUSIVE, never a silent
  pass.  The ``wf_t`` tiers apply only when the gate did not fire.
* **No peeking** - ``fit_fn`` receives the train slice only,
  structurally, and ``predict_fn`` receives the test frame WITHOUT the
  outcome column, so neither hook can touch realized test returns; the
  helper :func:`check_no_peeking` re-verifies ``train_end + purge_gap
  <= test_start`` for every recorded fold.
* **Overlapping folds de-duplicated** - with ``step < test_window`` the
  test slices overlap; ``wf_t`` and direction consistency are computed
  on the UNION of test rows (each row counted exactly once, with the
  prediction from the first fold that tested it), so folding frequency
  cannot manufacture significance.  ``overlap_factor`` records
  ``test_window / step``: a naive overlapping stack would count every
  row up to that many times and inflate ``wf_t`` by up to
  ``sqrt(overlap_factor)``.
* **Calendar terms forbidden** - any column whose name carries
  year / month / week / date / quarter / holiday / dow / doy (or a
  month / weekday abbreviation like jan or tue) as a separate letter
  subword raises ``ValueError``, because calendar dummies let a
  "signal" memorize regimes.  ``day`` / ``days`` are flagged only as
  the whole name or its first segment, so ``trading_day`` or ``5day``
  pass; see :func:`_forbidden_columns` for the exact rule and naming
  advice.
* **Threshold frozen on train** - use :func:`frozen_threshold` inside
  a custom ``fit_fn``: the threshold is selected on the train slice and
  stored in the model; the test slice is scored against it untouched
  (the default OLS fit is threshold-free).

Returned dict: ``{wf_t, wf_direction_consistency, wf_shrink_ratio,
verdict, verdict_reason, n_folds, in_sample_t, folds, warnings, mode,
purge_gap, n_pnl_unique, overlap_factor, preds_std}``.
``in_sample_t`` is a LOOKAHEAD-DIAGNOSTIC: computed by refitting on
the FULL sample - in-sample by construction; used only as the
``wf_shrink_ratio`` denominator, never for verdicts.
``preds_std`` is the std of the union OOS predictions: the
degeneracy read-out consumed by the ensemble pipeline's stage-4
CONSTANT_PREDICTIONS gate (non-finite or below 1e-12 means the model
emitted (numerically) a single prediction), and NaN in the
fewer-than-two-unique-predictions INCONCLUSIVE return where no
statistic is defined.

Usage::

    from walkforward_framework import walkforward

    result = walkforward(data, train_window=180, test_window=30, step=30,
                         fit_fn=my_fit, predict_fn=my_predict)
    print(result["wf_t"], result["verdict"])

Run ``python -m atropos.walkforward_framework`` for a synthetic demo.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["check_no_peeking", "frozen_threshold", "walkforward"]

# Calendar terms are matched as separate letter subwords (with optional
# plural): the term must be bounded by the string edge or a non-letter on
# both sides, so ordinary feature names carrying them as glued substrings
# - "window" (dow), "update" (date) - do not trigger false positives.
# Digit-letter bounds DO count as edges, so "2024year" is flagged.
# "day" is deliberately EXCLUDED here and handled by _DAY_HEAD_RE below,
# so suffix / digit-glued day names stay usable.
_ALWAYS_BAD_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:year|month|week|date|quarter|holiday|dow|doy)"
    r"s?(?:[^a-z]|$)"
)
# day / days are flagged only when they ARE the column name or start it
# ("day", "days", "day_of_week"); "trading_day" and "5day" pass.
_DAY_HEAD_RE = re.compile(r"(?i)^days?(?:[^a-z]|$)")
_MONTH_DOW_RE = re.compile(
    r"(?i)(?:^|[^a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"mon|tue|tues|wed|thu|thur|fri|sat|sun)(?:[^a-z]|$)"
)


def _forbidden_columns(columns: Any) -> list[str]:
    """Column names that smell like calendar features.

    Flagged is any name containing one of year / month / week / date /
    quarter / holiday / dow / doy (plus optional plural), or a month /
    weekday abbreviation (jan .. dec, mon .. sun), as a separate letter
    subword: bounded on both sides by the string edge or a non-letter
    character.  Digit-letter forms count as bounds, so ``2024year`` or
    ``q1_month`` are flagged, while glued substrings (``window``,
    ``update``) are not.  ``day`` / ``days`` are flagged only as the
    whole name or its FIRST segment - ``day``, ``days``,
    ``day_of_week`` - because suffix or digit-glued uses such as
    ``trading_day`` or ``5day`` are ordinary naming, not calendar
    dummies.  To feed a genuinely calendar feature past the guard,
    pass ``forbid_calendar=False`` deliberately; to keep a benign
    name, prefer a non-calendar synonym (``elapsed`` for ``n_days``)
    over reshuffling word order - the strong terms are flagged in any
    segment position.
    """
    bad: list[str] = []
    for col in columns:
        col_name = str(col)
        if (
            _ALWAYS_BAD_RE.search(col_name)
            or _DAY_HEAD_RE.search(col_name)
            or _MONTH_DOW_RE.search(col_name)
        ):
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
    "train_t": float}``.  The t-stat uses ``np.std(..., ddof=1)``,
    consistent with :func:`walkforward`.  Degenerate grid points (zero
    PnL standard deviation) are skipped; if NO grid point yields
    non-degenerate trades, ``ValueError`` is raised instead of
    silently returning a NaN threshold.  The threshold must then be
    reused verbatim on every test slice.

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
        sd = float(np.std(pnl, ddof=1))
        if not sd > 0:  # degenerate: all-flat / all-one positions
            continue
        t_stat = float(np.mean(pnl) / sd * np.sqrt(pnl.size))
        if t_stat > best_t:
            best_thr, best_t = float(thr), t_stat
    if not np.isfinite(best_t):
        raise ValueError("no threshold gives non-degenerate trades")
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
    if not np.all(np.isfinite(design)):
        # NaN/inf features would surface as an opaque LinAlgError from
        # the SVD instead of a diagnosable failure
        raise ValueError(
            "non-finite values in feature columns: clean or drop "
            "NaN/inf features before calling walkforward"
        )
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return {"cols": list(cols), "beta": beta}


def _default_predict(
    model: dict[str, Any], test_df: pd.DataFrame
) -> np.ndarray:
    """Default predict: linear model score from the frozen coefficients.

    Raises ``ValueError`` when the scores are constant (e.g. every
    feature coefficient is zero): a sign rule on a constant forecast
    carries no information, and failing loudly beats silently trading
    one side of every test slice.
    """
    features = test_df[model["cols"]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(features)), features])
    scores: np.ndarray = design @ model["beta"]
    if float(np.ptp(scores)) == 0.0:
        raise ValueError("model produced constant predictions")
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
    shrink_cap: float = 2.0,
    label_horizon: int = 1,
) -> dict[str, Any]:
    """Expanding-window walk-forward evaluation with anti-peeking checks.

    Parameters
    ----------
    data : DataFrame with feature column(s) and ``outcome_col`` (forward
        return).
    train_window : initial training length; the training set then EXPANDS
        with every fold (rolling mode is deliberately not implemented:
        for falsification, more accumulated history must mean more
        confidence, not a fresh tuning knob - see the module docstring).
    test_window, step : test-slice length and fold stride.  With
        ``step < test_window`` the test slices overlap; the statistics
        (``wf_t``, direction consistency) are then computed on the UNION
        of test rows - each row exactly once, taking the prediction from
        the FIRST fold that tested it - so overlapping cannot inflate
        significance (see ``overlap_factor`` in the result).
    fit_fn : ``fit_fn(train_df) -> model``.  Receives ONLY past data
        (including the outcome column - fitting needs labels).  None ->
        default OLS fit on every numeric non-outcome column, scored
        long/short as ``sign(forecast)``; the default path is
        threshold-free - entry thresholds via :func:`frozen_threshold`
        belong to custom ``fit_fn`` / ``predict_fn`` pairs.
    predict_fn : (keyword-only) ``predict_fn(model, test_df) -> array of
        return forecasts``.  The test frame arrives WITHOUT the outcome
        column, so a predict_fn structurally cannot peek at realized
        returns.  None -> default linear scorer (which raises on
        constant predictions).
    outcome_col : (keyword-only) name of the forward-return column.
    forbid_calendar : (keyword-only) raise on calendar-looking column
        names (recommended); see :func:`_forbidden_columns` for exactly
        what is flagged and how to rename benign features.
    purge_gap : (keyword-only) rows removed from the END of each train
        slice so overlapping forward-return labels cannot leak into the
        test period; set it to the label horizon (e.g. 1 for one-period
        forward returns).  A LEAKAGE GUARD only - never reused as the
        label horizon itself.
    shrink_cap : (keyword-only) strict upper bound for the shrink ratio
        ``wf_t / in_sample_t``.  A finite ratio STRICTLY above it is
        OVERFIT-DEATH: verdict FAIL with ``verdict_reason``
        ``OVERFIT_DEATH: shrinkage N.NNx`` - the FRAMEWORK.md stage-6
        kill criterion ("shrinkage > 2x -> OVERFIT-DEATH": the
        walk-forward result departs from the in-sample reference by
        more than the cap).  The boundary is STRICT (``>``): a ratio
        exactly equal to ``shrink_cap`` does NOT fire and the ``wf_t``
        tiers decide.  Must be finite and > 0; the gate is evaluated
        without consulting ``wf_t`` finiteness.
    label_horizon : (keyword-only) forward-label overlap length the
        ``wf_t`` statistic assumes: labels are treated as
        non-overlapping inside the test union, which is exact only for
        ``label_horizon == 1``.  This parameter DOCUMENTS that
        assumption; it does NOT change ``purge_gap`` semantics (the
        purge stays the leakage guard between train and test).  Values
        > 1 raise ``ValueError`` (overlap correction is not implemented
        - see KNOWN-ISSUES.md section 6); values below 1 are rejected
        as invalid.

    Returns:
    -------
    dict with ``wf_t`` (out-of-sample t-stat over the union of test
    rows), ``wf_direction_consistency``, ``wf_shrink_ratio`` (``wf_t``
    divided by the in-sample t), ``verdict``, ``verdict_reason`` (None
    for tier verdicts; the OVERFIT_DEATH, insufficient-predictions or
    shrinkage-NaN marker otherwise - downstream consumers detect a
    death from this field without recomputing), ``n_folds``,
    ``in_sample_t`` (LOOKAHEAD-DIAGNOSTIC: computed by refitting on
    the FULL sample - in-sample by construction; used only as the
    ``wf_shrink_ratio`` denominator, never for verdicts), ``folds``
    (per-fold frame; ``train_end`` is the
    EXCLUSIVE end of the train slice - the first row NOT in train - and
    the test slice spans ``[test_start, test_end)``), ``warnings``
    (list of strings), ``mode``, ``purge_gap``, ``n_pnl_unique`` (test
    rows counted once), ``overlap_factor`` (``test_window / step``;
    a naive overlapping stack would count every row up to this many
    times and inflate ``wf_t`` by up to ``sqrt(overlap_factor)``) and
    ``preds_std`` (std of the union OOS predictions - the degeneracy
    read-out the ensemble pipeline's stage-4 CONSTANT_PREDICTIONS gate
    consumes; NaN when fewer than two unique predictions exist).

    Verdict, in order: (0) fewer than two unique OOS predictions in
    the union -> INCONCLUSIVE with ``verdict_reason`` "insufficient
    walk-forward predictions (N unique) — the walk-forward is
    degenerate, not the signal": a sign rule over a single prediction
    carries no statistic, so the run is undecided, never a tier;
    (1) overfit gate - a finite shrink ratio
    STRICTLY above ``shrink_cap`` -> FAIL with ``verdict_reason``
    starting ``OVERFIT_DEATH`` (FRAMEWORK.md stage-6 kill criterion
    "shrinkage > 2x -> OVERFIT-DEATH"); (2) a NaN shrink ratio
    (degenerate or zero baseline t) -> INCONCLUSIVE with
    ``verdict_reason`` "OVERFIT check undefined: shrinkage NaN
    (baseline t degenerate)" - undecided, never a silent pass;
    (3) only then the wf_t tiers: PASS if ``wf_t >= 2`` and direction
    consistency >= 0.5; DEGRADED if ``wf_t >= 1``; FAIL otherwise.

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
    shrink_cap = float(shrink_cap)
    if not math.isfinite(shrink_cap) or shrink_cap <= 0.0:
        raise ValueError("shrink_cap must be finite and > 0")
    label_horizon = int(label_horizon)
    if label_horizon < 1:
        raise ValueError("label_horizon must be an int >= 1")
    if label_horizon > 1:
        # overlap inflates wf_t (anti-conservative) and the Newey-West
        # style correction does not exist yet: refuse loudly instead of
        # reporting an uncalibrated statistic
        raise ValueError(
            "label_horizon>1 requires overlap correction; not "
            "implemented - see KNOWN-ISSUES section 6; set "
            "label_horizon=1"
        )
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
    covered: np.ndarray = np.zeros(n_rows, dtype=bool)
    union_preds: list[np.ndarray] = []
    union_real: list[np.ndarray] = []

    t0 = train_window + purge_gap
    while t0 + test_window <= n_rows:
        # EXPANDING: all history up to t0, minus the purge window whose
        # forward-return labels overlap the test period
        train = data.iloc[: t0 - purge_gap]
        test = data.iloc[t0 : t0 + test_window]  # strictly after train
        model = fit_fn(train)  # fit_fn NEVER sees test data
        # predict_fn NEVER sees the outcome column of the test slice
        test_features = test.drop(columns=[outcome_col])
        preds = np.asarray(
            predict_fn(model, test_features), dtype=float
        ).reshape(-1)
        if preds.size != len(test):
            raise ValueError(
                f"predict_fn returned {preds.size} values"
                f" for {len(test)} rows"
            )
        realized = test[outcome_col].to_numpy(dtype=float)
        pnl = np.sign(preds) * realized
        # union bookkeeping: each data row enters the OOS sample once,
        # with the prediction of the FIRST fold that tested it
        positions = np.arange(t0, t0 + test_window)
        fresh = ~covered[positions]
        union_preds.append(preds[fresh])
        union_real.append(realized[fresh])
        covered[positions] = True
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

    wf_warnings: list[str] = []
    if step > test_window:
        wf_warnings.append(
            "step > test_window: folds skip periods between tests"
        )
    if step < test_window:
        wf_warnings.append(
            "step < test_window: test slices overlap - statistics use "
            "the union of test rows (n_pnl_unique), each row counted "
            "once, so the overlap cannot inflate wf_t"
        )

    preds_unique = np.concatenate(union_preds)
    real_unique = np.concatenate(union_real)
    n_pnl_unique = int(preds_unique.size)
    if preds_unique.size < 2:
        # fewer than two distinct OOS predictions cannot give a sign
        # rule any statistic (std undefined / zero): the walk-forward
        # is degenerate - undecided, never a tier (the pipeline's
        # stage-4 CONSTANT_PREDICTIONS gate must not fire here, so
        # this return carries a non-finite preds_std AND an
        # INCONCLUSIVE verdict the orchestrator surfaces unchanged)
        return {
            "wf_t": float("nan"),
            "wf_direction_consistency": float("nan"),
            "wf_shrink_ratio": float("nan"),
            "verdict": "INCONCLUSIVE",
            "verdict_reason": (
                "insufficient walk-forward predictions "
                f"({preds_unique.size} unique) — the walk-forward is "
                "degenerate, not the signal"
            ),
            "n_folds": len(fold_rows),
            "in_sample_t": float("nan"),
            "folds": pd.DataFrame(fold_rows),
            "warnings": wf_warnings,
            "mode": "expanding",
            "purge_gap": int(purge_gap),
            "n_pnl_unique": n_pnl_unique,
            "overlap_factor": float(test_window / step),
            "preds_std": float("nan"),
        }
    pnl_unique = np.sign(preds_unique) * real_unique
    sd_unique = float(np.std(pnl_unique, ddof=1))
    wf_t = (
        float(np.mean(pnl_unique) / sd_unique * np.sqrt(n_pnl_unique))
        if sd_unique > 0
        else float("nan")
    )
    direction = float(np.mean(np.sign(preds_unique) == np.sign(real_unique)))

    # in-sample reference - LOOKAHEAD-DIAGNOSTIC: computed by refitting
    # on the FULL sample (train == test), so it is in-sample by
    # construction and must never feed a verdict; it exists only as
    # the wf_shrink_ratio denominator.
    model_all = fit_fn(data)
    preds_is = np.asarray(
        predict_fn(model_all, data.drop(columns=[outcome_col])),
        dtype=float,
    ).reshape(-1)
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

    verdict_reason: str | None = None
    if math.isfinite(shrink) and shrink > shrink_cap:
        # FRAMEWORK.md stage-6 kill criterion: the walk-forward result
        # departs from the in-sample reference by more than the cap ->
        # OVERFIT-DEATH.  Strictly '>' (a ratio exactly at the cap does
        # not fire) and independent of wf_t finiteness by design.
        verdict = "FAIL"
        verdict_reason = f"OVERFIT_DEATH: shrinkage {shrink:.2f}x"
    elif not math.isfinite(shrink):
        # baseline t degenerate or zero: the overfit check is undefined,
        # so no tier is trustworthy - undecided, never a silent pass
        verdict = "INCONCLUSIVE"
        verdict_reason = (
            "OVERFIT check undefined: shrinkage NaN (baseline t degenerate)"
        )
    elif np.isfinite(wf_t) and wf_t >= 2.0 and direction >= 0.5:
        verdict = "PASS"
    elif np.isfinite(wf_t) and wf_t >= 1.0:
        verdict = "DEGRADED"
    else:
        verdict = "FAIL"

    # degeneracy read-out of the union predictions: guarded (NaN when
    # the union is somehow empty - the >= 2 branch above makes this
    # unreachable, kept defensive); consumed by the ensemble pipeline's
    # stage-4 CONSTANT_PREDICTIONS gate (non-finite or < 1e-12)
    preds_std = (
        float(np.std(preds_unique)) if preds_unique.size > 0 else float("nan")
    )

    folds_df = pd.DataFrame(fold_rows)
    return {
        "wf_t": wf_t,
        "wf_direction_consistency": direction,
        "wf_shrink_ratio": shrink,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "n_folds": int(folds_df.shape[0]),
        # LOOKAHEAD-DIAGNOSTIC: full-sample refit, in-sample by
        # construction; shrink-ratio denominator only, never a verdict.
        "in_sample_t": is_t,
        "folds": folds_df,
        "warnings": wf_warnings,
        "mode": "expanding",
        "purge_gap": int(purge_gap),
        "n_pnl_unique": n_pnl_unique,
        "overlap_factor": float(test_window / step),
        "preds_std": preds_std,
    }


def check_no_peeking(result: dict[str, Any]) -> bool:
    """Verify each fold's train slice ends before its test slice starts.

    Honours the recorded ``purge_gap``: the check is
    ``train_end + purge_gap <= test_start`` per fold, so a purged
    walk-forward is verified against its own, stricter boundary.

    Scope (read before trusting the name): this is a TAMPER DETECTOR
    on the fold frame, not an independent lookahead proof.  For a
    result this module produced, the check re-verifies bookkeeping the
    walk-forward loop itself wrote (it reduces to ``t0 <= t0``); it
    only fires when a caller edits the frame.  Structural
    anti-peeking is enforced by construction elsewhere — ``fit_fn``
    never receives test rows (pinned by
    ``test_predict_fn_gets_test_frame_without_outcome``) and the
    no-lookahead suite pins fold-0 invariance.  A data-level
    re-derivation of the boundaries is tracked in KNOWN-ISSUES §6.

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
    """Demo: genuine edge, purge, overlap dedup, thresholds, guard."""
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
    assert check_no_peeking(out)
    assert np.isfinite(out["wf_t"])
    # genuine-edge case: the seeded verdict is PASS; DEGRADED (wf_t >= 1)
    # is the acceptable floor, FAIL would mean the planted edge vanished
    assert out["verdict"] in {"PASS", "DEGRADED"}
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
    assert check_no_peeking(purged)
    print(
        f"wf_t={purged['wf_t']:.2f}  n_folds={purged['n_folds']}"
        f"  no_peeking={check_no_peeking(purged)}"
        f"  first fold train_end={int(purged['folds']['train_end'].iloc[0])}"
        f" < test_start={int(purged['folds']['test_start'].iloc[0])}"
    )

    print()
    print("overlapping folds (step=5 < test_window=30): union statistics")
    overlapped = walkforward(data, train_window=180, test_window=30, step=5)
    assert check_no_peeking(overlapped)
    assert np.isfinite(overlapped["wf_t"])
    # overlap_factor is exactly test_window / step (30 / 5 here)
    assert abs(overlapped["overlap_factor"] - 30 / 5) < 1e-9
    print(
        f"wf_t={overlapped['wf_t']:.2f}"
        f"  n_pnl_unique={overlapped['n_pnl_unique']}"
        f"  overlap_factor={overlapped['overlap_factor']:.1f}"
        f"  n_folds={overlapped['n_folds']}"
    )
    for line in overlapped["warnings"]:
        print(f"  warning: {line}")

    print()
    print("frozen-threshold entry rule (long/flat), custom fit/predict:")

    def thr_fit(train_df: pd.DataFrame) -> dict[str, float]:
        return frozen_threshold(train_df, "signal", "outcome")

    def thr_predict(
        model: dict[str, float], test_df: pd.DataFrame
    ) -> np.ndarray:
        above = test_df["signal"].to_numpy(dtype=float)
        return (above > model["threshold"]).astype(float)

    head_model = thr_fit(data.iloc[:180])
    assert np.isfinite(head_model["threshold"])
    frozen = walkforward(
        data,
        train_window=180,
        test_window=30,
        step=30,
        fit_fn=thr_fit,
        predict_fn=thr_predict,
    )
    assert check_no_peeking(frozen)
    print(
        f"threshold={head_model['threshold']:.4f}"
        f"  train_t={head_model['train_t']:.2f}"
        f"  wf_t={frozen['wf_t']:.2f}"
        f"  verdict={frozen['verdict']}"
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
