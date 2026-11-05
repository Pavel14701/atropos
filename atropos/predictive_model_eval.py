"""Predictive-model evaluation: leakage-safe CV, walk-forward OOS, stability.

Evaluation framework for any model that maps features -> next-period return
forecasts.  All functions take ``features`` (feature matrix) and ``target``
(outcome) with rows in CHRONOLOGICAL order.

IMPORTANT — positional alignment: every function pairs ``features`` with
``target`` (or ``predictions`` with ``outcomes``) POSITIONALLY.  Pandas
inputs are converted via ``.to_numpy()`` and their indexes are IGNORED —
there is no reindexing and no sorting inside.  Rows must already be
pairwise aligned AND in chronological order before the call.

Functions
---------
``purged_kfold_cv``               contiguous K-fold with purge gaps on both
                                  sides of each test block (avoids leakage
                                  from overlapping label horizons); optional
                                  model scoring.
``walkforward_model_eval``        rolling train/test OOS: out-of-sample IC,
                                  directional hit rate, sign-PnL t-stat and a
                                  PASS / DEGRADED / FAIL verdict.
``feature_importance_stability``  are feature importances stable across
                                  bootstrap refits?
``sign_consistency``              directional accuracy overall and by regime
                                  (time thirds + outcome-magnitude terciles).
``deflated_performance``          Sharpe deflated for ``n_strategies``
                                  trials (Bailey & Lopez de Prado) + drawdown
                                  context.

Model contract: ``model_fn(features_train, target_train)`` returns a
PREDICTOR - any callable mapping ``features_test -> (n_test,)`` forecasts.
The tooling optionally reads ``predictor.feature_importances_`` (per-feature
importances); the bundled :func:`ols_model_fn` (linear),
:func:`knn_model_fn` (distance-weighted k-NN) and :func:`kernel_model_fn`
(Nadaraya-Watson gaussian kernel) satisfy the full contract with zero
dependencies beyond numpy/pandas/scipy.

Run ``python -m atropos.predictive_model_eval`` for a synthetic demo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from atropos._utils import validate_int, validate_number
from atropos.evaluation_metrics import deflated_sharpe

__all__ = [
    "deflated_performance",
    "feature_importance_stability",
    "kernel_model_fn",
    "knn_model_fn",
    "ols_model_fn",
    "purged_kfold_cv",
    "sign_consistency",
    "walkforward_model_eval",
]

Predictor = Callable[[Any], np.ndarray]
ModelFn = Callable[[np.ndarray, np.ndarray], Predictor]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _as_arrays(
    features: Any, target: Any
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert features/target to float arrays, dropping NaN rows.

    IMPORTANT: alignment is POSITIONAL, never index-based.  Pandas inputs
    are converted with ``.to_numpy()`` and their indexes are IGNORED (no
    reindexing, no sorting).  Callers must pass rows already pairwise
    aligned and in CHRONOLOGICAL order — the purged/walk-forward splits
    below are time blocks and silently assume it.
    """
    if isinstance(features, pd.DataFrame):
        names = [str(column) for column in features.columns]
        features_arr = features.to_numpy(dtype=float)
    else:
        features_arr = np.asarray(features, dtype=float)
        if features_arr.ndim == 1:
            features_arr = features_arr.reshape(-1, 1)
        names = [f"f{idx}" for idx in range(features_arr.shape[1])]
    target_arr = (
        target.to_numpy(dtype=float).reshape(-1)
        if isinstance(target, pd.Series)
        else np.asarray(target, dtype=float).reshape(-1)
    )
    if features_arr.shape[0] != target_arr.size:
        raise ValueError(
            f"features has {features_arr.shape[0]} rows but target has "
            f"{target_arr.size} values"
        )
    ok = np.isfinite(target_arr) & np.isfinite(features_arr).all(axis=1)
    return features_arr[ok], target_arr[ok], names


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


# cap / base seed of the deterministic permutation-importance scheme
_IMPORTANCE_CAP = 256
_IMPORTANCE_SEED = 0

# keys probed (in this order) on fitted models for importances; the SAME
# set is used for dict models (entries) and object models (attributes)
_IMPORTANCE_KEYS = (
    "feature_importances_",
    "feature_importances",
    "coef_",
    "coef",
)
# coefficient-style sources: the sign carries no importance meaning, so
# they are mapped through |.|; importance-style keys are returned as-is
_COEFFICIENT_KEYS = frozenset({"coef_", "coef"})


def _as_matrix(features: Any) -> np.ndarray:
    """Coerce ``features`` to a float ``(n_rows, n_features)`` matrix."""
    features_arr = np.asarray(features, dtype=float)
    if features_arr.ndim == 1:
        features_arr = features_arr.reshape(-1, 1)
    return features_arr


def _standardize_fit(
    features_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns with train statistics (zero std -> 1.0).

    Args:
        features_arr: ``(n_rows, n_features)`` raw feature matrix.

    Returns:
        Tuple ``(standardized, center, spread)``: the standardized
        matrix, the per-column mean, and the per-column std with zeros
        replaced by 1.0 so constant columns stay finite.
    """
    center = features_arr.mean(axis=0)
    spread = features_arr.std(axis=0)
    spread = np.where(spread > 0.0, spread, 1.0)
    return (features_arr - center) / spread, center, spread


def _pairwise_dist_sq(
    test_std: np.ndarray, train_std: np.ndarray
) -> np.ndarray:
    """Squared euclidean distances between standardized row sets.

    Args:
        test_std: ``(n_test, n_features)`` standardized query rows.
        train_std: ``(n_train, n_features)`` standardized rows.

    Returns:
        ``(n_test, n_train)`` matrix of squared distances, clipped at
        0 so the gram-matrix expansion never yields tiny negatives.
    """
    test_norms = np.sum(test_std**2, axis=1)[:, None]
    train_norms = np.sum(train_std**2, axis=1)[None, :]
    cross = test_std @ train_std.T
    squared = np.maximum(test_norms + train_norms - 2.0 * cross, 0.0)
    return np.asarray(squared, dtype=float)


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination of ``predicted`` against ``actual``.

    Args:
        actual: ``(n,)`` observed values.
        predicted: ``(n,)`` forecasted values.

    Returns:
        ``1 - SS_res / SS_tot`` (may be negative); 0.0 when ``actual``
        is constant, because the ratio is undefined there.
    """
    total_ss = float(np.sum((actual - float(actual.mean())) ** 2))
    if total_ss <= 1e-12:
        return 0.0
    residual_ss = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual_ss / total_ss


def _permutation_importance(
    predict_fn: Callable[[Any], np.ndarray],
    features_train: np.ndarray,
    target_train: np.ndarray,
) -> np.ndarray:
    """Capped deterministic permutation importance for a predictor.

    The evaluation subsample is the full train set, or 256 rows drawn
    without replacement (fixed seed 0) when the train set is larger.
    Importance of feature ``j`` is the drop in R^2 after shuffling
    column ``j`` (fresh generator seeded ``0 + j``), clipped at 0 and
    normalized to sum 1; all-zero drops yield the uniform vector.  The
    scheme touches no global RNG state, so fits are reproducible.

    CAVEAT - the clip ``max(baseline - permuted, 0)``: a negative
    baseline R^2 (model worse than the mean predictor) clips to zero
    and can hide features whose permutation actually IMPROVES the
    model; interpret importances as RELATIVE rankings within one fit,
    not as absolute evidence that a feature helps.

    Args:
        predict_fn: fitted predictor closure (``features -> forecasts``).
        features_train: ``(n_train, n_features)`` raw training matrix.
        target_train: ``(n_train,)`` training outcomes.

    Returns:
        ``(n_features,)`` importance vector summing to 1.
    """
    n_rows, n_features = features_train.shape
    if n_rows > _IMPORTANCE_CAP:
        subsample = np.random.default_rng(_IMPORTANCE_SEED).choice(
            n_rows, size=_IMPORTANCE_CAP, replace=False
        )
        features_sub = features_train[subsample]
        target_sub = target_train[subsample]
    else:
        features_sub = features_train
        target_sub = target_train
    baseline = _r_squared(
        target_sub,
        np.asarray(predict_fn(features_sub), dtype=float).reshape(-1),
    )
    drops = np.zeros(n_features, dtype=float)
    for feature_idx in range(n_features):
        shuffled = features_sub.copy()
        column_rng = np.random.default_rng(_IMPORTANCE_SEED + feature_idx)
        shuffled[:, feature_idx] = features_sub[
            column_rng.permutation(features_sub.shape[0]), feature_idx
        ]
        permuted = _r_squared(
            target_sub,
            np.asarray(predict_fn(shuffled), dtype=float).reshape(-1),
        )
        drops[feature_idx] = max(baseline - permuted, 0.0)
    total = float(drops.sum())
    if total > 0.0:
        return drops / total
    return np.full(n_features, 1.0 / n_features)


# ----------------------------------------------------------------------------
# purged K-fold CV
# ----------------------------------------------------------------------------
def purged_kfold_cv(
    features: Any,
    target: Any,
    n_splits: int = 5,
    purge_gap: int = 1,
    model_fn: ModelFn | None = None,
) -> dict[str, Any]:
    """Purged, contiguous K-fold CV for overlapping financial samples.

    Folds are contiguous time blocks (no shuffling).  For each fold, training
    rows within ``purge_gap`` of either boundary of the test block are
    removed, so a training label whose horizon overlaps the test window
    cannot leak:

        train indices kept:  j < test_lo - purge_gap  or
                              j > test_hi + purge_gap

    Time-based variant with embargo: ``atropos._utils.purge_embargo_mask``.

    When ``model_fn`` is given, every fold is scored: out-of-fold
    predictions, per-fold information coefficient (Pearson corr of forecast
    vs realized) and directional hit rate.  A fold whose training set is
    too small to fit (``n_train < n_features + 2``) is SKIPPED instead of
    silently scored as NaN: its fold row carries ``skipped_reason`` (a
    short string) while ``fold_ic`` / ``fold_hit`` stay NaN; scored folds
    carry ``skipped_reason = None``.

    IMPORTANT — positional alignment: ``features`` and ``target`` are
    paired POSITIONALLY; pandas indexes are IGNORED (no reindexing, no
    sorting).  Rows must already be aligned and in CHRONOLOGICAL order —
    the contiguous fold blocks above are time blocks.

    Returns ``{n, n_splits, purge_gap, n_features, folds (DataFrame),
    mean_ic, ic_t, ic_p_value, oos_hit_rate, oof_predictions,
    model_used}``.  ``ic_t`` is a t-stat of the per-fold ICs;
    ``ic_p_value`` is its two-sided p-value, ``2 * t.sf(|ic_t|,
    n_splits - 1)``; ``oof_predictions`` is full-length with NaN outside
    test folds.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> feats = rng.standard_normal((500, 2))  # doctest: +SKIP
        >>> targ = (  # doctest: +SKIP
        ...     0.4 * feats[:, 0] + rng.standard_normal(500)
        ... )
        >>> cv = purged_kfold_cv(  # doctest: +SKIP
        ...     feats, targ, model_fn=ols_model_fn
        ... )
        >>> cv["mean_ic"] > 0.2  # doctest: +SKIP
        True
    """
    features_arr, target_arr, names = _as_arrays(features, target)
    n_rows, n_features = features_arr.shape
    ns = validate_int("n_splits", n_splits, minimum=2)
    gap = validate_int("purge_gap", purge_gap, minimum=0)
    if n_rows < 4 * ns:
        raise ValueError(
            f"n={n_rows} too small for {ns} folds (need >= 4*n_splits)"
        )

    bounds = [
        np.asarray(part) for part in np.array_split(np.arange(n_rows), ns)
    ]
    oof = np.full(n_rows, np.nan) if model_fn is not None else None
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, te in enumerate(bounds):
        lo, hi = int(te[0]), int(te[-1])
        tr = np.concatenate(
            [bounds[other] for other in range(ns) if other != fold_idx]
        )
        tr = tr[(tr < lo - gap) | (tr > hi + gap)]
        row: dict[str, Any] = {
            "fold": fold_idx,
            "train_lo": int(tr.min()) if tr.size else -1,
            "train_hi": int(tr.max()) if tr.size else -1,
            "test_lo": lo,
            "test_hi": hi,
            "n_train": int(tr.size),
            "n_test": int(te.size),
        }
        min_train = n_features + 2
        if model_fn is not None and tr.size >= min_train:
            model = model_fn(features_arr[tr], target_arr[tr])
            preds = np.asarray(model(features_arr[te]), dtype=float).reshape(
                -1
            )
            if oof is not None:
                oof[te] = preds
            ic = _pearson(preds, target_arr[te])
            hit = float(np.mean(np.sign(preds) == np.sign(target_arr[te])))
            row["fold_ic"] = ic
            row["fold_hit"] = hit
            row["skipped_reason"] = None
        else:
            row["fold_ic"] = float("nan")
            row["fold_hit"] = float("nan")
            row["skipped_reason"] = (
                None
                if model_fn is None
                else f"n_train={int(tr.size)} < n_features+2={min_train}"
            )
        fold_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    ics = folds["fold_ic"].to_numpy(dtype=float)
    ics_valid = ics[np.isfinite(ics)]
    mean_ic = float(np.mean(ics_valid)) if ics_valid.size else float("nan")
    ic_t = float("nan")
    if ics_valid.size >= 2:
        se = float(np.std(ics_valid, ddof=1))
        ic_t = (
            float(mean_ic / se * np.sqrt(ics_valid.size))
            if se > 0
            else float("nan")
        )
    hits = folds["fold_hit"].to_numpy(dtype=float)
    hits_valid = hits[np.isfinite(hits)]
    return {
        "n": n_rows,
        "n_splits": ns,
        "purge_gap": gap,
        "n_features": n_features,
        "feature_names": names,
        "folds": folds,
        "mean_ic": mean_ic,
        "ic_t": ic_t,
        "ic_p_value": (
            float(2.0 * stats.t.sf(abs(ic_t), ns - 1))
            if np.isfinite(ic_t)
            else float("nan")
        ),
        "oos_hit_rate": float(np.mean(hits_valid))
        if hits_valid.size
        else float("nan"),
        "oof_predictions": oof,
        "model_used": model_fn is not None,
    }


# ----------------------------------------------------------------------------
# rolling walk-forward evaluation
# ----------------------------------------------------------------------------
def walkforward_model_eval(
    features: Any,
    target: Any,
    model_fn: ModelFn,
    *,
    train_window: int = 180,
    test_window: int = 30,
    step: int | None = None,
    purge_gap: int = 0,
) -> dict[str, Any]:
    """Rolling train/test walk-forward: honest out-of-sample model quality.

    IMPORTANT — positional alignment: ``features`` and ``target`` are
    paired POSITIONALLY; pandas indexes are IGNORED (no reindexing, no
    sorting).  Rows must already be aligned and in CHRONOLOGICAL order —
    the rolling windows below are time windows.

    Every fold trains on the FIXED-length window
    ``[t0 - purge_gap - train_window, t0 - purge_gap)`` and tests on
    ``[t0, t0 + test_window)``; the window then slides by ``step``
    (default ``test_window``).  ``purge_gap`` rows directly before each
    test slice are removed from the train slice so overlapping
    forward-return labels cannot leak into the test period (set it to
    the label horizon; 1 for one-period forward returns).  Stacked
    out-of-sample forecasts drive:

    * ``oos_ic``        - Pearson corr of forecasts vs realized outcomes,
    * ``pnl_t``         - t-stat of ``sign(forecast) * realized`` (long/short
      sign rule),
    * ``oos_hit_rate``  - share of folds-rows where sign(forecast) matches.

    Verdict: PASS if ``pnl_t >= 2``; DEGRADED if ``>= 1``; FAIL otherwise.

    Returns ``{n_folds, oos_ic, pnl_t, oos_hit_rate, verdict, folds
    (DataFrame), predictions, realized, train_window, test_window, step,
    purge_gap}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> feats = rng.standard_normal((500, 1))  # doctest: +SKIP
        >>> targ = (  # doctest: +SKIP
        ...     0.5 * feats[:, 0] + rng.standard_normal(500)
        ... )
        >>> wfe = walkforward_model_eval(  # doctest: +SKIP
        ...     feats, targ, ols_model_fn, train_window=150, test_window=50
        ... )
        >>> wfe["verdict"] in {"PASS", "DEGRADED", "FAIL"}  # doctest: +SKIP
        True
    """
    features_arr, target_arr, _ = _as_arrays(features, target)
    n_rows = features_arr.shape[0]
    tw = validate_int("train_window", train_window)
    ew = validate_int("test_window", test_window)
    st = validate_int("step", step) if step is not None else ew
    pg = validate_int("purge_gap", purge_gap, minimum=0)
    if tw < 10 or ew < 1 or st < 1:
        raise ValueError(
            "need train_window >= 10, test_window >= 1, step >= 1"
        )
    if tw + pg + ew > n_rows:
        raise ValueError(
            f"n={n_rows} too small for train_window={tw}"
            f" + purge_gap={pg} + test_window={ew}"
        )

    preds_all: list[np.ndarray] = []
    real_all: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    t0 = tw + pg
    while t0 + ew <= n_rows:
        tr = slice(t0 - pg - tw, t0 - pg)
        te = slice(t0, t0 + ew)
        model = model_fn(features_arr[tr], target_arr[tr])
        preds = np.asarray(model(features_arr[te]), dtype=float).reshape(-1)
        realized = target_arr[te]
        if preds.size != realized.size:
            raise ValueError(
                f"predictor returned {preds.size} values for "
                f"{realized.size} rows"
            )
        pnl = np.sign(preds) * realized
        pnl_sd = float(np.std(pnl, ddof=1))
        fold_rows.append(
            {
                "fold": len(fold_rows),
                "train_lo": tr.start,
                "train_hi": tr.stop - 1,
                "test_lo": te.start,
                "test_hi": te.stop - 1,
                "n_train": tw,
                "n_test": ew,
                "purge_gap": pg,
                "fold_ic": _pearson(preds, realized),
                "fold_hit": float(
                    np.mean(np.sign(preds) == np.sign(realized))
                ),
                "fold_pnl_mean": float(np.mean(pnl)),
                "fold_pnl_t": (
                    float(np.mean(pnl) / pnl_sd * np.sqrt(pnl.size))
                    if pnl_sd > 0
                    else float("nan")
                ),
            }
        )
        preds_all.append(preds)
        real_all.append(realized)
        t0 += st

    if not fold_rows:
        raise ValueError("no walk-forward folds produced")
    preds = np.concatenate(preds_all)
    realized = np.concatenate(real_all)
    pnl = np.sign(preds) * realized
    sd = float(np.std(pnl, ddof=1))
    pnl_t = (
        float(np.mean(pnl) / sd * np.sqrt(pnl.size))
        if sd > 0
        else float("nan")
    )
    hit = float(np.mean(np.sign(preds) == np.sign(realized)))
    if np.isfinite(pnl_t) and pnl_t >= 2.0:
        verdict = "PASS"
    elif np.isfinite(pnl_t) and pnl_t >= 1.0:
        verdict = "DEGRADED"
    else:
        verdict = "FAIL"
    return {
        "n_folds": len(fold_rows),
        "train_window": tw,
        "test_window": ew,
        "step": st,
        "purge_gap": pg,
        "oos_ic": _pearson(preds, realized),
        "pnl_t": pnl_t,
        "oos_hit_rate": hit,
        "verdict": verdict,
        "folds": pd.DataFrame(fold_rows),
        "predictions": preds,
        "realized": realized,
    }


# ----------------------------------------------------------------------------
# feature importance stability
# ----------------------------------------------------------------------------
def _extract_importances(
    model: Any,
    n_features: int,
    importance_fn: Callable[[Any], Any] | None,
) -> np.ndarray:
    """Pull a length-``n_features`` importance vector from a fitted model.

    Dict models (entries) and object models (attributes) are probed with
    the SAME ordered key set ``("feature_importances_",
    "feature_importances", "coef_", "coef")`` and the SAME abs handling:
    coefficient-style keys (``coef_`` / ``coef``) are mapped through
    ``|.|`` because their sign carries no importance meaning;
    importance-style keys are returned as-is (downstream consumers such
    as :func:`feature_importance_stability` apply ``|.|`` themselves
    where needed).

    Args:
        model: fitted model (dict or arbitrary object).
        n_features: expected length of the importance vector.
        importance_fn: explicit extractor; when given it fully replaces
            the default key probing.

    Returns:
        ``(n_features,)`` importance vector.

    Raises:
        ValueError: if no probed key yields a length-``n_features``
            vector and no ``importance_fn`` was supplied.
    """
    if importance_fn is not None:
        values = np.asarray(importance_fn(model), dtype=float).reshape(-1)
        if values.size != n_features:
            raise ValueError(
                f"importance_fn returned {values.size} values, "
                f"expected {n_features}"
            )
        return values
    for key in _IMPORTANCE_KEYS:
        if isinstance(model, dict):
            raw = model.get(key)
        else:
            raw = getattr(model, key, None)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size == n_features:
            if key in _COEFFICIENT_KEYS:
                # asarray pins the ndarray type (stubs -> Any).
                return np.asarray(np.abs(values))
            return values
    raise ValueError(
        "cannot extract feature importances from the fitted model; "
        "pass importance_fn(model) -> (n_features,) array"
    )


def feature_importance_stability(
    features: Any,
    target: Any,
    model_fn: ModelFn,
    *,
    n_bootstrap: int = 100,
    importance_fn: Callable[[Any], Any] | None = None,
    top_k: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Are feature importances stable across bootstrap refits?

    IMPORTANT — positional alignment: ``features`` and ``target`` are
    paired POSITIONALLY; pandas indexes are IGNORED (no reindexing, no
    sorting).  Rows must already be aligned and in CHRONOLOGICAL order
    (each bootstrap draw then resamples rows iid).

    Each of ``n_bootstrap`` draws resamples rows (iid bootstrap), refits
    the model and extracts importances (default extractors:
    ``model.feature_importances_`` then ``|model.coef_|``; or an explicit
    ``importance_fn(model)``).  Vectors are normalized to sum 1 per draw,
    then summarized:

    * per-feature mean / std / CV of importance, frequency in the top
      ``top_k`` ranks, mean descending rank;
    * ``mean_pairwise_spearman`` - mean rank correlation between every pair
      of bootstrap importance vectors.

    Verdict: STABLE when at least 10 valid draws exist and the mean pairwise
    Spearman is >= 0.6; UNSTABLE otherwise (an UNSTABLE ranking means the
    "story" behind the model is not identifiable from this data).

    Returns ``{per_feature (DataFrame), mean_pairwise_spearman, verdict,
    n_bootstrap, n_valid, n, n_features, top_k}``.

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> feats = pd.DataFrame(  # doctest: +SKIP
        ...     {"a": rng.standard_normal(400), "b": rng.standard_normal(400)}
        ... )
        >>> targ = (  # doctest: +SKIP
        ...     0.6 * feats["a"] + rng.standard_normal(400)
        ... )
        >>> stab = feature_importance_stability(  # doctest: +SKIP
        ...     feats, targ, ols_model_fn, n_bootstrap=30, seed=0
        ... )
        >>> stab["verdict"]  # doctest: +SKIP
        'STABLE'
    """
    features_arr, target_arr, names = _as_arrays(features, target)
    n_rows, n_features = features_arr.shape
    nb = validate_int("n_bootstrap", n_bootstrap, minimum=2)
    tk = int(top_k)
    if not (1 <= tk <= n_features):
        raise ValueError(
            f"top_k must be in [1, n_features={n_features}], got {tk}"
        )
    tk = validate_int("top_k", top_k)
    rng = np.random.default_rng(seed)

    imps: list[np.ndarray] = []
    for _ in range(nb):
        idx = rng.integers(0, n_rows, size=n_rows)
        model = model_fn(features_arr[idx], target_arr[idx])
        imp = _extract_importances(model, n_features, importance_fn)
        imp = np.abs(imp)
        total = float(imp.sum())
        if np.isfinite(imp).all() and total > 0:
            imps.append(imp / total)
    if len(imps) < 2:
        return {
            "per_feature": pd.DataFrame(),
            "mean_pairwise_spearman": float("nan"),
            "verdict": "UNSTABLE",
            "n_bootstrap": nb,
            "n_valid": len(imps),
            "n": n_rows,
            "n_features": n_features,
            "top_k": tk,
            "note": "fewer than 2 valid bootstrap draws",
        }

    imps_arr = np.asarray(imps, dtype=float)
    mean = imps_arr.mean(axis=0)
    std = imps_arr.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.inf)
    desc_ranks = np.argsort(-imps_arr, axis=1)  # descending
    top_freq = np.array(
        [
            float(np.mean(desc_ranks[:, :tk] == feat_idx))
            for feat_idx in range(n_features)
        ]
    )
    mean_rank = np.array(
        [
            float(np.mean(np.where(desc_ranks == feat_idx)[1]) + 1.0)
            for feat_idx in range(n_features)
        ]
    )

    ranks = np.apply_along_axis(
        stats.rankdata, 1, imps_arr
    )  # avg ranks, ascending
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    cov = rc @ rc.T / float(n_features)
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(np.outer(sd, sd) > 0, cov / np.outer(sd, sd), np.nan)
    iu = np.triu_indices(len(imps), k=1)
    rhos = corr[iu]
    rhos = rhos[np.isfinite(rhos)]
    mean_rho = float(np.mean(rhos)) if rhos.size else float("nan")

    verdict = (
        "STABLE"
        if (len(imps) >= 10 and np.isfinite(mean_rho) and mean_rho >= 0.6)
        else "UNSTABLE"
    )
    per_feature = pd.DataFrame(
        {
            "feature": names,
            "mean_importance": mean,
            "std": std,
            "cv": cv,
            f"top{tk}_freq": top_freq,
            "mean_rank": mean_rank,
        }
    )
    return {
        "per_feature": per_feature,
        "mean_pairwise_spearman": mean_rho,
        "verdict": verdict,
        "n_bootstrap": nb,
        "n_valid": len(imps),
        "n": n_rows,
        "n_features": n_features,
        "top_k": tk,
    }


# ----------------------------------------------------------------------------
# sign consistency
# ----------------------------------------------------------------------------
def sign_consistency(
    predictions: Any,
    outcomes: Any,
    n_regimes: int = 3,
) -> dict[str, Any]:
    """Does the model get the SIGN right, everywhere (not just on average)?

    ``directional_accuracy`` is the overall share of rows where
    ``sign(prediction) == sign(outcome)``.  ``sign_stability_by_regime``
    reports the same statistic within (a) ``n_regimes`` contiguous time
    thirds and (b) terciles of ``|outcome|`` (calm / normal / violent rows).
    A model that only works in one regime is a regime bet, not an edge.

    IMPORTANT — positional alignment: ``predictions`` and ``outcomes``
    are paired POSITIONALLY; pandas indexes are IGNORED (no reindexing,
    no sorting).  Rows must already be aligned and in CHRONOLOGICAL
    order.

    Degenerate inputs are rejected explicitly — a constant predictor is
    a beta bet, not a model, and a constant outcome has no sign
    variation to explain:

    * constant predictions (``ptp(predictions) == 0``) -> verdict
      ``FAIL`` with ``reason="degenerate (constant) predictions"``;
    * constant outcomes (``ptp(outcomes) == 0``) -> verdict
      ``INCONCLUSIVE`` with ``reason="degenerate (constant) outcomes"``.

    Exact zeros: ``sign(0) == 0`` carries no directional information, so
    rows where EITHER side is exactly 0 are EXCLUDED from every accuracy
    denominator (overall and per-regime); ``n_usable`` reports how many
    sign-bearing rows remain.  A regime with no usable rows is omitted
    from ``sign_stability_by_regime``.

    Verdict: PASS when overall accuracy > 0.5 AND every regime accuracy
    > 0.5; FAIL otherwise (and always for degenerate predictions);
    INCONCLUSIVE for degenerate outcomes.

    Returns ``{directional_accuracy, sign_stability_by_regime (dict),
    stable, verdict, n, n_usable}`` (plus ``reason`` in the degenerate
    cases and ``note`` when fewer than 10 finite rows remain).

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> preds = rng.standard_normal(600)  # doctest: +SKIP
        >>> outcs = 0.5 * preds + rng.standard_normal(600)  # doctest: +SKIP
        >>> sign_consistency(preds, outcs)["verdict"]  # doctest: +SKIP
        'PASS'
    """
    pred_vals = np.asarray(predictions, dtype=float).reshape(-1)
    outc_vals = np.asarray(outcomes, dtype=float).reshape(-1)
    if pred_vals.size != outc_vals.size:
        raise ValueError(
            f"predictions has {pred_vals.size} values, "
            f"outcomes has {outc_vals.size}"
        )
    ok = np.isfinite(pred_vals) & np.isfinite(outc_vals)
    pred_vals, outc_vals = pred_vals[ok], outc_vals[ok]
    n_rows = int(pred_vals.size)
    if n_rows < 10:
        return {
            "directional_accuracy": float("nan"),
            "sign_stability_by_regime": {},
            "stable": False,
            "verdict": "FAIL",
            "n": n_rows,
            "n_usable": int(
                np.count_nonzero((pred_vals != 0.0) & (outc_vals != 0.0))
            ),
            "note": "insufficient data",
        }
    correct = np.sign(pred_vals) == np.sign(outc_vals)
    if float(np.ptp(pred_vals)) == 0.0:
        # "always +1" (or any constant) is a beta bet, not a model: it
        # must not PASS just because the outcome mean is positive.
        return {
            "directional_accuracy": float(np.mean(correct)),
            "sign_stability_by_regime": {},
            "stable": False,
            "verdict": "FAIL",
            "n": n_rows,
            "n_usable": int(np.count_nonzero(outc_vals != 0.0)),
            "reason": "degenerate (constant) predictions",
        }
    if float(np.ptp(outc_vals)) == 0.0:
        # constant outcome: sign accuracy is unidentifiable here
        return {
            "directional_accuracy": float(np.mean(correct)),
            "sign_stability_by_regime": {},
            "stable": False,
            "verdict": "INCONCLUSIVE",
            "n": n_rows,
            "n_usable": int(np.count_nonzero(pred_vals != 0.0)),
            "reason": "degenerate (constant) outcomes",
        }
    usable = (pred_vals != 0.0) & (outc_vals != 0.0)
    n_usable = int(np.count_nonzero(usable))
    acc = float(np.mean(correct[usable])) if n_usable else float("nan")

    regime_acc: dict[str, float] = {}
    nr = max(2, int(n_regimes))
    for regime_idx, chunk in enumerate(
        np.array_split(np.arange(n_rows), nr), start=1
    ):
        chunk_usable = usable[chunk]
        regime_acc[f"time_third_{regime_idx}"] = (
            float(np.mean(correct[chunk][chunk_usable]))
            if chunk_usable.any()
            else float("nan")
        )
    mag_bins = np.clip(
        np.digitize(
            np.abs(outc_vals),
            np.quantile(np.abs(outc_vals), [1.0 / 3.0, 2.0 / 3.0]),
        ),
        0,
        2,
    )
    for name, bin_idx in zip(
        ("vol_tercile_low", "vol_tercile_mid", "vol_tercile_high"),
        (0, 1, 2),
        strict=True,  # 3 regime names for 3 tercile bins, by construction
    ):
        sel = (mag_bins == bin_idx) & usable
        if sel.any():
            regime_acc[name] = float(np.mean(correct[sel]))

    stable = (
        bool(all(entry > 0.5 for entry in regime_acc.values()))
        and len(regime_acc) > 0
    )
    verdict = "PASS" if (acc > 0.5 and stable) else "FAIL"
    return {
        "directional_accuracy": acc,
        "sign_stability_by_regime": regime_acc,
        "stable": stable,
        "verdict": verdict,
        "n": n_rows,
        "n_usable": n_usable,
    }


# ----------------------------------------------------------------------------
# deflated performance
# ----------------------------------------------------------------------------
def deflated_performance(
    returns: Any,
    n_strategies: int,
    mdd: float,
    periods: int = 252,
    sr_var_annual: Any = 1.0,
) -> dict[str, Any]:
    """Sharpe deflated for multiple testing, with drawdown context.

    Delegates the statistics to ``evaluation_metrics.deflated_sharpe``
    (Bailey & Lopez de Prado, 2014): the observed Sharpe is compared against
    the expected MAXIMUM Sharpe of ``n_strategies`` unit-variance trials, so
    a "best of N" selection cannot masquerade as an edge.  ``n_strategies
    = 1`` disables the multiple-testing penalty (DSR == PSR): with a single
    trial there is no "best of N" selection to punish.  ``mdd`` (the
    strategy's max drawdown, negative) turns the excess Sharpe into a
    drawdown-scaled ``deflated_calmar`` diagnostic.

    Verdict: PASS when the deflated-Sharpe p-value < 0.05 (the edge survives
    having tried ``n_strategies`` strategies); FAIL otherwise.

    Args:
        returns: per-period return series of the selected strategy.
        n_strategies: number of trials the strategy was selected from;
            ``1`` disables the multiple-testing penalty (DSR == PSR).
        mdd: max drawdown of the strategy, NEGATIVE by convention
            (``mdd == 0`` means "no drawdown observed" and yields
            ``deflated_calmar = inf`` for a positive excess Sharpe).
        periods: periods per year (252 for daily data).
        sr_var_annual: variance of Sharpe ratios across trials, in
            annualized units (forwarded to ``deflated_sharpe``).

    Returns:
        ``{dsr, p_value, sr_annual, sr_benchmark_annual, excess_sr, mdd,
        deflated_calmar, n_strategies, verdict}``.

    Raises:
        ValueError: if ``n_strategies`` or ``periods`` is not an
            integer >= 1, or ``mdd > 0`` (a positive drawdown is a
            sign error, not a drawdown).

    Example::

        >>> rng = np.random.default_rng(0)  # doctest: +SKIP
        >>> rets = rng.normal(0.0008, 0.01, 1500)  # doctest: +SKIP
        >>> dp = deflated_performance(  # doctest: +SKIP
        ...     rets, n_strategies=10, mdd=-0.15
        ... )
        >>> dp["verdict"]  # doctest: +SKIP
        'PASS'
    """
    ns = validate_int("n_strategies", n_strategies)
    periods = validate_int("periods", periods)
    mdd_f = validate_number("mdd", mdd)
    if mdd_f > 0.0:
        raise ValueError(
            f"mdd must be <= 0 (max drawdown is negative), got {mdd_f}"
        )
    base = deflated_sharpe(
        returns, n_trials=ns, periods=periods, sr_var_annual=sr_var_annual
    )
    sr_a = float(base["sr_annual"])
    sr0 = float(base["sr_benchmark_annual"])
    excess = (
        sr_a - sr0 if np.isfinite(sr_a) and np.isfinite(sr0) else float("nan")
    )
    if np.isfinite(excess) and np.isfinite(mdd_f) and mdd_f < 0:
        deflated_calmar = float(excess / abs(mdd_f))
    elif np.isfinite(excess) and excess > 0:
        deflated_calmar = float("inf")
    else:
        deflated_calmar = float("nan")
    p_value = float(base["p_value"])
    verdict = "PASS" if (np.isfinite(p_value) and p_value < 0.05) else "FAIL"
    return {
        "dsr": float(base["dsr"]),
        "p_value": p_value,
        "sr_annual": sr_a,
        "sr_benchmark_annual": sr0,
        "excess_sr": excess,
        "mdd": mdd_f,
        "deflated_calmar": deflated_calmar,
        "n_strategies": ns,
        "verdict": verdict,
    }


# ----------------------------------------------------------------------------
# default model factory
# ----------------------------------------------------------------------------
def ols_model_fn(
    features_train: np.ndarray, target_train: np.ndarray
) -> Predictor:
    """Default ``model_fn``: OLS on ``[1 | features]``, returns a predictor.

    The predictor carries ``coef_`` (fitted coefficients incl. intercept)
    and ``feature_importances_`` (|beta_j| * sd(feature_j), normalized to
    sum 1) so the default importance extractor of
    :func:`feature_importance_stability` works without configuration.

    The fit is fail-closed on rank deficiency: with collinear,
    duplicated or constant feature columns the least-squares solution
    is not unique, so the fit raises instead of silently returning a
    minimum-norm coefficient pick.

    Raises:
        ValueError: if the design matrix ``[1 | features]`` is
            rank-deficient (collinear features) - the coefficients are
            not identifiable.

    Example::

        >>> model = ols_model_fn(  # doctest: +SKIP
        ...     features_train, target_train
        ... )
        >>> preds = model(features_test)  # doctest: +SKIP
    """
    features_arr = np.asarray(features_train, dtype=float)
    if features_arr.ndim == 1:
        features_arr = features_arr.reshape(-1, 1)
    target_arr = np.asarray(target_train, dtype=float).reshape(-1)
    design = np.column_stack([np.ones(target_arr.size), features_arr])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "design is rank-deficient (collinear, duplicated or "
            "constant feature columns): OLS coefficients are not "
            "identifiable - drop or de-duplicate the offending columns"
        )
    beta, _, _, _ = np.linalg.lstsq(design, target_arr, rcond=None)
    sd = features_arr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    imp = np.abs(beta[1:] * sd)
    total = float(imp.sum())
    imp_norm = imp / total if total > 0 else imp

    def predict(features_test: Any, _coeffs: np.ndarray = beta) -> np.ndarray:
        test_arr = np.asarray(features_test, dtype=float)
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(-1, 1)
        forecasts = (
            np.column_stack([np.ones(test_arr.shape[0]), test_arr]) @ _coeffs
        )
        return np.asarray(forecasts, dtype=float)

    predict.coef_ = beta  # type: ignore[attr-defined]
    predict.feature_importances_ = imp_norm  # type: ignore[attr-defined]
    return predict


# ----------------------------------------------------------------------------
# native non-parametric model factories (numpy only)
# ----------------------------------------------------------------------------
def knn_model_fn(
    features_train: np.ndarray,
    target_train: np.ndarray,
    n_neighbors: int = 5,
) -> Predictor:
    """Distance-weighted k-NN regressor packaged as a ``model_fn``.

    Features are standardized with the train mean/std (zero std -> 1.0)
    inside the fit.  A forecast is the inverse-distance weighted mean
    of the ``n_neighbors`` nearest training targets::

        prediction = sum(w_i * y_i) / sum(w_i),
        w_i = 1 / (d_i + 1e-12)

    with euclidean distances ``d_i`` measured in standardized space.

    ``feature_importances_`` is permutation importance on a capped
    train subsample (<= 256 rows drawn without replacement when the
    train set is larger, fixed seed 0): the drop in R^2 after shuffling
    one feature column (per-column generator seeded ``0 + column``),
    clipped at 0 and normalized to sum 1 (uniform when every drop is
    0).  Fitting is fully deterministic.

    Args:
        features_train: ``(n_train, n_features)`` feature matrix.
        target_train: ``(n_train,)`` outcome vector.
        n_neighbors: number of nearest training rows averaged per
            forecast.

    Returns:
        Predictor closure with attributes ``n_neighbors_`` (validated
        neighbor count) and ``feature_importances_``.

    Raises:
        ValueError: if ``n_neighbors < 1`` or it exceeds the number of
            training rows.

    Example::

        >>> model = knn_model_fn(  # doctest: +SKIP
        ...     features_train, target_train, n_neighbors=5
        ... )
        >>> preds = model(features_test)  # doctest: +SKIP
    """
    features_arr = _as_matrix(features_train)
    target_arr = np.asarray(target_train, dtype=float).reshape(-1)
    n_rows = features_arr.shape[0]
    neighbors = validate_int("n_neighbors", n_neighbors)
    if n_rows < 1:
        raise ValueError("need at least 1 training row")
    if neighbors > n_rows:
        raise ValueError(
            f"n_neighbors={neighbors} exceeds n_train={n_rows} rows"
        )
    train_std, center, spread = _standardize_fit(features_arr)

    def predict(features_test: Any) -> np.ndarray:
        test_std = (_as_matrix(features_test) - center) / spread
        dist_sq = _pairwise_dist_sq(test_std, train_std)
        neighbor_idx = np.argpartition(dist_sq, neighbors - 1, axis=1)[
            :, :neighbors
        ]
        neighbor_dist = np.sqrt(
            np.take_along_axis(dist_sq, neighbor_idx, axis=1)
        )
        weights = 1.0 / (neighbor_dist + 1e-12)
        weighted = weights * target_arr[neighbor_idx]
        return np.asarray(
            weighted.sum(axis=1) / weights.sum(axis=1), dtype=float
        )

    importances = _permutation_importance(predict, features_arr, target_arr)
    predict.n_neighbors_ = neighbors  # type: ignore[attr-defined]
    predict.feature_importances_ = importances  # type: ignore[attr-defined]
    return predict


def kernel_model_fn(
    features_train: np.ndarray,
    target_train: np.ndarray,
    bandwidth: float | None = None,
) -> Predictor:
    """Nadaraya-Watson gaussian-kernel regressor as a ``model_fn``.

    Features are standardized with the train mean/std (zero std -> 1.0)
    inside the fit.  A forecast is the kernel-weighted mean over ALL
    training rows::

        prediction = sum(K_i * y_i) / sum(K_i),
        K_i = exp(-0.5 * (d_i / bandwidth)**2)

    with euclidean distances ``d_i`` in standardized space; when the
    kernel weight sum is numerically negligible (``denom <= 1e-12``, e.g.
    every ``K_i`` underflows to 0 for a query very far from all training
    rows) the forecast falls back to the single nearest training target.

    The default bandwidth is Scott's rule ``n ** (-1 / (n_features +
    4))`` computed on the standardized train data (scalar).  Importances
    use the same capped deterministic permutation scheme as
    :func:`knn_model_fn`.

    Args:
        features_train: ``(n_train, n_features)`` feature matrix.
        target_train: ``(n_train,)`` outcome vector.
        bandwidth: positive kernel width in standardized units;
            ``None`` selects Scott's rule.

    Returns:
        Predictor closure with attributes ``bandwidth_`` (effective
        width) and ``feature_importances_``.

    Raises:
        ValueError: if an explicit ``bandwidth`` is not > 0.

    Example::

        >>> model = kernel_model_fn(  # doctest: +SKIP
        ...     features_train, target_train, bandwidth=0.5
        ... )
        >>> preds = model(features_test)  # doctest: +SKIP
    """
    features_arr = _as_matrix(features_train)
    target_arr = np.asarray(target_train, dtype=float).reshape(-1)
    n_rows, n_features = features_arr.shape
    if n_rows < 1:
        raise ValueError("need at least 1 training row")
    train_std, center, spread = _standardize_fit(features_arr)
    if bandwidth is None:
        width = float(n_rows ** (-1.0 / (n_features + 4)))
    else:
        width = validate_number("bandwidth", bandwidth)
        if not width > 0.0:
            raise ValueError(f"bandwidth must be > 0, got {width}")

    def predict(features_test: Any) -> np.ndarray:
        test_std = (_as_matrix(features_test) - center) / spread
        dist_sq = _pairwise_dist_sq(test_std, train_std)
        weights = np.exp(-0.5 * dist_sq / (width * width))
        denom = weights.sum(axis=1)
        # denom in (0, 1e-12] is numerically useless (the weights are
        # pure underflow noise): fall back to the nearest training row.
        usable_denom = denom > 1e-12
        smoothed = (weights @ target_arr) / np.where(usable_denom, denom, 1.0)
        nearest = target_arr[np.argmin(dist_sq, axis=1)]
        return np.asarray(
            np.where(usable_denom, smoothed, nearest), dtype=float
        )

    importances = _permutation_importance(predict, features_arr, target_arr)
    predict.bandwidth_ = width  # type: ignore[attr-defined]
    predict.feature_importances_ = importances  # type: ignore[attr-defined]
    return predict


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def _demo_linear_chain(rng: np.random.Generator, n_obs: int) -> None:
    """Demo stage 1: a linear problem through the full eval chain."""
    f1 = rng.standard_normal(n_obs)
    f2 = rng.standard_normal(n_obs)
    f3 = rng.standard_normal(n_obs)
    target = 0.6 * f1 + 0.3 * f2 + rng.standard_normal(n_obs)
    features = pd.DataFrame({"alpha_1": f1, "alpha_2": f2, "noise_3": f3})

    cv = purged_kfold_cv(
        features, target, n_splits=5, purge_gap=1, model_fn=ols_model_fn
    )
    print(
        f"purged_kfold_cv : mean_ic={cv['mean_ic']:.3f} "
        f"ic_t={cv['ic_t']:.2f} "
        f"hit={cv['oos_hit_rate']:.3f} (5 folds, gap=1)"
    )
    print(
        cv["folds"][
            ["fold", "n_train", "n_test", "fold_ic", "fold_hit"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    assert float(cv["mean_ic"]) > 0.5
    assert float(cv["ic_t"]) > 10.0
    assert 0.0 <= float(cv["ic_p_value"]) < 0.001
    assert bool(cv["folds"]["skipped_reason"].isna().all())

    wfe = walkforward_model_eval(
        features,
        target,
        ols_model_fn,
        train_window=200,
        test_window=50,
    )
    print(
        f"\nwalkforward     : n_folds={wfe['n_folds']} "
        f"oos_ic={wfe['oos_ic']:.3f} "
        f"pnl_t={wfe['pnl_t']:.2f} hit={wfe['oos_hit_rate']:.3f} "
        f"verdict={wfe['verdict']}"
    )
    # Soft floor, not an exact pin: the seeded demo config (seed 21,
    # n=800, train_window=200, test_window=50) yields exactly 12 folds;
    # that exact value is regression-pinned in
    # tests/test_predictive_model_eval.py::test_seeded_regression_*.
    assert int(wfe["n_folds"]) >= 10
    assert wfe["verdict"] == "PASS"
    assert float(wfe["pnl_t"]) > 5.0

    stab = feature_importance_stability(
        features, target, ols_model_fn, n_bootstrap=40, top_k=1, seed=1
    )
    print(
        f"\nimportance      : {stab['verdict']} "
        f"(mean pairwise spearman={stab['mean_pairwise_spearman']:.3f}, "
        f"{stab['n_valid']}/{stab['n_bootstrap']} valid draws)"
    )
    print(
        stab["per_feature"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    assert stab["verdict"] == "STABLE"
    # Soft floor, not an exact pin: at seed 1 with n_bootstrap=40 every
    # draw is valid (n_valid == 40); the exact value is regression-
    # pinned in tests/test_predictive_model_eval.py (seeded regression).
    assert int(stab["n_valid"]) >= 30
    assert list(stab["per_feature"]["feature"]) == [
        "alpha_1",
        "alpha_2",
        "noise_3",
    ]

    sc = sign_consistency(wfe["predictions"], wfe["realized"])
    print(
        f"\nsign_consistency: acc={sc['directional_accuracy']:.3f} "
        f"stable={sc['stable']} verdict={sc['verdict']}"
    )
    for name, entry in sc["sign_stability_by_regime"].items():
        print(f"  {name:18s} acc={entry:.3f}")
    assert sc["verdict"] == "PASS"
    assert float(sc["directional_accuracy"]) > 0.6

    pnl = np.sign(wfe["predictions"]) * wfe["realized"]
    dp = deflated_performance(pnl, n_strategies=10, mdd=-0.18)
    print(
        f"\ndeflated_perf   : dsr={dp['dsr']:.3f} p={dp['p_value']:.4f} "
        f"sr={dp['sr_annual']:.2f} "
        f"vs benchmark={dp['sr_benchmark_annual']:.2f} "
        f"defl_calmar={dp['deflated_calmar']:.2f} verdict={dp['verdict']}"
    )
    assert dp["verdict"] == "PASS"
    assert 0.0 <= float(dp["p_value"]) < 0.05


def _demo_nonlinear_shootout(rng: np.random.Generator, n_obs: int) -> None:
    """Demo stage 2: non-linear models on ``y = sin(3*x) + noise``."""
    print("\n" + "=" * 72)
    print("non-linear demo: y = sin(3*x) + noise (one feature)")
    print("=" * 72)
    grid = np.sort(rng.uniform(-1.5, 1.5, n_obs))
    nonlinear_target = np.sin(3.0 * grid) + 0.15 * rng.standard_normal(n_obs)
    nonlinear_features = pd.DataFrame({"wave": grid})
    model_factories: list[tuple[str, ModelFn]] = [
        ("ols", ols_model_fn),
        (
            "knn(n_neighbors=5)",
            lambda feats, targ: knn_model_fn(feats, targ, n_neighbors=5),
        ),
        ("kernel(scott)", kernel_model_fn),
    ]
    shootout: dict[str, float] = {}
    for label, model_factory in model_factories:
        nonlinear_cv = purged_kfold_cv(
            nonlinear_features,
            nonlinear_target,
            n_splits=5,
            purge_gap=1,
            model_fn=model_factory,
        )
        shootout[label] = float(nonlinear_cv["mean_ic"])
        print(f"{label:20s} purged-CV mean_ic={nonlinear_cv['mean_ic']:.3f}")
    assert shootout["knn(n_neighbors=5)"] > shootout["ols"]
    assert shootout["kernel(scott)"] > shootout["ols"]


def run_all_examples() -> None:
    """Demo of the full chain plus a non-linear model shoot-out.

    Every printed outcome is cross-checked with a deterministic assert
    (the data is seeded, so the demo doubles as a smoke test).
    """
    print("=" * 72)
    print(
        "predictive_model_eval demo: 2 informative features + 1 noise feature"
    )
    print("=" * 72)
    rng = np.random.default_rng(21)
    n_obs = 800
    _demo_linear_chain(rng, n_obs)
    _demo_nonlinear_shootout(rng, n_obs)


if __name__ == "__main__":
    run_all_examples()
