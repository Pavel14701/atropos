"""No-lookahead metamorphic tests: the future must not move the past.

Every honest backtest owes the same structural promise: changing data
that a fold could never have observed (outcomes AFTER its test start)
must leave that fold's training artifacts untouched.  These tests
mutate the future on purpose and assert the past does not move:

* ``walkforward`` (frame API) -- fold geometry unchanged, and the
  FIRST fold's fitted coefficients and predictions bitwise identical
  (captured through recording fit/predict hooks), while later folds
  do move (the mutation has teeth);
* ``walkforward_model_eval`` (array API) -- the same metamorphosis via
  its public ``predictions`` output, plus a peeking-predictor probe:
  the predictor receives a features-only ndarray, so asking for the
  outcome column fails structurally before any forecast exists;
* ``ensemble_pipeline._make_synthetic``'s ``mom12`` idiom -- lagging a
  series through a NaN-padded concatenation (never ``np.roll``) makes
  the lagged head provably independent of the tail's future values.

The frame-API peek guard itself (``test["outcome"]`` raising KeyError
because walkforward drops the outcome column) is already pinned in
``tests/test_walkforward_framework.py`` and is not duplicated here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.predictive_model_eval import (
    Predictor,
    ols_model_fn,
    walkforward_model_eval,
)
from atropos.walkforward_framework import check_no_peeking, walkforward

_GEOMETRY_COLS = [
    "fold",
    "train_end",
    "test_start",
    "test_end",
    "n_train",
    "purge_gap",
]

_WFE_GEOMETRY_COLS = [
    "fold",
    "train_lo",
    "train_hi",
    "test_lo",
    "test_hi",
    "n_train",
    "n_test",
    "purge_gap",
]


class _RecordingOls:
    """Deterministic fit/predict pair that records every call's model.

    Mirrors the default OLS hook of ``walkforward`` (intercept + the
    ``signal`` column) while appending a copy of the fitted
    coefficients and of every prediction, so two runs can be compared
    call by call.  ``walkforward`` also fits once more on the FULL
    sample for its in-sample reference, so ``models`` / ``preds``
    carry ``n_folds + 1`` entries: per-fold artifacts first, the
    in-sample refit last.
    """

    def __init__(self) -> None:
        self.models: list[np.ndarray] = []
        self.preds: list[np.ndarray] = []

    def fit(self, train_df: pd.DataFrame) -> dict[str, Any]:
        """Fit intercept + signal OLS on the train slice; record beta."""
        target = train_df["outcome"].to_numpy(dtype=float)
        feats = train_df[["signal"]].to_numpy(dtype=float)
        design = np.column_stack([np.ones(target.size), feats])
        beta = np.asarray(
            np.linalg.lstsq(design, target, rcond=None)[0], dtype=float
        )
        self.models.append(beta.copy())
        return {"beta": beta}

    def predict(
        self, model: dict[str, Any], test_df: pd.DataFrame
    ) -> np.ndarray:
        """Score the test slice with the frozen beta; record forecasts."""
        feats = test_df[["signal"]].to_numpy(dtype=float)
        design = np.column_stack([np.ones(feats.shape[0]), feats])
        scores = np.asarray(design @ model["beta"], dtype=float)
        self.preds.append(scores.copy())
        return scores


def test_walkforward_future_outcome_mutation_spares_first_fold() -> None:
    """Future outcomes spare the first fold and the fold geometry.

    Mutating outcomes from the first test start changes nothing the
    first fold could have fitted, and the fold table keeps its
    geometry.  The mutation point is the FIRST fold's test start
    (185 = 180 train + 5 purge), so fold 0's train slice ``[0, 180)``
    is strictly in the past: its recorded coefficients and predictions
    must be BITWISE identical between the runs, while at least one
    later fold (whose expanding train window swallows mutated
    outcomes) must move -- otherwise the mutation would be too weak
    to prove anything.
    """
    rng = np.random.default_rng(42)
    n_rows = 400
    signal = rng.standard_normal(n_rows)
    outcome = 0.4 * signal + rng.standard_normal(n_rows) * 0.7
    frame = pd.DataFrame({"signal": signal, "outcome": outcome})

    def run(data: pd.DataFrame) -> tuple[dict[str, Any], _RecordingOls]:
        recorder = _RecordingOls()
        result = walkforward(
            data,
            train_window=180,
            test_window=30,
            step=30,
            fit_fn=recorder.fit,
            predict_fn=recorder.predict,
            purge_gap=5,
        )
        return result, recorder

    base_result, base = run(frame)
    first_test_start = int(base_result["folds"]["test_start"].iloc[0])
    assert first_test_start == 185

    mutated_frame = frame.copy()
    replacement = rng.standard_normal(n_rows) * 3.0 + 10.0
    mutated_frame.loc[mutated_frame.index >= first_test_start, "outcome"] = (
        replacement[first_test_start:]
    )
    # the mutation must be effective, or the invariance below is vacuous
    assert not np.array_equal(
        frame["outcome"].to_numpy(dtype=float)[first_test_start:],
        mutated_frame["outcome"].to_numpy(dtype=float)[first_test_start:],
    )
    mutated_result, mutated = run(mutated_frame)

    # fold GEOMETRY is a pure function of the row count and the windows
    pd.testing.assert_frame_equal(
        base_result["folds"][_GEOMETRY_COLS],
        mutated_result["folds"][_GEOMETRY_COLS],
    )
    assert base_result["n_folds"] == mutated_result["n_folds"]
    assert check_no_peeking(base_result) is True
    assert check_no_peeking(mutated_result) is True

    # walkforward fits n_folds times plus one in-sample refit on the
    # full sample, so both recorders saw n_folds + 1 calls
    assert len(base.models) == base_result["n_folds"] + 1
    assert len(mutated.models) == len(base.models)

    # THE no-lookahead claim: fold 0 trained strictly before the
    # mutation point, so its coefficients and predictions cannot move
    assert np.array_equal(base.models[0], mutated.models[0])
    assert np.array_equal(base.preds[0], mutated.preds[0])

    # sensitivity: later folds (and the in-sample refit) train on rows
    # inside the mutated region and therefore DO move
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            base.models[1:], mutated.models[1:], strict=True
        )
    )
    assert not np.array_equal(base.models[-1], mutated.models[-1])


def test_mom12_lag_idiom_head_ignores_future_values() -> None:
    """The lag-12 head ignores future values of the mom process.

    Zeroing the TAIL of a momentum process cannot touch the lag-12
    head -- the exact shift idiom ``_make_synthetic`` uses for mom12.
    ``ensemble_pipeline._make_synthetic`` builds ``mom12`` as
    ``concat([NaN * 12, mom[:-12]])`` -- a true shift with a 12-row
    NaN head, because ``np.roll`` would wrap future values around to
    the front, which is a lookahead.  The latent mom process is
    internal to the function, so the metamorphosis is applied to the
    idiom itself on a locally built AR series: after replacing the
    last ``k_mut`` mom values, every shifted position below
    ``n - k_mut + lag`` is unchanged (the NaN head included), and the
    idiom agrees with ``pd.Series.shift`` bitwise.  The NaN-head guard
    on the real ``_make_synthetic`` output is pinned separately in
    ``tests/test_ensemble_pipeline.py``.
    """
    rng = np.random.default_rng(8)
    n_obs, lag, k_mut = 200, 12, 20
    mom = np.zeros(n_obs)
    shocks = rng.standard_normal(n_obs)
    for step_pos in range(1, n_obs):
        mom[step_pos] = 0.5 * mom[step_pos - 1] + shocks[step_pos]

    def lag12(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.full(lag, np.nan), values[:-lag]])

    base = lag12(mom)
    mutated_mom = mom.copy()
    mutated_mom[n_obs - k_mut :] = 0.0
    mutated = lag12(mutated_mom)

    # the 12-row NaN head survives the mutation untouched
    assert bool(np.isnan(base[:lag]).all())
    assert bool(np.isnan(mutated[:lag]).all())
    # only shifted positions fed by the mutated tail may move
    untouched_upto = n_obs - k_mut + lag
    assert np.array_equal(
        base[:untouched_upto], mutated[:untouched_upto], equal_nan=True
    )
    # sensitivity: the visible tail of the lagged series does move
    assert not np.array_equal(
        base[untouched_upto:], mutated[untouched_upto:], equal_nan=True
    )
    # the concat idiom IS a shift: identical to pandas' lag bitwise
    assert np.array_equal(
        base, pd.Series(mom).shift(lag).to_numpy(), equal_nan=True
    )
    # and that is exactly what makes it honest: np.roll WOULD leak the
    # future into the head (its head carries the mutated tail's values)
    rolled_head = np.roll(mutated_mom, lag)[:lag]
    assert bool(np.isfinite(rolled_head).all())


def test_wfe_predictor_structurally_cannot_see_test_outcome() -> None:
    """A peeking predictor structurally cannot see the test outcome.

    Its test input is a features-only ndarray, so asking for an
    ``"outcome"`` column dies.  ``walkforward_framework.walkforward``
    drops the outcome column from the test frame and the peek dies
    with KeyError (pinned in ``tests/test_walkforward_framework.py``).
    The array-API sibling ``walkforward_model_eval`` of
    ``predictive_model_eval`` enforces the same promise by
    construction: the predictor is called on ``features_arr[te]``
    alone, so there is no outcome to index into -- the probe dies with
    IndexError before any forecast exists.  The FIT side legitimately
    receives training data, and it receives EXACTLY the rows
    ``[0, train_window)`` -- never a test row.
    """
    rng = np.random.default_rng(4)
    n_rows = 300
    train_window = 120
    feats = rng.standard_normal((n_rows, 2))
    target = 0.3 * feats[:, 0] + rng.standard_normal(n_rows) * 0.5
    fit_feats: list[np.ndarray] = []
    fit_targets: list[np.ndarray] = []

    def peeking_model_fn(
        train_features: np.ndarray, train_target: np.ndarray
    ) -> Predictor:
        """Fit hook that records the exact slice it was handed."""

        def predict(features_test: Any) -> np.ndarray:
            """Predictor that tries to read realized test outcomes."""
            return np.asarray(features_test["outcome"], dtype=float)

        fit_feats.append(np.asarray(train_features, dtype=float).copy())
        fit_targets.append(np.asarray(train_target, dtype=float).copy())
        return predict

    with pytest.raises(IndexError, match="valid indices"):
        walkforward_model_eval(
            feats,
            target,
            peeking_model_fn,
            train_window=train_window,
            test_window=30,
        )
    # the (only, failing-first) fit saw exactly the train rows
    # [0, 120) -- features AND labels, nothing from any test slice
    assert len(fit_feats) == 1
    assert np.array_equal(fit_feats[0], feats[:train_window])
    assert np.array_equal(fit_targets[0], target[:train_window])


def test_wfe_future_target_mutation_freezes_first_fold() -> None:
    """Future targets leave fold 0's stacked predictions bitwise equal.

    Mutating targets from the first test start leaves fold 0's
    forecasts untouched.  ``walkforward_model_eval`` publishes its
    stacked out-of-sample ``predictions`` (fold order), so the
    no-lookahead claim can be made through the public output: the
    first ``test_window`` predictions come from a model fitted on
    ``[0, train_window)`` only and cannot move when every target from
    ``train_window + purge_gap`` on is replaced; later folds retrain
    on swallowed mutated rows and must move.  Fold geometry columns
    likewise cannot move (the per-fold IC / hit / PnL columns
    legitimately flip -- they score test rows against the mutated
    outcomes and are excluded here).
    """
    rng = np.random.default_rng(19)
    n_rows, train_window, test_window, purge_gap = 400, 180, 30, 5
    feats = rng.standard_normal((n_rows, 2))
    target = 0.5 * feats[:, 0] + rng.standard_normal(n_rows) * 0.5

    base = walkforward_model_eval(
        feats,
        target,
        ols_model_fn,
        train_window=train_window,
        test_window=test_window,
        purge_gap=purge_gap,
    )
    first_test_start = train_window + purge_gap
    assert int(base["folds"]["test_lo"].iloc[0]) == first_test_start

    mutated_target = target.copy()
    mutated_target[first_test_start:] = (
        rng.standard_normal(n_rows - first_test_start) * 5.0 + 2.0
    )
    assert not np.array_equal(
        target[first_test_start:], mutated_target[first_test_start:]
    )
    mutated = walkforward_model_eval(
        feats,
        mutated_target,
        ols_model_fn,
        train_window=train_window,
        test_window=test_window,
        purge_gap=purge_gap,
    )

    pd.testing.assert_frame_equal(
        base["folds"][_WFE_GEOMETRY_COLS],
        mutated["folds"][_WFE_GEOMETRY_COLS],
    )
    assert base["n_folds"] == mutated["n_folds"]
    # fold 0's forecasts: trained on [0, 180), tested on [185, 215)
    assert np.array_equal(
        base["predictions"][:test_window],
        mutated["predictions"][:test_window],
    )
    # sensitivity: at least one later fold retrains on mutated rows
    assert not np.array_equal(base["predictions"], mutated["predictions"])
