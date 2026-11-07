"""Ensemble pipeline: one call that chains every hypothesis-killing module.

:func:`run_full_pipeline` feeds a candidate signal through up to seven
adversarial stages, in increasing order of cost, and KILLS the candidate the
moment any stage fails (remaining stages are reported as SKIPPED and never
run):

1. ``null_control_suite``      6 null controls (permutation, placebo events,
                               momentum, known features, subperiod
                               concentration, calendar fixed effects).
2. ``incremental_regression``  specification curve - does ANY reasonable
                               spec keep |t| >= 2?
3. ``robustness_battery``      7-test mechanical-trading battery (threshold /
                               window sensitivity, subperiods, leave-one-out,
                               placebo lag, declustering, monotonicity).
4. ``walkforward_framework``   expanding-window walk-forward with anti-peeking
                               enforcement.
5. ``monte_carlo``             block-bootstrap distributions of Sharpe and max
                               drawdown of the threshold rule NET OF
                               TRANSACTION COSTS (proportional one-way cost
                               per unit turnover, :mod:`atropos.cost_model`).
6. ``predictive_model_eval``   purged K-fold CV, rolling walk-forward OOS,
                               feature-importance stability, sign consistency
                               by regime, deflated Sharpe.  The regression
                               model is configurable: ``"ols"`` (default),
                               ``"knn"``, ``"kernel"`` (Nadaraya-Watson) or
                               any callable factory (e.g. an sklearn
                               wrapper) via ``config["model"]``.
7. ``cross_sectional``         optional panel stage (runs when the data
                               carries ``panel_signal`` / ``panel_returns``
                               DataFrames): per-date cross-sectional IC,
                               net-of-cost quantile long-short spread,
                               quantile monotonicity.

Decision tree (first FAIL decides):

* stage 1, momentum / known-features control fails -> ``REPACKAGING``
* stage 1, other controls fail                     -> ``ARTIFACT``
* stage 2 curve death                              -> ``REPACKAGING``
  (a curve SURVIVES call backed by fewer than 5 testable
  specs is INCONCLUSIVE - a WARN, not a pass)
* stage 3 DEAD or FRAGILE (fewer than 5/7 battery
  tests passed: ROBUSTNESS-DEATH)                  -> ``DEAD``
* stage 4 walk-forward FAIL, shrinkage over
  the cap (OVERFIT_DEATH)                          -> ``DEAD``
* stage 4 walk-forward FAIL (weak OOS edge only)  -> ``NO_OOS_EDGE``
* stage 4 walk-forward cannot run (the
  configuration exceeds the aligned data)          -> ``INCONCLUSIVE``
* stage 5 drawdown-tail breach, or a Sharpe-floor
  breach that persists gross of costs              -> ``RISK``
* stage 5 net-of-cost Sharpe floor breach ONLY (the
  gross envelope passes, drawdown tail intact), or
  the same net breach with the stage-5 breakeven
  gate fired (``cost_bps >= breakeven``, drawdown
  tail intact)                                     -> ``COST_DEATH``
* stage 6 OOS evaluation fail                      -> ``NO_OOS_EDGE``
* stage 7 cross-sectional tests fail               -> ``CS_NO_EDGE``
* panel-only mode: stage 7 passes, no warnings     -> ``SURVIVES_CS_ONLY``
* panel-only mode: stage 7 passes, some warnings   ->
  ``SURVIVES_CS_ONLY_WITH_WARNINGS``
* all series stages pass, no warnings              -> ``SURVIVES``
* all series stages pass, some warnings            ->
  ``SURVIVES_WITH_WARNINGS``

A plain ``SURVIVES`` means EVERY executed attack missed; when only the
cross-sectional stage ran (panel-only mode), the verdict says so
(``SURVIVES_CS_ONLY`` / ``SURVIVES_CS_ONLY_WITH_WARNINGS``) instead of
borrowing the stronger all-stages label.

Input ``data`` keys: ``returns`` (per-period forward returns the signal is
supposed to predict), ``signal`` (same index), optional ``features``
(DataFrame of known controls), optional ``mom7`` / ``mom30`` (observed 7d/30d
momentum factors; fallback: trailing sums of past returns), optional
``hypothesis`` (description string).  Panel mode: ``panel_signal`` /
``panel_returns`` DataFrames (rows = dates, columns = assets) enable stage 7;
they may be supplied alone (stages 1-6 are then SKIPPED: panel-only mode,
verdict ``SURVIVES_CS_ONLY``) or together with the series inputs.

Run ``python -m atropos.ensemble_pipeline`` for the ten synthetic
verification cases: a genuine edge must SURVIVE; a SLOW genuine edge
(smoothed driver) must SURVIVE - the battery passes slow-robust edges
(9/10 swept seeds; seed 7 stops on the ATR-04 FRAGILE kill); a momentum
echo must die as REPACKAGING; an edge concentrated in the first third
of the sample is caught by the OOS evaluation as NO_OOS_EDGE (its
single stage-1 concentration FAIL is only a WARN under ATR-12); a
genuine edge behind an unreachable entry threshold must die as DEAD; a
genuine edge under prohibitive costs must die as COST_DEATH; a genuine
edge under a strict drawdown mandate must die as RISK; a genuine
cross-sectional panel must report SURVIVES_CS_ONLY (only the
cross-sectional attack ran); a thin panel must report
SURVIVES_CS_ONLY_WITH_WARNINGS; a pure-noise panel must die as
CS_NO_EDGE.  ARTIFACT and INCONCLUSIVE are the two verdicts without a
demo case: ARTIFACT needs two independent stage-1 rejections with
clean factor controls (ATR-12; no honest synthetic reaches it - pinned
in tests/test_ensemble_pipeline.py), and INCONCLUSIVE needs a
walk-forward configuration that outgrows the data (unit-tested).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from atropos.cost_model import (
    annual_turnover,
    apply_proportional_costs,
    breakeven_cost_bps,
)
from atropos.cross_sectional import run_cross_sectional_tests
from atropos.incremental_regression import curve_summary, specification_curve
from atropos.monte_carlo import max_drawdown_distribution, sharpe_distribution
from atropos.null_control_suite import (
    calendar_control,
    known_features_control,
    momentum_control,
    placebo_placebo,
    shuffled_permutation,
    subperiod_stability,
)
from atropos.predictive_model_eval import (
    ModelFn,
    Predictor,
    deflated_performance,
    feature_importance_stability,
    kernel_model_fn,
    knn_model_fn,
    ols_model_fn,
    purged_kfold_cv,
    sign_consistency,
    walkforward_model_eval,
)
from atropos.robustness_battery import battery_overall, run_battery
from atropos.walkforward_framework import check_no_peeking, walkforward

__all__ = ["DEFAULT_CONFIG", "run_full_pipeline"]

DEFAULT_CONFIG: dict[str, Any] = {
    "n_perm_draws": 200,  # permutation / placebo draws (stage 1)
    "placebo_top_quantile": 0.95,  # event days = signal above this quantile
    "battery": {
        "threshold": None,
        "window": 10,
        "n_boot": 1000,
        "block": 40,
    },  # threshold None -> median(signal)
    "walkforward": {
        "train_window": 180,
        "test_window": 30,
        "step": 30,
        # purge_gap: rows removed from the END of each train slice -
        # the leakage guard between train and test (the pipeline
        # predicts one-period forward returns -> 1).  Distinct from
        # label_horizon below; never conflated with it.
        "purge_gap": 1,
        # label_horizon: forward-label overlap length the wf_t
        # statistic assumes (non-overlapping labels inside the test
        # union) - a DOCUMENTED assumption, not a purge setting.
        # >1 raises (overlap correction unimplemented - see
        # KNOWN-ISSUES.md section 6).
        "label_horizon": 1,
    },
    "purged_cv": {"n_splits": 5, "purge_gap": 1},
    "wf_eval": {"train_window": 180, "test_window": 30, "purge_gap": 1},
    "importance": {"n_bootstrap": 40, "top_k": 1},
    "mc": {
        "n_sim": 2000,
        "block": 20,
        "min_sr_p5": 0.0,
        "worst_allowed_mdd_p5": -0.60,
        "target_ann_vol": 0.20,
    },
    "deflated": {"n_strategies": 10, "sr_var_annual": 1.0},
    "cost": {
        # one-way proportional cost per unit turnover, in basis points
        # (stage 5 nets it out of the strategy returns; stage 7 uses it
        # for the long-short spread unless overridden there)
        "cost_bps": 10.0,
        # optional annual-turnover WARN ceiling (None = off)
        "max_turnover_annual": None,
    },
    "model": "ols",  # "ols" | "knn" | "kernel" | callable factory
    "model_params": {},  # kwargs for knn_model_fn / kernel_model_fn
    "cross_sectional": {
        "n_quantiles": 5,
        "min_assets": 5,
        # cost_bps defaults to cost.cost_bps when absent here
    },
    "seed": 0,
}

_STAGE_NAMES = {
    1: "null_control_suite",
    2: "incremental_regression",
    3: "robustness_battery",
    4: "walkforward_framework",
    5: "monte_carlo",
    6: "predictive_model_eval",
    7: "cross_sectional",
}

_KILL_VERDICTS: dict[int, str] = {
    1: "ARTIFACT",  # refined to REPACKAGING when the factor controls fail
    2: "REPACKAGING",
    3: "DEAD",
    4: "NO_OOS_EDGE",
    5: "RISK",  # refined to COST_DEATH when only the net envelope dies
    6: "NO_OOS_EDGE",
    7: "CS_NO_EDGE",
}

_KILL_REASONS: dict[int, str] = {
    1: (
        "the effect is an artifact "
        "(alignment / placebo / concentration / calendar)"
    ),
    2: "no specification keeps |t| >= 2 - repackaging of known information",
    3: (
        "robustness battery failed (ROBUSTNESS-DEATH: fewer than 5 of 7 "
        "tests passed, or a catastrophic test)"
    ),
    4: "no significant out-of-sample edge (expanding walk-forward)",
    5: "Monte Carlo risk envelope breached (Sharpe floor / drawdown tail)",
    6: "no significant out-of-sample edge (purged CV / rolling walk-forward)",
    7: (
        "cross-sectional tests fail (per-date IC / net quantile spread / "
        "quantile monotonicity)"
    ),
}

_COST_DEATH_REASON = (
    "net-of-cost Sharpe does not clear the floor while the gross envelope "
    "passes - the edge is eaten by transaction costs (COST-DEATH)"
)

_OVERFIT_DEATH_MARKER = "OVERFIT_DEATH"
"""Marker inside the stage-4 result's ``verdict_reason`` (set by
:func:`atropos.walkforward_framework.walkforward` when the shrink ratio
is finite and strictly above ``shrink_cap``): the verdict is a DEAD,
not a plain missing-edge NO_OOS_EDGE (FRAMEWORK.md stage-6 kill
criterion)."""

_INCONCLUSIVE_REASON = "insufficient data for walk-forward configuration"
"""Stage-4 INCONCLUSIVE cause: the configuration cannot fit even one
fold on the aligned data (ATR-06).  "Could not test" is not evidence
of "no edge", so this never reads NO_OOS_EDGE.  A shrinkage-NaN
INCONCLUSIVE carries its own verdict_reason instead of this one."""

# stage 2: minimum number of TESTABLE specs a curve must reach before a
# SURVIVES call is trusted (max-of-few-specs is a multiple search too
# small to believe); fewer specs downgrade the stage to INCONCLUSIVE/WARN
_MIN_CURVE_SPECS = 5


def _merge_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-ish merge of user config over :data:`DEFAULT_CONFIG`."""
    cfg = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in DEFAULT_CONFIG.items()
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


# ----------------------------------------------------------------------------
# input alignment
# ----------------------------------------------------------------------------
def _align_inputs(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Index-align every input (inner join, NaN rows dropped).

    Returns ``{"returns", "signal", "mom7", "mom30", "features",
    "panel_signal", "panel_returns"}`` where the series entries are Series /
    DataFrames sharing one index (or ``None`` in panel-only mode) and the
    panel entries are the raw DataFrames (or ``None`` when absent), plus the
    aligned observation count (panel dates in panel-only mode).
    """
    panel_signal = data.get("panel_signal")
    panel_returns = data.get("panel_returns")
    has_panel = isinstance(panel_signal, pd.DataFrame) and isinstance(
        panel_returns, pd.DataFrame
    )
    ret = data.get("returns")
    sig = data.get("signal")
    if ret is None or sig is None:
        if not has_panel or panel_signal is None or panel_returns is None:
            raise ValueError(
                "data must contain 'returns' and 'signal' series "
                "(or 'panel_signal' and 'panel_returns' DataFrames)"
            )
        # panel-only mode: no single-series stages.  The >= 300 floor
        # below deliberately does NOT apply here: it guards the series
        # stages (battery / walk-forward / block bootstrap need that many
        # rows of a single history), while panel evidence scales with
        # dates x assets and stage 7 enforces its own per-date
        # ``min_assets`` floor instead.
        return {
            "returns": None,
            "signal": None,
            "mom7": None,
            "mom30": None,
            "features": None,
            "panel_signal": panel_signal,
            "panel_returns": panel_returns,
        }, len(panel_signal.index)
    if not isinstance(ret, pd.Series):
        ret = pd.Series(np.asarray(ret, dtype=float).reshape(-1))
    if not isinstance(sig, pd.Series):
        sig = pd.Series(
            np.asarray(sig, dtype=float).reshape(-1), index=ret.index
        )

    def _ser(value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.astype(float)
        return pd.Series(
            np.asarray(value, dtype=float).reshape(-1), index=ret.index
        )

    feats_df = data.get("features")
    feat_series: dict[str, pd.Series] = {}
    if isinstance(feats_df, pd.DataFrame) and feats_df.shape[1] > 0:
        for column in feats_df.columns:
            feat_series[f"__f::{column}"] = feats_df[column].astype(float)

    mom7_in, mom30_in = data.get("mom7"), data.get("mom30")
    # fallback: trailing sums of PAST returns only (shift(1) -> no lookahead)
    mom7 = (
        _ser(mom7_in)
        if mom7_in is not None
        else ret.astype(float).shift(1).rolling(7).sum()
    )
    mom30 = (
        _ser(mom30_in)
        if mom30_in is not None
        else ret.astype(float).shift(1).rolling(30).sum()
    )

    pieces: dict[str, pd.Series] = {
        "__returns": ret.astype(float),
        "__signal": sig.astype(float),
        "__mom7": mom7,
        "__mom30": mom30,
    }
    pieces.update(feat_series)
    comb = pd.concat(pieces.values(), axis=1, join="inner")
    comb.columns = list(pieces.keys())
    comb = comb.dropna()
    n_obs = len(comb)
    if n_obs < 300:
        raise ValueError(
            f"need >= 300 aligned observations after dropna, got {n_obs}"
        )

    feat_cols = [
        column for column in comb.columns if column.startswith("__f::")
    ]
    if feat_cols:
        features = comb[feat_cols].rename(columns=lambda column: column[5:])
    else:
        features = pd.DataFrame(
            {"mom7": comb["__mom7"], "mom30": comb["__mom30"]}
        )
    aligned = {
        "returns": comb["__returns"],
        "signal": comb["__signal"],
        "mom7": comb["__mom7"],
        "mom30": comb["__mom30"],
        "features": features,
        "panel_signal": panel_signal if has_panel else None,
        "panel_returns": panel_returns if has_panel else None,
    }
    return aligned, n_obs


# ----------------------------------------------------------------------------
# stage implementations
# ----------------------------------------------------------------------------
def _stage_1_null_controls(
    al: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Six adversarial null controls on the aligned signal/returns."""
    seed = int(cfg["seed"])
    draws = int(cfg["n_perm_draws"])
    ret, sig = al["returns"], al["signal"]
    n_obs = len(ret)
    results: dict[str, dict[str, Any]] = {}
    results["shuffled_permutation"] = shuffled_permutation(
        sig, ret, n_draws=draws, seed=seed
    )

    k_events = max(
        20, round((1.0 - float(cfg["placebo_top_quantile"])) * n_obs)
    )
    event_pos = np.argsort(sig.to_numpy(dtype=float))[-k_events:]
    results["placebo_placebo"] = placebo_placebo(
        event_pos, ret, n_draws=draws, seed=seed + 1
    )

    results["momentum_control"] = momentum_control(
        ret, sig, al["mom7"], al["mom30"]
    )
    results["known_features_control"] = known_features_control(
        ret, sig, al["features"]
    )

    daily_ev = sig.to_numpy(dtype=float) * ret.to_numpy(dtype=float)
    results["subperiod_stability"] = subperiod_stability(
        daily_ev, periods=3, n_perm=draws, seed=seed + 2
    )

    idx = ret.index
    years = (
        idx.year.to_numpy()
        if isinstance(idx, pd.DatetimeIndex)
        else np.arange(n_obs) // 250
    )
    # drop_first avoids the dummy trap: calendar_control's design already
    # carries an intercept, so a FULL one-hot year set is exactly
    # collinear with it (rank-deficient -> INCONCLUSIVE).  Dropping one
    # category spans the same column space and leaves the signal's
    # t-stat identical.
    year_dummies = pd.get_dummies(years, drop_first=True).to_numpy(
        dtype=float
    )
    results["calendar_control"] = calendar_control(ret, sig, year_dummies)

    failed = [
        name
        for name, result in results.items()
        if result["verdict"] == "FAIL"
    ]
    repackaging = bool(failed) and any(
        name in failed
        for name in ("momentum_control", "known_features_control")
    )
    # rank-deficient controls (signal exactly collinear with a control)
    # are neither a pass nor a kill: they could not be tested
    inconclusive = [
        name
        for name, result in results.items()
        if result["verdict"] == "INCONCLUSIVE"
    ]
    # ATR-12: null-control hierarchy — one FAIL flags (WARN), two or
    # more independent rejections kill.  A single false positive at
    # alpha=0.05 across 6 controls is expected by chance (~26%); two
    # simultaneous rejections are unlikely enough to warrant a kill.
    status = (
        "FAIL"
        if len(failed) >= 2
        else ("WARN" if failed or inconclusive else "PASS")
    )
    summary = f"{len(results) - len(failed)}/6 controls passed"
    if failed:
        summary += f"; failed: {', '.join(failed)}"
    if inconclusive:
        summary += (
            f"; inconclusive (collinear, untested): {', '.join(inconclusive)}"
        )
    return {
        "stage": 1,
        "name": _STAGE_NAMES[1],
        "status": status,
        "summary": summary,
        "results": results,
        "repackaging": repackaging,
    }


def _stage_2_specification_curve(
    al: dict[str, Any], _cfg: dict[str, Any]
) -> dict[str, Any]:
    """Specification curve: 5 transforms x {none, known-features} controls.

    ``_cfg`` (the merged config) is accepted for call-site symmetry with
    the other stage functions; the stage is intentionally not
    configurable - the 5 x 2 specification lattice is fixed by design so
    that every candidate faces exactly the same curve.

    Trust conditions on top of the lenient curve verdict:

    * at least one CONTROLLED spec must survive, otherwise the curve is
      kept alive by uncontrolled specs only (repackaging of known
      information) -> FAIL;
    * at least 5 testable specs must reach the summary at all: taking
      the max over 1-4 specs is a multiple search too small to trust,
      so a SURVIVES call on a thin curve is downgraded to INCONCLUSIVE
      (WARN, not a kill - the evidence cannot decide) instead of PASS.
    """
    table = specification_curve(
        al["returns"], [al["features"]], al["signal"], robust=True
    )
    cs = curve_summary(table)
    n_specs = int(cs["n_specs"])
    if cs["verdict"] == "SURVIVES" and n_specs < _MIN_CURVE_SPECS:
        # thin curve: max-of-few-specs is not evidence - INCONCLUSIVE,
        # never a silent pass
        return {
            "stage": 2,
            "name": _STAGE_NAMES[2],
            "status": "WARN",
            "summary": (
                f"INCONCLUSIVE: only {n_specs} testable specs "
                f"(< {_MIN_CURVE_SPECS}) reach the summary - max|t|="
                f"{cs['max_abs_t']:.2f} over so few specs is a multiple "
                "search too small to trust; the SURVIVES call is not "
                "treated as a pass"
            ),
            "results": {"curve_summary": cs, "spec_table": table},
        }
    # the lenient curve verdict is not enough: at least one CONTROLLED
    # spec must survive, otherwise the curve is kept alive by
    # uncontrolled specs only (repackaging of known information)
    status = (
        "PASS"
        if cs["verdict"] == "SURVIVES"
        and int(cs["n_surviving_controlled"]) >= 1
        else "FAIL"
    )
    summary = (
        f"{cs['verdict']}: max|t|={cs['max_abs_t']:.2f} across "
        f"{n_specs} specs ({cs['n_surviving']} with |t|>=2, "
        f"{cs['n_surviving_controlled']} controlled, "
        f"sign_consistent={cs['sign_consistent']})"
    )
    return {
        "stage": 2,
        "name": _STAGE_NAMES[2],
        "status": status,
        "summary": summary,
        "results": {"curve_summary": cs, "spec_table": table},
    }


def _stage_3_robustness_battery(
    al: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Seven-test mechanical battery on the threshold rule."""
    bcfg = cfg["battery"]
    sig = al["signal"]
    threshold = bcfg.get("threshold")
    threshold = (
        float(np.median(sig.to_numpy(dtype=float)))
        if threshold is None
        else float(threshold)
    )
    table = run_battery(
        sig,
        al["returns"],
        {
            "threshold": threshold,
            "window": int(
                bcfg.get("window", DEFAULT_CONFIG["battery"]["window"])
            ),
            "n_boot": int(bcfg.get("n_boot", 1000)),
            "block": int(
                bcfg.get("block", DEFAULT_CONFIG["battery"]["block"])
            ),
            "seed": int(cfg["seed"]),
        },
    )
    overall = battery_overall(table)
    n_pass = int(
        table.loc[~table["test"].isin(["OVERALL", "baseline"]), "verdict"]
        .eq("PASS")
        .sum()
    )
    # FRAGILE is a KILL, not a warning: the doctrine (FRAMEWORK.md,
    # ROBUSTNESS-DEATH) stops the candidate when fewer than 5 of 7
    # battery tests pass -- a stage status softer than the doctrine
    # would let sub-5/7 books through on a warning (ATR-04).
    status = {"ROBUST": "PASS", "FRAGILE": "FAIL", "DEAD": "FAIL"}[overall]
    summary = (
        f"{overall}: {n_pass}/7 tests passed "
        f"(baseline threshold={threshold:.3f})"
    )
    return {
        "stage": 3,
        "name": _STAGE_NAMES[3],
        "status": status,
        "summary": summary,
        "results": {
            "battery_table": table,
            "overall": overall,
            "threshold_used": threshold,
        },
    }


def _stage_4_walkforward(
    al: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Expanding-window walk-forward with anti-peeking enforcement.

    Verdict mapping: PASS -> PASS, DEGRADED -> WARN, FAIL -> FAIL and
    INCONCLUSIVE -> INCONCLUSIVE (a shrinkage-NaN walk-forward cannot
    hand out a tier).  The walk-forward's ``verdict_reason`` is carried
    through verbatim (it marks OVERFIT_DEATH or the shrinkage-NaN
    cause) so the caller can classify without recomputing; when set it
    is also appended to the summary.

    A :class:`ValueError` whose message marks a capacity the data
    cannot carry (``"too small"``: not enough rows for even one fold)
    is INCONCLUSIVE, not a FAIL (ATR-06) -- the configuration did not
    fit, which says nothing about the edge.  Any other walk-forward
    rejection is a caller/config error (bad window sizes, negative
    purge gap, ``label_horizon > 1``) and PROPAGATES loudly instead of
    being classified here.
    """
    wcfg = cfg["walkforward"]
    df_wf = pd.DataFrame({"signal": al["signal"], "outcome": al["returns"]})
    try:
        wf = walkforward(
            df_wf,
            train_window=int(wcfg.get("train_window", 180)),
            test_window=int(wcfg.get("test_window", 30)),
            step=int(wcfg.get("step", 30)),
            forbid_calendar=True,
            purge_gap=int(wcfg.get("purge_gap", 0)),
            label_horizon=int(wcfg.get("label_horizon", 1)),
        )
    except ValueError as exc:
        if "too small" not in str(exc):
            # caller/config error (e.g. train_window below the
            # framework floor): re-raise, never classify it as the
            # data-capacity INCONCLUSIVE (a typo'd config would
            # otherwise read as "could not test")
            raise
        return {
            "stage": 4,
            "name": _STAGE_NAMES[4],
            "status": "INCONCLUSIVE",
            "summary": (
                "walk-forward could not run: insufficient data for"
                f" the configuration ({exc})"
            ),
            "results": {"error": str(exc)},
        }
    no_peek = check_no_peeking(wf)
    status = {
        "PASS": "PASS",
        "DEGRADED": "WARN",
        "FAIL": "FAIL",
        "INCONCLUSIVE": "INCONCLUSIVE",
    }[wf["verdict"]]
    if not no_peek:
        status = "FAIL"
    summary = (
        f"{wf['verdict']}: wf_t={wf['wf_t']:.2f} "
        f"direction={wf['wf_direction_consistency']:.3f} "
        f"shrink={wf['wf_shrink_ratio']:.2f} ({wf['n_folds']} folds, "
        f"no_peeking={no_peek}, mode={wf['mode']}, "
        f"purge={int(wf.get('purge_gap', 0))})"
    )
    reason = wf.get("verdict_reason")
    if reason:
        summary = f"{summary} [{reason}]"
    return {
        "stage": 4,
        "name": _STAGE_NAMES[4],
        "status": status,
        "summary": summary,
        "results": {key: value for key, value in wf.items() if key != "folds"}
        | {"folds": wf["folds"], "no_peeking": no_peek},
    }


def _strategy_position_returns(
    al: dict[str, Any], threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Long/flat rule on the signal with a one-period lag (no lookahead).

    Returns ``(positions, strategy_returns)`` where ``strat[t] =
    pos[t-1] * ret[t]`` and ``strat[0] = 0`` (entry costs are handled by
    :func:`atropos.cost_model.apply_proportional_costs` via the position
    changes).
    """
    sig = al["signal"].to_numpy(dtype=float)
    ret = al["returns"].to_numpy(dtype=float)
    pos = (sig > threshold).astype(float)
    strat = np.zeros(ret.size, dtype=float)
    strat[1:] = pos[:-1] * ret[1:]
    return pos, strat


def _stage_5_monte_carlo(
    al: dict[str, Any], cfg: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """Bootstrap Sharpe / max-drawdown envelope of the threshold rule.

    The envelope is computed on returns NET of proportional transaction
    costs (``cost.cost_bps`` one-way per unit turnover).  Verdict matrix
    (first match wins; ``cost_breach`` is set only on the last row):

    * drawdown tail breached (``mDD p5 < worst_allowed_mdd_p5``)
      -> ``RISK``
    * gross Sharpe floor also breached (no edge even before costs)
      -> ``RISK``
    * net Sharpe floor breached while the gross envelope clears it
      -> ``COST_DEATH`` (``cost_breach=True``)
    * net Sharpe floor breached, drawdown tail intact, and the
      breakeven gate fires (``cost_bps >= breakeven_cost_bps`` with a
      finite positive breakeven) -> ``COST_DEATH`` too: costs at or
      above the edge-eating level attribute the death to costs even
      when the gross bootstrap envelope is too noisy to vouch for the
      edge (the breakeven number itself proves a positive gross mean).
      The boundary is inclusive on the cost side - a cost exactly
      equal to the breakeven kills, because the net edge is then
      exactly eaten.  A non-positive breakeven (no gross edge in the
      mean), NaN (neither edge nor trading) and inf (never trades) can
      never fire the gate.

    So a drawdown breach always dominates: ``COST_DEATH`` requires the
    drawdown tail to be intact, and either the gross envelope to pass
    or the breakeven gate to fire.  The breakeven comparison is a kill
    ATTRIBUTION, not a reported number: when it fires, the verdict is
    the existing ``COST_DEATH`` path (no new token), and
    ``results["cost"]["breakeven_breach"]`` records that the gate (not
    the gross-envelope comparison) drove the attribution.

    Entry-day exclusion: ``strat[0] = 0`` by construction (no position
    is held on the entry day) and it carries no turnover charge, so
    every per-period statistic of this stage - the Sharpe /
    drawdown envelope AND ``cost_drag_mean`` - is computed on the
    post-entry body ``net[1:]``: the entry day is excluded from both.
    """
    mc = cfg["mc"]
    cost_cfg = cfg["cost"]
    cost_bps = float(cost_cfg.get("cost_bps", 10.0))
    pos, strat = _strategy_position_returns(al, threshold)
    net = apply_proportional_costs(strat, pos, cost_bps)
    turn = annual_turnover(pos)
    breakeven = float(breakeven_cost_bps(strat, pos)["breakeven_cost_bps"])
    body = net[1:]
    cost_drag = (
        float(np.mean((strat - net)[1:])) if net.size > 1 else float("nan")
    )
    sd = float(np.std(body, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return {
            "stage": 5,
            "name": _STAGE_NAMES[5],
            "status": "FAIL",
            "summary": (
                "degenerate strategy returns "
                "(no active trades or zero variance)"
            ),
            "results": {"n_active": int(np.sum(body != 0))},
        }
    # rescale to a sane annualized vol so compounded drawdowns are meaningful
    scale = (float(mc["target_ann_vol"]) / np.sqrt(252.0)) / sd
    scaled = body * scale
    sr_d = sharpe_distribution(
        scaled,
        n_sim=int(mc["n_sim"]),
        block=int(mc["block"]),
        seed=int(cfg["seed"]) + 51,
    )
    mdd_d = max_drawdown_distribution(
        scaled,
        n_sim=int(mc["n_sim"]),
        block=int(mc["block"]),
        seed=int(cfg["seed"]) + 52,
    )
    sr_p5, mdd_p5 = sr_d["sr_p5"], mdd_d["mdd_p5"]
    sr_floor = float(mc["min_sr_p5"])
    mdd_limit = float(mc["worst_allowed_mdd_p5"])
    sr_fail = not np.isfinite(sr_p5) or sr_p5 <= sr_floor
    mdd_fail = not np.isfinite(mdd_p5) or mdd_p5 < mdd_limit
    gross_sr_p5: float | None = None
    if sr_fail:
        # attribute the breach: costs or genuine lack of edge?
        gross_d = sharpe_distribution(
            strat[1:],
            n_sim=int(mc["n_sim"]),
            block=int(mc["block"]),
            seed=int(cfg["seed"]) + 51,
        )
        gross_sr_p5 = float(gross_d["sr_p5"])
    gross_pass = (
        gross_sr_p5 is not None
        and np.isfinite(gross_sr_p5)
        and gross_sr_p5 > sr_floor
    )
    # breakeven gate: a finite positive breakeven proves the gross MEAN
    # is positive, so costs at or above it attribute a net-envelope
    # death to costs even when the gross bootstrap p5 is too noisy to
    # clear the floor.  NaN (no edge, no trading), inf (never trades)
    # and a non-positive breakeven (no gross edge in the mean) can
    # never fire: the comparison then cannot attribute a death to
    # costs because there is no edge to eat.  The verdict matrix (see
    # docstring): a drawdown breach always yields RISK, COST_DEATH
    # needs net fail + intact drawdown tail + (gross pass or this gate)
    breakeven_breach = bool(
        np.isfinite(breakeven) and breakeven > 0.0 and cost_bps >= breakeven
    )
    cost_breach = (
        sr_fail and not mdd_fail and (gross_pass or breakeven_breach)
    )
    reasons: list[str] = []
    if sr_fail:
        if cost_breach:
            reasons.append(
                f"net SR p5={sr_p5:.2f} <= floor {sr_floor:.2f} "
                f"while gross p5={gross_sr_p5:.2f} "
                f"(cost {cost_bps:.0f}bps vs breakeven {breakeven:.0f}bps)"
            )
        else:
            reasons.append(f"SR p5={sr_p5:.2f} <= floor {sr_floor:.2f}")
    if mdd_fail:
        reasons.append(f"mDD p5={mdd_p5:.1%} < limit {mdd_limit:.1%}")
    warns: list[str] = []
    max_turn = cost_cfg.get("max_turnover_annual")
    if max_turn is not None and turn["annual_turnover"] > float(max_turn):
        warns.append(
            f"annual turnover {turn['annual_turnover']:.0f}/yr "
            f"> {float(max_turn):.0f}"
        )
    status = "FAIL" if reasons else ("WARN" if warns else "PASS")
    summary = (
        f"net SR p5/p50/p95 = {sr_d['sr_p5']:.2f}/{sr_d['sr_p50']:.2f}/"
        f"{sr_d['sr_p95']:.2f} | "
        f"mDD p5/p50/p95 = {mdd_d['mdd_p5']:.1%}/{mdd_d['mdd_p50']:.1%}/"
        f"{mdd_d['mdd_p95']:.1%} | "
        f"breakeven {breakeven:.0f}bps (cost {cost_bps:.0f}bps, "
        f"turnover {turn['annual_turnover']:.0f}/yr)"
    )
    if reasons:
        summary = "FAILED: " + "; ".join(reasons)
    elif warns:
        summary = "WARN: " + "; ".join(warns) + " | " + summary
    return {
        "stage": 5,
        "name": _STAGE_NAMES[5],
        "status": status,
        "summary": summary,
        "results": {
            "sr_distribution": sr_d,
            "mdd_distribution": mdd_d,
            "vol_scale_factor": scale,
            "strategy_returns": net,
            "gross_strategy_returns": strat,
            "cost": {
                "cost_bps": cost_bps,
                "breakeven_cost_bps": breakeven,
                # True when the breakeven gate (cost_bps >= breakeven,
                # finite positive breakeven) drove the cost_breach
                # attribution - see the verdict matrix in the docstring
                "breakeven_breach": breakeven_breach,
                "annual_turnover": turn["annual_turnover"],
                "n_changes": turn["n_changes"],
                "cost_drag_mean": cost_drag,
            },
            "cost_breach": cost_breach,
        },
    }


def _resolve_model(cfg: dict[str, Any]) -> ModelFn:
    """Map ``config['model']`` to a model factory used by stage 6.

    ``"ols"`` / ``"knn"`` / ``"kernel"`` select the bundled factories
    (``model_params`` are forwarded to the non-linear ones); any callable
    ``(features, target) -> predictor`` passes through unchanged, so an
    sklearn-style wrapper plugs in without new dependencies.
    """
    model_spec = cfg.get("model", "ols")
    params: dict[str, Any] = dict(cfg.get("model_params") or {})
    if callable(model_spec):
        passthrough: ModelFn = model_spec
        return passthrough
    if model_spec == "ols":
        return ols_model_fn

    def _knn_factory(
        features_train: np.ndarray, target_train: np.ndarray
    ) -> Predictor:
        return knn_model_fn(features_train, target_train, **params)

    def _kernel_factory(
        features_train: np.ndarray, target_train: np.ndarray
    ) -> Predictor:
        return kernel_model_fn(features_train, target_train, **params)

    factories: dict[str, ModelFn] = {
        "knn": _knn_factory,
        "kernel": _kernel_factory,
    }
    if model_spec not in factories:
        raise ValueError(
            f"unknown model {model_spec!r}: "
            "expected 'ols' | 'knn' | 'kernel' or a callable factory"
        )
    return factories[model_spec]


def _model_label(cfg: dict[str, Any]) -> str:
    """Human-readable model name for the stage-6 summary line."""
    model_spec = cfg.get("model", "ols")
    if isinstance(model_spec, str):
        return model_spec
    return getattr(model_spec, "__name__", "custom")


def _stage_6_predictive_eval(
    al: dict[str, Any],
    cfg: dict[str, Any],
    strat_returns: np.ndarray,
    mdd_p50: float,
) -> dict[str, Any]:
    """Purged CV + walk-forward + stability + sign + deflated Sharpe."""
    features = pd.concat(
        [al["signal"].rename("signal"), al["features"]], axis=1
    )
    target = al["returns"]
    model_fn = _resolve_model(cfg)
    cv = purged_kfold_cv(
        features,
        target,
        n_splits=int(cfg["purged_cv"]["n_splits"]),
        purge_gap=int(cfg["purged_cv"]["purge_gap"]),
        model_fn=model_fn,
    )
    wfe = walkforward_model_eval(
        features,
        target,
        model_fn,
        train_window=int(cfg["wf_eval"]["train_window"]),
        test_window=int(cfg["wf_eval"]["test_window"]),
        purge_gap=int(cfg["wf_eval"].get("purge_gap", 0)),
    )
    stab = feature_importance_stability(
        features,
        target,
        model_fn,
        n_bootstrap=int(cfg["importance"]["n_bootstrap"]),
        top_k=int(cfg["importance"]["top_k"]),
        seed=int(cfg["seed"]) + 61,
    )
    sc = sign_consistency(wfe["predictions"], wfe["realized"])
    dp = deflated_performance(
        strat_returns,
        n_strategies=int(cfg["deflated"]["n_strategies"]),
        mdd=mdd_p50,
        sr_var_annual=cfg["deflated"].get("sr_var_annual", 1.0),
    )

    fails: list[str] = []
    if not np.isfinite(cv["mean_ic"]) or cv["mean_ic"] <= 0.0:
        fails.append(f"purged-CV mean IC={cv['mean_ic']:.3f} <= 0")
    if wfe["verdict"] == "FAIL":
        fails.append(f"rolling walk-forward FAIL (pnl_t={wfe['pnl_t']:.2f})")
    if sc["verdict"] == "FAIL":
        fails.append(
            f"sign consistency FAIL (acc={sc['directional_accuracy']:.3f})"
        )
    warns: list[str] = []
    if sc["verdict"] == "INCONCLUSIVE":
        warns.append(
            "sign consistency INCONCLUSIVE "
            f"({sc.get('reason', 'degenerate outcomes')})"
        )
    if wfe["verdict"] == "DEGRADED":
        warns.append(
            f"rolling walk-forward DEGRADED (pnl_t={wfe['pnl_t']:.2f})"
        )
    if stab["verdict"] == "UNSTABLE":
        warns.append(
            f"feature importances unstable "
            f"(rho={stab['mean_pairwise_spearman']:.2f})"
        )
    if dp["verdict"] == "FAIL":
        # ATR-08: DSR FAIL is a KILL in both chains (aligned with
        # ml_training _gate_deflated_sharpe hard DEAD)
        fails.append(
            f"deflated Sharpe p={dp['p_value']:.3f} "
            f"(n_strategies={dp['n_strategies']})"
        )

    status = "FAIL" if fails else ("WARN" if warns else "PASS")
    summary = (
        f"[model={_model_label(cfg)}] "
        f"purged-CV IC={cv['mean_ic']:.3f} | "
        f"wf_eval pnl_t={wfe['pnl_t']:.2f} "
        f"hit={wfe['oos_hit_rate']:.3f} | "
        f"sign acc={sc['directional_accuracy']:.3f} | "
        f"DSR p={dp['p_value']:.3f}"
    )
    if fails:
        summary = f"[model={_model_label(cfg)}] FAILED: " + "; ".join(fails)
    elif warns:
        summary = f"[model={_model_label(cfg)}] WARN: " + "; ".join(warns)
    return {
        "stage": 6,
        "name": _STAGE_NAMES[6],
        "status": status,
        "summary": summary,
        "results": {
            "purged_cv": {
                key: value for key, value in cv.items() if key != "folds"
            },
            "cv_folds": cv["folds"],
            "wf_eval": {
                key: value
                for key, value in wfe.items()
                if key not in ("folds", "predictions", "realized")
            },
            "wf_eval_folds": wfe["folds"],
            "importance_stability": stab,
            "sign_consistency": sc,
            "deflated_performance": dp,
        },
    }


def _stage_7_cross_sectional(
    al: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Cross-sectional IC / quantile spread / monotonicity on the panel."""
    if al["panel_signal"] is None or al["panel_returns"] is None:
        raise ValueError("stage 7 needs 'panel_signal' and 'panel_returns'")
    cs_cfg = cfg.get("cross_sectional", {})
    cost_bps = float(
        cs_cfg.get("cost_bps", cfg["cost"].get("cost_bps", 10.0))
    )
    cs = run_cross_sectional_tests(
        al["panel_signal"],
        al["panel_returns"],
        cost_bps=cost_bps,
        n_quantiles=int(cs_cfg.get("n_quantiles", 5)),
        min_assets=int(cs_cfg.get("min_assets", 5)),
    )
    ic_res, spread_res, mono_res = cs["ic"], cs["spread"], cs["monotonicity"]
    # INCONCLUSIVE (too few effective dates / degenerate stats) must not
    # kill the candidate the way a FAIL does: it means "could not test",
    # which surfaces as a warning on an otherwise surviving run
    status = (
        "PASS"
        if cs["overall_verdict"] == "PASS"
        else ("WARN" if cs["overall_verdict"] == "INCONCLUSIVE" else "FAIL")
    )
    summary = (
        f"{cs['overall_verdict']}: ic mean={ic_res['mean_ic']:+.3f} "
        f"(t={ic_res['ic_tstat']:.1f}) | "
        f"spread net SR={spread_res['net_sharpe']:.2f} "
        f"(turnover {spread_res['annual_turnover']:.0f}/yr) | "
        f"mono rho={mono_res['spearman']:+.2f}"
    )
    return {
        "stage": 7,
        "name": _STAGE_NAMES[7],
        "status": status,
        "summary": summary,
        "results": {"cross_sectional": cs, "cost_bps": cost_bps},
    }


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------
def _stage_by_id(
    stages: list[dict[str, Any]], stage_id: int
) -> dict[str, Any]:
    """Return the result dict of the stage with id ``stage_id``.

    Lets cross-stage dependencies (the stage-3 threshold feeding stage 5,
    the stage-5 returns / mdd feeding stage 6, the kill attribution in
    ``_finish``) be resolved by stable stage id instead of positional
    indices into the ``stages`` list.
    """
    for stage in stages:
        if stage["stage"] == stage_id:
            return stage
    raise KeyError(f"stage {stage_id} has not run yet")


def _json_safe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable copy of the merged config.

    Callable values (a custom ``model`` factory, callable ``model_params``
    entries) are replaced with the ``"<callable>"`` placeholder; the
    pipeline itself keeps using the real objects.
    """
    return {key: (_json_safe_value(value)) for key, value in cfg.items()}


def _json_safe_value(value: Any) -> Any:
    """Recursively replace callables with a serializable placeholder.

    Tuples become lists and sets become sorted lists (entries are
    assumed to be mutually sortable), so containers ``json.dumps``
    cannot handle - a bare ``set`` - or would only convert at dump
    time - a tuple that still hides a callable inside - leave this
    function already JSON-serializable.
    """
    if callable(value):
        return "<callable>"
    if isinstance(value, dict):
        return {
            inner_key: _json_safe_value(inner)
            for inner_key, inner in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe_value(item) for item in value)
    return value


# PLR0911 waived: one return per pipeline-stage verdict (stages 1-7, the
# panel-only branch, and the all-survived return), mirroring the
# kill-on-first-FAIL decision tree.  Intentionally NOT refactored into a
# class: the flat return-per-verdict shape is the decision tree.
def run_full_pipeline(  # noqa: PLR0911, PLR0915
    data: dict[str, Any],
    hypothesis_config: dict[str, Any] | None = None,
    *,
    prereg_hash: str | None = None,
) -> dict[str, Any]:
    """Chain the adversarial stages; kill on the first FAIL.

    Args:
        data: dict with ``returns`` (per-period forward returns),
            ``signal`` (same index), optional ``features`` (known
            control factors), optional ``mom7`` / ``mom30`` (observed
            momentum factors; trailing-return sums are used when
            absent), optional ``hypothesis`` (description).  Panel
            mode: ``panel_signal`` / ``panel_returns`` DataFrames
            (rows = dates, columns = assets) enable stage 7; given
            alone, stages 1-6 are SKIPPED (panel-only mode).
        hypothesis_config: optional overrides over
            :data:`DEFAULT_CONFIG` (nested dicts are merged
            key-wise).  Notable keys: ``cost.cost_bps`` (one-way cost
            in bps, netted out in stage 5), ``model`` (``"ols"`` |
            ``"knn"`` | ``"kernel"`` | callable factory, stage 6).
        prereg_hash: keyword-only hash of the frozen preregistration
            this run is bound to (e.g.
            :func:`atropos.prereg_generator.spec_hash` of the frozen
            params).  Purely an anchor: when given, the report echoes
            it as ``report["prereg_hash"]`` so a verdict can never
            drift away from the spec it was preregistered under;
            ``None`` (the default) leaves the key ``None`` and the
            run unanchored.

    Returns:
        One dict: ``{hypothesis, n_obs, date_range, overall_verdict,
        killed_by_stage, kill_reason, stages (list of per-stage
        dicts), config, runtime_seconds, prereg_hash}``.  Verdicts:
        ``REPACKAGING`` / ``ARTIFACT`` / ``DEAD`` / ``NO_OOS_EDGE`` /
        ``RISK`` / ``COST_DEATH`` / ``CS_NO_EDGE`` (kills; ``DEAD``
        also covers a stage-4 OVERFIT_DEATH - walk-forward shrinkage
        strictly above the cap - with the gate's reason as
        ``kill_reason``),
        ``INCONCLUSIVE`` (a walk-forward configuration the data
        cannot carry, or a NaN shrinkage -- undecided, not a kill; the
        reason names the blocker), ``SURVIVES`` /
        ``SURVIVES_WITH_WARNINGS`` (every executed series attack
        missed) and the panel-only ``SURVIVES_CS_ONLY`` /
        ``SURVIVES_CS_ONLY_WITH_WARNINGS`` (only the cross-sectional
        attack ran -- never a plain ``SURVIVES``).  Stages after the
        killer are ``SKIPPED`` (with a kill-context summary); in a
        series-only survival stage 7 is ``SKIPPED`` as an optional
        stage (no panel data).  The returned ``config`` is a
        JSON-serializable copy: a callable ``model`` shows up as
        ``"<callable>"`` while the pipeline itself runs the real
        factory.

    Raises:
        ValueError: if ``data`` lacks a usable ``returns`` stream or
            the panel inputs violate the aligner's label contract.
    """
    t0 = time.time()
    cfg = _merge_config(hypothesis_config)
    al, n_obs = _align_inputs(data)
    hypothesis = str(data.get("hypothesis", ""))
    has_series = al["returns"] is not None
    has_panel = al["panel_signal"] is not None

    stages: list[dict[str, Any]] = []

    def _finish(
        killed_stage: int | None,
        *,
        panel_only: bool = False,
        inconclusive_reason: str | None = None,
        kill_verdict: str | None = None,
        kill_reason_override: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the report; kill on ``killed_stage`` is final.

        ``kill_verdict`` / ``kill_reason_override`` refine the default
        kill label for one case: the stage-4 OVERFIT_DEATH gate reads
        DEAD (not NO_OOS_EDGE) with the gate's own reason - the
        killed_by_stage and kill_chain handling of DEAD are unchanged.
        """
        if killed_stage is None and inconclusive_reason is not None:
            # "could not test" is not a kill and not a pass: undecided,
            # with the blocker named (ATR-06)
            verdict = "INCONCLUSIVE"
            reason = inconclusive_reason
        elif killed_stage is not None:
            repack = killed_stage == 1 and bool(
                _stage_by_id(stages, 1).get("repackaging")
            )
            cost_kill = killed_stage == 5 and bool(
                _stage_by_id(stages, 5).get("results", {}).get("cost_breach")
            )
            if repack:
                verdict = "REPACKAGING"
                reason = (
                    "the new signal adds nothing beyond known momentum / "
                    "known features"
                )
            elif cost_kill:
                verdict = "COST_DEATH"
                reason = _COST_DEATH_REASON
            else:
                verdict = (
                    kill_verdict
                    if kill_verdict is not None
                    else _KILL_VERDICTS[killed_stage]
                )
                reason = (
                    kill_reason_override
                    if kill_reason_override is not None
                    else _KILL_REASONS[killed_stage]
                )
        elif panel_only:
            # only the cross-sectional attack ran: a plain SURVIVES
            # ("every attack missed") is unreachable -- the CS-only
            # verdicts keep the scope visible (ATR-03)
            has_warn = any(stage["status"] == "WARN" for stage in stages)
            verdict = (
                "SURVIVES_CS_ONLY_WITH_WARNINGS"
                if has_warn
                else "SURVIVES_CS_ONLY"
            )
            reason = None
        else:
            has_warn = any(stage["status"] == "WARN" for stage in stages)
            verdict = "SURVIVES_WITH_WARNINGS" if has_warn else "SURVIVES"
            reason = None
        # context-dependent skip summary: after a kill the remaining
        # stages are moot; an inconclusive stop left them untestable; a
        # survivor simply lacked panel data for the optional stage 7
        if killed_stage is not None:
            skip_summary = "not run: candidate killed earlier"
        elif inconclusive_reason is not None:
            skip_summary = (
                "not run: pipeline stopped inconclusive "
                "(insufficient walk-forward data)"
            )
        else:
            skip_summary = "not run: optional stage (no panel data)"
        next_stage = len(stages) + 1
        stages.extend(
            {
                "stage": skipped,
                "name": _STAGE_NAMES[skipped],
                "status": "SKIPPED",
                "summary": skip_summary,
                "results": {},
            }
            for skipped in range(next_stage, 8)
        )
        index = (
            al["returns"].index
            if al["returns"] is not None
            else al["panel_signal"].index
        )
        return {
            "hypothesis": hypothesis,
            "n_obs": n_obs,
            "date_range": (str(index.min()), str(index.max())),
            "overall_verdict": verdict,
            "killed_by_stage": killed_stage,
            "kill_reason": reason,
            "stages": stages,
            "config": _json_safe_config(cfg),
            "runtime_seconds": round(time.time() - t0, 2),
            "prereg_hash": prereg_hash,
        }

    if not has_series:  # panel-only mode: jump straight to stage 7
        stages.extend(
            {
                "stage": skipped,
                "name": _STAGE_NAMES[skipped],
                "status": "SKIPPED",
                "summary": (
                    "not run: panel-only mode - only the cross-sectional"
                    " stage was tested"
                ),
                "results": {},
            }
            for skipped in range(1, 7)
        )
        s7 = _stage_7_cross_sectional(al, cfg)
        stages.append(s7)
        if s7["status"] == "FAIL":
            return _finish(7)
        return _finish(None, panel_only=True)

    # stage 1 -------------------------------------------------------------
    s1 = _stage_1_null_controls(al, cfg)
    stages.append(s1)
    if s1["status"] == "FAIL":
        return _finish(1)

    # stage 2 -------------------------------------------------------------
    s2 = _stage_2_specification_curve(al, cfg)
    stages.append(s2)
    if s2["status"] == "FAIL":
        return _finish(2)

    # stage 3 -------------------------------------------------------------
    s3 = _stage_3_robustness_battery(al, cfg)
    stages.append(s3)
    if s3["status"] == "FAIL":
        return _finish(3)

    # stage 4 -------------------------------------------------------------
    s4 = _stage_4_walkforward(al, cfg)
    stages.append(s4)
    if s4["status"] == "INCONCLUSIVE":
        # a shrinkage-NaN INCONCLUSIVE names its own cause (visible,
        # not swallowed by the generic "insufficient data" reason)
        s4_reason = str(s4.get("results", {}).get("verdict_reason") or "")
        return _finish(
            None,
            inconclusive_reason=s4_reason or _INCONCLUSIVE_REASON,
        )
    if s4["status"] == "FAIL":
        s4_reason = str(s4.get("results", {}).get("verdict_reason") or "")
        if _OVERFIT_DEATH_MARKER in s4_reason:
            # FRAMEWORK.md stage-6 kill criterion: the shrink gate
            # fired - an overfit signature is a DEAD, not a plain
            # missing-edge NO_OOS_EDGE; kill_chain handles DEAD
            # already and killed_by_stage stays 4
            return _finish(
                4,
                kill_verdict="DEAD",
                kill_reason_override=s4_reason,
            )
        return _finish(4)

    # stage 5 (threshold dependency: the battery's stage-3 threshold) ------
    battery_threshold = float(
        _stage_by_id(stages, 3)["results"]["threshold_used"]
    )
    s5 = _stage_5_monte_carlo(al, cfg, battery_threshold)
    stages.append(s5)
    if s5["status"] == "FAIL":
        return _finish(5)
    stage_five_results = _stage_by_id(stages, 5)["results"]
    strat_returns = stage_five_results["strategy_returns"]
    mdd_p50 = float(stage_five_results["mdd_distribution"]["mdd_p50"])

    # stage 6 -------------------------------------------------------------
    s6 = _stage_6_predictive_eval(al, cfg, strat_returns, mdd_p50)
    stages.append(s6)
    if s6["status"] == "FAIL":
        return _finish(6)

    # stage 7 (optional: panel data present) --------------------------------
    if has_panel:
        s7 = _stage_7_cross_sectional(al, cfg)
        stages.append(s7)
        if s7["status"] == "FAIL":
            return _finish(7)

    return _finish(None)


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def _print_report(res: dict[str, Any], title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    if res["hypothesis"]:
        print(f"hypothesis : {res['hypothesis']}")
    config = res["config"]
    cost_cfg = config.get("cost", {})
    mc_cfg = config.get("mc", {})
    print(
        f"config    : seed={config.get('seed')} "
        f"model={config.get('model')} "
        f"cost_bps={cost_cfg.get('cost_bps')} "
        f"n_perm_draws={config.get('n_perm_draws')} "
        f"mc.n_sim={mc_cfg.get('n_sim')}"
    )
    print(
        f"n_obs={res['n_obs']}  "
        f"range={res['date_range'][0]} .. {res['date_range'][1]}"
    )
    if res.get("prereg_hash"):
        print(f"prereg    : {res['prereg_hash']}")
    header = f"{'stage':>5}  {'module':<24} {'status':<7} summary"
    print(header)
    print("-" * 78)
    for stage in res["stages"]:
        print(
            f"{stage['stage']:>5}  {stage['name']:<24} "
            f"{stage['status']:<7} {stage['summary']}"
        )
    print("-" * 78)
    if res["killed_by_stage"] is not None:
        print(
            f"OVERALL VERDICT: {res['overall_verdict']}  "
            f"(killed at stage {res['killed_by_stage']}: "
            f"{res['kill_reason']})"
        )
    else:
        print(f"OVERALL VERDICT: {res['overall_verdict']}")
    print(f"runtime: {res['runtime_seconds']}s")


# ----------------------------------------------------------------------------
# synthetic verification cases
# ----------------------------------------------------------------------------
# SLOW-class smoothing window (rows) for kind="slow_genuine_edge"; see
# _make_synthetic for why this is 8 and not the ~20 a first instinct
# suggests (subperiod evidence concentration of the squared state).
_SLOW_SMOOTHING_WINDOW = 8
# concentrated-edge world: the driver edge lives only in the first
# _CONCENTRATED_EDGE_FRACTION of the sample, at _CONCENTRATED_EDGE_STRENGTH
_CONCENTRATED_EDGE_FRACTION = 0.35
_CONCENTRATED_EDGE_STRENGTH = 2.0


def _smoothed_state(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of ``values`` (expanding head), rescaled to unit std."""
    smoothed = (
        pd.Series(values)
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    spread = float(np.std(smoothed))
    if not spread > 0.0:
        raise ValueError("smoothed state has zero variance")
    return smoothed / spread


def _make_synthetic(
    kind: str, n_obs: int = 900, seed: int = 7
) -> dict[str, Any]:
    """Synthetic market world: latent momentum factor + hidden driver.

    Latent states are AR processes - persistent enough to survive the
    battery's one-period lag, with memory short enough that run starts
    still carry edge.  A candidate signal must survive the one-period
    position lag, which kills any non-persistent "edge".  Returns are
    ``fwd = edge_term + 0.6*mom + 0.55*eps``; the two latent states are
    independent of each other by construction (separate Gaussian shocks).

    Two CLASSES of genuine edge (the fast/slow review contract):

    * ``kind='genuine_edge'`` (FAST class) - signal = hidden driver +
      30% momentum bleed; returns load on the raw AR(0.5) driver
      (lag-1 autocorr ~0.5, dead by lag 10).  The driver's memory is
      SHORTER than the battery's smoothing windows (5/10/15), so
      window-smoothing the signal inverts its alignment with the
      next-period return (a smoothed high reads as "the state was high
      over the whole window" -> mean-reversion anti-signal).  The
      battery correctly kills this class on a minority of data seeds -
      pinned, never loosened, in tests/test_pipeline_sweep.py.
    * ``kind='slow_genuine_edge'`` (SLOW class) - the same driver is
      averaged into a slow state (trailing mean over
      ``_SLOW_SMOOTHING_WINDOW`` rows, unit variance; lag-1 autocorr
      ~0.95); signal = slow state + 30% momentum bleed; returns load
      on the slow state.  The signal's information is spread over a
      multi-row window, so the battery's smoothing OVERLAPS the
      information window instead of washing past it: the edge
      attenuates but never inverts, and the battery correctly lets
      the slow class through on 9/10 swept seeds (seed 7 reads FRAGILE
      3/7 and stops under the ATR-04 ROBUSTNESS-DEATH - pinned in
      tests/test_pipeline_sweep.py).  The window is
      deliberately modest: per-day evidence is ~ slow**2, and a
      ~20-row window concentrates that evidence into subperiod
      chunks (max_share above the stage-1 cap -> ARTIFACT kill on
      ~4/10 swept seeds, measured), so the window sits where the
      squared-state memory stays inside one subperiod third while
      the level memory still dwarfs the battery's smoothing scale.

    Two kill worlds:

    * ``kind='momentum_echo'`` - signal = momentum + noise (no new
      information) -> stage-1 known-features FAIL (a WARN under the
      ATR-12 hierarchy) -> stage-2 REPACKAGING kill.
    * ``kind='concentrated_edge'`` - a genuine driver edge that only
      ever existed in the first ``_CONCENTRATED_EDGE_FRACTION`` of the
      sample (full-sample regressions stay significant); the evidence
      concentrates in one subperiod -> stage-1 subperiod control FAIL
      (a single FAIL is only a WARN under ATR-12, factor controls
      clean) -> the stage-6 OOS evaluation catches the edge-free tail
      -> NO_OOS_EDGE at n_obs=400 (at 900 rows the deeper stages keep
      passing and the run survives with the stage-1 warning).

    Known features are built from PAST information only (no lookahead):

    * ``mom_noisy`` - a noisy contemporaneous observation of latent
      momentum (NOT a 7-day sum, hence the honest name);
    * ``mom12`` - momentum lagged 12 periods, shifted with a 12-row NaN
      head (NOT ``np.roll``, which would wrap future values around to
      the front - a lookahead); ``_align_inputs`` drops the NaN rows.

    No ``mom7`` / ``mom30`` factors are supplied: the pipeline falls back
    to its own trailing sums of past returns, so alignment drops ~30
    leading rows and the usable ``n_obs`` shrinks accordingly.
    """
    rng = np.random.default_rng(seed)
    mom = np.zeros(n_obs)
    mom_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        mom[pos] = 0.5 * mom[pos - 1] + mom_shocks[pos]
    mom /= float(np.std(mom))
    driver = np.zeros(n_obs)  # hidden persistent driver
    driver_shocks = rng.standard_normal(n_obs)
    for pos in range(1, n_obs):
        driver[pos] = 0.5 * driver[pos - 1] + driver_shocks[pos]
    driver /= float(np.std(driver))  # independent of mom by construction

    if kind == "genuine_edge":
        signal = driver + 0.3 * mom
        edge_term = 1.0 * driver
        hyp = (
            "persistent order-flow state z predicts next-period return, "
            "incrementally to 7d/30d momentum (signal carries z plus a "
            "momentum bleed)"
        )
    elif kind == "slow_genuine_edge":
        slow = _smoothed_state(driver, _SLOW_SMOOTHING_WINDOW)
        signal = slow + 0.3 * mom
        edge_term = 1.0 * slow
        hyp = (
            "slow order-flow state (multi-period average of the hidden "
            "driver) predicts next-period return: robust to the "
            "battery's window smoothing, unlike the fast class"
        )
    elif kind == "momentum_echo":
        signal = mom + 0.4 * rng.standard_normal(n_obs)
        edge_term = 1.0 * driver
        hyp = (
            "claim: a fresh order-flow signal - but it is just observed "
            "momentum plus noise (echo of a known factor)"
        )
    elif kind == "concentrated_edge":
        signal = driver + 0.3 * mom
        loading = np.zeros(n_obs)
        edge_rows = int(_CONCENTRATED_EDGE_FRACTION * n_obs)
        loading[:edge_rows] = _CONCENTRATED_EDGE_STRENGTH
        edge_term = loading * driver
        hyp = (
            "claim: a persistent edge - but it only ever existed in the "
            "first third of the sample (in-sample concentration)"
        )
    else:
        raise ValueError(f"unknown kind {kind!r}")

    fwd = edge_term + 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
    # observed feature: noisy read of latent momentum (past-only: mom is
    # the state at t, observable when the signal is built)
    mom_noisy = mom + 0.10 * rng.standard_normal(n_obs)
    # observed feature: momentum lagged 12 periods - head padded with NaN
    # so nothing from the future leaks in (np.roll would wrap it around)
    mom12 = np.concatenate([np.full(12, np.nan), mom[:-12]]) + 0.10 * (
        rng.standard_normal(n_obs)
    )
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(fwd, index=idx),
        "signal": pd.Series(signal, index=idx),
        "features": pd.DataFrame(
            {"mom_noisy": mom_noisy, "mom12": mom12}, index=idx
        ),
        "hypothesis": hyp,
    }


def _make_panel(
    kind: str,
    n_dates: int = 240,
    n_assets: int = 40,
    seed: int = 13,
) -> dict[str, Any]:
    """Synthetic cross-sectional panel: per-asset persistent factor.

    Each asset carries a latent AR(0.85) state; the factor is z-scored
    across assets per date (pure cross-sectional information, no market
    timing).  Forward returns load on the same factor plus noise.
    Persistence keeps the long-short turnover realistic so the demo shows
    a meaningful cost drag rather than an iid churn.

    ``kind='genuine_edge'``  - signal = factor + 30% noise (real ranking).
    ``kind='cs_noise'``      - signal = pure noise (no ranking information).
    """
    rng = np.random.default_rng(seed)
    factor = np.zeros((n_dates, n_assets))
    shocks = rng.standard_normal((n_dates, n_assets))
    for row in range(1, n_dates):
        factor[row] = 0.85 * factor[row - 1] + shocks[row]
    mean_per_date = factor.mean(axis=1, keepdims=True)
    std_per_date = factor.std(axis=1, keepdims=True)
    std_per_date[std_per_date == 0.0] = 1.0
    zfactor = (factor - mean_per_date) / std_per_date

    if kind == "genuine_edge":
        signal = zfactor + 0.30 * rng.standard_normal((n_dates, n_assets))
        hyp = (
            "cross-sectional order-flow imbalance ranks assets: next-day "
            "relative returns are predictable from today's cross-section"
        )
    elif kind == "cs_noise":
        signal = rng.standard_normal((n_dates, n_assets))
        hyp = (
            "claim: a fresh cross-sectional ranking signal - but it is "
            "pure noise across assets"
        )
    else:
        raise ValueError(f"unknown kind {kind!r}")

    fwd = (
        0.0005
        + 0.004 * zfactor
        + 0.010 * rng.standard_normal((n_dates, n_assets))
    )
    idx = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    columns = [f"asset_{asset_no:02d}" for asset_no in range(n_assets)]
    return {
        "panel_signal": pd.DataFrame(signal, index=idx, columns=columns),
        "panel_returns": pd.DataFrame(fwd, index=idx, columns=columns),
        "hypothesis": hyp,
    }


def _verdict_matches(verdict: str, expected: str) -> bool:
    """Check a demo verdict against its expected label.

    ``"SURVIVES*"`` matches every alive form (``SURVIVES``,
    ``SURVIVES_WITH_WARNINGS``, ``SURVIVES_CS_ONLY``,
    ``SURVIVES_CS_ONLY_WITH_WARNINGS``); any other expected label is an
    exact equality.
    """
    if expected == "SURVIVES*":
        return verdict.startswith("SURVIVES")
    return verdict == expected


def _verify_demo_cases(
    cases: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """Assert the demo contract: verdicts, kill stages, kill causes.

    Layer-0 self-check of the demo: every case's overall verdict must
    match its expected label, the canonical kills must land at the right
    stage with the right cause, and the achieved verdict-matrix coverage
    is printed (10/12 - ARTIFACT needs a two-rejection stage-1 world no
    honest synthetic reaches, and INCONCLUSIVE is config-driven; both
    are pinned in tests/test_ensemble_pipeline.py).
    """
    checks = [
        (
            case["label"],
            result["overall_verdict"],
            case["expect"],
            _verdict_matches(result["overall_verdict"], case["expect"]),
        )
        for case, result in zip(cases, results, strict=True)
    ]
    for label, got, expected, ok in checks:
        print(
            f"{label} -> {got:<24} "
            f"expected {expected:<12} [{'OK' if ok else 'MISMATCH'}]"
        )
    if not all(ok for *_, ok in checks):
        raise AssertionError("ensemble_pipeline verification failed")
    (res_a, res_b, res_c, res_d, res_e, res_f, res_g, res_h, res_i, res_j) = (
        results
    )
    # canonical kills land at the right stages with the right causes
    assert res_a["killed_by_stage"] is None
    # momentum echo: the single stage-1 known-features FAIL is only a
    # WARN (ATR-12); the stage-2 curve finds no surviving CONTROLLED
    # spec and kills as REPACKAGING
    assert res_b["killed_by_stage"] == 2
    assert res_b["overall_verdict"] == "REPACKAGING"
    assert res_b["stages"][0]["status"] == "WARN"
    assert res_b["stages"][0]["repackaging"] is True
    assert res_b["stages"][1]["status"] == "FAIL"
    assert res_c["killed_by_stage"] is None
    # panel-only survivor: CS-scoped verdict, never a plain SURVIVES
    assert res_c["overall_verdict"] == "SURVIVES_CS_ONLY"
    assert res_c["stages"][6]["status"] == "PASS"
    assert res_d["killed_by_stage"] == 5
    assert res_d["stages"][4]["results"]["cost_breach"] is True
    assert res_e["killed_by_stage"] == 7
    assert res_e["stages"][6]["status"] == "FAIL"
    assert res_f["killed_by_stage"] == 5
    assert res_f["stages"][4]["status"] == "FAIL"
    assert res_f["stages"][4]["results"]["cost_breach"] is False
    assert "mDD" in res_f["stages"][4]["summary"]
    # the slow class survives with the battery stage passing it through
    # (ROBUST; a FRAGILE battery is a stage-3 kill since ATR-04)
    assert res_g["killed_by_stage"] is None
    assert res_g["stages"][2]["status"] == "PASS"
    # thin panel: INCONCLUSIVE is a warning, not a kill; the CS-only
    # scope stays visible in the verdict
    assert res_h["killed_by_stage"] is None
    assert res_h["overall_verdict"] == "SURVIVES_CS_ONLY_WITH_WARNINGS"
    assert res_h["stages"][6]["status"] == "WARN"
    assert res_h["stages"][6]["summary"].startswith("INCONCLUSIVE")
    # unreachable threshold: the rule never trades -> DEAD at stage 3
    assert res_i["killed_by_stage"] == 3
    assert res_i["stages"][2]["status"] == "FAIL"
    battery_table = res_i["stages"][2]["results"]["battery_table"]
    overall_note = str(
        battery_table.loc[battery_table["test"] == "OVERALL", "note"].iloc[0]
    )
    assert "no active days" in overall_note
    # concentrated edge: the single stage-1 concentration FAIL is only
    # a WARN (ATR-12), the factor controls stay clean (NOT repackaging),
    # and the stage-6 OOS evaluation kills the edge-free tail
    assert res_j["overall_verdict"] == "NO_OOS_EDGE"
    assert res_j["killed_by_stage"] == 6
    assert res_j["stages"][0]["status"] == "WARN"
    assert res_j["stages"][0]["repackaging"] is False
    assert (
        res_j["stages"][0]["results"]["subperiod_stability"]["verdict"]
        == "FAIL"
    )
    assert res_j["stages"][5]["name"] == "predictive_model_eval"
    assert res_j["stages"][5]["status"] == "FAIL"
    covered = {
        "SURVIVES",
        "SURVIVES_WITH_WARNINGS",
        "SURVIVES_CS_ONLY",
        "SURVIVES_CS_ONLY_WITH_WARNINGS",
        "REPACKAGING",
        "DEAD",
        "COST_DEATH",
        "RISK",
        "CS_NO_EDGE",
        "NO_OOS_EDGE",
    }
    print(
        f"verdict matrix coverage: {len(covered)}/12 "
        "(missing ARTIFACT: a stage-1 kill needs two independent "
        "rejections under ATR-12 and no honest synthetic fails two "
        "controls while keeping the factor controls clean, and "
        "INCONCLUSIVE: needs an undersized walk-forward configuration "
        "- both pinned in tests/test_ensemble_pipeline.py)"
    )
    print("all verification checks passed")


def run_all_examples() -> None:
    """Verify the ten cases of the 7-stage kill chain.

    Cases A-D are the four canonical verdicts (series survive,
    repackaging, panel-only CS-scoped survive, cost death); E-F pin the
    panel kill and the risk-mandate kill; G-J complete the verdict
    matrix (slow-class survive, thin-panel CS-scoped warnings,
    unreachable-threshold DEAD, concentration NO_OOS_EDGE).  ARTIFACT
    and INCONCLUSIVE are the two verdicts without a demo case:
    ARTIFACT needs two independent stage-1 rejections with clean
    factor controls (ATR-12; no honest synthetic reaches it), and
    INCONCLUSIVE needs a walk-forward configuration that outgrows the
    data - both are pinned in tests/test_ensemble_pipeline.py.
    """

    def thin_edge_data() -> dict[str, Any]:
        # realistic return amplitude: the same statistical edge, but
        # daily returns of ~2bp scale - the cost drag now dominates the
        # gross edge (breakeven ~75bps here, so 140bps one-way is
        # prohibitive while the net path stays inside the -60%
        # drawdown tail: a pure COST_DEATH, not a risk breach)
        thin = _make_synthetic("genuine_edge", seed=4)
        thin["returns"] = thin["returns"] * 0.02
        thin["hypothesis"] = (
            "genuine edge, but thin returns (~2bp daily): "
            "140bps one-way costs must eat it (COST-DEATH)"
        )
        return thin

    cases: list[dict[str, Any]] = [
        {
            "label": "case A genuine edge  ",
            "title": "CASE A: genuine edge (must SURVIVE, net of 10bps)",
            "data": _make_synthetic("genuine_edge", seed=4),
            "config": None,
            "expect": "SURVIVES*",
        },
        {
            "label": "case B momentum echo ",
            "title": "CASE B: momentum echo (must die as REPACKAGING)",
            "data": _make_synthetic("momentum_echo", seed=4),
            "config": None,
            "expect": "REPACKAGING",
        },
        {
            "label": "case C cs panel edge ",
            "title": (
                "CASE C: cross-sectional edge, panel-only "
                "(must report SURVIVES_CS_ONLY: only the CS attack ran)"
            ),
            "data": _make_panel("genuine_edge", seed=13),
            "config": None,
            "expect": "SURVIVES_CS_ONLY",
        },
        {
            "label": "case D cost death    ",
            "title": (
                "CASE D: genuine edge, thin returns + prohibitive costs "
                "(must die COST_DEATH)"
            ),
            "data": thin_edge_data(),
            "config": {"cost": {"cost_bps": 140.0}},
            "expect": "COST_DEATH",
        },
        {
            "label": "case E cs noise panel",
            "title": (
                "CASE E: cross-sectional noise panel (must die CS_NO_EDGE)"
            ),
            "data": _make_panel("cs_noise", seed=13),
            "config": None,
            "expect": "CS_NO_EDGE",
        },
        {
            "label": "case F drawdown risk ",
            "title": (
                "CASE F: genuine edge under a strict drawdown mandate "
                "(must die RISK)"
            ),
            # a strict risk mandate on the SAME genuine edge as case A:
            # both Sharpe floors pass comfortably (net SR p5 ~ 3.8 at the
            # mild default 10bps cost, gross likewise), but the bootstrap
            # mDD p5 sits near -16% on the 20%-vol scaling - a mandate
            # tolerating only a 5% drawdown-tail breach is breached while
            # the edge is real: RISK, not COST_DEATH, not a Sharpe kill.
            "data": _make_synthetic("genuine_edge", seed=4),
            "config": {"mc": {"worst_allowed_mdd_p5": -0.05}},
            "expect": "RISK",
        },
        {
            "label": "case G slow edge     ",
            "title": (
                "CASE G: slow genuine edge (must SURVIVE: the battery "
                "passes the slow-robust class)"
            ),
            "data": _make_synthetic("slow_genuine_edge", seed=4),
            "config": None,
            "expect": "SURVIVES*",
        },
        {
            "label": "case H thin panel    ",
            "title": (
                "CASE H: thin cross-sectional panel "
                "(must SURVIVES_CS_ONLY_WITH_WARNINGS)"
            ),
            # 15 dates < stage 7's min_dates: the cross-sectional killers
            # cannot be evaluated -> INCONCLUSIVE -> WARN; the run
            # survives WITH WARNINGS under the CS-only verdict (missing
            # data is not a kill, and only the CS attack ran)
            "data": _make_panel("genuine_edge", n_dates=15, seed=13),
            "config": None,
            "expect": "SURVIVES_CS_ONLY_WITH_WARNINGS",
        },
        {
            "label": "case I dead threshold",
            "title": (
                "CASE I: genuine edge, unreachable entry threshold "
                "(must die DEAD)"
            ),
            # honest DEAD by configuration: threshold 10.0 lies above
            # every draw of the signal (std ~1), so the long/flat rule
            # never trades - "no active days" DEAD at stage 3.  The
            # stage-1/2 controls are cost/rule-blind and still pass:
            # the edge is real, the RULE is untradeable.
            "data": _make_synthetic("genuine_edge", seed=4),
            "config": {"battery": {"threshold": 10.0}},
            "expect": "DEAD",
        },
        {
            "label": "case J concentration",
            "title": (
                "CASE J: edge concentrated in the first third "
                "(stage-1 concentration FAIL is only a WARN per ATR-12; "
                "the OOS evaluation catches the edge-free tail - "
                "NO_OOS_EDGE)"
            ),
            # n_obs=400 (the sweep scale the test suite pins): at the
            # demo-default 900 rows the deeper stages keep passing (the
            # expanding walk-forward never leaves the edge era) and the
            # single stage-1 FAIL would only WARN the run into a plain
            # SURVIVES_WITH_WARNINGS; at 400 rows stage 6 reads the
            # edge-free tail as a sign-consistency FAIL -> NO_OOS_EDGE.
            "data": _make_synthetic("concentrated_edge", n_obs=400, seed=4),
            "config": None,
            "expect": "NO_OOS_EDGE",
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        print()
        result = run_full_pipeline(case["data"], case["config"])
        _print_report(result, case["title"])
        results.append(result)

    print()
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    _verify_demo_cases(cases, results)


if __name__ == "__main__":
    run_all_examples()
