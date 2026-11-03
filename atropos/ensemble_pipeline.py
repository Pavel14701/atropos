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
* stage 3 DEAD                                     -> ``DEAD``
* stage 4 walk-forward FAIL                        -> ``NO_OOS_EDGE``
* stage 5 Sharpe floor / drawdown tail breach      -> ``RISK``
* stage 5 net-of-cost Sharpe floor breach (the gross
  envelope passes, the net one does not)           -> ``COST_DEATH``
* stage 6 OOS evaluation fail                      -> ``NO_OOS_EDGE``
* stage 7 cross-sectional tests fail               -> ``CS_NO_EDGE``
* all stages pass, no warnings                     -> ``SURVIVES``
* all stages pass, some warnings                   ->
  ``SURVIVES_WITH_WARNINGS``

Input ``data`` keys: ``returns`` (per-period forward returns the signal is
supposed to predict), ``signal`` (same index), optional ``features``
(DataFrame of known controls), optional ``mom7`` / ``mom30`` (observed 7d/30d
momentum factors; fallback: trailing sums of past returns), optional
``hypothesis`` (description string).  Panel mode: ``panel_signal`` /
``panel_returns`` DataFrames (rows = dates, columns = assets) enable stage 7;
they may be supplied alone (stages 1-6 are then SKIPPED: panel-only mode)
or together with the series inputs.

Run ``python -m atropos.ensemble_pipeline`` for the four synthetic
verification cases: a genuine edge must SURVIVE; a momentum echo must die
as REPACKAGING; a genuine cross-sectional panel must SURVIVE; a genuine
edge under prohibitive costs must die as COST_DEATH.
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

__all__ = ["DEFAULT_CONFIG", "run_all_examples", "run_full_pipeline"]

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
        # rows purged between train and test (label horizon; the pipeline
        # predicts one-period forward returns -> 1)
        "purge_gap": 1,
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
    3: "robustness battery catastrophic failure",
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
        # panel-only mode: no single-series stages
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
    year_dummies = pd.get_dummies(years).to_numpy(dtype=float)
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
    status = "FAIL" if failed else "PASS"
    summary = f"{len(results) - len(failed)}/6 controls passed"
    if failed:
        summary += f"; failed: {', '.join(failed)}"
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

    ``_cfg`` (the merged config) is accepted for signature symmetry with
    the other stage functions; stage 2 exposes no configurable knobs.
    """
    table = specification_curve(
        al["returns"], [al["features"]], al["signal"], robust=True
    )
    cs = curve_summary(table)
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
        f"{cs['n_specs']} specs ({cs['n_surviving']} with |t|>=2, "
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
            "window": int(bcfg.get("window", 20)),
            "n_boot": int(bcfg.get("n_boot", 1000)),
            "block": int(bcfg.get("block", 20)),
            "seed": int(cfg["seed"]),
        },
    )
    overall = battery_overall(table)
    n_pass = int(
        table.loc[~table["test"].isin(["OVERALL", "baseline"]), "verdict"]
        .eq("PASS")
        .sum()
    )
    status = {"ROBUST": "PASS", "FRAGILE": "WARN", "DEAD": "FAIL"}[overall]
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
    """Expanding-window walk-forward with anti-peeking enforcement."""
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
        )
    except ValueError as exc:
        return {
            "stage": 4,
            "name": _STAGE_NAMES[4],
            "status": "FAIL",
            "summary": f"walk-forward rejected the setup: {exc}",
            "results": {"error": str(exc)},
        }
    no_peek = check_no_peeking(wf)
    status = {"PASS": "PASS", "DEGRADED": "WARN", "FAIL": "FAIL"}[
        wf["verdict"]
    ]
    if not no_peek:
        status = "FAIL"
    summary = (
        f"{wf['verdict']}: wf_t={wf['wf_t']:.2f} "
        f"direction={wf['wf_direction_consistency']:.3f} "
        f"shrink={wf['wf_shrink_ratio']:.2f} ({wf['n_folds']} folds, "
        f"no_peeking={no_peek}, mode={wf['mode']}, "
        f"purge={int(wf.get('purge_gap', 0))})"
    )
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
    costs (``cost.cost_bps`` one-way per unit turnover).  When the net
    Sharpe floor is breached but the gross envelope clears it, the stage is
    flagged ``cost_breach`` and the overall verdict becomes ``COST_DEATH``
    instead of ``RISK``.
    """
    mc = cfg["mc"]
    cost_cfg = cfg["cost"]
    cost_bps = float(cost_cfg.get("cost_bps", 10.0))
    pos, strat = _strategy_position_returns(al, threshold)
    net = apply_proportional_costs(strat, pos, cost_bps)
    turn = annual_turnover(pos)
    breakeven = float(breakeven_cost_bps(strat, pos)["breakeven_cost_bps"])
    cost_drag = float(np.mean(strat - net))
    body = net[1:]
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
    reasons: list[str] = []
    cost_breach = False
    if not np.isfinite(sr_p5) or sr_p5 <= float(mc["min_sr_p5"]):
        # attribute the breach: costs or genuine lack of edge?
        gross_d = sharpe_distribution(
            strat[1:],
            n_sim=int(mc["n_sim"]),
            block=int(mc["block"]),
            seed=int(cfg["seed"]) + 51,
        )
        if np.isfinite(gross_d["sr_p5"]) and gross_d["sr_p5"] > float(
            mc["min_sr_p5"]
        ):
            cost_breach = True
            reasons.append(
                f"net SR p5={sr_p5:.2f} <= floor {mc['min_sr_p5']:.2f} "
                f"while gross p5={gross_d['sr_p5']:.2f} "
                f"(cost {cost_bps:.0f}bps vs breakeven {breakeven:.0f}bps)"
            )
        else:
            reasons.append(
                f"SR p5={sr_p5:.2f} <= floor {mc['min_sr_p5']:.2f}"
            )
    if not np.isfinite(mdd_p5) or mdd_p5 < float(mc["worst_allowed_mdd_p5"]):
        reasons.append(
            f"mDD p5={mdd_p5:.1%} < limit {mc['worst_allowed_mdd_p5']:.1%}"
        )
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
        warns.append(
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
    status = "PASS" if cs["overall_verdict"] == "PASS" else "FAIL"
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
# PLR0911 waived: one return per pipeline-stage verdict (stages 1-7, the
# panel-only branch, and the all-survived return), mirroring the
# kill-on-first-FAIL decision tree.
def run_full_pipeline(  # noqa: PLR0911, PLR0915
    data: dict[str, Any], hypothesis_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Chain the adversarial stages; kill on the first FAIL.

    Parameters
    ----------
    data : dict with ``returns`` (per-period forward returns), ``signal``
        (same index), optional ``features`` (known control factors), optional
        ``mom7`` / ``mom30`` (observed momentum factors; trailing-return sums
        are used when absent), optional ``hypothesis`` (description).
        Panel mode: ``panel_signal`` / ``panel_returns`` DataFrames (rows =
        dates, columns = assets) enable stage 7; given alone, stages 1-6 are
        SKIPPED (panel-only mode).
    hypothesis_config : optional overrides over :data:`DEFAULT_CONFIG`
        (nested dicts are merged key-wise).  Notable keys: ``cost.cost_bps``
        (one-way cost in bps, netted out in stage 5), ``model``
        (``"ols"`` | ``"knn"`` | ``"kernel"`` | callable factory, stage 6).

    Returns one dict: ``{hypothesis, n_obs, date_range, overall_verdict,
    killed_by_stage, kill_reason, stages (list of per-stage dicts),
    config, runtime_seconds}``.  Stages after the killer are ``SKIPPED``.
    """
    t0 = time.time()
    cfg = _merge_config(hypothesis_config)
    al, n_obs = _align_inputs(data)
    hypothesis = str(data.get("hypothesis", ""))
    has_series = al["returns"] is not None
    has_panel = al["panel_signal"] is not None

    stages: list[dict[str, Any]] = []

    def _finish(killed_stage: int | None) -> dict[str, Any]:
        if killed_stage is not None:
            repack = killed_stage == 1 and stages[0].get("repackaging")
            cost_kill = killed_stage == 5 and bool(
                stages[4].get("results", {}).get("cost_breach")
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
                verdict = _KILL_VERDICTS[killed_stage]
                reason = _KILL_REASONS[killed_stage]
        else:
            has_warn = any(stage["status"] == "WARN" for stage in stages)
            verdict = "SURVIVES_WITH_WARNINGS" if has_warn else "SURVIVES"
            reason = None
        next_stage = len(stages) + 1
        stages.extend(
            {
                "stage": skipped,
                "name": _STAGE_NAMES[skipped],
                "status": "SKIPPED",
                "summary": "not run: candidate killed earlier",
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
            "config": cfg,
            "runtime_seconds": round(time.time() - t0, 2),
        }

    if not has_series:  # panel-only mode: jump straight to stage 7
        stages.extend(
            {
                "stage": skipped,
                "name": _STAGE_NAMES[skipped],
                "status": "SKIPPED",
                "summary": "not run: panel-only mode (no series inputs)",
                "results": {},
            }
            for skipped in range(1, 7)
        )
        s7 = _stage_7_cross_sectional(al, cfg)
        stages.append(s7)
        if s7["status"] == "FAIL":
            return _finish(7)
        return _finish(None)

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
    threshold = float(s3["results"]["threshold_used"])

    # stage 4 -------------------------------------------------------------
    s4 = _stage_4_walkforward(al, cfg)
    stages.append(s4)
    if s4["status"] == "FAIL":
        return _finish(4)

    # stage 5 -------------------------------------------------------------
    s5 = _stage_5_monte_carlo(al, cfg, threshold)
    stages.append(s5)
    if s5["status"] == "FAIL":
        return _finish(5)
    strat_returns = s5["results"]["strategy_returns"]
    mdd_p50 = float(s5["results"]["mdd_distribution"]["mdd_p50"])

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
    print(
        f"n_obs={res['n_obs']}  "
        f"range={res['date_range'][0]} .. {res['date_range'][1]}"
    )
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
def _make_synthetic(
    kind: str, n_obs: int = 900, seed: int = 7
) -> dict[str, Any]:
    """Synthetic market world: latent momentum factor + hidden driver.

    Latent states are AR processes - persistent enough to survive the
    battery's one-period lag and its window smoothing (base window 10 ->
    5/10/15), but with memory short enough that run starts still carry edge
    and the lag-10 placebo is dead.  A candidate signal must survive the
    one-period position lag, which kills any non-persistent "edge".
    ``fwd = 1.0*driver + 0.6*mom + 0.55*eps``.

    ``kind='genuine_edge'``  - signal = hidden driver + 30% momentum bleed.
    ``kind='momentum_echo'`` - signal = momentum + noise (no new information).
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
    driver /= float(np.std(driver))  # orthogonal to mom by construction

    if kind == "genuine_edge":
        signal = driver + 0.3 * mom
        hyp = (
            "persistent order-flow state z predicts next-period return, "
            "incrementally to 7d/30d momentum (signal carries z plus a "
            "momentum bleed)"
        )
    elif kind == "momentum_echo":
        signal = mom + 0.4 * rng.standard_normal(n_obs)
        hyp = (
            "claim: a fresh order-flow signal - but it is just observed "
            "momentum plus noise (echo of a known factor)"
        )
    else:
        raise ValueError(f"unknown kind {kind!r}")

    fwd = 1.0 * driver + 0.6 * mom + 0.55 * rng.standard_normal(n_obs)
    mom7 = mom + 0.10 * rng.standard_normal(n_obs)  # observed 7d momentum
    mom30 = np.roll(mom, 12) + 0.10 * rng.standard_normal(
        n_obs
    )  # observed 30d factor
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(fwd, index=idx),
        "signal": pd.Series(signal, index=idx),
        "features": pd.DataFrame({"mom7": mom7, "mom30": mom30}, index=idx),
        "mom7": pd.Series(mom7, index=idx),
        "mom30": pd.Series(mom30, index=idx),
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


def run_all_examples() -> None:
    """Verify the four canonical cases of the 7-stage kill chain."""
    print()
    res_a = run_full_pipeline(_make_synthetic("genuine_edge", seed=4))
    _print_report(res_a, "CASE A: genuine edge (must SURVIVE, net of 10bps)")

    print()
    res_b = run_full_pipeline(_make_synthetic("momentum_echo", seed=4))
    _print_report(res_b, "CASE B: momentum echo (must die as REPACKAGING)")

    print()
    res_c = run_full_pipeline(_make_panel("genuine_edge", seed=13))
    _print_report(
        res_c, "CASE C: cross-sectional edge, panel-only (must SURVIVE)"
    )

    print()
    thin_edge = _make_synthetic("genuine_edge", seed=4)
    # realistic return amplitude: the same statistical edge, but daily
    # returns of ~2bp scale - the cost drag now dominates the gross edge
    thin_edge["returns"] = thin_edge["returns"] * 0.02
    thin_edge["hypothesis"] = (
        "genuine edge, but thin returns (~2bp daily): "
        "300bps one-way costs must eat it (COST-DEATH)"
    )
    res_d = run_full_pipeline(thin_edge, {"cost": {"cost_bps": 300.0}})
    _print_report(
        res_d,
        "CASE D: genuine edge, thin returns + prohibitive costs "
        "(must die COST_DEATH)",
    )

    print()
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    ok_a = res_a["overall_verdict"].startswith("SURVIVES")
    ok_b = res_b["overall_verdict"] == "REPACKAGING"
    ok_c = res_c["overall_verdict"].startswith("SURVIVES")
    ok_d = res_d["overall_verdict"] == "COST_DEATH"
    checks = (
        (
            "case A genuine edge  ",
            res_a["overall_verdict"],
            "SURVIVES*",
            ok_a,
        ),
        (
            "case B momentum echo ",
            res_b["overall_verdict"],
            "REPACKAGING",
            ok_b,
        ),
        (
            "case C cs panel edge ",
            res_c["overall_verdict"],
            "SURVIVES*",
            ok_c,
        ),
        (
            "case D cost death    ",
            res_d["overall_verdict"],
            "COST_DEATH",
            ok_d,
        ),
    )
    for label, got, expected, ok in checks:
        print(
            f"{label} -> {got:<24} "
            f"expected {expected:<12} [{'OK' if ok else 'MISMATCH'}]"
        )
    if not all(ok for *_, ok in checks):
        raise AssertionError("ensemble_pipeline verification failed")
    print("all verification checks passed")


if __name__ == "__main__":
    run_all_examples()
