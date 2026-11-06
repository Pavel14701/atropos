"""Unified kill-chain entry point: rule pipeline + optional ML chain.

:func:`run_full_kill_chain` always runs the rule-based 7-stage
pipeline (:func:`atropos.ensemble_pipeline.run_full_pipeline`) on
``rule_data`` and, when ``ml_dataset`` is supplied, the walk-forward
ML engine (:func:`atropos.ml_training.run_ml_kill_protocol`) on
``ml_dataset``.  It is a thin aggregator, not a third engine: no new
gates, no thresholds, no chain or model selection -- both engine
reports are passed through verbatim and the unified verdict is a pure
aggregate of the two chain verdicts.  A kill in either chain kills
the hypothesis; survival needs every executed chain to survive.

Unified verdict matrix (documented aggregate; ``SURVIVES*`` abbreviates
the four alive forms ``SURVIVES`` / ``SURVIVES_WITH_WARNINGS`` /
``SURVIVES_CS_ONLY`` / ``SURVIVES_CS_ONLY_WITH_WARNINGS``, ``[...]``
names ``unified["source"]``)::

    rule overall_verdict | ml verdict   | unified verdict
    ---------------------+--------------+----------------------
    SURVIVES*            | (not run)    | SURVIVES
    any kill verdict     | (not run)    | KILLED   [rule]
    INCONCLUSIVE         | (not run)    | INCONCLUSIVE [rule]
    SURVIVES*            | SURVIVES     | SURVIVES
    SURVIVES*            | DEAD(...)    | KILLED   [ml]
    SURVIVES*            | UNDERPOWERED | INCONCLUSIVE [ml]
    any kill verdict     | SURVIVES     | KILLED   [rule]
    any kill verdict     | DEAD(...)    | KILLED   [rule+ml]
    any kill verdict     | UNDERPOWERED | KILLED   [rule]
    INCONCLUSIVE         | any decision | INCONCLUSIVE [rule]

Rule kill verdicts are every ``overall_verdict`` outside the four alive
forms and ``INCONCLUSIVE``: ``REPACKAGING``, ``ARTIFACT``, ``DEAD``,
``NO_OOS_EDGE``, ``COST_DEATH``, ``RISK``, ``CS_NO_EDGE``.  Undecided
is not death on either chain: a rule ``INCONCLUSIVE`` (a walk-forward
configuration the data cannot carry, ATR-06) and the ML ``UNDERPOWERED``
(the declustered event floor is not met) both map to ``INCONCLUSIVE`` --
a fresh snapshot with more power may still decide the hypothesis.

Run ``python -m atropos.kill_chain`` for the asserted demo: a
rule-only noise case dies from the rule chain, a genuine rule edge
with a noise ML dataset dies from the ML chain, the full genuine
case survives both chains, and an underpowered ML dataset stays
INCONCLUSIVE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atropos.ensemble_pipeline import run_full_pipeline
from atropos.ml_training import run_ml_kill_protocol

__all__ = ["run_full_kill_chain"]

_RULE_ALIVE_VERDICTS = frozenset(
    {
        "SURVIVES",
        "SURVIVES_WITH_WARNINGS",
        # panel-only alive forms: only the cross-sectional attack ran
        # (ATR-03) -- still a survival, with the scope kept visible
        "SURVIVES_CS_ONLY",
        "SURVIVES_CS_ONLY_WITH_WARNINGS",
    }
)
_RULE_SERIES_KEYS = ("returns", "signal")
_RULE_PANEL_KEYS = ("panel_signal", "panel_returns")
_ML_DATASET_KEYS = ("features", "target", "exec_returns", "times_us")

# Frozen demo configs (deterministic, reduced budgets): the SURVIVES
# world is the planted quadrant-interaction edge at 600 rows, the DEAD
# world is powered pure noise under a logit (the engine's own demo
# recipes at demo scale).
_DEMO_ML_SURVIVES_CONFIG: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"gbm": {"n_estimators": 25}},
}
_DEMO_ML_DEAD_CONFIG: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"logit": {"max_iter": 500}},
}
_DEMO_ML_UNDERPOWERED_CONFIG: dict[str, Any] = {
    **_DEMO_ML_SURVIVES_CONFIG,
    "n_min_events": 100_000,
}


def _missing_keys(
    dataset: dict[str, Any], keys: tuple[str, ...]
) -> list[str]:
    """Return the required keys absent from ``dataset``."""
    return [key for key in keys if key not in dataset]


def _validate_rule_data(rule_data: dict[str, Any]) -> None:
    """Reject a rule dataset without the series or the panel pair.

    Raises:
        ValueError: when neither ``returns`` + ``signal`` nor
            ``panel_signal`` + ``panel_returns`` is present (the
            contract of ``run_full_pipeline``), before any stage runs.
    """
    series_missing = _missing_keys(rule_data, _RULE_SERIES_KEYS)
    panel_missing = _missing_keys(rule_data, _RULE_PANEL_KEYS)
    if series_missing and panel_missing:
        raise ValueError(
            "rule_data must provide 'returns' + 'signal' series (or "
            "'panel_signal' + 'panel_returns' DataFrames); missing: "
            f"{series_missing + panel_missing}"
        )


def _validate_ml_dataset(ml_dataset: dict[str, Any]) -> None:
    """Reject an ML dataset without the four contract keys.

    Raises:
        ValueError: when any of ``features`` / ``target`` /
            ``exec_returns`` / ``times_us`` is absent (the dataset
            contract of ``run_ml_kill_protocol``), before any model
            is fit.
    """
    missing = _missing_keys(ml_dataset, _ML_DATASET_KEYS)
    if missing:
        raise ValueError(
            "ml_dataset must provide 'features', 'target', "
            f"'exec_returns', 'times_us'; missing: {missing}"
        )


def _unify(rule_verdict: str, ml_verdict: str | None) -> dict[str, Any]:
    """Aggregate the two chain verdicts into the unified namespace.

    Kill dominates, undecidedness (ML ``UNDERPOWERED``) beats bare
    survival, and plain ``SURVIVES`` requires every executed chain to
    be alive.  Pure function of the two verdict strings: no winner
    picking, no re-interpretation of the engine reports.
    """
    rule_dead = rule_verdict not in _RULE_ALIVE_VERDICTS
    ml_dead = ml_verdict is not None and ml_verdict.startswith("DEAD")
    if rule_dead and ml_dead:
        return {
            "verdict": "KILLED",
            "rule_verdict": rule_verdict,
            "ml_verdict": ml_verdict,
            "source": "rule+ml",
            "reason": (
                f"both chains killed: rule {rule_verdict}, ml {ml_verdict}"
            ),
        }
    if rule_dead:
        return {
            "verdict": "KILLED",
            "rule_verdict": rule_verdict,
            "ml_verdict": ml_verdict,
            "source": "rule",
            "reason": f"rule chain killed: {rule_verdict}",
        }
    if ml_dead:
        return {
            "verdict": "KILLED",
            "rule_verdict": rule_verdict,
            "ml_verdict": ml_verdict,
            "source": "ml",
            "reason": f"ml chain killed: {ml_verdict}",
        }
    if rule_verdict == "INCONCLUSIVE":
        # ATR-06: a walk-forward configuration the data cannot carry is
        # undecided, not a kill -- "could not test" is not evidence of
        # "no edge", so it can never read KILLED [rule] or SURVIVES
        return {
            "verdict": "INCONCLUSIVE",
            "rule_verdict": rule_verdict,
            "ml_verdict": ml_verdict,
            "source": "rule",
            "reason": (
                f"rule chain undecided: {rule_verdict} (undecided is "
                "not death; decide on a wider snapshot)"
            ),
        }
    if ml_verdict is not None and ml_verdict != "SURVIVES":
        return {
            "verdict": "INCONCLUSIVE",
            "rule_verdict": rule_verdict,
            "ml_verdict": ml_verdict,
            "source": "ml",
            "reason": (
                f"ml chain undecided: {ml_verdict} (undecided is not "
                "death; decide on a wider snapshot)"
            ),
        }
    return {
        "verdict": "SURVIVES",
        "rule_verdict": rule_verdict,
        "ml_verdict": ml_verdict,
        "source": None,
        "reason": None,
    }


def run_full_kill_chain(
    rule_data: dict[str, Any],
    ml_dataset: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
    ml_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both kill chains on one hypothesis; aggregate one verdict.

    The rule chain (7 adversarial stages, 12 verdicts) always runs on
    ``rule_data``; the ML chain (walk-forward gates) runs only when
    ``ml_dataset`` is given.  The wrapper adds no gates and makes no
    choice between the chains: ``rule`` and ``ml`` carry the engine
    reports verbatim, ``unified`` is the documented aggregate (see
    the module docstring matrix).

    Args:
        rule_data: dataset of the rule chain -- ``returns`` +
            ``signal`` series (same index), optional ``features`` /
            ``mom7`` / ``mom30`` / ``hypothesis``, optional
            ``panel_signal`` + ``panel_returns`` DataFrames for the
            panel-only mode; passed to ``run_full_pipeline`` as-is.
        ml_dataset: dataset of the ML chain or ``None`` to skip it --
            ``features`` (DataFrame, chronological rows), ``target``
            (0/1 labels), ``exec_returns`` (costs already netted),
            ``times_us`` (int64 microsecond stamps); passed to
            ``run_ml_kill_protocol`` as-is.  Its ``battery_controls``
            argument is not re-exposed here: callers that need the
            incremental battery comparator call the engine directly.
        config: overrides over the rule chain ``DEFAULT_CONFIG``
            (merged key-wise, forwarded untouched).
        ml_config: overrides over the ML engine
            ``DEFAULT_TRAINING_CONFIG`` (merged key-wise, forwarded
            untouched).

    Returns:
        One dict: ``{"rule": <rule report>, "ml": <ml report or
        None>, "unified": {...}}`` where ``unified`` carries
        ``verdict`` (``KILLED`` / ``INCONCLUSIVE`` / ``SURVIVES``),
        ``rule_verdict``, ``ml_verdict`` (``None`` when the ML chain
        did not run), ``source`` (which chain decided: ``rule`` /
        ``ml`` / ``rule+ml`` / ``None``) and ``reason``.

    Raises:
        ValueError: from the dataset validators, BEFORE any engine
            runs -- when ``rule_data`` lacks both the series pair and
            the panel pair, or ``ml_dataset`` lacks any of its four
            contract keys.

    Example::

        >>> from atropos.kill_chain import run_full_kill_chain
        >>> noise = run_full_kill_chain(_demo_rule_world("noise"), None)
        >>> noise["unified"]["verdict"], noise["unified"]["source"]
        ('KILLED', 'rule')
        >>> noise["ml"] is None
        True
        >>> full = run_full_kill_chain(
        ...     _demo_rule_world("genuine"),
        ...     _demo_ml_world("planted"),
        ...     ml_config=_DEMO_ML_SURVIVES_CONFIG,
        ... )
        >>> full["unified"]["verdict"]
        'SURVIVES'
    """
    _validate_rule_data(rule_data)
    if ml_dataset is not None:
        _validate_ml_dataset(ml_dataset)
    rule_report = run_full_pipeline(rule_data, config)
    ml_report: dict[str, Any] | None = None
    if ml_dataset is not None:
        ml_report = run_ml_kill_protocol(
            ml_dataset["features"],
            ml_dataset["target"],
            ml_dataset["exec_returns"],
            ml_dataset["times_us"],
            config=ml_config,
        )
    ml_verdict = str(ml_report["verdict"]) if ml_report is not None else None
    unified = _unify(rule_report["overall_verdict"], ml_verdict)
    return {"rule": rule_report, "ml": ml_report, "unified": unified}


def _demo_rule_world(
    kind: str, n_obs: int = 900, seed: int = 4
) -> dict[str, Any]:
    """Synthetic rule-chain dataset: ``'genuine'`` or ``'noise'``.

    ``'genuine'`` is the persistent AR(0.5) order-flow driver with a
    momentum bleed in the signal and in the returns (the recipe of
    ``ensemble_pipeline``'s demo genuine edge): the pipeline lets it
    through.  ``'noise'`` is independent noise in both series: stage
    1 kills it as ``REPACKAGING``.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    if kind == "genuine":
        momentum = np.zeros(n_obs)
        momentum_shocks = rng.standard_normal(n_obs)
        driver = np.zeros(n_obs)
        driver_shocks = rng.standard_normal(n_obs)
        for position in range(1, n_obs):
            momentum[position] = (
                0.5 * momentum[position - 1] + momentum_shocks[position]
            )
            driver[position] = (
                0.5 * driver[position - 1] + driver_shocks[position]
            )
        momentum /= float(np.std(momentum))
        driver /= float(np.std(driver))
        signal = driver + 0.3 * momentum
        forward = driver + 0.6 * momentum + 0.55 * rng.standard_normal(n_obs)
        mom_noisy = momentum + 0.10 * rng.standard_normal(n_obs)
        mom12 = np.concatenate(
            [np.full(12, np.nan), momentum[:-12]]
        ) + 0.10 * rng.standard_normal(n_obs)
        return {
            "returns": pd.Series(forward, index=index),
            "signal": pd.Series(signal, index=index),
            "features": pd.DataFrame(
                {"mom_noisy": mom_noisy, "mom12": mom12}, index=index
            ),
            "hypothesis": "kill_chain demo: persistent driver edge",
        }
    if kind == "noise":
        return {
            "returns": pd.Series(
                0.01 * rng.standard_normal(n_obs), index=index
            ),
            "signal": pd.Series(rng.standard_normal(n_obs), index=index),
            "hypothesis": "kill_chain demo: pure noise",
        }
    raise ValueError(f"unknown demo kind {kind!r}")


def _demo_ml_world(kind: str, seed: int = 0) -> dict[str, Any]:
    """Synthetic ML dataset: ``'planted'``, ``'noise'`` or ``'thin'``.

    The engine's own demo recipes at demo scale: ``'planted'`` carries
    a quadrant-interaction edge (SURVIVES under
    ``_DEMO_ML_SURVIVES_CONFIG``), ``'noise'`` is powered symmetric
    noise (DEAD(cost_line) under ``_DEMO_ML_DEAD_CONFIG``),
    ``'thin'`` is the planted edge under a floor no run can meet
    (UNDERPOWERED under ``_DEMO_ML_UNDERPOWERED_CONFIG``).
    """
    if kind == "planted":
        n_rows = 600
    elif kind == "noise":
        n_rows = 1500
    elif kind == "thin":
        n_rows = 600
    else:
        raise ValueError(f"unknown demo kind {kind!r}")
    rng = np.random.default_rng(seed)
    raw_features = rng.standard_normal((n_rows, 4))
    noise_term = 0.0024 * rng.standard_normal(n_rows)
    if kind == "noise":
        exec_returns = 0.008 * np.sign(rng.standard_normal(n_rows))
    else:
        exec_returns = (
            0.012 * np.tanh(raw_features[:, 0] * raw_features[:, 1] / 2.0)
            + 0.0035 * raw_features[:, 2]
            + noise_term
        )
    feature_frame = pd.DataFrame(
        {
            "f1": raw_features[:, 0],
            "f2": raw_features[:, 1],
            "f3": raw_features[:, 2],
            "f4": raw_features[:, 3],
        }
    )
    stamps = np.arange(n_rows, dtype=np.int64) * 3_600_000_000
    labels = (exec_returns > 0.004).astype(np.int64)
    return {
        "features": feature_frame,
        "target": labels,
        "exec_returns": exec_returns,
        "times_us": stamps,
    }


def run_all_examples() -> None:
    """Verify the unified wrapper on four synthetic cases (asserted).

    Case 1: rule-only noise -- unified KILLED from the rule chain, no
    ML report.  Case 2: genuine rule edge + noise ML dataset --
    unified KILLED from the ML chain.  Case 3: genuine everywhere --
    unified SURVIVES.  Case 4: genuine rule edge + underpowered ML
    dataset -- unified INCONCLUSIVE (undecided is not death).
    """
    print("=" * 78)
    print("CASE 1: rule-only noise (must be unified KILLED, source rule)")
    print("=" * 78)
    rule_only = run_full_kill_chain(_demo_rule_world("noise"), None)
    assert rule_only["rule"]["overall_verdict"] == "REPACKAGING"
    assert rule_only["ml"] is None
    assert rule_only["unified"]["verdict"] == "KILLED"
    assert rule_only["unified"]["source"] == "rule"
    print(
        f"rule={rule_only['rule']['overall_verdict']}  "
        f"unified={rule_only['unified']['verdict']} "
        f"({rule_only['unified']['reason']})"
    )

    print()
    print("=" * 78)
    print("CASE 2: genuine rule edge + noise ML (unified KILLED, source ml)")
    print("=" * 78)
    ml_killed = run_full_kill_chain(
        _demo_rule_world("genuine"),
        _demo_ml_world("noise"),
        ml_config=_DEMO_ML_DEAD_CONFIG,
    )
    assert ml_killed["rule"]["overall_verdict"] == "SURVIVES"
    assert ml_killed["ml"] is not None
    assert ml_killed["ml"]["verdict"] == "DEAD(cost_line)"
    assert ml_killed["unified"]["verdict"] == "KILLED"
    assert ml_killed["unified"]["source"] == "ml"
    print(
        f"rule={ml_killed['rule']['overall_verdict']}  "
        f"ml={ml_killed['ml']['verdict']}  "
        f"unified={ml_killed['unified']['verdict']} "
        f"({ml_killed['unified']['reason']})"
    )

    print()
    print("=" * 78)
    print("CASE 3: genuine rule edge + planted ML (unified SURVIVES)")
    print("=" * 78)
    full = run_full_kill_chain(
        _demo_rule_world("genuine"),
        _demo_ml_world("planted"),
        ml_config=_DEMO_ML_SURVIVES_CONFIG,
    )
    assert full["rule"]["overall_verdict"] == "SURVIVES"
    assert full["ml"] is not None
    assert full["ml"]["verdict"] == "SURVIVES"
    assert full["unified"]["verdict"] == "SURVIVES"
    assert full["unified"]["source"] is None
    print(
        f"rule={full['rule']['overall_verdict']}  "
        f"ml={full['ml']['verdict']}  "
        f"unified={full['unified']['verdict']}"
    )

    print()
    print("=" * 78)
    print(
        "CASE 4: genuine rule edge + underpowered ML (unified INCONCLUSIVE)"
    )
    print("=" * 78)
    undecided = run_full_kill_chain(
        _demo_rule_world("genuine"),
        _demo_ml_world("thin"),
        ml_config=_DEMO_ML_UNDERPOWERED_CONFIG,
    )
    assert undecided["rule"]["overall_verdict"] == "SURVIVES"
    assert undecided["ml"] is not None
    assert undecided["ml"]["verdict"] == "UNDERPOWERED"
    assert undecided["unified"]["verdict"] == "INCONCLUSIVE"
    assert undecided["unified"]["source"] == "ml"
    print(
        f"rule={undecided['rule']['overall_verdict']}  "
        f"ml={undecided['ml']['verdict']}  "
        f"unified={undecided['unified']['verdict']} "
        f"({undecided['unified']['reason']})"
    )
    print()
    print("kill_chain demo: all four unified-verdict cases asserted.")


if __name__ == "__main__":
    run_all_examples()
