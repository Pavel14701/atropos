"""Contracts for the unified kill-chain wrapper (``atropos.kill_chain``).

Covers the unified verdict matrix (a kill in either chain dominates;
``UNDERPOWERED`` is undecided, not death; a rule ``INCONCLUSIVE`` is
undecided against EVERY ml outcome, an ml ``DEAD`` included; plain
``SURVIVES`` needs every executed chain alive), pass-through fidelity
of both engine reports, bit-identical determinism of repeated runs,
dataset-key validation (``ValueError`` before any engine runs) and the
panel-only rule mode the wrapper must not block.  All worlds are
deterministic synthetic constructions with fixed seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos.ensemble_pipeline import _make_synthetic, run_full_pipeline
from atropos.kill_chain import (
    _unify,
    run_all_examples,
    run_full_kill_chain,
)
from atropos.ml_training import run_ml_kill_protocol

DEAD_CONFIG: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"logit": {"max_iter": 500}},
}
SURVIVES_CONFIG: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 100,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"gbm": {"n_estimators": 25}},
}
UNDERPOWERED_CONFIG: dict[str, Any] = {
    **SURVIVES_CONFIG,
    "n_min_events": 100_000,
}

UNIFIED_KEYS = frozenset(
    {"verdict", "rule_verdict", "ml_verdict", "source", "reason"}
)


def _equivalent(
    left: Any,
    right: Any,
    skip_keys: frozenset[str] = frozenset(),
) -> bool:
    """Deep equality that understands dicts, lists, frames, arrays."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(
            key in skip_keys or _equivalent(value, right[key], skip_keys)
            for key, value in left.items()
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equivalent(one, other, skip_keys)
            for one, other in zip(left, right, strict=True)
        )
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        return pd.DataFrame(left).equals(pd.DataFrame(right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


def _rule_world(kind: str) -> dict[str, Any]:
    """Rule dataset: 'genuine' (SURVIVES) or 'noise' (REPACKAGING)."""
    if kind == "genuine":
        return _make_synthetic("genuine_edge", seed=4)
    rng = np.random.default_rng(9)
    index = pd.date_range("2022-01-03", periods=360, freq="B")
    return {
        "returns": pd.Series(0.01 * rng.standard_normal(360), index=index),
        "signal": pd.Series(rng.standard_normal(360), index=index),
        "hypothesis": "pure noise",
    }


def _ml_world(kind: str) -> dict[str, Any]:
    """ML dataset: 'planted', 'noise' (DEAD) or 'thin' (UNDERPOWERED)."""
    rng = np.random.default_rng(0)
    n_rows = 1500 if kind == "noise" else 600
    raw = rng.standard_normal((n_rows, 4))
    noise_term = 0.0024 * rng.standard_normal(n_rows)
    if kind == "noise":
        exec_returns = 0.008 * np.sign(rng.standard_normal(n_rows))
    else:
        exec_returns = (
            0.012 * np.tanh(raw[:, 0] * raw[:, 1] / 2.0)
            + 0.0035 * raw[:, 2]
            + noise_term
        )
    return {
        "features": pd.DataFrame(
            {
                "f1": raw[:, 0],
                "f2": raw[:, 1],
                "f3": raw[:, 2],
                "f4": raw[:, 3],
            }
        ),
        "target": (exec_returns > 0.004).astype(np.int64),
        "exec_returns": exec_returns,
        "times_us": np.arange(n_rows, dtype=np.int64) * 3_600_000_000,
    }


# ---------------------------------------------------------------------------
# unified verdict matrix
# ---------------------------------------------------------------------------
def test_rule_survives_ml_dead_unifies_killed() -> None:
    """Rule SURVIVES + ml DEAD(<gate>) -> KILLED, source ml."""
    result = run_full_kill_chain(
        _rule_world("genuine"),
        _ml_world("noise"),
        ml_config=DEAD_CONFIG,
    )
    assert result["rule"]["overall_verdict"] == "SURVIVES"
    assert result["ml"] is not None
    assert result["ml"]["verdict"].startswith("DEAD")
    assert result["unified"]["verdict"] == "KILLED"
    assert result["unified"]["source"] == "ml"


def test_rule_dead_ml_survives_unifies_killed() -> None:
    """Rule kill verdict + ml SURVIVES -> KILLED, source rule."""
    result = run_full_kill_chain(
        _rule_world("noise"),
        _ml_world("planted"),
        ml_config=SURVIVES_CONFIG,
    )
    assert result["rule"]["overall_verdict"] == "REPACKAGING"
    assert result["ml"] is not None
    assert result["ml"]["verdict"] == "SURVIVES"
    assert result["unified"]["verdict"] == "KILLED"
    assert result["unified"]["source"] == "rule"


def test_both_survive_unify_survives() -> None:
    """Rule alive + ml SURVIVES -> SURVIVES with no deciding source."""
    result = run_full_kill_chain(
        _rule_world("genuine"),
        _ml_world("planted"),
        ml_config=SURVIVES_CONFIG,
    )
    assert result["rule"]["overall_verdict"] == "SURVIVES"
    assert result["ml"] is not None
    assert result["ml"]["verdict"] == "SURVIVES"
    assert result["unified"]["verdict"] == "SURVIVES"
    assert result["unified"]["source"] is None
    assert result["unified"]["reason"] is None


def test_both_dead_unifies_killed_from_both() -> None:
    """Two dead chains -> KILLED attributed to rule+ml."""
    result = run_full_kill_chain(
        _rule_world("noise"),
        _ml_world("noise"),
        ml_config=DEAD_CONFIG,
    )
    assert result["rule"]["overall_verdict"] == "REPACKAGING"
    assert result["ml"] is not None
    assert result["ml"]["verdict"].startswith("DEAD")
    assert result["unified"]["verdict"] == "KILLED"
    assert result["unified"]["source"] == "rule+ml"


@pytest.mark.parametrize(
    ("rule_kind", "expected"),
    [("genuine", "SURVIVES"), ("noise", "KILLED")],
)
def test_ml_none_aggregates_rule_only(rule_kind: str, expected: str) -> None:
    """Without ml_dataset the rule verdict alone decides."""
    result = run_full_kill_chain(_rule_world(rule_kind), None)
    assert result["ml"] is None
    assert result["unified"]["ml_verdict"] is None
    assert result["unified"]["verdict"] == expected
    if expected == "KILLED":
        assert result["unified"]["source"] == "rule"


def test_ml_underpowered_is_inconclusive_not_dead() -> None:
    """ML UNDERPOWERED with nothing dead -> INCONCLUSIVE, source ml."""
    result = run_full_kill_chain(
        _rule_world("genuine"),
        _ml_world("thin"),
        ml_config=UNDERPOWERED_CONFIG,
    )
    assert result["rule"]["overall_verdict"] == "SURVIVES"
    assert result["ml"] is not None
    assert result["ml"]["verdict"] == "UNDERPOWERED"
    assert result["unified"]["verdict"] == "INCONCLUSIVE"
    assert result["unified"]["source"] == "ml"
    assert result["unified"]["reason"] is not None


@pytest.mark.parametrize(
    ("ml_verdict", "expected_source"),
    [
        (None, "rule"),
        ("SURVIVES", "rule"),
        ("DEAD(cost_line)", "rule"),
        ("UNDERPOWERED", "rule"),
    ],
)
def test_rule_inconclusive_dominates_every_ml_outcome(
    ml_verdict: str | None, expected_source: str
) -> None:
    """Rule INCONCLUSIVE x every ml cell -> INCONCLUSIVE [rule].

    Matrix rows "INCONCLUSIVE | (not run)" and "INCONCLUSIVE | any
    decision": a walk-forward configuration the data cannot carry
    (ATR-06) is undecided, never KILLED -- not even against an ml
    ``DEAD``.

    Regression: ``INCONCLUSIVE`` was missing from the rule
    alive-verdict set, so all four cells unified to KILLED -- three as
    ``KILLED [rule]`` and the ml-DEAD cell as ``KILLED [rule+ml]`` --
    and the ATR-06 branch of ``_unify`` was unreachable dead code.
    """
    result = _unify("INCONCLUSIVE", ml_verdict)
    assert frozenset(result) == UNIFIED_KEYS
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["rule_verdict"] == "INCONCLUSIVE"
    assert result["ml_verdict"] == ml_verdict
    assert result["source"] == expected_source
    assert result["reason"] is not None
    assert "undecided" in result["reason"]


def test_rule_stage4_walkforward_oversized_train_is_inconclusive() -> None:
    """E2E: a stage-4 config that cannot fit -> unified INCONCLUSIVE.

    A ``train_window`` no dataset can carry makes the rule pipeline's
    stage 4 report INCONCLUSIVE (ATR-06: the configuration, not the
    edge, failed), and the unified verdict must agree with source
    ``rule`` -- undecided is not death.

    Regression: the wrapper read the rule ``INCONCLUSIVE`` as a kill
    and unified to ``KILLED [rule]``, contradicting the module's own
    matrix row "INCONCLUSIVE | (not run) | INCONCLUSIVE [rule]".
    """
    result = run_full_kill_chain(
        _rule_world("genuine"),
        None,
        config={"walkforward": {"train_window": 100_000_000}},
    )
    assert result["rule"]["overall_verdict"] == "INCONCLUSIVE"
    assert result["rule"]["kill_reason"] == (
        "insufficient data for walk-forward configuration"
    )
    assert result["ml"] is None
    assert result["unified"]["verdict"] == "INCONCLUSIVE"
    assert result["unified"]["source"] == "rule"
    assert "undecided" in result["unified"]["reason"]


def test_unified_namespace_keys_are_pinned() -> None:
    """The unified namespace carries exactly the documented keys."""
    result = run_full_kill_chain(_rule_world("noise"), None)
    assert frozenset(result["unified"]) == UNIFIED_KEYS
    assert frozenset(result) == {"rule", "ml", "unified"}


# ---------------------------------------------------------------------------
# thin-wrapper fidelity
# ---------------------------------------------------------------------------
def test_reports_pass_through_verbatim() -> None:
    """rule/ml reports equal the direct engine calls (modulo runtime)."""
    rule_data = _rule_world("genuine")
    ml_data = _ml_world("planted")
    result = run_full_kill_chain(
        rule_data, ml_data, ml_config=SURVIVES_CONFIG
    )
    direct_rule = run_full_pipeline(dict(rule_data), None)
    direct_ml = run_ml_kill_protocol(
        ml_data["features"],
        ml_data["target"],
        ml_data["exec_returns"],
        ml_data["times_us"],
        config=dict(SURVIVES_CONFIG),
    )
    assert _equivalent(
        result["rule"], direct_rule, skip_keys=frozenset({"runtime_seconds"})
    )
    assert _equivalent(result["ml"], direct_ml)


def test_kill_chain_is_deterministic() -> None:
    """Same inputs twice: bit-identical aggregate and engine reports."""
    first = run_full_kill_chain(
        _rule_world("genuine"),
        _ml_world("planted"),
        ml_config=SURVIVES_CONFIG,
    )
    second = run_full_kill_chain(
        _rule_world("genuine"),
        _ml_world("planted"),
        ml_config=SURVIVES_CONFIG,
    )
    assert _equivalent(
        first, second, skip_keys=frozenset({"runtime_seconds"})
    )
    assert first["unified"] == second["unified"]


def test_demo_self_verifies() -> None:
    """The bundled demo asserts all four unified verdict cases."""
    run_all_examples()


# ---------------------------------------------------------------------------
# dataset validation: ValueError before any engine runs
# ---------------------------------------------------------------------------
def test_missing_rule_keys_raise_valueerror() -> None:
    """Empty rule_data fails loudly naming the rule contract."""
    with pytest.raises(ValueError, match="rule_data"):
        run_full_kill_chain({}, None)


def test_partial_rule_keys_raise_valueerror() -> None:
    """Having returns without signal (nor panels) is a ValueError."""
    partial = {"returns": pd.Series(np.ones(400))}
    with pytest.raises(ValueError, match="signal"):
        run_full_kill_chain(partial, None)


def test_missing_ml_keys_raise_valueerror() -> None:
    """An incomplete ml_dataset names every missing contract key."""
    incomplete = {"features": pd.DataFrame(np.ones((10, 2)))}
    with pytest.raises(ValueError, match="ml_dataset"):
        run_full_kill_chain(_rule_world("noise"), incomplete)


def test_empty_ml_dataset_reports_all_missing_keys() -> None:
    """{} carries none of the four keys; the message lists them all."""
    with pytest.raises(ValueError, match="exec_returns") as excinfo:
        run_full_kill_chain(_rule_world("noise"), {})
    message = str(excinfo.value)
    for key in ("features", "target", "exec_returns", "times_us"):
        assert key in message


def test_panel_only_rule_data_runs_without_ml() -> None:
    """Panel pair alone is a valid rule dataset: no rule ValueError."""
    rng = np.random.default_rng(5)
    latent = rng.standard_normal((60, 12))
    panel_data = {
        "panel_signal": pd.DataFrame(latent),
        "panel_returns": pd.DataFrame(
            0.004 * latent + 0.01 * rng.standard_normal((60, 12))
        ),
        "hypothesis": "panel-only mode",
    }
    result = run_full_kill_chain(panel_data, None)
    rule_verdict = result["rule"]["overall_verdict"]
    assert rule_verdict == result["unified"]["rule_verdict"]
    if rule_verdict.startswith("SURVIVES"):
        assert result["unified"]["verdict"] == "SURVIVES"
    else:
        assert result["unified"]["verdict"] == "KILLED"
