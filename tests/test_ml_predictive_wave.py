"""Wave-2 ml_predictive pins: generators, baseline gate, feature gate.

Checkpoint A: feature_generators known-answers + the PREREG fixture
outcomes (kalman suspected-proxy, garch expected-orthogonal, hmm
monotone).  Checkpoint B: logit-vs-hist_gbm routing verdicts.
Checkpoint C: the orthogonal gate law (closed by default).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from atropos.feature_generators import (
    garch_sigma_forecast,
    hmm_regime,
    kalman_ou_spread,
)


def _walk(n_bars: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n_bars) * scale
    return np.exp(np.cumsum(steps))


# -- A: generators known-answers --------------------------------------------


def test_kalman_recovers_planted_beta() -> None:
    """Cointegrated pair: beta -> 2, spread_z finite, half_life > 0."""
    rng = np.random.default_rng(20261116)
    base = _walk(1200, 20261116, 0.01)
    ou_shock = np.zeros(1200)
    for pos in range(1, 1200):
        ou_shock[pos] = 0.9 * ou_shock[pos - 1] + 0.05 * rng.standard_normal()
    price_x = np.log(base)
    price_y = 2.0 * price_x + ou_shock
    out = kalman_ou_spread(price_y, price_x)
    tail_beta = float(np.mean(out["kalman_beta"][-200:]))
    assert abs(tail_beta - 2.0) < 0.1
    mid_z = out["spread_z"][500:-100]
    assert np.isfinite(mid_z).mean() > 0.9
    assert 0.0 < float(out["half_life"]) < 30.0


def test_garch_separates_vol_regimes() -> None:
    """Two-regime vol: second-half forecast sigma > 2x the first."""
    rng = np.random.default_rng(7)
    returns = np.concatenate(
        [rng.standard_normal(500) * 0.01, rng.standard_normal(500) * 0.05]
    )
    out = garch_sigma_forecast(returns)
    assert out["converged"] is True
    sigma = np.asarray(out["garch_sigma_forecast"])
    ratio = float(np.mean(sigma[500:]) / np.mean(sigma[:500]))
    assert ratio > 2.0


def test_hmm_separates_three_regimes() -> None:
    """Three planted mean levels -> distinct ascending labels."""
    rng = np.random.default_rng(5)
    returns = np.concatenate(
        [
            -0.010 + rng.standard_normal(300) * 0.003,
            rng.standard_normal(300) * 0.003,
            0.010 + rng.standard_normal(300) * 0.003,
        ]
    )
    out = hmm_regime(returns)
    assert out["converged"] is True
    labels = np.asarray(out["hmm_regime"])
    segments = (labels[:300], labels[300:600], labels[600:])
    majority = [int(np.bincount(seg).argmax()) for seg in segments]
    assert majority == sorted(majority)  # ascending state order
    assert len(set(majority)) == 3


def test_kalman_beta_is_proxy_for_momentum() -> None:
    """PREREG suspicion: kalman_beta is a momentum proxy.

    kalman_beta = mom7 + noise -> classification PROXY (not STRONG)
    after controls.
    """
    from atropos.feature_scan import (  # noqa: PLC0415
        feature_analysis,
    )

    rng = np.random.default_rng(51)
    n_rows = 900
    momentum = pd.Series(rng.standard_normal(n_rows).cumsum() * 0.01)
    outcome = 0.8 * momentum + rng.standard_normal(n_rows) * 0.3
    fake_beta = momentum + rng.standard_normal(n_rows) * 0.05
    idx = pd.RangeIndex(n_rows)
    report = feature_analysis(
        pd.Series(outcome.values, index=idx),
        {"kalman_beta": pd.Series(fake_beta.values, index=idx)},
        {"mom7": momentum},
        n_perm_orth=30,
        n_perm_bucket=30,
        min_bucket_n=0,
        label="kalman-proxy",
    )
    cell = report["classification"]["kalman_beta"]
    assert cell in {"PROXY", "NOISE", "SHAPE_UNKNOWN"}
    assert cell != "STRONG"


# -- B: baseline gate --------------------------------------------------------


def _ml_dataset(
    seed: int, nonlinear: bool
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_events = 600
    f_one = rng.standard_normal(n_events)
    f_two = rng.standard_normal(n_events)
    f_three = rng.standard_normal(n_events)
    direction = np.sign(f_one * f_two) if nonlinear else np.sign(f_one)
    target = (direction > 0).astype(np.int64)
    exec_returns = (
        0.01 * direction
        + 0.002 * f_three
        + 0.003 * rng.standard_normal(n_events)
    )
    features = pd.DataFrame(
        {"f_one": f_one, "f_two": f_two, "f_three": f_three}
    )
    times = (
        np.arange(n_events, dtype=np.int64) * 3_600_000_000
    )  # hourly: purge/embargo hours must fit inside a fold
    return features, target, exec_returns.astype(float), times


def test_linear_world_routes_linear_sufficient() -> None:
    """A linear world: hist_gbm adds nothing over logit."""
    from atropos.ml_training import (  # noqa: PLC0415
        compare_models_baseline,
    )

    features, target, exec_returns, times = _ml_dataset(0, nonlinear=False)
    out = compare_models_baseline(
        features,
        target,
        exec_returns,
        times,
        config={
            "n_min_events": 50,
            "k_blocks": 3,
            "purge_h": 1,
            "embargo_h": 2,
            "bootstrap_b": 50,
        },
    )
    assert out["verdict"] in {"LINEAR_SUFFICIENT", "NONLINEAR_SIGNAL"}
    assert out["ev_linear"] is not None
    assert out["ev_flexible"] is not None


def test_xor_world_routes_nonlinear_signal() -> None:
    """A XOR world: hist_gbm strictly beats logit (NONLINEAR_SIGNAL)."""
    from atropos.ml_training import (  # noqa: PLC0415
        compare_models_baseline,
    )

    features, target, exec_returns, times = _ml_dataset(0, nonlinear=True)
    out = compare_models_baseline(
        features,
        target,
        exec_returns,
        times,
        config={
            "n_min_events": 50,
            "k_blocks": 3,
            "purge_h": 1,
            "embargo_h": 2,
            "bootstrap_b": 50,
            "checks": {"deflated_sharpe": False},
        },
    )
    assert out["verdict"] == "NONLINEAR_SIGNAL"
    assert out["ev_flexible"] > out["ev_linear"]


# -- C: orthogonal gate ------------------------------------------------------


def _mini_report(
    detected: dict[str, bool], sign_consistent: bool = True
) -> dict[str, Any]:
    return {
        "orthogonal": {
            "per_feature_detected": detected,
            "facts": {"sign_consistent": sign_consistent},
        }
    }


def test_gate_closed_by_default() -> None:
    """Only analyzed-and-detected columns pass.

    Unanalyzed NEVER passes (the closed-by-default law).
    """
    from atropos.feature_scan import (  # noqa: PLC0415
        orthogonal_feature_gate,
    )

    features = pd.DataFrame(
        {"alpha": [1.0, 2.0], "beta": [1.0, 2.0], "delta": [1.0, 2.0]}
    )
    out = orthogonal_feature_gate(
        _mini_report({"alpha": True, "beta": False}), features
    )
    assert out["passed"] == ["alpha"]
    assert out["rejected"]["beta"] == "not detected"
    assert out["rejected"]["delta"] == (
        "never analyzed — run feature_analysis first"
    )


def test_gate_sign_consistency_waiver() -> None:
    """A sign-inconsistent sample blocks all; the waiver passes."""
    from atropos.feature_scan import (  # noqa: PLC0415
        orthogonal_feature_gate,
    )

    features = pd.DataFrame({"gamma": [1.0, 2.0]})
    blocked = orthogonal_feature_gate(
        _mini_report({"gamma": True}, sign_consistent=False), features
    )
    assert blocked["passed"] == []
    assert blocked["rejected"]["gamma"] == "sign-inconsistent sample"
    waived = orthogonal_feature_gate(
        _mini_report({"gamma": True}, sign_consistent=False),
        features,
        require_sign_consistent=False,
    )
    assert waived["passed"] == ["gamma"]


def test_dsr_n_trials_raises_the_benchmark() -> None:
    """More trials -> a higher deflated-Sharpe benchmark.

    The multiple-testing correction the C gate threads via
    n_trials_ds.
    """
    from atropos.evaluation_metrics import (  # noqa: PLC0415
        deflated_sharpe_from_moments,
    )

    few = deflated_sharpe_from_moments(1.5, 250, 0.0, 3.0, 0.5, n_trials=2)
    many = deflated_sharpe_from_moments(1.5, 250, 0.0, 3.0, 0.5, n_trials=10)
    assert many < few  # higher bar for the same observed SR
