"""Chaos tests: corrupted input must yield refusal or sane output.

The contract under test: on ANY corrupted input a public function must
either (a) raise a documented ``ValueError`` / ``TypeError`` (a loud,
messaged refusal), or (b) return a sane payload: finite numbers, honest
NaN / None markers, or an ``inf`` that is part of that function's own
documented contract.  Never acceptable: plausible-looking finite
garbage, an undocumented ``inf`` fabricated by intermediate float
overflow while the true value is representable, or an opaque
third-party crash such as ``LinAlgError: SVD did not converge``.

Legal infs encoded per function (assumptions of this layer, each taken
from the function's own docstring or from exact float overflow):

* ``calmar_ratio`` -- ``calmar = inf`` for a zero-drawdown curve with
  positive CAGR (docstring); ``cagr`` may overflow to inf only when
  the true compounded growth exceeds float range.
* ``profit_factor`` -- ``pf = inf`` for losses-free positive trades
  (docstring); ``gross_win`` / ``gross_loss`` may overflow to inf only
  when the true gross sums exceed float range.
* ``turnover`` / ``turnover_series`` -- with positions of magnitude
  1e308 the true per-period change (2e308 and beyond) exceeds float
  range, so ``inf`` is the correctly-rounded honest result.
* ``breakeven_cost_bps`` / ``cost_death_test`` --
  ``breakeven_cost_bps = inf`` for a positive-edge strategy that never
  trades (docstring); ``annual_turnover`` may overflow honestly.
* NaN is allowed everywhere: every insufficient-data scaffold in this
  package honestly reports NaN instead of a fabricated number.

Contract violations found while writing this layer are frozen as
``xfail(strict=True)`` tests named ``test_finding_*``; the mark reason
carries the function, the corruption and why the output is garbage.

Corruption factories (shared, parametrized): all-NaN, all-inf,
NaN-through-one, alternating +/-inf, ``*1e308`` overflow, ``*1e-320``
denormal, duplicated concatenation, sorted monotone, empty,
single element, all zeros, one huge point outlier (1e12), one crash
outlier (-1.5), object dtype with None.  String arrays, negative
sizes and invalid confidence levels live in dedicated refusal tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest
from atropos._utils import block_indices, clean_1d, ols_hc0
from atropos.bootstrap_ci import (
    block_bootstrap_ci,
    percentile_ci,
    stationary_bootstrap_ci,
)
from atropos.cost_model import (
    apply_proportional_costs,
    breakeven_cost_bps,
    cost_death_test,
    turnover_series,
)
from atropos.evaluation_metrics import (
    calmar_ratio,
    deflated_sharpe,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    turnover,
    win_rate,
)
from atropos.incremental_regression import incremental_test
from atropos.ml_training import (
    decluster_count,
    deflated_sharpe_value,
    ev_of_taken,
    fold_edges,
    pick_threshold,
    purge_embargo_mask,
    rolling_z,
    run_ml_kill_protocol,
    spearman_median,
    stream_stats,
)
from atropos.monte_carlo import (
    bootstrap_resample,
    max_drawdown_distribution,
    monte_carlo_paths,
    sharpe_distribution,
)
from atropos.null_control_suite import known_features_control
from atropos.walkforward_framework import walkforward

# ---------------------------------------------------------------------------
# deterministic base data (fixed seeds; every result reproducible)
# ---------------------------------------------------------------------------
_CHAOS_RNG = np.random.default_rng(20261103)
BASE_RETURNS = _CHAOS_RNG.normal(0.0005, 0.01, 64)
BASE_TRADES = _CHAOS_RNG.normal(0.5, 2.0, 64)
BASE_POSITIONS = np.tile(np.array([1.0, -1.0]), 32)
BASE_EQUITY = np.cumprod(1.0 + BASE_RETURNS)
_STR_VALS = np.array(["not_a_number"] * 8)


# ---------------------------------------------------------------------------
# corruption factories
# ---------------------------------------------------------------------------
def _nan_alternating(base: np.ndarray) -> np.ndarray:
    """NaN at every odd position, base values at even ones."""
    out = base.astype(float).copy()
    out[1::2] = np.nan
    return out


def _inf_alternating(base: np.ndarray) -> np.ndarray:
    """+inf at every 4th, -inf at every 4th (offset) position."""
    out = base.astype(float).copy()
    out[::4] = np.inf
    out[1::4] = -np.inf
    return out


def _huge_outlier(base: np.ndarray) -> np.ndarray:
    """One enormous (1e12) point among the normal values."""
    out = base.astype(float).copy()
    out[17] = 1e12
    return out


def _crash_outlier(base: np.ndarray) -> np.ndarray:
    """One catastrophic (-1.5, i.e. a -150% 'return') point."""
    out = base.astype(float).copy()
    out[17] = -1.5
    return out


def _object_none(base: np.ndarray) -> np.ndarray:
    """Object dtype mixing None and floats (None coerces to NaN)."""
    out = np.empty(base.size, dtype=object)
    out[::2] = None
    out[1::2] = base[1::2]
    return out


CorruptionFn = Callable[[np.ndarray], np.ndarray]

CORRUPTIONS: dict[str, CorruptionFn] = {
    "all_nan": lambda base: np.full(base.shape, np.nan),
    "all_inf": lambda base: np.full(base.shape, np.inf),
    "nan_alternating": _nan_alternating,
    "inf_alternating": _inf_alternating,
    "overflow_1e308": lambda base: base * 1e308,
    "denormal_1e_320": lambda base: base * 1e-320,
    "duplicated": lambda base: np.concatenate([base, base]),
    "sorted": np.sort,
    "empty": lambda base: base[:0].copy(),
    "single": lambda base: base[:1].copy(),
    "zeros": lambda base: np.zeros_like(base, dtype=float),
    "huge_outlier": _huge_outlier,
    "crash_outlier": _crash_outlier,
    "object_none": _object_none,
}
CORRUPTION_ORDER: tuple[str, ...] = tuple(CORRUPTIONS)


# ---------------------------------------------------------------------------
# the chaos contract checker
# ---------------------------------------------------------------------------
def _check_number(field: str, value: float, spec: Mapping[str, Any]) -> None:
    """Check one float against the sanity rules of ``spec``."""
    if math.isnan(value):
        return  # honest NaN (insufficient-data scaffolds) is sane
    if math.isinf(value):
        assert field in spec.get("inf_fields", ()), (
            f"field {field!r} is {value}: undocumented inf (fabricated "
            "by overflow or garbage), the chaos contract is violated"
        )
        return
    bounds = spec.get("bounds", {})
    if field in bounds:
        low, high = bounds[field]
        assert low <= value <= high, (
            f"field {field!r} = {value} outside sane bounds [{low}, {high}]"
        )


def _walk_mapping(
    payload: Mapping[Any, Any], spec: Mapping[str, Any]
) -> None:
    """Check required keys, then recurse into every mapped value."""
    required = spec.get("keys")
    if required is not None:
        missing = set(required) - set(payload)
        assert not missing, f"result lacks required keys: {missing}"
    for key, item in payload.items():
        assert_sane_outcome(item, spec, str(key))


def _walk_frame(frame: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    """Recurse into every numeric column of a DataFrame payload."""
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            assert_sane_outcome(frame[column].to_numpy(), spec, str(column))


def assert_sane_outcome(
    result: Any, spec: Mapping[str, Any], field: str = "value"
) -> None:
    """Assert that ``result`` is a sane payload per ``spec``.

    ``spec`` keys (all optional):

    * ``inf_fields`` -- fields where +/-inf is contract-legal;
    * ``bounds`` -- field -> (low, high) for finite float values;
    * ``keys`` -- required keys of a dict result (subset check, so
      implementations may add keys);
    * ``extra`` -- callable for contract-specific semantic checks.

    Walked recursively over dicts, tuples/lists, float ndarrays and
    numeric DataFrame columns; NaN is allowed everywhere.
    """
    if isinstance(result, Mapping):
        _walk_mapping(result, spec)
    elif isinstance(result, pd.DataFrame):
        _walk_frame(result, spec)
    elif isinstance(result, np.ndarray):
        if result.dtype.kind == "f":
            for element in result.reshape(-1).tolist():
                _check_number(field, float(element), spec)
    elif isinstance(result, (tuple, list)):
        for item in result:
            assert_sane_outcome(item, spec, field)
    elif isinstance(result, (float, np.floating)):
        _check_number(field, float(result), spec)
    elif (
        isinstance(result, (bool, np.bool_, int, np.integer, str))
        or result is None
    ):
        pass  # opaque-but-sane members: verdicts, flags, counts, notes
    else:  # pragma: no cover - defensive: unknown payload type
        raise AssertionError(
            f"unexpected result type {type(result).__name__}"
        )


def assert_chaos_contract(
    spec: Mapping[str, Any], call: Callable[[], Any]
) -> None:
    """Run ``call``: a documented refusal passes, else sane payload.

    An exception whose type is listed in ``spec['exceptions']`` counts
    as a documented refusal (its message must be non-empty).
    ``numpy.linalg.LinAlgError`` is NEVER accepted as a refusal even
    though it subclasses ``ValueError``: "SVD did not converge" is an
    opaque LAPACK crash (the exact 'невнятная ошибка' the chaos layer
    exists to catch), not a designed, documented refusal.  Any other
    exception propagates and fails the test -- that is the finding.
    """
    allowed: tuple[type[BaseException], ...] = spec.get("exceptions", ())
    try:
        outcome = call()
    except Exception as exc:
        if isinstance(exc, np.linalg.LinAlgError):
            raise
        if not isinstance(exc, allowed):
            raise
        assert str(exc), "documented refusal must carry a message"
        return
    assert_sane_outcome(outcome, spec)
    extra = spec.get("extra")
    if extra is not None:
        extra(outcome)


# ---------------------------------------------------------------------------
# reusable semantic extras
# ---------------------------------------------------------------------------
def _ci_pair_ordered(outcome: Any) -> None:
    """Tuple CI bounds must be ordered when both are finite."""
    low, high = float(outcome[0]), float(outcome[1])
    if math.isfinite(low) and math.isfinite(high):
        assert low <= high, f"ci bounds out of order: {low} > {high}"


def _dict_ci_ordered(outcome: Any) -> None:
    """Dict fields ci_low / ci_high must be ordered when finite."""
    _ci_pair_ordered((outcome["ci_low"], outcome["ci_high"]))


def _sharpe_ci_ordered(outcome: Any) -> None:
    """sharpe_ratio's sr_ci tuple must be ordered when finite."""
    _ci_pair_ordered(outcome["sr_ci"])


def _wilson_contains_wr(outcome: Any) -> None:
    """The Wilson interval must contain the point win-rate."""
    wr = float(outcome["wr"])
    low = float(outcome["ci_low"])
    high = float(outcome["ci_high"])
    if all(math.isfinite(item) for item in (wr, low, high)):
        msg = f"win-rate {wr} outside CI [{low}, {high}]"
        assert low <= wr <= high, msg


def _triple_ordered(
    low_key: str, mid_key: str, high_key: str
) -> Callable[[Any], None]:
    """Build an ordering check for a (p5, p50, p95) percentile triple."""

    def check_triple(outcome: Any) -> None:
        """Percentiles must be non-decreasing when finite."""
        p5 = float(outcome[low_key])
        p50 = float(outcome[mid_key])
        p95 = float(outcome[high_key])
        if all(math.isfinite(item) for item in (p5, p50, p95)):
            assert p5 <= p50 <= p95, f"percentiles out of order: {outcome}"

    return check_triple


def _paths_add_up(outcome: Any) -> None:
    """Valid plus dropped paths must reassemble the simulated count."""
    total = int(outcome["n_valid_paths"]) + int(outcome["n_dropped_paths"])
    assert total == int(outcome["n_sim"]), "path accounting is broken"


def _verdict_in(allowed: tuple[str, ...]) -> Callable[[Any], None]:
    """Build a verdict-membership check."""

    def check_verdict(outcome: Any) -> None:
        """The verdict must come from the documented enum."""
        verdict = outcome["verdict"]
        assert verdict in allowed, f"verdict {verdict!r} not in {allowed}"

    return check_verdict


def _clean_1d_extras(outcome: Any) -> None:
    """clean_1d must return a flat float array of finite entries only."""
    assert outcome.ndim == 1, f"expected 1-D, got ndim={outcome.ndim}"
    assert outcome.dtype == np.float64
    assert not bool(np.isnan(outcome).any()), "NaN survived clean_1d"


def _mdd_dist_extras(outcome: Any) -> None:
    """Ordering and path accounting for max_drawdown_distribution."""
    _triple_ordered("mdd_p5", "mdd_p50", "mdd_p95")(outcome)
    _paths_add_up(outcome)


def _cost_death_extras(outcome: Any) -> None:
    """Verdict enum plus CI ordering for cost_death_test payloads."""
    _verdict_in(("PASS", "FAIL", "INCONCLUSIVE"))(outcome)
    _dict_ci_ordered(outcome)


def _walkforward_extras(outcome: Any) -> None:
    """Verdict enum for walkforward payloads."""
    _verdict_in(("PASS", "DEGRADED", "FAIL"))(outcome)


# ---------------------------------------------------------------------------
# finding reasons -- all six chaos findings were FIXED after being
# pinned here (scaled std / distributed drag / ols_hc0 finite guards /
# walkforward non-finite guard); the texts stay as the historical
# record of what each regression test protects against
# ---------------------------------------------------------------------------
_F_SE_OVERFLOW = (
    "chaos-1 FINDING: bootstrap of data scaled to ~1e306 returns "
    "se=inf while ci_low/ci_high stay finite (~1e305). The true se is "
    "representable; np.std overflows only in the squared deviations, "
    "so se=inf is undocumented intermediate-overflow garbage, not an "
    "honest inf."
)
_F_WF_LINALG = (
    "chaos-2 FINDING: a NaN in the feature column makes the default "
    "OLS fit raise numpy.linalg.LinAlgError('SVD did not converge in "
    "Linear Least Squares') -- an opaque LAPACK crash (it subclasses "
    "ValueError, but no docstring documents it), neither a designed "
    "refusal nor a sane verdict."
)
_F_NET_INF = (
    "chaos-3 FINDING: apply_proportional_costs with alternating "
    "+-1e308 positions silently returns net=-inf for every rebalanced "
    "period. The true net return (gross - 1e-3 * 2e308 ~ -2e305) is "
    "representable; the -inf leaks from |pos[t]-pos[t-1]| = 2e308 "
    "overflowing float range."
)
_F_DRAG_INF = (
    "chaos-3b FINDING (same turnover-overflow root as chaos-3): "
    "cost_death_test on +-1e308 positions reports cost_drag_mean=inf. "
    "The true mean drag is 1e-3 * 2e308 ~ 2e305, representable; the "
    "inf leaks through net=-inf from the overflowing |pos diff|. The "
    "verdict itself stays honestly INCONCLUSIVE."
)
_F_OLS_DENORMAL = (
    "chaos-4 FINDING: ols_hc0 on a subnormal design (entries ~1e-320) "
    "silently returns beta=[-inf,-inf,-inf] with rank_deficient=False. "
    "OLS is scale-equivariant, so the true beta is finite; the -inf is "
    "LAPACK overflow garbage leaking through an undocumented path."
)
_F_OLS_NAN = (
    "chaos-5 FINDING: ols_hc0 on a design with a NaN column raises "
    "numpy.linalg.LinAlgError('SVD did not converge') instead of the "
    "NaN payload the docstring promises (root cause of chaos-2)."
)

# combos covered by dedicated xfail tests, skipped in the main matrices
_SKIP_OVERFLOW: dict[str, frozenset[str]] = {
    "block_bootstrap_ci": frozenset({"overflow_1e308"}),
    "stationary_bootstrap_ci": frozenset({"overflow_1e308"}),
    "apply_proportional_costs": frozenset({"overflow_1e308"}),
    "cost_death_test": frozenset({"overflow_1e308"}),
}


# ---------------------------------------------------------------------------
# per-function specs and case tables
# ---------------------------------------------------------------------------
Case = dict[str, Any]

_SPEC_SHARPE: dict[str, Any] = {
    "keys": (
        "sr",
        "sr_ci",
        "sr_per_period",
        "ci_method",
        "n",
        "rf",
    ),
    "extra": _sharpe_ci_ordered,
}

_SPEC_DSR: dict[str, Any] = {
    "keys": (
        "dsr",
        "p_value",
        "sr_annual",
        "sr_benchmark_annual",
        "sr_var_annual",
        "sr_var_source",
        "skew",
        "kurtosis",
        "n_trials",
        "n",
    ),
    "bounds": {"dsr": (0.0, 1.0), "p_value": (0.0, 1.0)},
}

_SPEC_MAX_DD: dict[str, Any] = {
    "keys": (
        "mdd",
        "mdd_duration",
        "peak_index",
        "trough_index",
        "still_underwater",
        "n",
    ),
    # documented refusals: ndim > 1 and non-positive equity
    "exceptions": (ValueError,),
    "bounds": {"mdd": (-1.0, 0.0)},
}

_SPEC_CALMAR: dict[str, Any] = {
    "keys": ("calmar", "cagr", "mdd", "n"),
    # ValueError: compounded equity <= 0 is refused by the documented
    # max_drawdown strictly-positive guard (returns <= -1 corrupted in)
    "exceptions": (ValueError,),
    "inf_fields": {"calmar", "cagr"},
    "bounds": {"mdd": (-1.0, 0.0)},
}

_SPEC_WIN_RATE: dict[str, Any] = {
    "keys": ("wr", "ci_low", "ci_high", "ci_level", "n"),
    "bounds": {
        "wr": (0.0, 1.0),
        "ci_low": (0.0, 1.0),
        "ci_high": (0.0, 1.0),
    },
    "extra": _wilson_contains_wr,
}

_SPEC_PROFIT_FACTOR: dict[str, Any] = {
    "keys": (
        "pf",
        "gross_win",
        "gross_loss",
        "n_trades",
        "n_winners",
        "n_losers",
    ),
    "inf_fields": {"pf", "gross_win", "gross_loss"},
    "bounds": {
        "pf": (0.0, math.inf),
        "gross_win": (0.0, math.inf),
        "gross_loss": (0.0, math.inf),
    },
}

_SPEC_TURNOVER: dict[str, Any] = {
    "keys": ("rt_per_year", "turnover_per_period", "n_rebalances", "periods"),
    "inf_fields": {"rt_per_year", "turnover_per_period"},
    "bounds": {
        "turnover_per_period": (0.0, math.inf),
        "rt_per_year": (0.0, math.inf),
    },
}

_EVAL_CASES: dict[str, Case] = {
    "sharpe_ratio": {
        "base": BASE_RETURNS,
        "spec": _SPEC_SHARPE,
        "invoke": lambda values: sharpe_ratio(
            values, n_boot=50, block=8, seed=0
        ),
    },
    "deflated_sharpe": {
        "base": BASE_RETURNS,
        "spec": _SPEC_DSR,
        "invoke": lambda values: deflated_sharpe(values, n_trials=10, seed=0),
    },
    "max_drawdown": {
        "base": BASE_EQUITY,
        "spec": _SPEC_MAX_DD,
        "invoke": max_drawdown,
    },
    "calmar_ratio": {
        "base": BASE_RETURNS,
        "spec": _SPEC_CALMAR,
        "invoke": calmar_ratio,
    },
    "win_rate": {
        "base": BASE_TRADES,
        "spec": _SPEC_WIN_RATE,
        "invoke": win_rate,
    },
    "profit_factor": {
        "base": BASE_TRADES,
        "spec": _SPEC_PROFIT_FACTOR,
        "invoke": profit_factor,
    },
    "turnover": {
        "base": BASE_POSITIONS,
        "spec": _SPEC_TURNOVER,
        "invoke": turnover,
    },
}


_SPEC_PERCENTILE_CI: dict[str, Any] = {"extra": _ci_pair_ordered}

_SPEC_BLOCK_CI: dict[str, Any] = {
    "keys": (
        "point_estimate",
        "ci_low",
        "ci_high",
        "se",
        "n",
        "B",
        "method",
    ),
    "bounds": {"se": (0.0, math.inf)},
    "extra": _dict_ci_ordered,
}

_BOOTSTRAP_CASES: dict[str, Case] = {
    "percentile_ci": {
        "base": BASE_RETURNS,
        "spec": _SPEC_PERCENTILE_CI,
        "invoke": percentile_ci,
    },
    "block_bootstrap_ci": {
        "base": BASE_RETURNS,
        "spec": _SPEC_BLOCK_CI,
        "invoke": lambda values: block_bootstrap_ci(
            values, n_boot=50, block=8, seed=0
        ),
    },
    "stationary_bootstrap_ci": {
        "base": BASE_RETURNS,
        "spec": _SPEC_BLOCK_CI,
        "invoke": lambda values: stationary_bootstrap_ci(
            values, n_boot=50, mean_block=8.0, seed=0
        ),
    },
}


_SPEC_TURNOVER_SERIES: dict[str, Any] = {
    # honest overflow only: true |pos[t]-pos[t-1]| beyond float range
    "inf_fields": {"value"},
    "bounds": {"value": (0.0, math.inf)},
}

_SPEC_APPLY_COSTS: dict[str, Any] = {
    # documented refusals: negative cost_bps, raw length mismatch
    "exceptions": (ValueError,),
}

_SPEC_BREAKEVEN: dict[str, Any] = {
    "keys": ("breakeven_cost_bps",),
    # documented refusal: raw length mismatch
    "exceptions": (ValueError,),
    # documented: inf when a positive-edge strategy never trades
    "inf_fields": {"breakeven_cost_bps"},
}

_SPEC_COST_DEATH: dict[str, Any] = {
    "keys": (
        "net_mean",
        "p_value",
        "verdict",
        "net_sharpe",
        "gross_sharpe",
        "cost_drag_mean",
        "ci_low",
        "ci_high",
        "n_obs",
        "breakeven_cost_bps",
        "annual_turnover",
        "n_changes",
    ),
    # documented refusals: ci_level, periods, cost_bps, length mismatch
    "exceptions": (ValueError,),
    "inf_fields": {"breakeven_cost_bps", "annual_turnover"},
    "extra": _cost_death_extras,
}

_COST_CASES: dict[str, Case] = {
    "turnover_series": {
        "base": BASE_POSITIONS,
        "spec": _SPEC_TURNOVER_SERIES,
        "invoke": turnover_series,
    },
    "apply_proportional_costs": {
        # the corruption hits the POSITIONS argument; gross returns stay
        # sane so any garbage must come through the turnover channel
        "base": BASE_POSITIONS,
        "spec": _SPEC_APPLY_COSTS,
        "invoke": lambda values: apply_proportional_costs(
            BASE_RETURNS, values, 10.0
        ),
    },
    "breakeven_cost_bps": {
        "base": BASE_POSITIONS,
        "spec": _SPEC_BREAKEVEN,
        "invoke": lambda values: breakeven_cost_bps(
            np.abs(BASE_RETURNS), values
        ),
    },
    "cost_death_test": {
        "base": BASE_POSITIONS,
        "spec": _SPEC_COST_DEATH,
        "invoke": lambda values: cost_death_test(
            np.abs(BASE_RETURNS),
            values,
            10.0,
            n_boot=50,
            block=8,
            seed=0,
        ),
    },
}


_SPEC_BOOT_RESAMPLE: dict[str, Any] = {}

_SPEC_MDD_DIST: dict[str, Any] = {
    "keys": (
        "mdd_p5",
        "mdd_p50",
        "mdd_p95",
        "n_valid_paths",
        "n_dropped_paths",
        "n_sim",
        "n",
        "block",
        "note",
    ),
    # documented refusals: < 20 valid returns, n_sim < 1
    "exceptions": (ValueError,),
    "bounds": {
        "mdd_p5": (-1.0, 0.0),
        "mdd_p50": (-1.0, 0.0),
        "mdd_p95": (-1.0, 0.0),
    },
    "extra": _mdd_dist_extras,
}

_SPEC_SR_DIST: dict[str, Any] = {
    "keys": (
        "sr_p5",
        "sr_p50",
        "sr_p95",
        "n_valid_paths",
        "n_dropped_paths",
        "n_sim",
        "n",
        "block",
        "periods",
    ),
    # documented refusals: < 20 valid returns, n_sim < 1, periods < 1
    "exceptions": (ValueError,),
    "extra": _triple_ordered("sr_p5", "sr_p50", "sr_p95"),
}

_MC_CASES: dict[str, Case] = {
    "bootstrap_resample": {
        "base": BASE_RETURNS,
        "spec": _SPEC_BOOT_RESAMPLE,
        "invoke": lambda values: bootstrap_resample(
            values, np.mean, n_boot=20, seed=0
        ),
    },
    "max_drawdown_distribution": {
        "base": BASE_RETURNS,
        "spec": _SPEC_MDD_DIST,
        "invoke": lambda values: max_drawdown_distribution(
            values, n_sim=30, block=8, seed=0
        ),
    },
    "sharpe_distribution": {
        "base": BASE_RETURNS,
        "spec": _SPEC_SR_DIST,
        "invoke": lambda values: sharpe_distribution(
            values, n_sim=30, block=8, periods=252, seed=0
        ),
    },
}


def _run_case_matrix(
    case: Case, corruption: str, exclude: frozenset[str] = frozenset()
) -> None:
    """Apply one corruption to a case base and check the contract."""
    if corruption in exclude:
        pytest.skip("covered by a dedicated xfail finding test")
    values = CORRUPTIONS[corruption](case["base"])
    assert_chaos_contract(case["spec"], lambda: case["invoke"](values))


# ---------------------------------------------------------------------------
# evaluation_metrics chaos matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
@pytest.mark.parametrize("func_id", tuple(_EVAL_CASES))
def test_evaluation_metrics_chaos(func_id: str, corruption: str) -> None:
    """Metrics on corrupted return/trade/position streams stay sane."""
    _run_case_matrix(_EVAL_CASES[func_id], corruption)


# ---------------------------------------------------------------------------
# bootstrap_ci chaos matrix (+ dedicated finding tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
@pytest.mark.parametrize("func_id", tuple(_BOOTSTRAP_CASES))
def test_bootstrap_ci_chaos(func_id: str, corruption: str) -> None:
    """Bootstrap CIs on corrupted series stay sane or refuse loudly."""
    _run_case_matrix(
        _BOOTSTRAP_CASES[func_id],
        corruption,
        exclude=_SKIP_OVERFLOW.get(func_id, frozenset()),
    )


def test_finding_block_bootstrap_ci_overflow_se_inf() -> None:
    """chaos-1: se=inf (undocumented) on 1e306-scale input."""
    assert_chaos_contract(
        _SPEC_BLOCK_CI,
        lambda: block_bootstrap_ci(
            BASE_RETURNS * 1e308, n_boot=50, block=8, seed=0
        ),
    )


def test_finding_stationary_bootstrap_ci_overflow_se_inf() -> None:
    """chaos-1 (stationary variant): se=inf on 1e306-scale input."""
    assert_chaos_contract(
        _SPEC_BLOCK_CI,
        lambda: stationary_bootstrap_ci(
            BASE_RETURNS * 1e308, n_boot=50, mean_block=8.0, seed=0
        ),
    )


# ---------------------------------------------------------------------------
# cost_model chaos matrix (+ dedicated finding test)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
@pytest.mark.parametrize("func_id", tuple(_COST_CASES))
def test_cost_model_chaos(func_id: str, corruption: str) -> None:
    """Cost arithmetic on corrupted positions/returns stays sane."""
    _run_case_matrix(
        _COST_CASES[func_id],
        corruption,
        exclude=_SKIP_OVERFLOW.get(func_id, frozenset()),
    )


def test_finding_apply_proportional_costs_overflow_net_inf() -> None:
    """chaos-3: net returns silently become -inf on 1e308 positions."""
    assert_chaos_contract(
        _SPEC_APPLY_COSTS,
        lambda: apply_proportional_costs(
            BASE_RETURNS, BASE_POSITIONS * 1e308, 10.0
        ),
    )


def test_finding_cost_death_test_overflow_drag_inf() -> None:
    """chaos-3b: cost_drag_mean silently becomes inf on 1e308 pos."""
    assert_chaos_contract(
        _SPEC_COST_DEATH,
        lambda: cost_death_test(
            np.abs(BASE_RETURNS),
            BASE_POSITIONS * 1e308,
            10.0,
            n_boot=50,
            block=8,
            seed=0,
        ),
    )


# ---------------------------------------------------------------------------
# monte_carlo chaos matrix + parameter refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
@pytest.mark.parametrize("func_id", tuple(_MC_CASES))
def test_monte_carlo_chaos(func_id: str, corruption: str) -> None:
    """Resampling machinery on corrupted returns stays sane."""
    _run_case_matrix(_MC_CASES[func_id], corruption)


_PARAM_REFUSALS: dict[str, Callable[[], Any]] = {
    "percentile_ci ci_level=0": lambda: percentile_ci(BASE_RETURNS, 0.0),
    "percentile_ci ci_level=1": lambda: percentile_ci(BASE_RETURNS, 1.0),
    "percentile_ci ci_level=1.5": lambda: percentile_ci(BASE_RETURNS, 1.5),
    "percentile_ci ci_level=nan": lambda: percentile_ci(
        BASE_RETURNS, float("nan")
    ),
    "block_bootstrap_ci n_boot=0": lambda: block_bootstrap_ci(
        BASE_RETURNS, n_boot=0, block=8, seed=0
    ),
    "block_bootstrap_ci block=0": lambda: block_bootstrap_ci(
        BASE_RETURNS, n_boot=50, block=0, seed=0
    ),
    "block_bootstrap_ci block=-5": lambda: block_bootstrap_ci(
        BASE_RETURNS, n_boot=50, block=-5, seed=0
    ),
    "block_bootstrap_ci ci_level=1.5": lambda: block_bootstrap_ci(
        BASE_RETURNS, n_boot=50, block=8, ci_level=1.5, seed=0
    ),
    "stationary_bootstrap_ci n_boot=0": (
        lambda: stationary_bootstrap_ci(
            BASE_RETURNS, n_boot=0, mean_block=8.0, seed=0
        )
    ),
    "stationary_bootstrap_ci ci_level=2.0": (
        lambda: stationary_bootstrap_ci(
            BASE_RETURNS,
            n_boot=50,
            mean_block=8.0,
            ci_level=2.0,
            seed=0,
        )
    ),
    "sharpe_ratio ci_level=1.5": lambda: sharpe_ratio(
        BASE_RETURNS, ci_level=1.5, n_boot=50, block=8, seed=0
    ),
    "sharpe_ratio periods=0": lambda: sharpe_ratio(
        BASE_RETURNS, periods=0, n_boot=50, block=8, seed=0
    ),
    "sharpe_ratio n_boot=0": lambda: sharpe_ratio(
        BASE_RETURNS, n_boot=0, block=8, seed=0
    ),
    "sharpe_ratio block=-1": lambda: sharpe_ratio(
        BASE_RETURNS, n_boot=50, block=-1, seed=0
    ),
    "sharpe_ratio rf=inf": lambda: sharpe_ratio(
        BASE_RETURNS, rf=float("inf"), n_boot=50, block=8, seed=0
    ),
    "sharpe_ratio rf=nan": lambda: sharpe_ratio(
        BASE_RETURNS, rf=float("nan"), n_boot=50, block=8, seed=0
    ),
    "deflated_sharpe n_trials=0": lambda: deflated_sharpe(
        BASE_RETURNS, n_trials=0, seed=0
    ),
    "deflated_sharpe n_trials=-5": lambda: deflated_sharpe(
        BASE_RETURNS, n_trials=-5, seed=0
    ),
    "deflated_sharpe n_trials=1.5": lambda: deflated_sharpe(
        BASE_RETURNS,
        n_trials=1.5,  # type: ignore[arg-type]
        seed=0,
    ),
    "apply_proportional_costs negative_bps": (
        lambda: apply_proportional_costs(BASE_RETURNS, BASE_POSITIONS, -1.0)
    ),
    "apply_proportional_costs length_mismatch": (
        lambda: apply_proportional_costs(
            BASE_RETURNS, BASE_POSITIONS[:10], 10.0
        )
    ),
    "breakeven_cost_bps length_mismatch": (
        lambda: breakeven_cost_bps(BASE_RETURNS, BASE_POSITIONS[:10])
    ),
    "cost_death_test ci_level=0": lambda: cost_death_test(
        BASE_RETURNS, BASE_POSITIONS, 10.0, ci_level=0.0, seed=0
    ),
    "cost_death_test periods=-1": lambda: cost_death_test(
        BASE_RETURNS, BASE_POSITIONS, 10.0, periods=-1, seed=0
    ),
    "cost_death_test negative_bps": lambda: cost_death_test(
        BASE_RETURNS, BASE_POSITIONS, -1.0, seed=0
    ),
    "cost_death_test length_mismatch": lambda: cost_death_test(
        BASE_RETURNS, BASE_POSITIONS[:10], 10.0, seed=0
    ),
    "monte_carlo_paths s0=0": lambda: monte_carlo_paths(
        0.0, 0.05, 0.2, n_paths=4, n_steps=3
    ),
    "monte_carlo_paths s0=-1": lambda: monte_carlo_paths(
        -1.0, 0.05, 0.2, n_paths=4, n_steps=3
    ),
    "monte_carlo_paths sigma<0": lambda: monte_carlo_paths(
        1.0, 0.05, -0.2, n_paths=4, n_steps=3
    ),
    "monte_carlo_paths n_paths=0": lambda: monte_carlo_paths(
        1.0, 0.05, 0.2, n_paths=0, n_steps=3
    ),
    "monte_carlo_paths n_steps=-2": lambda: monte_carlo_paths(
        1.0, 0.05, 0.2, n_paths=4, n_steps=-2
    ),
    "monte_carlo_paths dt=0": lambda: monte_carlo_paths(
        1.0, 0.05, 0.2, n_paths=4, n_steps=3, dt=0.0
    ),
    "bootstrap_resample n_boot=0": lambda: bootstrap_resample(
        BASE_RETURNS, np.mean, n_boot=0
    ),
    "bootstrap_resample statistic_not_callable": (
        lambda: bootstrap_resample(
            BASE_RETURNS,
            "mean",  # type: ignore[arg-type]
            n_boot=5,
        )
    ),
    "max_drawdown_distribution n_sim=0": (
        lambda: max_drawdown_distribution(BASE_RETURNS, n_sim=0)
    ),
    "sharpe_distribution n_sim=0": lambda: sharpe_distribution(
        BASE_RETURNS, n_sim=0
    ),
    "sharpe_distribution periods=0": lambda: sharpe_distribution(
        BASE_RETURNS, n_sim=30, periods=0
    ),
    "walkforward purge_gap=-1": lambda: walkforward(
        _wf_base_frame(),
        train_window=100,
        test_window=50,
        step=50,
        purge_gap=-1,
    ),
    "walkforward non_dataframe": lambda: walkforward(
        np.ones((250, 2)),  # type: ignore[arg-type]
        train_window=100,
        test_window=50,
        step=50,
    ),
}


@pytest.mark.parametrize("case_name", sorted(_PARAM_REFUSALS))
def test_parameter_corruption_refuses_loudly(case_name: str) -> None:
    """Corrupted size/level parameters raise documented exceptions."""
    with pytest.raises((ValueError, TypeError)) as excinfo:
        _PARAM_REFUSALS[case_name]()
    assert str(excinfo.value), "a refusal must explain itself"


# ---------------------------------------------------------------------------
# _utils chaos
# ---------------------------------------------------------------------------
_SPEC_CLEAN_1D: dict[str, Any] = {"extra": _clean_1d_extras}


@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
def test_clean_1d_chaos(corruption: str) -> None:
    """clean_1d output is always a flat finite float array."""
    values = CORRUPTIONS[corruption](BASE_RETURNS)
    assert_chaos_contract(_SPEC_CLEAN_1D, lambda: clean_1d(values))


@pytest.mark.parametrize(
    ("n_obs", "n_sims"), [(0, 5), (10, 0), (-1, 5), (10, -3)]
)
def test_block_indices_rejects_non_positive_dimensions(
    n_obs: int, n_sims: int
) -> None:
    """Only the n_obs/n_sims < 1 ValueError path is in chaos scope."""
    with pytest.raises(ValueError, match="n_obs >= 1 and n_sims >= 1"):
        block_indices(n_obs, n_sims, 3, np.random.default_rng(0))


_SPEC_OLS: dict[str, Any] = {
    "keys": ("beta", "se", "t", "p", "df", "n", "r2", "rank_deficient"),
    "bounds": {"p": (0.0, 1.0), "se": (0.0, math.inf)},
}

_OLS_RNG = np.random.default_rng(77)
_OLS_TARGET = _OLS_RNG.normal(size=40)
_OLS_DESIGN = _OLS_RNG.normal(size=(40, 3))


def _with_column(
    design: np.ndarray, slot: int, values: np.ndarray
) -> np.ndarray:
    """Copy ``design`` with column ``slot`` replaced by ``values``."""
    out = design.copy()
    out[:, slot] = values
    return out


def _assert_nan_payload(result: Any) -> None:
    """Rank-deficient results must be all-NaN with the flag raised."""
    assert result["rank_deficient"] is True
    for key in ("beta", "se", "t", "p"):
        assert bool(np.isnan(result[key]).all()), f"{key} not all-NaN"


def test_ols_hc0_zero_column_flags_rank_deficient() -> None:
    """Documented: a zero column yields the NaN payload + flag."""
    design = _with_column(_OLS_DESIGN, 1, np.zeros(40))
    _assert_nan_payload(ols_hc0(_OLS_TARGET, design))


def test_ols_hc0_duplicate_column_flags_rank_deficient() -> None:
    """Documented: collinear columns yield the NaN payload + flag."""
    design = np.column_stack(
        [_OLS_DESIGN[:, 0], _OLS_DESIGN[:, 0], _OLS_DESIGN[:, 2]]
    )
    _assert_nan_payload(ols_hc0(_OLS_TARGET, design))


def test_ols_hc0_denormal_target_stays_sane() -> None:
    """Subnormal target: se may underflow to 0, t honestly NaN."""
    assert_chaos_contract(
        _SPEC_OLS, lambda: ols_hc0(_OLS_TARGET * 1e-320, _OLS_DESIGN)
    )


def test_ols_hc0_overflow_target_stays_sane() -> None:
    """Overflowing target: honest NaN payload (t not identifiable)."""
    assert_chaos_contract(
        _SPEC_OLS, lambda: ols_hc0(_OLS_TARGET * 1e308, _OLS_DESIGN)
    )


def test_finding_ols_hc0_denormal_design_beta_inf() -> None:
    """chaos-4: subnormal design silently yields beta = -inf."""
    assert_chaos_contract(
        _SPEC_OLS, lambda: ols_hc0(_OLS_TARGET, _OLS_DESIGN * 1e-320)
    )


def test_finding_ols_hc0_nan_column_linalg_error() -> None:
    """chaos-5: NaN design column crashes with LinAlgError."""
    design = _with_column(_OLS_DESIGN, 1, np.full(40, np.nan))
    assert_chaos_contract(_SPEC_OLS, lambda: ols_hc0(_OLS_TARGET, design))


# ---------------------------------------------------------------------------
# string-input refusals (dtype corruption, Any-typed inputs)
# ---------------------------------------------------------------------------
_STR_REFUSALS: dict[str, Callable[[], Any]] = {
    "clean_1d": lambda: clean_1d(_STR_VALS),
    "sharpe_ratio": lambda: sharpe_ratio(
        _STR_VALS, n_boot=10, block=4, seed=0
    ),
    "deflated_sharpe": lambda: deflated_sharpe(_STR_VALS, n_trials=3),
    "max_drawdown": lambda: max_drawdown(_STR_VALS),
    "calmar_ratio": lambda: calmar_ratio(_STR_VALS),
    "win_rate": lambda: win_rate(_STR_VALS),
    "profit_factor": lambda: profit_factor(_STR_VALS),
    "turnover": lambda: turnover(_STR_VALS),
    "percentile_ci": lambda: percentile_ci(_STR_VALS),
    "block_bootstrap_ci": lambda: block_bootstrap_ci(
        _STR_VALS, n_boot=10, block=4, seed=0
    ),
    "stationary_bootstrap_ci": lambda: stationary_bootstrap_ci(
        _STR_VALS, n_boot=10, mean_block=4.0, seed=0
    ),
    "turnover_series": lambda: turnover_series(_STR_VALS),
    "apply_proportional_costs": lambda: apply_proportional_costs(
        np.zeros(8), _STR_VALS, 10.0
    ),
    "breakeven_cost_bps": lambda: breakeven_cost_bps(np.zeros(8), _STR_VALS),
    "cost_death_test": lambda: cost_death_test(
        np.zeros(8), _STR_VALS, 10.0, n_boot=10, block=4, seed=0
    ),
    "bootstrap_resample": lambda: bootstrap_resample(
        _STR_VALS, np.mean, n_boot=5, seed=0
    ),
    "max_drawdown_distribution": lambda: max_drawdown_distribution(
        _STR_VALS, n_sim=25, seed=0
    ),
    "sharpe_distribution": lambda: sharpe_distribution(
        _STR_VALS, n_sim=25, seed=0
    ),
    "monte_carlo_paths": lambda: monte_carlo_paths(
        "not_a_number",  # type: ignore[arg-type]
        0.05,
        0.2,
        n_paths=4,
        n_steps=3,
    ),
}


@pytest.mark.parametrize("case_name", sorted(_STR_REFUSALS))
def test_string_input_refuses_loudly(case_name: str) -> None:
    """String arrays are refused with a messaged ValueError/TypeError."""
    with pytest.raises((ValueError, TypeError)) as excinfo:
        _STR_REFUSALS[case_name]()
    assert str(excinfo.value), "a refusal must explain itself"


# ---------------------------------------------------------------------------
# DataFrame chaos: walkforward
# ---------------------------------------------------------------------------
_WF_RNG = np.random.default_rng(11)
_WF_SIGNAL = _WF_RNG.standard_normal(250)
_WF_OUTCOME = 0.4 * _WF_SIGNAL + _WF_RNG.standard_normal(250)

_SPEC_WALKFORWARD: dict[str, Any] = {
    "keys": (
        "wf_t",
        "wf_direction_consistency",
        "wf_shrink_ratio",
        "verdict",
        "n_folds",
        "in_sample_t",
        "folds",
        "warnings",
        "mode",
        "purge_gap",
        "n_pnl_unique",
        "overlap_factor",
    ),
    # documented refusals: non-numeric features, frame too small
    "exceptions": (ValueError,),
    "bounds": {
        "wf_direction_consistency": (0.0, 1.0),
        "overlap_factor": (0.0, math.inf),
    },
    "extra": _walkforward_extras,
}


def _wf_base_frame() -> pd.DataFrame:
    """Deterministic 250-row signal/outcome frame for walkforward."""
    return pd.DataFrame({"signal": _WF_SIGNAL, "outcome": _WF_OUTCOME})


def _wf_nan_signal(every_other: bool) -> pd.DataFrame:
    """Frame with a fully (or half) NaN float feature column."""
    frame = _wf_base_frame()
    if every_other:
        frame.loc[frame.index[::2], "signal"] = np.nan
    else:
        frame["signal"] = np.nan
    return frame


def _wf_string_column(outcome: bool) -> pd.DataFrame:
    """Frame whose numeric signal (or outcome) column holds strings."""
    frame = _wf_base_frame()
    frame["outcome" if outcome else "signal"] = ["not_a_number"] * 250
    return frame


def _wf_duplicated_index() -> pd.DataFrame:
    """Frame with every index label duplicated twice."""
    frame = _wf_base_frame()
    frame.index = np.repeat(np.arange(125), 2)
    return frame


_WF_CORRUPTIONS: list[pytest.ParameterSet] = [
    pytest.param(
        _wf_nan_signal(every_other=False),
        id="nan_feature_column",
    ),
    pytest.param(
        _wf_nan_signal(every_other=True),
        id="nan_feature_column_partial",
    ),
    pytest.param(_wf_string_column(outcome=False), id="string_feature"),
    pytest.param(_wf_string_column(outcome=True), id="string_outcome"),
    pytest.param(_wf_duplicated_index(), id="duplicated_index"),
    pytest.param(
        pd.DataFrame(
            {
                "signal": np.array([], dtype=float),
                "outcome": np.array([], dtype=float),
            }
        ),
        id="empty_frame",
    ),
    pytest.param(_wf_base_frame().iloc[:1], id="single_row"),
]


@pytest.mark.parametrize("frame", _WF_CORRUPTIONS)
def test_walkforward_chaos(frame: pd.DataFrame) -> None:
    """Walk-forward on corrupted frames refuses or stays sane."""
    assert_chaos_contract(
        _SPEC_WALKFORWARD,
        lambda: walkforward(frame, train_window=100, test_window=50, step=50),
    )


def test_walkforward_nan_outcome_column_sane_fail() -> None:
    """All-NaN outcome degrades to NaN stats + FAIL, not garbage."""
    frame = _wf_base_frame()
    frame["outcome"] = np.nan
    assert_chaos_contract(
        _SPEC_WALKFORWARD,
        lambda: walkforward(frame, train_window=100, test_window=50, step=50),
    )


# ---------------------------------------------------------------------------
# DataFrame chaos: incremental_test
# ---------------------------------------------------------------------------
_INC_RNG = np.random.default_rng(12)
_INC_CONTROLS = pd.DataFrame(
    {
        "c0": _INC_RNG.normal(size=300),
        "c1": _INC_RNG.normal(size=300),
    }
)
_INC_Y = _INC_RNG.normal(size=300)
_INC_Z = 0.3 * _INC_CONTROLS["c0"].to_numpy() + 0.5 * _INC_RNG.normal(
    size=300
)

_SPEC_INCREMENTAL: dict[str, Any] = {
    "keys": (
        "t_new",
        "delta_r2",
        "verdict",
        "coef_new",
        "p_new",
        "r2_restricted",
        "r2_full",
        "n",
        "robust",
    ),
    # documented refusal: non-numeric input that fails float coercion
    "exceptions": (ValueError,),
    "extra": _verdict_in(
        ("SURVIVES", "REPACKAGING-DEATH", "NULL-DEATH", "INCONCLUSIVE")
    ),
}


def _inc_dup_index_controls() -> pd.DataFrame:
    """Controls frame with every index label duplicated twice."""
    frame = _INC_CONTROLS.copy()
    frame.index = np.repeat(np.arange(150), 2)
    return frame


_INC_CASES: dict[str, Callable[[], Any]] = {
    "nan_signal": lambda: incremental_test(
        _INC_Y, _INC_CONTROLS, np.full(300, np.nan)
    ),
    "nan_controls": lambda: incremental_test(
        _INC_Y, _INC_CONTROLS * np.nan, _INC_Z
    ),
    "nan_outcome": lambda: incremental_test(
        np.full(300, np.nan), _INC_CONTROLS, _INC_Z
    ),
    "zero_signal": lambda: incremental_test(
        _INC_Y, _INC_CONTROLS, np.zeros(300)
    ),
    "duplicated_control_index": lambda: incremental_test(
        _INC_Y, _inc_dup_index_controls(), _INC_Z
    ),
    "string_signal": lambda: incremental_test(
        _INC_Y, _INC_CONTROLS, np.array(["not_a_number"] * 300)
    ),
    "empty_controls_frame": lambda: incremental_test(
        _INC_Y, pd.DataFrame(), _INC_Z
    ),
    "empty_controls_columns": lambda: incremental_test(
        _INC_Y,
        pd.DataFrame(
            {
                "c0": np.array([], dtype=float),
                "c1": np.array([], dtype=float),
            }
        ),
        _INC_Z,
    ),
    "single_row": lambda: incremental_test(
        _INC_Y[:1], _INC_CONTROLS.iloc[:1], _INC_Z[:1]
    ),
}


@pytest.mark.parametrize("case_name", sorted(_INC_CASES))
def test_incremental_test_chaos(case_name: str) -> None:
    """Incremental test on corrupted inputs: INCONCLUSIVE or refusal."""
    assert_chaos_contract(_SPEC_INCREMENTAL, _INC_CASES[case_name])


# ---------------------------------------------------------------------------
# DataFrame chaos: known_features_control
# ---------------------------------------------------------------------------
_KFC_RNG = np.random.default_rng(13)
_KFC_FEATURES = pd.DataFrame(
    {
        "f0": _KFC_RNG.normal(size=300),
        "f1": _KFC_RNG.normal(size=300),
    }
)
_KFC_Y = _KFC_RNG.normal(size=300)
_KFC_Z = _KFC_RNG.normal(size=300)

_SPEC_KNOWN_FEATURES: dict[str, Any] = {
    "keys": ("stat", "p_value", "verdict", "n_features", "n", "r2"),
    # documented refusal: non-numeric input that fails float coercion
    "exceptions": (ValueError,),
    "bounds": {"p_value": (0.0, 1.0)},
    "extra": _verdict_in(("PASS", "FAIL", "INCONCLUSIVE")),
}


def _kfc_dup_index_features() -> pd.DataFrame:
    """Features frame with every index label duplicated twice."""
    frame = _KFC_FEATURES.copy()
    frame.index = np.repeat(np.arange(150), 2)
    return frame


_KFC_CASES: dict[str, Callable[[], Any]] = {
    "nan_outcome": lambda: known_features_control(
        np.full(300, np.nan), _KFC_Z, _KFC_FEATURES
    ),
    "nan_signal": lambda: known_features_control(
        _KFC_Y, np.full(300, np.nan), _KFC_FEATURES
    ),
    "nan_features": lambda: known_features_control(
        _KFC_Y, _KFC_Z, _KFC_FEATURES * np.nan
    ),
    "zero_variance_features": lambda: known_features_control(
        _KFC_Y,
        _KFC_Z,
        pd.DataFrame({"const": np.ones(300), "live": _KFC_Z}),
    ),
    "duplicated_feature_index": lambda: known_features_control(
        _KFC_Y, _KFC_Z, _kfc_dup_index_features()
    ),
    "string_features": lambda: known_features_control(
        _KFC_Y,
        _KFC_Z,
        pd.DataFrame({"bad": ["not_a_number"] * 300}),
    ),
    "empty_features": lambda: known_features_control(
        _KFC_Y,
        _KFC_Z,
        pd.DataFrame(
            {
                "f0": np.array([], dtype=float),
                "f1": np.array([], dtype=float),
            }
        ),
    ),
    "collinear_signal": lambda: known_features_control(
        _KFC_Y, _KFC_FEATURES["f0"].to_numpy(), _KFC_FEATURES
    ),
}


@pytest.mark.parametrize("case_name", sorted(_KFC_CASES))
def test_known_features_control_chaos(case_name: str) -> None:
    """Known-features control on corrupted inputs: sane verdicts."""
    assert_chaos_contract(_SPEC_KNOWN_FEATURES, _KFC_CASES[case_name])


# ---------------------------------------------------------------------------
# ml_training chaos: pure walk-forward mechanics under corruption
# ---------------------------------------------------------------------------
# The engine's pure primitives (rolling z, EV, threshold, declustering,
# fold edges, purge/embargo mask, stream moments, rank stability) and
# the orchestrator itself face the same corruption factories as the
# layers above.  Legal inf per docstring: ev_of_taken's mean of +-inf
# inputs is honestly +-inf (mixed signs -> NaN); stream_stats' mean of
# an all-inf stream is honestly +inf.  NaN is allowed everywhere.
_ML_HOUR_US = 3_600_000_000
_ML_RNG = np.random.default_rng(20261105)
BASE_ML_VALUES = _ML_RNG.normal(0.0005, 0.01, 64)
BASE_ML_STAMPS = np.arange(64, dtype=np.int64) * _ML_HOUR_US
BASE_ML_PROBA = np.linspace(0.05, 0.95, 64)
BASE_ML_TAKEN = BASE_ML_PROBA > 0.40
BASE_ML_RANK_A = _ML_RNG.permutation(np.arange(64, dtype=float))
BASE_ML_RANK_B = _ML_RNG.permutation(np.arange(64, dtype=float))


def _resized(pattern: np.ndarray, size: int) -> np.ndarray:
    """Tile/truncate ``pattern`` to ``size`` (np.resize, bool-safe)."""
    if size == 0:
        return pattern[:0].copy()
    return np.resize(pattern, size)


# |z_i| within a trailing window of length w is bounded by (w-1)/sqrt(w)
# (extremal case: one deviant against w-1 equals) -> 2.475 for w == 8
_SPEC_ML_ROLLING_Z: dict[str, Any] = {"bounds": {"value": (-2.5, 2.5)}}

_SPEC_ML_EV: dict[str, Any] = {"inf_fields": {"value"}}

_SPEC_ML_THRESHOLD: dict[str, Any] = {"bounds": {"value": (0.0, 1.0)}}

_SPEC_ML_DECLUSTER: dict[str, Any] = {
    "exceptions": (ValueError, TypeError),
    "bounds": {"value": (0.0, math.inf)},
}


def _fold_edges_extras(outcome: list[int]) -> None:
    """Edges must stay a strictly increasing int ladder."""
    assert all(low < high for low, high in pairwise(outcome))
    assert all(isinstance(edge, int) for edge in outcome)


def _purge_mask_extras(values: np.ndarray) -> Callable[[Any], None]:
    """Bind the formula recomputation to the corrupted stamps."""

    def check_mask(outcome: np.ndarray) -> None:
        """The mask must equal its definitional formula."""
        stamps = np.asarray(values, dtype=np.int64)
        test_start = 1000 * _ML_HOUR_US
        expected = (stamps + 3 * _ML_HOUR_US <= test_start) & (
            stamps + 24 * _ML_HOUR_US <= test_start
        )
        assert outcome.dtype == np.bool_
        assert np.array_equal(outcome, expected)

    return check_mask


_ML_CASES: dict[str, Case] = {
    "rolling_z": {
        "base": BASE_ML_VALUES,
        "spec": _SPEC_ML_ROLLING_Z,
        "invoke": lambda values: rolling_z(values, 8),
    },
    "ev_of_taken": {
        "base": BASE_ML_VALUES,
        "spec": _SPEC_ML_EV,
        "invoke": lambda values: ev_of_taken(
            values, _resized(BASE_ML_TAKEN, values.size)
        ),
    },
    "pick_threshold": {
        "base": BASE_ML_VALUES,
        "spec": _SPEC_ML_THRESHOLD,
        "invoke": lambda values: pick_threshold(
            _resized(BASE_ML_PROBA, values.size),
            values,
            grid=(0.30, 0.45, 0.60, 0.75),
        ),
    },
    "decluster_count": {
        "base": BASE_ML_STAMPS,
        "spec": _SPEC_ML_DECLUSTER,
        "invoke": decluster_count,
    },
    "fold_edges": {
        "base": BASE_ML_STAMPS,
        "spec": {
            "exceptions": (ValueError, TypeError),
            "extra": _fold_edges_extras,
        },
        "invoke": lambda values: fold_edges(values, 4),
    },
    "purge_embargo_mask": {
        "base": BASE_ML_STAMPS,
        "spec": {"exceptions": (ValueError, TypeError)},
        "invoke": lambda values: purge_embargo_mask(
            values, 1000 * _ML_HOUR_US
        ),
        "bound_extra": _purge_mask_extras,
    },
    "stream_stats": {
        "base": BASE_ML_VALUES,
        "spec": {
            "keys": ("sr", "skew", "kurt_pearson", "mean", "std"),
            "inf_fields": {"mean"},
            "bounds": {"sr": (-1e3, 1e3)},
        },
        "invoke": stream_stats,
    },
    "spearman_median": {
        "base": BASE_ML_RANK_A,
        "spec": {
            "exceptions": (TypeError, ValueError),
            "bounds": {"value": (-1.0, 1.0)},
        },
        "invoke": lambda values: spearman_median([values, BASE_ML_RANK_B]),
    },
}

# combos pinned by dedicated xfail finding tests below
_ML_SKIP: dict[str, frozenset[str]] = {
    "rolling_z": frozenset({"denormal_1e_320", "overflow_1e308"}),
    "decluster_count": frozenset(
        {"all_inf", "inf_alternating", "overflow_1e308"}
    ),
    "fold_edges": frozenset({"inf_alternating"}),
    "purge_embargo_mask": frozenset(
        {"all_inf", "all_nan", "inf_alternating", "overflow_1e308"}
    ),
    "stream_stats": frozenset({"overflow_1e308"}),
}


@pytest.mark.parametrize("corruption", CORRUPTION_ORDER)
@pytest.mark.parametrize("func_id", tuple(_ML_CASES))
def test_ml_training_pure_function_chaos(
    func_id: str, corruption: str
) -> None:
    """Walk-forward primitives on corrupted arrays stay sane or refuse."""
    if corruption in _ML_SKIP.get(func_id, frozenset()):
        pytest.skip("covered by a dedicated xfail finding test")
    case = _ML_CASES[func_id]
    values = CORRUPTIONS[corruption](case["base"])
    spec = case["spec"]
    binder = case.get("bound_extra")
    if binder is not None:
        spec = {**spec, "extra": binder(values)}
    assert_chaos_contract(spec, lambda: case["invoke"](values))


# ---------------------------------------------------------------------------
# ml_training findings (pinned xfail-strict; fix the module to flip them)
# ---------------------------------------------------------------------------
_F_ML_Z_DENORMAL = (
    "ml-chaos-1 FINDING: rolling_z on subnormal input (~1e-322) returns "
    "+-inf z-scores (54 of 64 entries measured). The rolling std "
    "underflows to exactly 0.0 and the docstring promises NaN whenever "
    "the window sd is zero; instead z = x/0.0 fabricates +-inf while "
    "the true z of every window is O(1)."
)
_F_ML_Z_OVERFLOW = (
    "ml-chaos-2 FINDING: rolling_z on 1e308-scaled input returns a "
    "finite -0.0 z-score while every other entry is NaN. The rolling "
    "std overflows to inf in the squared deviations, so z = (x - "
    "mean)/inf = -0.0 -- plausible-looking finite garbage fabricated "
    "by intermediate overflow; the true window z is O(1) and "
    "representable."
)
_F_ML_DECLUSTER_INF = (
    "ml-chaos-3 FINDING: decluster_count on infinite stamps (all-inf, "
    "alternating inf, or stamps*1e308 overflow) crashes with "
    "OverflowError('cannot convert float infinity to integer') -- an "
    "undocumented exception type; the int64-us contract owes the "
    "caller a messaged ValueError."
)
_F_ML_FOLD_EDGES_INF = (
    "ml-chaos-4 FINDING: fold_edges on alternating +/-inf stamps "
    "crashes with OverflowError from int(inf) -- same undocumented "
    "crash class as ml-chaos-3, not the designed ValueError the "
    "too-few-unique-stamps path takes."
)
_F_ML_PURGE_NONFINITE = (
    "ml-chaos-5 FINDING: purge_embargo_mask on non-finite stamps "
    "(inf/NaN) does NOT refuse -- np.asarray(..., dtype=np.int64) "
    "casts them to the platform sentinel int (only a RuntimeWarning) "
    "and the function silently returns a garbage keep-mask of cast "
    "dust, the exact 'plausible-looking garbage' this layer exists "
    "to catch."
)
_F_ML_STREAM_OVERFLOW = (
    "ml-chaos-6 FINDING: stream_stats on 1e308-scale finite input "
    "reports std=inf and sr=-0.0. The true std (~1e306) is "
    "representable; the inf leaks from the squared deviations "
    "overflowing inside np.std, and sr = mean/inf fabricates a "
    "plausible -0.0 Sharpe."
)
_F_ML_EV_SIZE_MISMATCH = (
    "ml-chaos-7 FINDING: ev_of_taken / pick_threshold with mismatched "
    "array lengths crash with IndexError('boolean index did not "
    "match') -- an opaque crash, not the messaged ValueError the "
    "chaos contract requires of a public statistic."
)


def _rolling_z_no_fabricated_dust(outcome: np.ndarray) -> None:
    """Finite z of random-normal windows must be O(1), never dust."""
    finite = outcome[np.isfinite(outcome)]
    assert finite.size == 0 or float(np.abs(finite).min()) > 1e-6, (
        f"finite z as small as {float(np.abs(finite).min())} on "
        "random-normal input: fabricated rounding dust, not a true z"
    )


def test_finding_rolling_z_denormal_inf_zscores() -> None:
    """ml-chaos-1: subnormal input fabricates +-inf z-scores."""
    assert_chaos_contract(
        _SPEC_ML_ROLLING_Z, lambda: rolling_z(BASE_ML_VALUES * 1e-320, 8)
    )


def test_finding_rolling_z_overflow_fabricated_zero() -> None:
    """ml-chaos-2: overflow input fabricates a -0.0 z-score."""
    spec = {**_SPEC_ML_ROLLING_Z, "extra": _rolling_z_no_fabricated_dust}
    assert_chaos_contract(spec, lambda: rolling_z(BASE_ML_VALUES * 1e308, 8))


def test_finding_decluster_count_inf_stamps_overflow_error() -> None:
    """ml-chaos-3: infinite stamps crash decluster with OverflowError."""
    assert_chaos_contract(
        _SPEC_ML_DECLUSTER, lambda: decluster_count(np.full(64, np.inf))
    )


def test_finding_fold_edges_inf_stamps_overflow_error() -> None:
    """ml-chaos-4: alternating inf stamps crash fold_edges."""
    corrupted = CORRUPTIONS["inf_alternating"](BASE_ML_STAMPS)
    assert_chaos_contract(
        {"exceptions": (ValueError,)}, lambda: fold_edges(corrupted, 4)
    )


def test_finding_purge_embargo_mask_nonfinite_stamps_refuse() -> None:
    """ml-chaos-5: non-finite stamps must refuse, not mask garbage."""
    stamps = np.array([np.inf, np.nan, 5.0, 2.0])
    with pytest.raises((ValueError, TypeError)):
        purge_embargo_mask(stamps, 10**12)


def test_finding_stream_stats_overflow_std_inf() -> None:
    """ml-chaos-6: 1e308-scale input leaks std=inf / sr=-0.0."""
    assert_chaos_contract(
        {
            "keys": ("sr", "skew", "kurt_pearson", "mean", "std"),
            "inf_fields": {"mean"},
        },
        lambda: stream_stats(BASE_ML_VALUES * 1e308),
    )


def test_finding_ev_of_taken_size_mismatch_index_error() -> None:
    """ml-chaos-7a: mismatched lengths crash ev_of_taken opaque."""
    assert_chaos_contract(
        {"exceptions": (ValueError, TypeError)},
        lambda: ev_of_taken(
            np.array([0.01, -0.005]), np.array([True, True, True])
        ),
    )


def test_finding_pick_threshold_size_mismatch_index_error() -> None:
    """ml-chaos-7b: mismatched lengths crash pick_threshold opaque."""
    assert_chaos_contract(
        {"exceptions": (ValueError, TypeError)},
        lambda: pick_threshold(
            np.array([0.5, 0.6]),
            np.array([0.01, -0.01, 0.02]),
            grid=(0.30, 0.50),
        ),
    )


# ---------------------------------------------------------------------------
# ml_training chaos: deflated_sharpe_value scalar corruption
# ---------------------------------------------------------------------------
_DSR_NAN_CASES: dict[str, tuple[float, int, float, float, float]] = {
    "sr_inf": (math.inf, 100, 0.0, 3.0, 0.0),
    "sr_nan": (math.nan, 100, 0.0, 3.0, 0.0),
    "sr_huge": (1e308, 100, 0.0, 3.0, 0.0),
    "var_sr_negative": (0.5, 100, 0.0, 3.0, -1e-9),
    "var_sr_inf": (0.5, 100, 0.0, 3.0, math.inf),
    "skew_inf": (0.5, 100, math.inf, 3.0, 0.1),
    "kurt_inf": (0.5, 100, 0.0, math.inf, 0.1),
    "denominator_zero": (2.0, 100, 2.0, 1.0, 0.0),
    "denominator_negative": (10.0, 100, 10.0, 1.0, 0.0),
    "n_obs_two": (0.5, 2, 0.0, 3.0, 0.1),
    "n_obs_one": (0.5, 1, 0.0, 3.0, 0.1),
}

_SPEC_ML_DSR: dict[str, Any] = {"bounds": {"value": (0.0, 1.0)}}


@pytest.mark.parametrize("case_name", sorted(_DSR_NAN_CASES))
def test_deflated_sharpe_value_scalar_chaos(case_name: str) -> None:
    """Degenerate scalar moments degrade to NaN, never a fabricated CDF.

    Non-finite moments, negative trial variance, a non-positive DSR
    denominator and n_obs < 3 are all unidentifiable inputs; the
    docstring promises NaN for each (never a fabricated 0 or 1).
    """
    sr, n_obs, skew_val, kurt_val, var_sr = _DSR_NAN_CASES[case_name]
    outcome = deflated_sharpe_value(
        sr, n_obs, skew_val, kurt_val, var_sr, n_trials=5
    )
    assert math.isnan(outcome), case_name


@pytest.mark.parametrize(
    ("n_obs", "n_trials"), [(0, 5), (-3, 5), (100, 1), (100, 2.5)]
)
def test_deflated_sharpe_value_rejects_bad_integers(
    n_obs: int, n_trials: float
) -> None:
    """Corrupted counts refuse with the validate_int message."""
    with pytest.raises(ValueError, match="must be"):
        deflated_sharpe_value(
            0.5,
            n_obs,
            0.0,
            3.0,
            0.1,
            n_trials=n_trials,  # type: ignore[arg-type]
        )


def test_deflated_sharpe_value_subnormal_moments_stay_in_unit_range() -> None:
    """Subnormal sr/skew degrade gracefully to Phi(0) = 0.5.

    A subnormal Sharpe is a finite number, so the CDF path runs: the
    deflation term is ~0 and the result is the honest midpoint 0.5
    (inside the documented [0, 1]), not NaN and not a fabricated
    extreme.
    """
    outcome = deflated_sharpe_value(1e-320, 100, 1e-320, 3.0, 0.0)
    assert 0.0 <= outcome <= 1.0
    assert outcome == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ml_training chaos: the orchestrator on a corrupted dataset
# ---------------------------------------------------------------------------
def _ml_dataset(n_rows: int = 64) -> dict[str, Any]:
    """Deterministic 64-row world for engine-level corruption."""
    rng = np.random.default_rng(20261105)
    feats = rng.standard_normal((n_rows, 3))
    frame = pd.DataFrame(
        {"f1": feats[:, 0], "f2": feats[:, 1], "ctrl1": feats[:, 2]}
    )
    stamps = np.arange(n_rows, dtype=np.int64) * _ML_HOUR_US
    exec_ret = 0.01 * np.sign(feats[:, 0]) + 0.002 * rng.standard_normal(
        n_rows
    )
    labels = (exec_ret > 0.004).astype(np.int64)
    return {
        "features": frame,
        "target": labels,
        "exec_returns": exec_ret,
        "times_us": stamps,
        "battery_controls": ["ctrl1"],
    }


_ML_ENGINE_CONFIG: dict[str, Any] = {
    "k_blocks": 3,
    "bootstrap_b": 50,
    "n_min_events": 5,
    "checks": {"deflated_sharpe": False},
    "model_params": {"gbm": {"n_estimators": 10}},
}


def _ml_engine_run(dataset: dict[str, Any]) -> Any:
    """Run the orchestrator over one (possibly corrupted) dataset."""
    return run_ml_kill_protocol(
        dataset["features"],
        dataset["target"],
        dataset["exec_returns"],
        dataset["times_us"],
        config=_ML_ENGINE_CONFIG,
        battery_controls=dataset["battery_controls"],
    )


def _ml_with(**overrides: Any) -> dict[str, Any]:
    """Copy the base dataset with named fields replaced."""
    dataset = _ml_dataset()
    dataset.update(overrides)
    return dataset


def _ml_nan_column() -> pd.DataFrame:
    """Feature frame whose f1 column is all NaN."""
    frame: pd.DataFrame = _ml_dataset()["features"].copy()
    frame["f1"] = np.nan
    return frame


def _ml_inf_column() -> pd.DataFrame:
    """Feature frame whose f2 column is all +inf."""
    frame: pd.DataFrame = _ml_dataset()["features"].copy()
    frame["f2"] = np.inf
    return frame


def _ml_nan_alternating_column() -> pd.DataFrame:
    """Feature frame with NaN in every second f1 row."""
    frame: pd.DataFrame = _ml_dataset()["features"].copy()
    frame.loc[frame.index[::2], "f1"] = np.nan
    return frame


def _ml_none_column() -> pd.DataFrame:
    """Feature frame with an object column of None (coerces to NaN)."""
    frame: pd.DataFrame = _ml_dataset()["features"].copy()
    frame["f1"] = np.empty(frame.shape[0], dtype=object)
    return frame


def _ml_string_column() -> pd.DataFrame:
    """Feature frame with a string column."""
    frame: pd.DataFrame = _ml_dataset()["features"].copy()
    frame["f2"] = ["not_a_number"] * frame.shape[0]
    return frame


_BASE_TARGET = _ml_dataset()["target"]
_BASE_EXEC = _ml_dataset()["exec_returns"]
_BASE_TIMES = _ml_dataset()["times_us"]
_BASE_TARGET[0] = 0  # keep both classes present in every derived world

_ML_ORCHESTRATOR_REFUSALS: dict[str, Callable[[], Any]] = {
    "features_nan_column": lambda: _ml_engine_run(
        _ml_with(features=_ml_nan_column())
    ),
    "features_inf_column": lambda: _ml_engine_run(
        _ml_with(features=_ml_inf_column())
    ),
    "features_nan_alternating": lambda: _ml_engine_run(
        _ml_with(features=_ml_nan_alternating_column())
    ),
    "features_none_column": lambda: _ml_engine_run(
        _ml_with(features=_ml_none_column())
    ),
    "features_string_column": lambda: _ml_engine_run(
        _ml_with(features=_ml_string_column())
    ),
    "features_not_a_frame": lambda: _ml_engine_run(
        _ml_with(features=_ml_dataset()["features"].to_numpy())
    ),
    "features_single_row": lambda: _ml_engine_run(
        _ml_with(
            features=_ml_dataset()["features"].iloc[:1],
            target=_BASE_TARGET[:1],
            exec_returns=_BASE_EXEC[:1],
            times_us=_BASE_TIMES[:1],
        )
    ),
    "features_empty": lambda: _ml_engine_run(
        _ml_with(
            features=_ml_dataset()["features"].iloc[:0],
            target=_BASE_TARGET[:0],
            exec_returns=_BASE_EXEC[:0],
            times_us=_BASE_TIMES[:0],
        )
    ),
    "exec_all_nan": lambda: _ml_engine_run(
        _ml_with(exec_returns=np.full(64, np.nan))
    ),
    "exec_one_inf": lambda: _ml_engine_run(
        _ml_with(
            exec_returns=np.where(np.arange(64) == 7, np.inf, _BASE_EXEC)
        )
    ),
    "target_not_binary": lambda: _ml_engine_run(
        _ml_with(target=np.where(np.arange(64) == 3, 2, _BASE_TARGET))
    ),
    "target_not_integer": lambda: _ml_engine_run(
        _ml_with(target=_BASE_TARGET.astype(float) + 0.5)
    ),
    "times_non_monotone": lambda: _ml_engine_run(
        _ml_with(times_us=_BASE_TIMES[::-1].copy())
    ),
    "times_not_integer": lambda: _ml_engine_run(
        _ml_with(times_us=_BASE_TIMES.astype(float) + 0.5)
    ),
    "length_mismatch_target": lambda: _ml_engine_run(
        _ml_with(target=_BASE_TARGET[:-1])
    ),
    "length_mismatch_exec": lambda: _ml_engine_run(
        _ml_with(exec_returns=_BASE_EXEC[:-3])
    ),
    "length_mismatch_times": lambda: _ml_engine_run(
        _ml_with(times_us=_BASE_TIMES[:-2])
    ),
    "battery_column_missing": lambda: _ml_engine_run(
        _ml_with(battery_controls=["no_such_column"])
    ),
}

_SPEC_ML_ENGINE_REFUSAL: dict[str, Any] = {"exceptions": (ValueError,)}


@pytest.mark.parametrize("case_name", sorted(_ML_ORCHESTRATOR_REFUSALS))
def test_ml_engine_corrupted_dataset_refuses_loudly(case_name: str) -> None:
    """The dataset contract refuses every corrupted input loudly.

    NaN/inf/object/string feature columns, non-finite exec returns,
    non-binary or non-integer targets, non-monotone or non-integer
    stamps, length mismatches, non-DataFrame features and missing
    battery columns all violate the documented dataset contract and
    must raise a messaged ValueError BEFORE any model is fit.
    """
    assert_chaos_contract(
        _SPEC_ML_ENGINE_REFUSAL, _ML_ORCHESTRATOR_REFUSALS[case_name]
    )


_ML_REFUSAL_MESSAGES: dict[str, str] = {
    "features_nan_column": "finite floats",
    "exec_one_inf": "finite floats",
    "target_not_binary": "binary",
    "target_not_integer": "exact integers",
    "times_non_monotone": "monoton",
    "times_not_integer": "exact integers",
    "length_mismatch_target": "must match",
    "features_not_a_frame": "DataFrame",
    "battery_column_missing": "not found in features",
}


@pytest.mark.parametrize(
    ("case_name", "message_pattern"),
    sorted(_ML_REFUSAL_MESSAGES.items()),
)
def test_ml_engine_refusal_messages_name_the_offender(
    case_name: str, message_pattern: str
) -> None:
    """Each documented refusal message names the offending contract."""
    with pytest.raises(ValueError, match=message_pattern):
        _ML_ORCHESTRATOR_REFUSALS[case_name]()


_ML_REPORT_KEYS = (
    "baseline",
    "battery",
    "config",
    "deflated_sharpe",
    "dsr_detail",
    "ev_baseline",
    "ev_battery",
    "ev_model",
    "folds",
    "importance_stability",
    "incremental",
    "kills",
    "models",
    "n_events",
    "n_min_events",
    "n_rows",
    "powered",
    "verdict",
)


def _valid_ml_verdict(verdict: Any) -> None:
    """A verdict must be one of the three documented forms."""
    is_dead = isinstance(verdict, str) and verdict.startswith("DEAD(")
    assert verdict in ("SURVIVES", "UNDERPOWERED") or is_dead, verdict


def _ml_report_extras(outcome: dict[str, Any]) -> None:
    """Top-level keys complete; verdict enums valid; rows counted.

    The generic walker cannot carry a required-keys spec here (it
    would demand the 18 top-level keys of every nested record too),
    so key completeness is asserted directly.
    """
    missing = set(_ML_REPORT_KEYS) - set(outcome)
    assert not missing, f"report lacks required keys: {missing}"
    _valid_ml_verdict(outcome["verdict"])
    for entry in outcome["models"].values():
        _valid_ml_verdict(entry["verdict"])
    assert outcome["n_rows"] > 0


def _ml_duplicated_stamp_dataset() -> dict[str, Any]:
    """Honest dataset whose every stamp appears twice (duplicates ok).

    The contract demands non-decreasing (not strictly increasing)
    stamps, so duplicated timestamps are legal input: fold edges
    deduplicate, both copies land in one fold, and the engine must
    still produce a fully sane report.  240 base hours leave the
    first fold enough pre-embargo history to train on.
    """
    base = _ml_dataset(240)
    frame = pd.DataFrame(
        np.repeat(base["features"].to_numpy(), 2, axis=0),
        columns=base["features"].columns,
    )
    return {
        "features": frame,
        "target": np.repeat(base["target"], 2),
        "exec_returns": np.repeat(base["exec_returns"], 2),
        "times_us": np.repeat(base["times_us"], 2),
        "battery_controls": base["battery_controls"],
    }


_SPEC_ML_REPORT: dict[str, Any] = {
    "extra": _ml_report_extras,
}


def test_ml_engine_duplicated_stamps_stay_sane() -> None:
    """Duplicate timestamps are legal: the report stays fully sane."""
    assert_chaos_contract(
        _SPEC_ML_REPORT,
        lambda: _ml_engine_run(_ml_duplicated_stamp_dataset()),
    )


_F_ML_ENGINE_EMPTY_TRAIN = (
    "ml-chaos-8 FINDING: run_ml_kill_protocol with k_blocks large "
    "relative to a short timeline (50 blocks over 300 hourly rows) "
    "does not refuse; the first fold retains ZERO train rows after "
    "purge+embargo and the engine crashes inside sklearn with "
    "'Found array with 0 sample(s) ... required by "
    "GradientBoostingClassifier' -- a third-party crash naming "
    "neither the empty fold nor the purge/embargo cause, instead of "
    "the designed pre-fit ValueError the orchestrator docstring "
    "promises."
)


def test_finding_engine_empty_first_train_fold_refuses_by_design() -> None:
    """ml-chaos-8: a purge-emptied first fold must refuse by name."""
    dataset = _ml_dataset()
    dataset["times_us"] = np.arange(300, dtype=np.int64) * _ML_HOUR_US
    dataset["features"] = pd.DataFrame(
        np.random.default_rng(4).standard_normal((300, 3)),
        columns=["f1", "f2", "ctrl1"],
    )
    dataset["exec_returns"] = 0.01 * np.sign(
        dataset["features"]["f1"].to_numpy()
    )
    dataset["target"] = (dataset["exec_returns"] > 0.004).astype(np.int64)
    with pytest.raises(ValueError, match=r"fold|train|purge|embargo"):
        run_ml_kill_protocol(
            dataset["features"],
            dataset["target"],
            dataset["exec_returns"],
            dataset["times_us"],
            config={**_ML_ENGINE_CONFIG, "k_blocks": 50},
            battery_controls=["ctrl1"],
        )
