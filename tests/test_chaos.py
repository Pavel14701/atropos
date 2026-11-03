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
