"""Shared private helpers for Atropos leaf modules.

Single home for the input-coercion, resampling and OLS helpers that leaf
modules need, so their behavior cannot drift apart between copies.
"""

import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "align_inputs",
    "apply_stat_rowwise",
    "block_indices",
    "clean_1d",
    "frameify",
    "ols_hc0",
    "purge_embargo_mask",
    "sane_variance_mask",
    "validate_int",
    "validate_number",
]

_HOUR_US = 3_600 * 1_000_000
"""Microseconds in one hour (the time unit of the purge/embargo mask)."""


def validate_int(name: str, value: Any, minimum: int = 1) -> int:
    """Reject non-integer or too-small integer parameters loudly.

    Systematic guard for every public integer parameter: a ``2.5`` used
    to be silently truncated by ``int(value)`` checks, and an
    ``object()`` used to escape as ``TypeError`` instead of the
    documented ``ValueError``.

    Args:
        name: parameter name quoted in the error message.
        value: candidate value; anything without an exact integer value
            (``2.5``, ``"3"``, ``object()``) is rejected.
        minimum: smallest accepted integer; defaults to 1.

    Returns:
        ``int(value)`` -- the validated integer.

    Raises:
        ValueError: if ``value`` is not an exact integer, or is below
            ``minimum``.

    Example::

        >>> validate_int("n_boot", 100)
        100
        >>> validate_int("n_boot", 2.5)
        Traceback (most recent call last):
            ...
        ValueError: n_boot must be an integer, got 2.5
    """
    try:
        as_int = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be an integer, got {value!r}"
        ) from error
    if as_int != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if as_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return as_int


def validate_number(name: str, value: Any) -> float:
    """Reject non-numeric or non-finite float parameters loudly.

    Coercion goes through ``float(value)`` with ``TypeError`` /
    ``ValueError`` re-raised as ``ValueError`` naming the parameter, so
    an ``object()`` (or a string, or NaN/inf) fails the documented
    ``Raises: ValueError`` contract instead of escaping as
    ``TypeError`` deep inside the arithmetic.

    Args:
        name: parameter name quoted in the error message.
        value: candidate value.

    Returns:
        ``float(value)`` -- the validated finite float.

    Raises:
        ValueError: if ``value`` cannot be converted to a finite float.

    Example::

        >>> validate_number("rf", 0.02)
        0.02
        >>> validate_number("rf", float("nan"))
        Traceback (most recent call last):
            ...
        ValueError: rf must be a finite number, got nan
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite number, got {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def clean_1d(values: Any) -> np.ndarray:
    """Coerce array-like input to a flat 1-D float array, non-finite dropped.

    Accepts lists, tuples, pandas/numpy arrays or scalars; returns a new
    1-D float ndarray containing only the finite entries in order.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    # np.asarray pins the ndarray type: stubs resolve this to Any.
    return np.asarray(arr[np.isfinite(arr)])


def frameify(source: Any, tag: str) -> pd.DataFrame:
    """Convert a Series / DataFrame / array-like input to a DataFrame.

    A Series becomes a single column named ``tag``; a DataFrame keeps
    its columns with a ``tag::`` prefix; 1-D array-likes become one
    ``tag`` column and 2-D array-likes get ``tag::0 .. tag::m-1``
    columns. Inputs of any other dimension raise ``ValueError``.
    """
    if isinstance(source, pd.Series):
        return source.to_frame(name=tag)
    if isinstance(source, pd.DataFrame):
        return source.add_prefix(tag + "::")
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 1:
        return pd.DataFrame({tag: arr})
    if arr.ndim != 2:
        raise ValueError(
            f"input {tag!r} must be 1-D or 2-D, got ndim={arr.ndim}"
        )
    return pd.DataFrame(
        arr, columns=[f"{tag}::{col}" for col in range(arr.shape[1])]
    )


def align_inputs(*arrays: Any) -> list[np.ndarray]:
    """Align pandas inputs by index, numpy by position; drop NaN rows.

    Mixing pandas and numpy inputs is allowed: numpy arrays inherit the
    index of the first pandas input when lengths match (positional
    alignment).

    Returns one float ndarray per input; 1-D inputs stay 1-D, wide
    inputs come back as (n, m) matrices.

    When NO row survives the NaN drop the returned arrays are empty
    (no exception is raised): public callers translate an empty
    alignment into an honest INCONCLUSIVE verdict instead of crashing
    deep inside a regression or silently killing the hypothesis on
    zero observations.
    """
    frames = [frameify(item, f"v{slot}") for slot, item in enumerate(arrays)]
    ref_index = None
    for item in arrays:
        if isinstance(item, (pd.Series, pd.DataFrame)) and not isinstance(
            item.index, pd.RangeIndex
        ):
            ref_index = item.index
            break
    if ref_index is not None:
        # frames is built 1:1 from arrays, so the strict zip is safe
        for slot, (item, frame) in enumerate(
            zip(arrays, frames, strict=True)
        ):
            if not isinstance(item, (pd.Series, pd.DataFrame)):
                if len(frame) == len(ref_index):
                    frame.index = ref_index
                else:
                    raise ValueError(
                        f"input {slot} (numpy, len={len(frame)}) cannot"
                        " be aligned with the pandas index"
                        f" (len={len(ref_index)}); pass pandas inputs or"
                        " equal-length arrays"
                    )
    comb = pd.concat(frames, axis=1, join="inner").dropna()
    out: list[np.ndarray] = []
    pos = 0
    for frame in frames:
        width = frame.shape[1]
        block = comb.iloc[:, pos : pos + width].to_numpy(dtype=float)
        out.append(block.reshape(-1) if width == 1 else block)
        pos += width
    return out


def sane_variance_mask(
    means: np.ndarray, sds: np.ndarray, rel_tolerance: float = 1e-12
) -> np.ndarray:
    """Flag rows whose standard deviation is signal, not rounding dust.

    A resampled row of near-constant values can carry an ``sd`` of float
    rounding scale (~1e-16 times the mean) instead of an exact 0.0, and
    the row's implied Sharpe (~1e15) then poisons percentile statistics.
    A row counts as usable when ``sd`` exceeds ``rel_tolerance * |mean|``
    (with an absolute epsilon floor for zero-mean rows).
    """
    dust = rel_tolerance * np.maximum(np.abs(means), np.finfo(float).eps)
    return sds > dust


def block_indices(
    n_obs: int, n_sims: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular moving-block resample indices with shape ``(n_sims, n_obs)``.

    ``block`` is clamped to ``[1, n_obs]``; blocks past the end wrap
    around (circular bootstrap). Indices are drawn in a single
    ``rng.integers`` call, so a fixed seed reproduces the historical
    per-module results exactly.
    """
    if n_obs < 1 or n_sims < 1:
        raise ValueError("block_indices needs n_obs >= 1 and n_sims >= 1")
    block = int(min(max(int(block), 1), n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=(int(n_sims), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    return idx.reshape(int(n_sims), n_blocks * block)[:, :n_obs]


def _axis_parameter_kind(stat: Callable[..., Any]) -> str | None:
    """Classify how ``stat`` can receive an ``axis`` argument.

    Returns ``"explicit"`` when an ``axis`` parameter is declared,
    ``"var_keyword"`` when only ``**kwargs`` is declared (support
    unclear), and ``None`` when ``stat`` cannot accept ``axis`` at all.
    """
    try:
        signature = inspect.signature(stat)
    except (TypeError, ValueError):
        return None
    for parameter in signature.parameters.values():
        if parameter.name == "axis" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return "explicit"
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return "var_keyword"
    return None


def apply_stat_rowwise(
    samples: np.ndarray, stat: Callable[..., Any]
) -> np.ndarray:
    """Apply ``stat`` to each row of ``samples``; return a 1-D float array.

    Uses the vectorized ``stat(samples, axis=1)`` call when ``stat``
    advertises an ``axis`` parameter (explicitly or via ``**kwargs``);
    otherwise applies ``stat`` row by row. Errors raised by ``stat``
    propagate unchanged -- a broken statistic must fail loudly instead
    of silently degrading to the loop path.
    """
    axis_kind = _axis_parameter_kind(stat)
    if axis_kind is not None:
        try:
            out = np.asarray(stat(samples, axis=1), dtype=float)
        except TypeError:
            # **kwargs did not actually accept axis: use the loop.
            if axis_kind == "explicit":
                raise
        else:
            if out.ndim == 1 and out.shape[0] == samples.shape[0]:
                return out
    return np.asarray([float(stat(row)) for row in samples], dtype=float)


def ols_hc0(target: np.ndarray, design: np.ndarray) -> dict[str, Any]:
    """Fit OLS with HC0 standard errors; NaN payload when unidentified.

    Returns the keys ``beta``, ``se``, ``t``, ``p``, ``df``, ``n``,
    ``r2`` and ``rank_deficient`` (same contract the per-module
    ``_fit``/``_ols`` helpers used, plus the flag). ``p``-values use the
    t distribution with ``df = n - k`` degrees of freedom -- an
    asymptotic approximation for HC0, kept as the house convention.

    When the design has at most as many rows as columns, or is not full
    column rank, the model is not identified and every statistic is NaN
    with ``rank_deficient=True``. A pseudo-inverse would otherwise
    fabricate plausible-looking minimum-norm t-stats exactly in the
    collinearity cases the library must flag, so callers are expected
    to treat ``rank_deficient`` (or non-finite ``t``) as INCONCLUSIVE.

    Non-finite inputs (NaN/inf columns or target) and fits whose
    coefficients overflow (denormal-scaled designs) return the same
    NaN payload: both are unidentified for reporting purposes, and
    pushing either into ``lstsq`` surfaces as an opaque ``LinAlgError``
    from the SVD instead of the promised NaN contract.
    """
    n_rows, n_cols = design.shape

    def degenerate_payload() -> dict[str, Any]:
        nan_vector = np.full(n_cols, np.nan)
        return {
            "beta": nan_vector,
            "se": nan_vector.copy(),
            "t": nan_vector.copy(),
            "p": nan_vector.copy(),
            "df": max(n_rows - n_cols, 1),
            "n": n_rows,
            "r2": float("nan"),
            "rank_deficient": True,
        }

    inputs_finite = bool(
        np.all(np.isfinite(design)) and np.all(np.isfinite(target))
    )
    if (
        not inputs_finite
        or n_rows <= n_cols
        or np.linalg.matrix_rank(design) < n_cols
    ):
        return degenerate_payload()
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    if not bool(np.all(np.isfinite(beta))):
        # e.g. a denormal column: lstsq overflows although OLS is
        # scale-equivariant and the true coefficients are finite
        return degenerate_payload()
    resid = target - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    design_res = design * resid[:, None]  # X' diag(e^2) X = (X*e)'(X*e)
    meat = design_res.T @ design_res
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    df = n_rows - n_cols
    pvalues = 2.0 * stats.t.sf(np.abs(t_stats), df)
    sst = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {
        "beta": np.asarray(beta, dtype=float),
        "se": np.asarray(se, dtype=float),
        "t": np.asarray(t_stats, dtype=float),
        "p": np.asarray(pvalues, dtype=float),
        "df": df,
        "n": n_rows,
        "r2": r2,
        "rank_deficient": False,
    }


def purge_embargo_mask(
    train_times: np.ndarray,
    test_start: int,
    horizon_h: int = 1,
    purge_h: int = 2,
    embargo_h: int = 24,
) -> np.ndarray:
    """Keep-mask for expanding walk-forward training rows (time-based).

    A training row at ``t`` is kept iff ``t + (horizon_h + purge_h)``
    hours <= ``test_start`` (its label window must not overlap the test
    fold) AND ``t + embargo_h`` hours <= ``test_start`` (residual
    information from rows NEAR the test boundary must not leak into
    training -- the embargo flank sits BEFORE the test start, not
    after it).
    ``train_times`` and ``test_start`` are int64 microseconds; the
    default hours mirror ``atropos.ml_training.DEFAULT_TRAINING_CONFIG``.

    Semantically distinct from
    :func:`atropos.predictive_model_eval.purged_kfold_cv`: that helper
    purges by ROW distance from both sides of a contiguous test block
    and has no embargo, while this mask works in wall-clock
    microseconds against an arbitrary test-start stamp and adds the
    embargo flank, which non-iid label streams require.

    Args:
        train_times: int64 microsecond stamps of candidate train rows.
        test_start: int64 microsecond stamp where the test fold starts.
        horizon_h: label horizon in hours (purge window component).
        purge_h: extra purge hours beyond the label horizon.
        embargo_h: embargo hours before the test fold start.

    Returns:
        Boolean keep-mask aligned with ``train_times``.

    Example::

        >>> hour = 3_600_000_000
        >>> purge_embargo_mask(
        ...     np.array([997, 998, 999]) * hour,
        ...     1000 * hour,
        ...     horizon_h=1,
        ...     purge_h=2,
        ...     embargo_h=2,
        ... ).tolist()
        [True, False, False]
    """
    raw = np.asarray(train_times)
    times_finite = bool(np.isfinite(raw.astype(float)).all())
    if raw.size and not times_finite:
        raise ValueError("train_times must contain finite microsecond stamps")
    stamps = raw.astype(np.int64)
    keep_purge = stamps + (horizon_h + purge_h) * _HOUR_US <= test_start
    keep_embargo = stamps + embargo_h * _HOUR_US <= test_start
    return keep_purge & keep_embargo
