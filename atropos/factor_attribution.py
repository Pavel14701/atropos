"""Factor attribution: the native factor_regression instrument.

The ``cross_sectional_factor`` plan (``atropos.instrument_map``)
judges a factor book by ``stage_7_cs`` + ``factor_regression``; this
module IS the ``factor_regression`` instrument.  Three public pieces:

* :func:`factor_regression` -- OLS of the strategy's return stream on
  factor proxies (intercept + HC0 statistics, the
  :class:`atropos.orthogonal.FitBlock` conventions), with an explicit
  collinearity screen (pairwise ``|pearson| >= 0.95`` drops the LATER
  factor in insertion order; an iterative VIF screen then drops the
  highest-VIF factor while any VIF exceeds 10 -- every drop lands in
  ``factors_dropped`` with its reason).
* :func:`factor_verdict` -- the ROUTING verdict of the report: a
  stream whose variance the factors explain is a ``FACTOR_EXPOSURE``
  (the book collects the factor; it is not alpha -- packaging-vs-
  direct comparison, §18), while a stream with a significant
  intercept AFTER the factor betas is an ``ALPHA_CANDIDATE`` (the
  orthogonal residual is the real finding).  Precedence: a
  significant alpha wins even when R2 is high -- the residual is the
  information the factors do not own.
* :func:`construct_factor_proxies` -- builds the standard proxy trio's
  honest subset (``mkt`` equal-weight mean + ``bab_proxy`` low-beta
  minus high-beta, trailing 60-row betas) from a dates x symbols
  returns panel, for callers who only have the panel.

The ``size_proxy`` is deliberately NOT constructed: a size factor
needs market-cap data the panel does not carry, and fabricating a
price-based stand-in would smuggle a different factor under the
name.  Callers with cap data pass their own ``size`` column into
``factor_regression``.

Alignment contract: ``factor_regression`` takes streams the CALLER
has already aligned by index/position (the shared ``align_inputs``
path still inner-joins pandas indexes and drops non-finite rows;
numpy inputs align positionally).

Import note: reuses :class:`atropos.orthogonal.FitBlock` and the
``ols_hc0`` helper so the fit statistics cannot drift from the
orthogonal check's conventions.

Run ``python -m atropos.factor_attribution`` for a synthetic demo.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from atropos._utils import align_inputs, frameify, ols_hc0, validate_number
from atropos.orthogonal import FitBlock

__all__ = [
    "FactorRegressionReport",
    "FactorVerdict",
    "construct_factor_proxies",
    "factor_regression",
    "factor_verdict",
]

_MIN_OBS = 30
"""Minimum aligned observations before a verdict may route anything."""

_PEARSON_DROP = 0.95
"""Pairwise |pearson| between factors that drops the later name."""

_VIF_DROP = 10.0
"""Variance-inflation-factor gate of the iterative second screen."""

_PROXY_WINDOW = 60
"""Trailing rows of the rolling beta estimate behind ``bab_proxy``."""

_OK = "OK"
_INCONCLUSIVE = "INCONCLUSIVE"


class FactorRegressionReport(TypedDict):
    """Fixed report schema; every key is present on every outcome.

    ``status`` is ``OK`` (identifiable design, ``n_obs >= 30``) or
    ``INCONCLUSIVE`` (empty alignment, degenerate outcome, all factors
    dropped, rank-deficient design, or ``n_obs < 30`` -- statistics are
    still carried when the fit ran, but the status forbids routing on
    them).  ``fit`` follows the :class:`atropos.orthogonal.FitBlock`
    conventions (``intercept`` first, then surviving factors in
    insertion order); ``alpha_mean`` / ``alpha_t`` duplicate the
    intercept's beta / t for the pinned reading.
    """

    label: str
    status: str
    n_obs: int
    factors_used: list[str]
    factors_dropped: dict[str, str]
    fit: FitBlock
    alpha_mean: float
    alpha_t: float
    r_squared: float
    notes: list[str]


class FactorVerdict(TypedDict):
    """Routing verdict of one :class:`FactorRegressionReport`.

    ``verdict`` is one of ``FACTOR_EXPOSURE`` / ``ALPHA_CANDIDATE`` /
    ``RESIDUAL_WEAK`` / ``INCONCLUSIVE``; ``informative`` is True only
    for the first two (the run separates the stream); the remaining
    fields follow the glossary verdict anatomy (reading + run facts,
    next step, what would change the verdict).
    """

    verdict: str
    informative: bool
    interpretation: str
    next_step: str
    what_would_change: str


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _is_degenerate(values: np.ndarray) -> bool:
    """True when a column cannot carry any statistical test.

    Same semantics as ``atropos.orthogonal._is_degenerate``: empty,
    all-NaN, no finite entry, or a zero peak-to-peak range over the
    finite slice.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.ptp(finite) == 0.0)


def _one_column_frame(value: Any, name: str) -> pd.DataFrame:
    """One-column frame view of an input, named for the alignment join.

    A multi-column value (DataFrame or 2-D array) cannot map onto one
    factor name, so it is rejected loudly (the same contract as the
    orthogonal check's input screen).
    """
    frame = frameify(value, name)
    if frame.shape[1] != 1:
        raise ValueError(
            f"factor {name!r} must be a single column, got"
            f" {frame.shape[1]} columns"
        )
    return frame


def _fit_block(fit: dict[str, Any], names: list[str]) -> FitBlock:
    """Pack one ``ols_hc0`` payload into the report's fit block.

    Same packing as ``atropos.orthogonal._fit_block``: the payload keys
    ``t`` / ``p`` become the report keys ``t_stat`` / ``p_value``.
    """
    return FitBlock(
        r2=float(fit["r2"]),
        beta={key: float(fit["beta"][pos]) for pos, key in enumerate(names)},
        se={key: float(fit["se"][pos]) for pos, key in enumerate(names)},
        t_stat={key: float(fit["t"][pos]) for pos, key in enumerate(names)},
        p_value={key: float(fit["p"][pos]) for pos, key in enumerate(names)},
    )


def _nan_fit_block() -> FitBlock:
    """Empty fit block for an outcome with no identifiable model."""
    return FitBlock(
        r2=float("nan"),
        beta={},
        se={},
        t_stat={},
        p_value={},
    )


def _pairwise_screen(
    names: list[str], arrays: list[np.ndarray]
) -> tuple[list[str], list[np.ndarray], dict[str, str]]:
    """Drop the LATER factor of any ``|pearson| >= 0.95`` pair.

    One deterministic pass in insertion order: factor ``j`` is dropped
    when any SURVIVING earlier factor ``i`` correlates with it at
    ``|pearson| >= 0.95``; the earlier name always survives.
    """
    kept: list[tuple[str, np.ndarray]] = []
    dropped: dict[str, str] = {}
    for name, arr in zip(names, arrays, strict=True):
        collinear_with = ""
        worst_rho = 0.0
        for kept_name, kept_arr in kept:
            rho = float(np.corrcoef(arr, kept_arr)[0, 1])
            if abs(rho) >= _PEARSON_DROP and abs(rho) > worst_rho:
                collinear_with = kept_name
                worst_rho = abs(rho)
        if collinear_with:
            dropped[name] = (
                f"|pearson| = {worst_rho:.2f} with {collinear_with}"
                f" (>= {_PEARSON_DROP} collinearity screen); the earlier"
                " name survives"
            )
        else:
            kept.append((name, arr))
    return (
        [name for name, _ in kept],
        [arr for _, arr in kept],
        dropped,
    )


def _vif_of(target: np.ndarray, others: list[np.ndarray]) -> float:
    """VIF of one factor against the remaining columns (inf on break)."""
    design = np.column_stack([np.ones(target.size), *others])
    r_squared = float(ols_hc0(target, design)["r2"])
    if not np.isfinite(r_squared) or r_squared >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - r_squared)


def _vif_screen(
    names: list[str], arrays: list[np.ndarray]
) -> tuple[list[str], list[np.ndarray], dict[str, str]]:
    """Iteratively drop the highest-VIF factor while any VIF > 10.

    Ties break to the LATER name in insertion order.  A single
    surviving factor has no VIF to compute (nothing left to inflate
    against), so the loop stops at one survivor.
    """
    kept_names = list(names)
    kept_arrays = list(arrays)
    dropped: dict[str, str] = {}
    while len(kept_names) >= 2:
        vifs = [
            _vif_of(
                kept_arrays[pos],
                kept_arrays[:pos] + kept_arrays[pos + 1 :],
            )
            for pos in range(len(kept_arrays))
        ]
        worst_pos = max(range(len(vifs)), key=lambda pos: (vifs[pos], pos))
        if vifs[worst_pos] <= _VIF_DROP:
            break
        dropped[kept_names[worst_pos]] = (
            f"VIF = {vifs[worst_pos]:.1f} > {_VIF_DROP:g} against the"
            " surviving factors (iterative VIF screen)"
        )
        del kept_names[worst_pos]
        del kept_arrays[worst_pos]
    return kept_names, kept_arrays, dropped


def _align_and_screen(
    returns: Any,
    factors: dict[str, Any],
) -> tuple[
    np.ndarray, list[str], list[np.ndarray], dict[str, str], list[str]
]:
    """Align the streams and run every factor-drop screen.

    Returns ``(returns_arr, usable_names, usable_arrays, dropped,
    notes)``: ``dropped`` maps every dropped factor name to its reason
    (degenerate column, pairwise ``|pearson| >= 0.95``, VIF > 10 --
    in that order) and ``notes`` carries the same reasons plus the
    row-drop note from the alignment.
    """
    notes: list[str] = []
    dropped: dict[str, str] = {}
    if not factors:
        notes.append("no factors supplied (factors dict is empty)")
    factor_frames = [
        _one_column_frame(value, name) for name, value in factors.items()
    ]
    returns_frame = frameify(returns, "returns")
    if returns_frame.shape[1] != 1:
        raise ValueError(
            "input 'returns' must be a single column, got"
            f" {returns_frame.shape[1]} columns"
        )
    n_before = len(
        pd.concat([returns_frame, *factor_frames], axis=1, join="inner")
    )
    aligned = align_inputs(returns, *factors.values())
    returns_arr = aligned[0]
    factor_arrs = aligned[1:]
    if n_before > returns_arr.size:
        notes.append(
            f"{n_before - int(returns_arr.size)} row(s) dropped by"
            " alignment (index mismatch or non-finite values)"
        )

    usable_names: list[str] = []
    usable_arrays: list[np.ndarray] = []
    for name, arr in zip(list(factors), factor_arrs, strict=True):
        if _is_degenerate(arr):
            dropped[name] = "degenerate column (constant or all-NaN)"
            notes.append(f"factor {name} dropped: degenerate column")
        else:
            usable_names.append(name)
            usable_arrays.append(arr)
    if not usable_names:
        return returns_arr, [], [], dropped, notes

    usable_names, usable_arrays, pairwise = _pairwise_screen(
        usable_names, usable_arrays
    )
    for name, reason in pairwise.items():
        dropped[name] = reason
        notes.append(f"factor {name} dropped: {reason}")
    usable_names, usable_arrays, vif_dropped = _vif_screen(
        usable_names, usable_arrays
    )
    for name, reason in vif_dropped.items():
        dropped[name] = reason
        notes.append(f"factor {name} dropped: {reason}")
    return returns_arr, usable_names, usable_arrays, dropped, notes


def _dominant_factor(
    fit: FitBlock, factors_used: list[str]
) -> tuple[str, float, float]:
    """Name / beta / t of the factor with the largest |t_stat|.

    Ties break to the earlier name in insertion order; with no finite
    factor t the name is ``""`` (the caller reads that as undecidable).
    """
    dominant_name = ""
    dominant_beta = float("nan")
    dominant_t = float("nan")
    for name in factors_used:
        t_stat = float(fit["t_stat"][name])
        if np.isfinite(t_stat) and abs(t_stat) > abs(dominant_t):
            dominant_name = name
            dominant_beta = float(fit["beta"][name])
            dominant_t = t_stat
    return dominant_name, dominant_beta, dominant_t


# ---------------------------------------------------------------------------
# public instruments
# ---------------------------------------------------------------------------
def factor_regression(
    returns: Any,
    factors: dict[str, Any],
    *,
    label: str = "unnamed",
) -> FactorRegressionReport:
    """OLS of a return stream on factor proxies (intercept + HC0).

    Fits ``returns ~ intercept + factors`` with the shared ``ols_hc0``
    (HC0 standard errors, t / p per column, R2, rank-deficiency flag --
    the orthogonal check's conventions), after two collinearity screens
    (pairwise ``|pearson| >= 0.95`` drops the later name; iterative VIF
    drops the highest-VIF factor while any VIF exceeds 10).  Degenerate
    factor columns (constant / all-NaN after alignment) are dropped
    with a note.  ``alpha_mean`` is the intercept beta, ``alpha_t`` its
    t-statistic: the mean and significance of the residual the factors
    do not explain.

    Args:
        returns: the strategy's return / R stream (pandas or numpy 1-D
            array-like), index/position-aligned with every factor by
            the CALLER; the shared alignment still drops non-finite
            rows.
        factors: mapping factor name -> single column (Series or 1-D
            array-like); insertion order fixes the design order.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`FactorRegressionReport`; ``status`` is ``OK`` only for
        an identifiable design with ``n_obs >= 30`` (see the class
        docstring for the INCONCLUSIVE causes).

    Raises:
        ValueError: if a factor value is not a single column.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(11)
        >>> mkt = rng.standard_normal(240)
        >>> strategy = pd.Series(0.8 * mkt + 0.1 * rng.standard_normal(240))
        >>> report = factor_regression(
        ...     strategy, {"mkt": pd.Series(mkt)}, label="pure beta"
        ... )
        >>> report["status"], round(report["fit"]["beta"]["mkt"], 2)
        ('OK', 0.8)
        >>> round(report["r_squared"], 2) >= 0.97
        True
        >>> twin = factor_regression(
        ...     strategy,
        ...     {"mkt": pd.Series(mkt), "mkt_copy": pd.Series(2.0 * mkt)},
        ...     label="collinear pair",
        ... )
        >>> list(twin["factors_dropped"]), twin["status"]
        (['mkt_copy'], 'OK')
    """
    (
        returns_arr,
        usable_names,
        usable_arrays,
        dropped,
        notes,
    ) = _align_and_screen(returns, factors)
    n_obs = int(returns_arr.size)

    status = _OK
    if n_obs == 0:
        status = _INCONCLUSIVE
        notes.append("no rows survived alignment - cannot fit any design")
    elif _is_degenerate(returns_arr):
        status = _INCONCLUSIVE
        notes.append("returns stream is constant after alignment")
    elif not usable_names:
        status = _INCONCLUSIVE
        notes.append(
            "no usable factor columns survived the screens - attribution"
            " is undecidable"
        )

    fit = _nan_fit_block()
    r_squared = float("nan")
    alpha_mean = float("nan")
    alpha_t = float("nan")
    if n_obs > 0 and usable_names and status == _OK:
        design = np.column_stack([np.ones(n_obs), *usable_arrays])
        payload = ols_hc0(returns_arr, design)
        fit = _fit_block(payload, ["intercept", *usable_names])
        r_squared = float(payload["r2"])
        alpha_mean = float(payload["beta"][0])
        alpha_t = float(payload["t"][0])
        if bool(payload["rank_deficient"]) or not np.isfinite(r_squared):
            status = _INCONCLUSIVE
            notes.append(
                "design is rank-deficient or saturated - statistics are"
                " NaN and must not be read as evidence"
            )
    if status == _OK and n_obs < _MIN_OBS:
        status = _INCONCLUSIVE
        notes.append(
            f"n_obs = {n_obs} < {_MIN_OBS} minimum - numbers carried"
            " for inspection only, routing is forbidden"
        )

    return FactorRegressionReport(
        label=label,
        status=status,
        n_obs=n_obs,
        factors_used=usable_names,
        factors_dropped=dropped,
        fit=fit,
        alpha_mean=alpha_mean,
        alpha_t=alpha_t,
        r_squared=r_squared,
        notes=notes,
    )


def factor_verdict(
    report: FactorRegressionReport,
    *,
    r2_exposure: float = 0.7,
    alpha_t_threshold: float = 2.0,
) -> FactorVerdict:
    """Routing verdict of a :class:`FactorRegressionReport` (§18).

    Semantics (precedence top-down):

    * ``ALPHA_CANDIDATE`` -- ``|alpha_t| >= alpha_t_threshold``: the
      orthogonal residual is the real finding, even when R2 is high.
    * ``FACTOR_EXPOSURE`` -- ``r_squared >= r2_exposure`` with an
      insignificant alpha: the book collects the factor; deploy it as
      an exposure, not as alpha (packaging-vs-direct comparison, §18).
    * ``RESIDUAL_WEAK`` -- neither factor nor alpha: noise on this
      alignment.
    * ``INCONCLUSIVE`` -- non-OK status (``n_obs < 30``, empty
      alignment, collinearity-undecidable or rank-deficient design).

    Args:
        report: a :class:`FactorRegressionReport` from
            :func:`factor_regression`.
        r2_exposure: R2 an exposure must clear; must lie in (0, 1).
        alpha_t_threshold: |alpha_t| that separates alpha from noise;
            must be a positive finite float.

    Returns:
        :class:`FactorVerdict` with the run facts (R2, alpha_t,
        dominant beta) in ``interpretation``.

    Raises:
        ValueError: if ``r2_exposure`` is not in (0, 1) or
            ``alpha_t_threshold`` is not a positive finite float.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(11)
        >>> mkt = rng.standard_normal(240)
        >>> pure_beta = pd.Series(0.8 * mkt + 0.1 * rng.standard_normal(240))
        >>> exposure = factor_verdict(
        ...     factor_regression(
        ...         pure_beta, {"mkt": pd.Series(mkt)}, label="pure beta"
        ...     )
        ... )
        >>> exposure["verdict"], exposure["informative"]
        ('FACTOR_EXPOSURE', True)
        >>> with_alpha = pd.Series(
        ...     0.05 + 0.8 * mkt + 0.05 * rng.standard_normal(240)
        ... )
        >>> factor_verdict(
        ...     factor_regression(
        ...         with_alpha, {"mkt": pd.Series(mkt)}, label="planted"
        ...     )
        ... )["verdict"]
        'ALPHA_CANDIDATE'
    """
    exposure_gate = validate_number("r2_exposure", r2_exposure)
    if not 0.0 < exposure_gate < 1.0:
        raise ValueError(
            f"r2_exposure must be in (0, 1), got {r2_exposure!r}"
        )
    alpha_gate = validate_number("alpha_t_threshold", alpha_t_threshold)
    if alpha_gate <= 0:
        raise ValueError(
            "alpha_t_threshold must be a positive finite float, got"
            f" {alpha_t_threshold!r}"
        )

    alpha_t = float(report["alpha_t"])
    r_squared = float(report["r_squared"])
    if (
        report["status"] != _OK
        or not report["factors_used"]
        or not np.isfinite(alpha_t)
        or not np.isfinite(r_squared)
    ):
        reason = (
            "; ".join(report["notes"]) if report["notes"] else "undecidable"
        )
        return FactorVerdict(
            verdict=_INCONCLUSIVE,
            informative=False,
            interpretation=f"undecidable: {reason}",
            next_step=(
                "collect at least 30 aligned observations and an"
                " identifiable factor set, then re-run"
            ),
            what_would_change=(
                "an OK report (full-rank factors, n_obs >= 30) with"
                " finite alpha_t and r_squared"
            ),
        )

    dominant_name, dominant_beta, dominant_t = _dominant_factor(
        report["fit"], report["factors_used"]
    )
    if abs(alpha_t) >= alpha_gate:
        return FactorVerdict(
            verdict="ALPHA_CANDIDATE",
            informative=True,
            interpretation=(
                f"|alpha_t| = {abs(alpha_t):.2f} >= {alpha_gate:g}: the"
                " residual after the factor betas is the finding"
                f" (R2 = {r_squared:.2f}; dominant exposure"
                f" {dominant_name} beta = {dominant_beta:.2f},"
                f" t = {dominant_t:.2f})"
            ),
            next_step=(
                "route the orthogonal residual to preregistration (§8)"
                " -- the alpha is what survives the factor block"
            ),
            what_would_change=(
                "alpha_t collapsing below the threshold under costs or"
                " extra controls reclassifies the stream"
                " (FACTOR_EXPOSURE if R2 holds, else RESIDUAL_WEAK)"
            ),
        )
    if r_squared >= exposure_gate:
        return FactorVerdict(
            verdict="FACTOR_EXPOSURE",
            informative=True,
            interpretation=(
                f"R2 = {r_squared:.2f} >= {exposure_gate:g} with"
                f" |alpha_t| = {abs(alpha_t):.2f} < {alpha_gate:g}: the"
                " stream IS the factor -- dominant exposure"
                f" {dominant_name} (beta = {dominant_beta:.2f},"
                f" t = {dominant_t:.2f})"
            ),
            next_step=(
                "collect the factor: deploy as a factor exposure (or"
                " run the packaging-vs-direct comparison, §18); do NOT"
                " book it as alpha"
            ),
            what_would_change=(
                "an orthogonal residual |alpha_t| >= threshold on a"
                " longer sample upgrades the verdict to ALPHA_CANDIDATE"
            ),
        )
    return FactorVerdict(
        verdict="RESIDUAL_WEAK",
        informative=False,
        interpretation=(
            f"R2 = {r_squared:.2f} < {exposure_gate:g} and |alpha_t| ="
            f" {abs(alpha_t):.2f} < {alpha_gate:g}: neither factor nor"
            " alpha -- the stream is noise on this alignment"
        ),
        next_step=(
            "graveyard entry (§8): nothing here is collectable; stop"
            " iterating on this stream"
        ),
        what_would_change=(
            "a longer aligned sample or a different factor set that"
            " lifts R2 (exposure) or |alpha_t| (alpha)"
        ),
    )


def construct_factor_proxies(
    panel_returns: pd.DataFrame,
    *,
    betas: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build the honest proxy subset from a dates x symbols panel.

    Constructs ``mkt`` (equal-weight mean of the row's finite symbols)
    and ``bab_proxy`` (equal-weight mean of the LOW-beta leg minus the
    HIGH-beta leg).  Betas are the caller's static ``betas`` mapping
    when given, else a trailing 60-row estimate ``cov(sym, mkt) /
    var(mkt)`` per symbol (the 60d convention); the rolling path
    yields NaN for the first 59 rows (no trailing window yet -- honest
    NaN, never a backfilled guess).  The median symbol of each split
    joins the LOW leg; a row whose betas tie everywhere (no high leg)
    reads NaN.  ``size_proxy`` is deliberately not constructed: a size
    factor needs market-cap data the panel does not carry, and a
    price-based stand-in would be a fabricated factor.

    Args:
        panel_returns: returns panel (rows = dates, columns = symbols);
            NaN cells are excluded symbol-by-symbol from ``mkt`` and
            poison any trailing window they touch.
        betas: optional static mapping symbol -> beta used for the
            low/high split instead of the trailing estimate; must cover
            every panel column.

    Returns:
        ``{"mkt": ndarray, "bab_proxy": ndarray}`` aligned with the
        panel's row order.

    Raises:
        ValueError: if ``panel_returns`` is not a DataFrame with at
            least two columns and one row, or ``betas`` does not cover
            every column with finite floats.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> steps = np.arange(120, dtype=float)
        >>> market = np.sin(steps / 10.0)
        >>> panel = pd.DataFrame(
        ...     {
        ...         "lo_1": 0.30 * market,
        ...         "lo_2": 0.40 * market,
        ...         "hi_1": 1.60 * market,
        ...         "hi_2": 1.90 * market + 0.01 * np.cos(steps / 7.0),
        ...     }
        ... )
        >>> static = construct_factor_proxies(
        ...     panel,
        ...     betas={"lo_1": 0.3, "lo_2": 0.4, "hi_1": 1.6, "hi_2": 1.9},
        ... )
        >>> sorted(static)
        ['bab_proxy', 'mkt']
        >>> round(float(static["mkt"][0]), 4)
        0.0025
        >>> round(float(static["bab_proxy"][0]), 4)
        -0.005
        >>> rolled = construct_factor_proxies(panel)
        >>> bool(np.isnan(rolled["bab_proxy"][:59]).all())
        True
        >>> bool(np.isnan(rolled["bab_proxy"][59]))
        False
    """
    if not isinstance(panel_returns, pd.DataFrame):
        raise ValueError(
            "panel_returns must be a pandas DataFrame (dates x symbols),"
            f" got {type(panel_returns).__name__}"
        )
    n_rows, n_cols = panel_returns.shape
    if n_cols < 2:
        raise ValueError(
            "panel_returns needs at least 2 symbol columns for a"
            f" cross-sectional split, got {n_cols}"
        )
    if n_rows < 1:
        raise ValueError("panel_returns needs at least 1 row")
    panel_arr = panel_returns.to_numpy(dtype=float)
    present = np.isfinite(panel_arr)
    counts = present.sum(axis=1)
    mkt = np.full(n_rows, np.nan)
    np.divide(
        np.where(present, panel_arr, 0.0).sum(axis=1),
        counts,
        out=mkt,
        where=counts > 0,
    )
    if betas is not None:
        bab = _bab_from_betas(panel_arr, panel_returns.columns, betas)
    elif n_rows >= _PROXY_WINDOW:
        bab = _bab_from_trailing_betas(panel_arr, mkt)
    else:
        bab = np.full(n_rows, np.nan)
    return {"mkt": mkt, "bab_proxy": bab}


def _bab_from_betas(
    panel_arr: np.ndarray,
    columns: pd.Index,
    betas: Mapping[str, float],
) -> np.ndarray:
    """Low-minus-high spread from the caller's static betas.

    Every panel column must appear in ``betas`` with a finite float;
    the median beta joins the LOW leg, an all-tie split reads NaN.
    """
    missing = [str(symbol) for symbol in columns if symbol not in betas]
    if missing:
        raise ValueError(
            f"betas must cover every panel column; missing: {missing}"
        )
    beta_values = np.asarray(
        [
            validate_number(f"betas[{symbol!r}]", betas[symbol])
            for symbol in columns
        ],
        dtype=float,
    )
    low_mask = beta_values <= float(np.median(beta_values))
    high_mask = ~low_mask
    if not low_mask.any() or not high_mask.any():
        return np.full(panel_arr.shape[0], np.nan)
    return panel_arr[:, low_mask].mean(axis=1) - panel_arr[:, high_mask].mean(
        axis=1
    )


def _bab_from_trailing_betas(
    panel_arr: np.ndarray, mkt: np.ndarray
) -> np.ndarray:
    """Low-minus-high spread from trailing 60-row betas vs ``mkt``.

    Rows whose 60-row window is incomplete (the first 59), contains a
    NaN cell or a zero-variance market window read NaN -- honest NaN,
    never a backfilled guess.  The median beta of a row joins the LOW
    leg; a fully-tied row (no high leg) reads NaN.
    """
    n_rows = panel_arr.shape[0]
    # sliding_window_view appends the window axis LAST:
    # win_panel is (windows, symbols, window), win_mkt (windows, window)
    win_panel = sliding_window_view(panel_arr, _PROXY_WINDOW, axis=0)
    win_mkt = sliding_window_view(mkt, _PROXY_WINDOW)
    dev_panel = win_panel - win_panel.mean(axis=2, keepdims=True)
    dev_mkt = win_mkt - win_mkt.mean(axis=1, keepdims=True)
    covariances = (dev_panel * dev_mkt[:, None, :]).sum(axis=2) / (
        _PROXY_WINDOW - 1
    )
    variances = (dev_mkt**2).sum(axis=1) / (_PROXY_WINDOW - 1)
    beta_frame = covariances / variances[:, None]

    bab = np.full(n_rows, np.nan)
    for row in range(_PROXY_WINDOW - 1, n_rows):
        beta_row = beta_frame[row - _PROXY_WINDOW + 1]
        if not np.isfinite(beta_row).all():
            continue
        low_mask = beta_row <= float(np.median(beta_row))
        high_mask = ~low_mask
        if not low_mask.any() or not high_mask.any():
            continue
        bab[row] = float(panel_arr[row, low_mask].mean()) - float(
            panel_arr[row, high_mask].mean()
        )
    return bab


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: planted factor world vs planted alpha world (asserted).

    A pure-beta stream must route FACTOR_EXPOSURE (the book collects
    the factor); the same world plus a constant daily alpha must route
    ALPHA_CANDIDATE (the orthogonal residual is the finding).  The
    proxy constructor is exercised on a deterministic four-symbol
    panel.  Every printed verdict is asserted, so a regression crashes
    the demo instead of printing a wrong verdict quietly.
    """
    n_rows = 300
    rng = np.random.default_rng(42)
    mkt = rng.standard_normal(n_rows)
    noise = 0.05 * rng.standard_normal(n_rows)
    factor_world = pd.Series(0.9 * mkt + noise)
    alpha_world = pd.Series(0.01 + 0.8 * mkt + noise)

    print("=" * 72)
    print("factor_attribution demo: planted worlds, routing verdicts")
    print("=" * 72)
    worlds = (
        (
            "planted factor world (beta 0.9, no alpha)",
            factor_world,
            "FACTOR_EXPOSURE",
        ),
        (
            "planted alpha world (beta 0.8 + 0.01 daily alpha)",
            alpha_world,
            "ALPHA_CANDIDATE",
        ),
    )
    for name, stream, expected in worlds:
        report = factor_regression(
            stream, {"mkt": pd.Series(mkt)}, label=name
        )
        verdict = factor_verdict(report)
        print(f"{name}:")
        print(
            f"  status={report['status']} n_obs={report['n_obs']}"
            f" R2={report['r_squared']:.3f}"
            f" alpha_t={report['alpha_t']:.2f}"
        )
        print(f"  verdict={verdict['verdict']}")
        print(f"  next_step: {verdict['next_step']}")
        assert report["status"] == "OK"
        assert verdict["verdict"] == expected
        assert verdict["informative"] is True
    short_report = factor_regression(
        factor_world, {"mkt": pd.Series(mkt[:20])}, label="short sample"
    )
    assert factor_verdict(short_report)["verdict"] == "INCONCLUSIVE"

    steps = np.arange(120, dtype=float)
    market = np.sin(steps / 10.0)
    panel = pd.DataFrame(
        {
            "lo_1": 0.30 * market,
            "lo_2": 0.40 * market,
            "hi_1": 1.60 * market,
            "hi_2": 1.90 * market,
        }
    )
    proxies = construct_factor_proxies(panel)
    print("proxy trio subset from a 4-symbol panel:")
    print(f"  keys={sorted(proxies)}")
    print(
        f"  bab warmup rows NaN: {int(np.isnan(proxies['bab_proxy']).sum())}"
        f" of {len(proxies['bab_proxy'])}"
    )
    assert sorted(proxies) == ["bab_proxy", "mkt"]
    assert bool(np.isnan(proxies["bab_proxy"][: _PROXY_WINDOW - 1]).all())
    assert bool(np.isfinite(proxies["bab_proxy"][_PROXY_WINDOW - 1 :]).all())
    assert float(np.nanmean(proxies["bab_proxy"])) < 0.0


if __name__ == "__main__":
    run_all_examples()
