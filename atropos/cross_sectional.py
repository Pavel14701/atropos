"""Cross-sectional (panel) hypothesis tests: many assets per date.

Extends Atropos beyond a single 1-D asset series to signal/return
panels: rows are dates (chronological), columns are assets.  Three
portfolio-level killers, priced net of transaction costs where
relevant (doctrine CASES_AND_FLAGS.md flags 6/8: gross amplitude below
the cost line, and turnover-driven COST-DEATH):

1. ``cross_sectional_ic`` - per-date rank (or raw) correlation between
   the signal and forward returns ACROSS assets; PASS needs a positive
   mean IC with ``ic_tstat >= tstat_cutoff`` and at least
   ``positive_share_min`` of the dates positive.
2. ``quantile_spread`` - equal-weight long top quantile / short bottom
   quantile portfolio, net of one-sided turnover costs; PASS needs
   ``net_tstat >= tstat_cutoff`` and a positive net Sharpe.
3. ``cs_monotonicity`` - mean forward return per signal quantile must
   rise with the quantile index (Spearman >= ``mono_spearman_min``).

Panels are aligned on the INTERSECTION of dates and assets; per date
only assets with BOTH a finite signal and a finite forward return are
used, and dates with fewer than ``min_assets`` valid assets are
dropped.  Every function returns the house contract
``{"stat", "p_value", "verdict": "PASS" | "FAIL" | "INCONCLUSIVE",
...extras}``.  Fully degenerate inputs (empty intersection, no
qualifying date) FAIL gracefully rather than raise; a THIN sample -
fewer than ``min_dates`` effective dates, or a non-finite t-statistic
caused by zero dispersion - is reported as ``"INCONCLUSIVE"`` instead
of lending a spurious PASS.

Run ``python -m atropos.cross_sectional`` for a seeded synthetic demo:
a planted cross-sectional edge must PASS, pure noise must FAIL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "cross_sectional_ic",
    "cs_monotonicity",
    "quantile_spread",
    "run_cross_sectional_tests",
]

_PERIODS_PER_YEAR = 252
_BPS_SCALE = 1e4


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _align_panels(
    signal_panel: pd.DataFrame, returns_panel: pd.DataFrame, min_assets: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], np.ndarray]:
    """Inner-join two panels by date and asset; find usable dates.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        min_assets: minimum number of assets with BOTH a finite signal
            and a finite forward return for the date to qualify.  The
            default of 5 with ``n_quantiles = 5`` leaves ONE asset per
            bucket; a diversified book needs
            ``min_assets >= 2 * n_quantiles``.

    Returns:
        ``(signal_frame, returns_frame, valid_positions, pair_mask)``:
        the two panels reindexed to the common dates/assets
        (a DatetimeIndex is sorted ascending defensively; a plain
        index must already be sorted), the integer
        row positions that pass the ``min_assets`` filter, and the
        ``(n_dates, n_assets)`` boolean mask of assets with both a
        finite signal and a finite forward return.

    Raises:
        ValueError: if an input is not a DataFrame, has non-unique
            date/asset labels, ``min_assets`` < 1, or the common
            index is a non-DatetimeIndex that is not sorted ascending
            (without dates the order cannot be recovered, and a
            scrambled reindex would make turnover order-dependent).

    Example::

        >>> aligned = _align_panels(sig_df, ret_df, 5)  # doctest: +SKIP
    """
    if min_assets < 1:
        raise ValueError(f"min_assets must be >= 1, got {min_assets}")
    for label, panel in (
        ("signal_panel", signal_panel),
        ("returns_panel", returns_panel),
    ):
        if not isinstance(panel, pd.DataFrame):
            raise ValueError(f"{label} must be a pandas DataFrame")
        if not panel.index.is_unique or not panel.columns.is_unique:
            raise ValueError(
                f"{label} must have unique date and asset labels"
            )
    common_index = signal_panel.index.intersection(returns_panel.index)
    if isinstance(common_index, pd.DatetimeIndex):
        # wall-clock dates: sort defensively (intersection preserves
        # the left operand's order, which a scrambled file could break)
        common_index = common_index.sort_values()
    elif not common_index.is_monotonic_increasing:
        # plain positional/label index: reindexing a scrambled index
        # would silently reorder rows and corrupt turnover, so refuse
        # loudly instead of guessing an order that was never given
        raise ValueError(
            "non-DatetimeIndex panels must be sorted ascending: "
            "got a scrambled index with no date information to "
            "recover the order from"
        )
    common_columns = signal_panel.columns.intersection(returns_panel.columns)
    signal_frame = signal_panel.reindex(
        index=common_index, columns=common_columns
    )
    returns_frame = returns_panel.reindex(
        index=common_index, columns=common_columns
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    pair_mask = np.isfinite(sig_matrix) & np.isfinite(ret_matrix)
    valid_positions = [
        date_pos
        for date_pos in range(sig_matrix.shape[0])
        if int(pair_mask[date_pos].sum()) >= min_assets
    ]
    return signal_frame, returns_frame, valid_positions, pair_mask


def _date_quantiles(
    signal_values: np.ndarray, n_quantiles: int
) -> np.ndarray:
    """Assign integer buckets 1..Q by ascending signal rank.

    Average ranks (``scipy.stats.rankdata``) handle ties, so tied
    signals share a bucket; ``ceil(rank/n * Q)`` clipped to [1, Q]
    mirrors the tercile scheme of ``robustness_battery``.

    Args:
        signal_values: finite signal values for one date (non-empty).
        n_quantiles: number of quantile buckets (Q >= 2).

    Returns:
        Integer bucket ids in 1..Q, same length and order as the
        input.

    Raises:
        ValueError: if ``signal_values`` is empty (bucketing an empty
            cross-section is always a caller bug).

    Example::

        >>> _date_quantiles(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        ... # doctest: +SKIP
        array([1, 1, 2, 2])
    """
    if signal_values.size == 0:
        raise ValueError(
            "signal_values must contain at least one element, got 0"
        )
    ranks: np.ndarray = (
        np.asarray(stats.rankdata(signal_values), dtype=float)
        / signal_values.size
    )
    buckets = np.clip(np.ceil(ranks * n_quantiles), 1, n_quantiles)
    return np.asarray(buckets).astype(np.int64)


def _per_date_ic(
    sig_vals: np.ndarray, ret_vals: np.ndarray, method: str
) -> float:
    """Cross-sectional correlation between signal and returns on one date.

    Args:
        sig_vals: finite signal values of the valid assets.
        ret_vals: matching finite forward returns.
        method: ``"spearman"`` (rank IC) or ``"pearson"`` (raw).

    Returns:
        The correlation, or NaN when either side is constant (zero
        cross-sectional dispersion makes a correlation undefined).

    Example::

        >>> _per_date_ic(np.array([1.0, 2.0, 3.0]),
        ...              np.array([3.0, 2.0, 1.0]), "spearman")
        ... # doctest: +SKIP
        -1.0
    """
    if np.unique(sig_vals).size < 2 or np.unique(ret_vals).size < 2:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(sig_vals, ret_vals).statistic)
    return float(np.corrcoef(sig_vals, ret_vals)[0, 1])


def _scaled_ratio(mean_val: float, disp_val: float, scale: float) -> float:
    """Compute ``mean / dispersion * scale`` with sign-safe infinities.

    A zero dispersion with a non-zero mean is a deterministic sample,
    so the ratio is defined as plus/minus infinity; NaN inputs stay
    NaN (e.g. ddof=1 dispersion of a single observation).

    Args:
        mean_val: sample mean.
        disp_val: sample dispersion (standard deviation, ddof=1).
        scale: multiplier such as ``sqrt(n)`` or ``sqrt(252)``.

    Returns:
        The scaled mean/displacement ratio, possibly infinite.

    Example::

        >>> _scaled_ratio(0.02, 0.0, 1.0)  # doctest: +SKIP
        inf
    """
    if disp_val > 0.0:
        return float(mean_val / disp_val * scale)
    if not (np.isfinite(mean_val) and np.isfinite(disp_val)):
        return float("nan")
    if mean_val > 0.0:
        return float("inf")
    if mean_val < 0.0:
        return float("-inf")
    return 0.0


def _t_pvalue(tstat: float, n_obs: int) -> float:
    """Two-sided p-value of ``tstat`` under Student's t distribution.

    Note:
        The tail assumes i.i.d. observations. Under autocorrelated IC
        series the effective df is smaller; the t-test is
        anti-conservative (over-kills) - the same caveat Lo (2002)
        flags for i.i.d. Sharpe standard errors. When IC persistence
        is material, prefer a block-bootstrap or Newey-West standard
        error over this tail.

    Args:
        tstat: t-statistic; infinities evaluate to 0.0, NaN stays
            NaN.
        n_obs: number of observations; ``df = n_obs - 1``.

    Returns:
        ``2 * stats.t.sf(|tstat|, df=n_obs - 1)`` - the exact tail of
        the mean test - or NaN when no degrees of freedom remain
        (``n_obs < 2``).

    Example::

        >>> _t_pvalue(2.0, 31)  # doctest: +SKIP
        0.0535...
    """
    if n_obs < 2:
        return float("nan")
    return float(2.0 * stats.t.sf(abs(tstat), df=n_obs - 1))


# ----------------------------------------------------------------------------
# private runners on pre-aligned data (the orchestrator aligns ONCE)
# ----------------------------------------------------------------------------
def _run_aligned_ic(
    sig_matrix: np.ndarray,
    ret_matrix: np.ndarray,
    pair_mask: np.ndarray,
    valid_positions: list[int],
    *,
    method: str,
    min_dates: int,
    tstat_cutoff: float,
    positive_share_min: float,
    return_series: bool,
) -> dict[str, Any]:
    """IC statistics on pre-aligned matrices (see ``cross_sectional_ic``).

    Args:
        sig_matrix: aligned signal values, rows = dates.
        ret_matrix: aligned forward returns, same shape.
        pair_mask: boolean per-date mask of valid (finite-pair)
            assets.
        valid_positions: row positions passing the ``min_assets``
            filter.
        method: ``"spearman"`` (rank IC) or ``"pearson"`` (raw).
        min_dates: minimum number of effective (finite-IC) dates for
            a decisive verdict; a smaller non-empty sample is
            INCONCLUSIVE.
        tstat_cutoff: minimum t-statistic for PASS.
        positive_share_min: minimum share of positive-IC dates for
            PASS.
        return_series: attach the per-date finite IC values under
            ``"ic_series"``.

    Returns:
        The ``cross_sectional_ic`` result contract.
    """
    ic_values: list[float] = []
    for date_pos in valid_positions:
        date_mask = pair_mask[date_pos]
        ic_values.append(
            _per_date_ic(
                sig_matrix[date_pos][date_mask],
                ret_matrix[date_pos][date_mask],
                method,
            )
        )
    ic_arr = np.asarray(
        [value for value in ic_values if np.isfinite(value)], dtype=float
    )
    n_dates = int(ic_arr.size)
    result: dict[str, Any] = {
        "stat": float("nan"),
        "p_value": 1.0,
        "verdict": "FAIL",
        "mean_ic": float("nan"),
        "ic_std": float("nan"),
        "ic_tstat": float("nan"),
        "ic_ir": float("nan"),
        "positive_share": float("nan"),
        "n_dates": n_dates,
        "method": method,
    }
    if return_series:
        result["ic_series"] = [float(value) for value in ic_arr]
    if n_dates == 0:
        return result
    mean_ic = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr, ddof=1)) if n_dates > 1 else float("nan")
    ic_tstat = _scaled_ratio(mean_ic, ic_std, float(np.sqrt(n_dates)))
    ic_ir = _scaled_ratio(mean_ic, ic_std, 1.0)
    positive_share = float(np.mean(ic_arr > 0.0))
    if n_dates < min_dates:
        verdict = "INCONCLUSIVE"
    elif not np.isfinite(ic_tstat):
        # Zero-dispersion IC sample: the t-stat is +-inf/NaN and must
        # not be read as overwhelming evidence for a PASS.
        verdict = "INCONCLUSIVE"
    else:
        verdict = (
            "PASS"
            if (
                mean_ic > 0.0
                and ic_tstat >= tstat_cutoff
                and positive_share >= positive_share_min
            )
            else "FAIL"
        )
    result.update(
        stat=mean_ic,
        p_value=_t_pvalue(ic_tstat, n_dates),
        verdict=verdict,
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_tstat=ic_tstat,
        ic_ir=ic_ir,
        positive_share=positive_share,
    )
    return result


def _run_aligned_spread(
    sig_matrix: np.ndarray,
    ret_matrix: np.ndarray,
    pair_mask: np.ndarray,
    valid_positions: list[int],
    *,
    n_quantiles: int,
    cost_bps: float,
    min_dates: int,
    tstat_cutoff: float,
) -> dict[str, Any]:
    """Spread statistics on pre-aligned data (see ``quantile_spread``).

    Implements the skip/hold semantics documented in
    ``quantile_spread``: dates that cannot form both extreme legs HOLD
    the previous book (mark-to-market gross return, missing leg
    returns count as 0.0, zero turnover) until trading resumes.

    Args:
        sig_matrix: aligned signal values, rows = dates.
        ret_matrix: aligned forward returns, same shape.
        pair_mask: boolean per-date mask of valid (finite-pair)
            assets.
        valid_positions: row positions passing the ``min_assets``
            filter.
        n_quantiles: number of quantile buckets (Q >= 2).
        cost_bps: one-way transaction cost per unit of turnover, in
            basis points.
        min_dates: minimum number of return observations for a
            decisive verdict; a smaller non-empty sample is
            INCONCLUSIVE.
        tstat_cutoff: minimum net t-statistic for PASS.

    Returns:
        The ``quantile_spread`` result contract.
    """
    row_count = int(sig_matrix.shape[0])
    column_count = int(sig_matrix.shape[1])
    traded_rows = np.zeros(row_count, dtype=bool)
    traded_rows[valid_positions] = True
    prev_weights = np.zeros(column_count, dtype=float)
    gross_list: list[float] = []
    turnover_list: list[float] = []
    assets_used: list[int] = []
    for date_pos in range(row_count):
        date_mask = pair_mask[date_pos]
        traded = False
        if traded_rows[date_pos]:
            sig_vals = sig_matrix[date_pos][date_mask]
            buckets = _date_quantiles(sig_vals, n_quantiles)
            top_sel = buckets == n_quantiles
            bottom_sel = buckets == 1
            if top_sel.any() and bottom_sel.any():
                ret_vals = ret_matrix[date_pos][date_mask]
                gross_list.append(
                    float(
                        np.mean(ret_vals[top_sel])
                        - np.mean(ret_vals[bottom_sel])
                    )
                )
                weights = np.zeros(column_count, dtype=float)
                col_ids = np.flatnonzero(date_mask)
                weights[col_ids[top_sel]] = 1.0 / float(top_sel.sum())
                weights[col_ids[bottom_sel]] = -1.0 / float(bottom_sel.sum())
                turnover_list.append(
                    float(np.abs(weights - prev_weights).sum())
                )
                assets_used.append(int(date_mask.sum()))
                prev_weights = weights
                traded = True
        if not traded and np.any(prev_weights != 0.0):
            # HOLD day: the previous book stays on, marked to market.
            # Legs whose return is missing on this date contribute
            # r = 0.0 (conservative convention); nothing is
            # rebalanced, so the day's turnover is exactly zero.
            day_returns = np.where(
                np.isfinite(ret_matrix[date_pos]),
                ret_matrix[date_pos],
                0.0,
            )
            gross_list.append(float(np.dot(prev_weights, day_returns)))
            turnover_list.append(0.0)
    n_dates = len(gross_list)
    if n_dates == 0:
        return {
            "stat": float("nan"),
            "p_value": 1.0,
            "verdict": "FAIL",
            "gross_mean": float("nan"),
            "net_mean": float("nan"),
            "cost_drag_mean": float("nan"),
            "gross_sharpe": float("nan"),
            "net_sharpe": float("nan"),
            "net_tstat": float("nan"),
            "annual_turnover": float("nan"),
            "n_dates": 0,
            "n_assets_mean": float("nan"),
            "n_quantiles": int(n_quantiles),
            "cost_bps": float(cost_bps),
        }
    gross_arr = np.asarray(gross_list, dtype=float)
    turnover_arr = np.asarray(turnover_list, dtype=float)
    cost_arr = turnover_arr * (float(cost_bps) / _BPS_SCALE)
    net_arr = gross_arr - cost_arr
    gross_mean = float(np.mean(gross_arr))
    net_mean = float(np.mean(net_arr))
    gross_std = (
        float(np.std(gross_arr, ddof=1)) if n_dates > 1 else float("nan")
    )
    net_std = float(np.std(net_arr, ddof=1)) if n_dates > 1 else float("nan")
    ann_scale = float(np.sqrt(_PERIODS_PER_YEAR))
    net_tstat = _scaled_ratio(net_mean, net_std, float(np.sqrt(n_dates)))
    net_sharpe = _scaled_ratio(net_mean, net_std, ann_scale)
    if n_dates < min_dates:
        verdict = "INCONCLUSIVE"
    elif not np.isfinite(net_tstat):
        # Zero-dispersion net series: +-inf/NaN t must not PASS.
        verdict = "INCONCLUSIVE"
    else:
        verdict = (
            "PASS"
            if (net_tstat >= tstat_cutoff and net_sharpe > 0.0)
            else "FAIL"
        )
    return {
        "stat": net_mean,
        "p_value": _t_pvalue(net_tstat, n_dates),
        "verdict": verdict,
        "gross_mean": gross_mean,
        "net_mean": net_mean,
        "cost_drag_mean": float(np.mean(cost_arr)),
        "gross_sharpe": _scaled_ratio(gross_mean, gross_std, ann_scale),
        "net_sharpe": net_sharpe,
        "net_tstat": net_tstat,
        "annual_turnover": float(np.mean(turnover_arr) * _PERIODS_PER_YEAR),
        "n_dates": n_dates,
        "n_assets_mean": (
            float(np.mean(assets_used)) if assets_used else float("nan")
        ),
        "n_quantiles": int(n_quantiles),
        "cost_bps": float(cost_bps),
    }


def _run_aligned_mono(
    sig_matrix: np.ndarray,
    ret_matrix: np.ndarray,
    pair_mask: np.ndarray,
    valid_positions: list[int],
    *,
    n_quantiles: int,
    min_dates: int,
    mono_spearman_min: float,
) -> dict[str, Any]:
    """Monotonicity statistics on pre-aligned data (see ``cs_monotonicity``).

    Args:
        sig_matrix: aligned signal values, rows = dates.
        ret_matrix: aligned forward returns, same shape.
        pair_mask: boolean per-date mask of valid (finite-pair)
            assets.
        valid_positions: row positions passing the ``min_assets``
            filter.
        n_quantiles: number of quantile buckets (Q >= 2).
        min_dates: minimum number of contributing dates for a
            decisive verdict; a smaller non-empty sample is
            INCONCLUSIVE.
        mono_spearman_min: minimum bucket-order Spearman correlation
            for PASS.

    Returns:
        The ``cs_monotonicity`` result contract.
    """
    sums = np.zeros(n_quantiles, dtype=float)
    counts = np.zeros(n_quantiles, dtype=float)
    dates_used = 0
    for date_pos in valid_positions:
        date_mask = pair_mask[date_pos]
        sig_vals = sig_matrix[date_pos][date_mask]
        ret_vals = ret_matrix[date_pos][date_mask]
        buckets = _date_quantiles(sig_vals, n_quantiles)
        if not ((buckets == n_quantiles).any() and (buckets == 1).any()):
            continue
        for bucket_id in range(1, n_quantiles + 1):
            sel = buckets == bucket_id
            sums[bucket_id - 1] += float(np.sum(ret_vals[sel]))
            counts[bucket_id - 1] += float(sel.sum())
        dates_used += 1
    means_arr = np.divide(
        sums,
        counts,
        out=np.full(n_quantiles, np.nan, dtype=float),
        where=counts > 0.0,
    )
    if not np.all(np.isfinite(means_arr)) or float(np.ptp(means_arr)) == 0.0:
        rho = float("nan")
        p_value = 1.0
    else:
        rank_corr = stats.spearmanr(np.arange(1, n_quantiles + 1), means_arr)
        rho = float(rank_corr.statistic)
        p_value = float(rank_corr.pvalue)
    if dates_used == 0:
        verdict = "FAIL"
    elif dates_used < min_dates:
        verdict = "INCONCLUSIVE"
    elif not np.isfinite(rho):
        verdict = "FAIL"
    else:
        verdict = "PASS" if rho >= mono_spearman_min else "FAIL"
    return {
        "stat": rho,
        "p_value": p_value,
        "verdict": verdict,
        "quantile_means": [float(value) for value in means_arr],
        "spearman": rho,
        "n_dates": dates_used,
        "n_quantiles": int(n_quantiles),
    }


# ----------------------------------------------------------------------------
# 1. cross-sectional information coefficient
# ----------------------------------------------------------------------------
def cross_sectional_ic(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    min_assets: int = 5,
    method: str = "spearman",
    min_dates: int = 20,
    tstat_cutoff: float = 2.0,
    positive_share_min: float = 0.5,
    return_series: bool = False,
) -> dict[str, Any]:
    """Mean per-date cross-sectional IC between signal and returns.

    For every qualifying date the signal and the forward returns are
    correlated ACROSS assets (rank Spearman IC by default, raw Pearson
    otherwise).  Dates where either side is constant contribute NaN
    and are dropped from the average.

    Note:
        ``ic_tstat`` and its ``p_value`` treat the per-date IC series
        as i.i.d.  Under autocorrelated IC series the effective df is
        smaller; the t-test is anti-conservative (over-kills) - see
        :func:`_t_pvalue`.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        min_assets: dates with fewer valid assets (finite signal AND
            finite return) are dropped from the analysis.  The default
            of 5 with ``n_quantiles = 5`` leaves ONE asset per
            bucket; prefer ``min_assets >= 2 * n_quantiles``.
        method: ``"spearman"`` or ``"pearson"``.
        min_dates: minimum number of effective (finite-IC) dates for
            a decisive verdict; ``1 <= n_dates < min_dates`` yields
            ``"INCONCLUSIVE"``.
        tstat_cutoff: minimum mean-IC t-statistic for PASS; must be
            > 0.
        positive_share_min: minimum share of positive-IC dates for
            PASS; strictly inside (0, 1).
        return_series: when True, add an ``"ic_series"`` key holding
            the per-date finite IC values (length ``n_dates``);
            absent by default.

    Returns:
        ``{stat, p_value, verdict, mean_ic, ic_std, ic_tstat, ic_ir,
        positive_share, n_dates, method}`` (plus ``ic_series`` when
        requested).  ``ic_tstat = mean/std*sqrt(n_dates)`` with
        ddof=1; ``p_value`` is the two-sided Student-t tail of
        ``ic_tstat`` (``df = n_dates - 1``).  Verdict is ``"PASS"``
        iff ``mean_ic > 0`` AND ``ic_tstat >= tstat_cutoff`` AND
        ``positive_share >= positive_share_min``; ``"INCONCLUSIVE"``
        when ``n_dates < min_dates`` or the t-statistic is not
        finite (zero dispersion); a fully empty panel FAILs
        gracefully.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``min_assets`` < 1, ``min_dates`` < 1, ``tstat_cutoff``
            <= 0, ``positive_share_min`` outside (0, 1) or an unknown
            ``method``.

    Example::

        >>> gen = np.random.default_rng(0)  # doctest: +SKIP
        >>> values = gen.standard_normal((60, 10))  # doctest: +SKIP
        >>> signal = pd.DataFrame(values)  # doctest: +SKIP
        >>> returns = pd.DataFrame(  # doctest: +SKIP
        ...     0.2 * values + gen.standard_normal((60, 10)))
        >>> cross_sectional_ic(signal, returns)["verdict"]  # doctest:+SKIP
        'PASS'
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(
            f"method must be 'spearman' or 'pearson', got {method!r}"
        )
    if min_dates < 1:
        raise ValueError(f"min_dates must be >= 1, got {min_dates}")
    if not float(tstat_cutoff) > 0.0:
        raise ValueError(f"tstat_cutoff must be > 0, got {tstat_cutoff}")
    if not 0.0 < float(positive_share_min) < 1.0:
        raise ValueError(
            f"positive_share_min must be in (0, 1), got {positive_share_min}"
        )
    signal_frame, returns_frame, valid_positions, pair_mask = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    return _run_aligned_ic(
        signal_frame.to_numpy(dtype=float),
        returns_frame.to_numpy(dtype=float),
        pair_mask,
        valid_positions,
        method=method,
        min_dates=min_dates,
        tstat_cutoff=tstat_cutoff,
        positive_share_min=positive_share_min,
        return_series=return_series,
    )


# ----------------------------------------------------------------------------
# 2. long-short quantile spread net of costs
# ----------------------------------------------------------------------------
def quantile_spread(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    cost_bps: float = 10.0,
    min_assets: int = 5,
    min_dates: int = 20,
    tstat_cutoff: float = 2.0,
) -> dict[str, Any]:
    """Daily long-top / short-bottom quantile spread, net of costs.

    Each date the valid assets are ranked by signal and split into
    ``n_quantiles`` buckets.  The portfolio is LONG the top bucket
    (weight ``+1/n_top`` per asset) and SHORT the bottom bucket
    (weight ``-1/n_bottom`` per asset): each leg sums to 1 in absolute
    value, so gross exposure is 2 and the gross long-short return is
    ``mean(top-bucket returns) - mean(bottom-bucket returns)``.

    Skip/hold semantics: a date that cannot form both extreme legs -
    fewer than ``min_assets`` valid assets, or a degenerate
    cross-section (e.g. a fully tied signal) leaving an extreme
    bucket empty - is a HOLD date while the previous book is
    non-zero: the previous weights are kept, the gross return
    accrues as ``sum_a w_a * r_a`` over the assets valid on that
    date (a leg whose return is missing that day contributes
    ``r = 0.0`` - a deliberately conservative, documented
    convention), and turnover is ZERO because nothing is rebalanced.
    Dates before the first trade (all-zero book) contribute no
    return observation at all.  This closes the old artefact where a
    skipped date earned no P&L while the next traded date still paid
    the full rebalancing cost against the stale weights.

    Turnover: weights change only on traded dates, so the one-sided
    turnover of a traded date ``t`` is
    ``sum_a |w_t[a] - w_{t-1}[a]|`` over all panel assets (assets
    without a valid weight contribute 0).  The FIRST traded date
    pays the full entry cost against an all-zero initial book, i.e.
    ``sum(|w_0|) = 2``.  ``cost_t = cost_bps/1e4 * turnover_t`` and
    the net return is ``gross_t - cost_t``.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        n_quantiles: number of quantile buckets (Q >= 2).
        cost_bps: one-way transaction cost per unit of turnover, in
            basis points (10 bps = 0.1% per unit traded).
        min_assets: dates with fewer valid assets are dropped (held,
            see above).  The default of 5 with ``n_quantiles = 5``
            leaves ONE asset per bucket; a diversified book needs
            ``min_assets >= 2 * n_quantiles`` (e.g. 10 for quintiles;
            the ~40-asset demo panel comfortably allows it).
        min_dates: minimum number of return observations for a
            decisive verdict; ``1 <= n_dates < min_dates`` yields
            ``"INCONCLUSIVE"``.
        tstat_cutoff: minimum net t-statistic (together with a
            positive net Sharpe) for PASS; must be > 0.

    Returns:
        ``{stat, p_value, verdict, gross_mean, net_mean,
        cost_drag_mean, gross_sharpe, net_sharpe, net_tstat,
        annual_turnover, n_dates, n_assets_mean, n_quantiles,
        cost_bps}``.  ``n_dates`` counts return observations (traded
        dates plus HOLD dates with an open book); ``n_assets_mean``
        is averaged over TRADED dates only.  Sharpes are annualized
        (``periods=252``, ddof=1); ``net_tstat =
        mean/std*sqrt(n_dates)`` of net daily returns with ddof=1;
        ``stat`` is ``net_mean`` and ``p_value`` the two-sided
        Student-t tail of ``net_tstat`` (``df = n_dates - 1``).
        Verdict is ``"PASS"`` iff ``net_tstat >= tstat_cutoff`` AND
        ``net_sharpe > 0``; ``"INCONCLUSIVE"`` when
        ``n_dates < min_dates`` or the t-statistic is not finite
        (zero dispersion); a fully empty panel FAILs gracefully.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``n_quantiles`` < 2, ``min_dates`` < 1, ``tstat_cutoff``
            <= 0 or negative / non-finite ``cost_bps``.

    Example::

        >>> gen = np.random.default_rng(0)  # doctest: +SKIP
        >>> latent = gen.standard_normal((240, 40))  # doctest: +SKIP
        >>> signal = pd.DataFrame(latent)  # doctest: +SKIP
        >>> returns = pd.DataFrame(  # doctest: +SKIP
        ...     0.004 * latent + 0.01 * gen.standard_normal((240, 40)))
        >>> quantile_spread(signal, returns)["verdict"]  # doctest:+SKIP
        'PASS'
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError(f"cost_bps must be finite and >= 0, got {cost_bps}")
    if min_dates < 1:
        raise ValueError(f"min_dates must be >= 1, got {min_dates}")
    if not float(tstat_cutoff) > 0.0:
        raise ValueError(f"tstat_cutoff must be > 0, got {tstat_cutoff}")
    signal_frame, returns_frame, valid_positions, pair_mask = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    return _run_aligned_spread(
        signal_frame.to_numpy(dtype=float),
        returns_frame.to_numpy(dtype=float),
        pair_mask,
        valid_positions,
        n_quantiles=n_quantiles,
        cost_bps=cost_bps,
        min_dates=min_dates,
        tstat_cutoff=tstat_cutoff,
    )


# ----------------------------------------------------------------------------
# 3. quantile monotonicity
# ----------------------------------------------------------------------------
def cs_monotonicity(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    min_assets: int = 5,
    min_dates: int = 20,
    mono_spearman_min: float = 0.5,
) -> dict[str, Any]:
    """Mean forward return per signal quantile must rise with the bucket.

    Assets are bucketed by signal rank per date (``_date_quantiles``);
    returns are pooled over dates within each bucket 1..Q.  The test
    statistic is the Spearman correlation between the quantile index
    and the pooled bucket means.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        n_quantiles: number of quantile buckets (Q >= 2).
        min_assets: dates with fewer valid assets are dropped.  The
            default of 5 with ``n_quantiles = 5`` leaves ONE asset
            per bucket; prefer ``min_assets >= 2 * n_quantiles``.
        min_dates: minimum number of contributing dates for a
            decisive verdict; ``1 <= n_dates < min_dates`` yields
            ``"INCONCLUSIVE"``.
        mono_spearman_min: minimum Spearman correlation between the
            bucket index and the bucket means for PASS; must lie in
            [-1, 1].

    Returns:
        ``{stat, p_value, verdict, quantile_means, spearman, n_dates,
        n_quantiles}``.  ``quantile_means[k-1]`` is the pooled mean
        forward return of bucket ``k``.  Verdict is ``"PASS"`` iff
        ``spearman >= mono_spearman_min``; ``"INCONCLUSIVE"`` when
        ``1 <= n_dates < min_dates``; flat means, empty buckets and
        fully degenerate panels yield NaN Spearman and FAIL.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``n_quantiles`` < 2, ``min_assets`` < 1, ``min_dates``
            < 1 or ``mono_spearman_min`` outside [-1, 1].

    Example::

        >>> gen = np.random.default_rng(0)  # doctest: +SKIP
        >>> latent = gen.standard_normal((240, 40))  # doctest: +SKIP
        >>> signal = pd.DataFrame(latent)  # doctest: +SKIP
        >>> returns = pd.DataFrame(  # doctest: +SKIP
        ...     0.004 * latent + 0.01 * gen.standard_normal((240, 40)))
        >>> cs_monotonicity(signal, returns)["spearman"]  # doctest:+SKIP
        1.0
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if min_dates < 1:
        raise ValueError(f"min_dates must be >= 1, got {min_dates}")
    if not -1.0 <= float(mono_spearman_min) <= 1.0:
        raise ValueError(
            f"mono_spearman_min must be in [-1, 1], got {mono_spearman_min}"
        )
    signal_frame, returns_frame, valid_positions, pair_mask = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    return _run_aligned_mono(
        signal_frame.to_numpy(dtype=float),
        returns_frame.to_numpy(dtype=float),
        pair_mask,
        valid_positions,
        n_quantiles=n_quantiles,
        min_dates=min_dates,
        mono_spearman_min=mono_spearman_min,
    )


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------
def run_cross_sectional_tests(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    n_quantiles: int = 5,
    min_assets: int = 5,
    min_dates: int = 20,
    tstat_cutoff: float = 2.0,
    positive_share_min: float = 0.5,
    mono_spearman_min: float = 0.5,
) -> dict[str, Any]:
    """Run all three cross-sectional killers and aggregate the verdict.

    The two panels are aligned ONCE (single intersection + validity
    mask) and the pre-aligned matrices are handed to the private
    per-test runners, so the alignment work is not repeated three
    times per call.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        cost_bps: one-way transaction cost in basis points, applied
            to the quantile-spread turnover.
        n_quantiles: quantile buckets for spread and monotonicity.
        min_assets: dates with fewer valid assets are dropped.  The
            default of 5 with ``n_quantiles = 5`` leaves ONE asset
            per bucket; prefer ``min_assets >= 2 * n_quantiles``
            (the ~40-asset demo panel comfortably allows it).
        min_dates: minimum number of effective dates per sub-test for
            decisive verdicts; thinner non-empty samples make the
            sub-tests INCONCLUSIVE.
        tstat_cutoff: minimum t-statistic for the IC and spread
            tests; must be > 0.
        positive_share_min: minimum share of positive-IC dates for
            the IC test; strictly inside (0, 1).
        mono_spearman_min: minimum bucket-order Spearman correlation
            for the monotonicity test; must lie in [-1, 1].

    Returns:
        ``{"ic", "spread", "monotonicity"}`` sub-reports (the dicts of
        the three individual tests) plus ``{"overall_verdict",
        "n_dates", "n_assets_mean", "cost_bps"}``.  ``n_dates`` is
        taken from the IC report (its effective finite-IC dates); the
        spread can have a DIFFERENT number of effective dates - its
        ``n_dates`` counts return observations (traded dates plus
        HOLD dates with an open book), and dates before the first
        trade contribute no observation - so read the spread sample
        size from ``report["spread"]["n_dates"]``.  The overall
        verdict is ``"PASS"`` iff all three sub-verdicts PASS,
        ``"FAIL"`` as soon as any sub-verdict FAILs, and
        ``"INCONCLUSIVE"`` otherwise (no kill, but not enough
        evidence to crown the hypothesis).

    Raises:
        ValueError: on invalid inputs (non-DataFrame panels,
            non-unique labels, ``n_quantiles`` < 2, ``min_assets`` <
            1, ``min_dates`` < 1, ``tstat_cutoff`` <= 0,
            ``positive_share_min`` outside (0, 1),
            ``mono_spearman_min`` outside [-1, 1], or
            negative/non-finite ``cost_bps``).

    Example::

        >>> report = run_cross_sectional_tests(  # doctest: +SKIP
        ...     signal_panel, returns_panel)
        >>> report["overall_verdict"]  # doctest:+SKIP
        'PASS'
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError(f"cost_bps must be finite and >= 0, got {cost_bps}")
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if min_dates < 1:
        raise ValueError(f"min_dates must be >= 1, got {min_dates}")
    if not float(tstat_cutoff) > 0.0:
        raise ValueError(f"tstat_cutoff must be > 0, got {tstat_cutoff}")
    if not 0.0 < float(positive_share_min) < 1.0:
        raise ValueError(
            f"positive_share_min must be in (0, 1), got {positive_share_min}"
        )
    if not -1.0 <= float(mono_spearman_min) <= 1.0:
        raise ValueError(
            f"mono_spearman_min must be in [-1, 1], got {mono_spearman_min}"
        )
    signal_frame, returns_frame, valid_positions, pair_mask = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    ic_report = _run_aligned_ic(
        sig_matrix,
        ret_matrix,
        pair_mask,
        valid_positions,
        method="spearman",
        min_dates=min_dates,
        tstat_cutoff=tstat_cutoff,
        positive_share_min=positive_share_min,
        return_series=False,
    )
    spread_report = _run_aligned_spread(
        sig_matrix,
        ret_matrix,
        pair_mask,
        valid_positions,
        n_quantiles=n_quantiles,
        cost_bps=cost_bps,
        min_dates=min_dates,
        tstat_cutoff=tstat_cutoff,
    )
    mono_report = _run_aligned_mono(
        sig_matrix,
        ret_matrix,
        pair_mask,
        valid_positions,
        n_quantiles=n_quantiles,
        min_dates=min_dates,
        mono_spearman_min=mono_spearman_min,
    )
    sub_verdicts = (
        ic_report["verdict"],
        spread_report["verdict"],
        mono_report["verdict"],
    )
    if all(verdict == "PASS" for verdict in sub_verdicts):
        overall_verdict = "PASS"
    elif any(verdict == "FAIL" for verdict in sub_verdicts):
        overall_verdict = "FAIL"
    else:
        overall_verdict = "INCONCLUSIVE"
    return {
        "ic": ic_report,
        "spread": spread_report,
        "monotonicity": mono_report,
        "overall_verdict": overall_verdict,
        "n_dates": int(ic_report["n_dates"]),
        "n_assets_mean": float(spread_report["n_assets_mean"]),
        "cost_bps": float(cost_bps),
    }


# ----------------------------------------------------------------------------
# demo (the ONLY place with randomness, always seeded)
# ----------------------------------------------------------------------------
def _planted_panel(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a seeded 240x40 panel with a persistent planted edge.

    A latent AR(0.85) cross-sectional factor is z-scored per date
    (``zsignal``); returns are ``0.0005 + 0.004*zsignal + noise`` and
    the observed signal is ``zsignal + 0.3*noise``.

    Args:
        seed: seed for ``np.random.default_rng``.

    Returns:
        ``(signal_panel, returns_panel)`` DataFrames, 240 business
        dates x 40 assets.
    """
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 240, 40
    latent = np.zeros((n_dates, n_assets))
    shocks = rng.standard_normal((n_dates, n_assets))
    for date_pos in range(1, n_dates):
        latent[date_pos] = 0.85 * latent[date_pos - 1] + shocks[date_pos]
    cross_std = latent.std(axis=1, keepdims=True)
    zsignal = (latent - latent.mean(axis=1, keepdims=True)) / np.where(
        cross_std > 0.0, cross_std, 1.0
    )
    returns = (
        0.0005
        + 0.004 * zsignal
        + 0.01 * rng.standard_normal((n_dates, n_assets))
    )
    signal = zsignal + 0.3 * rng.standard_normal((n_dates, n_assets))
    return _panel_from_array(signal), _panel_from_array(returns)


def _pure_noise_panel(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a seeded 240x40 panel with NO cross-sectional edge.

    Args:
        seed: seed for ``np.random.default_rng``.

    Returns:
        ``(signal_panel, returns_panel)`` of independent draws.
    """
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 240, 40
    signal = rng.standard_normal((n_dates, n_assets))
    returns = 0.0005 + 0.01 * rng.standard_normal((n_dates, n_assets))
    return _panel_from_array(signal), _panel_from_array(returns)


def _panel_from_array(values: np.ndarray) -> pd.DataFrame:
    """Wrap a 2-D array into a dated panel (rows = dates, cols = assets).

    Args:
        values: ``(n_dates, n_assets)`` float array.

    Returns:
        A DataFrame with a business-day DatetimeIndex and asset
        tickers ``TK00``.. as columns.
    """
    dates = pd.date_range("2024-01-02", periods=values.shape[0], freq="B")
    tickers = [f"TK{asset_pos:02d}" for asset_pos in range(values.shape[1])]
    return pd.DataFrame(values, index=dates, columns=tickers)


def _print_report(label: str, report: dict[str, Any]) -> None:
    """Print one ``run_cross_sectional_tests`` report block.

    Args:
        label: human-readable panel description.
        report: the dict returned by ``run_cross_sectional_tests``.
    """
    ic = report["ic"]
    spread = report["spread"]
    mono = report["monotonicity"]
    print(
        f"\n[{label}] overall={report['overall_verdict']} "
        f"n_dates={report['n_dates']} "
        f"n_assets_mean={report['n_assets_mean']:.1f} "
        f"cost_bps={report['cost_bps']:g}"
    )
    print(
        f"  ic           : mean={ic['mean_ic']:+.4f} "
        f"t={ic['ic_tstat']:.2f} "
        f"pos_share={ic['positive_share']:.2f} -> {ic['verdict']}"
    )
    print(
        f"  quantile_spread: gross={spread['gross_mean']:+.5f} "
        f"net={spread['net_mean']:+.5f} "
        f"net_sharpe={spread['net_sharpe']:.2f} "
        f"turnover/yr={spread['annual_turnover']:.1f} "
        f"-> {spread['verdict']}"
    )
    means_text = ", ".join(
        f"{value:+.5f}" for value in mono["quantile_means"]
    )
    print(
        f"  monotonicity : means=[{means_text}] "
        f"spearman={mono['spearman']:+.3f} -> {mono['verdict']}"
    )


def run_all_examples() -> None:
    """Demo the three cross-sectional tests on seeded synthetic panels.

    A planted persistent edge (seed 11, 240 dates x 40 assets) must
    PASS all three tests net of 10 bps costs; a pure-noise panel must
    FAIL.
    """
    print("=" * 72)
    print("cross_sectional demo: 240 dates x 40 assets, seed 11")
    print("=" * 72)
    signal_panel, returns_panel = _planted_panel(seed=11)
    planted_report = run_cross_sectional_tests(signal_panel, returns_panel)
    _print_report("planted edge", planted_report)
    # Layer-0 verification: the planted edge survives 10 bps costs
    assert planted_report["overall_verdict"] == "PASS"
    assert planted_report["n_dates"] == 240
    assert planted_report["n_assets_mean"] == 40.0
    assert planted_report["cost_bps"] == 10.0
    planted_ic = planted_report["ic"]
    planted_spread = planted_report["spread"]
    planted_mono = planted_report["monotonicity"]
    assert planted_ic["verdict"] == "PASS"
    assert 0.2 < planted_ic["mean_ic"] < 0.5
    assert planted_ic["ic_tstat"] >= 2.0
    assert planted_ic["positive_share"] >= 0.5
    assert planted_spread["verdict"] == "PASS"
    assert planted_spread["gross_mean"] > planted_spread["net_mean"] > 0.0
    assert planted_spread["net_sharpe"] > 0.0
    assert planted_spread["cost_drag_mean"] > 0.0
    assert planted_mono["verdict"] == "PASS"
    assert planted_mono["spearman"] >= 0.8
    bucket_means = list(planted_mono["quantile_means"])
    assert len(bucket_means) == 5
    # the pooled bucket means rise monotonically with the quantile index
    assert bucket_means == sorted(bucket_means)
    noise_signal, noise_returns = _pure_noise_panel(seed=99)
    noise_report = run_cross_sectional_tests(noise_signal, noise_returns)
    _print_report("pure noise", noise_report)
    # Layer-0 verification: pure noise must die in every killer
    assert noise_report["overall_verdict"] == "FAIL"
    for sub_report in (
        noise_report["ic"],
        noise_report["spread"],
        noise_report["monotonicity"],
    ):
        assert sub_report["verdict"] == "FAIL"
    assert abs(float(noise_report["ic"]["mean_ic"])) < 0.1
    assert noise_report["spread"]["net_mean"] < 0.0


if __name__ == "__main__":
    run_all_examples()
