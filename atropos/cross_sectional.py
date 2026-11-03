"""Cross-sectional (panel) hypothesis tests: many assets per date.

Extends Atropos beyond a single 1-D asset series to signal/return
panels: rows are dates (chronological), columns are assets.  Three
portfolio-level killers, priced net of transaction costs where
relevant (doctrine CASES_AND_FLAGS.md flags 6/8: gross amplitude below
the cost line, and turnover-driven COST-DEATH):

1. ``cross_sectional_ic`` - per-date rank (or raw) correlation between
   the signal and forward returns ACROSS assets; PASS needs a positive
   mean IC with t >= 2 and at least half of the dates positive.
2. ``quantile_spread`` - equal-weight long top quantile / short bottom
   quantile portfolio, net of one-sided turnover costs; PASS needs
   t >= 2 on the net daily long-short returns and a positive net
   Sharpe.
3. ``cs_monotonicity`` - mean forward return per signal quantile must
   rise with the quantile index (Spearman >= 0.5).

Panels are aligned on the INTERSECTION of dates and assets; per date
only assets with BOTH a finite signal and a finite forward return are
used, and dates with fewer than ``min_assets`` valid assets are
dropped.  Every function returns the house contract
``{"stat", "p_value", "verdict": "PASS" | "FAIL", ...extras}``;
degenerate inputs (empty intersection, no qualifying date) produce a
graceful FAIL rather than an exception.

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
_TSTAT_CUTOFF = 2.0
_POSITIVE_SHARE_MIN = 0.5
_MONO_SPEARMAN_MIN = 0.5
_BPS_SCALE = 1e4


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _align_panels(
    signal_panel: pd.DataFrame, returns_panel: pd.DataFrame, min_assets: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Inner-join two panels by date and asset; find usable dates.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        min_assets: minimum number of assets with BOTH a finite signal
            and a finite forward return for the date to qualify.

    Returns:
        ``(signal_frame, returns_frame, valid_positions)``: the two
        panels reindexed to the common dates/assets (DatetimeIndex is
        sorted ascending defensively) plus the integer row positions
        that pass the ``min_assets`` filter.

    Raises:
        ValueError: if an input is not a DataFrame, has non-unique
            date/asset labels, or ``min_assets`` < 1.

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
        common_index = common_index.sort_values()
    common_columns = signal_panel.columns.intersection(returns_panel.columns)
    signal_frame = signal_panel.reindex(
        index=common_index, columns=common_columns
    )
    returns_frame = returns_panel.reindex(
        index=common_index, columns=common_columns
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    valid_positions: list[int] = []
    for date_pos in range(sig_matrix.shape[0]):
        pair_mask = np.isfinite(sig_matrix[date_pos]) & np.isfinite(
            ret_matrix[date_pos]
        )
        if int(pair_mask.sum()) >= min_assets:
            valid_positions.append(date_pos)
    return signal_frame, returns_frame, valid_positions


def _date_quantiles(
    signal_values: np.ndarray, n_quantiles: int
) -> np.ndarray:
    """Assign integer buckets 1..Q by ascending signal rank.

    Average ranks (``scipy.stats.rankdata``) handle ties, so tied
    signals share a bucket; ``ceil(rank/n * Q)`` clipped to [1, Q]
    mirrors the tercile scheme of ``robustness_battery``.

    Args:
        signal_values: finite signal values for one date.
        n_quantiles: number of quantile buckets (Q >= 2).

    Returns:
        Integer bucket ids in 1..Q, same length and order as the
        input.

    Example::

        >>> _date_quantiles(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        ... # doctest: +SKIP
        array([1, 1, 2, 2])
    """
    ranks: np.ndarray = (
        np.asarray(stats.rankdata(signal_values), dtype=float)
        / signal_values.size
    )
    buckets = np.clip(np.ceil(ranks * n_quantiles), 1, n_quantiles)
    # np.asarray pins the ndarray type: stubs resolve this to Any.
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


# ----------------------------------------------------------------------------
# 1. cross-sectional information coefficient
# ----------------------------------------------------------------------------
def cross_sectional_ic(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    min_assets: int = 5,
    method: str = "spearman",
) -> dict[str, Any]:
    """Mean per-date cross-sectional IC between signal and returns.

    For every qualifying date the signal and the forward returns are
    correlated ACROSS assets (rank Spearman IC by default, raw Pearson
    otherwise).  Dates where either side is constant contribute NaN
    and are dropped from the average.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        min_assets: dates with fewer valid assets (finite signal AND
            finite return) are dropped from the analysis.
        method: ``"spearman"`` or ``"pearson"``.

    Returns:
        ``{stat, p_value, verdict, mean_ic, ic_std, ic_tstat, ic_ir,
        positive_share, n_dates, method}``.  ``ic_tstat =
        mean/std*sqrt(n_dates)`` with ddof=1; ``p_value`` is the
        two-sided normal tail of ``ic_tstat``.  Verdict is ``"PASS"``
        iff ``mean_ic > 0`` AND ``ic_tstat >= 2.0`` AND
        ``positive_share >= 0.5``; degenerate panels FAIL gracefully.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``min_assets`` < 1 or an unknown ``method``.

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
    signal_frame, returns_frame, valid_positions = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    ic_values: list[float] = []
    for date_pos in valid_positions:
        pair_mask = np.isfinite(sig_matrix[date_pos]) & np.isfinite(
            ret_matrix[date_pos]
        )
        ic_values.append(
            _per_date_ic(
                sig_matrix[date_pos][pair_mask],
                ret_matrix[date_pos][pair_mask],
                method,
            )
        )
    ic_arr = np.asarray(
        [value for value in ic_values if np.isfinite(value)], dtype=float
    )
    n_dates = int(ic_arr.size)
    if n_dates == 0:
        return {
            "stat": float("nan"),
            "p_value": 1.0,
            "verdict": "FAIL",
            "mean_ic": float("nan"),
            "ic_std": float("nan"),
            "ic_tstat": float("nan"),
            "ic_ir": float("nan"),
            "positive_share": float("nan"),
            "n_dates": 0,
            "method": method,
        }
    mean_ic = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr, ddof=1)) if n_dates > 1 else float("nan")
    ic_tstat = _scaled_ratio(mean_ic, ic_std, float(np.sqrt(n_dates)))
    ic_ir = _scaled_ratio(mean_ic, ic_std, 1.0)
    positive_share = float(np.mean(ic_arr > 0.0))
    verdict = (
        "PASS"
        if (
            mean_ic > 0.0
            and ic_tstat >= _TSTAT_CUTOFF
            and positive_share >= _POSITIVE_SHARE_MIN
        )
        else "FAIL"
    )
    return {
        "stat": mean_ic,
        "p_value": float(2.0 * stats.norm.sf(abs(ic_tstat))),
        "verdict": verdict,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_tstat": ic_tstat,
        "ic_ir": ic_ir,
        "positive_share": positive_share,
        "n_dates": n_dates,
        "method": method,
    }


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
) -> dict[str, Any]:
    """Daily long-top / short-bottom quantile spread, net of costs.

    Each date the valid assets are ranked by signal and split into
    ``n_quantiles`` buckets.  The portfolio is LONG the top bucket
    (weight ``+1/n_top`` per asset) and SHORT the bottom bucket
    (weight ``-1/n_bottom`` per asset): each leg sums to 1 in absolute
    value, so gross exposure is 2 and the gross long-short return is
    ``mean(top-bucket returns) - mean(bottom-bucket returns)``.
    Dates whose cross-section cannot populate BOTH extreme buckets
    (e.g. a fully tied signal) are skipped.

    Turnover: weights change daily, so the one-sided turnover of date
    ``t`` is ``sum_a |w_t[a] - w_{t-1}[a]|`` over all panel assets
    (assets without a valid weight contribute 0).  The FIRST traded
    date pays the full entry cost against an all-zero initial book,
    i.e. ``sum(|w_0|) = 2``.  ``cost_t = cost_bps/1e4 * turnover_t``
    and the net return is ``gross_t - cost_t``.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        n_quantiles: number of quantile buckets (Q >= 2).
        cost_bps: one-way transaction cost per unit of turnover, in
            basis points (10 bps = 0.1% per unit traded).
        min_assets: dates with fewer valid assets are dropped.

    Returns:
        ``{stat, p_value, verdict, gross_mean, net_mean,
        cost_drag_mean, gross_sharpe, net_sharpe, net_tstat,
        annual_turnover, n_dates, n_assets_mean, n_quantiles,
        cost_bps}``.  Sharpes are annualized (``periods=252``, ddof=1);
        ``net_tstat = mean/std*sqrt(n)`` of net daily returns;
        ``stat`` is ``net_mean`` and ``p_value`` the two-sided normal
        tail of ``net_tstat``.  Verdict is ``"PASS"`` iff
        ``net_tstat >= 2.0`` AND ``net_sharpe > 0``.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``n_quantiles`` < 2, ``min_assets`` < 1 or negative /
            non-finite ``cost_bps``.

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
    signal_frame, returns_frame, valid_positions = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    n_columns = int(sig_matrix.shape[1])
    prev_weights = np.zeros(n_columns, dtype=float)
    gross_list: list[float] = []
    turnover_list: list[float] = []
    assets_used: list[int] = []
    for date_pos in valid_positions:
        pair_mask = np.isfinite(sig_matrix[date_pos]) & np.isfinite(
            ret_matrix[date_pos]
        )
        sig_vals = sig_matrix[date_pos][pair_mask]
        ret_vals = ret_matrix[date_pos][pair_mask]
        buckets = _date_quantiles(sig_vals, n_quantiles)
        top_sel = buckets == n_quantiles
        bottom_sel = buckets == 1
        if not (top_sel.any() and bottom_sel.any()):
            continue  # tied/degenerate cross-section: no legs to hold
        gross_list.append(
            float(np.mean(ret_vals[top_sel]) - np.mean(ret_vals[bottom_sel]))
        )
        weights = np.zeros(n_columns, dtype=float)
        col_ids = np.flatnonzero(pair_mask)
        weights[col_ids[top_sel]] = 1.0 / float(top_sel.sum())
        weights[col_ids[bottom_sel]] = -1.0 / float(bottom_sel.sum())
        turnover_list.append(float(np.abs(weights - prev_weights).sum()))
        assets_used.append(int(pair_mask.sum()))
        prev_weights = weights
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
    verdict = (
        "PASS"
        if (net_tstat >= _TSTAT_CUTOFF and net_sharpe > 0.0)
        else "FAIL"
    )
    return {
        "stat": net_mean,
        "p_value": float(2.0 * stats.norm.sf(abs(net_tstat))),
        "verdict": verdict,
        "gross_mean": gross_mean,
        "net_mean": net_mean,
        "cost_drag_mean": float(np.mean(cost_arr)),
        "gross_sharpe": _scaled_ratio(gross_mean, gross_std, ann_scale),
        "net_sharpe": net_sharpe,
        "net_tstat": net_tstat,
        "annual_turnover": float(np.mean(turnover_arr) * _PERIODS_PER_YEAR),
        "n_dates": n_dates,
        "n_assets_mean": float(np.mean(assets_used)),
        "n_quantiles": int(n_quantiles),
        "cost_bps": float(cost_bps),
    }


# ----------------------------------------------------------------------------
# 3. quantile monotonicity
# ----------------------------------------------------------------------------
def cs_monotonicity(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    min_assets: int = 5,
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
        min_assets: dates with fewer valid assets are dropped.

    Returns:
        ``{stat, p_value, verdict, quantile_means, spearman, n_dates,
        n_quantiles}``.  ``quantile_means[k-1]`` is the pooled mean
        forward return of bucket ``k``.  Verdict is ``"PASS"`` iff
        ``spearman >= 0.5``; flat means, empty buckets and degenerate
        panels yield NaN Spearman and FAIL.

    Raises:
        ValueError: on a non-DataFrame input, non-unique labels,
            ``n_quantiles`` < 2 or ``min_assets`` < 1.

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
    signal_frame, returns_frame, valid_positions = _align_panels(
        signal_panel, returns_panel, min_assets
    )
    sig_matrix = signal_frame.to_numpy(dtype=float)
    ret_matrix = returns_frame.to_numpy(dtype=float)
    sums = np.zeros(n_quantiles, dtype=float)
    counts = np.zeros(n_quantiles, dtype=float)
    dates_used = 0
    for date_pos in valid_positions:
        pair_mask = np.isfinite(sig_matrix[date_pos]) & np.isfinite(
            ret_matrix[date_pos]
        )
        sig_vals = sig_matrix[date_pos][pair_mask]
        ret_vals = ret_matrix[date_pos][pair_mask]
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
    verdict = (
        "PASS" if (np.isfinite(rho) and rho >= _MONO_SPEARMAN_MIN) else "FAIL"
    )
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
# orchestrator
# ----------------------------------------------------------------------------
def run_cross_sectional_tests(
    signal_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    n_quantiles: int = 5,
    min_assets: int = 5,
) -> dict[str, Any]:
    """Run all three cross-sectional killers and aggregate the verdict.

    Args:
        signal_panel: rows = dates (chronological), columns = assets.
        returns_panel: forward returns, same layout.
        cost_bps: one-way transaction cost in basis points, applied
            to the quantile-spread turnover.
        n_quantiles: quantile buckets for spread and monotonicity.
        min_assets: dates with fewer valid assets are dropped.

    Returns:
        ``{"ic", "spread", "monotonicity"}`` sub-reports (the dicts of
        the three individual tests) plus ``{"overall_verdict",
        "n_dates", "n_assets_mean", "cost_bps"}``.  The overall
        verdict is ``"PASS"`` iff all three sub-verdicts PASS.

    Raises:
        ValueError: propagated from the sub-tests on invalid inputs.

    Example::

        >>> report = run_cross_sectional_tests(  # doctest: +SKIP
        ...     signal_panel, returns_panel)
        >>> report["overall_verdict"] in {"PASS", "FAIL"}  # doctest:+SKIP
        True
    """
    ic_report = cross_sectional_ic(
        signal_panel, returns_panel, min_assets=min_assets
    )
    spread_report = quantile_spread(
        signal_panel,
        returns_panel,
        n_quantiles=n_quantiles,
        cost_bps=cost_bps,
        min_assets=min_assets,
    )
    mono_report = cs_monotonicity(
        signal_panel,
        returns_panel,
        n_quantiles=n_quantiles,
        min_assets=min_assets,
    )
    all_pass = (
        ic_report["verdict"] == "PASS"
        and spread_report["verdict"] == "PASS"
        and mono_report["verdict"] == "PASS"
    )
    return {
        "ic": ic_report,
        "spread": spread_report,
        "monotonicity": mono_report,
        "overall_verdict": "PASS" if all_pass else "FAIL",
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
    _print_report(
        "planted edge",
        run_cross_sectional_tests(signal_panel, returns_panel),
    )
    noise_signal, noise_returns = _pure_noise_panel(seed=99)
    _print_report(
        "pure noise",
        run_cross_sectional_tests(noise_signal, noise_returns),
    )


if __name__ == "__main__":
    run_all_examples()
