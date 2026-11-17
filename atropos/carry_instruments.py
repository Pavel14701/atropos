"""Carry-family native instruments: funding alignment + risk envelope.

The ``carry`` plan (``atropos.instrument_map``) judges a carry book by
``stages_1_5`` + ``funding_alignment``; this module IS the carry-side
native instrument pair (plus the risk envelope read the plan's
identity promises):

* :func:`funding_alignment` -- does the ENGINE's funding bookkeeping
  match the settlement rate stream it claims to have seen?  Per hold:
  the settlement calendar count between entry and exit (half-open
  window ``[entry_ts, exit_ts)``) vs the count the engine applied, the
  accrued funding R vs the recomputation from the rate stream (perp
  convention: POSITIVE funding pays shorts -- a long hold accrues
  ``-sum(rates)``, a short hold ``+sum(rates)``), and the flip-cost
  share of the gross accrual.
* :func:`carry_risk_envelope` -- does the accrual stream survive its
  own drawdowns?  Cumulative-accrual max drawdown in R, the longest
  losing streak (consecutive negative-accrual periods), the tail
  (worst settlement raw, worst day net of flip costs), and the
  vol-targeting feasibility read (``mean / sigma >= 0.5``).  Breach
  gate: the max drawdown exceeds 3x the expected ANNUAL accrual
  (``mean * 252``) -- three years of expected carry burned in one
  drawdown -- or the mean accrual is non-positive (a stream that
  bleeds does not survive anything).

Both instruments follow the house honesty policy: absent inputs skip
their check with a note (never a silent zero), and nothing-checkable
runs read ``INCONCLUSIVE``, not ``ALIGNED`` / ``ENVELOPE_OK``.

Run ``python -m atropos.carry_instruments`` for a synthetic demo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from atropos._utils import (
    align_inputs,
    clean_1d,
    validate_int,
    validate_number,
)

__all__ = [
    "CarryRiskEnvelopeReport",
    "FundingAlignmentReport",
    "carry_risk_envelope",
    "funding_alignment",
]

_BPS = 10_000.0
"""Basis points per unit (the unit of ``rate_tolerance_bps``)."""

_PERIODS_PER_YEAR = 252
"""Settlements per year behind the expected-annual-accrual scaling."""

_MIN_OBS = 30
"""Minimum observations before an envelope verdict may fire."""

_DD_BREACH_MULTIPLE = 3.0
"""Max-drawdown multiples of the expected annual accrual that breach."""

_VOL_TARGET_RATIO = 0.5
"""mean / sigma of the daily accrual below which vol targeting fails."""

_ALIGNED = "ALIGNED"
_DRIFT = "DRIFT"
_INCONCLUSIVE = "INCONCLUSIVE"
_ENVELOPE_OK = "ENVELOPE_OK"
_ENVELOPE_BREACH = "ENVELOPE_BREACH"

_REQUIRED_HOLD_KEYS = ("entry_ts", "exit_ts", "side", "accrued_funding_r")
"""Every trade hold must carry these four fields (others are optional)."""


class FundingAlignmentReport(TypedDict):
    """Fixed report schema; every key is present on every outcome.

    ``drift_issues`` holds EVERY issue found, each tagged ``kind``:
    ``"settlement_count"`` (the engine's applied settlement count
    differs from the calendar between entry and exit) or
    ``"accrual_drift"`` (|accrued - recomputed| exceeded the
    tolerance in R).  ``flip_cost_share`` is NaN when no hold carried
    a flip cost or the gross accrual is zero (absence is not a free
    zero).  ``verdict`` is ``ALIGNED`` / ``DRIFT`` / ``INCONCLUSIVE``.
    """

    label: str
    n_holds: int
    n_settlements_expected: int
    n_aligned: int
    drift_issues: list[dict[str, Any]]
    flip_cost_share: float
    verdict: str
    notes: list[str]


class CarryRiskEnvelopeReport(TypedDict):
    """Fixed report schema; every key is present on every outcome.

    ``max_dd_r`` is the deepest peak-to-trough decline of the
    CUMULATIVE accrual curve (R); ``longest_losing_streak`` counts
    consecutive negative-accrual periods; ``worst_settlement_r`` is
    the raw most-negative single accrual, ``worst_day_r`` the same net
    of flip costs (equal to the settlement tail when none are given);
    ``vol_targeting_feasible`` applies the documented
    ``mean / sigma >= 0.5`` gate.  ``verdict`` is ``ENVELOPE_OK`` /
    ``ENVELOPE_BREACH`` / ``INCONCLUSIVE``.
    """

    label: str
    n: int
    mean_daily_r: float
    sigma_daily_r: float
    max_dd_r: float
    longest_losing_streak: int
    worst_settlement_r: float
    worst_day_r: float
    expected_annual_r: float
    mean_over_sigma: float
    vol_targeting_feasible: bool
    verdict: str
    notes: list[str]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _settlement_series(
    settlement_rates: Mapping[Any, Any] | pd.Series,
) -> pd.Series:
    """Sorted float rate series indexed by settlement timestamp.

    A ``pd.Series`` is used as-is; a mapping is rebuilt with its keys
    as the index.  Timestamps must share one comparable dtype across
    the index and the hold boundaries (int microseconds, epoch floats
    or ``pd.Timestamp`` all work; mixing them is the caller's bug).
    """
    if isinstance(settlement_rates, pd.Series):
        series = settlement_rates.astype(float)
    elif isinstance(settlement_rates, Mapping):
        series = pd.Series(
            np.asarray(list(settlement_rates.values()), dtype=float),
            index=pd.Index(list(settlement_rates.keys())),
        )
    else:
        raise ValueError(
            "settlement_rates must be a pandas Series or a mapping"
            " timestamp -> rate fraction, got"
            f" {type(settlement_rates).__name__}"
        )
    ordered = series.sort_index()
    if not np.all(np.isfinite(ordered.to_numpy(dtype=float))):
        raise ValueError(
            "settlement_rates must contain only finite rate fractions"
        )
    return ordered


def _normalize_holds(
    trade_holds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the holds to plain dicts, validating required fields."""
    holds: list[dict[str, Any]] = []
    for position, hold in enumerate(trade_holds):
        if not isinstance(hold, Mapping):
            raise ValueError(
                f"trade_holds[{position}] must be a mapping, got"
                f" {type(hold).__name__}"
            )
        missing = [key for key in _REQUIRED_HOLD_KEYS if key not in hold]
        if missing:
            raise ValueError(
                f"trade_holds[{position}] is missing required keys: {missing}"
            )
        holds.append(dict(hold))
    return holds


def _side_sign(side: Any) -> float:
    """+1.0 for a short hold, -1.0 for a long hold.

    Positive funding pays shorts (perp convention), so a SHORT hold's
    funding R is ``+sum(rates)`` and a LONG hold's ``-sum(rates)``.
    Accepted encodings: ``"long"`` / ``"short"`` (case-insensitive)
    or ``+1`` / ``-1``.
    """
    if isinstance(side, str):
        token = side.strip().lower()
        if token == "short":
            return 1.0
        if token == "long":
            return -1.0
    elif side in (-1, -1.0):
        return 1.0
    elif side in (1, 1.0):
        return -1.0
    raise ValueError(
        f"side must be 'long' / 'short' or +1 / -1, got {side!r}"
    )


def _positions_between(
    stamps: pd.Index, entry_ts: Any, exit_ts: Any
) -> np.ndarray:
    """Row positions of settlements in the half-open ``[entry, exit)``.

    A settlement exactly AT the exit timestamp belongs to the next
    holder; one exactly AT the entry belongs to this hold.
    """
    after_entry = np.asarray(stamps >= entry_ts, dtype=bool)
    before_exit = np.asarray(stamps < exit_ts, dtype=bool)
    return np.flatnonzero(after_entry & before_exit)


def _max_drawdown_r(accrued: np.ndarray) -> float:
    """Max peak-to-trough decline of the accrual cumsum (positive).

    The initial 0 counts as the first peak (an early negative accrual
    is already a drawdown), the same convention as the paper-protocol
    per-leg drawdown helper.
    """
    if accrued.size == 0:
        return float("nan")
    equity = np.concatenate([[0.0], np.cumsum(accrued)])
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks[1:] - equity[1:]))


def _longest_losing_streak(accrued: np.ndarray) -> int:
    """Longest run of consecutive negative accruals."""
    longest = 0
    current = 0
    for value in accrued.tolist():
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _scan_holds(
    stamps: pd.Index,
    rates_arr: np.ndarray,
    holds: list[dict[str, Any]],
    *,
    tolerance_r: float,
) -> tuple[int, int, list[dict[str, Any]], float, float, int, int]:
    """Run the per-hold checks against the settlement calendar.

    Returns ``(n_expected_total, n_aligned, issues, accrued_gross,
    flip_gross, n_with_count, n_with_flip)``.  A hold counts as
    aligned when every check that COULD run on it passed; each issue
    dict carries ``kind="settlement_count"`` or
    ``kind="accrual_drift"`` plus the hold's position.
    """
    n_expected_total = 0
    n_aligned = 0
    issues: list[dict[str, Any]] = []
    accrued_gross = 0.0
    flip_gross = 0.0
    n_with_count = 0
    n_with_flip = 0
    for position, hold in enumerate(holds):
        sign = _side_sign(hold["side"])
        accrued = validate_number(
            "accrued_funding_r", hold["accrued_funding_r"]
        )
        window = _positions_between(stamps, hold["entry_ts"], hold["exit_ts"])
        n_expected_total += int(window.size)
        recomputed = sign * float(rates_arr[window].sum())
        hold_issues = 0
        if "n_settlements" in hold:
            n_with_count += 1
            applied = validate_int(
                "n_settlements", hold["n_settlements"], minimum=0
            )
            if applied != int(window.size):
                issues.append(
                    {
                        "kind": "settlement_count",
                        "hold": position,
                        "expected": int(window.size),
                        "observed": applied,
                    }
                )
                hold_issues += 1
        if abs(accrued - recomputed) > tolerance_r:
            issues.append(
                {
                    "kind": "accrual_drift",
                    "hold": position,
                    "accrued_r": accrued,
                    "recomputed_r": recomputed,
                    "delta_r": accrued - recomputed,
                }
            )
            hold_issues += 1
        if hold_issues == 0:
            n_aligned += 1
        accrued_gross += abs(accrued)
        if "flip_cost_r" in hold:
            n_with_flip += 1
            flip_gross += abs(
                validate_number("flip_cost_r", hold["flip_cost_r"])
            )
    return (
        n_expected_total,
        n_aligned,
        issues,
        accrued_gross,
        flip_gross,
        n_with_count,
        n_with_flip,
    )


# ---------------------------------------------------------------------------
# public instruments
# ---------------------------------------------------------------------------
def funding_alignment(
    settlement_rates: Mapping[Any, Any] | pd.Series,
    trade_holds: Sequence[Mapping[str, Any]],
    *,
    rate_tolerance_bps: float = 1.0,
    label: str,
) -> FundingAlignmentReport:
    """Does the engine's funding bookkeeping match the rate stream?

    Three checks per hold (absent inputs skip their check with a
    note, never silently):

    * settlement calendar: settlements in ``[entry_ts, exit_ts)`` vs
      the engine-applied count, when the hold carries an
      ``n_settlements`` key -- a mismatch is a ``"settlement_count"``
      issue;
    * accrual recomputation: ``side_sign * sum(rates in window)`` vs
      the hold's ``accrued_funding_r`` -- ``|delta|`` beyond
      ``rate_tolerance_bps`` (converted to R) is an
      ``"accrual_drift"`` issue;
    * flip-cost share: ``sum(|flip_cost_r|) / sum(|accrued_funding_r|)``
      over the holds that carry a flip cost.

    Args:
        settlement_rates: the funding rate stream the ENGINE saw --
            a ``pd.Series`` or mapping timestamp -> rate FRACTION per
            settlement (sorted internally by timestamp).
        trade_holds: per-trade dicts with the required keys
            ``entry_ts`` / ``exit_ts`` / ``side`` / ``accrued_funding_r``
            and the optional keys ``n_settlements`` / ``flip_cost_r``.
        rate_tolerance_bps: accrued-vs-recomputed tolerance in basis
            points of R (default 1.0 bp = 0.0001 R); must be positive.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`FundingAlignmentReport`; the verdict is ``DRIFT`` when
        any issue fired, ``ALIGNED`` when every checkable hold passed,
        ``INCONCLUSIVE`` when nothing could be checked (no holds or an
        empty rate stream).

    Raises:
        ValueError: if ``settlement_rates`` is not a Series/mapping or
            holds non-finite rates; a hold is not a mapping or misses a
            required key; ``side`` uses an unknown encoding; a numeric
            field is not a finite number; or ``rate_tolerance_bps`` is
            not a positive finite float.

    Example::

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(7)
        >>> ou = np.zeros(90)
        >>> for pos in range(1, 90):
        ...     ou[pos] = (
        ...         ou[pos - 1]
        ...         + 0.30 * (0.0002 - ou[pos - 1])
        ...         + 0.00005 * rng.standard_normal()
        ...     )
        >>> stamps = pd.date_range("2026-01-01", periods=90, freq="8h")
        >>> rates = pd.Series(ou, index=stamps)
        >>> holds = [{
        ...     "entry_ts": stamps[0],
        ...     "exit_ts": stamps[-1] + pd.Timedelta(hours=8),
        ...     "side": "short",
        ...     "accrued_funding_r": float(ou.sum()),
        ...     "n_settlements": 90,
        ... }]
        >>> report = funding_alignment(rates, holds, label="OU demo")
        >>> report["verdict"], report["n_aligned"]
        ('ALIGNED', 1)
        >>> holds[0]["accrued_funding_r"] += 0.001
        >>> funding_alignment(rates, holds, label="OU drift")["verdict"]
        'DRIFT'
    """
    tolerance_bps = validate_number("rate_tolerance_bps", rate_tolerance_bps)
    if tolerance_bps <= 0:
        raise ValueError(
            "rate_tolerance_bps must be a positive finite float, got"
            f" {rate_tolerance_bps!r}"
        )
    tolerance_r = tolerance_bps / _BPS
    rates = _settlement_series(settlement_rates)
    stamps = rates.index
    rates_arr = rates.to_numpy(dtype=float)
    holds = _normalize_holds(trade_holds)
    notes: list[str] = []

    (
        n_expected_total,
        n_aligned,
        issues,
        accrued_gross,
        flip_gross,
        n_with_count,
        n_with_flip,
    ) = _scan_holds(stamps, rates_arr, holds, tolerance_r=tolerance_r)

    if holds and n_with_count < len(holds):
        notes.append(
            f"settlement-count check skipped for {len(holds) - n_with_count}"
            " hold(s) without an n_settlements key"
        )
    flip_cost_share = float("nan")
    if not n_with_flip:
        notes.append(
            "no hold carried a flip_cost_r; flip_cost_share is NaN"
            " (absence is not a zero share)"
        )
    elif accrued_gross > 0:
        flip_cost_share = flip_gross / accrued_gross
    else:
        notes.append(
            "gross accrual is zero; flip_cost_share is undefined (NaN)"
        )

    verdict = _ALIGNED
    if not holds or rates_arr.size == 0:
        verdict = _INCONCLUSIVE
        cause = (
            "no trade holds supplied"
            if not holds
            else "settlement rate stream is empty"
        )
        notes.append(
            f"status {_INCONCLUSIVE}: {cause}; the alignment check has"
            " nothing to judge"
        )
    elif issues:
        verdict = _DRIFT
        notes.append(
            f"{len(issues)} issue(s): the engine's funding bookkeeping"
            " diverges from the rate stream (count or accrual)"
        )
    return FundingAlignmentReport(
        label=label,
        n_holds=len(holds),
        n_settlements_expected=n_expected_total,
        n_aligned=n_aligned,
        drift_issues=issues,
        flip_cost_share=flip_cost_share,
        verdict=verdict,
        notes=notes,
    )


def carry_risk_envelope(
    accrued_r_stream: Any,
    *,
    flip_costs: Any = None,
    label: str,
) -> CarryRiskEnvelopeReport:
    """Does the accrual stream survive its own drawdowns?

    Reads the carry risk profile off the per-period accrued funding R
    stream: cumulative-accrual max drawdown, the longest run of
    negative-accrual periods, the tail (worst settlement raw and worst
    day net of flip costs), the mean / sigma ratio, and the vol
    targeting feasibility gate (``mean / sigma >= 0.5`` -- below it,
    the daily accrual signal is too noisy to hold a volatility
    target).

    Envelope gate (documented constants): ``ENVELOPE_BREACH`` when the
    max drawdown exceeds ``3x`` the expected annual accrual
    (``mean * 252`` -- three years of expected carry burned in one
    drawdown) or the mean accrual is non-positive; ``INCONCLUSIVE``
    below 30 observations; ``ENVELOPE_OK`` otherwise.

    Args:
        accrued_r_stream: per-period accrued funding in R (pandas or
            numpy 1-D array-like), chronological.
        flip_costs: optional per-period flip costs in R, aligned with
            the stream (non-finite rows drop pairwise); ``None`` reads
            the stream raw.
        label: free-text tag stored under ``report["label"]``.

    Returns:
        :class:`CarryRiskEnvelopeReport`; every key is present on
        every outcome (NaN where not computable).

    Raises:
        ValueError: if ``flip_costs`` is given and its length cannot
            align with the stream.

    Example::

        >>> import numpy as np
        >>> rng = np.random.default_rng(7)
        >>> ou = np.zeros(120)
        >>> for pos in range(1, 120):
        ...     ou[pos] = (
        ...         ou[pos - 1]
        ...         + 0.30 * (0.0002 - ou[pos - 1])
        ...         + 0.00005 * rng.standard_normal()
        ...     )
        >>> envelope = carry_risk_envelope(ou, label="OU funding")
        >>> envelope["verdict"], envelope["vol_targeting_feasible"]
        ('ENVELOPE_OK', True)
        >>> envelope["worst_day_r"] == envelope["worst_settlement_r"]
        True
        >>> carry_risk_envelope(ou[:20], label="short")["verdict"]
        'INCONCLUSIVE'
    """
    notes: list[str] = []
    if flip_costs is None:
        accrued = clean_1d(accrued_r_stream)
        costs = np.zeros(accrued.size, dtype=float)
    else:
        aligned = align_inputs(accrued_r_stream, flip_costs)
        accrued = np.asarray(aligned[0], dtype=float)
        costs = np.asarray(aligned[1], dtype=float)
        if accrued.size != costs.size:
            raise ValueError(
                "flip_costs could not align with the accrual stream"
                f" ({costs.size} vs {accrued.size} rows)"
            )
    n_obs = int(accrued.size)
    mean_daily = float(np.mean(accrued)) if n_obs else float("nan")
    sigma_daily = (
        float(np.std(accrued, ddof=1)) if n_obs >= 2 else float("nan")
    )
    mean_over_sigma = (
        mean_daily / sigma_daily
        if np.isfinite(sigma_daily) and sigma_daily > 0
        else float("nan")
    )
    feasible = bool(
        np.isfinite(mean_over_sigma) and mean_over_sigma >= _VOL_TARGET_RATIO
    )
    if n_obs >= 2 and sigma_daily == 0.0:
        notes.append("zero dispersion: mean / sigma is NaN, not zero")
    expected_annual = mean_daily * _PERIODS_PER_YEAR
    worst_settlement = float(np.min(accrued)) if n_obs else float("nan")
    worst_day = float(np.min(accrued + costs)) if n_obs else float("nan")
    max_dd = _max_drawdown_r(accrued)
    streak = _longest_losing_streak(accrued)

    verdict = _ENVELOPE_OK
    if n_obs < _MIN_OBS:
        verdict = _INCONCLUSIVE
        notes.append(
            f"status {_INCONCLUSIVE}: n = {n_obs} < {_MIN_OBS}"
            " observations; the envelope verdict is withheld"
        )
    elif mean_daily <= 0:
        verdict = _ENVELOPE_BREACH
        notes.append(
            f"mean accrual {mean_daily:.6g} R <= 0: the stream bleeds --"
            " its envelope is broken by construction"
        )
    elif max_dd > _DD_BREACH_MULTIPLE * expected_annual:
        verdict = _ENVELOPE_BREACH
        notes.append(
            f"max drawdown {max_dd:.4g} R exceeds {_DD_BREACH_MULTIPLE:g}x"
            f" the expected annual accrual {expected_annual:.4g} R"
        )
    else:
        notes.append(
            f"risk envelope gate: max drawdown {max_dd:.4g} R vs"
            f" {_DD_BREACH_MULTIPLE:g}x expected annual accrual"
            f" {expected_annual:.4g} R -- the stream survives its own"
            " drawdowns"
        )
    return CarryRiskEnvelopeReport(
        label=label,
        n=n_obs,
        mean_daily_r=mean_daily,
        sigma_daily_r=sigma_daily,
        max_dd_r=max_dd,
        longest_losing_streak=streak,
        worst_settlement_r=worst_settlement,
        worst_day_r=worst_day,
        expected_annual_r=expected_annual,
        mean_over_sigma=mean_over_sigma,
        vol_targeting_feasible=feasible,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def run_all_examples() -> None:
    """Demo: a planted OU funding world through both instruments.

    An Ornstein-Uhlenbeck funding rate (mean 2bp per settlement) is
    settled every 8h; a short hold over the whole span accrues exactly
    what the rate stream recomputes (ALIGNED), a perturbed accrual is
    DRIFT, and the rate stream's own risk envelope reads ENVELOPE_OK.
    Every printed verdict is asserted, so a regression crashes the
    demo instead of printing a wrong verdict quietly.
    """
    rng = np.random.default_rng(7)
    n_settle = 120
    ou = np.zeros(n_settle)
    for pos in range(1, n_settle):
        ou[pos] = (
            ou[pos - 1]
            + 0.30 * (0.0002 - ou[pos - 1])
            + 0.00005 * rng.standard_normal()
        )
    stamps = pd.date_range("2026-01-01", periods=n_settle, freq="8h")
    rates = pd.Series(ou, index=stamps)
    holds: list[dict[str, Any]] = [
        {
            "entry_ts": stamps[0],
            "exit_ts": stamps[-1] + pd.Timedelta(hours=8),
            "side": "short",
            "accrued_funding_r": float(ou.sum()),
            "n_settlements": n_settle,
            "flip_cost_r": 0.002,
        }
    ]

    print("=" * 72)
    print("carry_instruments demo: planted OU funding world")
    print("=" * 72)
    aligned = funding_alignment(rates, holds, label="OU demo")
    print(
        f"funding_alignment: verdict={aligned['verdict']}"
        f" n_holds={aligned['n_holds']}"
        f" n_settlements_expected={aligned['n_settlements_expected']}"
        f" n_aligned={aligned['n_aligned']}"
    )
    print(f"  flip_cost_share={aligned['flip_cost_share']:.4f}")
    assert aligned["verdict"] == _ALIGNED
    assert aligned["n_aligned"] == 1
    assert aligned["n_settlements_expected"] == n_settle
    assert 0.0 < aligned["flip_cost_share"] < 1.0

    drifted = [
        dict(
            holds[0],
            accrued_funding_r=holds[0]["accrued_funding_r"] + 0.001,
        )
    ]
    drift_report = funding_alignment(rates, drifted, label="OU drift")
    print(
        "funding_alignment (perturbed accrual):"
        f" verdict={drift_report['verdict']}"
    )
    print(f"  issues={drift_report['drift_issues']}")
    assert drift_report["verdict"] == _DRIFT
    assert drift_report["drift_issues"][0]["kind"] == "accrual_drift"

    envelope = carry_risk_envelope(ou, label="OU funding")
    print(
        f"carry_risk_envelope: verdict={envelope['verdict']}"
        f" n={envelope['n']}"
        f" max_dd_r={envelope['max_dd_r']:.6f}"
        f" streak={envelope['longest_losing_streak']}"
    )
    print(
        f"  mean/sigma={envelope['mean_over_sigma']:.2f}"
        f" vol_targeting_feasible={envelope['vol_targeting_feasible']}"
        f" worst_settlement_r={envelope['worst_settlement_r']:.6f}"
    )
    assert envelope["verdict"] == _ENVELOPE_OK
    assert envelope["vol_targeting_feasible"] is True
    assert envelope["worst_settlement_r"] < 0.0
    assert (
        carry_risk_envelope(ou[:20], label="short sample")["verdict"]
        == _INCONCLUSIVE
    )


if __name__ == "__main__":
    run_all_examples()
