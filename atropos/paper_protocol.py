"""Paper-vs-frozen parity checker: does paper execute the frozen spec?

Paper is a forward test of the IMPLEMENTATION, not the edge.  This
protocol answers: "does the paper engine execute the same strategy as
the frozen spec?" — a parity checker, not a strategy kill-protocol.
A ``PARITY_FAIL`` indicts the paper implementation or its feed; it
never indicts the hypothesis.

The stage machine runs in a FIXED order, per leg.  Parity and
operational stages are n-agnostic (any single issue kills regardless
of sample size); the n-gate fires only after they pass:

1. ``feed_granularity_check`` — the paper log's stamped
   ``bar_delta_min`` must equal the frozen spec's.
2. ``frozen_window_coverage`` — every paper ``entry_ts`` sits inside
   the spec's ``paper_window`` and the frozen frame covers it through
   its exits (1-bar tolerance on both bounds).  An EMPTY
   ``paper_window`` dict derives the window from the frozen frame's
   own span.
3. ``parity_check`` — nearest-entry matching per ``(coin, entry_ts)``
   within one bar, then per-pair comparisons: exit type through the
   frozen ``exit_type_normalization`` map, exit timestamp, ``pnl_r``,
   fee, size, funding rate, optional pnl components.  Paper trades
   without a frozen counterpart read as ``unmatched_entry`` or (when
   the coin is outside the frozen universe) ``universe_violation``.
4. ``operational_check`` — exactly TWO data-checks: ``latency``
   (needs the ``signal_ts`` column) and ``exit_direction`` (needs
   ``side`` + ``entry_price``/``exit_price``).  Anything that cannot
   run is recorded in ``checks_skipped``, never inferred.
5. ``n_check`` — ``n_matched >= min_n_trades`` or ``INSUFFICIENT_N``.
6. ``statistical_check`` — paired mean-drift bootstrap (``p_le0``)
   plus the paper-vs-frozen max-drawdown ratio gate.

Final-verdict contract:

* any stage 1/2/3 issue              -> ``PARITY_FAIL``
* any operational issue              -> ``OPERATIONAL_FAIL``
* clean stages 1-4, too few trades   -> ``INSUFFICIENT_N``
* mean-drift ``p_le0 < 0.05`` or a dd breach -> ``STATISTICAL_FAIL``
* statistical clean, at least one operational data-check RAN ->
  ``SURVIVES_TO_LIVE``: execution-quality evidence exists and passes
* statistical clean, BOTH data-checks skipped -> ``PARITY_PASS``: a
  parity-only attestation

``SURVIVES_TO_LIVE`` demands execution evidence (fill direction and
latency actually verified); ``PARITY_PASS`` attests the trade-level
match when the paper log carries no execution columns.  Both are
PASS-class; the distinction is evidence completeness, not correctness.

``dd_comparison`` is computed on EVERY verdict (including kills at any
earlier stage); the breach flag can only affect the verdict inside
stage 6.  Class-specific sections (``funding``,
``pnl_decomposition``) are opt-in via ``frozen_spec``: absence means
the related check is skipped and recorded, never inferred.

The function:

* :func:`run_paper_protocol` -- validate the paper log and the frozen
  spec, run the stage machine per leg, and aggregate the severity
  cascade into one :class:`PaperVerdict`.

Issue taxonomy: the parity report carries exactly 10 named issues,
the statistical report exactly one (``dd_divergence``); the ``facts``
dict pins the count.

Run ``python -m atropos.paper_protocol`` for a synthetic demo.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd

from atropos._utils import validate_int, validate_number

__all__ = ["LegVerdict", "PaperVerdict", "run_paper_protocol"]


VerdictToken = Literal[
    "PARITY_PASS",
    "PARITY_FAIL",
    "OPERATIONAL_FAIL",
    "STATISTICAL_FAIL",
    "INSUFFICIENT_N",
    "SURVIVES_TO_LIVE",
]
VerdictClass = Literal["PASS", "FAIL", "NEEDS_ANALYSIS", "INCONCLUSIVE"]


class LegVerdict(TypedDict):
    """Per-leg outcome of the paper-vs-frozen stage machine."""

    overall_verdict: VerdictToken
    verdict_class: VerdictClass
    informative: bool
    killed_by_stage: int | None
    kill_reason: str
    n_paper_trades: int
    n_frozen_matched: int
    parity_report: dict[str, Any]
    operational_report: dict[str, Any]
    statistical_report: dict[str, Any]
    interpretation: str
    next_step: str
    what_would_change: str
    portfolio_verdict: str
    feed_granularity: dict[str, Any]
    sizing_rule: dict[str, Any]
    funding_alignment: dict[str, Any] | None
    pnl_decomposition: dict[str, Any] | None
    dd_comparison: dict[str, Any]
    facts: dict[str, Any]
    label: str


class _LegStageResult(TypedDict):
    """Intermediate per-leg payload feeding the metadata layer."""

    verdict_token: VerdictToken
    killed_by_stage: int | None
    kill_reason: str
    n_paper_trades: int
    n_frozen_matched: int
    parity_report: dict[str, Any]
    operational_report: dict[str, Any]
    statistical_report: dict[str, Any]
    dd_comparison: dict[str, Any]
    feed_granularity: dict[str, Any]
    sizing_rule: dict[str, Any]
    funding_alignment: dict[str, Any] | None
    pnl_decomposition: dict[str, Any] | None


class PaperVerdict(LegVerdict):
    """Top-level outcome; ``per_leg`` is None for single-leg runs."""

    per_leg: dict[str, LegVerdict] | None


_PAPER_REQUIRED_COLUMNS = (
    "leg",
    "coin",
    "entry_ts",
    "exit_ts",
    "exit_type",
    "pnl_r",
    "sizing_rule",
)
_FROZEN_REQUIRED_COLUMNS = (
    "coin",
    "entry_ts",
    "exit_ts",
    "exit_type",
    "pnl_r",
)
_OPTIONAL_COLUMN_NAMES = (
    "signal_ts",
    "entry_price",
    "exit_price",
    "side",
    "fee_r",
    "size",
    "funding_rate",
    "pnl_delta",
    "pnl_gamma",
    "pnl_theta",
    "pnl_vega",
    "pnl_fee",
)
_PNL_COMPONENT_NAMES = ("delta", "gamma", "theta", "vega", "fee")
_PARITY_ISSUE_ORDER = (
    "feed_granularity_mismatch",
    "frozen_window_gap",
    "exit_mismatch",
    "fee_mismatch",
    "sizing_mismatch",
    "sizing_rule_mismatch",
    "funding_rate_mismatch",
    "pnl_component_mismatch",
    "universe_violation",
    "unmatched_entry",
)
_OPERATIONAL_ISSUE_ORDER = ("latency_breach", "exit_direction_violation")
_SEVERITY_CASCADE = (
    "PARITY_FAIL",
    "OPERATIONAL_FAIL",
    "STATISTICAL_FAIL",
    "INSUFFICIENT_N",
)
_SIZE_ABS_TOLERANCE = 1e-9
_SIZING_NOTE = "live sizing remains a policy decision outside this protocol"
_FACTS_NOTES = "10 parity + 1 statistical = 11 named issues"

# verdict -> (verdict_class, informative, interpretation template,
# next_step template, what_would_change template).  Placeholders are
# filled at finalize time from run facts: {label}, {min_n_trades},
# {n_paper_trades}, {n_frozen_matched}, {n_parity_issues},
# {n_operational_issues}, {n_<issue_name>} (pre-seeded at zero for
# every taxonomy name), {issue_summary} ("n_<issue>=<count>" join),
# {first_issue}, {paper_dd_r}, {frozen_dd_r}, {dd_ratio},
# {mean_drift_p_le0}, {dd_breach_ratio}, {n_skipped_checks},
# {n_run_checks}.  Filled strings are reported, never the bare
# templates.
_VERDICT_META: dict[str, tuple[str, bool, str, str, str]] = {
    "PARITY_PASS": (
        "PASS",
        True,
        (
            "parity-only attestation on label={label}: "
            "n_matched={n_frozen_matched} of "
            "n_paper_trades={n_paper_trades} trades match frozen within "
            "tolerance (0 named parity issues) and both operational "
            "data-checks were skipped ({n_skipped_checks} skips) — "
            "execution-quality evidence was unavailable"
        ),
        (
            "attach execution columns (signal_ts, side, entry_price, "
            "exit_price) to the paper log and rerun; a clean "
            "operational stage flips this to SURVIVES_TO_LIVE"
        ),
        "any operational data-check actually running and passing on a rerun",
    ),
    "PARITY_FAIL": (
        "FAIL",
        True,
        (
            "paper diverges from frozen on label={label}: "
            "{n_parity_issues} named parity issue(s) [{issue_summary}] "
            "over n_matched={n_frozen_matched} of "
            "n_paper_trades={n_paper_trades} trades"
        ),
        (
            "rerun after fixing {first_issue}; a clean rerun with "
            "n_matched >= {min_n_trades} flips this to PARITY_PASS or "
            "SURVIVES_TO_LIVE"
        ),
        (
            "rerun after fixing {first_issue}; a clean rerun with "
            "n_matched >= {min_n_trades} flips this to PARITY_PASS or "
            "SURVIVES_TO_LIVE"
        ),
    ),
    "OPERATIONAL_FAIL": (
        "FAIL",
        True,
        (
            "execution degraded on label={label}: parity clean "
            "({n_frozen_matched} of {n_paper_trades} trades matched) but "
            "{n_operational_issues} named operational issue(s) fired "
            "[{issue_summary}]"
        ),
        (
            "fix the paper execution path named in the issues (latency "
            "/ fill direction) and rerun; a clean rerun with "
            "n_matched >= {min_n_trades} flips this to SURVIVES_TO_LIVE"
        ),
        (
            "a rerun with clean latency and exit-direction checks and "
            "n_matched >= {min_n_trades}"
        ),
    ),
    "STATISTICAL_FAIL": (
        "NEEDS_ANALYSIS",
        True,
        (
            "distribution drift on label={label}: parity and execution "
            "clean ({n_frozen_matched} of {n_paper_trades} trades "
            "matched) but statistics failed [{issue_summary}]: "
            "paper_dd_r={paper_dd_r:.2f}R vs "
            "frozen_dd_r={frozen_dd_r:.2f}R (ratio {dd_ratio:.2f}, cap "
            "{dd_breach_ratio}), mean-drift "
            "p_le0={mean_drift_p_le0:.4f}"
        ),
        (
            "analyze the drift: {issue_summary}; a dd divergence means "
            "the paper risk profile diverges from frozen, a mean-drift "
            "failure means the paper engine books systematically LESS "
            "than the frozen spec"
        ),
        (
            "a rerun with dd ratio <= {dd_breach_ratio} and mean-drift "
            "p_le0 >= 0.05"
        ),
    ),
    "INSUFFICIENT_N": (
        "INCONCLUSIVE",
        False,
        (
            "parity and operational stages clean on label={label}, but "
            "n_matched={n_frozen_matched} < min_n_trades={min_n_trades} "
            "(of n_paper_trades={n_paper_trades}): parity holds, "
            "statistics cannot run"
        ),
        (
            "extend the paper window until n_matched >= {min_n_trades} "
            "and rerun"
        ),
        "n_matched >= {min_n_trades}",
    ),
    "SURVIVES_TO_LIVE": (
        "PASS",
        True,
        (
            "paper executes the frozen spec on label={label}: parity "
            "clean ({n_frozen_matched} of {n_paper_trades} trades "
            "matched), {n_run_checks} operational data-check(s) ran and "
            "passed, statistics pass (dd ratio {dd_ratio:.2f}, "
            "mean-drift p_le0={mean_drift_p_le0:.4f}) — this attests "
            "the implementation, not the edge"
        ),
        (
            "proceed to live tracking; spec correctness is a separate "
            "question (run_full_pipeline on the backtest feed)"
        ),
        (
            "nothing on this feed; live-vs-paper divergence is the "
            "next kill surface"
        ),
    ),
}


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _validate_paper_log(paper_log: pd.DataFrame) -> tuple[int, str]:
    """Contract-check the paper log (columns, attrs, sizing, side).

    Required columns: ``leg, coin, entry_ts, exit_ts, exit_type,
    pnl_r, sizing_rule``; required attribute
    ``paper_log.attrs["bar_delta_min"]`` (int minutes).  Optional
    columns are listed in ``_OPTIONAL_COLUMN_NAMES``; their absence
    sends the related check to ``checks_skipped``.

    Returns:
        The stamped bar granularity in minutes and the global
        ``sizing_rule`` value (read via ``.iloc[0]``).

    Raises:
        ValueError: on an empty log, missing columns/attr, a
            non-string ``sizing_rule`` value, or ``side`` values
            outside ``{"long", "short"}``.
    """
    if paper_log.empty:
        raise ValueError("paper_log is empty: nothing to parity-check")
    missing = [
        column_name
        for column_name in _PAPER_REQUIRED_COLUMNS
        if column_name not in paper_log.columns
    ]
    if missing:
        raise ValueError(f"paper_log is missing required columns {missing}")
    if "bar_delta_min" not in paper_log.attrs:
        raise ValueError(
            "paper_log.attrs['bar_delta_min'] is missing: the paper "
            "engine must stamp its bar granularity (int minutes) on "
            "the log"
        )
    bar_delta_min = validate_int(
        "paper_log.attrs['bar_delta_min']",
        paper_log.attrs["bar_delta_min"],
        minimum=1,
    )
    sizing_value = paper_log["sizing_rule"].iloc[0]
    if not isinstance(sizing_value, str) or not sizing_value.strip():
        raise ValueError(
            "paper_log['sizing_rule'] must carry a non-empty string "
            f"value at .iloc[0], got {sizing_value!r}"
        )
    if "side" in paper_log.columns:
        unknown_sides = sorted(
            {str(value) for value in paper_log["side"].tolist()}
            - {"long", "short"}
        )
        if unknown_sides:
            raise ValueError(
                "paper_log['side'] must only contain 'long'/'short', "
                f"got {unknown_sides}"
            )
    return bar_delta_min, sizing_value


def _validate_exit_rules(prefix: str, raw_exit_rules: Any) -> None:
    """``exit_rules`` must expose the three knobs (None allowed)."""
    if not isinstance(raw_exit_rules, dict):
        raise ValueError(f"{prefix}['exit_rules'] must be a dict")
    for key_name in ("mtm_bars", "tp_r", "stop_buffer_atr"):
        if key_name not in raw_exit_rules:
            raise ValueError(
                f"{prefix}['exit_rules'] is missing the key {key_name!r}"
            )
    if raw_exit_rules["mtm_bars"] is not None:
        validate_int(
            f"{prefix}['exit_rules']['mtm_bars']",
            raw_exit_rules["mtm_bars"],
            minimum=1,
        )
    for key_name in ("tp_r", "stop_buffer_atr"):
        value = raw_exit_rules[key_name]
        if value is not None:
            value_name = f"{prefix}['exit_rules']['{key_name}']"
            number = validate_number(value_name, value)
            if number < 0.0:
                raise ValueError(f"{value_name} must be >= 0, got {number}")


def _validate_normalization(prefix: str, raw_normalization: Any) -> None:
    """The alias map must be non-empty and unambiguous."""
    if not isinstance(raw_normalization, dict) or not raw_normalization:
        raise ValueError(
            f"{prefix}['exit_type_normalization'] must be a non-empty "
            "dict canonical -> alias list"
        )
    seen_aliases: dict[str, str] = {}
    for canonical, raw_aliases in raw_normalization.items():
        canonical_name = str(canonical)
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ValueError(
                f"{prefix}['exit_type_normalization'][{canonical_name!r}]"
                " must be a non-empty list of alias strings"
            )
        for raw_alias in raw_aliases:
            if not isinstance(raw_alias, str):
                raise ValueError(
                    f"{prefix}['exit_type_normalization']"
                    f"[{canonical_name!r}] must be a non-empty list of "
                    "alias strings"
                )
            owner = seen_aliases.get(raw_alias)
            if owner is not None and owner != canonical_name:
                raise ValueError(
                    f"{prefix}['exit_type_normalization']: alias "
                    f"{raw_alias!r} is claimed by both {owner!r} and "
                    f"{canonical_name!r}"
                )
            seen_aliases[raw_alias] = canonical_name


def _validate_fee_section(prefix: str, raw_fee: Any) -> None:
    """Fee schedule shape + the known ``round_trip`` literals."""
    if not isinstance(raw_fee, dict):
        raise ValueError(f"{prefix}['fee'] must be a dict")
    for key_name in ("taker_bps_per_leg", "round_trip"):
        if key_name not in raw_fee:
            raise ValueError(
                f"{prefix}['fee'] is missing the key {key_name!r}"
            )
    taker_bps = validate_number(
        f"{prefix}['fee']['taker_bps_per_leg']",
        raw_fee["taker_bps_per_leg"],
    )
    if taker_bps < 0.0:
        raise ValueError(
            f"{prefix}['fee']['taker_bps_per_leg'] must be >= 0, "
            f"got {taker_bps}"
        )
    if raw_fee["round_trip"] not in ("single", "double"):
        raise ValueError(
            f"{prefix}['fee']['round_trip'] must be 'single' or "
            f"'double', got {raw_fee['round_trip']!r}"
        )


def _component_short_name(prefix: str, raw_component: Any) -> str:
    """Normalize one decomposition component to its short name.

    ``"delta"`` and ``"pnl_delta"`` both name the ``pnl_delta`` column
    (the spec vocabulary accepts either spelling).
    """
    component = str(raw_component)
    if component.startswith("pnl_"):
        component = component[len("pnl_") :]
    if component not in _PNL_COMPONENT_NAMES:
        raise ValueError(
            f"{prefix}['pnl_decomposition']['components'] entry "
            f"{raw_component!r} is unknown: expected one of "
            f"{_PNL_COMPONENT_NAMES}"
        )
    return component


def _validate_paper_window(prefix: str, raw_window: Any) -> None:
    """``paper_window`` must parse to start <= end ({} = frozen span).

    An EMPTY ``paper_window`` dict is legal and means the frozen
    frame's own span defines the window; a non-empty dict must carry
    Timestamp-parseable ``start``/``end``.
    """
    if not isinstance(raw_window, dict):
        raise ValueError(
            f"{prefix}['paper_window'] must be a dict "
            "('start'/'end', pd.Timestamp-parseable; {} = the frozen "
            "frame's span)"
        )
    if not raw_window:
        return
    if "start" not in raw_window or "end" not in raw_window:
        raise ValueError(
            f"{prefix}['paper_window'] must carry 'start'/'end' "
            "(pd.Timestamp-parseable) or be empty ({} = frozen span)"
        )
    try:
        window_start = pd.Timestamp(raw_window["start"])
        window_end = pd.Timestamp(raw_window["end"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{prefix}['paper_window'] start/end must be "
            f"pd.Timestamp-parseable: {error}"
        ) from error
    if window_start > window_end:
        raise ValueError(
            f"{prefix}['paper_window'] start must not exceed end "
            f"({window_start} > {window_end})"
        )


def _validate_funding_section(prefix: str, raw_funding: Any) -> None:
    """Optional ``funding`` section: interval + rate tolerance."""
    if not isinstance(raw_funding, dict):
        raise ValueError(f"{prefix}['funding'] must be a dict")
    for key_name in ("settlement_interval_hours", "rate_tolerance_bps"):
        if key_name not in raw_funding:
            raise ValueError(
                f"{prefix}['funding'] is missing the key {key_name!r}"
            )
    validate_int(
        f"{prefix}['funding']['settlement_interval_hours']",
        raw_funding["settlement_interval_hours"],
        minimum=1,
    )
    rate_tolerance = validate_number(
        f"{prefix}['funding']['rate_tolerance_bps']",
        raw_funding["rate_tolerance_bps"],
    )
    if rate_tolerance < 0.0:
        raise ValueError(
            f"{prefix}['funding']['rate_tolerance_bps'] must be >= 0, "
            f"got {rate_tolerance}"
        )


def _validate_decomposition_section(
    prefix: str, raw_decomposition: Any
) -> None:
    """Optional ``pnl_decomposition`` section: known components."""
    if not isinstance(raw_decomposition, dict):
        raise ValueError(f"{prefix}['pnl_decomposition'] must be a dict")
    for key_name in ("enabled", "components", "tolerance_per_component_r"):
        if key_name not in raw_decomposition:
            raise ValueError(
                f"{prefix}['pnl_decomposition'] is missing the key "
                f"{key_name!r}"
            )
    if not isinstance(raw_decomposition["enabled"], bool):
        raise ValueError(
            f"{prefix}['pnl_decomposition']['enabled'] must be a bool"
        )
    components = raw_decomposition["components"]
    if not isinstance(components, list) or not components:
        raise ValueError(
            f"{prefix}['pnl_decomposition']['components'] must be a "
            "non-empty list[str]"
        )
    for raw_component in components:
        _component_short_name(prefix, raw_component)  # loud validation
    tolerance = validate_number(
        f"{prefix}['pnl_decomposition']['tolerance_per_component_r']",
        raw_decomposition["tolerance_per_component_r"],
    )
    if tolerance < 0.0:
        raise ValueError(
            f"{prefix}['pnl_decomposition']"
            "['tolerance_per_component_r'] must be >= 0, "
            f"got {tolerance}"
        )


def _validate_frozen_spec(leg: str, spec: dict[str, Any]) -> None:
    """Validate one leg's frozen spec loudly (shape + domains).

    Required keys: ``exit_rules``, ``exit_type_normalization``,
    ``fee``, ``universe``, ``sizing_rule``, ``bar_delta_min`` and
    ``paper_window``.  Optional: ``funding`` (carry) and
    ``pnl_decomposition`` (options).

    Raises:
        ValueError: on missing keys, wrong shapes, unknown
            ``round_trip`` literals, ambiguous alias claims and
            out-of-domain numbers; every message names the leg.
    """
    prefix = f"frozen_spec[{leg!r}]"
    if not isinstance(spec, dict):
        raise ValueError(f"{prefix} must be a dict")
    required = (
        "exit_rules",
        "exit_type_normalization",
        "fee",
        "universe",
        "sizing_rule",
        "bar_delta_min",
        "paper_window",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"{prefix} is missing required keys {missing}")
    _validate_exit_rules(prefix, spec["exit_rules"])
    _validate_normalization(prefix, spec["exit_type_normalization"])
    _validate_fee_section(prefix, spec["fee"])
    if not isinstance(spec["universe"], list):
        raise ValueError(f"{prefix}['universe'] must be a list[str]")
    if not all(isinstance(name, str) for name in spec["universe"]):
        raise ValueError(f"{prefix}['universe'] must be a list[str]")
    if (
        not isinstance(spec["sizing_rule"], str)
        or not spec["sizing_rule"].strip()
    ):
        raise ValueError(f"{prefix}['sizing_rule'] must be a non-empty str")
    validate_int(
        f"{prefix}['bar_delta_min']", spec["bar_delta_min"], minimum=1
    )
    _validate_paper_window(prefix, spec["paper_window"])
    if "funding" in spec:
        _validate_funding_section(prefix, spec["funding"])
    if "pnl_decomposition" in spec:
        _validate_decomposition_section(prefix, spec["pnl_decomposition"])


def _validate_leg_frames(
    leg: str,
    paper_leg: pd.DataFrame,
    frozen_leg: pd.DataFrame,
    spec: dict[str, Any],
) -> None:
    """Per-leg frame contract (columns, stamps, section demands).

    Raises:
        ValueError: on an empty frozen frame, missing required frozen
            columns, NaN entry/exit stamps, or missing
            ``funding_rate`` / ``pnl_*`` columns when the spec's
            ``funding`` / ``pnl_decomposition`` sections demand them.
    """
    prefix = f"frozen_outcomes[{leg!r}]"
    if frozen_leg.empty:
        raise ValueError(
            f"{prefix} is empty: the frozen window cannot be spanned"
        )
    missing = [
        column_name
        for column_name in _FROZEN_REQUIRED_COLUMNS
        if column_name not in frozen_leg.columns
    ]
    if missing:
        raise ValueError(f"{prefix} is missing required columns {missing}")
    for frame_name, frame in (
        ("paper_log leg", paper_leg),
        ("frozen_outcomes leg", frozen_leg),
    ):
        for stamp_column in ("entry_ts", "exit_ts"):
            if bool(frame[stamp_column].isna().any()):
                raise ValueError(
                    f"{frame_name} {leg!r}: {stamp_column} carries NaN "
                    "stamps; parity matching needs real timestamps"
                )
    raw_decomposition = spec.get("pnl_decomposition")
    needs_components = isinstance(raw_decomposition, dict) and bool(
        raw_decomposition["enabled"]
    )
    if needs_components:
        for frame_name, frame in (
            ("paper_log leg", paper_leg),
            ("frozen_outcomes leg", frozen_leg),
        ):
            missing_components = [
                f"pnl_{component_name}"
                for component_name in _PNL_COMPONENT_NAMES
                if f"pnl_{component_name}" not in frame.columns
            ]
            if missing_components:
                raise ValueError(
                    f"{frame_name} {leg!r} is missing the pnl "
                    f"decomposition columns {missing_components} "
                    "(pnl_decomposition.enabled)"
                )
    if "funding" in spec:
        for frame_name, frame in (
            ("paper_log leg", paper_leg),
            ("frozen_outcomes leg", frozen_leg),
        ):
            if "funding_rate" not in frame.columns:
                raise ValueError(
                    f"{frame_name} {leg!r} is missing the 'funding_rate' "
                    "column (frozen_spec funding section present)"
                )


def _exit_alias_map(spec: dict[str, Any]) -> dict[str, str]:
    """Build the alias -> canonical exit-type map (validated upstream).

    Canonical names map to themselves, so a log may carry either the
    canonical token or any registered alias.
    """
    alias_map: dict[str, str] = {}
    raw_normalization = spec["exit_type_normalization"]
    for canonical, raw_aliases in raw_normalization.items():
        canonical_name = str(canonical)
        alias_map[canonical_name] = canonical_name
        for raw_alias in raw_aliases:
            alias_map[str(raw_alias)] = canonical_name
    return alias_map


def _normalize_exit_type(
    token: str, alias_map: dict[str, str], leg: str
) -> str:
    """Resolve one exit-type token through the frozen alias map.

    A token found in NO alias list is a hard ``ValueError`` — never a
    silent mismatch (a normalization gap must be fixed in the spec,
    not papered over by the checker).
    """
    canonical = alias_map.get(str(token))
    if canonical is None:
        raise ValueError(
            f"leg {leg!r}: exit_type token {token!r} is not listed in "
            "exit_type_normalization (unknown tokens are errors, not "
            "silent mismatches)"
        )
    return canonical


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _max_drawdown_r(pnls: np.ndarray) -> float:
    """Max peak-to-trough decline of the pnl cumsum (positive).

    The cumsum runs over the FRAME'S ROW ORDER (as recorded, NOT
    sorted by time): paper journals append in resolution order which
    legitimately differs from the frozen sim's sequence, and that
    order divergence is exactly the ``dd_divergence`` signal.  The
    initial equity of 0 counts as the first peak, so an early loss
    already counts as a drawdown.
    """
    if pnls.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(pnls)])
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks[1:] - equity[1:]))


def _max_losing_streak(pnls: np.ndarray) -> int:
    """Longest run of consecutive ``pnl_r < 0`` in frame row order."""
    longest = 0
    current = 0
    for value in pnls.tolist():
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _paired_bootstrap_p(
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> float:
    """One-sided paired-bootstrap pseudo-p that ``mean(a) <= mean(b)``.

    Joint paired bootstrap: non-finite pairs are dropped, the
    surviving INDEX PAIRS are resampled jointly through
    ``np.random.default_rng(seed).integers`` and the statistic is
    ``mean(arm_a[idx]) - mean(arm_b[idx])``; ``p_le0`` is the fraction
    of draws with ``diff <= 0``.  Same inputs + seed give IDENTICAL
    p-values to ``atropos.ranking.paired_bootstrap_diff``'s ``p_le0``.

    Pinned against atropos.ranking.paired_bootstrap_diff @ commit
    5afc184 (convention 1:1).  If that implementation changes, this
    must be re-pinned via test_local_bootstrap_matches_ranking.
    """
    n_boot = validate_int("n_boot", n_boot, minimum=1)
    seed = validate_int("seed", seed, minimum=0)
    values_a = np.asarray(arm_a, dtype=np.float64).ravel()
    values_b = np.asarray(arm_b, dtype=np.float64).ravel()
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"arm_a and arm_b must have equal length, got "
            f"{values_a.size} and {values_b.size}"
        )
    rng = np.random.default_rng(seed)
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a, values_b = values_a[finite], values_b[finite]
    size = int(values_a.size)
    if size == 0:
        return 1.0
    diffs = values_a - values_b
    indices = rng.integers(0, size, size=(n_boot, size))
    draws = diffs[indices].mean(axis=1)
    return float((draws <= 0).mean())


# ---------------------------------------------------------------------------
# issue bookkeeping + frame readers
# ---------------------------------------------------------------------------
def _record_issue(
    issues: dict[str, dict[str, Any]], issue_name: str, label: str
) -> None:
    """Count one affected trade and keep up to three examples."""
    entry = issues.setdefault(issue_name, {"n_affected": 0, "examples": []})
    entry["n_affected"] += 1
    if len(entry["examples"]) < 3:
        entry["examples"].append(label)


def _record_issue_bulk(
    issues: dict[str, dict[str, Any]],
    issue_name: str,
    count: int,
    labels: list[str],
) -> None:
    """Count several affected trades at once (up to three examples)."""
    entry = issues.setdefault(issue_name, {"n_affected": 0, "examples": []})
    entry["n_affected"] += count
    for label in labels[: 3 - len(entry["examples"])]:
        entry["examples"].append(label)


def _issue_list(
    issues: dict[str, dict[str, Any]], order: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Materialize the issue dict in the taxonomy's canonical order."""
    return [
        {
            "name": issue_name,
            "n_affected": issues[issue_name]["n_affected"],
            "examples": issues[issue_name]["examples"],
        }
        for issue_name in order
        if issue_name in issues
    ]


def _string_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Read one column as plain strings (loud on a missing column)."""
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is missing from the frame")
    return [str(value) for value in frame[column].tolist()]


def _timestamp_values(frame: pd.DataFrame, column: str) -> list[pd.Timestamp]:
    """Read one column as ``pd.Timestamp`` values (loud on missing)."""
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is missing from the frame")
    return [pd.Timestamp(value) for value in frame[column].tolist()]


def _float_values(frame: pd.DataFrame, column: str) -> list[float] | None:
    """Read one column as floats; ``None`` marks the column absent."""
    if column not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column])
    return [float(value) for value in numeric.tolist()]


def _trade_labels(paper_leg: pd.DataFrame) -> list[str]:
    """``coin@entry_ts`` labels for the parity issue examples."""
    coins = _string_values(paper_leg, "coin")
    entries = _timestamp_values(paper_leg, "entry_ts")
    return [
        f"{coin_value}@{entry_stamp}"
        for coin_value, entry_stamp in zip(coins, entries, strict=True)
    ]


# ---------------------------------------------------------------------------
# per-leg stage checks
# ---------------------------------------------------------------------------
def _window_coverage_issue(
    paper_leg: pd.DataFrame,
    frozen_leg: pd.DataFrame,
    spec: dict[str, Any],
    labels: list[str],
) -> dict[str, dict[str, Any]] | None:
    """Stage 2: paper entries inside the window, frozen span covers it.

    Both bounds tolerate one bar (the spec's ``bar_delta_min``); the
    frozen frame must cover the window through its EXITS (frozen min
    entry at/before the start, frozen max exit at/after the end).  An
    empty ``paper_window`` derives the window from the frozen frame's
    own span.  Any out-of-window paper entry or a range that does not
    span the window becomes the ``frozen_window_gap`` issue.
    """
    frozen_entries = _timestamp_values(frozen_leg, "entry_ts")
    frozen_exits = _timestamp_values(frozen_leg, "exit_ts")
    raw_window = spec["paper_window"]
    if raw_window:
        window_start = pd.Timestamp(raw_window["start"])
        window_end = pd.Timestamp(raw_window["end"])
    else:
        # empty paper_window: the frozen frame's own span IS the window
        window_start = min(frozen_entries)
        window_end = max(frozen_exits)
    one_bar = pd.Timedelta(minutes=int(spec["bar_delta_min"]))
    paper_entries = _timestamp_values(paper_leg, "entry_ts")
    issues: dict[str, dict[str, Any]] = {}
    for position, entry_stamp in enumerate(paper_entries):
        if entry_stamp < window_start or entry_stamp > window_end:
            _record_issue(issues, "frozen_window_gap", labels[position])
    examples: list[str] = []
    span_violations = 0
    frozen_min = min(frozen_entries)
    if frozen_min > window_start + one_bar:
        span_violations += 1
        examples.append(f"frozen_min@{frozen_min}")
    if max(frozen_exits) < window_end - one_bar:
        span_violations += 1
        examples.append(f"frozen_max_exit@{max(frozen_exits)}")
    if not issues and not span_violations:
        return None
    slot = issues.setdefault(
        "frozen_window_gap", {"n_affected": 0, "examples": []}
    )
    slot["n_affected"] += span_violations
    for example in examples[: 3 - len(slot["examples"])]:
        slot["examples"].append(example)
    return issues


class _SideArrays(TypedDict):
    """Per-frame column arrays consumed by the parity comparisons."""

    exit_types: list[str]
    exits: list[pd.Timestamp]
    pnls: list[float]
    fees: list[float] | None
    sizes: list[float] | None
    rates: list[float]
    components: dict[str, list[float]]


class _ParityInputs(TypedDict):
    """Stage-3 spec inputs + columns whose absence skips a check."""

    universe: list[str]
    funding_enabled: bool
    rate_tolerance_bps: float
    decomposition_enabled: bool
    spec_components: list[str]
    component_tolerance: float
    checks_skipped: set[str]


def _collect_side_arrays(
    frame: pd.DataFrame, spec_components: list[str]
) -> _SideArrays:
    """Extract one frame's parity columns (optionals NaN-free None)."""
    components = {
        component_name: _float_values(frame, f"pnl_{component_name}") or []
        for component_name in spec_components
    }
    return {
        "exit_types": _string_values(frame, "exit_type"),
        "exits": _timestamp_values(frame, "exit_ts"),
        "pnls": _float_values(frame, "pnl_r") or [],
        "fees": _float_values(frame, "fee_r"),
        "sizes": _float_values(frame, "size"),
        "rates": _float_values(frame, "funding_rate") or [],
        "components": components,
    }


def _collect_parity_inputs(
    spec: dict[str, Any],
    paper_leg: pd.DataFrame,
    frozen_leg: pd.DataFrame,
) -> _ParityInputs:
    """Resolve the opt-in spec sections + record the skipped checks.

    A section's absence (no ``funding``, no enabled
    ``pnl_decomposition``) or a missing optional column sends the
    related comparison to ``checks_skipped`` — never inferred.
    """
    checks_skipped: set[str] = set()
    spec_funding = spec.get("funding")
    funding_enabled = spec_funding is not None
    if not funding_enabled:
        checks_skipped.add(
            "funding_rate_comparison (frozen_spec has no funding section)"
        )
    raw_decomposition = spec.get("pnl_decomposition")
    decomposition_enabled = isinstance(raw_decomposition, dict) and bool(
        raw_decomposition["enabled"]
    )
    spec_components: list[str] = []
    component_tolerance = 0.0
    if decomposition_enabled and isinstance(raw_decomposition, dict):
        spec_components = [
            str(name) for name in raw_decomposition["components"]
        ]
        component_tolerance = float(
            raw_decomposition["tolerance_per_component_r"]
        )
    else:
        checks_skipped.add(
            "pnl_component_comparison (pnl_decomposition absent or disabled)"
        )
    if (
        _float_values(paper_leg, "fee_r") is None
        or _float_values(frozen_leg, "fee_r") is None
    ):
        checks_skipped.add("fee_comparison (fee_r absent)")
    if (
        _float_values(paper_leg, "size") is None
        or _float_values(frozen_leg, "size") is None
    ):
        checks_skipped.add("size_comparison (size absent)")
    universe = [str(name) for name in spec["universe"]]
    if not universe:
        checks_skipped.add("universe_check (empty frozen universe)")
    return {
        "universe": universe,
        "funding_enabled": funding_enabled,
        "rate_tolerance_bps": (
            float(spec_funding["rate_tolerance_bps"])
            if spec_funding is not None
            else 0.0
        ),
        "decomposition_enabled": decomposition_enabled,
        "spec_components": spec_components,
        "component_tolerance": component_tolerance,
        "checks_skipped": checks_skipped,
    }


def _exit_record_broken(
    paper_side: _SideArrays,
    frozen_side: _SideArrays,
    paper_position: int,
    frozen_position: int,
    *,
    alias_map: dict[str, str],
    leg: str,
    tolerance: pd.Timedelta,
    parity_tolerance_r: float,
) -> bool:
    """Exit-record divergence: type token, exit ts, or pnl_r drift.

    All three read through one ``exit_mismatch`` bucket — the exit
    record as a whole disagrees with the frozen spec's.
    """
    paper_canonical = _normalize_exit_type(
        paper_side["exit_types"][paper_position], alias_map, leg
    )
    frozen_canonical = _normalize_exit_type(
        frozen_side["exit_types"][frozen_position], alias_map, leg
    )
    if paper_canonical != frozen_canonical:
        return True
    if (
        abs(
            paper_side["exits"][paper_position]
            - frozen_side["exits"][frozen_position]
        )
        > tolerance
    ):
        return True
    paper_pnl = paper_side["pnls"][paper_position]
    frozen_pnl = frozen_side["pnls"][frozen_position]
    return not (
        math.isfinite(paper_pnl)
        and math.isfinite(frozen_pnl)
        and abs(paper_pnl - frozen_pnl) <= parity_tolerance_r
    )


def _pair_extra_issues(
    paper_side: _SideArrays,
    frozen_side: _SideArrays,
    paper_position: int,
    frozen_position: int,
    *,
    inputs: _ParityInputs,
    paper_canonical: str,
    fee_tolerance_bps: float,
) -> list[str]:
    """Optional-comparison issues fired by ONE matched pair.

    Covers the fee, size, funding-rate and pnl-component comparisons
    (each runs only when its evidence exists on BOTH frames).
    """
    fired: list[str] = []
    paper_fees = paper_side["fees"]
    frozen_fees = frozen_side["fees"]
    if paper_fees is not None and frozen_fees is not None:
        paper_fee = paper_fees[paper_position]
        frozen_fee = frozen_fees[frozen_position]
        if not (
            math.isfinite(paper_fee)
            and math.isfinite(frozen_fee)
            and abs(paper_fee - frozen_fee) <= fee_tolerance_bps * 1e-4
        ):
            fired.append("fee_mismatch")
    paper_sizes = paper_side["sizes"]
    frozen_sizes = frozen_side["sizes"]
    if paper_sizes is not None and frozen_sizes is not None:
        paper_size = paper_sizes[paper_position]
        frozen_size = frozen_sizes[frozen_position]
        if not (
            math.isfinite(paper_size)
            and math.isfinite(frozen_size)
            and abs(paper_size - frozen_size) <= _SIZE_ABS_TOLERANCE
        ):
            fired.append("sizing_mismatch")
    if inputs["funding_enabled"] and paper_canonical == "funding_payment":
        paper_rate = paper_side["rates"][paper_position]
        frozen_rate = frozen_side["rates"][frozen_position]
        if not (
            math.isfinite(paper_rate)
            and math.isfinite(frozen_rate)
            and abs(paper_rate - frozen_rate)
            < inputs["rate_tolerance_bps"] * 1e-4
        ):
            # rates are fractions; the tolerance converts bps -> fraction
            fired.append("funding_rate_mismatch")
    if inputs["decomposition_enabled"]:
        for component_name in inputs["spec_components"]:
            paper_value = paper_side["components"][component_name][
                paper_position
            ]
            frozen_value = frozen_side["components"][component_name][
                frozen_position
            ]
            if (
                math.isfinite(paper_value)
                and math.isfinite(frozen_value)
                and abs(paper_value - frozen_value)
                <= inputs["component_tolerance"]
            ):
                continue
            fired.append("pnl_component_mismatch")
            break
    return fired


def _run_parity_stage(
    leg: str,
    paper_leg: pd.DataFrame,
    frozen_leg: pd.DataFrame,
    spec: dict[str, Any],
    *,
    alias_map: dict[str, str],
    bar_delta_min: int,
    paper_sizing_value: str,
    parity_tolerance_r: float,
    fee_tolerance_bps: float,
    labels: list[str],
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Stage 3: one-to-one trade matching + per-pair comparisons.

    Matching: every paper trade (frame order) takes the nearest
    not-yet-matched frozen trade of the same coin whose ``entry_ts``
    lies within one bar.  The ``min()`` key is
    ``(abs_entry_diff, frozen_entry_ts)``, so equidistant candidates
    resolve to the EARLIER frozen entry — deterministic, and
    independent of row order beyond ties.  Matching is one-to-one.

    ``exit_mismatch`` is the bucket for any divergence in the exit
    record: normalized exit type, exit timestamp beyond one bar, or a
    ``pnl_r`` difference beyond ``parity_tolerance_r``.

    Returns:
        The ``parity_report`` dict and the matched
        ``(paper_position, frozen_position)`` pairs in paper frame
        order (consumed by the stage-6 bootstrap).
    """
    issues: dict[str, dict[str, Any]] = {}
    tolerance = pd.Timedelta(minutes=bar_delta_min)
    if str(paper_sizing_value) != str(spec["sizing_rule"]):
        _record_issue_bulk(
            issues, "sizing_rule_mismatch", len(labels), labels
        )
    inputs = _collect_parity_inputs(spec, paper_leg, frozen_leg)
    paper_side = _collect_side_arrays(paper_leg, inputs["spec_components"])
    frozen_side = _collect_side_arrays(frozen_leg, inputs["spec_components"])
    paper_coins = _string_values(paper_leg, "coin")
    paper_entries = _timestamp_values(paper_leg, "entry_ts")
    frozen_coins = _string_values(frozen_leg, "coin")
    frozen_entries = _timestamp_values(frozen_leg, "entry_ts")
    available: dict[str, list[tuple[int, pd.Timestamp]]] = {}
    for frozen_position, (coin_value, entry_stamp) in enumerate(
        zip(frozen_coins, frozen_entries, strict=True)
    ):
        available.setdefault(coin_value, []).append(
            (frozen_position, entry_stamp)
        )
    matched_pairs: list[tuple[int, int]] = []
    for paper_position, (coin_value, entry_stamp) in enumerate(
        zip(paper_coins, paper_entries, strict=True)
    ):
        label = labels[paper_position]
        candidates = available.get(coin_value, [])
        eligible = [
            (abs(entry_stamp - frozen_stamp), frozen_stamp, frozen_position)
            for frozen_position, frozen_stamp in candidates
            if abs(entry_stamp - frozen_stamp) <= tolerance
        ]
        if not eligible:
            if inputs["universe"] and coin_value not in inputs["universe"]:
                _record_issue(issues, "universe_violation", label)
            else:
                _record_issue(issues, "unmatched_entry", label)
            continue
        _, chosen_stamp, frozen_position = min(
            eligible, key=lambda candidate: (candidate[0], candidate[1])
        )
        candidates.remove((frozen_position, chosen_stamp))
        matched_pairs.append((paper_position, frozen_position))
        if _exit_record_broken(
            paper_side,
            frozen_side,
            paper_position,
            frozen_position,
            alias_map=alias_map,
            leg=leg,
            tolerance=tolerance,
            parity_tolerance_r=parity_tolerance_r,
        ):
            _record_issue(issues, "exit_mismatch", label)
        paper_canonical = _normalize_exit_type(
            paper_side["exit_types"][paper_position], alias_map, leg
        )
        for issue_name in _pair_extra_issues(
            paper_side,
            frozen_side,
            paper_position,
            frozen_position,
            inputs=inputs,
            paper_canonical=paper_canonical,
            fee_tolerance_bps=fee_tolerance_bps,
        ):
            _record_issue(issues, issue_name, label)
    parity_report: dict[str, Any] = {
        "stage_verdict": "PARITY_FAIL" if issues else "PARITY_PASS",
        "n_matched": len(matched_pairs),
        "issues": _issue_list(issues, _PARITY_ISSUE_ORDER),
        "checks_skipped": sorted(inputs["checks_skipped"]),
    }
    return parity_report, matched_pairs


def _run_operational_stage(
    leg: str,
    paper_leg: pd.DataFrame,
    alias_map: dict[str, str],
    latency_max_sec: float,
    labels: list[str],
) -> dict[str, Any]:
    """Stage 4: exactly TWO operational data-checks.

    ``latency`` runs only when ``signal_ts`` is present; the bound is
    ``(entry_ts - signal_ts) < latency_max_sec``, and a NEGATIVE
    latency (signal stamped at-or-after the entry) is a lookahead
    defect that counts as a breach even though the plain bound would
    pass it.  ``exit_direction`` runs only when ``side`` and
    ``entry_price``/``exit_price`` are present: a long ``stop`` exit
    must sit below the entry (short mirrored) and a long ``tp`` exit
    above it; canonical tokens that name neither stop nor profit
    (``time_exit``, ``funding_payment``, ...) are unconstrained.
    Trades with non-finite prices are not checkable and are skipped.
    """
    issues: dict[str, dict[str, Any]] = {}
    checks_run: list[str] = []
    checks_skipped: list[str] = []
    entries = _timestamp_values(paper_leg, "entry_ts")
    if "signal_ts" in paper_leg.columns:
        checks_run.append("latency")
        signals = _timestamp_values(paper_leg, "signal_ts")
        raw_signals = list(paper_leg["signal_ts"])
        for position, entry_stamp in enumerate(entries):
            signal_stamp = signals[position]
            if pd.isna(raw_signals[position]):
                continue
            latency = (entry_stamp - signal_stamp).total_seconds()
            if latency < 0.0 or latency >= latency_max_sec:
                _record_issue(issues, "latency_breach", labels[position])
    else:
        checks_skipped.append("latency (no signal_ts column)")
    entry_prices = _float_values(paper_leg, "entry_price")
    exit_prices = _float_values(paper_leg, "exit_price")
    if (
        "side" in paper_leg.columns
        and entry_prices is not None
        and exit_prices is not None
    ):
        checks_run.append("exit_direction")
        exit_direction_issues = _exit_direction_issues(
            paper_leg,
            labels,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            alias_map=alias_map,
            leg=leg,
        )
        for label in exit_direction_issues:
            _record_issue(issues, "exit_direction_violation", label)
    else:
        checks_skipped.append("exit_direction (no side/price columns)")
    stage_verdict = "OPERATIONAL_FAIL" if issues else "OPERATIONAL_PASS"
    return {
        "stage_verdict": stage_verdict,
        "issues": _issue_list(issues, _OPERATIONAL_ISSUE_ORDER),
        "checks_run": checks_run,
        "checks_skipped": checks_skipped,
    }


def _exit_direction_issues(
    paper_leg: pd.DataFrame,
    labels: list[str],
    *,
    entry_prices: list[float],
    exit_prices: list[float],
    alias_map: dict[str, str],
    leg: str,
) -> list[str]:
    """Labels whose exit fill sits on the wrong side of the entry.

    Only stop-like and profit-like canonical tokens carry a direction
    constraint (long ``stop`` below the entry / long ``tp`` above it,
    short mirrored); ``time_exit``, ``funding_payment`` and other
    unconstrained canonicals skip the check.  Trades with non-finite
    prices are not checkable and are skipped.
    """
    paper_sides = _string_values(paper_leg, "side")
    paper_exit_types = _string_values(paper_leg, "exit_type")
    offending: list[str] = []
    for position, entry_price in enumerate(entry_prices):
        exit_price = exit_prices[position]
        if not math.isfinite(entry_price) or not math.isfinite(exit_price):
            continue
        canonical = _normalize_exit_type(
            paper_exit_types[position], alias_map, leg
        )
        adverse = "stop" in canonical
        favorable = "tp" in canonical or "profit" in canonical
        if not adverse and not favorable:
            continue
        if paper_sides[position] == "long":
            direction_ok = (
                exit_price < entry_price
                if adverse
                else exit_price > entry_price
            )
        else:
            direction_ok = (
                exit_price > entry_price
                if adverse
                else exit_price < entry_price
            )
        if not direction_ok:
            offending.append(labels[position])
    return offending


def _run_statistical_stage(
    matched_pairs: list[tuple[int, int]],
    paper_pnls: list[float],
    frozen_pnls: list[float],
    *,
    paper_streak: int,
    frozen_streak: int,
    n_matched: int,
    n_bootstrap: int,
    seed: int,
    dd_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Stage 6: paired mean-drift bootstrap + drawdown-ratio gate.

    Arm order is PINNED (``arm_a`` = frozen, ``arm_b`` = paper), so
    the resampled statistic is ``mean(frozen) - mean(paper)`` and a
    small ``p_le0`` flags a paper engine that books significantly
    LESS than its own frozen simulation (frozen - paper consistently
    positive).  Mean drift is reported through ``mean_drift_p_le0`` —
    it is the p-value path, NOT a named issue; ``dd_divergence`` is
    the only named statistical issue.
    """
    matched_paper = [
        paper_pnls[paper_position] for paper_position, _ in matched_pairs
    ]
    matched_frozen = [
        frozen_pnls[frozen_position] for _, frozen_position in matched_pairs
    ]
    drift_p = _paired_bootstrap_p(
        np.asarray(matched_frozen, dtype=np.float64),
        np.asarray(matched_paper, dtype=np.float64),
        n_boot=n_bootstrap,
        seed=seed,
    )
    dd_breach = bool(dd_comparison["breach"])
    return {
        "mean_drift_p_le0": drift_p,
        "n_matched": n_matched,
        "paper_max_streak": paper_streak,
        "frozen_max_streak": frozen_streak,
        "streak_ratio": _streak_ratio(paper_streak, frozen_streak),
        "issues": ["dd_divergence"] if dd_breach else [],
        "sizing_note": _SIZING_NOTE,
    }


def _streak_ratio(paper_streak: int, frozen_streak: int) -> float:
    """Paper/frozen losing-streak ratio (inf when frozen never lost)."""
    if frozen_streak > 0:
        return paper_streak / frozen_streak
    return math.inf


def _dd_comparison_dict(
    paper_dd_r: float, frozen_dd_r: float, dd_breach_ratio: float
) -> dict[str, Any]:
    """Drawdown comparison payload (computed on EVERY verdict).

    A positive frozen drawdown yields the plain ratio; a zero frozen
    drawdown with a positive paper drawdown is an unbounded ratio
    (``inf``), and a zero paper drawdown cannot breach.
    """
    if frozen_dd_r > 0.0:
        ratio = paper_dd_r / frozen_dd_r
    elif paper_dd_r == 0.0:
        ratio = 0.0
    else:
        ratio = math.inf
    return {
        "paper_dd_r": paper_dd_r,
        "frozen_dd_r": frozen_dd_r,
        "ratio": ratio,
        "breach": bool(ratio > dd_breach_ratio),
    }


def _leg_kill_reason(
    token: VerdictToken,
    parity_report: dict[str, Any],
    operational_report: dict[str, Any],
    statistical_report: dict[str, Any],
    *,
    killed_by_stage: int | None,
    n_matched: int,
    min_n_trades: int,
) -> str:
    """Human-readable kill reason naming the stage and the issues."""
    if token == "PARITY_FAIL":
        names = [str(entry["name"]) for entry in parity_report["issues"]]
        return f"parity failed at stage {killed_by_stage}: {', '.join(names)}"
    if token == "OPERATIONAL_FAIL":
        names = [str(entry["name"]) for entry in operational_report["issues"]]
        return f"operational failed at stage 4: {', '.join(names)}"
    if token == "STATISTICAL_FAIL":
        names = [str(name) for name in statistical_report["issues"]]
        if not names:
            names = ["mean_drift_p_le0"]
        return f"statistical failed at stage 6: {', '.join(names)}"
    if token == "INSUFFICIENT_N":
        return (
            f"stage 5: n_matched={n_matched} < min_n_trades={min_n_trades} "
            "after clean parity and operational stages"
        )
    if token == "PARITY_PASS":
        return (
            "not killed: parity proven; both operational data-checks "
            "skipped, so no execution-quality evidence exists"
        )
    return "not killed: all six stages clean with operational evidence"


def _run_leg_stages(  # noqa: PLR0911 (one return per stage gate, house style)
    leg: str,
    paper_leg: pd.DataFrame,
    frozen_leg: pd.DataFrame,
    spec: dict[str, Any],
    *,
    bar_delta_min_paper: int,
    paper_sizing_value: str,
    min_n_trades: int,
    parity_tolerance_r: float,
    fee_tolerance_bps: float,
    latency_max_sec: float,
    dd_breach_ratio: float,
    n_bootstrap: int,
    seed: int,
) -> _LegStageResult:
    """Run the fixed six-stage machine for ONE leg (order pinned).

    Stages 1-3 are parity stages (n-agnostic), stage 4 the operational
    data-checks, stage 5 the n-gate, stage 6 the statistics.  The
    drawdown comparison is computed BEFORE the gates: it is reported
    on every verdict (the pinned test asserts a populated
    ``dd_comparison`` even on ``PARITY_FAIL``) and can only affect the
    verdict inside stage 6.
    """
    # facts computed for EVERY verdict, kills included
    alias_map = _exit_alias_map(spec)
    labels = _trade_labels(paper_leg)
    n_paper_trades = int(paper_leg.shape[0])
    paper_pnls_all = np.asarray(
        _float_values(paper_leg, "pnl_r") or [], dtype=np.float64
    )
    frozen_pnls_all = np.asarray(
        _float_values(frozen_leg, "pnl_r") or [], dtype=np.float64
    )
    paper_streak = _max_losing_streak(paper_pnls_all)
    frozen_streak = _max_losing_streak(frozen_pnls_all)
    dd_comparison = _dd_comparison_dict(
        _max_drawdown_r(paper_pnls_all),
        _max_drawdown_r(frozen_pnls_all),
        dd_breach_ratio,
    )
    spec_bar_delta = int(spec["bar_delta_min"])
    feed_granularity = {
        "paper": bar_delta_min_paper,
        "frozen": spec_bar_delta,
        "match": bool(bar_delta_min_paper == spec_bar_delta),
    }
    sizing_rule = {
        "paper": str(paper_sizing_value),
        "frozen": str(spec["sizing_rule"]),
        "match": bool(str(paper_sizing_value) == str(spec["sizing_rule"])),
    }
    funding_alignment: dict[str, Any] | None = None
    if "funding" in spec:
        funding_alignment = {
            "settlement_interval_hours": int(
                spec["funding"]["settlement_interval_hours"]
            ),
            "rate_tolerance_bps": float(
                spec["funding"]["rate_tolerance_bps"]
            ),
            "status": "carry_only",
        }
    pnl_decomposition: dict[str, Any] | None = None
    raw_decomposition = spec.get("pnl_decomposition")
    if isinstance(raw_decomposition, dict) and bool(
        raw_decomposition["enabled"]
    ):
        pnl_decomposition = {
            "components": [
                str(name) for name in raw_decomposition["components"]
            ],
            "tolerance_per_component_r": float(
                raw_decomposition["tolerance_per_component_r"]
            ),
            "status": "checked",
        }

    def statistical_scaffold(n_matched: int) -> dict[str, Any]:
        """Report shape for runs where stage 6 never executed."""
        return {
            "mean_drift_p_le0": float("nan"),
            "n_matched": n_matched,
            "paper_max_streak": paper_streak,
            "frozen_max_streak": frozen_streak,
            "streak_ratio": _streak_ratio(paper_streak, frozen_streak),
            "issues": [],
            "sizing_note": _SIZING_NOTE,
        }

    operational_not_run = {
        "stage_verdict": "NOT_RUN",
        "issues": [],
        "checks_run": [],
        "checks_skipped": [],
    }

    def finish(
        token: VerdictToken,
        parity_report: dict[str, Any],
        operational_report: dict[str, Any],
        statistical_report: dict[str, Any],
        *,
        stage: int | None,
        n_matched: int,
    ) -> _LegStageResult:
        """Assemble the per-leg payload (the ensemble ``_finish`` idiom)."""
        return {
            "verdict_token": token,
            "killed_by_stage": stage,
            "kill_reason": _leg_kill_reason(
                token,
                parity_report,
                operational_report,
                statistical_report,
                killed_by_stage=stage,
                n_matched=n_matched,
                min_n_trades=min_n_trades,
            ),
            # the opt-in sections report the matched-pair count they
            # were checked against (carry-only alignment included)
            "n_paper_trades": n_paper_trades,
            "n_frozen_matched": n_matched,
            "parity_report": parity_report,
            "operational_report": operational_report,
            "statistical_report": statistical_report,
            "dd_comparison": dd_comparison,
            "feed_granularity": feed_granularity,
            "sizing_rule": sizing_rule,
            "funding_alignment": (
                dict(funding_alignment, n_matched=n_matched)
                if funding_alignment is not None
                else None
            ),
            "pnl_decomposition": (
                dict(pnl_decomposition, n_matched=n_matched)
                if pnl_decomposition is not None
                else None
            ),
        }

    # --- stage 1: feed granularity (n-agnostic) -------------------------
    if not feed_granularity["match"]:
        granularity_issues: dict[str, dict[str, Any]] = {}
        _record_issue_bulk(
            granularity_issues,
            "feed_granularity_mismatch",
            n_paper_trades,
            labels,
        )
        parity_report = {
            "stage_verdict": "PARITY_FAIL",
            "n_matched": 0,
            "issues": _issue_list(granularity_issues, _PARITY_ISSUE_ORDER),
            "checks_skipped": [],
        }
        return finish(
            "PARITY_FAIL",
            parity_report,
            operational_not_run,
            statistical_scaffold(0),
            stage=1,
            n_matched=0,
        )
    # --- stage 2: frozen window coverage (n-agnostic) -------------------
    window_issues = _window_coverage_issue(
        paper_leg, frozen_leg, spec, labels
    )
    if window_issues is not None:
        parity_report = {
            "stage_verdict": "PARITY_FAIL",
            "n_matched": 0,
            "issues": _issue_list(window_issues, _PARITY_ISSUE_ORDER),
            "checks_skipped": [],
        }
        return finish(
            "PARITY_FAIL",
            parity_report,
            operational_not_run,
            statistical_scaffold(0),
            stage=2,
            n_matched=0,
        )
    # --- stage 3: trade-level parity (n-agnostic) -----------------------
    parity_report, matched_pairs = _run_parity_stage(
        leg,
        paper_leg,
        frozen_leg,
        spec,
        alias_map=alias_map,
        bar_delta_min=spec_bar_delta,
        paper_sizing_value=paper_sizing_value,
        parity_tolerance_r=parity_tolerance_r,
        fee_tolerance_bps=fee_tolerance_bps,
        labels=labels,
    )
    n_matched = len(matched_pairs)
    if parity_report["stage_verdict"] == "PARITY_FAIL":
        return finish(
            "PARITY_FAIL",
            parity_report,
            operational_not_run,
            statistical_scaffold(n_matched),
            stage=3,
            n_matched=n_matched,
        )
    # --- stage 4: operational data-checks (latency + exit direction) ----
    operational_report = _run_operational_stage(
        leg,
        paper_leg,
        alias_map,
        latency_max_sec=latency_max_sec,
        labels=labels,
    )
    if operational_report["stage_verdict"] == "OPERATIONAL_FAIL":
        return finish(
            "OPERATIONAL_FAIL",
            parity_report,
            operational_report,
            statistical_scaffold(n_matched),
            stage=4,
            n_matched=n_matched,
        )
    # --- stage 5: the n-gate (fires only after stages 1-4 pass) ---------
    if n_matched < min_n_trades:
        return finish(
            "INSUFFICIENT_N",
            parity_report,
            operational_report,
            statistical_scaffold(n_matched),
            stage=5,
            n_matched=n_matched,
        )
    # --- stage 6: statistics --------------------------------------------
    statistical_report = _run_statistical_stage(
        matched_pairs,
        paper_pnls_all.tolist(),
        frozen_pnls_all.tolist(),
        paper_streak=paper_streak,
        frozen_streak=frozen_streak,
        n_matched=n_matched,
        n_bootstrap=n_bootstrap,
        seed=seed,
        dd_comparison=dd_comparison,
    )
    drift_failed = float(statistical_report["mean_drift_p_le0"]) < 0.05
    if drift_failed or statistical_report["issues"]:
        return finish(
            "STATISTICAL_FAIL",
            parity_report,
            operational_report,
            statistical_report,
            stage=6,
            n_matched=n_matched,
        )
    # --- final contract: evidence completeness picks the PASS token -----
    if operational_report["checks_run"]:
        return finish(
            "SURVIVES_TO_LIVE",
            parity_report,
            operational_report,
            statistical_report,
            stage=None,
            n_matched=n_matched,
        )
    return finish(
        "PARITY_PASS",
        parity_report,
        operational_report,
        statistical_report,
        stage=None,
        n_matched=n_matched,
    )


# ---------------------------------------------------------------------------
# portfolio aggregation + metadata layer
# ---------------------------------------------------------------------------
def _merge_issue_lists(
    issue_lists: list[list[dict[str, Any]]], order: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Merge per-leg issue lists (sum counts, pool up to 3 examples)."""
    merged: dict[str, dict[str, Any]] = {}
    for issue_list in issue_lists:
        for entry in issue_list:
            issue_name = str(entry["name"])
            slot = merged.setdefault(
                issue_name, {"n_affected": 0, "examples": []}
            )
            slot["n_affected"] += int(entry["n_affected"])
            for example in entry["examples"]:
                if len(slot["examples"]) < 3:
                    slot["examples"].append(str(example))
    return [
        {
            "name": issue_name,
            "n_affected": merged[issue_name]["n_affected"],
            "examples": merged[issue_name]["examples"],
        }
        for issue_name in order
        if issue_name in merged
    ]


def _portfolio_cascade(
    results: dict[str, _LegStageResult],
) -> tuple[str, str]:
    """Severity cascade over legs: PARITY > OPERATIONAL > STAT > N."""
    groups: dict[str, list[str]] = {token: [] for token in _SEVERITY_CASCADE}
    for leg_name, result in results.items():
        token = result["verdict_token"]
        if token in groups:
            groups[token].append(leg_name)
        # a leg's PARITY_PASS counts as the SURVIVES tier
    parts = [
        f"legs with {cascade_token}: [{', '.join(groups[cascade_token])}]"
        for cascade_token in _SEVERITY_CASCADE
        if groups[cascade_token]
    ]
    for cascade_token in _SEVERITY_CASCADE:
        if groups[cascade_token]:
            return cascade_token, "; ".join(parts)
    detail = "; ".join(
        f"{leg_name}={result['verdict_token']}"
        for leg_name, result in results.items()
    )
    return "SURVIVES_TO_LIVE", f"no leg killed: {detail}"


def _aggregate_stage_results(
    results: dict[str, _LegStageResult],
    token: VerdictToken,
) -> _LegStageResult:
    """Merge per-leg stage results into the portfolio-level payload.

    The drawdown comparison keeps the WORST leg's ratio; the drift
    p-value keeps the most significant (smallest finite) one.
    """
    result_values = list(results.values())
    parity_reports = [result["parity_report"] for result in result_values]
    operational_reports = [
        result["operational_report"] for result in result_values
    ]
    statistical_reports = [
        result["statistical_report"] for result in result_values
    ]
    parity_tokens = [str(entry["stage_verdict"]) for entry in parity_reports]
    merged_parity = {
        "stage_verdict": (
            "PARITY_FAIL" if "PARITY_FAIL" in parity_tokens else "PARITY_PASS"
        ),
        "n_matched": sum(int(entry["n_matched"]) for entry in parity_reports),
        "issues": _merge_issue_lists(
            [entry["issues"] for entry in parity_reports],
            _PARITY_ISSUE_ORDER,
        ),
        "checks_skipped": sorted(
            {
                str(note)
                for entry in parity_reports
                for note in entry["checks_skipped"]
            }
        ),
    }
    operational_tokens = [
        str(entry["stage_verdict"]) for entry in operational_reports
    ]
    if "OPERATIONAL_FAIL" in operational_tokens:
        operational_verdict = "OPERATIONAL_FAIL"
    elif "OPERATIONAL_PASS" in operational_tokens:
        operational_verdict = "OPERATIONAL_PASS"
    else:
        operational_verdict = "NOT_RUN"
    merged_operational = {
        "stage_verdict": operational_verdict,
        "issues": _merge_issue_lists(
            [entry["issues"] for entry in operational_reports],
            _OPERATIONAL_ISSUE_ORDER,
        ),
        "checks_run": sorted(
            {
                str(check_name)
                for entry in operational_reports
                for check_name in entry["checks_run"]
            }
        ),
        "checks_skipped": sorted(
            {
                str(note)
                for entry in operational_reports
                for note in entry["checks_skipped"]
            }
        ),
    }
    finite_ps = [
        float(entry["mean_drift_p_le0"])
        for entry in statistical_reports
        if math.isfinite(float(entry["mean_drift_p_le0"]))
    ]
    merged_drift_p = min(finite_ps) if finite_ps else float("nan")
    paper_streak = max(
        int(entry["paper_max_streak"]) for entry in statistical_reports
    )
    frozen_streak = max(
        int(entry["frozen_max_streak"]) for entry in statistical_reports
    )
    merged_statistical = {
        "mean_drift_p_le0": merged_drift_p,
        "n_matched": sum(
            int(entry["n_matched"]) for entry in statistical_reports
        ),
        "paper_max_streak": paper_streak,
        "frozen_max_streak": frozen_streak,
        "streak_ratio": _streak_ratio(paper_streak, frozen_streak),
        "issues": sorted(
            {
                str(issue_name)
                for entry in statistical_reports
                for issue_name in entry["issues"]
            }
        ),
        "sizing_note": _SIZING_NOTE,
    }
    worst_dd = max(
        result_values,
        key=lambda result: float(result["dd_comparison"]["ratio"]),
    )
    frozen_bar_values = [
        int(result["feed_granularity"]["frozen"]) for result in result_values
    ]
    merged_feed = {
        "paper": int(result_values[0]["feed_granularity"]["paper"]),
        "frozen": min(frozen_bar_values),
        "match": all(
            bool(result["feed_granularity"]["match"])
            for result in result_values
        ),
    }
    sizing_values = [result["sizing_rule"] for result in result_values]
    merged_sizing = {
        "paper": str(sizing_values[0]["paper"]),
        "frozen": str(sizing_values[0]["frozen"]),
        "match": all(bool(entry["match"]) for entry in sizing_values),
    }
    funding_by_leg = {
        leg_name: result["funding_alignment"]
        for leg_name, result in results.items()
        if result["funding_alignment"] is not None
    }
    decomposition_by_leg = {
        leg_name: result["pnl_decomposition"]
        for leg_name, result in results.items()
        if result["pnl_decomposition"] is not None
    }
    if len(result_values) == 1:
        # single leg: the top-level section dict IS the leg's dict
        merged_funding = result_values[0]["funding_alignment"]
        merged_decomposition = result_values[0]["pnl_decomposition"]
    else:
        merged_funding = funding_by_leg or None
        merged_decomposition = decomposition_by_leg or None
    kill_stages = [
        int(result["killed_by_stage"])
        for result in result_values
        if result["killed_by_stage"] is not None
    ]
    return {
        "verdict_token": token,
        "killed_by_stage": min(kill_stages) if kill_stages else None,
        "kill_reason": "",
        "n_paper_trades": sum(
            int(result["n_paper_trades"]) for result in result_values
        ),
        "n_frozen_matched": sum(
            int(result["n_frozen_matched"]) for result in result_values
        ),
        "parity_report": merged_parity,
        "operational_report": merged_operational,
        "statistical_report": merged_statistical,
        "dd_comparison": dict(worst_dd["dd_comparison"]),
        "feed_granularity": merged_feed,
        "sizing_rule": merged_sizing,
        "funding_alignment": merged_funding,
        "pnl_decomposition": merged_decomposition,
    }


def _issue_summary(result: _LegStageResult, token: VerdictToken) -> str:
    """Compact ``n_<issue>=<count>`` summary for the firing stage."""
    if token == "PARITY_FAIL":
        entries = result["parity_report"]["issues"]
    elif token == "OPERATIONAL_FAIL":
        entries = result["operational_report"]["issues"]
    elif token == "STATISTICAL_FAIL":
        entries = result["statistical_report"]["issues"]
        if not entries:
            drift_p = float(result["statistical_report"]["mean_drift_p_le0"])
            return f"n_mean_drift_p_le0={drift_p:.4f}"
        return ", ".join(
            f"n_{issue_name}={entries.count(issue_name)}"
            for issue_name in sorted(set(entries))
        )
    else:
        return "none"
    return ", ".join(
        f"n_{entry['name']}={int(entry['n_affected'])}" for entry in entries
    )


def _first_issue(result: _LegStageResult, token: VerdictToken) -> str:
    """First (canonically ordered) issue name behind the verdict."""
    if token == "PARITY_FAIL":
        entries = result["parity_report"]["issues"]
    elif token == "OPERATIONAL_FAIL":
        entries = result["operational_report"]["issues"]
    elif token == "STATISTICAL_FAIL":
        entries = result["statistical_report"]["issues"]
        if not entries:
            return "mean_drift_p_le0"
        return str(entries[0])
    else:
        return "none"
    if not entries:
        return "none"
    return str(entries[0]["name"])


def _meta_fills(
    result: _LegStageResult,
    label: str,
    min_n_trades: int,
    dd_breach_ratio: float,
) -> dict[str, Any]:
    """Format fills for the ``_VERDICT_META`` templates (superset).

    Every template can be ``.format``-ed with the returned mapping:
    all taxonomy counts are pre-seeded at zero and the run facts
    (n_paper_trades, n_frozen_matched, dd figures, drift p-value,
    skip/run counters) are always present.
    """
    parity_issues = result["parity_report"]["issues"]
    operational_issues = result["operational_report"]["issues"]
    statistical_issues = result["statistical_report"]["issues"]
    fills: dict[str, Any] = {
        "label": label,
        "min_n_trades": min_n_trades,
        "dd_breach_ratio": dd_breach_ratio,
        "n_paper_trades": result["n_paper_trades"],
        "n_frozen_matched": result["n_frozen_matched"],
        "paper_dd_r": result["dd_comparison"]["paper_dd_r"],
        "frozen_dd_r": result["dd_comparison"]["frozen_dd_r"],
        "dd_ratio": result["dd_comparison"]["ratio"],
        "mean_drift_p_le0": result["statistical_report"]["mean_drift_p_le0"],
        "n_skipped_checks": len(
            result["operational_report"]["checks_skipped"]
        ),
        "n_run_checks": len(result["operational_report"]["checks_run"]),
        "n_parity_issues": len(parity_issues),
        "n_operational_issues": len(operational_issues),
    }
    counts: dict[str, int] = {}
    for entry in list(parity_issues) + list(operational_issues):
        counts[str(entry["name"])] = int(entry["n_affected"])
    # the statistical report carries plain issue-name strings
    for issue_name in statistical_issues:
        counts[str(issue_name)] = counts.get(str(issue_name), 0) + 1
    for issue_name in (
        _PARITY_ISSUE_ORDER + _OPERATIONAL_ISSUE_ORDER + ("dd_divergence",)
    ):
        fills[f"n_{issue_name}"] = counts.get(issue_name, 0)
    token = result["verdict_token"]
    fills["issue_summary"] = _issue_summary(result, token)
    fills["first_issue"] = _first_issue(result, token)
    return fills


def _build_leg_verdict(
    result: _LegStageResult,
    portfolio_verdict: str,
    kill_reason: str,
    *,
    label: str,
    min_n_trades: int,
    dd_breach_ratio: float,
) -> LegVerdict:
    """Fill the ``_VERDICT_META`` templates with THIS run's facts."""
    token = result["verdict_token"]
    (
        verdict_class,
        informative,
        interpretation_tpl,
        next_tpl,
        wwc_tpl,
    ) = _VERDICT_META[token]
    fills = _meta_fills(result, label, min_n_trades, dd_breach_ratio)
    return {
        "overall_verdict": token,
        "verdict_class": cast(VerdictClass, verdict_class),
        "informative": bool(informative),
        "killed_by_stage": result["killed_by_stage"],
        "kill_reason": kill_reason,
        "n_paper_trades": result["n_paper_trades"],
        "n_frozen_matched": result["n_frozen_matched"],
        "parity_report": result["parity_report"],
        "operational_report": result["operational_report"],
        "statistical_report": result["statistical_report"],
        "interpretation": interpretation_tpl.format(**fills),
        "next_step": next_tpl.format(**fills),
        "what_would_change": wwc_tpl.format(**fills),
        "portfolio_verdict": portfolio_verdict,
        "label": label,
        "feed_granularity": result["feed_granularity"],
        "sizing_rule": result["sizing_rule"],
        "funding_alignment": result["funding_alignment"],
        "pnl_decomposition": result["pnl_decomposition"],
        "dd_comparison": result["dd_comparison"],
        "facts": {"notes": _FACTS_NOTES},
    }


def _build_paper_verdict(
    aggregate: _LegStageResult,
    portfolio_verdict: str,
    kill_reason: str,
    *,
    per_leg: dict[str, LegVerdict] | None,
    label: str,
    min_n_trades: int,
    dd_breach_ratio: float,
) -> PaperVerdict:
    """Assemble the top-level ``PaperVerdict`` (``per_leg`` attached)."""
    common = _build_leg_verdict(
        aggregate,
        portfolio_verdict,
        kill_reason,
        label=label,
        min_n_trades=min_n_trades,
        dd_breach_ratio=dd_breach_ratio,
    )
    return {
        **common,
        "per_leg": per_leg,
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
class _RunParams(TypedDict):
    """Validated scalar run parameters."""

    min_n_trades: int
    parity_tolerance_r: float
    fee_tolerance_bps: float
    latency_max_sec: float
    dd_breach_ratio: float
    n_bootstrap: int
    seed: int


def _validated_run_params(
    min_n_trades: int,
    *,
    parity_tolerance_r: float,
    fee_tolerance_bps: float,
    latency_max_sec: float,
    dd_breach_ratio: float,
    n_bootstrap: int,
    seed: int,
) -> _RunParams:
    """Validate the scalar protocol parameters loudly (shared contract).

    Raises:
        ValueError: on non-integer / below-minimum integer parameters
            and non-positive tolerances or ``dd_breach_ratio``.
    """
    min_n_trades = validate_int("min_n_trades", min_n_trades, minimum=1)
    parity_tolerance_r = validate_number(
        "parity_tolerance_r", parity_tolerance_r
    )
    if parity_tolerance_r <= 0.0:
        raise ValueError(
            f"parity_tolerance_r must be > 0, got {parity_tolerance_r}"
        )
    fee_tolerance_bps = validate_number(
        "fee_tolerance_bps", fee_tolerance_bps
    )
    if fee_tolerance_bps <= 0.0:
        raise ValueError(
            f"fee_tolerance_bps must be > 0, got {fee_tolerance_bps}"
        )
    latency_max_sec = validate_number("latency_max_sec", latency_max_sec)
    if latency_max_sec <= 0.0:
        raise ValueError(
            f"latency_max_sec must be > 0, got {latency_max_sec}"
        )
    dd_breach_ratio = validate_number("dd_breach_ratio", dd_breach_ratio)
    if dd_breach_ratio <= 0.0:
        raise ValueError(
            f"dd_breach_ratio must be > 0, got {dd_breach_ratio}"
        )
    n_bootstrap = validate_int("n_bootstrap", n_bootstrap, minimum=1)
    seed = validate_int("seed", seed, minimum=0)
    return {
        "min_n_trades": min_n_trades,
        "parity_tolerance_r": parity_tolerance_r,
        "fee_tolerance_bps": fee_tolerance_bps,
        "latency_max_sec": latency_max_sec,
        "dd_breach_ratio": dd_breach_ratio,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def _resolve_requested_legs(
    paper_log: pd.DataFrame,
    frozen_outcomes: dict[str, pd.DataFrame],
    frozen_spec: dict[str, dict[str, Any]],
    legs: list[str] | None,
) -> list[str]:
    """Resolve + validate the leg list (names, coverage, spec keys).

    Default (``legs=None``) is every ``paper_log`` leg in sorted order.
    Every paper leg needs a frozen counterpart; every requested leg
    needs both a frozen frame and a frozen spec entry; no leg name may
    contain "portfolio".

    Raises:
        ValueError: on unknown leg names, missing ``frozen_spec`` keys,
            reserved leg names and malformed ``legs`` entries.
    """
    if legs is not None:
        for leg_name in legs:
            if not isinstance(leg_name, str) or not leg_name.strip():
                raise ValueError(
                    f"legs entries must be non-empty strs, got {leg_name!r}"
                )
    paper_leg_names = sorted(
        {str(name) for name in paper_log["leg"].tolist()}
    )
    requested_legs = (
        [str(name) for name in legs] if legs is not None else paper_leg_names
    )
    if not requested_legs:
        raise ValueError(
            "no legs to process: pass legs or a paper_log with legs"
        )
    for leg_name in sorted(set(requested_legs) | set(paper_leg_names)):
        if "portfolio" in leg_name:
            raise ValueError(
                f"leg name {leg_name!r} contains 'portfolio': the name "
                "is reserved for portfolio-level aggregation"
            )
    for leg_name in paper_leg_names:
        if leg_name not in frozen_outcomes:
            raise ValueError(
                f"paper_log leg {leg_name!r} has no frozen_outcomes "
                "entry: every paper leg needs its frozen counterpart"
            )
    for leg_name in requested_legs:
        if leg_name not in frozen_outcomes:
            raise ValueError(
                f"requested leg {leg_name!r} is not in frozen_outcomes"
            )
        if leg_name not in frozen_spec:
            raise ValueError(
                f"frozen_spec is missing the required leg {leg_name!r}"
            )
        _validate_frozen_spec(leg_name, frozen_spec[leg_name])
    return requested_legs


def run_paper_protocol(
    paper_log: pd.DataFrame,
    frozen_outcomes: dict[str, pd.DataFrame],
    frozen_spec: dict[str, dict[str, Any]],
    *,
    legs: list[str] | None = None,
    min_n_trades: int = 20,
    parity_tolerance_r: float = 0.01,
    fee_tolerance_bps: float = 1.0,
    latency_max_sec: float = 60.0,
    dd_breach_ratio: float = 1.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    label: str = "unnamed",
) -> PaperVerdict:
    """Parity-check the paper log against the frozen outcomes.

    Paper is a forward test of the IMPLEMENTATION, not the edge.  This
    protocol answers: "does the paper engine execute the same strategy
    as the frozen spec?" — a parity checker, not a strategy
    kill-protocol.

    Spec correctness is out of scope. This protocol validates that
    paper matches frozen, not that frozen is correct. Frozen
    correctness → ``run_full_pipeline`` on the backtest feed. A
    ``PARITY_PASS`` means "implementation executes the spec", not
    "the spec is valid".

    The per-leg stage machine runs in the fixed order documented in
    the module docstring: parity stages 1-3 are n-agnostic, the
    operational data-checks (latency, exit direction) run only when
    their columns exist, and the n-gate fires only after they pass.
    ``SURVIVES_TO_LIVE`` demands execution evidence (fill direction
    and latency actually verified); ``PARITY_PASS`` attests the
    trade-level match when both data-checks were skipped — a
    parity-only attestation, never proof the edge is real.
    ``dd_comparison`` is populated on EVERY verdict, including
    ``PARITY_FAIL`` killed at any stage.

    Class-specific sections (``funding``, ``pnl_decomposition``) are
    opt-in via ``frozen_spec``: absence = skipped, not inferred.

    Args:
        paper_log: journal of the paper engine.  Required columns
            ``leg, coin, entry_ts, exit_ts, exit_type, pnl_r`` plus
            ``sizing_rule`` (value read via ``.iloc[0]``), and the
            required attribute ``paper_log.attrs["bar_delta_min"]``
            (int minutes).  Optional columns (absence sends the
            related check to ``checks_skipped``): ``signal_ts,
            entry_price, exit_price, side`` ("long"/"short"),
            ``fee_r, size, funding_rate, pnl_delta, pnl_gamma,
            pnl_theta, pnl_vega, pnl_fee``.
        frozen_outcomes: per-leg frozen (spec-executed) trade frames.
            Required columns ``coin, entry_ts, exit_ts, exit_type,
            pnl_r``; optional ``fee_r, size, funding_rate, pnl_*``.
        frozen_spec: per-leg spec dict.  Required keys:
            ``exit_rules`` ({mtm_bars, tp_r, stop_buffer_atr}),
            ``exit_type_normalization`` (canonical -> alias list),
            ``fee`` ({taker_bps_per_leg, round_trip: "single"|
            "double"}), ``universe`` (list[str]; empty list = universe
            check skipped), ``sizing_rule``, ``bar_delta_min`` and
            ``paper_window`` ({start, end}, pd.Timestamp-parseable;
            an empty dict derives the window from the frozen frame's
            own span).  Optional: ``funding`` ({settlement_interval_hours,
            rate_tolerance_bps}; carry only) and ``pnl_decomposition``
            ({enabled, components, tolerance_per_component_r};
            options only).
        legs: legs to process (each must exist in ``frozen_outcomes``
            and ``frozen_spec``); None processes every ``paper_log``
            leg in sorted order.
        min_n_trades: stage-5 gate on the matched-pair count.
        parity_tolerance_r: max per-pair ``|pnl_r|`` difference.
        fee_tolerance_bps: max per-pair fee difference (basis points).
        latency_max_sec: max ``(entry_ts - signal_ts)`` in seconds.
        dd_breach_ratio: stage-6 cap on paper/frozen max-drawdown
            ratio (drawdowns computed in each frame's row order).
        n_bootstrap: bootstrap draws for the mean-drift p-value.
        seed: RNG seed for the paired bootstrap.
        label: run label echoed in the interpretation strings.

    Returns:
        :class:`PaperVerdict` — every key documented on the TypedDict
        is always present (``per_leg`` is None for single-leg runs);
        ``interpretation`` / ``next_step`` / ``what_would_change`` are
        filled strings quoting this run's facts (n_paper_trades,
        n_frozen_matched and n_<issue> per firing issue), never the
        bare templates.

    Raises:
        ValueError: on an empty ``paper_log`` or per-leg frames,
            missing required columns or the ``bar_delta_min`` attr,
            missing ``frozen_spec`` keys, unknown leg names (requested
            legs absent from ``frozen_outcomes``/``frozen_spec``,
            paper legs without a frozen entry, or any name containing
            "portfolio"), unknown ``round_trip`` literals, exit_type
            tokens absent from ``exit_type_normalization``, missing
            ``funding_rate`` / ``pnl_*`` columns when their spec
            sections demand them, non-positive tolerances,
            ``dd_breach_ratio <= 0``, ``n_bootstrap < 1`` and
            non-integer / below-minimum integer parameters.

    Example::

        Deterministic three-trade parity run; one matched pair books a
        fee 5 bps apart (tolerance 1 bp) -> stage-3 kill:

        >>> import pandas as pd
        >>> from atropos.paper_protocol import run_paper_protocol
        >>> hour = pd.Timedelta(hours=1)
        >>> entries = pd.date_range("2024-03-04", periods=3, freq="h")
        >>> paper = pd.DataFrame(
        ...     {
        ...         "leg": ["btc"] * 3,
        ...         "coin": ["BTC"] * 3,
        ...         "entry_ts": entries,
        ...         "exit_ts": entries + hour,
        ...         "exit_type": ["mtm", "tp", "mtm"],
        ...         "pnl_r": [0.5, 3.0, -1.0],
        ...         "fee_r": [4e-4, 4e-4, 4e-4],
        ...         "sizing_rule": ["fixed_risk"] * 3,
        ...     }
        ... )
        >>> paper.attrs["bar_delta_min"] = 60
        >>> frozen = paper.copy()
        >>> frozen.loc[0, "fee_r"] = 9e-4  # 5 bps apart -> fee_mismatch
        >>> spec = {
        ...     "exit_rules": {"mtm_bars": 8, "tp_r": 3.0,
        ...                    "stop_buffer_atr": 0.5},
        ...     "exit_type_normalization": {
        ...         "time_exit": ["mtm"],
        ...         "tp_exit": ["tp"],
        ...         "stop_exit": ["stop"],
        ...     },
        ...     "fee": {"taker_bps_per_leg": 4.0, "round_trip": "double"},
        ...     "universe": ["BTC"],
        ...     "sizing_rule": "fixed_risk",
        ...     "bar_delta_min": 60,
        ...     "paper_window": {"start": "2024-03-04",
        ...                      "end": "2024-03-04 02:00:00"},
        ... }
        >>> verdict = run_paper_protocol(
        ...     paper, {"btc": frozen}, {"btc": spec}, label="doctest"
        ... )
        >>> verdict["overall_verdict"], verdict["killed_by_stage"]
        ('PARITY_FAIL', 3)
        >>> [issue["name"] for issue in verdict["parity_report"]["issues"]]
        ['fee_mismatch']
    """
    params = _validated_run_params(
        min_n_trades,
        parity_tolerance_r=parity_tolerance_r,
        fee_tolerance_bps=fee_tolerance_bps,
        latency_max_sec=latency_max_sec,
        dd_breach_ratio=dd_breach_ratio,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"label must be a non-empty str, got {label!r}")
    bar_delta_min_paper, paper_sizing_value = _validate_paper_log(paper_log)
    requested_legs = _resolve_requested_legs(
        paper_log, frozen_outcomes, frozen_spec, legs
    )
    for leg_name in requested_legs:
        paper_leg = paper_log[paper_log["leg"] == leg_name]
        if paper_leg.empty:
            raise ValueError(
                f"paper_log has no rows for requested leg {leg_name!r}"
            )
        _validate_leg_frames(
            leg_name,
            paper_leg,
            frozen_outcomes[leg_name],
            frozen_spec[leg_name],
        )
    stage_results: dict[str, _LegStageResult] = {}
    for leg_name in requested_legs:
        stage_results[leg_name] = _run_leg_stages(
            leg_name,
            paper_log[paper_log["leg"] == leg_name],
            frozen_outcomes[leg_name],
            frozen_spec[leg_name],
            bar_delta_min_paper=bar_delta_min_paper,
            paper_sizing_value=paper_sizing_value,
            min_n_trades=params["min_n_trades"],
            parity_tolerance_r=params["parity_tolerance_r"],
            fee_tolerance_bps=params["fee_tolerance_bps"],
            latency_max_sec=params["latency_max_sec"],
            dd_breach_ratio=params["dd_breach_ratio"],
            n_bootstrap=params["n_bootstrap"],
            seed=params["seed"],
        )
    portfolio_verdict, cascade_reason = _portfolio_cascade(stage_results)
    if len(stage_results) == 1:
        only_result = next(iter(stage_results.values()))
        aggregate_token = only_result["verdict_token"]
        top_kill_reason = only_result["kill_reason"]
    else:
        aggregate_token = cast(VerdictToken, portfolio_verdict)
        top_kill_reason = cascade_reason
    aggregate = _aggregate_stage_results(stage_results, aggregate_token)
    per_leg = {
        leg_name: _build_leg_verdict(
            result,
            portfolio_verdict,
            result["kill_reason"],
            label=label,
            min_n_trades=params["min_n_trades"],
            dd_breach_ratio=params["dd_breach_ratio"],
        )
        for leg_name, result in stage_results.items()
    }
    return _build_paper_verdict(
        aggregate,
        portfolio_verdict,
        top_kill_reason,
        per_leg=per_leg if len(per_leg) > 1 else None,
        label=label,
        min_n_trades=params["min_n_trades"],
        dd_breach_ratio=params["dd_breach_ratio"],
    )


# ---------------------------------------------------------------------------
# synthetic demo (python -m atropos.paper_protocol)
# ---------------------------------------------------------------------------
def _demo_spec(universe: list[str], window_end: str) -> dict[str, Any]:
    """Frozen spec shared by both demo scenarios (hourly bars)."""
    return {
        "exit_rules": {"mtm_bars": 8, "tp_r": 3.0, "stop_buffer_atr": 0.5},
        "exit_type_normalization": {
            "time_exit": ["mtm"],
            "tp_exit": ["tp"],
            "stop_exit": ["stop"],
        },
        "fee": {"taker_bps_per_leg": 4.0, "round_trip": "double"},
        "universe": universe,
        "sizing_rule": "fixed_risk",
        "bar_delta_min": 60,
        "paper_window": {"start": "2024-06-03 00:00:00", "end": window_end},
    }


def _demo_mismatch_run() -> PaperVerdict:
    """Demo block: exit mismatches + an alien coin must PARITY_FAIL."""
    hour = pd.Timedelta(hours=1)
    entries = pd.date_range("2024-06-03", periods=5, freq="h")
    paper = pd.DataFrame(
        {
            "leg": ["majors"] * 5,
            "coin": ["BTC", "BTC", "ETH", "BTC", "DOGE"],
            "entry_ts": entries,
            "exit_ts": entries + hour,
            "exit_type": ["mtm", "stop", "stop", "stop", "mtm"],
            "pnl_r": [0.5, -1.05, -1.05, -1.05, 0.4],
            "sizing_rule": ["fixed_risk"] * 5,
        }
    )
    paper.attrs["bar_delta_min"] = 60
    frozen = paper.iloc[:4].copy()
    frozen["exit_type"] = ["mtm", "mtm", "mtm", "mtm"]
    frozen["pnl_r"] = [0.5, 0.5, 0.6, 0.5]
    spec = _demo_spec(["BTC", "ETH"], "2024-06-03 04:00:00")
    verdict = run_paper_protocol(
        paper, {"majors": frozen}, {"majors": spec}, label="mismatch_demo"
    )
    assert verdict["overall_verdict"] == "PARITY_FAIL", verdict
    assert verdict["killed_by_stage"] == 3, verdict
    issue_names = {
        str(entry["name"]) for entry in verdict["parity_report"]["issues"]
    }
    assert {"exit_mismatch", "universe_violation"} <= issue_names, issue_names
    assert set(verdict["dd_comparison"]) == {
        "paper_dd_r",
        "frozen_dd_r",
        "ratio",
        "breach",
    }, verdict["dd_comparison"]
    return verdict


def _demo_survival_run() -> PaperVerdict:
    """Demo block: 40 clean matched trades with execution columns.

    The paper frame carries ``signal_ts`` / ``side`` / prices, so both
    operational data-checks RUN and pass -> ``SURVIVES_TO_LIVE`` (not
    the parity-only ``PARITY_PASS`` token).
    """
    entries = pd.date_range("2024-06-03", periods=40, freq="h")
    exit_pattern = (
        (3.0, "tp", 103.5),
        (-1.05, "stop", 97.0),
        (0.5, "mtm", 100.4),
    )
    rows = []
    for position, entry_stamp in enumerate(entries):
        pnl_r, exit_type, exit_price = exit_pattern[position % 3]
        rows.append(
            {
                "leg": "majors",
                "coin": "BTC",
                "entry_ts": entry_stamp,
                "exit_ts": entry_stamp + pd.Timedelta(hours=1),
                "exit_type": exit_type,
                "pnl_r": pnl_r,
                "signal_ts": entry_stamp - pd.Timedelta(seconds=10),
                "entry_price": 100.0,
                "exit_price": exit_price,
                "side": "long",
                "sizing_rule": "fixed_risk",
            }
        )
    paper = pd.DataFrame(rows)
    paper.attrs["bar_delta_min"] = 60
    frozen = paper[
        ["coin", "entry_ts", "exit_ts", "exit_type", "pnl_r"]
    ].copy()
    spec = _demo_spec(["BTC"], "2024-06-04 15:00:00")
    verdict = run_paper_protocol(
        paper,
        {"majors": frozen},
        {"majors": spec},
        label="clean_forward_test",
    )
    assert verdict["overall_verdict"] == "SURVIVES_TO_LIVE", verdict
    assert verdict["operational_report"]["checks_run"], verdict
    assert verdict["operational_report"]["issues"] == [], verdict
    assert verdict["dd_comparison"]["ratio"] == 1.0, verdict["dd_comparison"]
    return verdict


def run_all_examples() -> None:
    """Demo the parity checker on two synthetic scenarios (asserted).

    Every printed verdict is asserted by the scenario builders, so a
    regression crashes the demo instead of printing a wrong verdict
    quietly.  Scenario 1 plants parity defects (three exit-record
    divergences plus a coin outside the frozen universe); scenario 2
    replays a clean 40-trade forward test with full execution columns.
    """
    print("=" * 72)
    print("paper_protocol demo: paper-vs-frozen parity (two scenarios)")
    print("=" * 72)
    mismatch_verdict = _demo_mismatch_run()
    mismatch_issues = [
        str(entry["name"])
        for entry in mismatch_verdict["parity_report"]["issues"]
    ]
    print(
        "scenario 1 (5 trades: 3 exit mismatches + alien coin)"
        f" -> {mismatch_verdict['overall_verdict']}"
        f" | issues: {mismatch_issues}"
    )
    survival_verdict = _demo_survival_run()
    survival_issues = [
        str(entry["name"])
        for entry in survival_verdict["parity_report"]["issues"]
    ]
    print(
        "scenario 2 (40 clean matched trades, execution columns present)"
        f" -> {survival_verdict['overall_verdict']}"
        f" | issues: {survival_issues}"
    )


if __name__ == "__main__":
    run_all_examples()
