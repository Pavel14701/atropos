"""Portfolio kill-protocol pre-registration and calibration progression.

Pre-registration by construction: the portfolio kill gates are frozen
(hash-anchored, committed) BEFORE the private system is run blind.
Calibration on public data (null FPR, positive/negative controls)
proceeds while the gates are already locked; the private blind run is
verifiable because the prereg hash was committed first.

Workflow:

1. Build a ``PortfolioKillPrereg`` dict (hypothesis, gates, thresholds,
   data window, instruments, snapshot checksums).
2. ``validate_kill_prereg`` — every field checked; empty list = valid.
3. ``hash_kill_prereg`` — sha256 of the canonical JSON (sans hash).
4. ``save_prereg_snapshot`` — write ``prereg.json`` to disk
   (idempotent: refuses overwrite).
5. Commit the hash.  Run calibration phases on public data.
6. ``check_calibration_progression`` — all phases PASS? Proceed to
   the private blind run.  Any FAIL?  Stop and recalibrate.

Boundary semantics (implemented by the engine in
:mod:`atropos.portfolio`; a gate's ``comparison`` string is
DESCRIPTIVE for human readers and is never interpreted):

* ``dsr_min`` — the DSR gate triggers at or BELOW the threshold (the
  prereg's ``>`` comparison: equality fails);
* ``dd_floor`` — the drawdown gate triggers when the drawdown
  REACHES the floor;
* ``side_concentration`` — the concentration gate triggers at or
  ABOVE the cap (the prereg's ``<`` comparison: equality fails).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from atropos._utils import validate_number

__all__ = [
    "anchor_prereg",
    "check_calibration_progression",
    "gates_kwargs_from_prereg",
    "hash_kill_prereg",
    "save_prereg_snapshot",
    "validate_kill_prereg",
]

_REQUIRED_PREREG_KEYS = (
    "hypothesis",
    "gates",
    "slots",
    "policy",
    "corr_cap",
    "alpha",
    "data_window",
    "snapshot_sha256",
    "instruments",
    "frozen_date",
)

_GATE_NAMES = ("dsr_min", "dd_floor", "side_concentration")


def _is_real_number(value: Any) -> bool:
    """True for a plain ``int``/``float``; bools are labels, not numbers."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_plain_int(value: Any) -> bool:
    """True for an ``int`` that is not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_hypothesis(value: Any, problems: list[str]) -> None:
    """The hypothesis must say what is being killed and why."""
    if not isinstance(value, str) or not value.strip():
        problems.append("hypothesis must be a non-empty string")


def _threshold_problem(gate_name: str, threshold: float) -> str | None:
    """Domain check for ONE gate's threshold (the engine's ranges).

    ``dsr_min`` lives in ``[0, 1]``, ``dd_floor`` in ``[-1, 0]`` (a
    drawdown floor is a negative equity fraction) and the
    ``side_concentration`` share cap in ``(0, 1]`` (0 would trigger on
    every held bar).  Returns ``None`` when the threshold is in domain.
    """
    if gate_name == "dsr_min":
        in_domain = 0.0 <= threshold <= 1.0
        domain = "[0, 1]"
    elif gate_name == "dd_floor":
        in_domain = -1.0 <= threshold <= 0.0
        domain = "[-1, 0]"
    else:
        in_domain = 0.0 < threshold <= 1.0
        domain = "(0, 1]"
    if in_domain:
        return None
    return (
        f"gates.{gate_name}.threshold must lie in {domain}, got {threshold}"
    )


def _check_gate(gate_cfg: Any, gate_name: str, problems: list[str]) -> None:
    """Validate one gate config dict against the gate contract.

    The threshold is domain-checked per gate name (see
    :func:`_threshold_problem`).  ``comparison`` is only checked for
    shape: it is DESCRIPTIVE for human readers and never interpreted --
    the engine implements the frozen boundary semantics documented in
    this module's docstring.
    """
    if not isinstance(gate_cfg, dict):
        problems.append(f"gates.{gate_name} must be a dict")
        return
    if not isinstance(gate_cfg.get("enabled"), bool):
        problems.append(f"gates.{gate_name}.enabled must be a bool")
    threshold: Any = gate_cfg.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        problems.append(f"gates.{gate_name}.threshold must be numeric")
    else:
        domain_problem = _threshold_problem(gate_name, float(threshold))
        if domain_problem is not None:
            problems.append(domain_problem)
    if gate_cfg.get("comparison") not in ("<", ">"):
        problems.append(f"gates.{gate_name}.comparison must be '<' or '>'")


def _check_gates(gates: Any, problems: list[str]) -> None:
    """All three preregistered gates must be well-formed."""
    if not isinstance(gates, dict):
        problems.append("gates must be a dict")
        return
    for gate_name in _GATE_NAMES:
        _check_gate(gates.get(gate_name), gate_name, problems)


def _check_slots(slots: Any, problems: list[str]) -> None:
    """Slot count must be a plain integer >= 1."""
    if not _is_plain_int(slots) or slots < 1:
        problems.append("slots must be an integer >= 1")


def _check_policy(policy: Any, problems: list[str]) -> None:
    """Slot-replacement policy must be one of the frozen choices."""
    if policy not in ("fcfs", "replace_low", "replace_high"):
        problems.append(
            "policy must be 'fcfs', 'replace_low', or 'replace_high', "
            f"got {policy!r}"
        )


def _check_corr_cap(corr_cap: Any, problems: list[str]) -> None:
    """Correlation cap must be a real number in ``(0, 1]``."""
    if not _is_real_number(corr_cap) or not 0.0 < corr_cap <= 1.0:
        problems.append("corr_cap must be in (0, 1]")


def _check_alpha(alpha: Any, problems: list[str]) -> None:
    """Family-wise alpha must be a real number in ``(0, 0.2]``."""
    if not _is_real_number(alpha) or not 0.0 < alpha <= 0.2:
        problems.append("alpha must be in (0, 0.2]")


def _check_data_window(data_window: Any, problems: list[str]) -> None:
    """The data window must carry string ``start``/``end`` bounds."""
    if not isinstance(data_window, dict):
        problems.append("data_window must be a dict with start/end")
    elif not all(
        isinstance(data_window.get(bound), str) for bound in ("start", "end")
    ):
        problems.append("data_window.start and .end must be strings")


def _check_snapshot_sha(sha: Any, problems: list[str]) -> None:
    """The snapshot checksum must be a 64-character hex string."""
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or not all(char in "0123456789abcdef" for char in sha.lower())
    ):
        problems.append("snapshot_sha256 must be a 64-char hex string")


def _check_instruments(instruments: Any, problems: list[str]) -> None:
    """At least two non-empty instrument names are required."""
    if (
        not isinstance(instruments, list)
        or len(instruments) < 2
        or not all(
            isinstance(instrument, str) and instrument
            for instrument in instruments
        )
    ):
        problems.append(
            "instruments must be a list of >= 2 non-empty strings"
        )


_FROZEN_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _check_frozen_date(frozen: Any, problems: list[str]) -> None:
    """The freeze stamp must parse as a strict ISO ``YYYY-MM-DD`` date.

    The shape is checked by regex first (``date.fromisoformat`` alone
    also reads compact ``YYYYMMDD`` forms), then the value itself must
    be a real calendar date -- ``"2026-13-99"`` is malformed, not a
    freeze stamp.
    """
    if not isinstance(frozen, str) or not _FROZEN_DATE_RE.fullmatch(frozen):
        problems.append("frozen_date must be an ISO date string (YYYY-MM-DD)")
        return
    try:
        date.fromisoformat(frozen)
    except ValueError:
        problems.append(
            f"frozen_date must be a real calendar date, got {frozen!r}"
        )


def validate_kill_prereg(prereg: dict[str, Any]) -> list[str]:
    """Validate a portfolio kill preregistration dict.

    Returns an empty list when the prereg is valid; a non-empty list
    of human-readable problems otherwise.  Each problem names the
    offending field so a broken prereg can be fixed in one pass:
    missing keys are reported first and EVERY present field is still
    checked (an early return on the first missing key used to hide the
    remaining problems).

    Raises:
        ValueError: if ``prereg`` is not a dict.

    Example::

        >>> prereg = {
        ...     "hypothesis": "sector momentum top-3 / bottom-3",
        ...     "gates": {
        ...         "dsr_min": {"enabled": True, "threshold": 0.95,
        ...                      "comparison": ">"},
        ...         "dd_floor": {"enabled": True, "threshold": -0.20,
        ...                       "comparison": ">"},
        ...         "side_concentration": {"enabled": True,
        ...                                "threshold": 0.80,
        ...                                "comparison": "<"},
        ...     },
        ...     "slots": 3,
        ...     "policy": "replace_low",
        ...     "corr_cap": 0.70,
        ...     "alpha": 0.05,
        ...     "data_window": {"start": "2005-01-01",
        ...                     "end": "2024-12-31"},
        ...     "snapshot_sha256": "c0ffee" + "0" * 58,
        ...     "instruments": ["XLF", "XLE"],
        ...     "frozen_date": "2026-11-06",
        ... }
        >>> validate_kill_prereg(prereg)
        []
        >>> prereg["alpha"] = 0.9  # outside the allowed (0, 0.2]
        >>> validate_kill_prereg(prereg)
        ['alpha must be in (0, 0.2]']
    """
    if not isinstance(prereg, dict):
        raise ValueError("prereg must be a dict")
    missing = [
        f"missing required field: {key}"
        for key in _REQUIRED_PREREG_KEYS
        if key not in prereg
    ]
    field_checks: tuple[tuple[str, Callable[[Any, list[str]], None]], ...] = (
        ("hypothesis", _check_hypothesis),
        ("gates", _check_gates),
        ("slots", _check_slots),
        ("policy", _check_policy),
        ("corr_cap", _check_corr_cap),
        ("alpha", _check_alpha),
        ("data_window", _check_data_window),
        ("snapshot_sha256", _check_snapshot_sha),
        ("instruments", _check_instruments),
        ("frozen_date", _check_frozen_date),
    )
    problems: list[str] = []
    for key, check in field_checks:
        if key in prereg:
            check(prereg[key], problems)
    return missing + problems


def hash_kill_prereg(prereg: dict[str, Any]) -> str:
    """SHA256 hex digest of the canonical JSON (sans hash field).

    Deterministic: keys sorted, separators compact, UTF-8 encoded,
    ``allow_nan=False``.  The prereg ``hash`` field (if present) is
    excluded to avoid a circular dependency.

    Raises:
        ValueError: if the prereg carries a value that canonical JSON
            cannot express losslessly -- a non-serializable object, or
            a NaN/inf float (``NaN`` is not a number a frozen prereg
            can be re-hashed from, so hashing it would silently anchor
            nothing).

    Example::

        >>> h1 = hash_kill_prereg({"hypothesis": "test", "slots": 2})
        >>> h2 = hash_kill_prereg({"hypothesis": "test", "slots": 2})
        >>> h1 == h2
        True
    """
    frozen = {fn: fv for fn, fv in prereg.items() if fn != "hash"}
    try:
        canonical = json.dumps(
            frozen, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prereg must be JSON-serializable with finite numbers only"
            f" (canonical JSON is the hash preimage): {exc}"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_prereg_snapshot(prereg: dict[str, Any], root: Path) -> Path:
    """Write ``prereg.json`` (canonical JSON) into ``root``.

    Idempotent: refuses to overwrite an existing snapshot (edit
    manually or delete the old file first).

    Returns:
        The path of the written snapshot.

    Example::

        >>> import shutil
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> snapshot = save_prereg_snapshot(
        ...     {"hypothesis": "noise book", "slots": 1}, root)
        >>> snapshot.name
        'prereg.json'
        >>> json.loads(snapshot.read_text(encoding="utf-8")) == {
        ...     "hypothesis": "noise book", "slots": 1}
        True
        >>> try:
        ...     save_prereg_snapshot({"hypothesis": "noise book",
        ...                           "slots": 1}, root)
        ... except FileExistsError:
        ...     print("refused: never overwrite a frozen prereg")
        refused: never overwrite a frozen prereg
        >>> shutil.rmtree(root)
    """
    dest = root / "prereg.json"
    if dest.exists():
        raise FileExistsError(
            f"prereg snapshot already exists at {dest} — edit manually "
            "or delete it first; never overwrite a frozen prereg"
        )
    root.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(prereg, sort_keys=True, indent=2)
    dest.write_text(canonical, encoding="utf-8")
    return dest


def check_calibration_progression(
    null_fpr: float,
    positive_control: str,
    negative_control: str,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Gate the calibration progression: proceed to private blind run?

    Args:
        null_fpr: measured false-positive rate on the null signal world.
        positive_control: verdict of the positive control (e.g. SPY
            TSMOM momentum portfolio).  Must start with ``SURVIVES``.
        negative_control: verdict of the negative control (e.g. RSI
            mean-reversion or random portfolio).  Must start with
            ``DEAD``.
        alpha: nominal false-positive rate for the two-sided tolerance
            (``PASS`` requires ``null_fpr <= 2 * alpha``); must lie in
            the prereg domain ``(0, 0.2]``.

    Returns:
        ``{"null_fpr_pass": bool, "positive_pass": bool,
        "negative_pass": bool, "all_pass": bool,
        "next_step": str}``

    Raises:
        ValueError: if ``null_fpr`` is outside ``[0, 1]`` or ``alpha``
            is outside ``(0, 0.2]`` (a measured FPR cannot leave the
            unit interval and ``alpha`` is bound by the prereg domain).

    Example::

        >>> verdict = check_calibration_progression(
        ...     0.03, "SURVIVES", "DEAD(cost_line)")
        >>> verdict["all_pass"]
        True
        >>> verdict["next_step"]
        'proceed to private blind run'
        >>> verdict = check_calibration_progression(
        ...     0.30, "DEAD(incremental)", "SURVIVES")
        >>> verdict["next_step"]
        'recalibrate: null FPR, positive control, negative control'
    """
    if not 0.0 <= validate_number("null_fpr", null_fpr) <= 1.0:
        raise ValueError(f"null_fpr must be in [0, 1], got {null_fpr}")
    if not 0.0 < validate_number("alpha", alpha) <= 0.2:
        raise ValueError(f"alpha must be in (0, 0.2], got {alpha}")
    null_pass = null_fpr <= 2 * alpha
    pos_pass = positive_control.startswith("SURVIVES")
    neg_pass = negative_control.startswith("DEAD")
    all_pass = null_pass and pos_pass and neg_pass
    next_step = (
        "proceed to private blind run"
        if all_pass
        else "recalibrate: "
        + ", ".join(
            label_text
            for label_text, gate_ok in (
                ("null FPR", null_pass),
                ("positive control", pos_pass),
                ("negative control", neg_pass),
            )
            if not gate_ok
        )
    )
    return {
        "null_fpr_pass": null_pass,
        "positive_pass": pos_pass,
        "negative_pass": neg_pass,
        "all_pass": all_pass,
        "next_step": next_step,
    }


def gates_kwargs_from_prereg(prereg: dict[str, Any]) -> dict[str, float]:
    """Extract the engine gate thresholds from a frozen prereg.

    Returns ``{"dsr_min": ..., "dd_floor": ..., "share_cap": ...}`` --
    the gate thresholds :func:`atropos.portfolio.run_portfolio` and
    :func:`atropos.portfolio.portfolio_gates` execute with -- read
    ONLY from ``prereg["gates"][name]["threshold"]`` so the executed
    gates cannot silently drift from the hashed prereg.

    Boundary semantics the engine implements (the prereg's
    ``comparison`` strings are descriptive): the DSR gate triggers at
    or below ``dsr_min``, the drawdown gate when it reaches
    ``dd_floor``, the concentration gate at or above ``share_cap``.

    Fail-closed: every one of the three gates must be present, enabled
    and carry a numeric threshold -- a disabled or missing gate is an
    error, never a skipped check.

    Raises:
        ValueError: if the prereg carries no ``gates`` dict, or any of
            the three gates is missing, disabled (``enabled: False``)
            or lacks a numeric threshold.

    Example::

        >>> prereg = {
        ...     "gates": {
        ...         "dsr_min": {"enabled": True, "threshold": 0.95,
        ...                      "comparison": ">"},
        ...         "dd_floor": {"enabled": True, "threshold": -0.20,
        ...                       "comparison": ">"},
        ...         "side_concentration": {"enabled": True,
        ...                                "threshold": 0.80,
        ...                                "comparison": "<"},
        ...     }
        ... }
        >>> gates_kwargs_from_prereg(prereg)
        {'dsr_min': 0.95, 'dd_floor': -0.2, 'share_cap': 0.8}
    """
    gates = prereg.get("gates")
    if not isinstance(gates, dict):
        raise ValueError(
            "prereg must carry a 'gates' dict: thresholds can only be "
            "derived from the frozen preregistration"
        )
    extracted: dict[str, float] = {}
    for gate_name, result_key in (
        ("dsr_min", "dsr_min"),
        ("dd_floor", "dd_floor"),
        ("side_concentration", "share_cap"),
    ):
        gate_cfg = gates.get(gate_name)
        if not isinstance(gate_cfg, dict):
            raise ValueError(
                f"gate '{gate_name}' is missing from the prereg: every "
                "preregistered gate must be present (fail-closed)"
            )
        if gate_cfg.get("enabled") is not True:
            raise ValueError(
                f"gate '{gate_name}' is disabled or lacks enabled=True: "
                "a disabled prereg gate must not run (fail-closed)"
            )
        threshold = gate_cfg.get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(
            threshold, bool
        ):
            raise ValueError(
                f"gate '{gate_name}' threshold must be a real number, "
                f"got {threshold!r}"
            )
        extracted[result_key] = float(threshold)
    return extracted


def anchor_prereg(report: dict[str, Any], prereg_hash: str) -> dict[str, Any]:
    """Verify the prereg hash and embed it in the report.

    Raises:
        ValueError: if ``prereg_hash`` is not a 64-character sha256 hex
            string (the exact digest width
            :func:`hash_kill_prereg` emits).

    Example::

        >>> committed = "c0ffee" + "0" * 58
        >>> report = anchor_prereg({"verdict": "SURVIVES"}, committed)
        >>> report["prereg_hash"] == committed
        True
    """
    if (
        not isinstance(prereg_hash, str)
        or len(prereg_hash) != 64
        or not all(char in "0123456789abcdef" for char in prereg_hash.lower())
    ):
        raise ValueError(
            f"prereg_hash must be a 64-char sha256 hex string, got "
            f"{prereg_hash!r}"
        )
    report["prereg_hash"] = prereg_hash
    return report
