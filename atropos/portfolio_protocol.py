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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "anchor_prereg",
    "check_calibration_progression",
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


def _check_gate(gate_cfg: Any, gate_name: str, problems: list[str]) -> None:
    """Validate one gate config dict against the gate contract."""
    if not isinstance(gate_cfg, dict):
        problems.append(f"gates.{gate_name} must be a dict")
        return
    if not isinstance(gate_cfg.get("enabled"), bool):
        problems.append(f"gates.{gate_name}.enabled must be a bool")
    if not _is_real_number(gate_cfg.get("threshold")):
        problems.append(f"gates.{gate_name}.threshold must be numeric")
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


def _check_frozen_date(frozen: Any, problems: list[str]) -> None:
    """The freeze stamp must look like an ISO ``YYYY-MM-DD`` date."""
    if not isinstance(frozen, str) or len(frozen) != 10 or frozen[4] != "-":
        problems.append("frozen_date must be an ISO date string (YYYY-MM-DD)")


def validate_kill_prereg(prereg: dict[str, Any]) -> list[str]:
    """Validate a portfolio kill preregistration dict.

    Returns an empty list when the prereg is valid; a non-empty list
    of human-readable problems otherwise.  Each problem names the
    offending field so a broken prereg can be fixed in one pass.

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
    if missing:
        return missing
    problems: list[str] = []
    _check_hypothesis(prereg["hypothesis"], problems)
    _check_gates(prereg["gates"], problems)
    _check_slots(prereg["slots"], problems)
    _check_policy(prereg["policy"], problems)
    _check_corr_cap(prereg["corr_cap"], problems)
    _check_alpha(prereg["alpha"], problems)
    _check_data_window(prereg["data_window"], problems)
    _check_snapshot_sha(prereg["snapshot_sha256"], problems)
    _check_instruments(prereg["instruments"], problems)
    _check_frozen_date(prereg["frozen_date"], problems)
    return problems


def hash_kill_prereg(prereg: dict[str, Any]) -> str:
    """SHA256 hex digest of the canonical JSON (sans hash field).

    Deterministic: keys sorted, separators compact, UTF-8 encoded.
    The prereg ``hash`` field (if present) is excluded to avoid a
    circular dependency.

    Example::

        >>> h1 = hash_kill_prereg({"hypothesis": "test", "slots": 2})
        >>> h2 = hash_kill_prereg({"hypothesis": "test", "slots": 2})
        >>> h1 == h2
        True
    """
    frozen = {fn: fv for fn, fv in prereg.items() if fn != "hash"}
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
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
            (``PASS`` requires ``null_fpr <= 2 * alpha``).

    Returns:
        ``{"null_fpr_pass": bool, "positive_pass": bool,
        "negative_pass": bool, "all_pass": bool,
        "next_step": str}``

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


def anchor_prereg(report: dict[str, Any], prereg_hash: str) -> dict[str, Any]:
    """Verify the prereg hash and embed it in the report.

    Raises:
        ValueError: if ``prereg_hash`` is empty or not a hex string.

    Example::

        >>> report = anchor_prereg({"verdict": "SURVIVES"}, "abcdef12")
        >>> report["prereg_hash"]
        'abcdef12'
    """
    if (
        not isinstance(prereg_hash, str)
        or len(prereg_hash) < 8
        or not all(char in "0123456789abcdef" for char in prereg_hash.lower())
    ):
        raise ValueError(
            f"prereg_hash must be a hex string of >= 8 chars, got "
            f"{prereg_hash!r}"
        )
    report["prereg_hash"] = prereg_hash
    return report
