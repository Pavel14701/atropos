"""Tests for atropos.portfolio_protocol: prereg + calibration progression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from atropos.portfolio_protocol import (
    anchor_prereg,
    check_calibration_progression,
    gates_kwargs_from_prereg,
    hash_kill_prereg,
    save_prereg_snapshot,
    validate_kill_prereg,
)


def _valid_prereg() -> dict[str, Any]:
    return {
        "hypothesis": "Sector momentum portfolio (top-3 / bottom-3).",
        "gates": {
            "dsr_min": {
                "enabled": True,
                "threshold": 0.95,
                "comparison": ">",
            },
            "dd_floor": {
                "enabled": True,
                "threshold": -0.20,
                "comparison": ">",
            },
            "side_concentration": {
                "enabled": True,
                "threshold": 0.80,
                "comparison": "<",
            },
        },
        "slots": 3,
        "policy": "replace_low",
        "corr_cap": 0.70,
        "alpha": 0.05,
        "data_window": {"start": "2005-01-01", "end": "2024-12-31"},
        "snapshot_sha256": "a" * 64,
        "instruments": [
            "XLF",
            "XLE",
            "XLK",
            "XLI",
            "XLV",
            "XLY",
            "XLP",
            "XRE",
            "XLU",
            "XLB",
        ],
        "frozen_date": "2026-11-06",
    }


class TestValidateKillPrereg:
    def test_valid_prereg_passes(self) -> None:
        assert validate_kill_prereg(_valid_prereg()) == []

    def test_missing_required_field_reported(self) -> None:
        prereg = _valid_prereg()
        del prereg["hypothesis"]
        problems = validate_kill_prereg(prereg)
        assert any("hypothesis" in problem_text for problem_text in problems)

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            validate_kill_prereg(cast(Any, "not a dict"))

    def test_bad_gate_comparison(self) -> None:
        prereg = _valid_prereg()
        prereg["gates"]["dsr_min"]["comparison"] = "!="
        problems = validate_kill_prereg(prereg)
        assert any("comparison" in problem_text for problem_text in problems)

    def test_bad_alpha(self) -> None:
        prereg = _valid_prereg()
        prereg["alpha"] = 0.5
        problems = validate_kill_prereg(prereg)
        assert any("alpha" in problem_text for problem_text in problems)

    def test_bad_sha(self) -> None:
        prereg = _valid_prereg()
        prereg["snapshot_sha256"] = "not-hex"
        problems = validate_kill_prereg(prereg)
        assert any(
            "snapshot_sha256" in problem_text for problem_text in problems
        )

    def test_slots_zero(self) -> None:
        prereg = _valid_prereg()
        prereg["slots"] = 0
        problems = validate_kill_prereg(prereg)
        assert any("slots" in problem_text for problem_text in problems)

    def test_missing_field_does_not_hide_other_problems(self) -> None:
        """Missing keys AND field problems are reported together.

        Regression: the validator early-returned on the first missing
        key, hiding every other problem (here a broken alpha) from the
        one-pass fix the docstring promises.
        """
        prereg = _valid_prereg()
        del prereg["slots"]
        del prereg["frozen_date"]
        prereg["alpha"] = 0.9
        problems = validate_kill_prereg(prereg)
        assert any(
            "missing required field: slots" in problem_text
            for problem_text in problems
        )
        assert any(
            "missing required field: frozen_date" in problem_text
            for problem_text in problems
        )
        assert any("alpha" in problem_text for problem_text in problems)

    def test_gate_threshold_domains(self) -> None:
        """Each gate's threshold is domain-checked per gate name.

        Regression: any numeric threshold was accepted regardless of
        the gate -- a dsr_min of 1.5 (unsatisfiable), a dd_floor of
        0.5 (not a drawdown floor) and a share cap of 0.0 (triggers on
        every held bar) all hashed silently.
        """
        prereg = _valid_prereg()
        prereg["gates"]["dsr_min"]["threshold"] = 1.5
        prereg["gates"]["dd_floor"]["threshold"] = 0.5
        prereg["gates"]["side_concentration"]["threshold"] = 0.0
        problems = validate_kill_prereg(prereg)
        assert any(
            "dsr_min.threshold" in problem_text for problem_text in problems
        )
        assert any(
            "dd_floor.threshold" in problem_text for problem_text in problems
        )
        assert any(
            "side_concentration.threshold" in problem_text
            for problem_text in problems
        )

    def test_frozen_date_must_be_a_real_date(self) -> None:
        """A strict YYYY-MM-DD check rejects impossible dates.

        Regression: only the length and a dash position were checked,
        so ``"2026-13-99"`` passed as a freeze stamp.
        """
        prereg = _valid_prereg()
        prereg["frozen_date"] = "2026-13-99"
        problems = validate_kill_prereg(prereg)
        assert any("frozen_date" in problem_text for problem_text in problems)

    def test_frozen_date_must_be_dashed_iso_shape(self) -> None:
        """Compact ``YYYYMMDD`` and short-month shapes are rejected."""
        for bad_date in ("20261106", "2026-1-6"):
            prereg = _valid_prereg()
            prereg["frozen_date"] = bad_date
            problems = validate_kill_prereg(prereg)
            assert any(
                "ISO date" in problem_text for problem_text in problems
            )


class TestHashAndSave:
    def test_hash_deterministic(self) -> None:
        prereg = _valid_prereg()
        assert hash_kill_prereg(prereg) == hash_kill_prereg(dict(prereg))

    def test_hash_changes_with_content(self) -> None:
        prereg_a = _valid_prereg()
        prereg_b = _valid_prereg()
        prereg_b["slots"] = 5
        assert hash_kill_prereg(prereg_a) != hash_kill_prereg(prereg_b)

    def test_hash_ignores_self_hash_field(self) -> None:
        prereg = _valid_prereg()
        with_hash = dict(prereg, hash="deadbeef")
        assert hash_kill_prereg(prereg) == hash_kill_prereg(with_hash)

    def test_save_then_refuse_overwrite(self, tmp_path: Path) -> None:
        prereg = _valid_prereg()
        saved = save_prereg_snapshot(prereg, tmp_path)
        assert saved.name == "prereg.json"
        with pytest.raises(FileExistsError):
            save_prereg_snapshot(prereg, tmp_path)

    def test_save_creates_parents(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        save_prereg_snapshot(_valid_prereg(), nested)
        assert (nested / "prereg.json").is_file()

    def test_hash_rejects_nan_threshold(self) -> None:
        """A NaN prereg value must not silently produce a hash.

        Regression: the canonical JSON allowed ``NaN``, so a prereg
        with a NaN threshold hashed happily -- an anchor over a value
        that can never be part of a valid prereg.
        """
        prereg = _valid_prereg()
        prereg["gates"]["dsr_min"]["threshold"] = float("nan")
        with pytest.raises(ValueError, match="JSON-serializable"):
            hash_kill_prereg(prereg)

    def test_hash_rejects_non_serializable_value(self) -> None:
        """A non-JSON object raises ValueError, not raw TypeError.

        Regression: a non-serializable value (a set) leaked json.dumps'
        TypeError straight to the caller instead of the documented
        ValueError contract.
        """
        prereg = _valid_prereg()
        prereg["hypothesis"] = {"not", "serializable"}  # a set
        with pytest.raises(ValueError, match="JSON-serializable"):
            hash_kill_prereg(prereg)


class TestGatesKwargsFromPrereg:
    def test_extracts_all_three_thresholds(self) -> None:
        kwargs = gates_kwargs_from_prereg(_valid_prereg())
        assert kwargs == {
            "dsr_min": 0.95,
            "dd_floor": -0.20,
            "share_cap": 0.80,
        }

    def test_disabled_gate_raises(self) -> None:
        """A disabled gate is fail-closed, never a skipped check."""
        prereg = _valid_prereg()
        prereg["gates"]["side_concentration"]["enabled"] = False
        with pytest.raises(ValueError, match="disabled"):
            gates_kwargs_from_prereg(prereg)

    def test_missing_gate_raises(self) -> None:
        prereg = _valid_prereg()
        del prereg["gates"]["dd_floor"]
        with pytest.raises(ValueError, match="missing"):
            gates_kwargs_from_prereg(prereg)

    def test_missing_gates_dict_raises(self) -> None:
        prereg = {"hypothesis": "no gates here"}
        with pytest.raises(ValueError, match="gates"):
            gates_kwargs_from_prereg(prereg)

    def test_non_numeric_threshold_raises(self) -> None:
        prereg = _valid_prereg()
        prereg["gates"]["dsr_min"]["threshold"] = "0.95"
        with pytest.raises(ValueError, match="real number"):
            gates_kwargs_from_prereg(prereg)


class TestCalibrationProgression:
    def test_all_pass(self) -> None:
        result = check_calibration_progression(
            0.03, "SURVIVES", "DEAD(importance_stability)"
        )
        assert result["all_pass"] is True
        assert result["next_step"] == "proceed to private blind run"

    def test_null_fpr_too_high(self) -> None:
        result = check_calibration_progression(
            0.20, "SURVIVES", "DEAD(cost_line)"
        )
        assert result["all_pass"] is False
        assert "null FPR" in result["next_step"]

    def test_positive_control_killed(self) -> None:
        result = check_calibration_progression(
            0.03, "DEAD(incremental)", "DEAD(cost_line)"
        )
        assert result["all_pass"] is False
        assert "positive control" in result["next_step"]

    def test_negative_control_survives(self) -> None:
        result = check_calibration_progression(0.03, "SURVIVES", "SURVIVES")
        assert result["all_pass"] is False
        assert "negative control" in result["next_step"]

    def test_null_fpr_domain_checked(self) -> None:
        """A measured FPR cannot leave [0, 1].

        Regression: null_fpr was never domain-checked, so a slip like
        1.5 or -0.1 flowed straight into the ``<= 2 * alpha`` verdict.
        """
        with pytest.raises(ValueError, match="null_fpr"):
            check_calibration_progression(1.5, "SURVIVES", "DEAD(cost_line)")
        with pytest.raises(ValueError, match="null_fpr"):
            check_calibration_progression(-0.1, "SURVIVES", "DEAD(cost_line)")

    def test_alpha_domain_checked(self) -> None:
        """Prereg alpha domain (0, 0.2] is enforced.

        Regression: any positive alpha was accepted, silently widening
        the null-FPR tolerance past the preregistered ceiling.
        """
        with pytest.raises(ValueError, match="alpha"):
            check_calibration_progression(
                0.03, "SURVIVES", "DEAD(x)", alpha=0.5
            )
        with pytest.raises(ValueError, match="alpha"):
            check_calibration_progression(
                0.03, "SURVIVES", "DEAD(x)", alpha=0.0
            )


class TestAnchorPrereg:
    def test_valid_hash_anchored(self) -> None:
        report: dict[str, str] = {"verdict": "SURVIVES"}
        anchored = anchor_prereg(report, "a" * 64)
        assert anchored["prereg_hash"] == "a" * 64

    def test_short_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            anchor_prereg({}, "xyz")

    def test_non_sha256_width_hex_raises(self) -> None:
        """Exactly 64 chars are required, not merely >= 8.

        Regression: any hex string of at least 8 chars was accepted, so
        a truncated or otherwise malformed digest could be anchored as
        if it were a full sha256 commitment.
        """
        with pytest.raises(ValueError, match="64-char sha256"):
            anchor_prereg({}, "a" * 40)
