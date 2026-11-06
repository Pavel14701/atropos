"""Tests for atropos.portfolio_protocol: prereg + calibration progression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from atropos.portfolio_protocol import (
    anchor_prereg,
    check_calibration_progression,
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


class TestAnchorPrereg:
    def test_valid_hash_anchored(self) -> None:
        report: dict[str, str] = {"verdict": "SURVIVES"}
        anchored = anchor_prereg(report, "a" * 8)
        assert anchored["prereg_hash"] == "a" * 8

    def test_short_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="hex string"):
            anchor_prereg({}, "xyz")
