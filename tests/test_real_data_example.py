"""The real-data example stays honest to its committed snapshot.

`examples/real_data/` demonstrates calling the atropos library on real
market data (SPY 12-1 time-series momentum).  These checks pin the
three things that make the example trustworthy: the committed snapshot
bytes match the sha256 sidecar (the offline reproducibility anchor),
the `--quick` run exits cleanly with the recorded verdicts, and the
preregistration is printed BEFORE any walk-forward result (the
prereg-then-results order is the caller's discipline -- the test keeps
the example honest about it).  Floats are deliberately not compared:
they legitimately vary across BLAS builds.

The RSI(2) tests below pin the SECOND real-data example, the negative
control: a book whose recorded verdict is a kill, on the same
committed snapshot.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "real_data"
RUNNER = EXAMPLE_DIR / "run_example.py"
STRATEGY = EXAMPLE_DIR / "strategies" / "spy_tsmom.py"
SNAPSHOT = EXAMPLE_DIR / "data" / "SPY.csv"
SIDECAR = EXAMPLE_DIR / "data" / "SPY.csv.sha256"

QUICK_TIMEOUT_S = 120
# the deterministic --quick verdicts on the committed snapshot.
# Verdict history: until the incremental gate was recalibrated (the
# full-procedure threshold-sensitive bootstrap replaced the retired
# fixed-threshold paired bootstrap t, which false-passed 43-46 of 100
# pure-noise worlds), this run read DEAD(importance_stability) -- the
# old t spuriously certified "beats the battery incrementally" and
# instability named the kill.  The calibrated statistic measures NO
# incremental edge on this snapshot (bootstrap mean EV diff ~ -0.0020,
# 95% CI straddling zero), so the incremental gate -- gate #1 --
# triggers and names the verdict while importance_stability still
# triggers alongside it.
QUICK_RUN_VERDICT = "run verdict : DEAD(incremental)"
QUICK_GBM_VERDICT = "verdict=DEAD(incremental)"


def test_snapshot_matches_sha256_sidecar() -> None:
    """data/SPY.csv is byte-identical to the committed hash pin."""
    assert SNAPSHOT.is_file(), "committed snapshot is missing"
    assert SIDECAR.is_file(), "checksum sidecar is missing"
    pinned = SIDECAR.read_text(encoding="ascii").split()[0]
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert digest == pinned, (
        "snapshot drifted from data/SPY.csv.sha256; re-pin the vintage "
        "consciously or restore the committed bytes"
    )


def _load_strategy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "spy_tsmom_under_test", STRATEGY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_contract_on_committed_snapshot() -> None:
    """The event dataset is powered and lookahead-free by construction."""
    strategy = _load_strategy_module()
    dataset = strategy.build_dataset(SNAPSHOT)
    events = dataset["events"]
    assert len(events) >= 30, "event book too thin for the frozen floor"
    # every entry happens after its signal month-end (skip-1 lag)
    assert bool((events["entry_date"] > events["signal_month_end"]).all())
    # engine contract: binary target, finite exec, monotone int64 stamps
    target = np.asarray(dataset["target"])
    assert set(np.unique(target)).issubset({0, 1})
    exec_returns = np.asarray(dataset["exec_returns"], dtype=float)
    assert bool(np.isfinite(exec_returns).all())
    times_us = np.asarray(dataset["times_us"])
    assert times_us.dtype == np.int64
    assert bool((np.diff(times_us) > 0).all())


def _run_quick() -> str:
    completed = subprocess.run(
        [sys.executable, "-I", str(RUNNER), "--quick"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=QUICK_TIMEOUT_S,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_quick_run_prints_prereg_before_verdicts() -> None:
    """--quick exits 0 and prints the frozen prereg BEFORE any verdict."""
    stdout = _run_quick()
    assert "# Preregistration: spy_tsmom_12_1_ml" in stdout
    assert "- **Spec hash:**" in stdout
    first_verdict = stdout.find("verdict")
    assert first_verdict != -1, "no verdict printed at all"
    assert stdout.find("# Preregistration") < first_verdict, (
        "prereg must be printed before the first walk-forward verdict"
    )
    assert stdout.find("## Kill criteria") < first_verdict
    assert "KILLED if any of the following holds" in stdout


def test_quick_run_records_expected_verdicts() -> None:
    """The deterministic --quick verdicts match the committed transcript."""
    stdout = _run_quick()
    assert QUICK_RUN_VERDICT in stdout
    assert QUICK_GBM_VERDICT in stdout
    assert "UNDERPOWERED" not in stdout, (
        "the committed snapshot powers the run; an UNDERPOWERED here "
        "means the dataset or floor drifted"
    )
    assert "TRIGGERED" in stdout, "the kill gate must be shown as triggered"


# ---------------------------------------------------------------------------
# RSI(2) mean-reversion negative control (strategies/spy_rsi.py)
# ---------------------------------------------------------------------------

RSI_STRATEGY = EXAMPLE_DIR / "strategies" / "spy_rsi.py"
RSI_RUNNER = EXAMPLE_DIR / "run_rsi_example.py"
RSI_QUICK_TIMEOUT_S = 120
# the deterministic --quick verdict on the committed snapshot (the
# kill comes from importance_stability -- cost_line actually passes;
# expected_rsi_output.md records that anatomy honestly)
RSI_QUICK_RUN_VERDICT = "run verdict : DEAD(importance_stability)"


def _load_rsi_strategy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "spy_rsi_under_test", RSI_STRATEGY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_rsi_quick() -> str:
    completed = subprocess.run(
        [sys.executable, "-I", str(RSI_RUNNER), "--quick"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=RSI_QUICK_TIMEOUT_S,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_rsi_example_quick_runs() -> None:
    """--quick exits 0 and prints the frozen prereg BEFORE any verdict."""
    stdout = _run_rsi_quick()
    assert "# Preregistration: spy_rsi2_meanrev_ml" in stdout
    assert "- **Spec hash:**" in stdout
    first_verdict = stdout.find("verdict")
    assert first_verdict != -1, "no verdict printed at all"
    assert stdout.find("# Preregistration") < first_verdict, (
        "prereg must be printed before the first walk-forward verdict"
    )
    assert stdout.find("## Kill criteria") < first_verdict
    assert "run verdict" in stdout, "no run verdict line printed"


def test_rsi_dataset_contract() -> None:
    """The RSI event dataset is valid, powered and lookahead-free."""
    strategy = _load_rsi_strategy_module()
    dataset = strategy.build_dataset(SNAPSHOT)
    events = dataset["events"]
    assert len(events) >= 30, "event book too thin for the frozen floor"
    # engine contract: binary target, finite exec, monotone int64 stamps
    target = np.asarray(dataset["target"])
    assert set(np.unique(target)).issubset({0, 1})
    exec_returns = np.asarray(dataset["exec_returns"], dtype=float)
    assert bool(np.isfinite(exec_returns).all())
    times_us = np.asarray(dataset["times_us"])
    assert times_us.dtype == np.int64
    assert bool((np.diff(times_us) > 0).all())
    # no lookahead: entry executes strictly after the signal close,
    # and the book never overlaps itself (entry follows the prior exit)
    assert bool((events["entry_date"] > events["signal_date"]).all())
    assert bool((events["exit_date"] >= events["entry_date"]).all())
    entries = events["entry_date"].iloc[1:].to_numpy()
    prior_exits = events["exit_date"].iloc[:-1].to_numpy()
    assert bool((entries > prior_exits).all())
    # frozen construction: hold within the 5-bar cap, both sides present,
    # feature frame is exactly the model columns (no battery control)
    assert bool(events["hold_bars"].between(1, 5).all())
    assert set(events["side"]) == {"long", "short"}
    assert list(dataset["feature_frame"].columns) == list(strategy.FEATURES)


def test_rsi_is_negative_control() -> None:
    """The engine kills the RSI book on --quick: recorded verdict is DEAD.

    This is the honest recorded outcome on the committed snapshot, not
    a wish: the preregistered expectation was a kill and the engine
    delivered one (via importance_stability -- cost_line passes, see
    expected_rsi_output.md).  If this verdict ever changes, the finding
    is recorded here and in the transcript, not quietly re-tuned.
    """
    stdout = _run_rsi_quick()
    run_verdict = next(
        line.split(":", 1)[1].strip()
        for line in stdout.splitlines()
        if line.startswith("run verdict :")
    )
    assert run_verdict.startswith("DEAD"), (
        f"negative control no longer dies: verdict={run_verdict!r}; "
        "record the new outcome honestly in expected_rsi_output.md"
    )
    assert RSI_QUICK_RUN_VERDICT in stdout, (
        "verdict drifted from the committed transcript"
    )
    assert "UNDERPOWERED" not in stdout, (
        "the committed snapshot powers the run; an UNDERPOWERED here "
        "means the dataset or floor drifted"
    )
    assert "cost_line            pass" in stdout, (
        "the recorded anatomy matters: the book dies on instability "
        "while clearing the cost line -- keep that honest"
    )
