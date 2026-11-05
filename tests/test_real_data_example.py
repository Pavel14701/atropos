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
# the deterministic --quick verdicts on the committed snapshot
QUICK_RUN_VERDICT = "run verdict : DEAD(importance_stability)"
QUICK_GBM_VERDICT = "verdict=DEAD(importance_stability)"


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
