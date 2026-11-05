"""Committed demo transcripts stay truthful to the live demos.

`docs/examples/` holds captured demo outputs so a reader can see real
framework behavior without running anything.  These checks pin the
transcripts to the demo source: the recorded command must reference a
module that exists, and the verdicts recorded in the transcript must
still be produced by a fresh demo run (verdicts only -- float details
legitimately vary across BLAS builds).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"

# (transcript file, case banner token, recorded verdict token)
TRANSCRIPT_CASES = [
    (
        "ml_training_demo.log",
        "CASE 5: documented-kill reproduction (VRP panic)",
        "DEAD(incremental)",
    ),
]


def test_examples_dir_exists_with_readme() -> None:
    """The examples catalog ships with a regeneration README."""
    assert (EXAMPLES_DIR / "README.md").is_file()


@pytest.fixture(scope="module")
def fresh_ml_demo_output() -> str:
    """Fresh output of the ml_training demo, captured once."""
    module_ref = pytest.importorskip("atropos.ml_training")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module_ref.run_all_examples()
    return buffer.getvalue()


@pytest.mark.parametrize("transcript, banner, verdict", TRANSCRIPT_CASES)
def test_transcript_matches_fresh_demo_output(
    fresh_ml_demo_output: str,
    transcript: str,
    banner: str,
    verdict: str | None,
) -> None:
    """Recorded verdicts are still produced by the live demo.

    Guards against the transcript drifting from reality while
    deliberately NOT comparing floats, which vary across BLAS builds.
    """
    transcript_path = EXAMPLES_DIR / transcript
    assert transcript_path.is_file(), "committed transcript is missing"
    recorded = transcript_path.read_text(encoding="utf-8")
    assert banner in recorded, f"transcript lacks banner {banner!r}"
    if verdict is not None:
        assert verdict in recorded, f"transcript lacks verdict {verdict!r}"

    assert banner in fresh_ml_demo_output, (
        f"fresh demo output no longer contains {banner!r}"
    )
    if verdict is not None:
        assert verdict in fresh_ml_demo_output, (
            f"fresh demo verdict no longer matches recorded {verdict!r}"
        )


def test_vrp_transcript_carries_tombstone_references(
    fresh_ml_demo_output: str,
) -> None:
    """The VRP reproduction transcript cites the tombstone statistics."""
    transcript = (EXAMPLES_DIR / "ml_training_demo.log").read_text(
        encoding="utf-8"
    )
    assert "-7.98" in transcript  # tombstone per-trade EV
    assert "closed donor corpus" in transcript  # honest labeling
    assert "TRIGGERED" in fresh_ml_demo_output  # kill gates fired


def test_examples_source_command_is_real() -> None:
    """The recorded capture command runs the module that exists."""
    transcript = (EXAMPLES_DIR / "ml_training_demo.log").read_text(
        encoding="utf-8"
    )
    match = re.search(r"Executed: (.+)", transcript)
    assert match is not None, "transcript lacks the capture command"
    command = match.group(1).strip()
    assert command.startswith("uv run python -m atropos.")
    module_name = command.removeprefix("uv run python -m ").strip()

    assert importlib.util.find_spec(module_name) is not None, (
        f"capture command references missing module {module_name!r}"
    )
