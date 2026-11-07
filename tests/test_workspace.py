"""Workspace scaffolding tests: idempotent skeleton generation.

``atropos.workspace.init_workspace`` must create the documented
skeleton exactly once, never overwrite user files, and produce content
that matches the methodology documents (append-only graveyard with the
COD legend, the registry header, the prereg-file convention).  The
CLI-level behaviour (dry-run, default root, output) lives in
tests/test_cli.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from atropos import workspace as atropos_workspace
from atropos.workspace import (
    DEFAULT_WORKSPACE_FILES,
    init_workspace,
    run_all_examples,
)

_REL_PATHS = [
    "GRAVEYARD.md",
    "hypotheses/INDEX.csv",
    "hypotheses/README.md",
]

_COD_TAGS = (
    "REPACKAGING",
    "ARTIFACT",
    "NO_OOS_EDGE",
    "COST_DEATH",
    "RISK",
    "CS_NO_EDGE",
    "DEAD",
)


def test_init_creates_every_default_file(tmp_path: Path) -> None:
    """Fresh root: all skeleton files are created, parents included."""
    root = tmp_path / "fresh"
    result = init_workspace(root)
    assert result["created"] == _REL_PATHS
    assert result["skipped"] == []
    assert result["root"] == str(root)
    for rel_text in _REL_PATHS:
        assert (root / rel_text).is_file()


def test_init_is_idempotent(tmp_path: Path) -> None:
    """A second call creates nothing and skips every existing file."""
    root = tmp_path / "ws"
    init_workspace(root)
    second = init_workspace(root)
    assert second["created"] == []
    assert second["skipped"] == _REL_PATHS


def test_init_never_overwrites_user_edits(tmp_path: Path) -> None:
    """User edits survive a re-init byte-for-byte."""
    root = tmp_path / "ws"
    init_workspace(root)
    graveyard = root / "GRAVEYARD.md"
    index = root / "hypotheses" / "INDEX.csv"
    graveyard.write_text("# my own graveyard\n", encoding="utf-8")
    index.write_text("id,name\n", encoding="utf-8")
    result = init_workspace(root)
    assert result["created"] == []
    assert graveyard.read_text(encoding="utf-8") == "# my own graveyard\n"
    assert index.read_text(encoding="utf-8") == "id,name\n"


def test_init_race_reports_skip_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file created between the existence check and the write wins.

    Regression: init used check-then-write (``target.exists()`` ->
    ``write_text``), so a file materializing in that window was
    silently overwritten with the skeleton content.  The atomic
    create-or-fail ``open(target, "x")`` turns the same race into a
    skip.  The patched opener emulates the racing creator: when init
    asks to create ``GRAVEYARD.md``, a user file is planted first and
    ``FileExistsError`` is raised.  The old check-then-write never
    called this opener, wrote the skeleton anyway and reported the
    file as created.
    """
    real_open = open

    def racing_open(
        path: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if mode == "x" and Path(str(path)).name == "GRAVEYARD.md":
            with real_open(path, "w", encoding="utf-8") as handle:
                handle.write("user edit won the race\n")
            raise FileExistsError
        return real_open(path, mode, *args, **kwargs)

    # raising=False: the module only carries "open" while the atomic
    # create-or-fail implementation is in place; the old check-then-
    # write never bound (nor called) it, which is what fails this test
    # there.
    monkeypatch.setattr(atropos_workspace, "open", racing_open, raising=False)
    result = init_workspace(tmp_path / "ws")
    graveyard = tmp_path / "ws" / "GRAVEYARD.md"
    assert "GRAVEYARD.md" in result["skipped"]
    assert "GRAVEYARD.md" not in result["created"]
    assert graveyard.read_text(encoding="utf-8") == (
        "user edit won the race\n"
    )
    # the files that did not lose a race are still created normally
    assert result["created"] == [
        "hypotheses/INDEX.csv",
        "hypotheses/README.md",
    ]


def test_init_over_an_existing_file_root_reports_os_error(
    tmp_path: Path,
) -> None:
    """``root`` pointing at an existing file raises OSError (propagated)."""
    blocker = tmp_path / "occupied"
    blocker.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(OSError):
        init_workspace(blocker)


def test_graveyard_skeleton_markers(tmp_path: Path) -> None:
    """Graveyard skeleton: append-only banner + full COD legend."""
    root = tmp_path / "ws"
    init_workspace(root)
    graveyard = (root / "GRAVEYARD.md").read_text(encoding="utf-8")
    assert "Graveyard — append only" in graveyard
    assert "H-NNNN" in graveyard
    assert "## Entry format" in graveyard
    assert "## COD legend" in graveyard
    for cod_tag in _COD_TAGS:
        assert cod_tag in graveyard


def test_index_and_readme_skeleton_markers(tmp_path: Path) -> None:
    """INDEX header columns and the prereg README convention."""
    root = tmp_path / "ws"
    init_workspace(root)
    index_text = (root / "hypotheses" / "INDEX.csv").read_text(
        encoding="utf-8"
    )
    assert index_text == "id,name,date,family,cod,verdict,file\n"
    readme_text = (root / "hypotheses" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "PREREG_<name>_<ISO-date>.md" in readme_text
    assert "verdict" in readme_text
    assert "atropos init" in readme_text


def test_default_workspace_files_shape() -> None:
    """The registry maps relative paths to non-empty text templates."""
    assert list(DEFAULT_WORKSPACE_FILES) == [
        Path(rel_text) for rel_text in _REL_PATHS
    ]
    assert all(
        content.strip() for content in DEFAULT_WORKSPACE_FILES.values()
    )


def test_demo_run_all_examples_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module demo runs its asserts on a throwaway temp root."""
    run_all_examples()
    assert "workspace demo" in capsys.readouterr().out
