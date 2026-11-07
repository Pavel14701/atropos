"""CLI tests for the four ``atropos`` subcommands.

Everything runs through ``main()`` with a pinned user layer and a
temporary working directory (the project layer is ``<cwd>/.atropos/
agent``), captured via ``capsys`` - offline, deterministic, no
subprocesses.  The last test walks the full B.12 verification scenario:
default context -> project replacement -> show-sources -> disable.
"""

from __future__ import annotations

import io
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from atropos import __version__ as atropos_version
from atropos.cli import (
    _doctor_package,
    _load_manifest,
    _package_version,
    main,
)

PACKAGE_TWELVE = [
    "01-preregister",
    "02-run-pipeline",
    "03-read-graveyard",
    "04-run-ml-kill-protocol",
    "11-premise-check",
    "12-null-control",
    "13-signal-test",
    "14-incremental-test",
    "15-robustness-battery",
    "16-walkforward-calendar",
    "17-backtest-pipeline",
    "18-read-portfolio-results",
]


def _write(path: Path, text: str) -> Path:
    """Write a UTF-8 file (with parent dirs) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Isolated cwd plus its pinned (nonexistent) user-layer path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path, tmp_path / "user-layer"


def _run(
    argv: list[str], user_dir: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    """Call main() with a pinned user layer; return (code, out, err)."""
    code = main(argv, user_dir=user_dir)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# agent-context
# ---------------------------------------------------------------------------
def test_main_without_command_exits_with_usage() -> None:
    """No subcommand: argparse fails with its usage exit code."""
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_list_skills_pinned_layer(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--list-skills prints the twelve packaged skills, exit 0."""
    _, user_dir = workdir
    code, out, err = _run(
        ["agent-context", "--list-skills"], user_dir, capsys
    )
    assert code == 0
    assert err == ""
    assert out.splitlines() == PACKAGE_TWELVE


def test_content_is_default_action(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Bare agent-context prints the same text as --content."""
    _, user_dir = workdir
    code_default, out_default, _ = _run(["agent-context"], user_dir, capsys)
    code_flag, out_flag, _ = _run(
        ["agent-context", "--content"], user_dir, capsys
    )
    assert code_default == code_flag == 0
    assert out_default == out_flag
    assert "===== AGENTS.md =====" in out_default
    assert "===== CLAUDE.md =====" in out_default


def test_show_sources_defaults(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--show-sources lists layers, default statuses and 12 skills."""
    _, user_dir = workdir
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert code == 0
    assert "layers:" in out
    assert "package  " in out
    assert "AGENTS.md: default" in out
    assert "CLAUDE.md: default" in out
    assert "skills: 12 active, 0 disabled/empty" in out
    for name in PACKAGE_TWELVE:
        assert name in out


def test_skill_flags(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--skill prints one skill; unknown names exit non-zero."""
    _, user_dir = workdir
    code, out, err = _run(
        ["agent-context", "--skill", "01-preregister"], user_dir, capsys
    )
    assert code == 0
    assert out.startswith("---")
    assert err == ""
    code, out, err = _run(
        ["agent-context", "--skill", "no-such-skill"], user_dir, capsys
    )
    assert code == 1
    assert out == ""
    assert "unknown or empty skill" in err


def test_skills_only_prints_every_active_skill(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--skills-only prints each active skill under a name header."""
    _, user_dir = workdir
    code, out, _ = _run(["agent-context", "--skills-only"], user_dir, capsys)
    assert code == 0
    for name in PACKAGE_TWELVE:
        assert f"===== skill: {name} =====" in out


# ---------------------------------------------------------------------------
# agent-context: project-layer overrides (B.12 scenario)
# ---------------------------------------------------------------------------
def test_project_override_scenario(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Replace -> show-sources -> content reflects the project layer."""
    tmp_path, user_dir = workdir
    project_ag = _write(
        tmp_path / ".atropos" / "agent" / "AGENTS.md", "# PROJ RULES"
    )
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert code == 0
    assert "AGENTS.md: replaced" in out
    assert str(project_ag) in out
    assert "CLAUDE.md: default" in out
    code, out, _ = _run(["agent-context", "--content"], user_dir, capsys)
    assert code == 0
    assert "# PROJ RULES" in out
    assert "default layer" not in out
    assert "===== CLAUDE.md =====" in out


def test_project_append_and_disable(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Append adds to AGENTS.md; disabled drops CLAUDE.md."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    _write(project / "AGENTS.append.md", "PROJ APPEND")
    _write(project / "CLAUDE.disabled", "")
    code, out, _ = _run(["agent-context", "--content"], user_dir, capsys)
    assert code == 0
    assert "PROJ APPEND" in out
    assert "===== CLAUDE.md =====" not in out
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert "AGENTS.md: default + appends" in out
    assert "CLAUDE.md: disabled" in out


def test_all_disabled_content_fails_loudly(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Both files disabled: --content exits 1 with a stderr note."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    _write(project / "AGENTS.disabled", "")
    _write(project / "CLAUDE.disabled", "")
    code, out, err = _run(["agent-context", "--content"], user_dir, capsys)
    assert code == 1
    assert out == ""
    assert "no agent context resolves" in err


def test_skill_override_operations(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Replace one skill, add one, disable one - statuses and lists."""
    tmp_path, user_dir = workdir
    skills_dir = tmp_path / ".atropos" / "agent" / "skills"
    _write(skills_dir / "01-preregister.md", "---\nname: 01-preregister")
    _write(skills_dir / "new-thing.md", "# NEW THING")
    _write(skills_dir / ".disabled", "13-signal-test\n")
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert code == 0
    assert "skills: 12 active, 1 disabled/empty" in out
    assert f"{'01-preregister':<26} replaced" in out
    assert f"{'new-thing':<26} added" in out
    assert f"{'13-signal-test':<26} disabled" in out
    code, out, _ = _run(["agent-context", "--list-skills"], user_dir, capsys)
    listed = out.splitlines()
    assert "new-thing" in listed
    assert "13-signal-test" not in listed
    code, out, _ = _run(
        ["agent-context", "--skill", "01-preregister"], user_dir, capsys
    )
    assert code == 0
    assert out.strip() == "---\nname: 01-preregister"


# ---------------------------------------------------------------------------
# agent-setup
# ---------------------------------------------------------------------------
def test_setup_requires_scope_when_not_interactive(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-tty stdin without --scope: message, exit 1, no writes."""
    tmp_path, user_dir = workdir
    code, _out, err = _run(["agent-setup"], user_dir, capsys)
    assert code == 1
    assert "--scope" in err
    assert not (tmp_path / ".atropos").exists()


def test_setup_dry_run_writes_nothing(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run prints the plan and creates no files."""
    tmp_path, user_dir = workdir
    code, out, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--mode=append",
            "--dry-run",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert "[create]" in out
    assert "dry-run" in out
    assert not (tmp_path / ".atropos").exists()


def test_setup_refuses_to_write_without_yes(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A full plan without --yes exits 1 and writes nothing."""
    tmp_path, user_dir = workdir
    code, _out, err = _run(
        ["agent-setup", "--scope=project", "--mode=append"],
        user_dir,
        capsys,
    )
    assert code == 1
    assert "nothing written" in err
    assert not (tmp_path / ".atropos").exists()


def test_setup_append_mode_creates_stubs(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--mode=append scaffolds stubs + README and appends visibly."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    code, out, _ = _run(
        ["agent-setup", "--scope=project", "--mode=append", "--yes"],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (project / "AGENTS.append.md").is_file()
    assert (project / "CLAUDE.append.md").is_file()
    assert (project / "README.md").is_file()
    code, out, _ = _run(["agent-context", "--content"], user_dir, capsys)
    assert "AGENTS.append.md - project additions" in out
    # rerun never overwrites: everything exists, zero files written
    code, out, _ = _run(
        ["agent-setup", "--scope=project", "--mode=append", "--yes"],
        user_dir,
        capsys,
    )
    assert code == 0
    assert "0 file(s) written" in out
    assert "skip (exists)" in out


def test_setup_disable_and_replace_single_file(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--file restricts the stub; disable and replace modes differ."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--mode=disable",
            "--file=CLAUDE.md",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (project / "CLAUDE.disabled").is_file()
    assert not (project / "AGENTS.disabled").exists()
    code, out, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--mode=replace",
            "--file=AGENTS.md",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (project / "AGENTS.md").is_file()
    code, out, _ = _run(["agent-context", "--content"], user_dir, capsys)
    assert "AGENTS.md - project override (replace)" in out
    assert "===== CLAUDE.md =====" not in out


def test_setup_default_mode_creates_only_readme(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """mode=default scaffolds the layer without touching the files."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    code, _, _ = _run(
        ["agent-setup", "--scope=project", "--yes"], user_dir, capsys
    )
    assert code == 0
    assert (project / "README.md").is_file()
    assert list(project.iterdir()) == [project / "README.md"]


def test_setup_skills_none_disables_all_defaults(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--skills=none writes skills/.disabled-all; list is empty."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--skills=none",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (project / "skills" / ".disabled-all").is_file()
    code, out, _ = _run(["agent-context", "--list-skills"], user_dir, capsys)
    assert code == 0
    assert out == ""


def test_setup_disable_skill_names_accumulate(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--disable adds names to skills/.disabled exactly once."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--disable=13-signal-test",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    disabled_file = project / "skills" / ".disabled"
    assert disabled_file.read_text(encoding="utf-8") == "13-signal-test\n"
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--disable=13-signal-test",
            "--disable=12-null-control",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    content = disabled_file.read_text(encoding="utf-8")
    assert content.count("13-signal-test") == 1
    assert "12-null-control" in content
    code, out, _ = _run(["agent-context", "--list-skills"], user_dir, capsys)
    listed = out.splitlines()
    assert "13-signal-test" not in listed
    assert "12-null-control" not in listed
    assert "14-incremental-test" in listed


def test_setup_user_scope_uses_pinned_user_dir(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--scope=user writes into the injected user directory."""
    tmp_path, user_dir = workdir
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=user",
            "--mode=disable",
            "--file=AGENTS.md",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (user_dir / "AGENTS.disabled").is_file()
    assert not (tmp_path / ".atropos").exists()


# ---------------------------------------------------------------------------
# agent-doctor
# ---------------------------------------------------------------------------
def test_doctor_clean_environment(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """No overrides anywhere: zero errors, layers reported as none."""
    _, user_dir = workdir
    code, out, err = _run(["agent-doctor"], user_dir, capsys)
    assert code == 0
    assert err == ""
    assert "doctor: 0 error(s)" in out
    assert "user layer: none" in out
    assert "project layer: none" in out


def test_doctor_empty_project_directory_is_ok(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """An existing but empty project layer is fine (no orphans)."""
    tmp_path, user_dir = workdir
    (tmp_path / ".atropos" / "agent").mkdir(parents=True)
    code, out, _ = _run(["agent-doctor"], user_dir, capsys)
    assert code == 0
    assert "directory exists, no overrides" in out


def test_doctor_reports_orphans_and_unknown_disabled(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Orphan appends and unknown .disabled names become warnings."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    _write(project / "WEIRD.append.md", "orphan")
    _write(project / "skills" / "ghost.append.md", "orphan skill")
    _write(project / "skills" / ".disabled", "not-a-skill\n")
    code, out, _ = _run(["agent-doctor"], user_dir, capsys)
    assert code == 0
    assert "orphan append WEIRD.append.md" in out
    assert "orphan skill append ghost.append.md" in out
    assert "unknown skill 'not-a-skill'" in out


def test_doctor_reports_override_inventory(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing overrides are inventoried in the ok section."""
    tmp_path, user_dir = workdir
    _write(tmp_path / ".atropos" / "agent" / "AGENTS.append.md", "x")
    code, out, _ = _run(["agent-doctor"], user_dir, capsys)
    assert code == 0
    assert "project layer: 1 file(s): AGENTS.append.md" in out


def test_doctor_missing_package_layer_fails(
    workdir: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gutted package layer is reported as errors, exit 1."""
    tmp_path, user_dir = workdir
    nowhere = tmp_path / "gutted-package"
    monkeypatch.setattr("atropos.cli.AGENT_DIR", nowhere)
    monkeypatch.setattr("atropos.cli.AGENTS_MD", nowhere / "AGENTS.md")
    monkeypatch.setattr("atropos.cli.CLAUDE_MD", nowhere / "CLAUDE.md")
    monkeypatch.setattr("atropos.cli.SKILLS_DIR", nowhere / "skills")
    code, out, _ = _run(["agent-doctor"], user_dir, capsys)
    assert code == 1
    assert "package layer missing" in out
    assert "MANIFEST.toml unreadable" in out
    assert "doctor: 6 error(s)" in out


# ---------------------------------------------------------------------------
# full B.12-style verification walk
# ---------------------------------------------------------------------------
def test_full_verification_scenario(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Default -> replace -> show-sources -> disable, end to end."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"

    # 1. defaults: twelve skills, both files from the package
    code, out, _ = _run(["agent-context", "--list-skills"], user_dir, capsys)
    assert out.splitlines() == PACKAGE_TWELVE

    # 2. project replacement changes the effective AGENTS.md
    _write(project / "AGENTS.md", "# VERIFICATION REPLACEMENT")
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert "AGENTS.md: replaced" in out
    code, out, _ = _run(["agent-context", "--content"], user_dir, capsys)
    assert "# VERIFICATION REPLACEMENT" in out

    # 3. disabling the file leaves append-only (or empty) content
    _write(project / "AGENTS.disabled", "")
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert "AGENTS.md: disabled" in out
    code, out, err = _run(["agent-context", "--content"], user_dir, capsys)
    assert code == 0  # CLAUDE.md still resolves
    assert "===== AGENTS.md =====" not in out
    _write(project / "CLAUDE.disabled", "")
    code, out, err = _run(["agent-context", "--content"], user_dir, capsys)
    assert code == 1
    assert "no agent context resolves" in err


# ---------------------------------------------------------------------------
# remaining branch coverage: --path, interactive setup, doctor internals
# ---------------------------------------------------------------------------
def test_path_flag_lists_contributing_paths(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--path prints each contributing path with kind and layer."""
    tmp_path, user_dir = workdir
    project = tmp_path / ".atropos" / "agent"
    _write(project / "AGENTS.append.md", "extra")
    _write(project / "CLAUDE.disabled", "")
    code, out, _ = _run(["agent-context", "--path"], user_dir, capsys)
    assert code == 0
    assert "AGENTS.md:" in out
    assert "base     package" in out
    assert "append   project" in out
    assert "CLAUDE.md:" in out
    assert "[dropped: disabled]" in out


def test_path_and_sources_with_gutted_package(
    workdir: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package layer without files prints (none) / none statuses."""
    tmp_path, user_dir = workdir
    nowhere = tmp_path / "gutted"
    monkeypatch.setattr("atropos.cli.AGENT_DIR", nowhere)
    code, out, _ = _run(["agent-context", "--path"], user_dir, capsys)
    assert code == 0
    assert "  (none)" in out
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert "AGENTS.md: none" in out
    assert "CLAUDE.md: none" in out


def test_show_sources_empty_and_append_skill_lines(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty replacements and skill appends render their own statuses."""
    tmp_path, user_dir = workdir
    skills_dir = tmp_path / ".atropos" / "agent" / "skills"
    _write(skills_dir / "13-signal-test.md", "  ")
    _write(skills_dir / "01-preregister.append.md", "extra bit")
    code, out, _ = _run(["agent-context", "--show-sources"], user_dir, capsys)
    assert code == 0
    assert f"{'13-signal-test':<26} replaced (empty)" in out
    assert f"{'01-preregister':<26} default + appends" in out
    assert f"{'append':<24} project" in out


def test_setup_interactive_scope_and_abort(
    workdir: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive scope prompt writes to the user layer; empty aborts."""
    tmp_path, user_dir = workdir

    class _FakeTty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _FakeTty())
    answers = iter(["typo", "user"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    code, _out, _ = _run(
        ["agent-setup", "--mode=disable", "--file=AGENTS.md", "--yes"],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (user_dir / "AGENTS.disabled").is_file()
    assert not (tmp_path / ".atropos").exists()

    monkeypatch.setattr("sys.stdin", _FakeTty())
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    code, _out, err = _run(["agent-setup"], user_dir, capsys)
    assert code == 1
    assert "no scope chosen" in err


def test_setup_merges_into_disabled_list_without_newline(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A .disabled file lacking a trailing newline is repaired on merge."""
    tmp_path, user_dir = workdir
    disabled_file = _write(
        tmp_path / ".atropos" / "agent" / "skills" / ".disabled",
        "13-signal-test",
    )
    code, _, _ = _run(
        [
            "agent-setup",
            "--scope=project",
            "--disable=12-null-control",
            "--yes",
        ],
        user_dir,
        capsys,
    )
    assert code == 0
    assert (
        disabled_file.read_text(encoding="utf-8")
        == "13-signal-test\n12-null-control\n"
    )


def test_package_version_falls_back_to_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without installed metadata, the package version falls back."""

    def _missing(dist_name: str) -> str:
        raise PackageNotFoundError(dist_name)

    monkeypatch.setattr("atropos.cli._dist_version", _missing)
    assert _package_version() == atropos_version


def test_load_manifest_reports_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid TOML is returned as (None, reason), not raised."""
    fake_package = tmp_path / "broken-manifest"
    _write(fake_package / "MANIFEST.toml", "not [valid toml ===")
    monkeypatch.setattr("atropos.cli.AGENT_DIR", fake_package)
    manifest, reason = _load_manifest()
    assert manifest is None
    assert reason


def test_doctor_package_integrity_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Version mismatch and phantom manifest skills become errors."""
    fake_package = tmp_path / "fake-package"
    _write(fake_package / "AGENTS.md", "# AGENTS")
    _write(fake_package / "CLAUDE.md", "# CLAUDE")
    _write(fake_package / "VERSION", "0.1.0")
    _write(
        fake_package / "MANIFEST.toml",
        '[agent]\nversion = "9.9.9"\n\n'
        '[[skills]]\nname = "ghost"\n'
        'path = "skills/ghost.md"\n',
    )
    _write(fake_package / "skills" / "undeclared.md", "# X")
    monkeypatch.setattr("atropos.cli.AGENT_DIR", fake_package)
    monkeypatch.setattr("atropos.cli.AGENTS_MD", fake_package / "AGENTS.md")
    monkeypatch.setattr("atropos.cli.CLAUDE_MD", fake_package / "CLAUDE.md")
    monkeypatch.setattr("atropos.cli.SKILLS_DIR", fake_package / "skills")
    errors: list[str] = []
    notes: list[str] = []
    _doctor_package(errors, notes)
    assert any("!= VERSION" in error for error in errors)
    assert any("declares missing skill: ghost" in error for error in errors)
    assert any("undeclared" in note for note in notes)


# ---------------------------------------------------------------------------
# init (workspace skeleton)
# ---------------------------------------------------------------------------
def test_init_creates_workspace_skeleton(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The init subcommand scaffolds the workspace under --root."""
    pytest.importorskip("atropos.workspace")
    tmp_path, user_dir = workdir
    target = tmp_path / "fresh-ws"
    code, out, err = _run(["init", "--root", str(target)], user_dir, capsys)
    assert code == 0
    assert err == ""
    assert (target / "GRAVEYARD.md").is_file()
    assert (target / "hypotheses" / "INDEX.csv").is_file()
    assert (target / "hypotheses" / "README.md").is_file()
    assert "created" in out
    assert "workspace ready" in out


def test_init_default_root_is_cwd(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --root the workspace lands in the current directory."""
    pytest.importorskip("atropos.workspace")
    tmp_path, user_dir = workdir
    code, out, _ = _run(["init"], user_dir, capsys)
    assert code == 0
    assert (tmp_path / "GRAVEYARD.md").is_file()
    assert (tmp_path / "hypotheses" / "INDEX.csv").is_file()
    assert "workspace ready" in out


def test_init_dry_run_writes_nothing(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run prints the plan and creates no files."""
    pytest.importorskip("atropos.workspace")
    tmp_path, user_dir = workdir
    target = tmp_path / "never-created"
    code, out, _ = _run(
        ["init", "--root", str(target), "--dry-run"], user_dir, capsys
    )
    assert code == 0
    assert "[create]" in out
    assert "dry-run, nothing written" in out
    assert not target.exists()
    # an existing workspace shows up as skip-only plan
    target.mkdir()
    (target / "GRAVEYARD.md").write_text("user edit\n", encoding="utf-8")
    code, out, _ = _run(
        ["init", "--root", str(target), "--dry-run"], user_dir, capsys
    )
    assert code == 0
    assert "[skip (exists)]" in out


def test_init_second_run_skips_and_preserves_edits(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-init never overwrites: all files skipped, edits survive."""
    pytest.importorskip("atropos.workspace")
    tmp_path, user_dir = workdir
    target = tmp_path / "ws"
    code, _out, _ = _run(["init", "--root", str(target)], user_dir, capsys)
    assert code == 0
    graveyard = target / "GRAVEYARD.md"
    graveyard.write_text("user edit\n", encoding="utf-8")
    code, out, _ = _run(["init", "--root", str(target)], user_dir, capsys)
    assert code == 0
    assert "skipped (exists)" in out
    assert "created " not in out
    assert graveyard.read_text(encoding="utf-8") == "user edit\n"


# ---------------------------------------------------------------------------
# demo (packaged backtest demo pipeline)
# ---------------------------------------------------------------------------
def test_demo_quick_completes(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--mode quick prints every stage plus the verdict, exit 0."""
    pytest.importorskip("atropos.demo_pipeline")
    _, user_dir = workdir
    code, out, err = _run(["demo", "--mode", "quick"], user_dir, capsys)
    assert code == 0
    assert err == ""
    for stage in ("STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "STAGE 5"):
        assert stage in out
    assert "backtest pipeline: all stage checks passed" in out
    assert "VERDICT" in out
    assert "pipeline completed" in out


def test_demo_same_seed_is_bit_identical(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Two quick runs with one seed produce identical stdout."""
    pytest.importorskip("atropos.demo_pipeline")
    _, user_dir = workdir
    code_first, out_first, _ = _run(
        ["demo", "--mode", "quick", "--seed", "41"], user_dir, capsys
    )
    code_second, out_second, _ = _run(
        ["demo", "--mode", "quick", "--seed", "41"], user_dir, capsys
    )
    assert code_first == code_second == 0
    assert out_first == out_second
    assert out_first != ""


def test_demo_help_documents_mode_and_budget(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--help lists --mode quick/full with the runtime expectation."""
    with pytest.raises(SystemExit) as exit_info:
        main(["demo", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--mode" in out
    assert "quick" in out
    assert "full" in out
    assert "--seed" in out


def test_demo_failure_exits_nonzero(
    workdir: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A not-completed pipeline summary maps to exit code 1.

    Regression: only the success exit path was pinned before, so a
    refactor that made ``atropos demo`` always return 0 (reporting
    success for a failed pipeline) would have shipped unnoticed.
    """
    pytest.importorskip("atropos.demo_pipeline")
    import atropos.demo_pipeline as demo_module  # noqa: PLC0415

    _, user_dir = workdir
    monkeypatch.setattr(
        demo_module,
        "run_demo_pipeline",
        lambda **_kwargs: {"pipeline_completed": False},
    )
    code = main(["demo", "--mode", "quick"], user_dir=user_dir)
    assert code == 1


# ---------------------------------------------------------------------------
# non-ASCII output on legacy consoles (cp1251)
# ---------------------------------------------------------------------------
def test_skill_03_non_ascii_prints_cleanly(
    workdir: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Skill 03 (Cyrillic + arrows) prints via main() without crashing."""
    _, user_dir = workdir
    code, out, err = _run(
        ["agent-context", "--skill", "03-read-graveyard"], user_dir, capsys
    )
    assert code == 0
    assert "READ GRAVEYARD" in out
    assert err == ""


def test_cp1251_console_reconfigured_not_fatal(
    workdir: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strict cp1251 stdout is reconfigured; non-cp1251 text prints."""
    tmp_path, user_dir = workdir
    _write(
        tmp_path / ".atropos" / "agent" / "skills" / "arrow-skill.md",
        "trend arrow U+2192: \u2192",
    )
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1251", errors="strict")
    monkeypatch.setattr("sys.stdout", console)
    code = main(
        ["agent-context", "--skill", "arrow-skill"], user_dir=user_dir
    )
    assert code == 0
    console.flush()
    # without the UTF-8 reconfigure this encode would have raised
    # UnicodeEncodeError ('\u2192' has no cp1251 mapping)
    assert "arrow U+2192: \u2192" in raw.getvalue().decode("utf-8")
