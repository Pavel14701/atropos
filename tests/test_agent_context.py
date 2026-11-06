"""Unit tests for the three-layer agent-context resolution engine.

Every override operation (replace / append / disable / disabled-all /
add / append-only-survivor) is exercised on ``tmp_path`` layers, plus
the platform default-dir logic and the public ``atropos.agent`` API
against the real packaged layer.  All tests are offline and
deterministic: layers are explicit directories, never the real user
or project directories.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

import pytest
from atropos.agent import (
    AGENT_DIR,
    AGENTS_MD,
    CLAUDE_MD,
    SKILLS_DIR,
    list_skills,
    read_agent_context,
    read_skill,
)
from atropos.agent._resolution import (
    DISABLE_ALL_NAME,
    Layer,
    ResolvedSkill,
    _is_windows,
    _parse_disabled_list,
    _read_text,
    build_layers,
    default_project_dir,
    default_user_dir,
    resolve_context_file,
    resolve_skills,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

PACKAGE_ELEVEN = [
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
]


def _write(path: Path, text: str) -> Path:
    """Write a UTF-8 file (with parent dirs) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def mini_package(tmp_path: Path) -> Layer:
    """A minimal package layer: two context files, two skills."""
    package_dir = tmp_path / "package"
    _write(package_dir / "AGENTS.md", "# PACKAGE AGENTS\n\nbase rules.")
    _write(package_dir / "CLAUDE.md", "# PACKAGE CLAUDE\n\nclaude notes.")
    _write(package_dir / "skills" / "alpha-skill.md", "# ALPHA\npackaged")
    _write(package_dir / "skills" / "beta-skill.md", "# BETA\npackaged")
    return Layer("package", package_dir)


@pytest.fixture()
def user_layer(tmp_path: Path) -> Layer:
    """An empty user layer directory (tests add files per case)."""
    return Layer("user", tmp_path / "user")


@pytest.fixture()
def project_layer(tmp_path: Path) -> Layer:
    """An empty project layer directory (tests add files per case)."""
    return Layer("project", tmp_path / "project")


def _stack(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> list[Layer]:
    """The full three-layer stack (package, user, project)."""
    return [mini_package, user_layer, project_layer]


def _find_skill(skills: Sequence[ResolvedSkill], name: str) -> ResolvedSkill:
    """Return one ResolvedSkill by name (fails loudly when absent)."""
    for skill in skills:
        if skill.name == name:
            return skill
    raise AssertionError(f"skill {name!r} not resolved")


# ---------------------------------------------------------------------------
# context-file resolution: every operation and combination
# ---------------------------------------------------------------------------
def test_file_default_is_package_base(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """No overrides: package AGENTS.md resolves verbatim."""
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == "# PACKAGE AGENTS\n\nbase rules."
    assert resolved.parts[0].kind == "base"
    assert resolved.parts[0].layer == "package"
    assert not resolved.disabled
    assert not resolved.replaced


def test_file_replace_in_user_layer(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """user/AGENTS.md replaces the package base entirely."""
    _write(user_layer.directory / "AGENTS.md", "# USER REPLACEMENT")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == "# USER REPLACEMENT"
    assert resolved.replaced
    assert resolved.parts[0].kind == "replace"
    assert resolved.parts[0].layer == "user"


def test_file_project_replace_beats_user_replace(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Project is the highest layer: its AGENTS.md wins over user's."""
    _write(user_layer.directory / "AGENTS.md", "# USER")
    _write(project_layer.directory / "AGENTS.md", "# PROJECT")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == "# PROJECT"
    assert resolved.parts[0].layer == "project"


def test_file_append_accumulates_in_layer_order(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Appends from every layer stack below the (kept) package base."""
    _write(user_layer.directory / "AGENTS.append.md", "user extra")
    _write(project_layer.directory / "AGENTS.append.md", "project extra")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == (
        "# PACKAGE AGENTS\n\nbase rules.\n\nuser extra\n\nproject extra"
    )
    assert [part.kind for part in resolved.parts] == [
        "base",
        "append",
        "append",
    ]


def test_file_replace_plus_append_from_lower_layer(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """A project replacement does not silence a user append."""
    _write(user_layer.directory / "AGENTS.append.md", "user extra")
    _write(project_layer.directory / "AGENTS.md", "# PROJECT")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == "# PROJECT\n\nuser extra"


def test_file_disabled_drops_base(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """AGENTS.disabled in any layer disables the base entirely."""
    _write(user_layer.directory / "AGENTS.disabled", "")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content is None
    assert resolved.disabled
    # the dropped base is still reported for --show-sources
    assert resolved.parts[0].kind == "base"


def test_file_disabled_beats_project_replacement(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """A disable marker kills replacements from higher layers too."""
    _write(user_layer.directory / "AGENTS.disabled", "")
    _write(project_layer.directory / "AGENTS.md", "# PROJECT")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content is None


def test_file_disabled_with_append_leaves_appends_only(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Append + disabled resolves to append-only content."""
    _write(user_layer.directory / "AGENTS.disabled", "")
    _write(project_layer.directory / "AGENTS.append.md", "only extra")
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.content == "only extra"


def test_file_missing_everywhere_is_none(
    user_layer: Layer, project_layer: Layer
) -> None:
    """No layer provides the file: content None, no source parts."""
    empty_package = Layer("package", project_layer.directory / "nowhere")
    resolved = resolve_context_file(
        "AGENTS.md", [empty_package, user_layer, project_layer]
    )
    assert resolved.content is None
    assert resolved.parts == ()


def test_file_whitespace_only_base_is_none(
    user_layer: Layer, project_layer: Layer, tmp_path: Path
) -> None:
    """An empty result (whitespace-only base, no appends) -> None."""
    package_dir = tmp_path / "ws-package"
    _write(package_dir / "AGENTS.md", "  \n\n")
    resolved = resolve_context_file(
        "AGENTS.md",
        [Layer("package", package_dir), user_layer, project_layer],
    )
    assert resolved.content is None


# ---------------------------------------------------------------------------
# skill resolution: every operation and combination
# ---------------------------------------------------------------------------
def test_skills_defaults_sorted_and_active(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """No overrides: both package skills active, sorted by name."""
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    assert [skill.name for skill in skills] == ["alpha-skill", "beta-skill"]
    assert all(skill.status == "default" for skill in skills)
    assert all(skill.active for skill in skills)
    assert _find_skill(skills, "alpha-skill").content == "# ALPHA\npackaged"


def test_skill_replaced_in_user_layer(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """user/skills/alpha-skill.md replaces the packaged skill."""
    _write(user_layer.directory / "skills" / "alpha-skill.md", "# USER A")
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    alpha = _find_skill(skills, "alpha-skill")
    assert alpha.status == "replaced"
    assert alpha.content == "# USER A"
    assert not alpha.appended


def test_skill_project_replace_beats_user_replace(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Highest layer wins the {name}.md replacement."""
    _write(user_layer.directory / "skills" / "alpha-skill.md", "# USER A")
    _write(project_layer.directory / "skills" / "alpha-skill.md", "# PROJ")
    alpha = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "alpha-skill",
    )
    assert alpha.content == "# PROJ"


def test_new_skill_added_from_user_layer(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """An unknown {name}.md in a layer adds a brand-new skill."""
    _write(user_layer.directory / "skills" / "zzz-new.md", "# NEW")
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    names = [skill.name for skill in skills]
    assert names == ["alpha-skill", "beta-skill", "zzz-new"]
    assert _find_skill(skills, "zzz-new").status == "added"


def test_skill_append_from_any_layer(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """{name}.append.md stacks below the base in layer order."""
    _write(user_layer.directory / "skills" / "alpha-skill.append.md", "u+")
    _write(project_layer.directory / "skills" / "alpha-skill.append.md", "p+")
    alpha = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "alpha-skill",
    )
    assert alpha.content == "# ALPHA\npackaged\n\nu+\n\np+"
    assert alpha.appended


def test_skill_append_to_added_skill(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Appends attach to skills introduced by an override layer."""
    _write(user_layer.directory / "skills" / "zzz-new.md", "# NEW")
    _write(project_layer.directory / "skills" / "zzz-new.append.md", "x")
    added = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "zzz-new",
    )
    assert added.status == "added"
    assert added.content == "# NEW\n\nx"


def test_skill_disabled_by_name(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """skills/.disabled drops the named skill; comments are honored."""
    disabled_list = _write(
        user_layer.directory / "skills" / ".disabled",
        "# whole-line comment\nalpha-skill  # trailing comment\n\n",
    )
    assert _parse_disabled_list(disabled_list) == {"alpha-skill"}
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    alpha = _find_skill(skills, "alpha-skill")
    assert alpha.disabled
    assert not alpha.active
    assert alpha.content is None
    assert _find_skill(skills, "beta-skill").active


def test_skill_disabled_beats_replacement(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """Disabling applies by name even when a layer replaces the file."""
    _write(user_layer.directory / "skills" / ".disabled", "alpha-skill\n")
    _write(project_layer.directory / "skills" / "alpha-skill.md", "# PROJ")
    alpha = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "alpha-skill",
    )
    assert alpha.disabled
    assert alpha.content is None


def test_skill_disabled_with_append_stays_active_append_only(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """A disabled skill with an append resolves append-only."""
    _write(user_layer.directory / "skills" / ".disabled", "alpha-skill\n")
    _write(
        project_layer.directory / "skills" / "alpha-skill.append.md",
        "left",
    )
    alpha = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "alpha-skill",
    )
    assert alpha.disabled
    assert alpha.active
    assert alpha.content == "left"


def test_disabled_all_kills_defaults_keeps_new(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """.disabled-all removes every default skill; added skills stay."""
    _write(user_layer.directory / "skills" / DISABLE_ALL_NAME, "")
    _write(project_layer.directory / "skills" / "zzz-new.md", "# NEW")
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    active_names = [skill.name for skill in skills if skill.active]
    assert active_names == ["zzz-new"]


def test_disabled_all_plus_named_disable_of_new_skill(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """A newly added skill can still be killed via skills/.disabled."""
    _write(user_layer.directory / "skills" / DISABLE_ALL_NAME, "")
    _write(project_layer.directory / "skills" / "zzz-new.md", "# NEW")
    _write(project_layer.directory / "skills" / ".disabled", "zzz-new\n")
    skills = resolve_skills(_stack(mini_package, user_layer, project_layer))
    assert not any(skill.active for skill in skills)


def test_skill_replaced_by_empty_file_is_inactive(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """A replacement that is empty deactivates the skill."""
    _write(user_layer.directory / "skills" / "alpha-skill.md", "   ")
    alpha = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "alpha-skill",
    )
    assert alpha.status == "replaced"
    assert not alpha.active


def test_orphan_skill_append_resolves_append_only(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """An append naming no known skill is kept as an append-only skill."""
    _write(project_layer.directory / "skills" / "ghost.append.md", "g+")
    ghost = _find_skill(
        resolve_skills(_stack(mini_package, user_layer, project_layer)),
        "ghost",
    )
    assert ghost.status == "added"
    assert ghost.content == "g+"
    assert ghost.appended


# ---------------------------------------------------------------------------
# platform default directories and layer building
# ---------------------------------------------------------------------------
def test_default_user_dir_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows: %APPDATA%/atropos/agent; unset -> ~/AppData/Roaming.

    The APPDATA branch is compared via ``PureWindowsPath`` so the
    Windows-path semantics hold on any host: a plain ``Path`` on posix
    does not treat backslashes as separators, which made the original
    literal-path assertion pass only when the suite ran on Windows.
    """
    monkeypatch.setattr("atropos.agent._resolution._is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", r"C:\Users\example-user\AppData\Roaming")
    assert (
        PureWindowsPath(str(default_user_dir())).as_posix()
        == "C:/Users/example-user/AppData/Roaming/atropos/agent"
    )
    monkeypatch.delenv("APPDATA", raising=False)
    expected = Path.home() / "AppData" / "Roaming" / "atropos" / "agent"
    assert default_user_dir() == expected


def test_default_user_dir_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX: XDG_CONFIG_HOME wins, else ~/.config."""
    monkeypatch.setattr(
        "atropos.agent._resolution._is_windows", lambda: False
    )
    monkeypatch.setattr(
        Path, "home", classmethod(lambda _cls: Path("/fake/home"))
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/cfg")
    assert default_user_dir() == Path("/custom/cfg/atropos/agent")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    expected = Path("/fake/home/.config/atropos/agent")
    assert default_user_dir() == expected


def test_default_project_dir_uses_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project layer is <cwd>/.atropos/agent, resolved at call time."""
    monkeypatch.chdir(tmp_path)
    assert default_project_dir() == tmp_path / ".atropos" / "agent"


def test_build_layers_defaults_and_injection(
    tmp_path: Path,
    user_layer: Layer,
    project_layer: Layer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None dirs become platform defaults; explicit dirs are verbatim."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "atropos.agent._resolution._is_windows", lambda: False
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")
    layers = build_layers(tmp_path / "pkg")
    assert [layer.label for layer in layers] == [
        "package",
        "user",
        "project",
    ]
    assert layers[0].directory == tmp_path / "pkg"
    assert layers[1].directory == Path("/custom/xdg/atropos/agent")
    assert layers[2].directory == tmp_path / ".atropos" / "agent"
    injected = build_layers(
        tmp_path / "pkg",
        user_dir=user_layer.directory,
        project_dir=project_layer.directory,
    )
    assert [layer.directory for layer in injected] == [
        tmp_path / "pkg",
        user_layer.directory,
        project_layer.directory,
    ]


# ---------------------------------------------------------------------------
# public API against the real packaged layer
# ---------------------------------------------------------------------------
def test_package_paths_exist() -> None:
    """The wheel-shipped package layer is importable and complete."""
    assert AGENT_DIR.is_dir()
    assert AGENTS_MD.exists()
    assert CLAUDE_MD.exists()
    assert SKILLS_DIR.is_dir()
    assert (AGENT_DIR / "VERSION").exists()
    assert (AGENT_DIR / "MANIFEST.toml").exists()
    assert len(list(SKILLS_DIR.glob("*.md"))) == 11


def test_public_api_with_pinned_layers(tmp_path: Path) -> None:
    """list_skills/read_agent_context/read_skill on pinned layers."""
    nowhere = tmp_path / "nowhere"
    assert list_skills(user_dir=nowhere, project_dir=nowhere) == (
        PACKAGE_ELEVEN
    )
    context = read_agent_context(user_dir=nowhere, project_dir=nowhere)
    assert context is not None
    assert "===== AGENTS.md =====" in context
    assert "===== CLAUDE.md =====" in context
    assert read_skill(
        "02-run-pipeline", user_dir=nowhere, project_dir=nowhere
    ).startswith("---")


def test_read_skill_unknown_name_raises(tmp_path: Path) -> None:
    """An unknown skill name fails with KeyError listing actives."""
    nowhere = tmp_path / "nowhere"
    with pytest.raises(KeyError, match="unknown or empty skill"):
        read_skill("no-such-skill", user_dir=nowhere, project_dir=nowhere)


def test_read_agent_context_none_when_all_disabled(tmp_path: Path) -> None:
    """Both context files disabled: read_agent_context -> None."""
    project_dir = tmp_path / "project"
    _write(project_dir / "AGENTS.disabled", "")
    _write(project_dir / "CLAUDE.disabled", "")
    assert (
        read_agent_context(
            user_dir=tmp_path / "nowhere", project_dir=project_dir
        )
        is None
    )


def test_public_api_respects_project_overrides(tmp_path: Path) -> None:
    """The public API applies project-layer file and skill overrides."""
    project_dir = tmp_path / "project"
    _write(project_dir / "AGENTS.append.md", "project addition")
    _write(project_dir / "skills" / ".disabled", "11-premise-check\n")
    nowhere = tmp_path / "nowhere"
    assert list_skills(user_dir=nowhere, project_dir=project_dir) == [
        name for name in PACKAGE_ELEVEN if name != "11-premise-check"
    ]
    context = read_agent_context(user_dir=nowhere, project_dir=project_dir)
    assert context is not None
    assert "project addition" in context
    with pytest.raises(KeyError, match="11-premise-check"):
        read_skill(
            "11-premise-check", user_dir=nowhere, project_dir=project_dir
        )


def test_resolved_file_active_property(
    mini_package: Layer, user_layer: Layer, project_layer: Layer
) -> None:
    """ResolvedFile.active mirrors content presence."""
    resolved = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert resolved.active
    _write(user_layer.directory / "AGENTS.disabled", "")
    disabled = resolve_context_file(
        "AGENTS.md", _stack(mini_package, user_layer, project_layer)
    )
    assert not disabled.active


def test_read_text_unreadable_path_is_empty(tmp_path: Path) -> None:
    """_read_text swallows OSError (e.g. a directory) as empty text."""
    directory_path = tmp_path / "a-directory.md"
    directory_path.mkdir()
    assert _read_text(directory_path) == ""


def test_is_windows_matches_platform() -> None:
    """The platform seam reports the real os.name branch."""
    assert _is_windows() == (os.name == "nt")
