"""Three-layer agent-context resolution engine.

One uniform engine resolves both the context files (``AGENTS.md`` /
``CLAUDE.md``) and the skills, stacking three override layers from
low to high priority:

* ``package``  - the default layer shipped inside ``atropos/agent/``;
* ``user``     - ``%APPDATA%/atropos/agent`` on Windows,
  ``$XDG_CONFIG_HOME/atropos/agent`` (else ``~/.config/atropos/agent``)
  on POSIX;
* ``project``  - ``./.atropos/agent`` under the working directory.

Per-file semantics (``X`` is ``AGENTS`` or ``CLAUDE``):

* ``X.md`` in a higher layer replaces the lower-layer base entirely
  (project beats user beats package);
* ``X.append.md`` appends; multiple layers append in layer order;
* ``X.disabled`` (an empty marker file) disables the base - appends
  still resolve, so ``append + disabled`` yields append-only content;
* an empty resolved result is reported as ``None``.

Per-skill semantics inside each layer's ``skills/`` directory:

* ``{name}.md`` replaces the package skill of that name, or adds a new
  skill when the name is unknown to the package;
* ``{name}.append.md`` appends to the skill from any layer;
* ``skills/.disabled`` lists skill names to disable (one per line,
  ``#`` starts a comment);
* ``skills/.disabled-all`` disables every package-default skill (new
  skills from user/project layers stay active);
* a disabled skill with appends resolves append-only, mirroring the
  file semantics; a skill whose resolved content is empty is inactive.

The engine is pure directory arithmetic: every input is an explicit
layer path, so tests inject ``tmp_path`` layers instead of touching
the real user or project directories.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CONTEXT_FILES",
    "DISABLE_ALL_NAME",
    "DISABLE_LIST_NAME",
    "SKILLS_SUBDIR",
    "Layer",
    "ResolvedFile",
    "ResolvedSkill",
    "SourcePart",
    "build_layers",
    "default_project_dir",
    "default_user_dir",
    "resolve_context_file",
    "resolve_skills",
]

CONTEXT_FILES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
"""Context file names resolved per project by :func:`resolve_context_file`."""

SKILLS_SUBDIR = "skills"
DISABLE_LIST_NAME = ".disabled"
DISABLE_ALL_NAME = ".disabled-all"
_APPEND_SUFFIX = ".append.md"


@dataclass(frozen=True)
class Layer:
    """One override layer: a display label and its directory."""

    label: str
    directory: Path


@dataclass(frozen=True)
class SourcePart:
    """One physical file contributing to a resolved artifact."""

    layer: str
    path: Path
    kind: str  # "base" (package default) | "replace" | "append"
    lines: int


@dataclass(frozen=True)
class ResolvedFile:
    """Resolution result for one context file (``AGENTS.md``/...).

    ``content`` is ``None`` when nothing resolves (all layers disabled
    or every contributing file is empty).
    """

    name: str
    content: str | None
    parts: tuple[SourcePart, ...]
    disabled: bool
    replaced: bool

    @property
    def active(self) -> bool:
        """True when the file resolves to non-empty content."""
        return self.content is not None


@dataclass(frozen=True)
class ResolvedSkill:
    """Resolution result for one skill.

    ``status`` is ``"default"`` (package base), ``"replaced"`` (an
    override layer provides ``{name}.md`` for a packaged skill) or
    ``"added"`` (the name is new to the package).  ``active`` is False
    for disabled skills without appends and for empty resolutions.
    """

    name: str
    content: str | None
    parts: tuple[SourcePart, ...]
    status: str
    disabled: bool
    appended: bool

    @property
    def active(self) -> bool:
        """True when the skill resolves to non-empty content."""
        return self.content is not None


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file; unreadable files resolve as empty.

    A deliberate narrow fallback: ``agent-doctor`` lists unreadable
    overrides instead of crashing on permissions races.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _line_count(text: str) -> int:
    """Number of lines in ``text`` (for source reports)."""
    return len(text.splitlines())


def _join_blocks(blocks: Sequence[str]) -> str | None:
    """Join content blocks with a blank line; ``None`` when empty."""
    joined = "\n\n".join(block.strip("\n") for block in blocks).strip()
    return joined if joined else None


def resolve_context_file(
    file_name: str, layers: Sequence[Layer]
) -> ResolvedFile:
    """Resolve one context file across the layer stack.

    The first layer of ``layers`` is the package default (its ``X.md``
    is the ``base``); ``X.md`` in any later layer replaces it.  Later
    layers win, appends accumulate in layer order, and an ``X.disabled``
    marker in any layer drops the base while keeping appends.
    """
    stem = file_name[: -len(".md")]
    base_part: SourcePart | None = None
    replaced = False
    append_parts: list[SourcePart] = []
    disabled = False
    for index, layer in enumerate(layers):
        directory = layer.directory
        base_path = directory / file_name
        if base_path.is_file():
            text = _read_text(base_path)
            kind = "base" if index == 0 else "replace"
            base_part = SourcePart(
                layer.label, base_path, kind, _line_count(text)
            )
            replaced = index > 0
        append_path = directory / (stem + _APPEND_SUFFIX)
        if append_path.is_file():
            text = _read_text(append_path)
            append_parts.append(
                SourcePart(
                    layer.label, append_path, "append", _line_count(text)
                )
            )
        if (directory / (stem + ".disabled")).is_file():
            disabled = True
    blocks: list[str] = []
    if not disabled and base_part is not None:
        blocks.append(_read_text(base_part.path))
    blocks.extend(_read_text(part.path) for part in append_parts)
    parts = ((base_part,) if base_part is not None else ()) + tuple(
        append_parts
    )
    return ResolvedFile(
        name=file_name,
        content=_join_blocks(blocks),
        parts=parts,
        disabled=disabled,
        replaced=replaced,
    )


def _parse_disabled_list(path: Path) -> set[str]:
    """Names listed in a ``skills/.disabled`` file.

    One name per line; ``#`` starts a comment (full-line or trailing).
    """
    if not path.is_file():
        return set()
    names: set[str] = set()
    for raw_line in _read_text(path).splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            names.add(entry)
    return names


def _layer_skills_dir(layer: Layer) -> Path:
    """The ``skills/`` subdirectory of one layer."""
    return layer.directory / SKILLS_SUBDIR


def resolve_skills(layers: Sequence[Layer]) -> list[ResolvedSkill]:
    """Resolve every skill across the layer stack, sorted by name.

    The returned list contains active and disabled skills alike (callers
    filter on ``.active``); disabled entries carry their status so
    ``--show-sources`` can report them.
    """
    base_parts: dict[str, SourcePart] = {}
    default_names: set[str] = set()
    append_parts: dict[str, list[SourcePart]] = {}
    disabled_names: set[str] = set()
    disabled_all = False
    for index, layer in enumerate(layers):
        skills_dir = _layer_skills_dir(layer)
        if not skills_dir.is_dir():
            continue
        disabled_names |= _parse_disabled_list(skills_dir / DISABLE_LIST_NAME)
        if (skills_dir / DISABLE_ALL_NAME).is_file():
            disabled_all = True
        for skill_path in sorted(skills_dir.glob("*.md")):
            stem = skill_path.stem
            text = _read_text(skill_path)
            if stem.endswith(".append"):
                append_name = stem[: -len(".append")]
                append_parts.setdefault(append_name, []).append(
                    SourcePart(
                        layer.label,
                        skill_path,
                        "append",
                        _line_count(text),
                    )
                )
                continue
            kind = "base" if index == 0 else "replace"
            base_parts[stem] = SourcePart(
                layer.label, skill_path, kind, _line_count(text)
            )
            if index == 0:
                default_names.add(stem)
    resolved: list[ResolvedSkill] = []
    for name in sorted(base_parts):
        base_part = base_parts[name]
        appends = append_parts.pop(name, [])
        is_default = name in default_names
        if is_default:
            status = "default" if base_part.kind == "base" else "replaced"
        else:
            status = "added"
        disabled = name in disabled_names or (disabled_all and is_default)
        blocks: list[str] = []
        if not disabled:
            blocks.append(_read_text(base_part.path))
        blocks.extend(_read_text(part.path) for part in appends)
        parts = (base_part, *appends)
        resolved.append(
            ResolvedSkill(
                name=name,
                content=_join_blocks(blocks),
                parts=parts,
                status=status,
                disabled=disabled,
                appended=bool(appends),
            )
        )
    # appends naming a skill that no layer provides: still report the
    # append-only resolution so orphan appends are visible, not dropped
    for name in sorted(append_parts):
        appends = append_parts[name]
        blocks = [_read_text(part.path) for part in appends]
        resolved.append(
            ResolvedSkill(
                name=name,
                content=_join_blocks(blocks),
                parts=tuple(appends),
                status="added",
                disabled=False,
                appended=True,
            )
        )
    return resolved


def _is_windows() -> bool:
    """True on Windows (a seam so tests can pin the platform branch)."""
    return os.name == "nt"


def default_user_dir() -> Path:
    """Platform user-layer directory (never created, only computed).

    Windows: ``%APPDATA%/atropos/agent`` (falling back to
    ``~/AppData/Roaming`` when ``APPDATA`` is unset).  POSIX:
    ``$XDG_CONFIG_HOME/atropos/agent``, else ``~/.config/atropos/agent``.
    """
    if _is_windows():
        appdata = os.environ.get("APPDATA")
        config_root = (
            Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        )
    else:
        xdg_root = os.environ.get("XDG_CONFIG_HOME")
        config_root = Path(xdg_root) if xdg_root else Path.home() / ".config"
    return config_root / "atropos" / "agent"


def default_project_dir() -> Path:
    """Project-layer directory under the current working directory."""
    return Path.cwd() / ".atropos" / "agent"


def build_layers(
    package_dir: Path,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> list[Layer]:
    """Layer stack from low to high priority (package, user, project).

    ``None`` dirs fall back to the platform defaults
    (:func:`default_user_dir`, :func:`default_project_dir`); explicit
    paths are used verbatim, which is how tests and the CLI inject
    layers.
    """
    user_layer = default_user_dir() if user_dir is None else user_dir
    project_layer = (
        default_project_dir() if project_dir is None else project_dir
    )
    return [
        Layer("package", package_dir),
        Layer("user", user_layer),
        Layer("project", project_layer),
    ]
