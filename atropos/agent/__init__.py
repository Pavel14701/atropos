"""Atropos agent context: packaged default layer + override resolution.

This package ships the default agent context (``AGENTS.md``,
``CLAUDE.md`` and twelve methodology skills) and resolves it against
the user layer (``%APPDATA%/atropos/agent`` or ``~/.config/atropos/agent``)
and the project layer (``./.atropos/agent``).  The resolution engine
and its layer semantics live in :mod:`atropos.agent._resolution`.

Typical use::

    from atropos.agent import list_skills, read_agent_context, read_skill

    context = read_agent_context()        # AGENTS.md + CLAUDE.md text
    names = list_skills()                 # active skill names, sorted
    text = read_skill("01-preregister")   # one resolved skill

Every public function accepts ``user_dir`` / ``project_dir`` overrides
so callers (tests, the CLI) can pin the layers deterministically;
``None`` means "detect the platform defaults".
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from atropos.agent._resolution import (
    CONTEXT_FILES,
    ResolvedFile,
    ResolvedSkill,
    build_layers,
    resolve_context_file,
    resolve_skills,
)

__all__ = [
    "list_skills",
    "read_agent_context",
    "read_skill",
]

# importlib.resources locates the package layer both in a source tree
# and inside an installed wheel; str() adapts the Traversable to Path.
# The four path constants below are importable but deliberately NOT in
# __all__: pdoc renders the VALUE of every __all__ variable, which
# would bake an absolute, install-specific machine path (and the
# platform repr) into the committed API reference.
AGENT_DIR = Path(str(files("atropos.agent")))
AGENTS_MD = AGENT_DIR / "AGENTS.md"
CLAUDE_MD = AGENT_DIR / "CLAUDE.md"
SKILLS_DIR = AGENT_DIR / "skills"


def _resolved_files(
    user_dir: Path | None, project_dir: Path | None
) -> list[ResolvedFile]:
    """Resolve every context file over package/user/project layers."""
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    resolved_files = [
        resolve_context_file(name, layers) for name in CONTEXT_FILES
    ]
    return [
        resolved
        for resolved in resolved_files
        if resolved.content is not None
    ]


def _resolved_skills(
    user_dir: Path | None, project_dir: Path | None
) -> list[ResolvedSkill]:
    """Resolve every skill over package/user/project layers."""
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    return [skill for skill in resolve_skills(layers) if skill.active]


def read_agent_context(
    user_dir: Path | None = None, project_dir: Path | None = None
) -> str | None:
    """Resolved ``AGENTS.md`` + ``CLAUDE.md`` text, or ``None`` if empty.

    Each active file is prefixed with an ``===== NAME =====`` header so
    the consumer can attribute the text; disabled files contribute
    appends only (see :mod:`atropos.agent._resolution`).

    Example::

        >>> from pathlib import Path
        >>> nowhere = Path("atropos-no-such-layer")
        >>> context = read_agent_context(
        ...     user_dir=nowhere, project_dir=nowhere
        ... )
        >>> context is not None and "===== AGENTS.md =====" in context
        True
    """
    blocks = [
        f"===== {resolved.name} =====\n{resolved.content}"
        for resolved in _resolved_files(user_dir, project_dir)
    ]
    return "\n\n".join(blocks) if blocks else None


def list_skills(
    user_dir: Path | None = None, project_dir: Path | None = None
) -> list[str]:
    """Active skill names, stably sorted; overrides already applied.

    Example::

        >>> from pathlib import Path
        >>> nowhere = Path("atropos-no-such-layer")
        >>> names = list_skills(user_dir=nowhere, project_dir=nowhere)
        >>> len(names) >= 3 and names == sorted(names)
        True
    """
    return [skill.name for skill in _resolved_skills(user_dir, project_dir)]


def read_skill(
    name: str,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> str:
    """Resolved text of one skill, with appends/replacements applied.

    Raises ``KeyError`` when the name matches no skill in any layer or
    the skill resolves empty (disabled without appends).

    Example::

        >>> from pathlib import Path
        >>> nowhere = Path("atropos-no-such-layer")
        >>> text = read_skill(
        ...     "01-preregister", user_dir=nowhere, project_dir=nowhere
        ... )
        >>> text.startswith("---")
        True
    """
    active_skills = _resolved_skills(user_dir, project_dir)
    for skill in active_skills:
        if skill.name == name and skill.content is not None:
            return skill.content
    known = ", ".join(skill.name for skill in active_skills)
    raise KeyError(f"unknown or empty skill {name!r}; active: {known}")
