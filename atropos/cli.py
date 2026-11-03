"""Atropos command-line interface (stdlib argparse, no new deps).

Three subcommands around the :mod:`atropos.agent` context package:

* ``atropos agent-context`` - print the resolved agent context
  (``--content`` default, ``--path``, ``--show-sources``,
  ``--skill=NAME``, ``--skills-only``, ``--list-skills``);
* ``atropos agent-setup`` - scaffold user/project override layers
  (replace/append/disable stubs, skill switches, README hint; never
  writes without ``--yes`` or an interactive confirmation);
* ``atropos agent-doctor`` - verify the package layer (files, VERSION
  vs MANIFEST, skill inventory) and report user/project overrides,
  orphan appends and version notes.

Every handler accepts explicit ``user_dir`` / ``project_dir`` layer
paths (``None`` = platform defaults), which is how the tests pin the
layers deterministically.  Usage errors go through argparse (exit
code 2); command failures return non-zero from :func:`main`.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

from atropos import __version__
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
    CONTEXT_FILES,
    DISABLE_ALL_NAME,
    DISABLE_LIST_NAME,
    SKILLS_SUBDIR,
    Layer,
    ResolvedFile,
    ResolvedSkill,
    _parse_disabled_list,
    build_layers,
    default_project_dir,
    default_user_dir,
    resolve_context_file,
    resolve_skills,
)

__all__ = ["main"]

PROG = "atropos"
_APPEND_SUFFIX = ".append.md"


# ---------------------------------------------------------------------------
# agent-context
# ---------------------------------------------------------------------------
def _print_paths(user_dir: Path | None, project_dir: Path | None) -> int:
    """Print the contributing file path(s) of each context file."""
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    for file_name in CONTEXT_FILES:
        resolved = resolve_context_file(file_name, layers)
        print(f"{file_name}:")
        if not resolved.parts:
            print("  (none)")
            continue
        for part in resolved.parts:
            dropped = resolved.disabled and part.kind != "append"
            marker = "  [dropped: disabled]" if dropped else ""
            print(f"  {part.kind:<8} {part.layer:<8} {part.path}{marker}")
    return 0


def _file_state(resolved: ResolvedFile) -> str:
    """Human status of one resolved context file."""
    has_appends = any(part.kind == "append" for part in resolved.parts)
    if resolved.disabled:
        return "disabled (appends only)" if has_appends else "disabled"
    if not resolved.parts:
        return "none"
    if resolved.replaced:
        return "replaced" + (" + appends" if has_appends else "")
    return "default" + (" + appends" if has_appends else "")


def _print_file_sources(resolved: ResolvedFile) -> None:
    """Print the source report block for one context file."""
    print(f"{resolved.name}: {_file_state(resolved)}")
    for part in resolved.parts:
        dropped = resolved.disabled and part.kind != "append"
        marker = "  [dropped: disabled]" if dropped else ""
        print(
            f"  {part.kind:<8} {part.layer:<8} {part.lines:>3} lines"
            f"  {part.path}{marker}"
        )


def _print_skill_sources(skill: ResolvedSkill) -> None:
    """Print the source report line(s) for one skill."""
    if skill.disabled:
        state = "disabled" + (" (appends only)" if skill.appended else "")
    elif not skill.active:
        state = f"{skill.status} (empty)"
    else:
        state = skill.status + (" + appends" if skill.appended else "")
    origin = "  (no base file)"
    if skill.parts:
        base = skill.parts[0]
        origin = f"{base.layer:<8} {base.path}"
    print(f"  {skill.name:<26} {state:<24} {origin}")
    for part in skill.parts[1:]:
        print(f"  {'':26} {'append':<24} {part.layer:<8} {part.path}")


def _print_sources(user_dir: Path | None, project_dir: Path | None) -> int:
    """Print the full layer/file/skill source report."""
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    print("layers:")
    for layer in layers:
        state = "" if layer.directory.is_dir() else " (not found)"
        print(f"  {layer.label:<8} {layer.directory}{state}")
    print()
    print("context files:")
    for file_name in CONTEXT_FILES:
        _print_file_sources(resolve_context_file(file_name, layers))
    print()
    skills = resolve_skills(layers)
    active = [skill for skill in skills if skill.active]
    inactive = [skill for skill in skills if not skill.active]
    print(f"skills: {len(active)} active, {len(inactive)} disabled/empty")
    for skill in skills:
        _print_skill_sources(skill)
    return 0


def _print_skill_names(
    user_dir: Path | None, project_dir: Path | None
) -> int:
    """Print active skill names, one per line."""
    for name in list_skills(user_dir, project_dir):
        print(name)
    return 0


def _print_one_skill(
    skill_name: str, user_dir: Path | None, project_dir: Path | None
) -> int:
    """Print one resolved skill; non-zero when the name is unknown."""
    try:
        text = read_skill(skill_name, user_dir, project_dir)
    except KeyError as error:
        print(f"atropos: {error}", file=sys.stderr)
        return 1
    print(text)
    return 0


def _print_all_skills(user_dir: Path | None, project_dir: Path | None) -> int:
    """Print the resolved text of every active skill."""
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    for skill in resolve_skills(layers):
        if skill.active and skill.content is not None:
            print(f"===== skill: {skill.name} =====")
            print(skill.content)
    return 0


def _print_context(user_dir: Path | None, project_dir: Path | None) -> int:
    """Print the resolved AGENTS.md + CLAUDE.md context."""
    context = read_agent_context(user_dir, project_dir)
    if context is None:
        print(
            "atropos: no agent context resolves (all layers disabled)",
            file=sys.stderr,
        )
        return 1
    print(context)
    return 0


def _cmd_agent_context(
    args: argparse.Namespace,
    user_dir: Path | None,
    project_dir: Path | None,
) -> int:
    """Handle ``atropos agent-context``."""
    if args.list_skills:
        return _print_skill_names(user_dir, project_dir)
    if args.skill is not None:
        return _print_one_skill(str(args.skill), user_dir, project_dir)
    if args.show_sources:
        return _print_sources(user_dir, project_dir)
    if args.path:
        return _print_paths(user_dir, project_dir)
    if args.skills_only:
        return _print_all_skills(user_dir, project_dir)
    return _print_context(user_dir, project_dir)


# ---------------------------------------------------------------------------
# agent-setup
# ---------------------------------------------------------------------------
_SETUP_README_TEMPLATE = """\
# Atropos agent overrides ({scope} layer)

Files in this directory override the packaged agent context
(`atropos/agent/`): this layer {precedes} the package defaults.

- `AGENTS.md` / `CLAUDE.md`  - replace the packaged file entirely;
- `AGENTS.append.md` / `CLAUDE.append.md` - appended to the resolved
  file;
- `AGENTS.disabled` / `CLAUDE.disabled` - empty marker that disables
  the base file (appends still apply);
- `skills/{name}.md` - replace or add a skill; `skills/{name}.append.md`
  - append to a skill;
- `skills/.disabled` - one skill name per line (`#` comments);
- `skills/.disabled-all` - disable every packaged default skill.

Inspect the effective result: `atropos agent-context --show-sources`.
Scaffold this directory: `atropos agent-setup --scope={scope}`.
"""


@dataclass(frozen=True)
class _SetupAction:
    """One planned filesystem mutation of ``agent-setup``.

    ``overwrite`` marks merge-style actions (the ``skills/.disabled``
    list) whose content already preserves the existing lines; every
    other action never touches an existing file.
    """

    path: Path
    content: str
    create: bool  # False = already exists, will be skipped
    overwrite: bool = False


def _setup_readme(scope: str) -> str:
    """README hint text placed into a freshly scaffolded layer."""
    precedes = (
        "sits between the package defaults and the project layer"
        if scope == "user"
        else "beats every other layer"
    )
    # plain replaces: the template keeps literal {name} placeholders
    # that str.format would treat as arguments
    return _SETUP_README_TEMPLATE.replace("{scope}", scope).replace(
        "{precedes}", precedes
    )


def _replace_stub(stem: str, scope: str) -> str:
    """Placeholder for a full file replacement."""
    return (
        f"# {stem}.md - {scope} override (replace)\n"
        "\n"
        f"This file REPLACES the packaged default {stem}.md for the\n"
        f"{scope} layer. Replace this placeholder with your content;\n"
        "inspect the result with `atropos agent-context --show-sources`.\n"
    )


def _append_stub(stem: str, scope: str) -> str:
    """Placeholder appended below the packaged file."""
    return (
        f"# {stem}.append.md - {scope} additions\n"
        "\n"
        f"Everything below is APPENDED to the resolved {stem}.md\n"
        "(after the package default and any lower-layer appends).\n"
    )


def _ask_scope() -> str | None:
    """Interactively ask for the setup scope; ``None`` = aborted."""
    while True:
        answer = input("agent-setup scope [project/user]: ").strip().lower()
        if answer in {"project", "user"}:
            return answer
        if answer in {"", "q", "quit"}:
            return None
        print("answer 'project' or 'user' (or just Enter to abort)")


def _setup_plan(
    args: argparse.Namespace, scope: str, target: Path
) -> list[_SetupAction]:
    """Plan every file agent-setup would create in ``target``."""
    actions: list[tuple[Path, str, bool]] = []
    file_names = (
        ["AGENTS.md", "CLAUDE.md"] if args.file is None else [args.file]
    )
    mode = str(args.mode)
    for file_name in file_names:
        stem = str(file_name)[: -len(".md")]
        if mode == "replace":
            actions.append(
                (target / str(file_name), _replace_stub(stem, scope), False)
            )
        elif mode == "append":
            actions.append(
                (
                    target / f"{stem}{_APPEND_SUFFIX}",
                    _append_stub(stem, scope),
                    False,
                )
            )
        elif mode == "disable":
            actions.append((target / f"{stem}.disabled", "", False))
    skills_dir = target / SKILLS_SUBDIR
    if args.skills == "none":
        actions.append((skills_dir / DISABLE_ALL_NAME, "", False))
    disable_names = [str(name) for name in args.disable]
    if disable_names:
        disabled_path = skills_dir / DISABLE_LIST_NAME
        existing = _parse_disabled_list(disabled_path)
        pending = [name for name in disable_names if name not in existing]
        if pending:
            content = ""
            if disabled_path.is_file():
                content = disabled_path.read_text(encoding="utf-8")
                if content and not content.endswith("\n"):
                    content += "\n"
            content += "".join(f"{name}\n" for name in pending)
            # merge-style write: content already keeps the old entries
            actions.append((disabled_path, content, True))
    actions.append((target / "README.md", _setup_readme(scope), False))
    return [
        _SetupAction(
            path=path,
            content=content,
            create=not path.exists(),
            overwrite=overwrite,
        )
        for path, content, overwrite in actions
    ]


def _setup_target(
    scope: str, user_dir: Path | None, project_dir: Path | None
) -> Path:
    """Layer directory a setup run writes into (injected or default)."""
    if scope == "user":
        return default_user_dir() if user_dir is None else user_dir
    return default_project_dir() if project_dir is None else project_dir


def _cmd_agent_setup(
    args: argparse.Namespace,
    user_dir: Path | None,
    project_dir: Path | None,
) -> int:
    """Handle ``atropos agent-setup``; writes only with ``--yes``."""
    scope = args.scope
    if scope is None:
        if not sys.stdin.isatty():
            print(
                "atropos: agent-setup needs --scope=project or"
                " --scope=user (stdin is not interactive)",
                file=sys.stderr,
            )
            return 1
        scope = _ask_scope()
        if scope is None:
            print("atropos: no scope chosen; nothing done.", file=sys.stderr)
            return 1
    target = _setup_target(str(scope), user_dir, project_dir)
    plan = _setup_plan(args, str(scope), target)
    print(f"agent-setup: scope={scope}, mode={args.mode}, target={target}")
    for action in plan:
        if action.create:
            marker = "create"
        elif action.overwrite:
            marker = "update"
        else:
            marker = "skip (exists)"
        print(f"  [{marker}] {action.path}")
    if args.dry_run:
        print("agent-setup: dry-run, nothing written.")
        return 0
    if not args.yes:
        print(
            "agent-setup: nothing written; rerun with --yes to apply.",
            file=sys.stderr,
        )
        return 1
    written = 0
    for action in plan:
        existed = action.path.exists()
        if existed and not action.overwrite:
            print(f"  skipped (exists) {action.path}")
            continue
        action.path.parent.mkdir(parents=True, exist_ok=True)
        action.path.write_text(action.content, encoding="utf-8")
        marker = "updated" if existed else "created"
        print(f"  {marker} {action.path}")
        written += 1
    print(f"agent-setup: {written} file(s) written to {target}")
    return 0


# ---------------------------------------------------------------------------
# agent-doctor
# ---------------------------------------------------------------------------
def _package_version() -> str:
    """Installed atropos version; source-tree fallback on failure."""
    try:
        return _dist_version("atropos")
    except PackageNotFoundError:
        return str(__version__)


def _load_manifest() -> tuple[dict[str, Any] | None, str]:
    """Parse MANIFEST.toml; ``(dict, "")`` or ``(None, reason)``."""
    manifest_path = AGENT_DIR / "MANIFEST.toml"
    try:
        with manifest_path.open("rb") as handle:
            return tomllib.load(handle), ""
    except OSError as error:
        return None, str(error)
    except tomllib.TOMLDecodeError as error:
        return None, str(error)


def _doctor_package(errors: list[str], notes: list[str]) -> None:
    """Verify the packaged layer: files, VERSION vs MANIFEST, skills."""
    version_path = AGENT_DIR / "VERSION"
    agent_version = (
        version_path.read_text(encoding="utf-8").strip()
        if version_path.is_file()
        else ""
    )
    required_paths = (
        AGENTS_MD,
        CLAUDE_MD,
        version_path,
        AGENT_DIR / "MANIFEST.toml",
        SKILLS_DIR,
    )
    errors.extend(
        f"package layer missing: {required}"
        for required in required_paths
        if not required.exists()
    )
    manifest, manifest_error = _load_manifest()
    if manifest is None:
        errors.append(f"MANIFEST.toml unreadable: {manifest_error}")
        return
    agent_section = manifest.get("agent", {})
    manifest_version = str(agent_section.get("version", ""))
    if manifest_version != agent_version:
        errors.append(
            f"MANIFEST version {manifest_version!r} != VERSION"
            f" {agent_version!r}"
        )
    declared: set[str] = set()
    for entry in manifest.get("skills", []):
        name = str(entry.get("name", ""))
        rel_path = str(entry.get("path", ""))
        declared.add(name)
        if not (AGENT_DIR / rel_path).is_file():
            errors.append(
                f"MANIFEST declares missing skill: {name} ({rel_path})"
            )
    notes.extend(
        f"skill file not declared in MANIFEST: {skill_path.name}"
        for skill_path in sorted(SKILLS_DIR.glob("*.md"))
        if skill_path.stem not in declared
    )


def _doctor_layer(
    layer: Layer,
    default_names: set[str],
    warnings: list[str],
    notes: list[str],
) -> None:
    """Report one override layer: inventory and orphan diagnostics."""
    if not layer.directory.is_dir():
        notes.append(f"{layer.label} layer: none ({layer.directory})")
        return
    inventory = [
        path.name
        for path in sorted(layer.directory.iterdir())
        if path.is_file()
    ]
    skills_dir = layer.directory / SKILLS_SUBDIR
    if skills_dir.is_dir():
        inventory.extend(
            f"skills/{path.name}"
            for path in sorted(skills_dir.iterdir())
            if path.is_file()
        )
    if inventory:
        notes.append(
            f"{layer.label} layer: {len(inventory)} file(s): "
            + ", ".join(inventory)
        )
    else:
        notes.append(f"{layer.label} layer: directory exists, no overrides")
    context_append_names = {
        f"AGENTS{_APPEND_SUFFIX}",
        f"CLAUDE{_APPEND_SUFFIX}",
    }
    warnings.extend(
        f"{layer.label} layer: orphan append {append_path.name}"
        for append_path in sorted(layer.directory.glob(f"*{_APPEND_SUFFIX}"))
        if append_path.name not in context_append_names
    )
    if not skills_dir.is_dir():
        return
    layer_skill_names = {
        path.stem
        for path in skills_dir.glob("*.md")
        if not path.stem.endswith(".append")
    }
    known_names = default_names | layer_skill_names
    warnings.extend(
        f"{layer.label} layer: orphan skill append {append_path.name}"
        for append_path in sorted(skills_dir.glob(f"*{_APPEND_SUFFIX}"))
        if append_path.name[: -len(_APPEND_SUFFIX)] not in known_names
    )
    warnings.extend(
        f"{layer.label} layer: .disabled lists unknown skill {name!r}"
        for name in sorted(
            _parse_disabled_list(skills_dir / DISABLE_LIST_NAME)
        )
        if name not in known_names
    )


def _cmd_agent_doctor(user_dir: Path | None, project_dir: Path | None) -> int:
    """Handle ``atropos agent-doctor``; exit 1 on any error."""
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    _doctor_package(errors, notes)
    default_names = {
        path.stem
        for path in SKILLS_DIR.glob("*.md")
        if not path.stem.endswith(".append")
    }
    layers = build_layers(AGENT_DIR, user_dir, project_dir)
    for layer in layers:
        if layer.label == "package":
            continue
        _doctor_layer(layer, default_names, warnings, notes)
    version_path = AGENT_DIR / "VERSION"
    agent_version = (
        version_path.read_text(encoding="utf-8").strip()
        if version_path.is_file()
        else ""
    )
    package_version = _package_version()
    if package_version != agent_version:
        warnings.append(
            f"version note: agent context {agent_version} vs atropos"
            f" package {package_version}"
        )
    for note in notes:
        print(f"ok   {note}")
    for warning in warnings:
        print(f"warn {warning}")
    for error in errors:
        print(f"ERROR {error}")
    print(f"doctor: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# parser + entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree for the three agent subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Atropos hypothesis-killing framework: agent context tools."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    context_parser = subparsers.add_parser(
        "agent-context",
        help="print the resolved agent context (files and skills)",
    )
    context_group = context_parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--content",
        action="store_true",
        help="resolved AGENTS.md + CLAUDE.md text (default)",
    )
    context_group.add_argument(
        "--path",
        action="store_true",
        help="contributing file path(s) per context file",
    )
    context_group.add_argument(
        "--show-sources",
        action="store_true",
        help="layer/file/skill source report with statuses",
    )
    context_group.add_argument(
        "--skill",
        metavar="NAME",
        help="resolved text of one skill",
    )
    context_group.add_argument(
        "--skills-only",
        action="store_true",
        help="resolved text of every active skill",
    )
    context_group.add_argument(
        "--list-skills",
        action="store_true",
        help="active skill names, one per line",
    )

    setup_parser = subparsers.add_parser(
        "agent-setup",
        help="scaffold a user/project override layer",
    )
    setup_parser.add_argument(
        "--scope",
        choices=["project", "user"],
        help="which layer to write (asked interactively when omitted)",
    )
    setup_parser.add_argument(
        "--mode",
        choices=["default", "append", "replace", "disable"],
        default="default",
        help="override stub for AGENTS.md/CLAUDE.md (default: none)",
    )
    setup_parser.add_argument(
        "--file",
        choices=["AGENTS.md", "CLAUDE.md"],
        help="restrict the file stubs to one context file",
    )
    setup_parser.add_argument(
        "--skills",
        choices=["all", "none"],
        help="all: keep defaults; none: write skills/.disabled-all",
    )
    setup_parser.add_argument(
        "--disable",
        metavar="NAME",
        action="append",
        default=[],
        help="add a skill name to skills/.disabled (repeatable)",
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without writing anything",
    )
    setup_parser.add_argument(
        "--yes",
        action="store_true",
        help="apply the plan without asking",
    )

    subparsers.add_parser(
        "agent-doctor",
        help="verify the agent context layers and overrides",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> int:
    """CLI entry point; returns the process exit code.

    ``argv`` defaults to ``sys.argv[1:]``; ``user_dir`` / ``project_dir``
    pin the override layers (``None`` = platform defaults) so tests and
    embedding code stay deterministic.

    Example::

        >>> from pathlib import Path
        >>> nowhere = Path("atropos-cli-no-such-layer")
        >>> main(
        ...     ["agent-context", "--list-skills"],
        ...     user_dir=nowhere,
        ...     project_dir=nowhere,
        ... )
        01-preregister
        02-run-pipeline
        03-read-graveyard
        11-premise-check
        12-null-control
        13-signal-test
        14-incremental-test
        15-robustness-battery
        16-walkforward-calendar
        0
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)
    if command == "agent-context":
        return _cmd_agent_context(args, user_dir, project_dir)
    if command == "agent-setup":
        return _cmd_agent_setup(args, user_dir, project_dir)
    return _cmd_agent_doctor(user_dir, project_dir)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
