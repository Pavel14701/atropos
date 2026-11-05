"""Portability guard: shipped files carry no owner-machine traces.

The distribution must be machine-independent: no personal user names,
no absolute Windows user directories, and no references to the private
donor repository the case-study numbers were taken from.  This module
scans every text artifact the repository ships (``*.py``, ``*.md``,
``*.toml``, ``*.ipynb``) for a fixed pattern list and fails on any
occurrence that is not whitelisted.  The whitelist is exact: any *new*
occurrence anywhere in the tree fails the suite.

The only whitelisted occurrences are synthetic path-semantics fixtures
that exercise Windows ``APPDATA`` resolution against the fictional
``example-user`` account (never a real user name) in
``tests/test_agent_context.py``.

The scanner skips its own source file: the pattern table below must
be written in plain text and would otherwise self-match.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]

_SCAN_SUFFIXES: frozenset[str] = frozenset({".py", ".md", ".toml", ".ipynb"})

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        "dist",
    }
)

_SELF: str = "tests/test_portability.py"

# Sensitive substrings, matched case-insensitively.  Backslash
# literals use doubled backslashes so this module's own source never
# contains the raw byte sequence it scans for.
_PATTERNS: tuple[str, ...] = (
    "pashtet",
    "c:\\users",
    "c:/users",
    "desktop\\projects",
    "desktop/projects",
    "deterministic-trading",
    "toothcake",
    "/home/",
    "/users/",
)

# Exact whitelisted (file, line) occurrences.  Keys are stripped,
# casefolded source lines; a hit on any other line fails.  Entries are
# the synthetic ``example-user`` Windows fixtures of the agent-context
# path-semantics test only.
_ALLOWED_LINES: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "tests/test_agent_context.py",
            'monkeypatch.setenv("appdata", '
            'r"c:\\users\\example-user\\appdata\\roaming")',
        ),
        (
            "tests/test_agent_context.py",
            '== "c:/users/example-user/appdata/roaming/atropos/agent"',
        ),
        # synthetic posix fixture of the platform-defaults test
        (
            "tests/test_agent_context.py",
            'expected = path("/fake/home/.config/atropos/agent")',
        ),
    }
)


def _scan_files() -> list[Path]:
    """Collect scannable repository files, skipping caches and venvs."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _SKIP_DIRS
        )
        for name in sorted(filenames):
            path = Path(dirpath, name)
            if path.suffix not in _SCAN_SUFFIXES:
                continue
            if path.relative_to(ROOT).as_posix() == _SELF:
                continue
            found.append(path)
    return found


def _line_at(text: str, start: int, end: int) -> str:
    """Return the stripped, casefolded line holding ``text[start:end]``."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip().casefold()


def _violations() -> list[str]:
    """Report non-whitelisted hits as ``"file: pattern -> line"``."""
    reports: list[str] = []
    for path in _scan_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in _PATTERNS:
            for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                line = _line_at(text, match.start(), match.end())
                if (rel, line) not in _ALLOWED_LINES:
                    reports.append(f"{rel}: {pattern!r} -> {line!r}")
    return reports


def test_shipped_files_have_no_owner_machine_traces() -> None:
    """No personal names, user paths, or donor-repo names anywhere.

    Scans every ``*.py``/``*.md``/``*.toml``/``*.ipynb`` artifact in
    the repository (``.venv``, caches, ``dist`` excluded) against the
    pattern list; only the exact whitelisted synthetic ``example-user``
    fixture lines may match.  A new occurrence of any pattern fails.
    """
    reports = _violations()
    assert not reports, "owner-machine traces found:\n" + "\n".join(reports)


def test_portability_whitelist_stays_exact() -> None:
    """Every whitelisted line still occurs verbatim in its file.

    Keeps the whitelist from rotting: if a whitelisted fixture line is
    reworded or removed, the whitelist must be updated in the same
    commit instead of silently allowing stale patterns.
    """
    for rel, line in sorted(_ALLOWED_LINES):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert line in text.casefold(), f"stale whitelist: {rel}: {line!r}"
