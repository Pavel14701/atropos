"""Code-convention checks that ruff cannot express.

Ruff's E741 only bans the visually ambiguous names (``l``, ``I``, ``O``).
Atropos bans EVERY single-character identifier outright: a name that short
never says what it holds, and this codebase kills hypotheses — the reader
must always know what is being killed and why.  The only exception is the
throwaway ``_``.

Checked identifier kinds:

* function, async function, class and lambda argument names,
* function, async function and class names,
* assignment, annotation-assignment, loop, comprehension and ``with``
  targets (every ``ast.Name`` in a Store context),
* ``except ... as name`` bindings,
* ``global`` / ``nonlocal`` declarations.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCAN_DIRS = (_PROJECT_ROOT / "atropos", Path(__file__).resolve().parent)

_ALLOWED_SINGLE_CHAR = {"_"}  # conventional throwaway placeholder


def _iter_python_files() -> Iterator[Path]:
    """Yield every tracked .py file of the package and the test suite."""
    for scan_dir in _SCAN_DIRS:
        yield from sorted(scan_dir.glob("*.py"))


def _argument_names(args: ast.arguments) -> Iterator[str]:
    """Yield all argument names of a function or lambda signature."""
    for arg_list in (
        args.posonlyargs,
        args.args,
        args.kwonlyargs,
        [args.vararg] if args.vararg else [],
        [args.kwarg] if args.kwarg else [],
    ):
        yield from (arg.arg for arg in arg_list)


def _single_letter_violations(tree: ast.Module, path: Path) -> list[str]:
    """Return human-readable violations found in one parsed module."""
    violations: list[str] = []

    def report(kind: str, name: str, lineno: int) -> None:
        if len(name) == 1 and name not in _ALLOWED_SINGLE_CHAR:
            violations.append(f"{path.name}:{lineno} {kind} {name!r}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            report("function name", node.name, node.lineno)
            for arg_name in _argument_names(node.args):
                report("function argument", arg_name, node.lineno)
        elif isinstance(node, ast.Lambda):
            for arg_name in _argument_names(node.args):
                report("lambda argument", arg_name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            report("class name", node.name, node.lineno)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            report("assigned name", node.id, node.lineno)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                report("except binding", node.name, node.lineno)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for declared in node.names:
                report("global/nonlocal name", declared, node.lineno)
    return violations


def test_no_single_letter_identifiers() -> None:
    """Every identifier in atropos/ and tests/ is at least two characters."""
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_single_letter_violations(tree, path))
    assert not violations, (
        "single-character identifiers are banned "
        "(only '_' is allowed):\n  " + "\n  ".join(violations)
    )
