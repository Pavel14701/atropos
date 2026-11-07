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

Single-letter KEYS of TypedDicts are historical public API, but only
the grandfathered pair (``n``, ``B`` -- users index the bootstrap
result dicts with ``out["n"]``); any OTHER single-letter TypedDict
field is banned exactly like a single-letter variable, so a fresh
TypedDict cannot smuggle a meaningless ``q`` key past the rule.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCAN_DIRS = (_PROJECT_ROOT / "atropos", Path(__file__).resolve().parent)

_ALLOWED_SINGLE_CHAR = {"_"}  # conventional throwaway placeholder

# Grandfathered single-letter TypedDict KEYS (historical public API of
# the bootstrap_ci result dicts); every other single-letter field of a
# TypedDict stays banned.
_TYPED_DICT_ALLOWED_FIELDS = frozenset({"n", "B"})


def _iter_python_files() -> Iterator[Path]:
    """Yield every tracked .py file of the package and the test suite.

    Recursive: subpackages such as ``atropos/agent/`` are inside the
    ban, not a blind spot (the old non-recursive ``glob`` missed them).
    """
    for scan_dir in _SCAN_DIRS:
        yield from sorted(scan_dir.rglob("*.py"))


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


def _is_typed_dict_base(base: ast.expr) -> bool:
    """True when a ClassDef base expression names ``TypedDict``."""
    if isinstance(base, ast.Name):
        return base.id == "TypedDict"
    return isinstance(base, ast.Attribute) and base.attr == "TypedDict"


def _typed_dict_store_name_ids(tree: ast.Module) -> set[int]:
    """``id()`` of allowed single-letter Store names in TypedDict bodies.

    Only the grandfathered ``n`` / ``B`` KEYS (historical public API of
    the bootstrap_ci result dicts) are exempt from the single-letter
    ban; any other single-letter TypedDict field is reported like a
    single-letter variable, so a new TypedDict cannot widen the hole.
    """
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_typed_dict_base(base) for base in node.bases):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Store)
                and sub.id in _TYPED_DICT_ALLOWED_FIELDS
            ):
                skipped.add(id(sub))
    return skipped


def _single_letter_violations(tree: ast.Module, path: Path) -> list[str]:
    """Return human-readable violations found in one parsed module."""
    violations: list[str] = []
    typed_dict_fields = _typed_dict_store_name_ids(tree)

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
            if id(node) not in typed_dict_fields:
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


def test_typed_dict_single_letter_fields_allowed_variables_banned() -> None:
    """Only ``n`` / ``B`` TypedDict KEYS pass; everything else fails.

    Covers both the plain ``TypedDict`` base and the
    ``typing.TypedDict`` attribute form, pins that any OTHER
    single-letter TypedDict field (``q``) is banned despite living in
    a TypedDict body, and that the exemption never leaks to ordinary
    assignments.
    """
    sample = Path("sample.py")
    for typed_dict_source in (
        "from typing import TypedDict\n"
        "\n"
        "class Result(TypedDict):\n"
        "    n: int\n"
        "    B: int\n",
        "import typing\n"
        "\n"
        "class Result(typing.TypedDict):\n"
        "    n: int\n"
        "    B: int\n",
    ):
        tree = ast.parse(typed_dict_source)
        assert _single_letter_violations(tree, sample) == []
    # a single-letter TypedDict field OUTSIDE the {"n", "B"} whitelist
    # is reported like any other single-letter assignment: the old
    # blanket exemption let any new TypedDict smuggle such keys past
    bad_field_source = (
        "from typing import TypedDict\n"
        "\n"
        "class Result(TypedDict):\n"
        "    q: float\n"
    )
    tree = ast.parse(bad_field_source)
    assert _single_letter_violations(tree, sample) == [
        "sample.py:4 assigned name 'q'"
    ]
    variable_source = "q = 1\n"
    tree = ast.parse(variable_source)
    assert _single_letter_violations(tree, sample) == [
        "sample.py:1 assigned name 'q'"
    ]
