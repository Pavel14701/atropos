"""Meta-architecture tests: invariants of the codebase itself.

These catch the systemic patterns found in the 14-module review:
swallowed exceptions, lookahead-prone ``np.roll``, and helper copies
that drift apart. Cheap AST scans instead of 14 code-review sessions.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "atropos"
MODULE_PATHS = sorted(PACKAGE_DIR.glob("*.py"))

# Modules whose run_all_examples already assert their outcomes. After
# the layer-0 wave every module verifies its demo; a new module must be
# added here when it gains a demo.  kill_chain (follow-up from the ML
# unification wave) and workspace (runtime scaffolding) assert too.
ASSERTING_DEMO_MODULES = [
    "bootstrap_ci",
    "cost_model",
    "cross_sectional",
    "ensemble_pipeline",
    "evaluation_metrics",
    "incremental_regression",
    "kill_chain",
    "monte_carlo",
    "null_control_suite",
    "power_calculator",
    "predictive_model_eval",
    "prereg_generator",
    "robustness_battery",
    "track_record",
    "walkforward_framework",
    "workspace",
]

# Helpers that must live ONLY in atropos/_utils.py after the fix wave.
# _align/_frameify were consolidated there as align_inputs/frameify, so
# the old private copies are banned outright: re-adding one in a leaf
# module is a regression, not a new copy under some ceiling.
BANNED_HELPER_NAMES = {
    "_clean",
    "_apply_stat",
    "_block_index_matrix",
    "_align",
    "_frameify",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]


def _is_broad_catch(handler: ast.ExceptHandler) -> bool:
    """Bare ``except:`` or ``except Exception/BaseException``."""
    if handler.type is None:
        return True
    names = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [
            element.id
            for element in handler.type.elts
            if isinstance(element, ast.Name)
        ]
    return any(name in {"Exception", "BaseException"} for name in names)


def test_no_silently_swallowed_exceptions() -> None:
    """Broad ``except: pass`` hides caller-visible bugs (review: x2).

    Narrow typed fallbacks (``except (TypeError, ValueError): pass``
    followed by an explicit second path) are deliberate and allowed;
    only bare or ``Exception``-wide catches that silently discard the
    error are banned.
    """
    offenders = []
    for path in MODULE_PATHS:
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
                and _is_broad_catch(node)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"silent exception swallowing: {offenders}"


def test_no_np_roll_calls() -> None:
    """``np.roll`` wraps future values into the past (review: x4)."""
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in MODULE_PATHS
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "roll"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
    ]
    assert not offenders, f"lookahead-prone np.roll: {offenders}"


def test_removed_helper_copies_stay_removed() -> None:
    """Banned helper copies (_clean, _align, _frameify, ...) stay dead.

    The input-coercion helpers live in ``atropos/_utils.py`` under the
    public names ``align_inputs``/``frameify`` (which do not match the
    ban); the old private per-module copies must not come back.
    """
    offenders = [
        f"{path.name}:{function.name}"
        for path in MODULE_PATHS
        for function in _function_defs(_parse(path))
        if function.name in BANNED_HELPER_NAMES
    ]
    assert not offenders, f"helper copies resurrected: {offenders}"


def test_block_indices_has_a_single_home() -> None:
    """Block-bootstrap indices are generated in exactly one place."""
    homes = [
        path.name
        for path in MODULE_PATHS
        if any(
            function.name == "block_indices"
            for function in _function_defs(_parse(path))
        )
    ]
    assert homes == ["_utils.py"]


def _has_verification(node: ast.AST) -> bool:
    """An assert statement or an explicit verification failure."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if (
            isinstance(sub, ast.Raise)
            and isinstance(sub.exc, ast.Call)
            and isinstance(sub.exc.func, ast.Name)
            and sub.exc.func.id == "AssertionError"
        ):
            return True
    return False


@pytest.mark.parametrize("module_name", ASSERTING_DEMO_MODULES)
def test_demos_assert_their_outcomes(module_name: str) -> None:
    """run_all_examples must fail on regression, not just print.

    Verification counts whether it lives in the demo body, in a private
    demo helper it calls (one level deep), or as an explicit
    ``raise AssertionError``.
    """
    path = PACKAGE_DIR / f"{module_name}.py"
    tree = _parse(path)
    functions = {function.name: function for function in _function_defs(tree)}
    demo = functions.get("run_all_examples")
    assert demo is not None, f"{module_name}: no run_all_examples"
    verified = _has_verification(demo)
    if not verified:
        called = {
            node.func.id
            for node in ast.walk(demo)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
        }
        verified = any(_has_verification(functions[name]) for name in called)
    assert verified, (
        f"{module_name}.run_all_examples prints without asserting"
    )
