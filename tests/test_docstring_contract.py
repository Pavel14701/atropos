"""Docstring contracts for every ``atropos/*.py`` module.

Six checks:

* every module's doctest examples (the ones not marked
  ``# doctest: +SKIP``) actually run and pass;
* a documented ``Raises: ValueError`` section is backed by a real
  ``raise ...Error`` in the function body or in a module-level helper
  it directly or transitively calls;
* Google-style ``Args:`` sections match ``inspect.signature`` exactly;
* ``# doctest: +SKIP`` is not used on deterministic functions whose
  docstrings never mention randomness (allowlist documents the debt);
* every module-level function exported via ``__all__`` carries a
  ``>>>`` example in its docstring (snapshot
  ``PUBLIC_FNS_WITHOUT_EXAMPLES`` documents the current debt);
* the ``run_all_examples`` demo runners are never exported through
  any ``__all__``.
"""

from __future__ import annotations

import ast
import doctest
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import atropos
import pytest

MODULE_NAMES: tuple[str, ...] = tuple(
    sorted(name for _, name, _ in pkgutil.iter_modules(atropos.__path__))
)


# ---------------------------------------------------------------------------
# 1. executable doctests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module_name",
    MODULE_NAMES,
)
def test_module_doctests_pass(module_name: str) -> None:
    """doctest.testmod must report zero failed examples."""
    module = importlib.import_module(f"atropos.{module_name}")
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"atropos.{module_name}: {results.failed} of {results.attempted}"
        " doctest examples failed"
    )


# ---------------------------------------------------------------------------
# 2. documented "Raises ValueError" <=> a real raise exists
# ---------------------------------------------------------------------------
_RAISES_HEADER = re.compile(r"^\s*Raises\s*:\s*$", re.M)
_INLINE_RAISES = re.compile(r"[Rr]aises\s+``?ValueError")


def _documents_value_error(docstring: str) -> bool:
    """True when the docstring promises a ValueError to callers."""
    if _INLINE_RAISES.search(docstring):
        return True
    header = _RAISES_HEADER.search(docstring)
    if header is None:
        return False
    tail = docstring[header.end() :]
    section = tail.split("\n\n", 1)[0]
    return "ValueError" in section


def _raises_value_error_directly(statement: ast.Raise) -> bool:
    """True when the raise statement names ValueError."""
    exception = statement.exc
    if isinstance(exception, ast.Call):
        exception = exception.func
    return isinstance(exception, ast.Name) and exception.id == "ValueError"


def _value_error_raisers(
    tree: ast.Module, extra_raisers: frozenset[str] = frozenset()
) -> set[str]:
    """Module-level functions that can raise ValueError (transitively).

    A function qualifies when its body raises ValueError directly or it
    calls, by bare name, another module-level function that qualifies
    (validator/helper delegation), computed to a fixpoint.
    ``extra_raisers`` seeds the fixpoint with bare names that raise in
    ANOTHER module (the shared ``atropos._utils`` validators).
    """
    direct: dict[str, bool] = {}
    called: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        raises = False
        callees: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise) and _raises_value_error_directly(
                sub
            ):
                raises = True
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                callees.add(sub.func.id)
        direct[node.name] = raises
        called[node.name] = callees

    raisers = {name for name, raises in direct.items() if raises}
    raisers |= set(extra_raisers)
    changed = True
    while changed:
        changed = False
        for name, raises in direct.items():
            if raises or name in raisers:
                continue
            if called[name] & raisers:
                raisers.add(name)
                changed = True
    return raisers


# Shared cross-module validators: leaf modules import ``validate_int``
# and ``validate_number`` from ``atropos._utils`` by bare name, so the
# fixpoint above (which is per-module) cannot follow the call. They do
# raise ValueError in their own module, which keeps the contract honest.
_SHARED_RAISING_VALIDATORS = frozenset({"validate_int", "validate_number"})


def test_documented_valueerror_is_actually_raised() -> None:
    """Every 'Raises: ValueError' promise has a raise behind it."""
    utils_module = importlib.import_module("atropos._utils")
    assert utils_module.__file__ is not None
    utils_tree = ast.parse(
        Path(utils_module.__file__).read_text(encoding="utf-8")
    )
    shared_validators = _SHARED_RAISING_VALIDATORS & _value_error_raisers(
        utils_tree
    )
    # guard: renamed or no-longer-raising helpers must not linger here
    assert shared_validators == _SHARED_RAISING_VALIDATORS
    offenders: list[str] = []
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"atropos.{module_name}")
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        raisers = _value_error_raisers(tree, shared_validators)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            docstring = ast.get_docstring(node) or ""
            if _documents_value_error(docstring) and (
                node.name not in raisers
            ):
                offenders.append(f"{module_name}.{node.name}")
    assert not offenders, (
        "functions documenting ValueError without any raise in their"
        " body or called validators: "
        f"{sorted(offenders)}"
    )


# ---------------------------------------------------------------------------
# 3. Google-style Args: <=> signature
# ---------------------------------------------------------------------------
_ARGS_HEADER = re.compile(r"^\s*Args\s*:\s*$", re.M)
_ARG_LINE = re.compile(r"^\s+\*{0,2}([A-Za-z_]\w*)\s*(\([^)]*\))?\s*:")
_SECTION_STOP = re.compile(
    r"^\s{0,8}("
    r"Returns|Raises|Yields|Example|Examples|Note|Notes|Warning|"
    r"Warnings|See Also|References|Todo|Deprecated"
    r")\s*:"
)


def _documented_args(docstring: str) -> list[str]:
    """Parameter names listed under the Google-style ``Args:`` header."""
    header = _ARGS_HEADER.search(docstring)
    if header is None:
        return []
    names: list[str] = []
    for line in docstring[header.end() :].splitlines():
        if _SECTION_STOP.match(line) or (
            line.strip() and not line[0].isspace()
        ):
            break
        match = _ARG_LINE.match(line)
        if match:
            names.append(match.group(1))
    return names


ARGS_SIGNATURE_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("bootstrap_ci", "percentile_ci"),
    ("cost_model", "apply_proportional_costs"),
    ("cost_model", "cost_death_test"),
    ("cost_model", "breakeven_cost_bps"),
    ("cross_sectional", "cross_sectional_ic"),
    ("cross_sectional", "run_cross_sectional_tests"),
    ("track_record", "prereg_divergence"),
    ("track_record", "live_verdict"),
    ("predictive_model_eval", "deflated_performance"),
    ("predictive_model_eval", "knn_model_fn"),
)


@pytest.mark.parametrize(
    ("module_name", "function_name"), ARGS_SIGNATURE_FUNCTIONS
)
def test_documented_args_match_signature(
    module_name: str, function_name: str
) -> None:
    """Args entries and inspect.signature describe the same parameters."""
    module = importlib.import_module(f"atropos.{module_name}")
    function = getattr(module, function_name)
    docstring = inspect.getdoc(function) or ""
    documented = _documented_args(docstring)
    assert documented, f"{module_name}.{function_name} has no Args section"
    signature = list(inspect.signature(function).parameters)
    assert sorted(documented) == sorted(signature), (
        f"{module_name}.{function_name}: documented={sorted(documented)}"
        f" signature={sorted(signature)}"
    )


# ---------------------------------------------------------------------------
# 4. no '# doctest: +SKIP' on deterministic functions
# ---------------------------------------------------------------------------
_RANDOM_MENTION = re.compile(
    r"(?i)\b("
    r"random|rng|seed|noisy|noise|stochastic|shuffle|shuffled|"
    r"permutation|bootstrap|resample|monte[ ]carlo|simulat\w*|draws?"
    r")\b"
)

SKIPPED_BUT_DETERMINISTIC: dict[str, frozenset[str]] = {
    # Documented debt: these docstrings carry '# doctest: +SKIP' (or the
    # inert '# +SKIP') although the function is deterministic and the
    # docstring never mentions randomness. Most examples also reference
    # variables the snippet never defines (run_battery table, df, f1...)
    # or use '...' ellipsis without the ELLIPSIS flag, so simply removing
    # the marker would break doctest. Kept as an explicit allowlist so
    # NEW skips on pure functions fail this test.
    "cross_sectional": frozenset(
        {
            "_align_panels",
            "_date_quantiles",
            "_per_date_ic",
            "_scaled_ratio",
            "run_cross_sectional_tests",
        }
    ),
    "evaluation_metrics": frozenset({"equity_from_returns", "max_drawdown"}),
    "incremental_regression": frozenset({"curve_summary"}),
    "monte_carlo": frozenset({"minimum_detectable_effect"}),
    "power_calculator": frozenset(
        {
            "power_for_effect",
            "detectable_effect",
            "t_stat_for_partial_r2",
        }
    ),
    "predictive_model_eval": frozenset({"ols_model_fn"}),
    "prereg_generator": frozenset({"validate_prereg", "generate_prereg"}),
    "robustness_battery": frozenset({"battery_overall"}),
    "track_record": frozenset({"prereg_divergence", "drawdown_floor_check"}),
    "walkforward_framework": frozenset(
        {"frozen_threshold", "check_no_peeking"}
    ),
}


def _skipped_pure_functions(tree: ast.Module) -> set[str]:
    """Module-level functions with a +SKIP example and no random prose."""
    pure_skipped: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        docstring = ast.get_docstring(node) or ""
        if "+SKIP" not in docstring:
            continue
        if _RANDOM_MENTION.search(docstring):
            continue
        pure_skipped.add(node.name)
    return pure_skipped


def test_no_doctest_skip_on_deterministic_functions() -> None:
    """'+SKIP' is reserved for genuinely random outputs."""
    violations: list[str] = []
    stale: list[str] = []
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"atropos.{module_name}")
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        actual = _skipped_pure_functions(tree)
        allowed = set(SKIPPED_BUT_DETERMINISTIC.get(module_name, frozenset()))
        violations.extend(
            f"{module_name}.{name}" for name in sorted(actual - allowed)
        )
        stale.extend(
            f"{module_name}.{name}" for name in sorted(allowed - actual)
        )
    assert not violations, (
        "deterministic functions newly carrying '+SKIP' (either make the"
        " example runnable or justify randomness in the docstring): "
        f"{violations}"
    )
    assert not stale, (
        "allowlist entries that no longer skip a pure example (remove"
        f" them): {stale}"
    )


def test_allowlist_covers_only_real_modules() -> None:
    """The allowlist must not mention modules that do not exist."""
    unknown = sorted(set(SKIPPED_BUT_DETERMINISTIC) - set(MODULE_NAMES))
    assert not unknown, f"unknown allowlist modules: {unknown}"


# ---------------------------------------------------------------------------
# 5. every __all__-exported public function has a '>>>' example
# ---------------------------------------------------------------------------
PUBLIC_FNS_WITHOUT_EXAMPLES: dict[str, frozenset[str]] = {
    # Documented debt (snapshot): public functions exported through
    # __all__ whose docstring carries no '>>>' example yet.  The ratchet
    # cuts both ways - a NEW public function without an example fails
    # the test, and an entry whose function has gained an example must
    # be pruned (the debt may only shrink).  Non-function exports
    # (result classes, DEFAULT_CONFIG, REQUIRED_FIELDS, ...) are out of
    # scope: only module-level `def`s listed in __all__ are checked.
    "_utils": frozenset(
        {
            "align_inputs",
            "apply_stat_rowwise",
            "block_indices",
            "clean_1d",
            "frameify",
            "ols_hc0",
            "sane_variance_mask",
        }
    ),
    "ensemble_pipeline": frozenset({"run_full_pipeline"}),
}


def _module_level_function_docs(tree: ast.Module) -> dict[str, str]:
    """Map module-level function name -> docstring (``''`` if absent)."""
    docs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docs[node.name] = ast.get_docstring(node) or ""
    return docs


def _exported_names(module: ModuleType) -> frozenset[str]:
    """Names the module lists in ``__all__`` (empty when undefined)."""
    exported = getattr(module, "__all__", ())
    return frozenset(str(name) for name in exported)


def test_public_functions_have_examples() -> None:
    """Every ``__all__``-exported module-level function has ``>>>``.

    Ratchet on the debt snapshot
    ``PUBLIC_FNS_WITHOUT_EXAMPLES``: both directions are enforced -
    a new public function without a docstring example fails (write
    the example rather than growing the snapshot), and a snapshot
    entry whose function has since gained an example fails as stale
    (prune it: the debt may only shrink).
    """
    unknown_modules = sorted(
        set(PUBLIC_FNS_WITHOUT_EXAMPLES) - set(MODULE_NAMES)
    )
    assert not unknown_modules, f"unknown snapshot modules: {unknown_modules}"
    violations: list[str] = []
    stale: list[str] = []
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"atropos.{module_name}")
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        docs = _module_level_function_docs(tree)
        exported = _exported_names(module)
        actual = {
            name
            for name, docstring in docs.items()
            if name in exported and ">>>" not in docstring
        }
        allowed = set(PUBLIC_FNS_WITHOUT_EXAMPLES.get(module_name, ()))
        violations.extend(
            f"{module_name}.{name}" for name in sorted(actual - allowed)
        )
        stale.extend(
            f"{module_name}.{name}" for name in sorted(allowed - actual)
        )
    assert not violations, (
        "public functions exported via __all__ without a '>>>' example "
        "(write the example; grow the snapshot only for a genuinely "
        f"undemoable function): {violations}"
    )
    assert not stale, (
        "snapshot entries whose function now HAS an example (prune them "
        f"- the doctest debt may only shrink): {stale}"
    )


def test_demo_runners_are_not_exported() -> None:
    """``run_all_examples`` demo runners stay out of every ``__all__``.

    They are ``python -m atropos.<module>`` entry points, not public
    API; re-exporting one would put a printer on the import surface.
    """
    offenders: list[str] = []
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"atropos.{module_name}")
        if "run_all_examples" in _exported_names(module):
            offenders.append(module_name)
    assert not offenders, (
        "run_all_examples is a __main__ demo, not public API: remove it "
        f"from __all__ in {offenders}"
    )


# ---------------------------------------------------------------------------
# helper export sanity (silences unused-import confusion for readers)
# ---------------------------------------------------------------------------
def _docstring_of(module_name: str, function_name: str) -> Any:
    """Fetch a function object by module and name (test helper)."""
    module = importlib.import_module(f"atropos.{module_name}")
    return getattr(module, function_name)


def test_inline_raises_pattern_finds_known_case() -> None:
    """The inline 'Raises ``ValueError``' prose form is detected too."""
    function = _docstring_of("prereg_generator", "generate_prereg")
    docstring = inspect.getdoc(function) or ""
    assert _documents_value_error(docstring)
