"""Meta-test: every ``__main__`` module demo runs clean (rot-proof smoke).

CI used to smoke ONE hand-picked module (``uv run python -m
atropos.ranking`` in the ranking-extra job) -- and a hand-maintained
list rots silently: ranking.py itself shipped NO ``__main__`` block
when that step was added, so the smoke exited 0 while checking
nothing.  This suite regenerates its target list from the source
instead (the same ``rglob`` + source-scan technique as
tests/test_conventions.py): every atropos module declaring an
``if __name__ == "__main__"`` guard must run as
``python -m atropos.<module>`` and exit 0 in a clean interpreter.  A
module that loses its guard, or a new module demo that never runs,
fails here by construction -- no list to forget.

Measured budget (Windows laptop, BLAS threads pinned to 1): the 22
demos take ~122 s total, dominated by ml_training (~59 s) and
demo_pipeline (~17 s) -- over the 60 s suite budget, so the
parametrized run carries the repo's ``slow`` marker; the membership
checks below stay unmarked (they are instant).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAIN_GUARD = 'if __name__ == "__main__"'

# Wall-clock ceiling for ONE module demo, in seconds.  Must clear the
# heaviest HEALTHY demo -- ml_training measured ~59 s on a Windows
# laptop -- with headroom for slower CI hosts; a 20 s ceiling would
# time out a correct run.  This constant is the knob: raise it (or
# skip the file) on a constrained host rather than weakening the
# exit-0 contract.
_PER_MODULE_TIMEOUT_S = 90

# Modules whose ``__main__`` guard is an argparse ENTRY POINT, not a
# bare-invocation demo: ``python -m atropos.cli`` without a
# subcommand is a usage error and exits 2 BY DESIGN (the subparser is
# ``required=True``); its dispatch is exercised by tests/test_cli.py.
# The exit-2 contract itself is behaviorally pinned below
# (test_cli_bare_invocation_is_a_usage_error) — dropping
# ``required=True`` fails that pin instead of silently turning cli
# into a checks-nothing "demo".
_ARGPARSE_ENTRYPOINTS = frozenset({"cli"})

# OpenBLAS reserves its thread pool lazily per process; interpreters
# spawned from a long pytest session can exhaust the allocation arena
# on memory-tight hosts (the environment limitation documented in
# tests/test_ranking.py::test_module_imports_in_clean_interpreter).
# Pin the children to one BLAS thread: the demos are single-threaded
# anyway, and ~20 sequential interpreters must not oversubscribe.
_CHILD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _discovered_main_guard_modules() -> list[str]:
    """Return atropos modules declaring a ``__main__`` guard (dotted).

    A plain substring match, on purpose -- not an AST walk -- exactly
    like the repo's other convention scans: a guard ANYWHERE in the
    source (with or without a trailing pragma comment) makes the
    module a smoke target, so deleting the guard is the single edit
    that removes a module from this suite and trips the membership
    test below.
    """
    package_root = _PROJECT_ROOT / "atropos"
    discovered: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if _MAIN_GUARD not in path.read_text(encoding="utf-8"):
            continue
        parts = [*path.relative_to(package_root).parts[:-1], path.stem]
        discovered.append(".".join(parts))
    return discovered


def _runnable_demo_modules() -> list[str]:
    """Discovered demos minus the argparse entry points (see above)."""
    return [
        module_name
        for module_name in _discovered_main_guard_modules()
        if module_name not in _ARGPARSE_ENTRYPOINTS
    ]


@pytest.mark.slow
@pytest.mark.parametrize("module_name", _runnable_demo_modules())
def test_module_demo_exits_zero(module_name: str) -> None:
    """``python -m atropos.<module>`` runs its demo and exits 0."""
    completed = subprocess.run(
        [sys.executable, "-m", f"atropos.{module_name}"],
        capture_output=True,
        text=True,
        env=_CHILD_ENV,
        timeout=_PER_MODULE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, (
        f"atropos.{module_name} demo exited "
        f"{completed.returncode}; stderr tail:\n"
        f"{(completed.stderr or '')[-2000:]}"
    )


def test_smoke_covers_ranking_and_demo_pipeline() -> None:
    """``ranking`` / ``demo_pipeline`` stay in the discovered set.

    Pins both rot classes behind the old hand-written CI step:
    ranking (the module that step pointed at) losing its freshly
    added ``__main__`` demo, and a golden-path demo such as
    demo_pipeline silently dropping out of the smoke.  Both names are
    asserted on the DISCOVERED set, so a hand-maintained list cannot
    come back through a side door either.
    """
    discovered = set(_discovered_main_guard_modules())
    assert {"ranking", "demo_pipeline"} <= discovered


def test_argparse_exclusion_stays_honest() -> None:
    """cli.py keeps its guard; the single exemption hides nothing.

    ``cli`` is the only module exempt from the bare-invocation run
    (its guard is ``sys.exit(main())``, a usage error without a
    subcommand).  If cli.py ever drops its ``__main__`` block this
    fails, so the exemption can neither outlive its reason nor widen
    unnoticed.
    """
    assert "cli" in set(_discovered_main_guard_modules())


def test_cli_bare_invocation_is_a_usage_error() -> None:
    """Bare ``python -m atropos.cli`` exits 2 — the exemption's teeth.

    Behavioral pin behind the ``_ARGPARSE_ENTRYPOINTS`` exclusion: the
    subparser is ``required=True``, so a bare invocation is a usage
    error (exit 2), NOT a silent success.  If cli.py ever drops the
    ``required=True`` (or the dispatch default changes), bare
    ``python -m atropos.cli`` would exit 0 and this pin fails — the
    exemption would then be hiding a runnable-but-checks-nothing demo,
    the exact rot class the old hand-written CI step had.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "atropos.cli"],
        capture_output=True,
        text=True,
        env=_CHILD_ENV,
        timeout=_PER_MODULE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 2, (
        "bare atropos.cli no longer exits 2 (usage error); rc = "
        f"{completed.returncode}, stderr tail:\n"
        f"{(completed.stderr or '')[-2000:]}"
    )
