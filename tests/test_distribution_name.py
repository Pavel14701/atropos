"""Distribution/import name pins (wave v15, checkpoint V-1).

The distribution is ``atropos-quant``; the IMPORT is ``atropos``
and stays.  The tomllib pin (a) is the MUTATABLE one: it reads
pyproject.toml directly, so reverting the name in pyproject fails
it WITHOUT reinstalling the package -- a version()-based pin alone
would false-survive (importlib.metadata reads the INSTALLED
metadata, not the source).  The version() pin (b) checks the
installed state and is NOT mutated.
"""

from __future__ import annotations

import importlib
import tomllib
from importlib.metadata import version

import pytest

PYPROJECT = "pyproject.toml"


def test_pyproject_name_is_atropos_quant() -> None:
    """Pyproject name is atropos-quant -- the MUTATABLE pin (a).

    Reads the SOURCE file directly (tomllib), so a pyproject revert
    fails this test with no reinstall step.  Mutation fact: reverting
    name to "atropos" fails the equality.
    """
    with open(PYPROJECT, "rb") as handle:
        data = tomllib.load(handle)
    assert data["project"]["name"] == "atropos-quant"


def test_installed_distribution_version() -> None:
    """The INSTALLED distribution is atropos-quant (pin b, not mutated).

    importlib.metadata reads installed metadata -- it validates the
    shipped state, not the source; the runner must uv sync for this
    to reflect a pyproject change.
    """
    assert version("atropos-quant")  # raises PackageNotFoundError if absent


def test_import_name_unchanged() -> None:
    """`import atropos` still works (the promise of Option A)."""
    atropos = importlib.import_module("atropos")
    assert hasattr(atropos, "__version__") or hasattr(atropos, "__all__")


def test_import_atropos_quant_is_import_error() -> None:
    """NEGATIVE PIN: `import atropos_quant` must stay broken (d).

    Guarantees the import rename was not done silently: if anyone
    adds an atropos_quant package later, this test fails and forces
    the decision through a dedicated wave (shim + deprecation cycle).
    """
    with pytest.raises(ImportError):
        importlib.import_module("atropos_quant")


def test_cli_command_is_atropos_quant() -> None:
    """The CLI command is atropos-quant (bioinformatics collision fix).

    Mutation fact: reverting the [project.scripts] key to "atropos"
    fails the equality (reads pyproject directly, no reinstall).
    """
    with open(PYPROJECT, "rb") as handle:
        data = tomllib.load(handle)
    scripts = data["project"].get("scripts", {})
    assert scripts.get("atropos-quant") == "atropos.cli:main"
    assert "atropos" not in scripts
