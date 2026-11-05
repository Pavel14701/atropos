"""Runtime workspace scaffolding: generate the user hypothesis files.

The framework never assumes ``GRAVEYARD.md`` or ``hypotheses/INDEX.csv``
exist on the user's disk: :func:`init_workspace` materialises the
skeleton on demand (``atropos init``).  Generation is strictly additive
- an existing file is reported in ``"skipped"`` and left byte-for-byte
untouched, so the command is safe to re-run at any time (idempotent)
and there is deliberately no force option: edit the generated files
manually after init.

The skeletons mirror ``METHODOLOGY.md`` / ``TEMPLATES.md`` /
``CASES_AND_FLAGS.md``: an append-only graveyard carrying the
cause-of-death (COD) legend, a one-row-per-hypothesis registry and a
``hypotheses/`` README pointing at the ``PREREG_<name>_<ISO-date>.md``
convention.

Run ``python -m atropos.workspace`` for a verified demo.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_WORKSPACE_FILES", "init_workspace"]

_GRAVEYARD_TEMPLATE = """\
# Graveyard — append only

Every finished hypothesis gets a dated entry below: KILLED, SURVIVED
or INCONCLUSIVE — a verdict is mandatory; an open hypothesis without
one accumulates duplicate testing. Append new entries at the END of
this file; never edit or delete existing ones (an error in an entry is
fixed by a NEW entry that references it).

Full per-hypothesis tombstones (prediction, kill criterion,
resurrection conditions) live in `hypotheses/graveyard/H-NNNN.md` and
are registered in `hypotheses/INDEX.csv` — see
`docs/how-to/03-read-graveyard.md`.

## Entry format

    ### <ISO-date> — <hypothesis name>
    verdict: KILLED | SURVIVED | INCONCLUSIVE; cod: <COD tag>
    <one-line cause of death with the deciding numbers>

## COD legend (cause-of-death tags)

| COD tag     | Meaning                                          |
|-------------|--------------------------------------------------|
| REPACKAGING | echo of a known factor (incremental t < 2)       |
| ARTIFACT    | thresholds fitted after the data                 |
| NO_OOS_EDGE | raw effect gone on fresh out-of-sample days      |
| COST_DEATH  | gross < cost line or > 20 RT/year turnover       |
| RISK        | drawdown/vol outside the prereg risk frame       |
| CS_NO_EDGE  | per-date cross-sectional IC consistent with zero |
| DEAD        | pipeline kill on any other frozen gate           |

## Entries (append below — newest last)
"""

_INDEX_CSV_HEADER = "id,name,date,family,cod,verdict,file\n"

_HYPOTHESES_README_TEMPLATE = """\
# hypotheses/

One hypothesis = one prereg file `PREREG_<name>_<ISO-date>.md`,
committed BEFORE any data is loaded (the commit hash freezes the
protocol). Every hypothesis ends with a verdict row in `INDEX.csv`
and an entry in the append-only root `GRAVEYARD.md`. Regenerate
missing skeleton files any time with `atropos init` — existing files
are never overwritten; edit them manually after init.
"""

#: Skeleton files created by :func:`init_workspace` (relative POSIX
#: paths -> file content, in creation order).
DEFAULT_WORKSPACE_FILES: dict[Path, str] = {
    Path("GRAVEYARD.md"): _GRAVEYARD_TEMPLATE,
    Path("hypotheses/INDEX.csv"): _INDEX_CSV_HEADER,
    Path("hypotheses/README.md"): _HYPOTHESES_README_TEMPLATE,
}


def init_workspace(root: Path = Path(".")) -> dict[str, Any]:
    """Create the workspace skeleton under ``root``; never overwrite.

    For every file of :data:`DEFAULT_WORKSPACE_FILES`: an existing file
    is reported in ``"skipped"`` and left untouched, a missing one is
    created (parents included) and reported in ``"created"``.  The
    operation is idempotent: a second run on the same root creates
    nothing.

    Args:
        root: workspace root directory (created when missing).

    Returns:
        Report dict with ``"created"`` / ``"skipped"`` (relative
        POSIX-style paths in :data:`DEFAULT_WORKSPACE_FILES` order)
        and ``"root"`` (``str(root)``).

    Raises:
        OSError: propagated from directory/file creation (``root``
            exists as a file, or the tree is not writable).

    Example::

        >>> import shutil
        >>> import tempfile
        >>> from pathlib import Path
        >>> base = Path(tempfile.mkdtemp(prefix="atropos_ws_doctest_"))
        >>> first = init_workspace(base / "fresh")
        >>> first["created"]
        ['GRAVEYARD.md', 'hypotheses/INDEX.csv', 'hypotheses/README.md']
        >>> first["skipped"]
        []
        >>> second = init_workspace(base / "fresh")
        >>> second["created"]
        []
        >>> len(second["skipped"])
        3
        >>> shutil.rmtree(base, ignore_errors=True)
    """
    created: list[str] = []
    skipped: list[str] = []
    for rel_path, content in DEFAULT_WORKSPACE_FILES.items():
        target = root / rel_path
        if target.exists():
            skipped.append(rel_path.as_posix())
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created.append(rel_path.as_posix())
    return {"created": created, "skipped": skipped, "root": str(root)}


def run_all_examples() -> None:
    """Demo: scaffold a throwaway workspace and verify the skeleton.

    Doubles as the layer-0 verification: a fresh root gets every
    default file, the second call skips everything (idempotency), the
    graveyard carries the append-only banner and the full COD legend,
    the INDEX header matches the documented registry columns, and a
    user edit survives a re-init byte-for-byte.
    """
    base = Path(tempfile.mkdtemp(prefix="atropos_ws_demo_"))
    try:
        root = base / "ws"
        result = init_workspace(root)
        assert result["created"] == [
            "GRAVEYARD.md",
            "hypotheses/INDEX.csv",
            "hypotheses/README.md",
        ]
        assert result["skipped"] == []
        graveyard_path = root / "GRAVEYARD.md"
        graveyard_text = graveyard_path.read_text(encoding="utf-8")
        assert "Graveyard — append only" in graveyard_text
        assert "H-NNNN" in graveyard_text
        for cod_tag in (
            "REPACKAGING",
            "ARTIFACT",
            "NO_OOS_EDGE",
            "COST_DEATH",
            "RISK",
            "CS_NO_EDGE",
            "DEAD",
        ):
            assert cod_tag in graveyard_text
        index_text = (root / "hypotheses" / "INDEX.csv").read_text(
            encoding="utf-8"
        )
        assert index_text == "id,name,date,family,cod,verdict,file\n"
        readme_text = (root / "hypotheses" / "README.md").read_text(
            encoding="utf-8"
        )
        assert "PREREG_<name>_<ISO-date>.md" in readme_text
        # idempotent: the second call creates nothing, skips everything
        again = init_workspace(root)
        assert again["created"] == []
        assert again["skipped"] == [
            "GRAVEYARD.md",
            "hypotheses/INDEX.csv",
            "hypotheses/README.md",
        ]
        # user edits survive: init never overwrites an existing file
        graveyard_path.write_text("user edit\n", encoding="utf-8")
        third = init_workspace(root)
        assert graveyard_path.read_text(encoding="utf-8") == "user edit\n"
        assert "GRAVEYARD.md" in third["skipped"]
        print(f"workspace demo: skeleton verified under {root}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    run_all_examples()
