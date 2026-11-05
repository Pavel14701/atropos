"""Agent-docs contracts for the ML kill chain (regression guards).

The agent context layer (``atropos/agent/``) is the user-facing
documentation surface; these tests pin that the ML chain is actually
documented there: the workflow text mentions the ML engine, and the
ML skill (04) is registered, resolvable and names the public entry
point.  Read-only against the packaged layer: user/project layers are
pinned to nonexistent directories so platform overrides cannot flake
the assertions.
"""

from __future__ import annotations

from pathlib import Path

from atropos.agent import list_skills, read_agent_context, read_skill

_NOWHERE = Path("atropos-no-such-layer")
"""Pinned empty override layers: only the package layer resolves."""


def test_agent_context_mentions_ml_chain() -> None:
    """Resolved AGENTS.md/CLAUDE.md text names the ML kill chain."""
    context = read_agent_context(user_dir=_NOWHERE, project_dir=_NOWHERE)
    assert context is not None, "package agent context must resolve"
    assert "run_ml_kill_protocol" in context, (
        "agent context no longer mentions the ML kill chain "
        "(atropos.ml_training.run_ml_kill_protocol)"
    )


def test_ml_skill_registered() -> None:
    """Skill 04 is listed and its text names the ML engine entry point."""
    names = list_skills(user_dir=_NOWHERE, project_dir=_NOWHERE)
    assert "04-run-ml-kill-protocol" in names
    text = read_skill(
        "04-run-ml-kill-protocol", user_dir=_NOWHERE, project_dir=_NOWHERE
    )
    assert text, "skill 04 resolves empty"
    assert "run_ml_kill_protocol" in text
