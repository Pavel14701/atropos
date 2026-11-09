# CLAUDE.md — notes for Claude (Atropos package layer)

- Read this layer's `AGENTS.md` first: the 5-step workflow and the
  NEVER-list are mandatory before any hypothesis work.
- Positioning: Atropos is a falsification toolkit (backtest kill
  protocol, feature decomposition, paper parity + machine-checkable
  instrument routing) — verdicts are the product, never a strategy
  factory.
- Never guess skills by name — list them: `atropos agent-context
  --list-skills`; read one: `atropos agent-context --skill=NAME`.
- Before writing your own prereg text, call
  `atropos.prereg_generator.generate_prereg(params)` — the format is
  already frozen in code; the params dict is validated by
  `validate_prereg`.
- Always take the verdict from `report["overall_verdict"]`, not from
  your interpretation of intermediate numbers.
- These files are packaged. Personal/project additions go into the
  user/project layer (`atropos agent-setup`); never edit packaged
  files.
- Environment check before work: `atropos agent-doctor`.
