# CLAUDE.md — заметки для Claude (пакетный слой Atropos)

- Читай `AGENTS.md` этого слоя первым: рабочий цикл из 5 шагов и
  NEVER-список обязательны до любой работы с гипотезами.
- Скиллы не угадывай по имени — список: `atropos agent-context
  --list-skills`; текст: `atropos agent-context --skill=NAME`.
- Прежде чем писать свой текст пререга, вызывай
  `atropos.prereg_generator.generate_prereg(params)` — формат уже
  заморожен в коде; params-dict валидируется `validate_prereg`.
- Вердикт всегда берётся из `report["overall_verdict"]`, не из твоей
  интерпретации промежуточных чисел.
- Эти файлы — пакетные. Личные/проектные дополнения клади в user/project
  слой (`atropos agent-setup`), packaged-файлы не редактируй.
- Проверка окружения перед работой: `atropos agent-doctor`.
