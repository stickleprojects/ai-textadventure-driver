"""DeterministicParseStrategy — a plain-Python (no LLM call), config-driven
parse strategy. This is the stub to fill in with regex/pattern logic."""
from .base import ParseStrategy


class DeterministicParseStrategy(ParseStrategy):
    """Skeleton, not an implementation — fill in parse() with regex/pattern
    logic, game-config-driven rather than hardcoded (matching how every
    other Knight-Orc-specific pattern in this codebase already lives in
    configs/*.json, loaded via game_config.config, not in Python).

    Starting points already available to reuse rather than reinvent:
    - config.hard_failure_pattern / config.soft_failure_pattern — for
      action_result. These already have the exact phrasings this game
      uses for "that failed" (see configs/knight_orc.json).
    - A new config-driven room-name pattern would go the same place,
      e.g. something matching "You are in/at X." — add it to
      game_config.py/configs/knight_orc.json/schemas/game_config.schema.json
      the same way hard_failure_patterns already works, not as a Python
      constant here.
    - "Exits: X, Y" / "Exits lead X" phrasings for exits.
    - "You can see X" / "you notice X" phrasings for objects.

    Return any subset of the ParseResult fields documented in
    parse_strategies/base.py's module docstring — apply_parse_result()
    treats every field as optional, exactly like every other strategy here.
    """

    def parse(self, response_text, action_taken):
        raise NotImplementedError(
            "Write your parsing logic here — see the class docstring for where "
            "to start. Return a dict using any subset of the ParseResult fields "
            "documented in parse_strategies/base.py."
        )
