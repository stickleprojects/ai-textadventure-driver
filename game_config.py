"""Game-specific configuration, loadable from a JSON file at startup.

Default values match Knight Orc (Level 9 Computing, 1987).
Override by calling config.load_from_file(path) before starting the game.

JSON schema (all keys optional — missing keys keep their defaults):
{
    "prompt_pattern":      string  — pexpect pattern that ends every game response
    "inspection_sequence": [str]   — action verbs applied in order to discovered objects
    "creature_words":      [str]   — words that identify living creatures (case-insensitive)
    "failure_patterns":    [str]   — regex fragments; a response matching any is a refusal
}
"""
import json
import re


class GameConfig:
    def __init__(self):
        self.prompt_pattern = r'What now\?'
        self.inspection_sequence = ["take", "examine", "read", "look inside"]
        self.creature_words = frozenset({
            "horse", "pony", "mare", "stallion",
            "knight", "orc", "guard", "soldier",
            "man", "woman", "person", "peasant",
            "troll", "goblin", "dwarf", "elf",
            "creature", "beast", "monster", "demon",
        })
        self._failure_patterns = [
            r"you can'?t",
            r"can'?t see",
            r"can'?t do that",
            r"don'?t understand",
            r"you don'?t have",
            r"nothing happens",
            r"that'?s not something",
            r"there('?s| is) no \w+ here",
            r"i don'?t know (that word|what)",
        ]
        self._compile()

    def _compile(self):
        self.failure_pattern = re.compile(
            "|".join(self._failure_patterns),
            re.IGNORECASE,
        )

    def load_from_file(self, path):
        """Load overrides from a JSON config file."""
        with open(path) as f:
            data = json.load(f)
        if "prompt_pattern" in data:
            self.prompt_pattern = data["prompt_pattern"]
        if "inspection_sequence" in data:
            self.inspection_sequence = list(data["inspection_sequence"])
        if "creature_words" in data:
            self.creature_words = frozenset(data["creature_words"])
        if "failure_patterns" in data:
            self._failure_patterns = list(data["failure_patterns"])
            self._compile()


config = GameConfig()
