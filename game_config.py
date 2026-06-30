"""Game-specific configuration, loadable from a JSON file at startup.

Default values match Knight Orc (Level 9 Computing, 1987).
Override by calling config.load_from_file(path) before starting the game.

JSON schema (all keys optional — missing keys keep their defaults):
{
    "prompt_pattern":         string  — pexpect pattern that ends every game response
    "candidate_verbs":        [str]   — verbs to attempt on each discovered object (take first)
    "creature_words":         [str]   — words that identify living creatures (case-insensitive)
    "hard_failure_patterns":  [str]   — regex fragments; response matching any means verb is permanently invalid for this object
    "soft_failure_patterns":  [str]   — regex fragments; response matching any means verb is valid but blocked by current state
    "end_state_patterns":     {       — regex fragments keyed by category for run outcome detection
        "death":    [str],
        "finished": [str],
        "score":    [str]
    },
    "navigation": {
        "fast_nav_command":   string  — template for fast navigation (e.g. "run to {target}"); omit to use graph-based step-by-step
        "full_nav_command":   string  — template for full navigation showing intermediate rooms (e.g. "go to {target}"); omit to use graph-based
    },
    "npc_commands": {
        "wait_for_command":   string  — template to wait for a specific NPC (e.g. "wait for {npc}"); omit if game has no such command
    }
}

Legacy keys still accepted: "inspection_sequence" (alias for candidate_verbs),
"failure_patterns" (alias for hard_failure_patterns).
"""
import json
import re


class GameConfig:
    _DEFAULT_END_STATE_PATTERNS = {
        "death": [r"you have died", r"you are dead", r"killed"],
        "finished": [r"congratulations", r"you have finished", r"the end"],
        "score": [r"you score \d+ out of \d+"],
    }

    def __init__(self):
        # Matches both the verbose prompt and the terse ">" prompt that
        # Knight Orc switches to after a few successful commands.
        self.prompt_pattern = r'What now\?|\r?\n>'
        self.candidate_verbs = ["take", "examine", "read", "look inside"]
        self.creature_words = frozenset({
            "horse", "pony", "mare", "stallion",
            "knight", "orc", "guard", "soldier",
            "man", "woman", "person", "peasant",
            "troll", "goblin", "dwarf", "elf",
            "creature", "beast", "monster", "demon",
        })
        self._hard_failure_patterns = [
            r"you can'?t",
            r"can'?t see",
            r"can'?t do that",
            r"don'?t understand",
            r"you don'?t have",
            r"nothing happens",
            r"that'?s not something",
            r"there('?s| is) no \w+ here",
            r"i don'?t know (that word|what)",
            r"don'?t need to use the word",
        ]
        self._soft_failure_patterns = [
            r"right now",
            r"not now",
            r"(you'?re|you are) already (wearing|carrying|holding)",
            r"while (you'?re|you are) (wearing|carrying|holding)",
            r"can'?t do that yet",
            r"not yet",
        ]
        self.end_state_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in patterns]
            for k, patterns in self._DEFAULT_END_STATE_PATTERNS.items()
        }
        self.fast_nav_command = None   # e.g. "run to {target}"
        self.full_nav_command = None   # e.g. "go to {target}"
        self.wait_for_command = None   # e.g. "wait for {npc}"
        self._compile()

    def _compile(self):
        self.hard_failure_pattern = re.compile(
            "|".join(self._hard_failure_patterns), re.IGNORECASE)
        self.soft_failure_pattern = re.compile(
            "|".join(self._soft_failure_patterns), re.IGNORECASE)
        # Union for backward compat — matches either hard or soft failure
        self.failure_pattern = re.compile(
            "|".join(self._hard_failure_patterns + self._soft_failure_patterns),
            re.IGNORECASE,
        )

    @property
    def inspection_sequence(self):
        """Backward-compat alias for candidate_verbs."""
        return self.candidate_verbs

    def load_from_file(self, path):
        """Load overrides from a JSON config file."""
        with open(path) as f:
            data = json.load(f)
        if "prompt_pattern" in data:
            self.prompt_pattern = data["prompt_pattern"]
        if "candidate_verbs" in data:
            self.candidate_verbs = list(data["candidate_verbs"])
        elif "inspection_sequence" in data:
            self.candidate_verbs = list(data["inspection_sequence"])
        if "creature_words" in data:
            self.creature_words = frozenset(data["creature_words"])
        recompile = False
        if "hard_failure_patterns" in data:
            self._hard_failure_patterns = list(data["hard_failure_patterns"])
            recompile = True
        elif "failure_patterns" in data:
            self._hard_failure_patterns = list(data["failure_patterns"])
            recompile = True
        if "soft_failure_patterns" in data:
            self._soft_failure_patterns = list(data["soft_failure_patterns"])
            recompile = True
        if recompile:
            self._compile()
        if "end_state_patterns" in data:
            self.end_state_patterns = {
                k: [re.compile(p, re.IGNORECASE) for p in patterns]
                for k, patterns in data["end_state_patterns"].items()
            }
        if "navigation" in data:
            nav = data["navigation"]
            if "fast_nav_command" in nav:
                self.fast_nav_command = nav["fast_nav_command"]
            if "full_nav_command" in nav:
                self.full_nav_command = nav["full_nav_command"]
        if "npc_commands" in data:
            npc = data["npc_commands"]
            if "wait_for_command" in npc:
                self.wait_for_command = npc["wait_for_command"]


config = GameConfig()
