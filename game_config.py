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
    "scenery_patterns":       [str]   — regex fragments; response to a failed "take" matching any means the
                                         object is non-interactive scenery with nothing further to reveal, so
                                         the remaining queued inspection verbs (examine, read, ...) are skipped.
                                         A failed take that does NOT match this still runs the full inspection
                                         sequence — e.g. "too heavy" or "fixed in place" objects can still
                                         reveal sub-objects when examined (see docs/bugs/70.md).
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
    },
    "theft_patterns":        [str]   — regex fragments matching "<npc> stole/takes <item> from <anyone>"
                                       narration; each must have a named group (?P<item>...).
                                       Matched item is removed from inventory only if it's
                                       currently held — deliberately does not try to parse who
                                       the "victim" is (the game may narrate the player in
                                       third person, e.g. by creature name, not just "you"),
                                       so add new phrasings here rather than trying to also
                                       identify the victim from text.
    "room_patterns":          [str]  — regex fragments identifying a room-description sentence;
                                        each must have a named group (?P<room>...). Used by
                                        parse_strategies.DeterministicParseStrategy (no LLM) —
                                        the captured text is passed through as-is (room/exit
                                        canonicalization already happens downstream in
                                        world_graph.resolve_room_name), so a pattern can over-capture
                                        trailing description without needing to trim it itself.
    "exit_clause_patterns":   [str]  — regex fragments identifying the exits sentence; each must
                                        have a named group (?P<exits>...) capturing the raw
                                        comma/and-separated list text (further split and filtered
                                        against known direction words downstream — deliberately
                                        does NOT expand "in all directions" into every compass
                                        point: that's a hallucination, not a grounded read, the
                                        same reasoning as bug 76's LLM-side fix).
    "visible_entity_patterns": [str] — regex fragments identifying a "you can see X" style
                                        sentence; each must have a named group (?P<entities>...)
                                        capturing the raw comma/and-separated list text.
    "take_confirmation_patterns": [str] — regex fragments; a response to a "take" action matching
                                        any of these confirms the item was actually taken (as
                                        opposed to merely not matching a failure pattern) — the
                                        item name itself comes from the action text, not the
                                        response (see docs/agent_behavior_spec.md's "Handling
                                        uncertainty": don't guess a take succeeded just because
                                        nothing matched a known failure phrase).
    "hints": [str]  — user-authored strategy hints injected into the LLM prompt as additional context
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
            # Knight-Orc-specific NPC roles narrated as "the X" rather than a
            # proper name — found by scanning saved run logs for "the <word>
            # <action verb>" narration and "You can see the X" phrasings.
            "gripper", "hermit", "innkeeper", "prophet", "valkyrie",
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
        self._scenery_patterns = [
            r"probably just scenery",
        ]
        # Deliberately does not try to capture/parse who the theft's "victim" is —
        # the game can narrate the player in third person (e.g. by creature name),
        # not just "you", so it's unreliable to key off that text. Instead any
        # matched item is removed from inventory only if currently held.
        self._theft_patterns = [
            r"(?:just\s+)?stole\s+(?:the\s+)?(?P<item>.+?)\s+from\s+",
            r"takes?\s+(?:the\s+)?(?P<item>.+?)\s+from\s+",
        ]
        self._room_patterns = [
            # "in"/"on"/"at"/"beside" are grammatical connectors, dropped from
            # the captured room text (agent._LEADING_PREP_RE strips them too,
            # so either would work — dropped here to match room_patterns'
            # captured text to the LLM path's own convention). "outside"/
            # "inside" are NOT connectors here — Knight Orc treats "outside
            # a cave" and "inside a cave" as distinct rooms (see
            # agent.py's _LEADING_PREP_RE comment), so the second pattern
            # keeps that word as part of the captured room text.
            r"you (?:go \w+ and )?are (?:in|on|at|beside) (?P<room>.+?)"
            r"(?=\s*(?:exits? leads?|an exit leads|you can see|in the distance is)|$)",
            r"you (?:go \w+ and )?are (?P<room>(?:outside|inside) .+?)"
            r"(?=\s*(?:exits? leads?|an exit leads|you can see|in the distance is)|$)",
        ]
        self._exit_clause_patterns = [
            r"(?:exits? leads?|an exit leads) (?P<exits>[^.]+)\.",
        ]
        self._visible_entity_patterns = [
            r"you can see (?P<entities>[^.]+)\.",
        ]
        self._take_confirmation_patterns = [
            r"you take\b",
            r"^taken\.?$",
            r"^ok\.?$",
        ]
        self.end_state_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in patterns]
            for k, patterns in self._DEFAULT_END_STATE_PATTERNS.items()
        }
        self.fast_nav_command = None   # e.g. "run to {target}"
        self.full_nav_command = None   # e.g. "go to {target}"
        self.wait_for_command = None   # e.g. "wait for {npc}"
        self.hints = []
        self._compile()

    def _compile(self):
        self.hard_failure_pattern = re.compile(
            "|".join(self._hard_failure_patterns), re.IGNORECASE)
        self.soft_failure_pattern = re.compile(
            "|".join(self._soft_failure_patterns), re.IGNORECASE)
        self.scenery_pattern = re.compile(
            "|".join(self._scenery_patterns), re.IGNORECASE)
        # Union for backward compat — matches either hard or soft failure
        self.failure_pattern = re.compile(
            "|".join(self._hard_failure_patterns + self._soft_failure_patterns),
            re.IGNORECASE,
        )
        death_pats = self._DEFAULT_END_STATE_PATTERNS["death"]
        self.death_pattern = re.compile("|".join(death_pats), re.IGNORECASE)
        # Each pattern compiled separately (not OR-joined) since named groups
        # can't repeat within a single compiled pattern.
        self.theft_patterns = [re.compile(p, re.IGNORECASE) for p in self._theft_patterns]
        self.room_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self._room_patterns
        ]
        self.exit_clause_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self._exit_clause_patterns
        ]
        self.visible_entity_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self._visible_entity_patterns
        ]
        self.take_confirmation_pattern = re.compile(
            "|".join(self._take_confirmation_patterns), re.IGNORECASE | re.MULTILINE)

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
        if "scenery_patterns" in data:
            self._scenery_patterns = list(data["scenery_patterns"])
            recompile = True
        if "theft_patterns" in data:
            self._theft_patterns = list(data["theft_patterns"])
            recompile = True
        if "room_patterns" in data:
            self._room_patterns = list(data["room_patterns"])
            recompile = True
        if "exit_clause_patterns" in data:
            self._exit_clause_patterns = list(data["exit_clause_patterns"])
            recompile = True
        if "visible_entity_patterns" in data:
            self._visible_entity_patterns = list(data["visible_entity_patterns"])
            recompile = True
        if "take_confirmation_patterns" in data:
            self._take_confirmation_patterns = list(data["take_confirmation_patterns"])
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
        if "hints" in data:
            self.hints = list(data["hints"])


config = GameConfig()
