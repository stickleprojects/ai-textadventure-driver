"""DeterministicParseStrategy — a plain-Python (no LLM call), config-driven
parse strategy. Turns raw Knight Orc game text into a ParseResult using
regex patterns from game_config.config (configs/knight_orc.json), not an
LLM call — see game_config.py's docstring for what each pattern key means.

Patterns were derived from real saved run logs (logs/*.json), not guessed:
room/exit/visible-entity phrasing, the "the X"-narrated NPC roles now in
creature_words (gripper, hermit, innkeeper, prophet, valkyrie), and the
narrated direction words "downwards"/"upwards" (now in
world_graph.DIRECTION_NORMALIZE) all came from scanning actual game output
rather than assumed conventions.
"""
import re

from game_config import config
import response_classification
import world_graph

from .base import ParseStrategy, split_verb_object

_LIST_SPLIT_RE = re.compile(r",\s*|\s+and\s+")
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+(.*)$", re.IGNORECASE)
_ALL_DIRECTIONS_PREFIX_RE = re.compile(
    r"^in all directions?\s*(?:and\s*)?|^in every direction\s*(?:and\s*)?", re.IGNORECASE
)


class DeterministicParseStrategy(ParseStrategy):
    """Config-driven regex parser — no LLM call, game-specific phrasing lives
    in configs/*.json (game_config.room_patterns / exit_clause_patterns /
    visible_entity_patterns / take_confirmation_patterns), not hardcoded here.

    Deliberately does NOT implement: gifts/theft narration (received_from_npc/
    taken_by_npc — theft is already handled unconditionally by
    apply_parse_result's own config.theft_patterns cross-check, regardless of
    strategy), learned_spells, anomalies/blocked_by, or notable_events. These
    require either narrative judgment a regex can't reliably make, or (theft)
    are already covered elsewhere — see this module's docstring in a future
    revision if patterns for them are added.
    """

    def parse(self, response_text, action_taken):
        verb, obj = split_verb_object(action_taken) if action_taken else (None, None)
        action_result = self._classify_action_result(response_text, verb)

        result = {
            "action_result": action_result,
            "exits": self._extract_exits(response_text),
        }

        room = self._extract_room(response_text)
        if room:
            result["room_quote"] = room

        objects, npcs = self._extract_visible_entities(response_text)
        result["objects"] = objects
        result["npcs"] = npcs

        if verb == "take" and obj and action_result["succeeded"]:
            result["inventory_changes"] = [{"item": obj, "change": "gained"}]

        return result

    @staticmethod
    def _classify_action_result(response_text, verb):
        """succeeded/failed via config.hard_failure_pattern/soft_failure_pattern
        — either means the action didn't succeed, and apply_parse_result's own
        _classify_action_result re-derives hard vs. invalid/blocked from the
        response text regardless of what's reported here, so this only needs
        to get succeeded right, not which kind of failure.

        "take" needs a stricter, positive confirmation
        (config.take_confirmation_pattern) rather than just "no failure
        pattern matched" — the same "don't guess a take succeeded" caution
        the legacy and tool-calling paths already apply (see
        docs/agent_behavior_spec.md's "Handling uncertainty"); every other
        verb keeps the simpler "no failure pattern matched" = succeeded
        default already used elsewhere in this codebase.
        """
        if response_classification.is_hard_failure(response_text) or response_classification.is_soft_failure(response_text):
            return {"succeeded": False, "reason_if_failed": None}
        if verb == "take":
            succeeded = bool(config.take_confirmation_pattern.search(response_text))
        else:
            succeeded = True
        return {"succeeded": succeeded, "reason_if_failed": None}

    @staticmethod
    def _extract_room(response_text):
        """Last match of the last configured pattern that matches anything —
        picks up the post-narration location on a response that mentions
        "you are in X" more than once (e.g. a death-and-respawn narrated in
        the same response), since the final mention reflects where the
        player actually ended up. Returns the raw matched text unshortened;
        world_graph.resolve_room_name (called downstream by apply_parse_result)
        already truncates at the first comma/semicolon and strips leading
        articles/prepositions, so over-capturing trailing description here
        is harmless."""
        room = None
        for pattern in config.room_patterns:
            matches = list(pattern.finditer(response_text))
            if matches:
                room = matches[-1].group("room")
        return room

    @staticmethod
    def _extract_exits(response_text):
        """Splits each matched exits clause on comma/"and" and keeps only
        recognized direction words. Deliberately strips a leading "in all
        directions"/"in every direction" phrase before splitting rather than
        expanding it to every compass point — that phrase isn't a grounded
        read of which exits actually exist (the same reasoning as bug 76's
        LLM-side fix), though a genuine additional exit named after it (e.g.
        "Exits lead in all directions and inside.") still gets kept."""
        # Computed here, not at module level, to match the surrounding
        # strategy pattern and keep direction vocabulary lookup local.
        known_direction_words = world_graph.DIRECTIONS | set(world_graph.DIRECTION_NORMALIZE)
        exits = []
        for pattern in config.exit_clause_patterns:
            for m in pattern.finditer(response_text):
                clause = _ALL_DIRECTIONS_PREFIX_RE.sub("", m.group("exits").strip())
                if not clause:
                    continue
                for part in _LIST_SPLIT_RE.split(clause):
                    word = part.strip().lower()
                    if word in known_direction_words:
                        exits.append(word)
        return exits

    @classmethod
    def _extract_visible_entities(cls, response_text):
        objects, npcs = [], []
        for pattern in config.visible_entity_patterns:
            for m in pattern.finditer(response_text):
                clause = m.group("entities").strip()
                for part in _LIST_SPLIT_RE.split(clause):
                    part = part.strip()
                    if not part:
                        continue
                    category, name = cls._classify_entity(part)
                    (npcs if category == "npc" else objects).append(name)
        return objects, npcs

    @staticmethod
    def _classify_entity(phrase):
        """"the prophet"/"a golden disk"/"Denzyl"/"the Annihilator" all appear
        in the same "you can see X[, Y] and Z" list with no other marker
        distinguishing NPCs from objects, so this relies on how Knight Orc's
        own narration happens to phrase each: a common-noun phrase always
        carries a leading article ("a golden disk", "the prophet"), while a
        proper-noun NPC never does ("Denzyl", "Mighty Flynn") — except when
        it's a title referred to generically, hence also checking
        config.creature_words (extended with this game's own "the X" NPC
        roles — prophet, valkyrie, hermit, innkeeper, gripper) and checking
        capitalization even after stripping an article, to catch a named
        NPC introduced with one ("the Annihilator")."""
        m = _LEADING_ARTICLE_RE.match(phrase)
        bare = m.group(1) if m else phrase
        if response_classification.is_creature(bare) or bare[:1].isupper():
            return "npc", bare
        return "object", bare
