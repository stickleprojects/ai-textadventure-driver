"""Game-response classification — pure, stateless, config-driven text
classifiers wrapping game_config.config's compiled patterns.

Moved out of agent.py (feature 64): parse_strategies/base.py and
parse_strategies/deterministic.py already reached into these as if they
were public (agent._is_hard_failure, agent._is_soft_failure, etc.) — the
same shape of problem feature 63 solved for map/world-graph logic. None
of these functions depend on agent state or any other project module, so
this is a true leaf module (only `re` and `game_config.config`).
"""
import re

from game_config import config


def is_hard_failure(text):
    return bool(config.hard_failure_pattern.search(text))


def is_soft_failure(text):
    return bool(config.soft_failure_pattern.search(text))


def is_scenery_response(text):
    return bool(config.scenery_pattern.search(text))


def is_failure_response(text):
    return bool(config.failure_pattern.search(text))


def is_death(text):
    return bool(config.death_pattern.search(text))


_CARRYING_RE = re.compile(r"carrying[:\s]+(.+?)(?:\.\s*$|$)", re.IGNORECASE | re.DOTALL)
_NOT_CARRYING_RE = re.compile(r"not carrying|carrying nothing|nothing", re.IGNORECASE)
# Bug 72: Knight Orc's real INVENTORY response is "You own X[. You are
# wearing Y]." — never "You are carrying...". Confirmed across every
# historical run log with an "inventory" action; _CARRYING_RE never once
# matched real game output, so recheck_inventory's resync silently never
# fired. Checked first (a positive match takes precedence over the broad
# "nothing" scan below, which risks a false trigger from unrelated trailing
# narration, e.g. an NPC's shouted line); "carrying" stays as a fallback for
# other games/configs that might phrase it that way.
_OWN_RE = re.compile(r"you own\s+(.+?)\.", re.IGNORECASE)
_WEARING_RE = re.compile(r"you(?:'re| are) wearing\s+(.+?)\.", re.IGNORECASE)


def _split_item_list(raw):
    # Split on comma or " and " (handles "item1, item2 and item3")
    parts = re.split(r",|\s+and\s+", raw.strip().rstrip("."), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def parse_inventory_response(text):
    """Return item list from a game inventory response, or None if unparseable."""
    items = []
    own_match = _OWN_RE.search(text)
    if own_match:
        items.extend(_split_item_list(own_match.group(1)))
    wearing_match = _WEARING_RE.search(text)
    if wearing_match:
        items.extend(_split_item_list(wearing_match.group(1)))
    if items:
        return items

    if _NOT_CARRYING_RE.search(text):
        return []
    m = _CARRYING_RE.search(text)
    if not m:
        return None
    return _split_item_list(m.group(1))


def is_creature(name):
    return bool(set(name.lower().split()) & config.creature_words)
