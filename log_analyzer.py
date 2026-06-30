"""Analyze a game log list and return a list of structured issue dicts.

Each issue has at minimum: type (str), description (str), count (int).
Additional keys are issue-specific.

Imported by scripts/run_and_analyze.py and tested in tests/test_log_analyzer.py.
"""
from collections import Counter

from game_config import config

# Verbs and directions that should never form part of a real room name
_ACTION_VERBS = frozenset({
    "take", "examine", "read", "look", "inside", "use", "on", "with",
    "cast", "wear", "drop", "put", "get", "pick", "go",
    "north", "south", "east", "west", "up", "down", "ne", "nw", "se", "sw",
})

# Room names the LLM produces when it has nothing real to say
_PLACEHOLDER_ROOMS = frozenset({
    "current location", "unknown location", "unknown", "current room",
    "your location", "here", "this room", "the room",
})


def _is_suspicious_room(room, action):
    """Return True if room looks hallucinated rather than extracted from game output."""
    if not room:
        return False
    room_lower = room.lower()
    if room_lower in _PLACEHOLDER_ROOMS:
        return True
    # Room name built from the action's object words (e.g. "putty knife room"
    # from action "take putty knife")
    object_words = {w for w in action.lower().split()
                    if w not in _ACTION_VERBS and len(w) > 3}
    if object_words and object_words.issubset(set(room_lower.split())):
        return True
    return False


def analyze_log(game_log):
    """Analyze a list of game log entry dicts and return detected issues."""
    issues = []
    n = len(game_log)
    if n == 0:
        return [{"type": "no_steps", "count": 0,
                 "description": "No steps were recorded — startup may have failed"}]

    # ── Empty LLM extractions ─────────────────────────────────────────────────
    empty = [e for e in game_log if not e["extracted"]]
    if empty:
        issues.append({
            "type": "empty_llm_extraction",
            "count": len(empty),
            "pct": round(100 * len(empty) / n),
            "description": (
                f"LLM returned empty dict on {len(empty)}/{n} steps "
                f"({round(100 * len(empty) / n)}%)"
            ),
            "example_inputs": [e["response"][:120] for e in empty[:3]],
        })

    # ── Hallucinated room names ───────────────────────────────────────────────
    suspicious = []
    for e in game_log:
        room = e["extracted"].get("room")
        if room and _is_suspicious_room(room, e["action"]):
            suspicious.append({
                "action": e["action"],
                "hallucinated_room": room,
                "response_snippet": e["response"][:80],
            })
    if suspicious:
        issues.append({
            "type": "hallucinated_room_name",
            "count": len(suspicious),
            "description": (
                "LLM invented a room name instead of omitting the field — "
                "fix: tell the LLM to omit 'room' when the response contains no room description"
            ),
            "examples": suspicious[:5],
        })

    # ── Loop detection fires ──────────────────────────────────────────────────
    loops = [e for e in game_log if e.get("loop_detected")]
    if loops:
        issues.append({
            "type": "loop_detected",
            "count": len(loops),
            "looping_actions": list({e["loop_detected"] for e in loops}),
            "description": "Agent got stuck repeating the same action",
        })

    # ── Timeouts / critical errors ────────────────────────────────────────────
    timeouts = [e for e in game_log
                if "WARNING" in e["response"] or "CRITICAL" in e["response"]]
    if timeouts:
        issues.append({
            "type": "command_timeout_or_error",
            "count": len(timeouts),
            "description": "Game commands timed out or produced critical errors",
            "examples": [{"action": e["action"], "snippet": e["response"][:200]}
                         for e in timeouts[:3]],
        })

    # ── Creature words in objects list (LLM misclassification) ───────────────
    misclassified = []
    for e in game_log:
        for obj in e["extracted"].get("objects", []):
            if any(w in obj.lower().split() for w in config.creature_words):
                misclassified.append({
                    "action": e["action"],
                    "object": obj,
                    "response_snippet": e["response"][:80],
                })
    if misclassified:
        issues.append({
            "type": "creature_misclassified_as_object",
            "count": len(misclassified),
            "description": "LLM put a living creature into the 'objects' list",
            "examples": misclassified[:5],
        })

    # ── Inspection failures ───────────────────────────────────────────────────
    inspection_fails = []
    for e in game_log:
        action = e["action"].lower()
        if any(action.startswith(v) for v in ("read ", "look inside ", "examine ")):
            if config.failure_pattern.search(e["response"]):
                inspection_fails.append({
                    "action": e["action"],
                    "response": e["response"][:100],
                })
    if inspection_fails:
        issues.append({
            "type": "inspection_failure",
            "count": len(inspection_fails),
            "description": "Agent tried to inspect items/creatures that refused the action",
            "examples": inspection_fails[:5],
        })

    # ── Blocked verb rate ─────────────────────────────────────────────────────
    soft_counts = Counter(
        e["action"] for e in game_log if config.soft_failure_pattern.search(e["response"])
    )
    persistently_blocked = {action: count for action, count in soft_counts.items() if count >= 3}
    if persistently_blocked:
        issues.append({
            "type": "blocked_verb_rate",
            "count": len(persistently_blocked),
            "description": (
                "Verbs repeatedly blocked by game state — "
                "the blocking state may never be resolving"
            ),
            "examples": [{"action": a, "count": c}
                         for a, c in list(persistently_blocked.items())[:5]],
        })

    # ── Stuck in a single room ────────────────────────────────────────────────
    rooms = [e["extracted"].get("room") for e in game_log if e["extracted"].get("room")]
    if rooms:
        most_common_room, count = Counter(rooms).most_common(1)[0]
        if count > n * 0.6:
            issues.append({
                "type": "stuck_in_room",
                "count": count,
                "room": most_common_room,
                "total_steps": n,
                "description": (
                    f"Agent spent {count}/{n} steps in '{most_common_room}'"
                ),
            })

    return issues
