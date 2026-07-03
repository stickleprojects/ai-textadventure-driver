import re
from collections import Counter
from datetime import datetime

import networkx as nx

from game_config import config
from game_engine import execute_game_command
from llm import extract_knowledge

_SCORE_RE = re.compile(r"you score\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)
_SCORE_INTERVAL = 20

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b\s*", re.IGNORECASE)
# Strip leading positional prepositions — LLM says "in an alder ghostwood",
# "on a jousting field" etc.  Two passes needed: preposition first, then article.
# "inside"/"outside" are NOT stripped: they denote distinct rooms.
_LEADING_PREP_RE = re.compile(r"^(in|on|at)\s+", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _short_room_name(name):
    """Truncate at the first comma or semicolon and strip trailing period.

    Knight Orc room names follow the pattern '<short name>[, | ; <description>]'.
    The LLM sometimes returns the full description, sometimes just the short name.
    Truncating at the first separator and stripping trailing punctuation gives a
    stable short form that matches across variants.
    """
    return re.split(r"[,;]", name, maxsplit=1)[0].rstrip(".")


def _normalize_room(name):
    name = _short_room_name(name)
    name = _LEADING_PREP_RE.sub("", name)
    return _ARTICLE_RE.sub("", name).strip().lower()


def _canonicalize_room(name):
    """Return a clean short node name for storage.

    Truncates at the first comma/semicolon (bug 57), strips leading
    prepositions and articles.  Mid-string articles and spatial prefixes
    like 'inside'/'outside' are preserved.
    """
    name = _short_room_name(name)
    name = _LEADING_PREP_RE.sub("", name)
    name = _LEADING_ARTICLE_RE.sub("", name)
    return name.strip()


def _resolve_room_name(graph, room_name):
    """Return an existing graph node matching room_name after article normalisation.

    Prevents the same physical room being stored twice when the LLM returns
    slight article variations ('cave in juniper scrubland' vs 'cave in a juniper
    scrubland').  If no match exists, returns the canonicalised form of room_name
    so new nodes are stored without noisy leading prepositions/articles.
    """
    target = _normalize_room(room_name)
    for node in graph.nodes:
        if not node.startswith("Unknown") and _normalize_room(node) == target:
            return node
    return _canonicalize_room(room_name)


def _is_hard_failure(text):
    return bool(config.hard_failure_pattern.search(text))


def _is_soft_failure(text):
    return bool(config.soft_failure_pattern.search(text))


def _is_failure_response(text):
    return bool(config.failure_pattern.search(text))


def _is_death(text):
    return bool(config.death_pattern.search(text))


_CARRYING_RE = re.compile(r"carrying[:\s]+(.+?)(?:\.\s*$|$)", re.IGNORECASE | re.DOTALL)
_NOT_CARRYING_RE = re.compile(r"not carrying|carrying nothing|nothing", re.IGNORECASE)


def _parse_inventory_response(text):
    """Return item list from a game inventory response, or None if unparseable."""
    if _NOT_CARRYING_RE.search(text):
        return []
    m = _CARRYING_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().rstrip(".")
    # Split on comma or " and " (handles "item1, item2 and item3")
    parts = re.split(r",|\s+and\s+", raw, flags=re.IGNORECASE)
    items = [p.strip() for p in parts if p.strip()]
    return items if items else []


_DIRECTIONS = {"north", "south", "east", "west", "up", "down", "ne", "nw", "se", "sw"}


def _is_creature(name):
    return bool(set(name.lower().split()) & config.creature_words)


def _parse_inspection_action(action, target):
    """Return the verb if action is 'verb target', else None."""
    if not target:
        return None
    suffix = f" {target.lower()}"
    if action.lower().endswith(suffix):
        return action[:-(len(suffix))].strip()
    return None


def _record_verb_outcome(state, target, verb, outcome):
    """Record that verb was attempted on target with the given outcome."""
    entity = state["known_entities"].setdefault(
        target, {"status": "discovered", "location": state["current_room"], "verb_outcomes": {}}
    )
    entity.setdefault("verb_outcomes", {})[verb] = outcome


def _detect_loop(game_log, window=10, threshold=8):
    """Returns the repeated action if any single action appears >= threshold times
    in the last `window` log entries, else None.

    Direction actions that produced a room change are excluded — they are
    productive exploration, not repetition. A direction that keeps landing on
    the same room still counts toward the threshold.
    """
    recent = game_log[-window:]

    def _is_productive_move(i):
        if recent[i]["action"] not in _DIRECTIONS or i == 0:
            return False
        prev_room = recent[i - 1].get("extracted", {}).get("room")
        curr_room = recent[i].get("extracted", {}).get("room")
        return bool(prev_room and curr_room and prev_room != curr_room)

    actions = [e["action"] for i, e in enumerate(recent) if not _is_productive_move(i)]
    for action in set(actions):
        if actions.count(action) >= threshold:
            return action
    return None


_REVERSE = {
    "north": "south", "south": "north",
    "east": "west",   "west": "east",
    "up": "down",     "down": "up",
    "ne": "sw",       "sw": "ne",
    "nw": "se",       "se": "nw",
    "in": "out",      "out": "in",
}

# LLM sometimes returns full direction names; normalise to the abbreviated form
# used throughout the codebase so BFS vectors and placeholder keys stay consistent.
_DIRECTION_NORMALIZE = {
    "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    "inside": "in",    "outside": "out",
}


def update_graph(state, room_name, exits, previous_room, action):
    """Adds the current room and its exits to the world graph."""
    if not room_name:
        return
    exits = [_DIRECTION_NORMALIZE.get(d.lower(), d.lower()) for d in exits]
    if room_name not in state["world_graph"]:
        state["world_graph"].add_node(room_name)

    reverse_action = _REVERSE.get(action, "")
    if previous_room and previous_room != room_name and reverse_action:
        # Remove outbound placeholder from previous_room and the inbound placeholder
        # from room_name — both are now resolved by this traversal.
        for placeholder in (
            f"Unknown ({action} from {previous_room})",
            f"Unknown ({reverse_action} from {room_name})",
        ):
            if state["world_graph"].has_node(placeholder):
                state["world_graph"].remove_node(placeholder)
        # If the edge already exists (direction alias, e.g. south==down at start), merge
        # the new label rather than overwriting the existing one.
        if state["world_graph"].has_edge(previous_room, room_name):
            edge_data = state["world_graph"][previous_room][room_name]
            existing = edge_data.get("label", "").split("/")
            if action not in existing:
                edge_data["label"] = "/".join(existing + [action])
        else:
            state["world_graph"].add_edge(previous_room, room_name, label=action)

    for direction in exits:
        # Split merged labels (e.g. "south/down") so each component is checked individually
        existing_labels = set()
        for _, _, d in state["world_graph"].edges(room_name, data=True):
            for lbl in d.get("label", "").split("/"):
                existing_labels.add(lbl)
        if direction in existing_labels:
            continue
        # If this exit points back the way we came, wire the real return edge so
        # the placeholder is never created (and can't be re-added on the same step).
        if direction == reverse_action and previous_room and previous_room != room_name:
            state["world_graph"].add_edge(room_name, previous_room, label=direction)
            continue
        target_node = f"Unknown ({direction} from {room_name})"
        if not state["world_graph"].has_edge(room_name, target_node):
            state["world_graph"].add_edge(room_name, target_node, label=direction)


def get_next_move_to_target(state, target_room):
    """Returns the next movement command toward target_room via the shortest known path."""
    if target_room == state["current_room"]:
        return None
    try:
        path = nx.shortest_path(state["world_graph"], source=state["current_room"], target=target_room)
        if len(path) > 1:
            edge_data = state["world_graph"].get_edge_data(state["current_room"], path[1])
            # Edge label may store multiple aliases ("down/out") — only the first is needed.
            return edge_data['label'].split("/")[0]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _nav_command(state, target, fast=True):
    """Return a navigation command using the game's native nav if configured, else graph-based.

    Fast/full nav templates are only used when the target is in visited_rooms —
    the game only accepts 'run to X' / 'go to X' for rooms it has already seen
    the player enter. Unvisited targets fall back to graph-based step navigation.
    """
    template = config.fast_nav_command if fast else config.full_nav_command
    if template and target in state.get("visited_rooms", set()):
        return template.format(target=target)
    return get_next_move_to_target(state, target)


def determine_next_action(state):
    """Returns (action, reason) based on agent priority logic."""
    if state.get("position_lost"):
        state["position_lost"] = False
        return "look", "re-establishing position after lost room extraction"

    if state.get("recheck_inventory"):
        return "inventory", "post-death inventory check"

    if state["active_goal"]:
        target_room = state["active_goal"]["room"]
        if state["current_room"] == target_room:
            solution = state['active_goal']['solution']
            target = state['active_goal']['target']
            verb = "cast" if solution in state["spellbook"] else "use"
            state["active_goal"] = None
            return f"{verb} {solution} on {target}", f"goal: apply {solution} to {target}"
        move = _nav_command(state, target_room, fast=True)
        if move:
            return move, f"goal: navigating to {target_room}"
        state["active_goal"] = None

    all_capabilities = state["inventory"] + state["spellbook"]
    for target, anomaly_data in state["unresolved_anomalies"].items():
        solution = anomaly_data.get("potential_solution", "").lower()
        matching = [cap for cap in all_capabilities
                    if cap.lower() in solution or solution in cap.lower()]
        if matching:
            state["active_goal"] = {
                "room": anomaly_data["room"],
                "target": target,
                "solution": matching[0],
            }
            return determine_next_action(state)

    inspection = state["current_inspection"]
    if inspection["target"]:
        action = f"{inspection['sequence'][inspection['step_index']]} {inspection['target']}"
        inspection["step_index"] += 1
        if inspection["step_index"] >= len(inspection["sequence"]):
            inspection["target"] = None
            inspection["step_index"] = 0
        return action, f"inspecting {inspection['target'] or action.split()[1]}"

    if state["uninspected_objects"]:
        new_target = state["uninspected_objects"].pop(0)
        entity_data = state["known_entities"].get(new_target, {})
        verb_outcomes = entity_data.get("verb_outcomes", {})
        post_take_verbs = [
            v for v in config.candidate_verbs
            if v != "take" and verb_outcomes.get(v) != "invalid"
        ]
        inspection["target"] = new_target
        inspection["sequence"] = post_take_verbs
        inspection["step_index"] = 0
        return f"take {new_target}", f"new object: take {new_target}"

    for _, v, data in state["world_graph"].edges(state["current_room"], data=True):
        if v.startswith("Unknown") and not data.get("futile"):
            direction = data["label"].split("/")[0]
            return direction, f"exploring exit '{direction}' from {state['current_room']}"

    # Current room fully explored — navigate to a room with an Unknown exit.
    # Score by path_len + recent-direction penalty to avoid chasing the same diagonal
    # indefinitely (e.g. alternating sw/east across a grid of similar rooms).
    recent_dir_freq = Counter(
        e["action"] for e in state["game_log"][-20:]
        if e.get("action") in _DIRECTIONS
    )
    best_target = None
    best_score = float("inf")
    for node in state["world_graph"].nodes:
        if node.startswith("Unknown"):
            continue
        unknown_dirs = [
            data["label"].split("/")[0]
            for _, v, data in state["world_graph"].edges(node, data=True)
            if v.startswith("Unknown") and not data.get("futile")
        ]
        if not unknown_dirs:
            continue
        try:
            path_len = nx.shortest_path_length(
                state["world_graph"], state["current_room"], node
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        # Penalise rooms whose Unknown exits are in overused directions; prefer fresh ones
        min_dir_freq = min(recent_dir_freq.get(d, 0) for d in unknown_dirs)
        score = path_len + min_dir_freq
        if score < best_score:
            best_score = score
            best_target = node
    if best_target:
        move = _nav_command(state, best_target, fast=True)
        if move:
            return move, f"navigating to {best_target} (has unexplored exits)"

    # No Unknown exits anywhere — navigate to nearest unvisited known room.
    # This handles a pre-seeded graph where all edges are real but rooms haven't
    # been visited this run (so Unknown placeholders were never generated).
    # Guard: if current_room is None or absent from the graph, nx.shortest_path_length
    # treats None as "all sources" and returns a dict, crashing the < comparison.
    visited = state.get("visited_rooms", set())
    best_unvisited = None
    best_len = float("inf")
    current = state["current_room"]
    _known_nodes = state["world_graph"].nodes if (current and current in state["world_graph"]) else []
    for node in _known_nodes:
        if node.startswith("Unknown") or node in visited:
            continue
        try:
            path_len = nx.shortest_path_length(
                state["world_graph"], current, node
            )
            if path_len < best_len:
                best_len = path_len
                best_unvisited = node
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    if best_unvisited:
        move = _nav_command(state, best_unvisited, fast=True)
        if move:
            return move, f"navigating to unvisited room {best_unvisited}"

    step_count = len(state["game_log"])
    if step_count > 0 and step_count % _SCORE_INTERVAL == 0:
        return "score", "periodic score check"

    if state["pending_npc_tasks"]:
        cmd = state["pending_npc_tasks"][0]["wait_command"]
        return cmd, f"npc wait: {cmd}"

    return "look", "fallback: no unexplored exits or unvisited rooms reachable"


def _mark_edge_futile(state, from_room, direction):
    """Mark a direction from a room as permanently futile — skip in future unknown-exit scans."""
    state["futile_edges"].add((from_room, direction))
    for _, _, data in state["world_graph"].edges(from_room, data=True):
        if data.get("label") == direction:
            data["futile"] = True
            break


def _snapshot_state(state):
    return {
        "room": state["current_room"],
        "entity_count": len(state["known_entities"]),
        "npc_count": len(state["known_npcs"]),
        "inventory_count": len(state["inventory"]),
    }


def _compute_utility(action, response, snap_before, snap_after, insp_verb, effective_target, pre_verb_outcomes):
    """Classify action utility from state diff. No LLM — deterministic."""
    if (
        snap_before["room"] != snap_after["room"]
        or snap_after["entity_count"] > snap_before["entity_count"]
        or snap_after["npc_count"] > snap_before["npc_count"]
        or snap_after["inventory_count"] > snap_before["inventory_count"]
    ):
        return "productive"
    if _is_hard_failure(response):
        return "futile"
    if insp_verb and effective_target and insp_verb in pre_verb_outcomes:
        return "redundant"
    return "informative"


def process_agent_step(state, child, llm_instance):
    """Executes one agent cycle: decide → act → extract → update state."""
    previous_room = state["current_room"]

    # Capture inspection target before determine_next_action may change it
    pre_insp_target = state["current_inspection"]["target"]
    action_taken, action_reason = determine_next_action(state)
    post_insp_target = state["current_inspection"]["target"]

    # The effective target is the one associated with this action:
    # post_insp_target covers new takes; pre_insp_target covers ongoing inspection verbs.
    effective_target = post_insp_target or pre_insp_target
    insp_verb = _parse_inspection_action(action_taken, effective_target)

    snap_before = _snapshot_state(state)
    pre_verb_outcomes = (
        state["known_entities"].get(effective_target, {}).get("verb_outcomes", {}).copy()
        if effective_target else {}
    )

    response = execute_game_command(child, action_taken)

    # Record verb outcomes and handle take failure
    if insp_verb and effective_target:
        if insp_verb == "take":
            if _is_failure_response(response):
                _record_verb_outcome(state, effective_target, "take", "invalid")
                state["current_inspection"]["target"] = None
                state["current_inspection"]["step_index"] = 0
            else:
                _record_verb_outcome(state, effective_target, "take", "succeeded")
        elif _is_soft_failure(response):
            # Valid verb, blocked by current game state — retry later
            _record_verb_outcome(state, effective_target, insp_verb, "blocked")
        elif _is_hard_failure(response):
            # Verb is permanently invalid for this object
            _record_verb_outcome(state, effective_target, insp_verb, "invalid")
        else:
            _record_verb_outcome(state, effective_target, insp_verb, "succeeded")

    extracted = extract_knowledge(response, action_taken, llm_instance)
    token_usage = extracted.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
    llm_trace = extracted.pop("_trace", None)

    if extracted.get("room"):
        state["current_room"] = _resolve_room_name(state["world_graph"], extracted["room"])
        extracted["room"] = state["current_room"]
        state.setdefault("visited_rooms", set()).add(state["current_room"])
    elif action_taken in _DIRECTIONS and not _is_hard_failure(response) and not _is_soft_failure(response):
        # Movement appeared to succeed but LLM returned no room — position is unknown.
        # Force a look on the next step to re-establish where we are.
        state["position_lost"] = True

    if "exits" in extracted:
        resp_lower = response.lower()
        # The LLM infers "up"/"down" from "Exits lead in all directions" even when those
        # exits don't exist. Only trust them when literally present in the response text.
        exits = [e for e in extracted["exits"] if e not in ("up", "down") or e in resp_lower]
        extracted["exits"] = exits
        update_graph(state, state["current_room"], exits, previous_room, action_taken)

    # If a goal action hard-failed, clear the active_goal / drop the anomaly
    if _is_hard_failure(response):
        parts = action_taken.split(" on ", 1)
        if len(parts) == 2 and action_taken.startswith(("use ", "cast ")):
            state["unresolved_anomalies"].pop(parts[1], None)
        # go to / run to failure — game rejected the room name.  Drop both the
        # active_goal and the anomaly that triggered it so it is not immediately
        # re-queued.  With visited_rooms gating in _nav_command this should be
        # rare; the anomaly will be re-detected if the LLM sees it again later.
        if action_taken.startswith(("go to ", "run to ")):
            if state["active_goal"]:
                state["unresolved_anomalies"].pop(state["active_goal"].get("target", ""), None)
            state["active_goal"] = None

    for npc in extracted.get("npcs", []):
        if npc not in state["known_npcs"]:
            state["known_npcs"][npc] = {"location": state["current_room"], "greeted": False}

    if "objects" in extracted:
        for obj in extracted["objects"]:
            if obj in state["known_npcs"] or _is_creature(obj):
                if obj not in state["known_npcs"]:
                    state["known_npcs"][obj] = {"location": state["current_room"], "greeted": False}
                continue
            if obj not in state["known_entities"] and obj not in state["uninspected_objects"] and obj not in state["inventory"]:
                state["uninspected_objects"].append(obj)
                state["known_entities"][obj] = {"status": "discovered", "location": state["current_room"]}

    if _is_hard_failure(response):
        # Suppress LLM inventory hallucinations on failure responses so the log
        # reflects what was actually applied to state.
        extracted.pop("added_to_inventory", None)
    else:
        for item in extracted.get("added_to_inventory", []):
            if item not in state["inventory"]:
                state["inventory"].append(item)
            if item in state["known_entities"]:
                state["known_entities"][item]["status"] = "held"

    for spell in extracted.get("learned_spells", []):
        if spell not in state["spellbook"]:
            state["spellbook"].append(spell)

    for anomaly in extracted.get("anomalies", []):
        target = anomaly.get("target")
        if target and target not in state["unresolved_anomalies"]:
            state["unresolved_anomalies"][target] = {
                "room": state["current_room"],
                "reason": anomaly.get("reason"),
                "potential_solution": anomaly.get("potential_solution") or "",
            }

    for resolved in extracted.get("resolved_anomalies", []):
        state["unresolved_anomalies"].pop(resolved, None)

    if action_taken == "score":
        m = _SCORE_RE.search(response)
        if m:
            state["current_score"] = int(m.group(1))
            state["max_score"] = int(m.group(2))

    utility = _compute_utility(
        action_taken, response, snap_before, _snapshot_state(state),
        insp_verb, effective_target, pre_verb_outcomes,
    )

    if _is_death(response):
        utility = "death"
        state["recheck_inventory"] = True

    if action_taken == "inventory" and state.get("recheck_inventory"):
        parsed = _parse_inventory_response(response)
        if parsed is not None:
            for item in state["inventory"]:
                if item in state["known_entities"]:
                    state["known_entities"][item]["status"] = "discovered"
            state["inventory"] = parsed
            for item in parsed:
                if item in state["known_entities"]:
                    state["known_entities"][item]["status"] = "held"
        state["recheck_inventory"] = False

    if utility == "futile" and action_taken in _DIRECTIONS:
        _mark_edge_futile(state, previous_room, action_taken)

    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action_taken,
        "reason": action_reason,
        "response": response,
        "extracted": extracted,
        "score": state.get("current_score"),
        "utility": utility,
        "token_usage": token_usage,
    }
    if llm_trace:
        entry["llm_trace"] = llm_trace
    state["game_log"].append(entry)

    loop_action = _detect_loop(state["game_log"])
    if loop_action:
        entry["loop_detected"] = loop_action
