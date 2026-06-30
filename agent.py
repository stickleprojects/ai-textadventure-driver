import re
from datetime import datetime

import networkx as nx

from game_config import config
from game_engine import execute_game_command
from llm import extract_knowledge

_SCORE_RE = re.compile(r"you score\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)
_SCORE_INTERVAL = 20


def _is_hard_failure(text):
    return bool(config.hard_failure_pattern.search(text))


def _is_soft_failure(text):
    return bool(config.soft_failure_pattern.search(text))


def _is_failure_response(text):
    return bool(config.failure_pattern.search(text))


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


def _detect_loop(game_log, window=10, threshold=4):
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


def update_graph(state, room_name, exits, previous_room, action):
    """Adds the current room and its exits to the world graph."""
    if not room_name:
        return
    if room_name not in state["world_graph"]:
        state["world_graph"].add_node(room_name)

    if (previous_room and previous_room != room_name
            and action in ["north", "south", "east", "west", "up", "down", "ne", "nw", "se", "sw"]):
        state["world_graph"].add_edge(previous_room, room_name, label=action)

    for direction in exits:
        target_node = f"Unknown ({direction} from {room_name})"
        existing_labels = {d.get('label') for _, _, d in state["world_graph"].edges(room_name, data=True)}
        if not state["world_graph"].has_edge(room_name, target_node) and direction not in existing_labels:
            state["world_graph"].add_edge(room_name, target_node, label=direction)


def get_next_move_to_target(state, target_room):
    """Returns the next movement command toward target_room via the shortest known path."""
    if target_room == state["current_room"]:
        return None
    try:
        path = nx.shortest_path(state["world_graph"], source=state["current_room"], target=target_room)
        if len(path) > 1:
            edge_data = state["world_graph"].get_edge_data(state["current_room"], path[1])
            return edge_data['label']
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _nav_command(state, target, fast=True):
    """Return a navigation command using the game's native nav if configured, else graph-based."""
    template = config.fast_nav_command if fast else config.full_nav_command
    if template:
        return template.format(target=target)
    return get_next_move_to_target(state, target)


def determine_next_action(state):
    """Returns the next command string based on agent priority logic."""
    if state["active_goal"]:
        target_room = state["active_goal"]["room"]
        if state["current_room"] == target_room:
            solution = state['active_goal']['solution']
            target = state['active_goal']['target']
            verb = "cast" if solution in state["spellbook"] else "use"
            state["active_goal"] = None
            return f"{verb} {solution} on {target}"
        move = _nav_command(state, target_room, fast=True)
        if move:
            return move
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
        return action

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
        return f"take {new_target}"

    for _, v, data in state["world_graph"].edges(state["current_room"], data=True):
        if v.startswith("Unknown") and not data.get("futile"):
            return data["label"]

    step_count = len(state["game_log"])
    if step_count > 0 and step_count % _SCORE_INTERVAL == 0:
        return "score"

    if state["pending_npc_tasks"]:
        return state["pending_npc_tasks"][0]["wait_command"]

    return "look"


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
    action_taken = determine_next_action(state)
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

    if extracted.get("room"):
        state["current_room"] = extracted["room"]

    if "exits" in extracted:
        update_graph(state, state["current_room"], extracted["exits"], previous_room, action_taken)

    # If a goal action hard-failed, drop the anomaly so it is not re-queued
    if _is_hard_failure(response):
        parts = action_taken.split(" on ", 1)
        if len(parts) == 2 and action_taken.startswith(("use ", "cast ")):
            state["unresolved_anomalies"].pop(parts[1], None)

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

    if utility == "futile" and action_taken in _DIRECTIONS:
        _mark_edge_futile(state, previous_room, action_taken)

    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action_taken,
        "response": response,
        "extracted": extracted,
        "score": state.get("current_score"),
        "utility": utility,
    }
    state["game_log"].append(entry)

    loop_action = _detect_loop(state["game_log"])
    if loop_action:
        entry["loop_detected"] = loop_action
