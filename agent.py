import re
from collections import Counter
from datetime import datetime

import networkx as nx

import parse_strategies
from game_config import config
from game_engine import execute_game_command
from llm import extract_knowledge  # noqa: F401 — kept as agent.extract_knowledge: parse_strategies.LLMJsonModeParseStrategy calls it via module-attribute lookup so tests patching "agent.extract_knowledge" still work
from world_graph import DIRECTIONS, REVERSE, mark_edge_futile, nav_command

_SCORE_RE = re.compile(r"you score\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)
_SCORE_INTERVAL = 20
# P007: cap on consecutive forced re-establishing "look"s before giving up and
# resuming normal priorities, so a game state that genuinely never confirms a
# room (or a run of unparseable "look" responses) can't loop forever.
_POSITION_LOST_MAX_ATTEMPTS = 3


def _is_hard_failure(text):
    return bool(config.hard_failure_pattern.search(text))


def _is_soft_failure(text):
    return bool(config.soft_failure_pattern.search(text))


def _is_scenery_response(text):
    return bool(config.scenery_pattern.search(text))


def _is_failure_response(text):
    return bool(config.failure_pattern.search(text))


def _is_death(text):
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


def _parse_inventory_response(text):
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
        if recent[i]["action"] not in DIRECTIONS or i == 0:
            return False
        prev_room = recent[i - 1].get("extracted", {}).get("room")
        curr_room = recent[i].get("extracted", {}).get("room")
        return bool(prev_room and curr_room and prev_room != curr_room)

    actions = [e["action"] for i, e in enumerate(recent) if not _is_productive_move(i)]
    for action in set(actions):
        if actions.count(action) >= threshold:
            return action
    return None


def determine_next_action(state):
    """Returns (action, reason) based on agent priority logic."""
    if state.get("position_lost"):
        # P007: previously cleared the flag unconditionally on the first forced
        # look, so a "look" that itself failed to yield a room left the agent
        # blind with a stale current_room and no further re-establishing look
        # (the re-arm branch in process_agent_step only fires on directional
        # moves). Now the flag only clears once a room is actually resolved
        # (in process_agent_step), and repeated attempts are capped rather
        # than looping forever.
        attempts = state.get("position_lost_attempts", 0) + 1
        if attempts <= _POSITION_LOST_MAX_ATTEMPTS:
            state["position_lost_attempts"] = attempts
            return "look", "re-establishing position after lost room extraction"
        state["position_lost"] = False
        state["position_lost_attempts"] = 0

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
        move = nav_command(state, target_room, fast=True)
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

    unknown_exits = [
        data for _, v, data in state["world_graph"].edges(state["current_room"], data=True)
        if v.startswith("Unknown") and not data.get("futile")
    ]
    if unknown_exits:
        # P004: prefer a genuinely fresh direction over the one that reverses the
        # move that just brought us here — that direction is what produces the
        # two-room oscillation (a real exit list can legitimately re-list it as
        # "Unknown" if the return trip wasn't wired in update_graph). Only fall
        # back to it when it's the sole remaining exit, so it's still explored.
        last_entry = state["game_log"][-1] if state["game_log"] else None
        last_direction = last_entry["action"] if last_entry and last_entry.get("action") in DIRECTIONS else None
        reverse_of_last = REVERSE.get(last_direction)
        fresh_exits = [data for data in unknown_exits if data["label"].split("/")[0] != reverse_of_last]
        direction = (fresh_exits or unknown_exits)[0]["label"].split("/")[0]
        return direction, f"exploring exit '{direction}' from {state['current_room']}"

    # Current room fully explored — navigate to a room with an Unknown exit.
    # Score by path_len + recent-direction penalty to avoid chasing the same diagonal
    # indefinitely (e.g. alternating sw/east across a grid of similar rooms), plus a
    # recent-room-visit penalty (P004) so a two-room shuttle loses out to a farther
    # but genuinely unexplored frontier once both sides of the shuttle have been
    # revisited a few times.
    recent_dir_freq = Counter(
        e["action"] for e in state["game_log"][-20:]
        if e.get("action") in DIRECTIONS
    )
    recent_room_visits = Counter(
        e.get("extracted", {}).get("room") for e in state["game_log"][-10:]
        if e.get("extracted", {}).get("room")
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
        score = path_len + min_dir_freq + recent_room_visits.get(node, 0)
        if score < best_score:
            best_score = score
            best_target = node
    if best_target:
        move = nav_command(state, best_target, fast=True)
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
        move = nav_command(state, best_unvisited, fast=True)
        if move:
            return move, f"navigating to unvisited room {best_unvisited}"

    step_count = len(state["game_log"])
    if step_count > 0 and step_count % _SCORE_INTERVAL == 0:
        return "score", "periodic score check"

    if state["pending_npc_tasks"]:
        cmd = state["pending_npc_tasks"][0]["wait_command"]
        return cmd, f"npc wait: {cmd}"

    return "look", "fallback: no unexplored exits or unvisited rooms reachable"


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


def process_agent_step(state, child, llm_instance, parse_strategy=None):
    """Executes one agent cycle: decide → act → parse → apply → update state.

    parse_strategy defaults to LLMJsonModeParseStrategy(llm_instance) — the
    unchanged extract_knowledge-based behavior every existing caller relies
    on. Pass a different parse_strategies.ParseStrategy (e.g. a
    DeterministicParseStrategy) to swap out parsing without touching
    determine_next_action or anything below.
    """
    if parse_strategy is None:
        parse_strategy = parse_strategies.LLMJsonModeParseStrategy(llm_instance)

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

    # Record verb outcomes for non-take inspection verbs directly from
    # current_inspection's own target/sequence (works for any verb, not just
    # config.candidate_verbs — current_inspection's sequence isn't guaranteed
    # to be drawn from there, e.g. a caller-supplied custom sequence). "take"
    # is handled after parsing below, since it needs added_to_inventory to
    # tell a confirmed success apart from an unrecognized response (see bug:
    # "That's too heavy." matched no failure pattern and was silently
    # recorded as "succeeded" even though the item was never actually
    # taken). apply_parse_result below does its own, independent verb-outcome
    # recording keyed off config.candidate_verbs (needed by agent_tools.py's
    # tool-calling path, which has no current_inspection concept at all) —
    # redundant with this block whenever the two derivations agree, but never
    # conflicting, since both read the same response text.
    if insp_verb and effective_target and insp_verb != "take":
        if _is_soft_failure(response):
            # Valid verb, blocked by current game state — retry later
            _record_verb_outcome(state, effective_target, insp_verb, "blocked")
        elif _is_hard_failure(response):
            # Verb is permanently invalid for this object
            _record_verb_outcome(state, effective_target, insp_verb, "invalid")
        else:
            _record_verb_outcome(state, effective_target, insp_verb, "succeeded")

    result = parse_strategy.parse(response, action_taken)
    token_usage = result.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
    llm_trace = result.pop("_trace", None)

    unrecognized_failure = None
    if insp_verb == "take" and effective_target:
        take_failed = False
        if _is_soft_failure(response):
            # State-dependent (e.g. "you're already carrying that", hands full) —
            # may succeed on a later attempt/run, so never persist as permanent.
            _record_verb_outcome(state, effective_target, "take", "blocked")
            take_failed = True
        elif _is_hard_failure(response):
            # Permanently un-takeable (e.g. too heavy, fixed in place, scenery) —
            # safe to persist cross-run.
            _record_verb_outcome(state, effective_target, "take", "invalid")
            take_failed = True
        else:
            taken = {i.lower() for i in result.get("added_to_inventory", [])}
            taken |= {
                c["item"].lower() for c in result.get("inventory_changes", [])
                if c.get("change") == "gained" and c.get("item")
            }
            if effective_target.lower() in taken:
                _record_verb_outcome(state, effective_target, "take", "succeeded")
            else:
                # Neither a known failure pattern nor confirmed by parsing as
                # actually taken — don't guess. Verify via INVENTORY (see
                # "Handling uncertainty" in docs/agent_behavior_spec.md) instead
                # of assuming success, and surface the raw response so a new
                # failure pattern can be classified and reviewed later (see
                # anomaly_detector._unrecognized_failure_responses).
                state["recheck_inventory"] = True
                take_failed = True
                unrecognized_failure = {
                    "target": effective_target,
                    "action": action_taken,
                    "response": response,
                }
        # A failed take doesn't mean the object isn't worth examining — e.g. "too
        # heavy" or "fixed in place" objects can still reveal sub-objects (bug 31)
        # or other useful text. Only abandon the queued inspection sequence when
        # the game itself says there's nothing there to look at (bug 70).
        if take_failed and _is_scenery_response(response):
            state["current_inspection"]["target"] = None
            state["current_inspection"]["step_index"] = 0

    # Everything else extract_knowledge used to drive directly (theft/gifts,
    # exit and room resolution with grounding, npcs/objects, inventory
    # apply, spells, anomalies, blocked_by, score) now lives in
    # parse_strategies.apply_parse_result, shared with agent_tools.py's
    # tool-calling path — see that module's docstring for the full schema.
    # Its own take/unrecognized_failure computation is redundant with the
    # block above (same inputs, same answer) — the locally-computed
    # unrecognized_failure above is authoritative for this path's log entry.
    extracted, is_death, _ = parse_strategies.apply_parse_result(
        state, result, action_taken, response, previous_room
    )

    utility = _compute_utility(
        action_taken, response, snap_before, _snapshot_state(state),
        insp_verb, effective_target, pre_verb_outcomes,
    )

    if is_death:
        utility = "death"
        state["recheck_inventory"] = True

    if utility == "futile" and action_taken in DIRECTIONS:
        mark_edge_futile(state, previous_room, action_taken)

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
    if unrecognized_failure:
        entry["unrecognized_failure"] = unrecognized_failure
    state["game_log"].append(entry)

    loop_action = _detect_loop(state["game_log"])
    if loop_action:
        entry["loop_detected"] = loop_action
