from datetime import datetime

import networkx as nx

from game_engine import execute_game_command
from llm import extract_knowledge


def update_graph(state, room_name, exits, previous_room, action):
    """Adds the current room and its exits to the world graph."""
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
        move = get_next_move_to_target(state, target_room)
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
        inspection["target"] = new_target
        inspection["step_index"] = 1
        return f"take {new_target}"

    for _, v, data in state["world_graph"].edges(state["current_room"], data=True):
        if v.startswith("Unknown"):
            return data["label"]

    return "look"


def process_agent_step(state, child, llm_instance):
    """Executes one agent cycle: decide → act → extract → update state."""
    previous_room = state["current_room"]
    action_taken = determine_next_action(state)

    response = execute_game_command(child, action_taken)
    extracted = extract_knowledge(response, action_taken, llm_instance)

    if "room" in extracted:
        state["current_room"] = extracted["room"]

    if "exits" in extracted:
        update_graph(state, state["current_room"], extracted["exits"], previous_room, action_taken)

    if "objects" in extracted:
        for obj in extracted["objects"]:
            if obj not in state["known_entities"] and obj not in state["uninspected_objects"] and obj not in state["inventory"]:
                state["uninspected_objects"].append(obj)
                state["known_entities"][obj] = {"status": "discovered", "location": state["current_room"]}

    for item in extracted.get("added_to_inventory", []):
        if item not in state["inventory"]:
            state["inventory"].append(item)

    for spell in extracted.get("learned_spells", []):
        if spell not in state["spellbook"]:
            state["spellbook"].append(spell)

    for anomaly in extracted.get("anomalies", []):
        target = anomaly.get("target")
        if target and target not in state["unresolved_anomalies"]:
            state["unresolved_anomalies"][target] = {
                "room": state["current_room"],
                "reason": anomaly.get("reason"),
                "potential_solution": anomaly.get("potential_solution"),
            }

    for resolved in extracted.get("resolved_anomalies", []):
        state["unresolved_anomalies"].pop(resolved, None)

    state["game_log"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action_taken,
        "response": response,
        "extracted": extracted,
    })
