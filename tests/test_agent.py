from unittest.mock import patch

import pytest

from agent import _detect_loop, _is_failure_response, determine_next_action, process_agent_step
from tests.conftest import make_state


@pytest.mark.parametrize("text", [
    "You can't do that.",
    "You can't see the huge knight.",
    "I don't understand that.",
    "Nothing happens.",
    "That's not something you can take.",
    "There is no sword here.",
    "I don't know that word.",
])
def test_failure_response_detected(text):
    assert _is_failure_response(text)


@pytest.mark.parametrize("text", [
    "You take the sword.",
    "Taken.",
    "You are in the Forest Path.",
])
def test_failure_response_not_detected(text):
    assert not _is_failure_response(text)


def test_detect_loop_finds_repeated_action():
    game_log = [{"action": "look"} for _ in range(10)]
    assert _detect_loop(game_log) == "look"


def test_detect_loop_ignores_varied_actions():
    actions = ["look", "wait", "north"] * 3
    game_log = [{"action": a} for a in actions]
    assert _detect_loop(game_log) is None


def test_detect_loop_ignores_productive_navigation():
    # "north" repeated 4 times but each time the extracted room changed — exploration, not a loop
    rooms = ["Room A", "Room B", "Room C", "Room D", "Room E"]
    game_log = [
        {"action": "north", "extracted": {"room": rooms[i + 1]}}
        if i < 4 else {"action": "look", "extracted": {"room": rooms[4]}}
        for i in range(5)
    ]
    # Prepend a prior entry so index 0 has a previous room to diff against
    game_log = [{"action": "look", "extracted": {"room": "Room A"}}] + game_log
    assert _detect_loop(game_log, window=6, threshold=4) is None


def test_detect_loop_catches_navigation_stuck_in_same_room():
    # "north" repeated but extracted room never changes — genuinely stuck
    game_log = [{"action": "north", "extracted": {"room": "Dead End"}} for _ in range(10)]
    assert _detect_loop(game_log) == "north"


def test_active_goal_in_target_room_uses_solution():
    state = make_state(
        current_room="Throne Room",
        inventory=["key"],
        active_goal={"room": "Throne Room", "target": "chest", "solution": "key"},
    )
    assert determine_next_action(state) == "use key on chest"
    assert state["active_goal"] is None


def test_active_goal_in_target_room_casts_spell():
    state = make_state(
        current_room="Throne Room",
        spellbook=["fireball"],
        active_goal={"room": "Throne Room", "target": "chest", "solution": "fireball"},
    )
    assert determine_next_action(state) == "cast fireball on chest"


def test_anomaly_resolved_by_inventory_sets_active_goal():
    import networkx as nx
    g = nx.DiGraph()
    g.add_edge("Forest", "Cave", label="north")
    state = make_state(
        current_room="Forest",
        inventory=["torch"],
        unresolved_anomalies={"dark cave": {"room": "Cave", "reason": "too dark", "potential_solution": "torch"}},
        world_graph=g,
    )
    action = determine_next_action(state)
    assert action == "north"
    assert state["active_goal"] == {"room": "Cave", "target": "dark cave", "solution": "torch"}


def test_inspection_sequence_continues():
    state = make_state(current_inspection={"target": "sword", "sequence": ["take", "examine", "read", "look inside"], "step_index": 1})
    assert determine_next_action(state) == "examine sword"


def test_uninspected_objects_starts_inspection():
    state = make_state(uninspected_objects=["sword", "key"])
    assert determine_next_action(state) == "take sword"
    assert state["current_inspection"]["target"] == "sword"
    assert state["uninspected_objects"] == ["key"]


def test_unknown_exit_explored():
    import networkx as nx
    g = nx.DiGraph()
    g.add_edge("Forest", "Unknown (north from Forest)", label="north")
    state = make_state(current_room="Forest", world_graph=g)
    assert determine_next_action(state) == "north"


def test_fallback_to_look():
    state = make_state()
    assert determine_next_action(state) == "look"


def test_inspection_sequence_full_order():
    state = make_state(uninspected_objects=["sword"])
    actions = [determine_next_action(state) for _ in range(4)]
    assert actions == ["take sword", "examine sword", "read sword", "look inside sword"]
    assert state["current_inspection"]["target"] is None


def test_npc_not_added_to_inspection_queue(stub_child):
    state = make_state(known_npcs={"horse": {"location": "Stable", "greeted": False}})
    with patch("agent.execute_game_command", return_value="A horse is here."), \
         patch("agent.extract_knowledge", return_value={"room": "Stable", "objects": ["horse"]}):
        process_agent_step(state, stub_child, None)
    assert "horse" not in state["uninspected_objects"]
    assert "horse" not in state["known_entities"]
