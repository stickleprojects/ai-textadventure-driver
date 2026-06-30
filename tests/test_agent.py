from unittest.mock import patch, PropertyMock

import pytest
import networkx as nx

from agent import (
    _compute_utility,
    _detect_loop,
    _is_failure_response,
    _is_hard_failure,
    _is_soft_failure,
    _mark_edge_futile,
    _parse_inspection_action,
    _record_verb_outcome,
    _snapshot_state,
    determine_next_action,
    process_agent_step,
    update_graph,
)
from tests.conftest import make_state


# ── _is_failure_response (backward compat — matches hard OR soft) ─────────────

@pytest.mark.parametrize("text", [
    "You can't do that.",
    "You can't see the huge knight.",
    "I don't understand that.",
    "Nothing happens.",
    "That's not something you can take.",
    "There is no sword here.",
    "I don't know that word.",
    "You don't need to use the word PUSH to finish this part of the game.",
    "You don't need to use the word EAT to finish this part of the game.",
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


# ── _is_hard_failure ──────────────────────────────────────────────────────────

class TestIsHardFailure:
    def test_cant_is_hard(self):
        assert _is_hard_failure("You can't do that.")

    def test_nothing_happens_is_hard(self):
        assert _is_hard_failure("Nothing happens.")

    def test_dont_know_word_is_hard(self):
        assert _is_hard_failure("I don't know that word.")

    def test_success_not_hard(self):
        assert not _is_hard_failure("Taken.")

    def test_not_yet_not_hard(self):
        assert not _is_hard_failure("Not yet.")

    def test_already_wearing_not_hard(self):
        assert not _is_hard_failure("You're already wearing armour.")


# ── _is_soft_failure ──────────────────────────────────────────────────────────

class TestIsSoftFailure:
    def test_not_right_now_is_soft(self):
        assert _is_soft_failure("You can't do that right now.")

    def test_already_wearing_is_soft(self):
        assert _is_soft_failure("You're already wearing armour.")

    def test_not_yet_is_soft(self):
        assert _is_soft_failure("Not yet.")

    def test_while_wearing_is_soft(self):
        assert _is_soft_failure("You can't do that while you're wearing a cloak.")

    def test_success_not_soft(self):
        assert not _is_soft_failure("Taken.")

    def test_nothing_happens_not_soft(self):
        assert not _is_soft_failure("Nothing happens.")


# ── _parse_inspection_action ──────────────────────────────────────────────────

class TestParseInspectionAction:
    def test_simple_verb(self):
        assert _parse_inspection_action("examine sword", "sword") == "examine"

    def test_multi_word_verb(self):
        assert _parse_inspection_action("look inside chest", "chest") == "look inside"

    def test_multi_word_object(self):
        assert _parse_inspection_action("wear leather gloves", "leather gloves") == "wear"

    def test_take(self):
        assert _parse_inspection_action("take key", "key") == "take"

    def test_no_match_returns_none(self):
        assert _parse_inspection_action("go north", "sword") is None

    def test_none_target_returns_none(self):
        assert _parse_inspection_action("take sword", None) is None

    def test_case_insensitive(self):
        assert _parse_inspection_action("Examine Sword", "sword") == "Examine"


# ── _record_verb_outcome ──────────────────────────────────────────────────────

class TestRecordVerbOutcome:
    def test_records_outcome_on_existing_entity(self):
        state = make_state()
        state["known_entities"]["sword"] = {"status": "held", "location": "Hall", "verb_outcomes": {}}
        _record_verb_outcome(state, "sword", "read", "invalid")
        assert state["known_entities"]["sword"]["verb_outcomes"]["read"] == "invalid"

    def test_creates_entity_if_missing(self):
        state = make_state()
        _record_verb_outcome(state, "new_item", "examine", "succeeded")
        assert "new_item" in state["known_entities"]
        assert state["known_entities"]["new_item"]["verb_outcomes"]["examine"] == "succeeded"

    def test_adds_verb_outcomes_to_entity_without_it(self):
        state = make_state()
        state["known_entities"]["lamp"] = {"status": "discovered", "location": "Hall"}
        _record_verb_outcome(state, "lamp", "wear", "blocked")
        assert state["known_entities"]["lamp"]["verb_outcomes"]["wear"] == "blocked"

    def test_overwrites_existing_outcome(self):
        state = make_state()
        state["known_entities"]["hat"] = {"status": "held", "location": "Hall",
                                           "verb_outcomes": {"wear": "blocked"}}
        _record_verb_outcome(state, "hat", "wear", "succeeded")
        assert state["known_entities"]["hat"]["verb_outcomes"]["wear"] == "succeeded"


# ── _detect_loop ──────────────────────────────────────────────────────────────

def test_detect_loop_finds_repeated_action():
    game_log = [{"action": "look"} for _ in range(10)]
    assert _detect_loop(game_log) == "look"


def test_detect_loop_ignores_varied_actions():
    actions = ["look", "wait", "north"] * 3
    game_log = [{"action": a} for a in actions]
    assert _detect_loop(game_log) is None


def test_detect_loop_ignores_productive_navigation():
    rooms = ["Room A", "Room B", "Room C", "Room D", "Room E"]
    game_log = [
        {"action": "north", "extracted": {"room": rooms[i + 1]}}
        if i < 4 else {"action": "look", "extracted": {"room": rooms[4]}}
        for i in range(5)
    ]
    game_log = [{"action": "look", "extracted": {"room": "Room A"}}] + game_log
    assert _detect_loop(game_log, window=6, threshold=4) is None


def test_detect_loop_catches_navigation_stuck_in_same_room():
    game_log = [{"action": "north", "extracted": {"room": "Dead End"}} for _ in range(10)]
    assert _detect_loop(game_log) == "north"


# ── determine_next_action — existing priority logic ───────────────────────────

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
    state = make_state(current_inspection={"target": "sword", "sequence": ["examine", "read", "look inside"], "step_index": 0})
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


# ── determine_next_action — dynamic verb sequence (issue 18) ─────────────────

class TestDetermineNextActionDynamicSequence:
    def test_invalid_verbs_skipped_in_sequence(self):
        state = make_state()
        state["known_entities"]["sword"] = {
            "status": "discovered",
            "location": "Hall",
            "verb_outcomes": {"read": "invalid", "look inside": "invalid"},
        }
        state["uninspected_objects"] = ["sword"]
        determine_next_action(state)  # take
        seq = state["current_inspection"]["sequence"]
        assert "read" not in seq
        assert "look inside" not in seq
        assert "examine" in seq

    def test_blocked_verbs_included_in_sequence(self):
        test_verbs = ["take", "examine", "wear", "eat"]
        with patch("agent.config") as mock_cfg:
            mock_cfg.candidate_verbs = test_verbs
            state = make_state()
            state["known_entities"]["cloak"] = {
                "status": "discovered",
                "location": "Hall",
                "verb_outcomes": {"wear": "blocked"},
            }
            state["uninspected_objects"] = ["cloak"]
            determine_next_action(state)  # take
            seq = state["current_inspection"]["sequence"]
            assert "wear" in seq

    def test_no_prior_outcomes_uses_full_candidate_list(self):
        from game_config import config
        state = make_state(uninspected_objects=["lamp"])
        determine_next_action(state)  # take
        seq = state["current_inspection"]["sequence"]
        expected = [v for v in config.candidate_verbs if v != "take"]
        assert seq == expected

    def test_full_inspection_exhausts_all_candidate_verbs(self):
        test_verbs = ["take", "examine", "read", "wear"]
        with patch("agent.config") as mock_cfg:
            mock_cfg.candidate_verbs = test_verbs
            state = make_state(uninspected_objects=["sword"])
            n = len(test_verbs)  # take + 3 post-take verbs
            actions = [determine_next_action(state) for _ in range(n)]
        assert actions[0] == "take sword"
        assert "examine sword" in actions
        assert "read sword" in actions
        assert "wear sword" in actions
        assert state["current_inspection"]["target"] is None


# ── process_agent_step — verb outcome recording (issue 18) ───────────────────

class TestProcessAgentStepOutcomes:
    def test_take_failure_records_invalid_and_clears_inspection(self, stub_child):
        state = make_state(uninspected_objects=["wall"])
        state["known_entities"]["wall"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="You can't take that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["wall"]["verb_outcomes"].get("take") == "invalid"
        assert state["current_inspection"]["target"] is None

    def test_take_success_records_succeeded(self, stub_child):
        state = make_state(uninspected_objects=["key"])
        state["known_entities"]["key"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="Taken."), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["key"]}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["key"]["verb_outcomes"].get("take") == "succeeded"
        assert state["current_inspection"]["target"] == "key"

    def test_hard_failure_records_invalid_does_not_clear_inspection(self, stub_child):
        state = make_state(current_inspection={
            "target": "sword", "sequence": ["read", "wear"], "step_index": 0,
        })
        state["known_entities"]["sword"] = {"status": "held", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="Nothing happens."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["sword"]["verb_outcomes"].get("read") == "invalid"
        assert state["current_inspection"]["target"] == "sword"

    def test_soft_failure_records_blocked_does_not_clear_inspection(self, stub_child):
        state = make_state(current_inspection={
            "target": "cloak", "sequence": ["wear", "examine"], "step_index": 0,
        })
        state["known_entities"]["cloak"] = {"status": "held", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="You're already wearing armour."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["cloak"]["verb_outcomes"].get("wear") == "blocked"
        assert state["current_inspection"]["target"] == "cloak"

    def test_success_records_succeeded(self, stub_child):
        state = make_state(current_inspection={
            "target": "hat", "sequence": ["wear", "examine"], "step_index": 0,
        })
        state["known_entities"]["hat"] = {"status": "held", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="You put on the hat."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["hat"]["verb_outcomes"].get("wear") == "succeeded"

    def test_npc_not_added_to_inspection_queue(self, stub_child):
        state = make_state(known_npcs={"horse": {"location": "Stable", "greeted": False}})
        with patch("agent.execute_game_command", return_value="A horse is here."), \
             patch("agent.extract_knowledge", return_value={"room": "Stable", "objects": ["horse"]}):
            process_agent_step(state, stub_child, None)
        assert "horse" not in state["uninspected_objects"]
        assert "horse" not in state["known_entities"]


# ── issue 19 regression tests ─────────────────────────────────────────────────

class TestNullRoomHandling:
    def test_null_room_in_extracted_does_not_overwrite_current_room(self, stub_child):
        state = make_state(current_room="Dingy Stable")
        with patch("agent.execute_game_command", return_value="Nothing happens."), \
             patch("agent.extract_knowledge", return_value={"room": None, "exits": []}):
            process_agent_step(state, stub_child, None)
        assert state["current_room"] == "Dingy Stable"

    def test_absent_room_in_extracted_does_not_overwrite_current_room(self, stub_child):
        state = make_state(current_room="Dingy Stable")
        with patch("agent.execute_game_command", return_value="Taken."), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["key"]}):
            process_agent_step(state, stub_child, None)
        assert state["current_room"] == "Dingy Stable"

    def test_update_graph_none_room_is_a_no_op(self):
        state = make_state(current_room="Hall")
        initial_nodes = set(state["world_graph"].nodes)
        update_graph(state, None, ["north"], "Hall", "north")
        assert set(state["world_graph"].nodes) == initial_nodes


class TestStuckAnomalyLoop:
    def test_hard_failure_on_goal_action_removes_anomaly(self, stub_child):
        g = nx.DiGraph()
        g.add_node("Garbage Pile")
        state = make_state(
            current_room="Garbage Pile",
            inventory=["putty knife"],
            unresolved_anomalies={"rubbish": {
                "room": "Garbage Pile",
                "reason": "blocking path",
                "potential_solution": "putty knife",
            }},
            world_graph=g,
        )
        with patch("agent.execute_game_command", return_value="Nothing happens."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert "rubbish" not in state["unresolved_anomalies"]

    def test_success_on_goal_action_keeps_anomaly_for_llm_to_resolve(self, stub_child):
        g = nx.DiGraph()
        g.add_node("Garbage Pile")
        state = make_state(
            current_room="Garbage Pile",
            inventory=["putty knife"],
            unresolved_anomalies={"rubbish": {
                "room": "Garbage Pile",
                "reason": "blocking path",
                "potential_solution": "putty knife",
            }},
            world_graph=g,
        )
        with patch("agent.execute_game_command", return_value="The rubbish clears."), \
             patch("agent.extract_knowledge", return_value={"resolved_anomalies": ["rubbish"]}):
            process_agent_step(state, stub_child, None)
        assert "rubbish" not in state["unresolved_anomalies"]


# ── utility tagging (_compute_utility + process_agent_step integration) ───────

class TestComputeUtility:
    def _snap(self, room="Hall", entities=0, npcs=0, inventory=0):
        return {"room": room, "entity_count": entities, "npc_count": npcs, "inventory_count": inventory}

    def test_room_change_is_productive(self):
        before = self._snap(room="Hall")
        after = self._snap(room="Courtyard")
        assert _compute_utility("north", "You go north.", before, after, None, None, {}) == "productive"

    def test_new_entity_is_productive(self):
        before = self._snap(entities=0)
        after = self._snap(entities=1)
        assert _compute_utility("look", "You see a sword.", before, after, None, None, {}) == "productive"

    def test_new_npc_is_productive(self):
        before = self._snap(npcs=0)
        after = self._snap(npcs=1)
        assert _compute_utility("look", "A knight is here.", before, after, None, None, {}) == "productive"

    def test_new_inventory_item_is_productive(self):
        before = self._snap(inventory=0)
        after = self._snap(inventory=1)
        assert _compute_utility("take sword", "Taken.", before, after, "take", "sword", {}) == "productive"

    def test_hard_failure_with_no_state_change_is_futile(self):
        snap = self._snap()
        assert _compute_utility("north", "You can't do that.", snap, snap, None, None, {}) == "futile"

    def test_first_examine_with_no_state_change_is_informative(self):
        snap = self._snap()
        # verb not in pre_verb_outcomes → informative
        assert _compute_utility("examine sword", "A fine blade.", snap, snap, "examine", "sword", {}) == "informative"

    def test_repeated_examine_is_redundant(self):
        snap = self._snap()
        pre = {"examine": "succeeded"}
        assert _compute_utility("examine sword", "A fine blade.", snap, snap, "examine", "sword", pre) == "redundant"

    def test_look_with_no_target_is_informative(self):
        snap = self._snap()
        assert _compute_utility("look", "You are in the Hall.", snap, snap, None, None, {}) == "informative"

    def test_productive_beats_hard_failure(self):
        # If room changed despite a confusing failure message, still productive
        before = self._snap(room="Hall")
        after = self._snap(room="Courtyard")
        assert _compute_utility("north", "Nothing happens. You stumble through.", before, after, None, None, {}) == "productive"


class TestUtilityTaggedInLog:
    def test_futile_direction_tagged_in_log(self, stub_child):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        with patch("agent.execute_game_command", return_value="You can't do that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["game_log"][-1]["utility"] == "futile"

    def test_room_change_tagged_productive(self, stub_child):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        with patch("agent.execute_game_command", return_value="You go north."), \
             patch("agent.extract_knowledge", return_value={"room": "Courtyard", "exits": []}):
            process_agent_step(state, stub_child, None)
        assert state["game_log"][-1]["utility"] == "productive"

    def test_first_examine_tagged_informative(self, stub_child):
        state = make_state(
            current_room="Hall",
            current_inspection={"target": "sword", "sequence": ["examine"], "step_index": 0},
            known_entities={"sword": {"status": "held", "location": "Hall", "verb_outcomes": {}}},
        )
        with patch("agent.execute_game_command", return_value="A fine blade."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["game_log"][-1]["utility"] == "informative"

    def test_repeated_examine_tagged_redundant(self, stub_child):
        state = make_state(
            current_room="Hall",
            current_inspection={"target": "sword", "sequence": ["examine"], "step_index": 0},
            known_entities={"sword": {"status": "held", "location": "Hall", "verb_outcomes": {"examine": "succeeded"}}},
        )
        with patch("agent.execute_game_command", return_value="A fine blade."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["game_log"][-1]["utility"] == "redundant"


# ── futile_edges ──────────────────────────────────────────────────────────────

class TestMarkEdgeFutile:
    def test_adds_to_futile_edges_set(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        _mark_edge_futile(state, "Hall", "north")
        assert ("Hall", "north") in state["futile_edges"]

    def test_tags_graph_edge_data(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        _mark_edge_futile(state, "Hall", "north")
        data = state["world_graph"].get_edge_data("Hall", "Unknown (north from Hall)")
        assert data.get("futile") is True

    def test_no_crash_when_edge_absent(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        _mark_edge_futile(state, "Hall", "north")  # edge doesn't exist — should not raise
        assert ("Hall", "north") in state["futile_edges"]


class TestFutileEdgesInNavigation:
    def test_futile_edge_skipped_prefers_non_futile(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north", futile=True)
        state["world_graph"].add_edge("Hall", "Unknown (east from Hall)", label="east")
        state["futile_edges"].add(("Hall", "north"))
        assert determine_next_action(state) == "east"

    def test_all_exits_futile_falls_through_to_look(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north", futile=True)
        state["futile_edges"].add(("Hall", "north"))
        assert determine_next_action(state) == "look"

    def test_known_room_exit_never_skipped(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_node("Courtyard")
        state["world_graph"].add_edge("Hall", "Courtyard", label="north")
        # Even if we incorrectly mark it futile, it won't be picked by the Unknown scan
        # (it's not an Unknown node — this tests the Unknown filter still works)
        assert determine_next_action(state) == "look"


class TestFutileEdgesMarkedOnStep:
    def test_futile_direction_marked_after_hard_failure(self, stub_child):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        with patch("agent.execute_game_command", return_value="You can't do that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert ("Hall", "north") in state["futile_edges"]

    def test_productive_direction_not_marked_futile(self, stub_child):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        with patch("agent.execute_game_command", return_value="You go north."), \
             patch("agent.extract_knowledge", return_value={"room": "Courtyard", "exits": []}):
            process_agent_step(state, stub_child, None)
        assert ("Hall", "north") not in state["futile_edges"]

    def test_non_direction_hard_failure_not_marked(self, stub_child):
        state = make_state(
            current_room="Hall",
            current_inspection={"target": "rock", "sequence": ["eat"], "step_index": 0},
            known_entities={"rock": {"status": "held", "location": "Hall", "verb_outcomes": {}}},
        )
        with patch("agent.execute_game_command", return_value="You can't do that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert len(state["futile_edges"]) == 0
