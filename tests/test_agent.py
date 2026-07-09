from unittest.mock import MagicMock, patch

import pytest
import networkx as nx

import agent_tools
import parse_strategies
from agent import (
    _POSITION_LOST_MAX_ATTEMPTS,
    _compute_utility,
    _detect_loop,
    _parse_inspection_action,
    _record_verb_outcome,
    determine_next_action,
    process_agent_step,
)
from game_config import config
from response_classification import (
    is_death,
    is_failure_response,
    is_hard_failure,
    is_scenery_response,
    is_soft_failure,
    parse_inventory_response,
)
from tests.conftest import make_state
from world_graph import mark_edge_futile, nav_command, normalize_direction, update_graph


# ── is_failure_response (backward compat — matches hard OR soft) ─────────────

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
    assert is_failure_response(text)


@pytest.mark.parametrize("text", [
    "You take the sword.",
    "Taken.",
    "You are in the Forest Path.",
])
def test_failure_response_not_detected(text):
    assert not is_failure_response(text)


# ── is_hard_failure ──────────────────────────────────────────────────────────

class TestIsHardFailure:
    def test_cant_is_hard(self):
        assert is_hard_failure("You can't do that.")

    def test_nothing_happens_is_hard(self):
        assert is_hard_failure("Nothing happens.")

    def test_dont_know_word_is_hard(self):
        assert is_hard_failure("I don't know that word.")

    def test_success_not_hard(self):
        assert not is_hard_failure("Taken.")

    def test_not_yet_not_hard(self):
        assert not is_hard_failure("Not yet.")

    def test_already_wearing_not_hard(self):
        assert not is_hard_failure("You're already wearing armour.")


# ── is_soft_failure ──────────────────────────────────────────────────────────

class TestIsSoftFailure:
    def test_not_right_now_is_soft(self):
        assert is_soft_failure("You can't do that right now.")

    def test_already_wearing_is_soft(self):
        assert is_soft_failure("You're already wearing armour.")

    def test_not_yet_is_soft(self):
        assert is_soft_failure("Not yet.")

    def test_while_wearing_is_soft(self):
        assert is_soft_failure("You can't do that while you're wearing a cloak.")

    def test_success_not_soft(self):
        assert not is_soft_failure("Taken.")

    def test_nothing_happens_not_soft(self):
        assert not is_soft_failure("Nothing happens.")


# ── is_scenery_response ────────────────────────────────────────────────────────

class TestIsSceneryResponse:
    def test_probably_just_scenery_is_scenery(self):
        assert is_scenery_response("That's probably just scenery.")

    def test_too_heavy_not_scenery(self):
        assert not is_scenery_response("That's too heavy.")

    def test_cant_take_that_not_scenery(self):
        assert not is_scenery_response("You can't take that.")

    def test_success_not_scenery(self):
        assert not is_scenery_response("Taken.")


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

    def test_case_insensitive_target_reuses_existing_key(self):
        state = make_state()
        state["known_entities"]["Putty Knife"] = {
            "status": "held",
            "location": "Hall",
            "verb_outcomes": {"read": "invalid"},
        }
        resolved = _record_verb_outcome(state, "putty knife", "examine", "succeeded")
        assert resolved == "Putty Knife"
        assert "putty knife" not in state["known_entities"]
        assert state["known_entities"]["Putty Knife"]["verb_outcomes"]["read"] == "invalid"
        assert state["known_entities"]["Putty Knife"]["verb_outcomes"]["examine"] == "succeeded"


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


def test_detect_loop_casefolds_mixed_casing_actions():
    game_log = [{"action": "look", "extracted": {"room": "Hall"}} for _ in range(3)]
    game_log.extend({"action": "LOOK", "extracted": {"room": "Hall"}} for _ in range(5))
    assert _detect_loop(game_log, window=8, threshold=8) == "look"


def test_detect_loop_exempts_productive_uppercase_direction_moves():
    game_log = [
        {"action": "look", "extracted": {"room": "Room A"}},
        {"action": "NORTH", "extracted": {"room": "Room B"}},
        {"action": "NORTH", "extracted": {"room": "Room C"}},
        {"action": "NORTH", "extracted": {"room": "Room D"}},
    ]
    assert _detect_loop(game_log, window=4, threshold=3) is None


def test_resolve_entity_key_case_insensitive_match_returns_existing_key():
    entities = {"Putty Knife": {"verb_outcomes": {"read": "invalid"}}}
    assert parse_strategies.resolve_entity_key(entities, "putty knife") == "Putty Knife"


# ── determine_next_action — existing priority logic ───────────────────────────

def test_active_goal_in_target_room_uses_solution():
    state = make_state(
        current_room="Throne Room",
        inventory=["key"],
        active_goal={"room": "Throne Room", "target": "chest", "solution": "key"},
    )
    assert determine_next_action(state)[0] == "use key on chest"
    assert state["active_goal"] is None


def test_active_goal_in_target_room_casts_spell():
    state = make_state(
        current_room="Throne Room",
        spellbook=["fireball"],
        active_goal={"room": "Throne Room", "target": "chest", "solution": "fireball"},
    )
    assert determine_next_action(state)[0] == "cast fireball on chest"


def test_anomaly_resolved_by_inventory_sets_active_goal():
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("Forest", "Cave", label="north")
    state = make_state(
        current_room="Forest",
        inventory=["torch"],
        unresolved_anomalies={"dark cave": {"room": "Cave", "reason": "too dark", "potential_solution": "torch"}},
        world_graph=g,
    )
    action, _ = determine_next_action(state)
    assert action == "north"
    assert state["active_goal"] == {"room": "Cave", "target": "dark cave", "solution": "torch"}


def test_inspection_sequence_continues():
    state = make_state(current_inspection={"target": "sword", "sequence": ["examine", "read", "look inside"], "step_index": 0})
    assert determine_next_action(state)[0] == "examine sword"


def test_uninspected_objects_starts_inspection():
    state = make_state(uninspected_objects=["sword", "key"])
    assert determine_next_action(state)[0] == "take sword"
    assert state["current_inspection"]["target"] == "sword"
    assert state["uninspected_objects"] == ["key"]


def test_uninspected_objects_uses_case_insensitive_preseeded_entity_history():
    state = make_state(
        known_entities={
            "Putty Knife": {
                "status": "discovered",
                "location": "Hall",
                "verb_outcomes": {"read": "invalid"},
            }
        },
        uninspected_objects=["putty knife"],
    )
    action, _ = determine_next_action(state)
    assert action == "take putty knife"
    assert "read" not in state["current_inspection"]["sequence"]


def test_unknown_exit_explored():
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("Forest", "Unknown (north from Forest)", label="north")
    state = make_state(current_room="Forest", world_graph=g)
    assert determine_next_action(state)[0] == "north"


def test_fallback_to_look():
    state = make_state()
    assert determine_next_action(state)[0] == "look"


# ── determine_next_action — P004: two-room oscillation loop ──────────────────

def test_unknown_exit_prefers_fresh_direction_over_reverse_of_last_move():
    g = nx.MultiDiGraph()
    # We just arrived in B by moving "north", so "south" would double back.
    g.add_edge("B", "Unknown (south from B)", label="south")
    g.add_edge("B", "Unknown (east from B)", label="east")
    state = make_state(
        current_room="B", world_graph=g,
        game_log=[{"action": "north", "extracted": {"room": "B"}}],
    )
    assert determine_next_action(state)[0] == "east"


def test_unknown_exit_falls_back_to_reverse_when_it_is_the_only_option():
    g = nx.MultiDiGraph()
    g.add_edge("B", "Unknown (south from B)", label="south")
    state = make_state(
        current_room="B", world_graph=g,
        game_log=[{"action": "north", "extracted": {"room": "B"}}],
    )
    assert determine_next_action(state)[0] == "south"


def test_best_target_scoring_penalizes_recently_shuttled_room():
    g = nx.MultiDiGraph()
    g.add_edge("X", "Y", label="north")
    g.add_edge("Y", "X", label="south")
    g.add_edge("Y", "Unknown (east from Y)", label="east")
    g.add_edge("X", "Z", label="west")
    g.add_edge("Z", "X", label="east")
    g.add_edge("Z", "Unknown (north from Z)", label="north")
    game_log = [
        {"action": "look", "extracted": {"room": "Y"}},
        {"action": "look", "extracted": {"room": "X"}},
        {"action": "look", "extracted": {"room": "Y"}},
        {"action": "look", "extracted": {"room": "X"}},
    ]
    state = make_state(current_room="X", world_graph=g, game_log=game_log)
    action, reason = determine_next_action(state)
    assert action == "west"
    assert "Z" in reason


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
            actions = [determine_next_action(state)[0] for _ in range(n)]
        assert actions[0] == "take sword"
        assert "examine sword" in actions
        assert "read sword" in actions
        assert "wear sword" in actions
        assert state["current_inspection"]["target"] is None


# ── process_agent_step — verb outcome recording (issue 18) ───────────────────

class TestProcessAgentStepOutcomes:
    def test_take_hard_failure_records_invalid_keeps_inspecting(self, stub_child):
        # Bug 70: "You can't take that." isn't scenery -- a permanently
        # un-takeable object is still worth running the inspection sequence on.
        state = make_state(uninspected_objects=["wall"])
        state["known_entities"]["wall"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="You can't take that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["wall"]["verb_outcomes"].get("take") == "invalid"
        assert state["current_inspection"]["target"] == "wall"

    def test_take_scenery_clears_inspection(self, stub_child):
        # Bug 58: "probably just scenery" was not in hard_failure_patterns so take
        # was recorded as succeeded and the full inspect sequence ran on scenery items.
        import re
        state = make_state(uninspected_objects=["grass"])
        state["known_entities"]["grass"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        scenery_pattern = re.compile("probably just scenery", re.IGNORECASE)
        with patch("agent.execute_game_command", return_value="That's probably just scenery."), \
             patch("agent.extract_knowledge", return_value={}), \
             patch.object(config, "hard_failure_pattern", scenery_pattern), \
             patch.object(config, "failure_pattern", scenery_pattern):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["grass"]["verb_outcomes"].get("take") == "invalid"
        assert state["current_inspection"]["target"] is None

    def test_take_failure_then_next_action_is_examine_not_abandoned(self, stub_child):
        # End-to-end regression for bug 70: after "take flagpole" hard-fails with a
        # non-scenery response, the next decision should be the queued inspection
        # verb (examine), not skip the object entirely.
        state = make_state(uninspected_objects=["flagpole"])
        state["known_entities"]["flagpole"] = {
            "status": "discovered", "location": "Field", "verb_outcomes": {},
        }
        with patch("agent.execute_game_command", return_value="You can't take that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        next_action, _ = determine_next_action(state)
        assert next_action == "examine flagpole"

    def test_take_success_records_succeeded(self, stub_child):
        state = make_state(uninspected_objects=["key"])
        state["known_entities"]["key"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="Taken."), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["key"]}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["key"]["verb_outcomes"].get("take") == "succeeded"
        assert state["current_inspection"]["target"] == "key"

    def test_take_unrecognized_response_triggers_recheck_not_succeeded(self, stub_child):
        # A genuinely novel phrase the config doesn't classify as hard, soft, or
        # scenery -- extraction correctly did not add the item to inventory, so
        # this must not be silently recorded as "succeeded" (that's how the agent
        # kept re-taking "pile of garbage" forever: never blacklisted since never
        # "invalid"). Deliberately nonsense text, not a real game phrase like "too
        # heavy" -- those get classified by configs/knight_orc.json (loaded as a
        # side effect of importing scripts.watch_run elsewhere in a full test run,
        # see bug 70), which would make this test's premise state-dependent.
        state = make_state(uninspected_objects=["pile of garbage"])
        state["known_entities"]["pile of garbage"] = {
            "status": "discovered", "location": "Field", "verb_outcomes": {},
        }
        with patch("agent.execute_game_command", return_value="Zarglebop refuses."), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": []}):
            process_agent_step(state, stub_child, None)
        assert "take" not in state["known_entities"]["pile of garbage"]["verb_outcomes"]
        assert state["recheck_inventory"] is True
        assert state["game_log"][-1]["unrecognized_failure"] == {
            "target": "pile of garbage",
            "action": "take pile of garbage",
            "response": "Zarglebop refuses.",
        }
        # Bug 70: an unrecognized (non-scenery) take failure must not abandon the
        # object -- it's still worth running the queued inspection sequence on,
        # just as soon as the pending inventory recheck is out of the way.
        assert state["current_inspection"]["target"] == "pile of garbage"

    def test_take_confirmed_by_extraction_records_succeeded_even_with_no_keyword_match(self, stub_child):
        # Regression guard: a genuine success with unusual phrasing (no "Taken."
        # keyword) must still record "succeeded" via added_to_inventory, and must
        # NOT be treated as unrecognized.
        state = make_state(uninspected_objects=["putty knife"])
        state["known_entities"]["putty knife"] = {
            "status": "discovered", "location": "Hall", "verb_outcomes": {},
        }
        with patch("agent.execute_game_command", return_value="You slip the putty knife into your pocket."), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["putty knife"]}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["putty knife"]["verb_outcomes"].get("take") == "succeeded"
        assert "unrecognized_failure" not in state["game_log"][-1]

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

    def test_hard_failure_does_not_add_to_inventory(self, stub_child):
        state = make_state(uninspected_objects=["fish-heads"])
        state["known_entities"]["fish-heads"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        response = "You don't need to use the word \"fish-heads\" to finish this part of the game."
        with patch("agent.execute_game_command", return_value=response), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["fish-heads"]}):
            process_agent_step(state, stub_child, None)
        assert "fish-heads" not in state["inventory"]

    def test_hard_failure_clears_added_to_inventory_from_log(self, stub_child):
        # LLM hallucination must not appear in the log entry — log should reflect
        # what was actually applied to state, not what the LLM erroneously extracted.
        state = make_state(uninspected_objects=["fish-heads"])
        state["known_entities"]["fish-heads"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        response = "You don't need to use the word \"fish-heads\" to finish this part of the game."
        with patch("agent.execute_game_command", return_value=response), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["fish-heads"]}):
            process_agent_step(state, stub_child, None)
        assert "added_to_inventory" not in state["game_log"][-1]["extracted"]

    def test_hard_failure_with_hallucinated_inventory_is_futile(self, stub_child):
        state = make_state(uninspected_objects=["fish-heads"])
        state["known_entities"]["fish-heads"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        response = "You don't need to use the word \"fish-heads\" to finish this part of the game."
        with patch("agent.execute_game_command", return_value=response), \
             patch("agent.extract_knowledge", return_value={"added_to_inventory": ["fish-heads"]}):
            process_agent_step(state, stub_child, None)
        assert state["game_log"][-1]["utility"] == "futile"

    def test_npc_not_added_to_inspection_queue(self, stub_child):
        state = make_state(known_npcs={"horse": {"location": "Stable", "greeted": False}})
        with patch("agent.execute_game_command", return_value="A horse is here."), \
             patch("agent.extract_knowledge", return_value={"room": "Stable", "objects": ["horse"]}):
            process_agent_step(state, stub_child, None)
        assert "horse" not in state["uninspected_objects"]
        assert "horse" not in state["known_entities"]

    def test_object_known_from_prior_run_is_still_queued_for_take(self, stub_child):
        # Bug: known_entities is pre-seeded cross-run from the strategy sidecar, so
        # any object ever seen in *any* prior run was blocking re-queueing here —
        # the agent would walk past a takeable item without ever attempting "take".
        state = make_state(known_entities={
            "lantern": {"status": "discovered", "location": None, "verb_outcomes": {}}
        })
        with patch("agent.execute_game_command", return_value="A lantern is here."), \
             patch("agent.extract_knowledge", return_value={"room": "Cellar", "objects": ["lantern"]}):
            process_agent_step(state, stub_child, None)
        assert "lantern" in state["uninspected_objects"]

    def test_object_permanently_untakeable_is_not_requeued(self, stub_child):
        # Cross-run "take": "invalid" (e.g. scenery) is a permanent fact about the
        # object and should still block re-queueing, unlike bare known_entities membership.
        state = make_state(known_entities={
            "grass": {"status": "discovered", "location": None, "verb_outcomes": {"take": "invalid"}}
        })
        with patch("agent.execute_game_command", return_value="Grass is here."), \
             patch("agent.extract_knowledge", return_value={"room": "Field", "objects": ["grass"]}):
            process_agent_step(state, stub_child, None)
        assert "grass" not in state["uninspected_objects"]

    def test_object_already_held_is_not_requeued(self, stub_child):
        state = make_state(
            inventory=["sword"],
            known_entities={"sword": {"status": "held", "location": None, "verb_outcomes": {}}},
        )
        with patch("agent.execute_game_command", return_value="A sword is here."), \
             patch("agent.extract_knowledge", return_value={"room": "Armoury", "objects": ["sword"]}):
            process_agent_step(state, stub_child, None)
        assert "sword" not in state["uninspected_objects"]

    def test_take_soft_failure_records_blocked_not_invalid(self, stub_child):
        # A carry-capacity/"already carrying" style failure is state-dependent, not a
        # permanent fact about the object — must not be persisted as "invalid" or the
        # object would be permanently blacklisted from ever being taken again.
        state = make_state(uninspected_objects=["cloak"])
        state["known_entities"]["cloak"] = {"status": "discovered", "location": "Hall", "verb_outcomes": {}}
        with patch("agent.execute_game_command", return_value="You're already carrying that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert state["known_entities"]["cloak"]["verb_outcomes"].get("take") == "blocked"
        # Bug 70: a soft (non-scenery) take failure must not abandon inspection —
        # the object is still there and still worth examining.
        assert state["current_inspection"]["target"] == "cloak"


# ── requirement 23 — NPC inventory transfers ───────────────────────────────────

class TestNpcInventoryTransfers:
    def test_received_from_npc_adds_to_inventory(self, stub_child):
        state = make_state()
        with patch("agent.execute_game_command", return_value="Denzyl gives you a spear."), \
             patch("agent.extract_knowledge",
                   return_value={"received_from_npc": [{"item": "spear", "npc": "Denzyl"}]}):
            process_agent_step(state, stub_child, None)
        assert "spear" in state["inventory"]
        assert state["known_entities"]["spear"]["status"] == "held"

    def test_received_from_npc_does_not_duplicate_already_held_item(self, stub_child):
        state = make_state(inventory=["spear"])
        with patch("agent.execute_game_command", return_value="Denzyl gives you a spear."), \
             patch("agent.extract_knowledge",
                   return_value={"received_from_npc": [{"item": "spear", "npc": "Denzyl"}]}):
            process_agent_step(state, stub_child, None)
        assert state["inventory"].count("spear") == 1

    def test_taken_by_npc_removes_from_inventory(self, stub_child):
        state = make_state(inventory=["gold plate", "sword"])
        with patch("agent.execute_game_command", return_value="The troll snatches the gold plate from you."), \
             patch("agent.extract_knowledge",
                   return_value={"taken_by_npc": [{"item": "gold plate", "npc": "troll"}]}):
            process_agent_step(state, stub_child, None)
        assert "gold plate" not in state["inventory"]
        assert "sword" in state["inventory"]

    def test_taken_by_npc_item_not_held_is_noop(self, stub_child):
        state = make_state(inventory=["sword"])
        with patch("agent.execute_game_command", return_value="The troll snatches the gold plate from you."), \
             patch("agent.extract_knowledge",
                   return_value={"taken_by_npc": [{"item": "gold plate", "npc": "troll"}]}):
            process_agent_step(state, stub_child, None)
        assert state["inventory"] == ["sword"]


# ── requirement 25 — route blockages ───────────────────────────────────────────

class TestRouteBlockages:
    def test_blocked_by_adds_unresolved_anomaly(self, stub_child):
        state = make_state(current_room="Field")
        with patch("agent.execute_game_command", return_value="You are blocked by the drawbridge."), \
             patch("agent.extract_knowledge",
                   return_value={"blocked_by": [{"obstacle": "drawbridge", "blocking": "route to castle"}]}):
            process_agent_step(state, stub_child, None)
        assert state["unresolved_anomalies"]["drawbridge"] == {
            "room": "Field", "reason": "route to castle", "potential_solution": "",
        }

    def test_blocked_by_does_not_overwrite_existing_anomaly(self, stub_child):
        state = make_state(
            current_room="Field",
            unresolved_anomalies={"drawbridge": {
                "room": "Field", "reason": "route to castle", "potential_solution": "lever",
            }},
        )
        with patch("agent.execute_game_command", return_value="You are blocked by the drawbridge."), \
             patch("agent.extract_knowledge",
                   return_value={"blocked_by": [{"obstacle": "drawbridge", "blocking": "route to castle"}]}):
            process_agent_step(state, stub_child, None)
        assert state["unresolved_anomalies"]["drawbridge"]["potential_solution"] == "lever"

    def test_blocked_by_anomaly_resolves_via_existing_capability_matching(self):
        # Integration check: blocked_by reuses the same unresolved_anomalies
        # mechanism as "anomalies", so determine_next_action's existing
        # inventory/spellbook matching picks it up with no new logic needed.
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_edge("Field", "Castle", label="north")
        state = make_state(
            current_room="Field",
            inventory=["lever"],
            unresolved_anomalies={"drawbridge": {
                "room": "Castle", "reason": "route to castle", "potential_solution": "lever",
            }},
            world_graph=g,
        )
        action, _ = determine_next_action(state)
        assert action == "north"
        assert state["active_goal"] == {"room": "Castle", "target": "drawbridge", "solution": "lever"}


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

    def test_exits_not_attributed_to_stale_room_when_room_lost(self, stub_child):
        """P008 regression: a combat-interlude-style response can report exits
        for the room just moved into while omitting the room name. Before the
        fix, update_graph attributed those exits to the stale current_room
        (the room just left), wiring phantom Unknown exits that don't actually
        exist there."""
        state = make_state(current_room="Dingy Stable")
        # An Unknown "east" exit so determine_next_action picks a direction
        # (matching the real scenario: a productive-looking move that then
        # loses the room).
        state["world_graph"].add_edge("Dingy Stable", "Unknown (east from Dingy Stable)", label="east")
        with patch("agent.execute_game_command",
                   return_value="A male voice shouts. You are shoved onward."), \
             patch("agent.extract_knowledge", return_value={"exits": ["north", "out"]}):
            process_agent_step(state, stub_child, None)
        assert state["position_lost"] is True
        assert not state["world_graph"].has_node("Unknown (north from Dingy Stable)")
        assert not state["world_graph"].has_node("Unknown (out from Dingy Stable)")

    def test_exits_still_recorded_when_room_extracted_alongside(self, stub_child):
        # Exits must be literally present in the response text (bug 76's
        # grounding, retroactively applied to this path) — a response that
        # only names the room, without naming its exits, can't demonstrate
        # this case.
        state = make_state(current_room="Dingy Stable")
        response = "You are in a huge pile of garbage. Exits lead north and out."
        with patch("agent.execute_game_command", return_value=response), \
             patch("agent.extract_knowledge",
                   return_value={"room": "huge pile of garbage", "exits": ["north", "out"]}):
            process_agent_step(state, stub_child, None)
        assert state["world_graph"].has_node("Unknown (north from huge pile of garbage)")
        assert state["world_graph"].has_node("Unknown (out from huge pile of garbage)")

    def test_update_graph_none_room_is_a_no_op(self):
        state = make_state(current_room="Hall")
        initial_nodes = set(state["world_graph"].nodes)
        update_graph(state, None, ["north"], "Hall", "north")
        assert set(state["world_graph"].nodes) == initial_nodes

    def test_traversal_removes_unknown_placeholder(self):
        state = make_state(current_room="Forest")
        state["world_graph"].add_edge("Forest", "Unknown (north from Forest)", label="north")
        update_graph(state, "Clearing", [], "Forest", "north")
        assert not state["world_graph"].has_node("Unknown (north from Forest)")

    def test_traversal_removes_reverse_placeholder_from_destination(self):
        state = make_state(current_room="Clearing")
        state["world_graph"].add_edge("Clearing", "Unknown (south from Clearing)", label="south")
        update_graph(state, "Clearing", [], "Forest", "north")
        assert not state["world_graph"].has_node("Unknown (south from Clearing)")

    def test_traversal_does_not_remove_other_direction_placeholders(self):
        state = make_state(current_room="Forest")
        state["world_graph"].add_edge("Forest", "Unknown (east from Forest)", label="east")
        update_graph(state, "Clearing", [], "Forest", "north")
        assert state["world_graph"].has_node("Unknown (east from Forest)")

    def test_reverse_exit_not_readded_as_placeholder_after_traversal(self):
        # Bug 47: going north from A→B removes Unknown(south from B), but then
        # processing B's exits sees "south" and re-adds the placeholder.
        # The fix wires the real return edge instead.
        state = make_state(current_room="Hawthorn Coppice")
        state["world_graph"].add_node("Blackthorn Underbrush")
        state["world_graph"].add_edge(
            "Blackthorn Underbrush", "Unknown (south from Blackthorn Underbrush)", label="south"
        )
        update_graph(state, "Blackthorn Underbrush", ["north", "south", "east"], "Hawthorn Coppice", "north")
        assert not state["world_graph"].has_node("Unknown (south from Blackthorn Underbrush)")
        # The real return edge should exist instead
        assert state["world_graph"].has_edge("Blackthorn Underbrush", "Hawthorn Coppice")

    def test_reverse_exit_not_created_as_placeholder_when_never_existed(self):
        # Variant: Unknown(south from B) was never added — arriving north from A
        # should still wire B→south→A and not create the Unknown placeholder.
        state = make_state(current_room="Hawthorn Coppice")
        update_graph(state, "Blackthorn Underbrush", ["north", "south"], "Hawthorn Coppice", "north")
        assert not state["world_graph"].has_node("Unknown (south from Blackthorn Underbrush)")
        assert state["world_graph"].has_edge("Blackthorn Underbrush", "Hawthorn Coppice")

    def test_direction_aliases_stored_as_separate_edges(self):
        # Bug 48: south==down at starting room — each alias is a separate edge
        # in a MultiDiGraph rather than a merged compound label.
        state = make_state(current_room="Start")
        update_graph(state, "Cellar", [], "Start", "south")
        update_graph(state, "Cellar", [], "Start", "down")
        edges = state["world_graph"].get_edge_data("Start", "Cellar")
        assert edges is not None
        labels = {d["label"] for d in edges.values()}
        assert "south" in labels
        assert "down" in labels

    def test_direction_alias_does_not_readd_unknown_for_merged_direction(self):
        # Bug 48: after south/down are merged, exits listing "south" again should
        # not create a new Unknown placeholder.
        state = make_state(current_room="Start")
        update_graph(state, "Cellar", [], "Start", "south")
        update_graph(state, "Cellar", [], "Start", "down")
        # Now revisit Start with exits that include "south" and "down"
        update_graph(state, "Start", ["south", "down", "east"], None, "")
        assert not state["world_graph"].has_node("Unknown (south from Start)")
        assert not state["world_graph"].has_node("Unknown (down from Start)")

    def test_article_variant_resolves_to_existing_node(self):
        # LLM sometimes adds/drops articles ("cave in juniper scrubland" vs
        # "cave in a juniper scrubland") — both should resolve to the same node.
        from world_graph import resolve_room_name
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("cave in juniper scrubland")
        assert resolve_room_name(g, "cave in a juniper scrubland") == "cave in juniper scrubland"
        assert resolve_room_name(g, "Cave In Juniper Scrubland") == "cave in juniper scrubland"
        assert resolve_room_name(g, "the cave in juniper scrubland") == "cave in juniper scrubland"

    def test_leading_in_prefix_resolves_to_existing_node(self):
        # Bug 46: LLM returns "in an alder ghostwood" / "in alder forest" when the
        # stored node is "alder ghostwood" / "alder forest".
        from world_graph import resolve_room_name
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("alder ghostwood")
        g.add_node("cedar tangle")
        g.add_node("alder forest")
        # "in an X" → same as "X" (leading preposition + article variant)
        assert resolve_room_name(g, "in an alder ghostwood") == "alder ghostwood"
        assert resolve_room_name(g, "an alder ghostwood") == "alder ghostwood"
        # article-only variant still works
        assert resolve_room_name(g, "a cedar tangle") == "cedar tangle"
        assert resolve_room_name(g, "an alder forest") == "alder forest"

        assert resolve_room_name(g, "on a jousting field") == "jousting field"

        # "inside" / "outside" must NOT be stripped — they are distinct locations
        g2 = nx.MultiDiGraph()
        g2.add_node("inside a cave")
        g2.add_node("outside a cave")
        assert resolve_room_name(g2, "inside a cave") == "inside a cave"
        assert resolve_room_name(g2, "outside a cave") == "outside a cave"
        # bare "a cave" must NOT accidentally match "inside a cave"; new node gets
        # canonicalised name ("cave"), not the raw LLM string
        assert resolve_room_name(g2, "a cave") == "cave"

    def test_description_suffix_resolves_to_existing_short_node(self):
        # Bug 57: LLM sometimes appends the full room description after the short name,
        # separated by comma or semicolon. Both forms must resolve to the same node.
        from world_graph import resolve_room_name
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("jousting field")
        g.add_node("dingy stable")
        g.add_node("dismal fairground in a rowan coppice")
        # semicolon-separated description variant
        assert resolve_room_name(g, "on a jousting field; an acre of firm meadow, divided by a fence") == "jousting field"
        # trailing period variant
        assert resolve_room_name(g, "on a jousting field; an acre of firm meadow.") == "jousting field"
        # comma-separated description variant
        assert resolve_room_name(g, "a dingy stable, a temporary building with canvas walls") == "dingy stable"
        # short form still works
        assert resolve_room_name(g, "a dingy stable") == "dingy stable"
        # no separator — unchanged
        assert resolve_room_name(g, "on a dismal fairground in a rowan coppice") == "dismal fairground in a rowan coppice"

    def test_new_node_stored_under_short_name(self):
        # Bug 57: when the LLM returns a long-form name for a room not yet in the graph,
        # the new node should be stored under the short name, not the full description.
        from world_graph import resolve_room_name
        import networkx as nx
        g = nx.MultiDiGraph()
        result = resolve_room_name(g, "a dingy stable, a temporary building with canvas walls")
        assert result == "dingy stable"
        result2 = resolve_room_name(g, "on a jousting field; an acre of firm meadow")
        assert result2 == "jousting field"

    # ── Bug 45: maze rooms sharing a name but not exits must not collapse ─────

    def test_disjoint_exits_on_repeated_name_creates_new_node(self):
        # Two physically distinct rooms both reported as "Alder Clump" by the LLM
        # (a maze). Their exit sets share nothing, so the second must resolve to
        # a fresh, disambiguated node rather than merging into the first.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        first = resolve_room_name(g, "Alder Clump", ["north", "east"])
        assert first == "Alder Clump"
        # Record the first room's exits as real edges, the way update_graph would
        # after a live step — a bare node with no edges has no recorded exits yet.
        update_graph({"world_graph": g}, first, ["north", "east"], None, "")

        second = resolve_room_name(g, "Alder Clump", ["south", "west"])
        assert second == "Alder Clump #2"
        assert second != first

    def test_third_disjoint_visit_gets_next_suffix(self):
        # Suffix numbering must scale past #2, not just handle the binary case.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Alder Clump")
        g.add_edge("Alder Clump", "Unknown (north from Alder Clump)", label="north")
        g.add_node("Alder Clump #2")
        g.add_edge("Alder Clump #2", "Unknown (south from Alder Clump #2)", label="south")

        third = resolve_room_name(g, "Alder Clump", ["up", "down"])
        assert third == "Alder Clump #3"

    def test_overlapping_exits_on_repeated_name_merges_into_existing_node(self):
        # A revisit where the LLM under-reports one exit must still merge into
        # the same node, not fragment into a new one — only a fully disjoint
        # exit set indicates a genuinely different room.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Alder Clump")
        g.add_edge("Alder Clump", "Unknown (north from Alder Clump)", label="north")
        g.add_edge("Alder Clump", "Unknown (east from Alder Clump)", label="east")

        resolved = resolve_room_name(g, "Alder Clump", ["north"])
        assert resolved == "Alder Clump"

    def test_node_with_no_recorded_exits_yet_is_treated_as_same_room(self):
        # A node created from a terse first visit (no exits known yet) must not
        # be treated as incompatible just because it has nothing recorded.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Alder Clump")  # no edges yet — exits unknown

        resolved = resolve_room_name(g, "Alder Clump", ["north", "east"])
        assert resolved == "Alder Clump"

    def test_no_exits_this_step_falls_back_to_name_only_match(self):
        # Bug 45's own edge case: a terse response with no exit list this step
        # gives nothing to disambiguate with — fall back to pre-fix behavior
        # (first name match) rather than guessing.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Alder Clump")
        g.add_node("Alder Clump #2")

        resolved = resolve_room_name(g, "Alder Clump", exits=[])
        assert resolved == "Alder Clump"
        resolved_default = resolve_room_name(g, "Alder Clump")
        assert resolved_default == "Alder Clump"

    def test_known_exits_splits_compound_alias_label(self):
        # Bug 48/59: a merged alias label ("south/down") must count as both
        # exits known, not just the first.
        from world_graph import known_exits
        g = nx.MultiDiGraph()
        g.add_edge("Start", "Cellar", label="south/down")
        assert known_exits(g, "Start") == {"south", "down"}

    # ── Bug 78: verbose-but-grounded room quotes must not fragment the graph ──

    def test_verbose_narrator_quote_resolves_to_existing_short_node(self):
        # A model quotes the whole sentence instead of just the room name, but
        # the room already exists under its short form.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("dingy stable")
        assert resolve_room_name(g, "You are in the dingy stable") == "dingy stable"
        assert resolve_room_name(g, "You are in a dingy stable") == "dingy stable"
        assert resolve_room_name(g, "You go south and are in the dingy stable") == "dingy stable"

    def test_verbose_narrator_quote_on_empty_graph_canonicalizes_correctly(self):
        # First visit, empty graph, nothing to match against — the fix must
        # also live in _canonicalize_room (new-node naming), not just
        # candidate matching.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        assert resolve_room_name(g, "You are in the dingy stable") == "dingy stable"
        assert resolve_room_name(g, "You go south and are in an alder ghostwood") == "alder ghostwood"

    def test_narrator_prefix_strip_preserves_outside_inside_distinction(self):
        # Must not regress: "outside"/"inside" stay part of the room name.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("outside a cave")
        g.add_node("inside a cave")
        assert resolve_room_name(g, "You are outside a cave") == "outside a cave"
        assert resolve_room_name(g, "You are inside a cave") == "inside a cave"

    def test_shared_word_different_rooms_do_not_fuzzy_merge(self):
        # Two genuinely different rooms sharing one generic word must never
        # merge via the fuzzy containment tier.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("jousting field")
        g.add_node("trampled field")
        assert resolve_room_name(g, "You are on a jousting field") == "jousting field"
        assert resolve_room_name(g, "You are on a trampled field") == "trampled field"
        # A bare, under-specified quote sharing only the generic word must not
        # silently resolve to either existing node.
        ambiguous = resolve_room_name(g, "a field")
        assert ambiguous not in ("jousting field", "trampled field")

    def test_fuzzy_containment_merges_verbose_quote_into_existing_short_node(self):
        # Real latent case found in configs/knight_orc_rooms.json's own data:
        # "Headhunters" vs "Headhunters saloon bar of the Orc's Head Inn" —
        # not a narrator-prefix artifact, genuinely needs the word-containment
        # tier (the game itself narrates the same room at two levels of detail).
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Headhunters")
        assert resolve_room_name(g, "Headhunters saloon bar of the Orc's Head Inn") == "Headhunters"

    def test_fuzzy_containment_still_respects_exit_disambiguation(self):
        # Bug 45 must not regress: a fuzzy (contains-existing-name) match is
        # fed through the exact same exit-compatibility gate as an exact match.
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        first = resolve_room_name(g, "Alder Clump", ["north", "east"])
        update_graph({"world_graph": g}, first, ["north", "east"], None, "")
        second = resolve_room_name(g, "You are in the Alder Clump maze", ["south", "west"])
        assert second == "Alder Clump #2"

    def test_fuzzy_containment_merges_when_exits_compatible(self):
        from world_graph import resolve_room_name
        g = nx.MultiDiGraph()
        g.add_node("Alder Clump")  # no exits recorded yet
        assert resolve_room_name(g, "You are in the Alder Clump maze", ["north"]) == "Alder Clump"

    def test_process_agent_step_disjoint_maze_produces_two_distinct_rooms(self, stub_child):
        # End-to-end: two consecutive steps reporting the same room name with
        # disjoint exits must leave current_room pointing at two different nodes.
        state = make_state(current_room="Alder Clump")
        state["world_graph"].add_node("Alder Clump")
        state["world_graph"].add_edge(
            "Alder Clump", "Unknown (north from Alder Clump)", label="north"
        )

        with patch("agent.execute_game_command",
                    return_value="You are in the Alder Clump. Exits: south, west."), \
             patch("agent.extract_knowledge",
                    return_value={"room": "Alder Clump", "exits": ["south", "west"]}):
            process_agent_step(state, stub_child, None)

        assert state["current_room"] == "Alder Clump #2"

    def test_in_direction_cleans_up_placeholder_and_wires_out_return(self):
        # Bug 49: "in" was missing from _REVERSE so traversal via "in" skipped
        # placeholder cleanup entirely.
        state = make_state(current_room="Juniper Scrubland")
        state["world_graph"].add_edge(
            "Juniper Scrubland", "Unknown (in from Juniper Scrubland)", label="in"
        )
        update_graph(state, "Cave", ["out"], "Juniper Scrubland", "in")
        assert not state["world_graph"].has_node("Unknown (in from Juniper Scrubland)")
        assert state["world_graph"].has_edge("Juniper Scrubland", "Cave")
        # The "out" exit should wire the real return edge, not create a placeholder
        assert not state["world_graph"].has_node("Unknown (out from Cave)")
        assert state["world_graph"].has_edge("Cave", "Juniper Scrubland")

    def test_uppercase_direction_action_still_cleans_placeholders_and_wires_reverse(self):
        state = make_state(current_room="Forest")
        state["world_graph"].add_edge("Forest", "Unknown (west from Forest)", label="west")
        state["world_graph"].add_edge("Clearing", "Unknown (east from Clearing)", label="east")
        update_graph(state, "Clearing", ["east"], "Forest", "WEST")
        assert not state["world_graph"].has_node("Unknown (west from Forest)")
        assert not state["world_graph"].has_node("Unknown (east from Clearing)")
        labels = {
            data["label"]
            for data in state["world_graph"].get_edge_data("Forest", "Clearing").values()
        }
        assert "west" in labels


def test_normalize_direction_covers_case_and_aliases():
    assert normalize_direction("Northeast") == "ne"
    assert normalize_direction("LOOK") == "look"


def test_query_entity_history_resolves_case_insensitive_entity_key():
    state = make_state(known_entities={
        "Putty Knife": {
            "status": "discovered",
            "location": "Hall",
            "verb_outcomes": {"examine": "succeeded"},
            "last_result_summary": "already checked",
        }
    })
    result = agent_tools._impl_query_entity_history(state, "putty knife")
    assert result["verb_outcomes"]["examine"] == "succeeded"
    assert result["last_result_summary"] == "already checked"


class TestStuckAnomalyLoop:
    def test_hard_failure_on_goal_action_removes_anomaly(self, stub_child):
        g = nx.MultiDiGraph()
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
        g = nx.MultiDiGraph()
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
        # The claimed room must be a literal (grounded) substring of the
        # response text (bug 74's grounding, retroactively applied to this
        # path) — a response that never names the destination can't
        # demonstrate a real room change.
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        with patch("agent.execute_game_command", return_value="You go north. You are in the Courtyard."), \
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
        mark_edge_futile(state, "Hall", "north")
        assert ("Hall", "north") in state["futile_edges"]

    def test_tags_graph_edge_data(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        mark_edge_futile(state, "Hall", "north")
        data = state["world_graph"].get_edge_data("Hall", "Unknown (north from Hall)")
        assert data is not None
        # MultiDiGraph: get_edge_data returns {key: data_dict}
        assert any(d.get("futile") for d in data.values())

    def test_no_crash_when_edge_absent(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        mark_edge_futile(state, "Hall", "north")  # edge doesn't exist — should not raise
        assert ("Hall", "north") in state["futile_edges"]


class TestFutileEdgesInNavigation:
    def test_futile_edge_skipped_prefers_non_futile(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north", futile=True)
        state["world_graph"].add_edge("Hall", "Unknown (east from Hall)", label="east")
        state["futile_edges"].add(("Hall", "north"))
        assert determine_next_action(state)[0] == "east"

    def test_all_exits_futile_falls_through_to_look(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north", futile=True)
        state["futile_edges"].add(("Hall", "north"))
        assert determine_next_action(state)[0] == "look"

    def test_known_room_exit_never_skipped(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_node("Courtyard")
        state["world_graph"].add_edge("Hall", "Courtyard", label="north")
        # Even if we incorrectly mark it futile, it won't be picked by the Unknown scan
        # (it's not an Unknown node — this tests the Unknown filter still works)
        assert determine_next_action(state)[0] == "look"


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

    def test_futile_out_direction_marked_after_hard_failure(self, stub_child):
        """P005/P006 regression: "out" is a real direction (see _REVERSE), but was
        missing from _DIRECTIONS, so this guard never fired for it and a failed
        "out" was never marked futile — it kept getting re-offered indefinitely."""
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (out from Hall)", label="out")
        with patch("agent.execute_game_command", return_value="You can't go that way."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert ("Hall", "out") in state["futile_edges"]
        # And the Unknown-exit scan must actually skip it afterward.
        assert determine_next_action(state)[0] != "out"

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


class TestNavBlacklistMarkedOnStep:
    def _goal_state(self):
        g = nx.MultiDiGraph()
        g.add_node("Hall")
        g.add_node("Courtyard")
        g.add_edge("Hall", "Courtyard", label="north")
        return make_state(
            current_room="Hall",
            inventory=["key"],
            active_goal={"room": "Courtyard", "target": "door", "solution": "key"},
            world_graph=g,
            visited_rooms={"Courtyard"},
        )

    def test_nav_blacklist_updated_when_game_rejects_run_to(self, stub_child):
        state = self._goal_state()
        with patch.object(config, "fast_nav_command", "run to {target}"), \
             patch("agent.execute_game_command", return_value="You can't do that."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, stub_child, None)
        assert "Courtyard" in state["nav_blacklist"]
        assert state["active_goal"] is None

    def test_nav_blacklist_not_updated_when_run_to_succeeds(self, stub_child):
        state = self._goal_state()
        with patch.object(config, "fast_nav_command", "run to {target}"), \
             patch("agent.execute_game_command", return_value="You are in the Courtyard."), \
             patch("agent.extract_knowledge", return_value={"room": "Courtyard", "exits": []}):
            process_agent_step(state, stub_child, None)
        assert "Courtyard" not in state.get("nav_blacklist", set())

    def test_blacklisted_target_bypassed_on_next_step(self, stub_child):
        state = self._goal_state()
        state["nav_blacklist"] = {"Courtyard"}
        with patch.object(config, "fast_nav_command", "run to {target}"), \
             patch("agent.execute_game_command", return_value="You go north.") as mock_exec, \
             patch("agent.extract_knowledge", return_value={"room": "Courtyard", "exits": []}):
            process_agent_step(state, stub_child, None)
        mock_exec.assert_called_once_with(stub_child, "north")


# ── navigation config (_nav_command) ─────────────────────────────────────────

class TestNavCommand:
    def test_fast_template_used_when_configured(self):
        state = make_state(current_room="Hall", visited_rooms={"Courtyard"})
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert nav_command(state, "Courtyard", fast=True) == "run to Courtyard"

    def test_fast_template_not_used_for_unvisited_room(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_node("Courtyard")
        state["world_graph"].add_edge("Hall", "Courtyard", label="north")
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert nav_command(state, "Courtyard", fast=True) == "north"

    def test_full_template_used_when_configured(self):
        state = make_state(current_room="Hall", visited_rooms={"Courtyard"})
        with patch.object(config, "full_nav_command", "go to {target}"):
            assert nav_command(state, "Courtyard", fast=False) == "go to Courtyard"

    def test_falls_back_to_graph_when_no_template(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_node("Courtyard")
        state["world_graph"].add_edge("Hall", "Courtyard", label="north")
        with patch.object(config, "fast_nav_command", None):
            assert nav_command(state, "Courtyard", fast=True) == "north"

    def test_graph_fallback_returns_none_when_no_path(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        with patch.object(config, "fast_nav_command", None):
            assert nav_command(state, "Courtyard", fast=True) is None

    def test_blacklisted_target_falls_back_to_graph_even_if_visited(self):
        state = make_state(
            current_room="Hall", visited_rooms={"Courtyard"}, nav_blacklist={"Courtyard"}
        )
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_node("Courtyard")
        state["world_graph"].add_edge("Hall", "Courtyard", label="north")
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert nav_command(state, "Courtyard", fast=True) == "north"

    def test_non_blacklisted_target_still_uses_fast_template(self):
        state = make_state(
            current_room="Hall", visited_rooms={"Courtyard"}, nav_blacklist={"Elsewhere"}
        )
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert nav_command(state, "Courtyard", fast=True) == "run to Courtyard"


class TestActiveGoalNavigation:
    def _goal_state(self):
        g = nx.MultiDiGraph()
        g.add_node("Hall")
        g.add_node("Courtyard")
        g.add_edge("Hall", "Courtyard", label="north")
        return make_state(
            current_room="Hall",
            inventory=["key"],
            active_goal={"room": "Courtyard", "target": "door", "solution": "key"},
            world_graph=g,
            visited_rooms={"Courtyard"},
        )

    def test_active_goal_emits_fast_nav_when_configured_and_visited(self):
        state = self._goal_state()
        state["visited_rooms"] = {"Courtyard"}
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert determine_next_action(self._goal_state())[0] == "run to Courtyard"

    def test_active_goal_uses_graph_nav_for_unvisited_room(self):
        g = nx.MultiDiGraph()
        g.add_node("Hall")
        g.add_node("Courtyard")
        g.add_edge("Hall", "Courtyard", label="north")
        state = make_state(
            current_room="Hall",
            inventory=["key"],
            active_goal={"room": "Courtyard", "target": "door", "solution": "key"},
            world_graph=g,
        )
        with patch.object(config, "fast_nav_command", "run to {target}"):
            assert determine_next_action(state)[0] == "north"

    def test_active_goal_falls_back_to_graph_step_when_no_template(self):
        with patch.object(config, "fast_nav_command", None):
            assert determine_next_action(self._goal_state())[0] == "north"


# ── pending_npc_tasks ─────────────────────────────────────────────────────────

class TestPendingNpcTasks:
    def test_wait_command_emitted_when_task_pending_and_nothing_else(self):
        state = make_state(
            pending_npc_tasks=[
                {"npc": "Denzyl", "task_description": "fetch the spear", "wait_command": "wait for denzyl"},
            ]
        )
        assert determine_next_action(state)[0] == "wait for denzyl"

    def test_first_task_wait_command_used_when_multiple_pending(self):
        state = make_state(
            pending_npc_tasks=[
                {"npc": "Denzyl", "task_description": "fetch spear", "wait_command": "wait for denzyl"},
                {"npc": "Gripper", "task_description": "find key", "wait_command": "wait for gripper"},
            ]
        )
        assert determine_next_action(state)[0] == "wait for denzyl"

    def test_empty_pending_tasks_falls_through_to_look(self):
        assert determine_next_action(make_state())[0] == "look"

    def test_inspection_takes_priority_over_pending_task(self):
        state = make_state(
            current_inspection={"target": "sword", "sequence": ["examine"], "step_index": 0},
            known_entities={"sword": {"status": "held", "location": "Hall", "verb_outcomes": {}}},
            pending_npc_tasks=[
                {"npc": "Denzyl", "task_description": "fetch spear", "wait_command": "wait for denzyl"},
            ],
        )
        assert determine_next_action(state)[0] == "examine sword"

    def test_unknown_exit_takes_priority_over_pending_task(self):
        state = make_state(current_room="Hall")
        state["world_graph"].add_node("Hall")
        state["world_graph"].add_edge("Hall", "Unknown (north from Hall)", label="north")
        state["pending_npc_tasks"] = [
            {"npc": "Denzyl", "task_description": "fetch spear", "wait_command": "wait for denzyl"},
        ]
        assert determine_next_action(state)[0] == "north"


class TestPositionLost:
    """Bug 60: when LLM returns no room after apparent movement, force look next step."""

    def _state_with_unknown_exit(self):
        """State where determine_next_action will emit a direction (north) to explore."""
        import networkx as nx
        state = make_state()
        g = nx.MultiDiGraph()
        g.add_node("Forest")
        g.add_edge("Forest", "Unknown (north from Forest)", label="north")
        state["world_graph"] = g
        state["current_room"] = "Forest"
        return state

    def test_position_lost_set_when_direction_no_room_no_failure(self):
        state = self._state_with_unknown_exit()
        child = MagicMock()
        with patch("agent.execute_game_command", return_value="You go north and are in a cedar glade."), \
             patch("agent.extract_knowledge", return_value={"exits": ["east", "south"]}):
            process_agent_step(state, child, None)
        assert state.get("position_lost") is True

    def test_position_lost_returns_look_and_stays_set(self):
        state = make_state()
        state["position_lost"] = True
        action, reason = determine_next_action(state)
        assert action == "look"
        # P007: previously cleared unconditionally after one look, so a look that
        # itself failed to yield a room left the agent blind with no further
        # re-establishing look. Now it stays set until a room is actually resolved.
        assert state.get("position_lost") is True

    def test_position_lost_persists_across_repeated_unresolved_looks(self):
        state = make_state()
        state["position_lost"] = True
        for _ in range(_POSITION_LOST_MAX_ATTEMPTS):
            action, _ = determine_next_action(state)
            assert action == "look"
        assert state.get("position_lost") is True

    def test_position_lost_gives_up_after_max_attempts(self):
        # uninspected_objects gives a distinctive fallthrough action so this
        # doesn't coincidentally land on "look" via the unrelated final fallback.
        state = make_state(uninspected_objects=["sword"])
        state["position_lost"] = True
        for _ in range(_POSITION_LOST_MAX_ATTEMPTS):
            determine_next_action(state)
        # One more attempt beyond the cap gives up rather than looking forever.
        action, _ = determine_next_action(state)
        assert action == "take sword"
        assert state.get("position_lost") is False
        assert state.get("position_lost_attempts") == 0

    def test_position_lost_cleared_once_room_resolved_after_repeated_looks(self, stub_child):
        state = make_state(current_room="Forest")
        state["position_lost"] = True
        state["position_lost_attempts"] = 2
        with patch("agent.execute_game_command", return_value="You are in a cedar glade."), \
             patch("agent.extract_knowledge", return_value={"room": "cedar glade", "exits": []}):
            process_agent_step(state, stub_child, None)
        assert state.get("position_lost") is False
        assert state.get("position_lost_attempts") == 0

    def test_position_not_lost_when_room_extracted(self):
        state = self._state_with_unknown_exit()
        child = MagicMock()
        with patch("agent.execute_game_command", return_value="You go north and are in a cedar glade."), \
             patch("agent.extract_knowledge", return_value={"room": "cedar glade", "exits": ["south"]}):
            process_agent_step(state, child, None)
        assert not state.get("position_lost")

    def test_position_not_lost_on_soft_failure(self):
        state = self._state_with_unknown_exit()
        child = MagicMock()
        with patch("agent.execute_game_command", return_value="You can't go that way."), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, child, None)
        assert not state.get("position_lost")



    def test_config_loads_wait_for_command(self, tmp_path):
        cfg_file = tmp_path / "game.json"
        cfg_file.write_text('{"npc_commands": {"wait_for_command": "wait for {npc}"}}')
        from game_config import GameConfig
        cfg = GameConfig()
        cfg.load_from_file(cfg_file)
        assert cfg.wait_for_command == "wait for {npc}"


# ── Death detection ────────────────────────────────────────────────────────────

class TestDeathDetection:
    @pytest.mark.parametrize("text", [
        "You are dead.",
        "You have died.",
        "You were killed by the knight.",
        "killed in action",
    ])
    def test_is_death_detects_death_phrases(self, text):
        assert is_death(text)

    @pytest.mark.parametrize("text", [
        "You take the sword.",
        "You are in the courtyard.",
        "Nothing happens.",
    ])
    def test_is_death_ignores_non_death(self, text):
        assert not is_death(text)

    def test_parse_inventory_response_items(self):
        text = "You are carrying: a sword, a lantern and a rope."
        result = parse_inventory_response(text)
        assert result == ["a sword", "a lantern", "a rope"]

    def test_parse_inventory_response_nothing(self):
        text = "You are not carrying anything."
        result = parse_inventory_response(text)
        assert result == []

    def test_parse_inventory_response_no_match_returns_none(self):
        assert parse_inventory_response("You are in a dark room.") is None

    def test_parse_inventory_response_knight_orc_own_phrasing(self):
        # Bug 72: Knight Orc's real INVENTORY response — confirmed across
        # every historical run log, never once matched by _CARRYING_RE.
        text = "You own a putty knife and a hooded cloak."
        assert parse_inventory_response(text) == ["a putty knife", "a hooded cloak"]

    def test_parse_inventory_response_knight_orc_own_and_wearing(self):
        text = "You own a putty knife. You are wearing a hooded cloak."
        assert parse_inventory_response(text) == ["a putty knife", "a hooded cloak"]

    def test_parse_inventory_response_own_ignores_trailing_narration(self):
        # A false "nothing" match in unrelated trailing narration (an NPC's
        # shouted line, say) must not suppress a real positive "own" match —
        # positive extraction is tried before the broad not-carrying scan.
        text = 'You own a putty knife and a hooded cloak. \nA voice shouts, "I found nothing of value!"'
        assert parse_inventory_response(text) == ["a putty knife", "a hooded cloak"]

    def test_determine_next_action_returns_inventory_when_recheck_set(self):
        state = make_state()
        state["recheck_inventory"] = True
        action, reason = determine_next_action(state)
        assert action == "inventory"
        assert "death" in reason.lower() or "inventory" in reason.lower()

    def test_recheck_inventory_clears_after_parse(self):
        state = make_state()
        state["recheck_inventory"] = True
        state["inventory"] = ["sword"]
        state["known_entities"]["sword"] = {"status": "held", "location": None, "verb_outcomes": {}}
        response_text = "You are not carrying anything."

        with patch("agent.extract_knowledge", return_value={}), \
             patch("agent.execute_game_command", return_value=response_text):
            process_agent_step(state, child=object(), llm_instance=None)

        assert state["recheck_inventory"] is False
        assert state["inventory"] == []
        assert state["known_entities"]["sword"]["status"] == "discovered"

    def test_recheck_inventory_updates_items_to_held(self):
        state = make_state()
        state["recheck_inventory"] = True
        state["inventory"] = []
        state["known_entities"]["lantern"] = {"status": "discovered", "location": None, "verb_outcomes": {}}
        response_text = "You are carrying: a lantern."

        with patch("agent.extract_knowledge", return_value={}), \
             patch("agent.execute_game_command", return_value=response_text):
            process_agent_step(state, child=object(), llm_instance=None)

        assert "a lantern" in state["inventory"] or "lantern" in state["inventory"]
        assert state["recheck_inventory"] is False
