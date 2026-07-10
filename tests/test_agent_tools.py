"""Tests for agent_tools.py — tool-calling decide loop and run_tool_calling_step."""
import os
from unittest.mock import patch

import pytest

from env_utils import load_env_file
from tests.conftest import make_state

load_env_file()  # populate os.environ from .env before any os.environ.get calls below


# ---------------------------------------------------------------------------
# Stub tool adapter
# ---------------------------------------------------------------------------

class _StubToolAdapter:
    """Drives the decide loop through a canned sequence of tool-call batches.

    Each element in `turns` is a list of (name, call_id, input_dict) tuples
    describing the tool calls the model "returns" on that turn.  The adapter
    cycles through the sequence: the first call to start_turn returns turns[0],
    subsequent calls to continue_with_results return turns[1], turns[2], …
    """

    def __init__(self, turns):
        self._turns = turns
        self._index = 0

    def _response_for(self, calls):
        return {
            "tool_calls": [{"name": n, "id": cid, "input": inp} for n, cid, inp in calls],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def start_turn(self, system_prompt, tool_schemas, user_message):
        resp = self._response_for(self._turns[self._index])
        self._index += 1
        return resp

    def continue_with_results(self, results):
        resp = self._response_for(self._turns[self._index])
        self._index += 1
        return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_RESPONSE = "You are in the Hall. Exits: north."


def _run_step(state, adapter, game_response=_GAME_RESPONSE, run_id="test_run", step_num=1):
    from agent_tools import run_tool_calling_step
    from parse_strategies import DeterministicParseStrategy
    parse_strategy = DeterministicParseStrategy()
    with patch("agent_tools._run_game_command", return_value=game_response):
        run_tool_calling_step(state, None, adapter, parse_strategy, run_id, step_num)
    return state["game_log"][-1]


# ---------------------------------------------------------------------------
# tool_trace tests
# ---------------------------------------------------------------------------

class TestToolTrace:
    def test_tool_trace_captures_non_terminal_calls_in_order(self):
        """tool_trace records each non-terminal call's name, input, and output."""
        turns = [
            [("query_map", "c1", {"room": None})],
            [("query_entity_history", "c2", {"object": "sword"})],
            [("execute_game_command", "c3", {"command": "go north", "reason": "explore"})],
        ]
        state = make_state()
        entry = _run_step(state, _StubToolAdapter(turns))
        assert "tool_trace" in entry
        trace = entry["tool_trace"]
        assert len(trace) == 2
        assert trace[0]["tool"] == "query_map"
        assert "input" in trace[0]
        assert "output" in trace[0]
        assert trace[1]["tool"] == "query_entity_history"
        assert trace[1]["input"]["object"] == "sword"

    def test_no_tool_trace_when_execute_called_immediately(self):
        """tool_trace is absent when the model calls execute_game_command first."""
        turns = [
            [("execute_game_command", "c1", {"command": "look", "reason": "check"})],
        ]
        state = make_state()
        entry = _run_step(state, _StubToolAdapter(turns))
        assert "tool_trace" not in entry

    def test_tool_trace_contains_nudge_entry(self):
        """A _nudge entry appears in tool_trace when the tool-call cap is reached."""
        from agent_tools import _NON_TERMINAL_CALL_CAP
        # Build enough query_map calls to hit the cap, then after the nudge the
        # model complies with execute_game_command.
        pre_cap_turns = [
            [("query_map", f"c{i}", {"room": None})]
            for i in range(_NON_TERMINAL_CALL_CAP)
        ]
        # After nudge is attached, model finally calls execute_game_command.
        post_nudge_turn = [
            [("execute_game_command", "cx", {"command": "look", "reason": "nudged"})],
        ]
        turns = pre_cap_turns + post_nudge_turn
        state = make_state()
        entry = _run_step(state, _StubToolAdapter(turns))
        assert "tool_trace" in entry
        nudge_entries = [t for t in entry["tool_trace"] if t["tool"] == "_nudge"]
        assert len(nudge_entries) == 1
        assert "message" in nudge_entries[0]

    def test_tool_trace_output_matches_dispatch_result(self):
        """The output in tool_trace matches what the tool actually returned."""
        turns = [
            [("query_map", "c1", {"room": "Hall"})],
            [("execute_game_command", "c2", {"command": "go north"})],
        ]
        state = make_state(current_room="Hall")
        entry = _run_step(state, _StubToolAdapter(turns))
        assert "tool_trace" in entry
        qm = entry["tool_trace"][0]
        assert qm["tool"] == "query_map"
        # The output should be a dict with at minimum the "room" key.
        assert "room" in qm["output"]

    def test_forced_fallback_no_tool_calls_has_empty_trace(self):
        """When the model returns no tool calls the entry has no tool_trace."""
        adapter = _StubToolAdapter([[]])  # empty tool_calls list
        state = make_state()
        with patch("agent_tools._run_game_command", return_value=_GAME_RESPONSE), \
             patch("agent_tools._log_tool_loop_exhausted"):
            from agent_tools import run_tool_calling_step
            from parse_strategies import DeterministicParseStrategy
            run_tool_calling_step(
                state, None, adapter, DeterministicParseStrategy(),
                "test_run", 1,
            )
        entry = state["game_log"][-1]
        # Forced fallback: no non-terminal calls happened, so no tool_trace.
        assert "tool_trace" not in entry
        assert entry.get("forced_fallback") is True


# ---------------------------------------------------------------------------
# Bug 81: ungrounded execute_game_command target — unit tests for the
# grounding/matching helpers themselves.
# ---------------------------------------------------------------------------

class TestFuzzyObjectMatch:
    def test_exact_match(self):
        from agent_tools import _fuzzy_object_match
        assert _fuzzy_object_match("putty knife", "putty knife")
        assert _fuzzy_object_match("Putty Knife", "putty knife")

    def test_fuzzy_substring_word_match(self):
        """A single-word command target fuzzy-matches a known multi-word
        entity name (design constraint 1: 'examine knife' must match
        'putty knife')."""
        from agent_tools import _fuzzy_object_match
        assert _fuzzy_object_match("knife", "putty knife")
        assert _fuzzy_object_match("putty", "putty knife")
        assert _fuzzy_object_match("hooded cloak", "hooded cloak")
        assert _fuzzy_object_match("cloak", "hooded cloak")

    def test_no_match_for_unrelated_words(self):
        from agent_tools import _fuzzy_object_match
        assert not _fuzzy_object_match("sword", "putty knife")
        assert not _fuzzy_object_match("sword", "hooded cloak")

    def test_short_target_below_floor_does_not_match(self):
        """A very short word-set-containment candidate below the floor
        doesn't spuriously match everything."""
        from agent_tools import _fuzzy_object_match
        assert not _fuzzy_object_match("a", "a golden orb")


class TestExtractGroundingChecks:
    def test_movement_command_not_checked(self):
        from agent_tools import _extract_grounding_checks
        assert _extract_grounding_checks("north") == []
        assert _extract_grounding_checks("go north") == []

    def test_meta_command_not_checked(self):
        from agent_tools import _extract_grounding_checks
        assert _extract_grounding_checks("look") == []
        assert _extract_grounding_checks("inventory") == []
        assert _extract_grounding_checks("score") == []

    def test_single_object_verb_checked_as_room(self):
        from agent_tools import _extract_grounding_checks
        assert _extract_grounding_checks("examine sword") == [("sword", "room")]
        assert _extract_grounding_checks("take putty knife") == [("putty knife", "room")]

    def test_use_on_splits_held_and_room(self):
        from agent_tools import _extract_grounding_checks
        checks = _extract_grounding_checks("use rusty key on door")
        assert ("rusty key", "held") in checks
        assert ("door", "room") in checks

    def test_cast_on_splits_held_and_room(self):
        from agent_tools import _extract_grounding_checks
        checks = _extract_grounding_checks("cast fireball on troll")
        assert ("fireball", "held") in checks
        assert ("troll", "room") in checks

    def test_bare_use_is_either(self):
        from agent_tools import _extract_grounding_checks
        assert _extract_grounding_checks("use lever") == [("lever", "either")]


class TestFindUngroundedTarget:
    def _state(self, **overrides):
        return make_state(
            current_room="huge pile of garbage",
            known_entities={
                "putty knife": {"status": "discovered", "location": "huge pile of garbage", "verb_outcomes": {}},
                "hooded cloak": {"status": "discovered", "location": "huge pile of garbage", "verb_outcomes": {}},
            },
            **overrides,
        )

    def test_grounded_exact_target_no_nudge(self):
        from agent_tools import _find_ungrounded_target
        state = self._state()
        assert _find_ungrounded_target(state, "examine putty knife", "") is None

    def test_grounded_fuzzy_target_no_nudge(self):
        """'examine knife' fuzzy-matches the known 'putty knife' entity."""
        from agent_tools import _find_ungrounded_target
        state = self._state()
        assert _find_ungrounded_target(state, "examine knife", "") is None

    def test_hallucinated_target_flagged(self):
        """The bug's exact repro: 'sword' was never a real object here."""
        from agent_tools import _find_ungrounded_target
        state = self._state()
        result = _find_ungrounded_target(
            state, "examine sword",
            "It is rusty and blunt, barely able to carve slime off a puddle.",
        )
        assert result == ("sword", "room")

    def test_raw_text_only_match_not_flagged(self):
        """A target absent from known_entities but genuinely present in the
        raw last response text is treated as real, under-extracted, not a
        hallucination (bug 77 precedent, design constraint 3)."""
        from agent_tools import _find_ungrounded_target
        state = self._state()
        result = _find_ungrounded_target(
            state, "examine iron gate",
            "Beyond the rubble you notice an iron gate set into the wall.",
        )
        assert result is None

    def test_movement_and_meta_commands_never_flagged(self):
        from agent_tools import _find_ungrounded_target
        state = self._state()
        assert _find_ungrounded_target(state, "north", "") is None
        assert _find_ungrounded_target(state, "look", "") is None
        assert _find_ungrounded_target(state, "inventory", "") is None

    def test_pronoun_target_not_flagged(self):
        from agent_tools import _find_ungrounded_target
        state = self._state()
        assert _find_ungrounded_target(state, "examine it", "") is None

    def test_held_target_checked_against_inventory_only(self):
        """'use X on Y': X (held side) must not be satisfied by a room
        object — the two ground-truth sets aren't conflated (constraint 2)."""
        from agent_tools import _find_ungrounded_target
        state = self._state(inventory=["rusty key"])
        # 'putty knife' is a real room object, but not held — as the actor
        # of 'use', it should still be flagged.
        result = _find_ungrounded_target(state, "use putty knife on door", "")
        assert result is not None
        assert result[1] == "held"

    def test_use_on_target_grounds_against_inventory(self):
        from agent_tools import _find_ungrounded_target
        state = self._state(inventory=["rusty key"])
        assert _find_ungrounded_target(state, "use rusty key on gate", "you see a gate") is None

    def test_empty_known_entities_falls_back_to_raw_text(self):
        from agent_tools import _find_ungrounded_target
        state = make_state(current_room="Unknown Location")
        assert _find_ungrounded_target(state, "examine putty knife", "putty knife lies here") is None
        assert _find_ungrounded_target(state, "examine sword", "putty knife lies here") == ("sword", "room")


class TestBuildUngroundedNudgeMessage:
    def test_message_lists_known_objects_and_forbids_invention(self):
        from agent_tools import _build_ungrounded_nudge_message
        state = make_state(
            current_room="huge pile of garbage",
            known_entities={
                "putty knife": {"status": "discovered", "location": "huge pile of garbage", "verb_outcomes": {}},
                "hooded cloak": {"status": "discovered", "location": "huge pile of garbage", "verb_outcomes": {}},
            },
        )
        message = _build_ungrounded_nudge_message(state, "sword", "room")
        assert "sword" in message
        assert "putty knife" in message
        assert "hooded cloak" in message
        assert "don't invent new object names" in message

    def test_held_kind_message_lists_inventory_and_spellbook(self):
        from agent_tools import _build_ungrounded_nudge_message
        state = make_state(inventory=["rusty key"], spellbook=["heal"])
        message = _build_ungrounded_nudge_message(state, "wand", "held")
        assert "rusty key" in message
        assert "heal" in message
        assert "don't invent new object names" in message

    def test_message_scoped_to_current_room_not_whole_run(self):
        """The displayed candidate list is the narrower current-room set,
        not the broader whole-run set used for the grounding decision —
        showing a far-away object as 'here right now' would be misleading."""
        from agent_tools import _build_ungrounded_nudge_message
        state = make_state(
            current_room="huge pile of garbage",
            known_entities={
                "putty knife": {"status": "discovered", "location": "huge pile of garbage", "verb_outcomes": {}},
                "golden orb": {"status": "discovered", "location": "a distant tower", "verb_outcomes": {}},
            },
        )
        message = _build_ungrounded_nudge_message(state, "sword", "room")
        assert "putty knife" in message
        assert "golden orb" not in message


# ---------------------------------------------------------------------------
# Bug 81: behavior-level test — the nudge must actually fire inside the
# decide loop and prevent the hallucinated command from ever reaching
# _run_game_command, mirroring this bug's exact original scenario.
# ---------------------------------------------------------------------------

class TestUngroundedTargetNudgeBehavior:
    def test_hallucinated_object_is_nudged_not_dispatched(self):
        """Reproduces docs/bugs/81.md's exact scenario: after examining a
        'rusty and blunt' putty knife, the model hallucinates a 'sword' from
        the putty knife's own description and tries to examine it. The
        ungrounded target must be intercepted inside the decide loop — the
        model gets a corrective nudge and, in this scenario, corrects
        itself — so 'examine sword' is never sent to the game engine."""
        state = make_state()

        # Step 1: establish the room and its real objects.
        step1_turns = [
            [("execute_game_command", "c1", {"command": "look", "reason": "initial look"})],
        ]
        step1_response = (
            "You are in the huge pile of garbage. Exits lead north. "
            "You can see a putty knife and a hooded cloak."
        )
        with patch("agent_tools._run_game_command", return_value=step1_response):
            from agent_tools import run_tool_calling_step
            from parse_strategies import DeterministicParseStrategy
            run_tool_calling_step(
                state, None, _StubToolAdapter(step1_turns), DeterministicParseStrategy(),
                "test_run", 1,
            )
        assert "putty knife" in state["known_entities"]
        assert "hooded cloak" in state["known_entities"]

        # Step 2: the model first hallucinates 'sword', then — after the
        # nudge — corrects to the real object.
        step2_turns = [
            [("execute_game_command", "c2", {
                "command": "examine sword",
                "reason": "The game just described a rusty, blunt sword — I should examine it properly.",
            })],
            [("execute_game_command", "c3", {
                "command": "examine putty knife",
                "reason": "corrected: there is no sword, examine the putty knife instead",
            })],
        ]
        step2_response = "It is rusty and blunt, barely able to carve slime off a puddle."
        with patch("agent_tools._run_game_command", return_value=step2_response) as mock_run:
            from agent_tools import run_tool_calling_step
            from parse_strategies import DeterministicParseStrategy
            run_tool_calling_step(
                state, None, _StubToolAdapter(step2_turns), DeterministicParseStrategy(),
                "test_run", 2,
            )

        # The hallucinated command was never sent to the game engine — only
        # the corrected one, exactly once (one step == one real command).
        mock_run.assert_called_once_with(None, "examine putty knife")

        entry = state["game_log"][-1]
        assert entry["action"] == "examine putty knife"
        nudge_entries = [t for t in entry["tool_trace"] if t["tool"] == "_ungrounded_target_nudge"]
        assert len(nudge_entries) == 1
        assert nudge_entries[0]["target"] == "sword"
        assert "putty knife" in nudge_entries[0]["message"]
        assert "don't invent new object names" in nudge_entries[0]["message"]

    def test_grounded_command_dispatched_without_nudge(self):
        """A legitimately-named object (fuzzy-matched) is dispatched
        immediately, no nudge — the fix must not false-positive on real
        objects (bias toward permissive, per bug 81's design constraints)."""
        state = make_state()
        step1_turns = [
            [("execute_game_command", "c1", {"command": "look", "reason": "initial look"})],
        ]
        step1_response = (
            "You are in the huge pile of garbage. Exits lead north. "
            "You can see a putty knife and a hooded cloak."
        )
        with patch("agent_tools._run_game_command", return_value=step1_response):
            from agent_tools import run_tool_calling_step
            from parse_strategies import DeterministicParseStrategy
            run_tool_calling_step(
                state, None, _StubToolAdapter(step1_turns), DeterministicParseStrategy(),
                "test_run", 1,
            )

        step2_turns = [
            # Fuzzy: "knife" alone should match the known "putty knife".
            [("execute_game_command", "c2", {"command": "examine knife", "reason": "look closer"})],
        ]
        step2_response = "It is rusty and blunt."
        with patch("agent_tools._run_game_command", return_value=step2_response) as mock_run:
            from agent_tools import run_tool_calling_step
            from parse_strategies import DeterministicParseStrategy
            run_tool_calling_step(
                state, None, _StubToolAdapter(step2_turns), DeterministicParseStrategy(),
                "test_run", 2,
            )

        mock_run.assert_called_once_with(None, "examine knife")
        entry = state["game_log"][-1]
        assert "tool_trace" not in entry


# ---------------------------------------------------------------------------
# Bug 81: a small, real live-model check — proves the new tool_result content
# shape (the corrective nudge message replacing the terminal call's own
# placeholder result) round-trips cleanly through a real cloud tool-calling
# adapter's continue_with_results(), not just the stubbed adapter above.
# Forcing the exact hallucination deterministically against a live model
# isn't practical (that's what the mocked TestUngroundedTargetNudgeBehavior
# tests above are for) — this test's job is compatibility/regression-safety
# against the real API's tool_use/tool_result protocol, run over a handful
# of steps of the bug's actual scenario. Skipped automatically when no
# DeepSeek/OpenAI-compatible credentials are configured.
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")


@pytest.mark.llm
@pytest.mark.slow
def test_ungrounded_target_nudge_survives_real_tool_calling_round_trip():
    """Live smoke test (small — 3 steps) against a real cloud tool-calling
    model, replaying docs/bugs/81.md's exact scenario turn-by-turn (room with
    a putty knife + hooded cloak, "examine putty knife", then the putty
    knife's own "rusty and blunt" description) so the model's step-3
    user_message is the actual hallucination-triggering context, not just a
    fixed canned response regardless of what it chose on step 2.

    This isn't a flaky assertion that the model *must* hallucinate — some
    runs it won't (see the PR description for confirmed live reproductions
    where it did, including the bug's literal "SWORD"). Either way, the
    step must complete cleanly, and if it does pick an ungrounded target,
    the nudge/dispatch invariants the mocked tests above prove
    deterministically must also hold here against the real API protocol."""
    from llm import OPENAI_AVAILABLE, load_openai_tool_llm
    if not OPENAI_AVAILABLE:
        pytest.skip("openai package not installed (required for the DeepSeek/OpenAI tool adapter)")
    if not LLM_API_KEY:
        pytest.skip(
            "no cloud tool-calling credentials configured; export LLM_API_KEY or "
            "DEEPSEEK_API_KEY (see scripts/watch_run.py)"
        )
    base_url = "https://api.deepseek.com" if LLM_PROVIDER == "deepseek" else None
    adapter = load_openai_tool_llm(LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, base_url=base_url)

    from agent_tools import run_tool_calling_step
    from parse_strategies import DeterministicParseStrategy

    state = make_state()
    parse_strategy = DeterministicParseStrategy()

    # Each real command dispatched gets the NEXT scripted response in order —
    # step 3's user_message is genuinely built from step 2's real response
    # (the putty knife's own description), matching the bug's exact shape.
    scripted_responses = [
        "You are in the huge pile of garbage. Exits lead north. You can see a putty knife and a hooded cloak.",
        "It is rusty and blunt, barely able to carve slime off a puddle.",
        "You find nothing further of interest.",
    ]
    dispatched_commands = []

    def _fake_run(child, command):
        dispatched_commands.append(command)
        index = min(len(dispatched_commands) - 1, len(scripted_responses) - 1)
        return scripted_responses[index]

    with patch("agent_tools._run_game_command", side_effect=_fake_run):
        for step_num in range(1, 4):
            run_tool_calling_step(state, None, adapter, parse_strategy, "test_llm_run_81", step_num)

    assert len(state["game_log"]) == 3
    for entry, dispatched in zip(state["game_log"], dispatched_commands):
        assert entry["action"] == dispatched  # what got logged is what got sent, always

    step3_entry = state["game_log"][-1]
    ground_nudges = [t for t in step3_entry.get("tool_trace", []) if t["tool"] == "_ungrounded_target_nudge"]
    if ground_nudges:
        # The model picked an ungrounded target on this run (confirmed to
        # happen live — see PR description) — the nudge fired at most once,
        # and the command actually dispatched to the game is not the
        # hallucinated one it names.
        assert len(ground_nudges) == 1
        assert ground_nudges[0]["target"].lower() not in dispatched_commands[-1].lower()
