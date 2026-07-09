"""Tests for agent_tools.py — tool-calling decide loop and run_tool_calling_step."""
from unittest.mock import patch

from tests.conftest import make_state


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
