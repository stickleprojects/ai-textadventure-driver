"""Tests for LLM request tracing (prompt + raw attempts stored in log entry)."""
from llm import extract_knowledge


def _make_fake_llm(json_text):
    def fake_llm(system_prompt, user_message):
        return {"choices": [{"text": json_text}], "usage": {}}
    return fake_llm


class TestTraceInResult:
    def test_trace_present_on_success(self):
        llm = _make_fake_llm('{"exits": ["north"]}')
        result = extract_knowledge("You are in the hall.", "look", llm)
        assert "_trace" in result

    def test_trace_contains_system_and_user(self):
        llm = _make_fake_llm('{"exits": ["north"]}')
        result = extract_knowledge("You are in the hall.", "look", llm)
        trace = result["_trace"]
        assert "system" in trace
        assert "user" in trace
        assert "look" in trace["user"]
        assert "You are in the hall." in trace["user"]

    def test_trace_records_successful_attempt(self):
        llm = _make_fake_llm('{"exits": ["north"]}')
        result = extract_knowledge("You are in the hall.", "look", llm)
        attempts = result["_trace"]["attempts"]
        assert len(attempts) == 1
        assert attempts[0]["parsed"] is True
        assert '{"exits": ["north"]}' in attempts[0]["output"]

    def test_trace_records_failed_attempt_before_success(self):
        call_count = [0]
        def flaky_llm(system_prompt, user_message):
            call_count[0] += 1
            text = "not json" if call_count[0] < 2 else '{"exits": ["south"]}'
            return {"choices": [{"text": text}], "usage": {}}

        result = extract_knowledge("You are in the forest.", "look", flaky_llm)
        attempts = result["_trace"]["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["parsed"] is False
        assert attempts[1]["parsed"] is True

    def test_no_trace_when_llm_is_none(self):
        result = extract_knowledge("You are in the hall.", "look", None)
        assert result == {}


class TestTraceInLogEntry:
    def _run_one_step(self, fake_llm_json):
        import networkx as nx
        from agent import process_agent_step

        state = {
            "current_room": "Hall",
            "inventory": [], "spellbook": [],
            "known_entities": {}, "world_graph": nx.MultiDiGraph(),
            "uninspected_objects": [],
            "current_inspection": {"target": None, "sequence": [], "step_index": 0},
            "known_npcs": {}, "unresolved_anomalies": {}, "active_goal": None,
            "game_log": [], "is_running": False, "current_score": None,
            "max_score": None, "futile_edges": set(), "pending_npc_tasks": [],
        }

        class FakeChild:
            def sendline(self, _): pass
            def expect(self, _): pass
            @property
            def before(self): return b"You are in the Hall."

        from unittest.mock import patch
        with patch("agent.execute_game_command", return_value="You are in the Hall."):
            process_agent_step(state, FakeChild(), _make_fake_llm(fake_llm_json))

        return state["game_log"][-1]

    def test_llm_trace_written_to_log_entry(self):
        entry = self._run_one_step('{"room": "Hall", "exits": ["north"]}')
        assert "llm_trace" in entry
        assert "system" in entry["llm_trace"]
        assert "user" in entry["llm_trace"]
        assert entry["llm_trace"]["attempts"][0]["parsed"] is True

    def test_no_llm_trace_when_extraction_returns_empty(self):
        entry = self._run_one_step("not json at all")
        assert "llm_trace" not in entry
