"""Tests for issue 10.a (death/end detection) and 10.b (score tracking)."""
import re
from unittest.mock import MagicMock, patch

import pytest

from agent import _SCORE_INTERVAL, _SCORE_RE, determine_next_action, process_agent_step
from game_config import config
from run_evaluator import classify_finding, classify_run
from tests.conftest import make_state


# ── 10.a — end_state_patterns in config ──────────────────────────────────────

class TestEndStatePatterns:
    def test_death_pattern_matches(self):
        pats = config.end_state_patterns["death"]
        assert any(p.search("you have died") for p in pats)

    def test_you_are_dead_matches(self):
        pats = config.end_state_patterns["death"]
        assert any(p.search("You are dead.") for p in pats)

    def test_finished_pattern_matches_congratulations(self):
        pats = config.end_state_patterns["finished"]
        assert any(p.search("Congratulations! You have won.") for p in pats)

    def test_finished_pattern_matches_the_end(self):
        pats = config.end_state_patterns["finished"]
        assert any(p.search("The End.") for p in pats)

    def test_score_pattern_matches(self):
        pats = config.end_state_patterns["score"]
        assert any(p.search("you score 5 out of 1000") for p in pats)

    def test_normal_response_does_not_match_any(self):
        text = "You walk north into the forest."
        for pats in config.end_state_patterns.values():
            assert not any(p.search(text) for p in pats)


# ── 10.a — run_evaluator.classify_run ────────────────────────────────────────

def _log_entry(response):
    return {"action": "look", "response": response, "extracted": {}}


class TestClassifyRun:
    def test_finished_takes_priority(self):
        log = [_log_entry("Congratulations! You have won.")]
        assert classify_run(log, [], final_score=10) == "finished"

    def test_agent_failure_on_loop(self):
        log = [_log_entry("You are in the forest.")]
        findings = [{"type": "loop", "action": "look"}]
        assert classify_run(log, findings) == "agent_failure"

    def test_agent_failure_on_crash(self):
        log = [_log_entry("You are in the forest.")]
        findings = [{"type": "crash", "error": "AttributeError"}]
        assert classify_run(log, findings) == "agent_failure"

    def test_game_ended_on_death(self):
        log = [_log_entry("You have died. Game over.")]
        assert classify_run(log, []) == "game_ended"

    def test_score_improved_when_score_positive(self):
        log = [_log_entry("You are in the forest.")]
        assert classify_run(log, [], final_score=5) == "score_improved"

    def test_ambiguous_when_nothing_notable(self):
        log = [_log_entry("You are in the forest.")]
        assert classify_run(log, [], final_score=None) == "ambiguous"

    def test_ambiguous_when_score_is_zero(self):
        log = [_log_entry("You are in the forest.")]
        assert classify_run(log, [], final_score=0) == "ambiguous"

    def test_interrupted_outcome(self):
        log = [_log_entry("You are in the forest.")]
        findings = [{"type": "interrupted"}]
        assert classify_run(log, findings) == "interrupted"

    def test_interrupted_takes_priority_over_score(self):
        log = [_log_entry("You are in the forest.")]
        findings = [{"type": "interrupted"}]
        assert classify_run(log, findings, final_score=10) == "interrupted"


# ── 10.a — run_evaluator.classify_finding ────────────────────────────────────

class TestClassifyFinding:
    def test_crash_is_python_fix(self):
        finding = {"type": "crash", "error": "AttributeError: foo"}
        assert classify_finding(finding, []) == "python_fix"

    def test_unmatched_response_is_config_fix(self):
        finding = {"type": "timeout_or_error", "response": "The magic barrier blocks your way."}
        assert classify_finding(finding, []) == "config_fix"

    def test_matched_response_is_ambiguous(self):
        finding = {"type": "timeout_or_error", "response": "You can't do that."}
        assert classify_finding(finding, []) == "ambiguous"

    def test_no_response_is_ambiguous(self):
        finding = {"type": "loop", "action": "look"}
        assert classify_finding(finding, []) == "ambiguous"


# ── 10.b — score command issued every N steps ─────────────────────────────────

class TestScoreAction:
    def _state_with_steps(self, n):
        """Return a stuck state (no goals, no objects, no unknown exits) with n log entries."""
        state = make_state()
        state["game_log"] = [{"action": "look", "response": "", "extracted": {}} for _ in range(n)]
        return state

    def test_score_issued_at_interval(self):
        state = self._state_with_steps(_SCORE_INTERVAL)
        assert determine_next_action(state)[0] == "score"

    def test_score_issued_at_double_interval(self):
        state = self._state_with_steps(_SCORE_INTERVAL * 2)
        assert determine_next_action(state)[0] == "score"

    def test_score_not_issued_at_zero(self):
        state = self._state_with_steps(0)
        assert determine_next_action(state)[0] != "score"

    def test_score_not_issued_between_intervals(self):
        state = self._state_with_steps(_SCORE_INTERVAL - 1)
        assert determine_next_action(state)[0] != "score"


# ── 10.b — score parsed from response ────────────────────────────────────────

class TestScoreParsing:
    def test_score_re_matches(self):
        m = _SCORE_RE.search("you score 42 out of 1000")
        assert m and m.group(1) == "42" and m.group(2) == "1000"

    def test_score_re_case_insensitive(self):
        assert _SCORE_RE.search("You Score 5 Out Of 1000")

    def test_score_re_no_match_on_normal_text(self):
        assert not _SCORE_RE.search("You are in the forest.")

    def test_process_agent_step_parses_score(self):
        state = make_state(game_log=[{"action": "look", "response": "", "extracted": {}} for _ in range(_SCORE_INTERVAL)])
        child = MagicMock()
        child.isalive.return_value = True

        with patch("agent.execute_game_command", return_value="you score 15 out of 1000"), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, child, None)

        assert state["current_score"] == 15
        assert state["max_score"] == 1000

    def test_process_agent_step_score_in_log_entry(self):
        state = make_state(game_log=[{"action": "look", "response": "", "extracted": {}} for _ in range(_SCORE_INTERVAL)])
        child = MagicMock()

        with patch("agent.execute_game_command", return_value="you score 7 out of 1000"), \
             patch("agent.extract_knowledge", return_value={}):
            process_agent_step(state, child, None)

        last = state["game_log"][-1]
        assert last["score"] == 7

    def test_score_none_in_log_before_any_score_command(self):
        state = make_state()
        child = MagicMock()

        with patch("agent.execute_game_command", return_value="You are in the forest."), \
             patch("agent.extract_knowledge", return_value={"room": "Forest"}):
            process_agent_step(state, child, None)

        assert state["game_log"][-1]["score"] is None
