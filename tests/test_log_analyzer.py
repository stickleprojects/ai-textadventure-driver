"""Unit tests for log_analyzer.analyze_log and its helper detectors.

These tests are pure Python — no LLM, no game process. Each test constructs
a minimal synthetic log and asserts that the right issue type is (or isn't)
detected. Add a new test here whenever you add a new detector to log_analyzer.py.
"""
import pytest

from log_analyzer import analyze_log, _is_suspicious_room


# ── _is_suspicious_room ───────────────────────────────────────────────────────

class TestIsSuspiciousRoom:
    def test_placeholder_current_location(self):
        assert _is_suspicious_room("current location", "take sword")

    def test_placeholder_unknown_location(self):
        assert _is_suspicious_room("unknown location", "look")

    def test_room_derived_from_action_object(self):
        # "putty knife room" contains both "putty" and "knife" from the action
        assert _is_suspicious_room("putty knife room", "take putty knife")

    def test_room_derived_partial_match_only(self):
        # Only one object word in room name — not suspicious
        assert not _is_suspicious_room("knife workshop", "take putty knife")

    def test_real_room_name_not_flagged(self):
        assert not _is_suspicious_room("Dingy Stable", "take pitchfork")

    def test_direction_words_not_flagged(self):
        # "East Wing" should not be flagged for action "go east"
        assert not _is_suspicious_room("East Wing", "go east")

    def test_short_action_words_ignored(self):
        # Words <= 3 chars are ignored so "key" won't trigger "key room"
        assert not _is_suspicious_room("key room", "take key")

    def test_none_room_not_flagged(self):
        assert not _is_suspicious_room(None, "look")

    def test_empty_string_not_flagged(self):
        assert not _is_suspicious_room("", "look")


# ── analyze_log — hallucinated_room_name ─────────────────────────────────────

def _entry(action, response, extracted):
    return {"action": action, "response": response, "extracted": extracted}


class TestHallucinatedRoomName:
    def test_detects_placeholder_room(self):
        log = [_entry("take sword", "Taken.", {"room": "current location", "added_to_inventory": ["sword"]})]
        issues = analyze_log(log)
        types = [i["type"] for i in issues]
        assert "hallucinated_room_name" in types

    def test_detects_room_from_action_words(self):
        log = [_entry("take putty knife", "Taken.", {"room": "putty knife room"})]
        issues = analyze_log(log)
        assert any(i["type"] == "hallucinated_room_name" for i in issues)

    def test_counts_multiple_hallucinations(self):
        log = [
            _entry("take sword", "Taken.", {"room": "current location"}),
            _entry("examine sword", "You examine the sword.", {"room": "sword examination room"}),
        ]
        issues = analyze_log(log)
        issue = next(i for i in issues if i["type"] == "hallucinated_room_name")
        assert issue["count"] == 2

    def test_real_room_not_flagged(self):
        log = [
            _entry("north", "You go north. You are in the Courtyard.", {"room": "Courtyard"}),
            _entry("look", "You see a stone.", {"room": "Courtyard", "objects": ["stone"]}),
        ]
        issues = analyze_log(log)
        assert not any(i["type"] == "hallucinated_room_name" for i in issues)

    def test_absent_room_not_flagged(self):
        # If LLM correctly omits room on a terse response, no issue
        log = [_entry("take sword", "Taken.", {"added_to_inventory": ["sword"]})]
        issues = analyze_log(log)
        assert not any(i["type"] == "hallucinated_room_name" for i in issues)


# ── analyze_log — empty_llm_extraction ───────────────────────────────────────

class TestEmptyExtraction:
    def test_detects_empty_dict(self):
        log = [_entry("look", "You see nothing.", {})]
        issues = analyze_log(log)
        assert any(i["type"] == "empty_llm_extraction" for i in issues)

    def test_no_false_positive_on_nonempty(self):
        log = [_entry("look", "You see a stone.", {"objects": ["stone"]})]
        issues = analyze_log(log)
        assert not any(i["type"] == "empty_llm_extraction" for i in issues)

    def test_count_and_pct(self):
        log = [
            _entry("look", "response", {}),
            _entry("look", "response", {}),
            _entry("north", "response", {"room": "Hall"}),
            _entry("look", "response", {}),
        ]
        issues = analyze_log(log)
        issue = next(i for i in issues if i["type"] == "empty_llm_extraction")
        assert issue["count"] == 3
        assert issue["pct"] == 75


# ── analyze_log — creature_misclassified_as_object ───────────────────────────

class TestCreatureMisclassified:
    def test_detects_horse_in_objects(self):
        log = [_entry("look", "A horse stands here.", {"objects": ["horse"]})]
        issues = analyze_log(log)
        assert any(i["type"] == "creature_misclassified_as_object" for i in issues)

    def test_no_false_positive_for_inanimate(self):
        log = [_entry("look", "A sword lies here.", {"objects": ["sword"]})]
        issues = analyze_log(log)
        assert not any(i["type"] == "creature_misclassified_as_object" for i in issues)


# ── analyze_log — edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_log_returns_no_steps_issue(self):
        issues = analyze_log([])
        assert issues[0]["type"] == "no_steps"

    def test_clean_run_returns_no_issues(self):
        log = [
            _entry("look", "You are in the Courtyard. Exits: north.", {"room": "Courtyard", "exits": ["north"]}),
            _entry("north", "You go north.", {"room": "Hall"}),
        ]
        assert analyze_log(log) == []
