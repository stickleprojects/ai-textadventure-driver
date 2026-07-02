"""Tests for issue 27 — persist invalid verb outcomes across runs."""
import json

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scripts.watch_run import make_initial_state
from run_evaluator import load_strategy, merge_run_record
from game_config import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_record(run_id="watch_test", outcome="ambiguous", entity_verb_outcomes=None, futile_edges=None):
    return {
        "run_id": run_id,
        "outcome": outcome,
        "final_score": None,
        "futile_edges": futile_edges or [],
        "entity_verb_outcomes": entity_verb_outcomes or {},
    }


# ── load_strategy ─────────────────────────────────────────────────────────────

class TestLoadStrategyEntityVerbOutcomes:
    def test_returns_empty_dict_when_file_absent(self, tmp_path):
        result = load_strategy(tmp_path / "nonexistent.json")
        assert result["entity_verb_outcomes"] == {}

    def test_returns_empty_dict_when_key_absent_in_file(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({"futile_edges": [], "run_history": []}))
        result = load_strategy(f)
        assert result["entity_verb_outcomes"] == {}

    def test_loads_persisted_invalid_outcomes(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [],
            "entity_verb_outcomes": {"fence": {"read": "invalid", "wear": "invalid"}},
        }))
        result = load_strategy(f)
        assert result["entity_verb_outcomes"]["fence"]["read"] == "invalid"
        assert result["entity_verb_outcomes"]["fence"]["wear"] == "invalid"


# ── merge_run_record ──────────────────────────────────────────────────────────

class TestMergeRunRecordEntityVerbOutcomes:
    def test_persists_invalid_outcomes_from_run(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        record = _make_record(entity_verb_outcomes={"fence": {"read": "invalid", "wear": "invalid"}})
        merge_run_record(record, strategy_path)
        result = load_strategy(strategy_path)
        assert result["entity_verb_outcomes"]["fence"]["read"] == "invalid"
        assert result["entity_verb_outcomes"]["fence"]["wear"] == "invalid"

    def test_does_not_persist_blocked_outcomes(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        record = _make_record(entity_verb_outcomes={"door": {"open": "blocked", "unlock": "invalid"}})
        merge_run_record(record, strategy_path)
        result = load_strategy(strategy_path)
        assert "open" not in result["entity_verb_outcomes"].get("door", {})
        assert result["entity_verb_outcomes"]["door"]["unlock"] == "invalid"

    def test_unions_outcomes_across_runs(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(run_id="r1", entity_verb_outcomes={"fence": {"read": "invalid"}}), strategy_path)
        merge_run_record(_make_record(run_id="r2", entity_verb_outcomes={"fence": {"wear": "invalid"}}), strategy_path)
        result = load_strategy(strategy_path)
        assert result["entity_verb_outcomes"]["fence"]["read"] == "invalid"
        assert result["entity_verb_outcomes"]["fence"]["wear"] == "invalid"

    def test_unions_across_different_entities(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(run_id="r1", entity_verb_outcomes={"fence": {"read": "invalid"}}), strategy_path)
        merge_run_record(_make_record(run_id="r2", entity_verb_outcomes={"stone": {"eat": "invalid"}}), strategy_path)
        result = load_strategy(strategy_path)
        assert "fence" in result["entity_verb_outcomes"]
        assert "stone" in result["entity_verb_outcomes"]

    def test_run_with_no_entity_verb_outcomes_field(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        record = {"run_id": "r1", "outcome": "ambiguous", "final_score": None, "futile_edges": []}
        merge_run_record(record, strategy_path)
        result = load_strategy(strategy_path)
        assert result["entity_verb_outcomes"] == {}

    def test_entity_verb_outcomes_written_to_file(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(entity_verb_outcomes={"fence": {"read": "invalid"}}), strategy_path)
        # entity_verb_outcomes now lives in the items sidecar, not the main strategy file
        items_path = tmp_path / "strategy_items.json"
        on_disk = json.loads(items_path.read_text())
        assert on_disk["fence"]["read"] == "invalid"


# ── make_initial_state startup load ──────────────────────────────────────────

class TestMakeInitialStateLoadsVerbOutcomes:
    def test_known_entities_empty_when_no_strategy(self, tmp_path):
        state = make_initial_state(strategy_path=tmp_path / "nonexistent.json")
        assert state["known_entities"] == {}

    def test_pre_populates_known_entities_from_strategy(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [],
            "entity_verb_outcomes": {"fence": {"read": "invalid", "wear": "invalid"}},
        }))
        state = make_initial_state(strategy_path=f)
        assert "fence" in state["known_entities"]
        assert state["known_entities"]["fence"]["verb_outcomes"]["read"] == "invalid"
        assert state["known_entities"]["fence"]["verb_outcomes"]["wear"] == "invalid"

    def test_pre_populated_entity_has_discovered_status(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [],
            "entity_verb_outcomes": {"fence": {"read": "invalid"}},
        }))
        state = make_initial_state(strategy_path=f)
        assert state["known_entities"]["fence"]["status"] == "discovered"

    def test_pre_populated_entity_has_null_location(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [],
            "entity_verb_outcomes": {"fence": {"read": "invalid"}},
        }))
        state = make_initial_state(strategy_path=f)
        assert state["known_entities"]["fence"]["location"] is None

    def test_futile_edges_still_loaded_alongside_verb_outcomes(self, tmp_path):
        f = tmp_path / "strategy.json"
        f.write_text(json.dumps({
            "futile_edges": [["Hall", "north"]],
            "run_history": [],
            "entity_verb_outcomes": {"fence": {"read": "invalid"}},
        }))
        state = make_initial_state(strategy_path=f)
        assert ("Hall", "north") in state["futile_edges"]
        assert "fence" in state["known_entities"]


# ── soft_failure_pattern: reason-qualified refusals ───────────────────────────

class TestReasonQualifiedRefusals:
    @pytest.fixture(autouse=True)
    def load_config(self):
        config.load_from_file("configs/knight_orc.json")

    def test_cant_because_is_soft_failure(self):
        assert config.soft_failure_pattern.search("you can't read this because it's too small")

    def test_cant_because_is_not_hard_failure_when_soft_matches_first(self):
        # Verifies the intent: reason-qualified refusals should be blocked, not invalid.
        # soft_failure checked before hard_failure in agent.py.
        response = "you can't read the inscription because it's too dark"
        assert config.soft_failure_pattern.search(response)

    def test_plain_cant_is_still_hard_failure(self):
        assert not config.soft_failure_pattern.search("you can't do that")
        assert config.hard_failure_pattern.search("you can't do that")

    def test_right_now_still_soft(self):
        assert config.soft_failure_pattern.search("you can't do that right now")

    def test_not_yet_still_soft(self):
        assert config.soft_failure_pattern.search("not yet")
