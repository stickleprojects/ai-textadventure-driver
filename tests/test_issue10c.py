"""Tests for issue 10c — cross-run learning: runs/ persistence, strategy merge, startup load."""
import json

import pytest

# Import via sys.path manipulation that watch_run itself uses
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scripts.watch_run import _write_run_record, make_initial_state, RUNS_DIR
from run_evaluator import load_strategy, merge_run_record
from tests.conftest import make_state


# ── _write_run_record ─────────────────────────────────────────────────────────

def _make_record(run_id="watch_test", outcome="score_improved", final_score=5,
                 max_score=1000, steps=50, futile_edges=None):
    return {
        "run_id": run_id,
        "outcome": outcome,
        "final_score": final_score,
        "max_score": max_score,
        "steps": steps,
        "futile_edges": futile_edges or [],
    }


class TestWriteRunRecord:
    def test_creates_file_in_runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _write_run_record(_make_record())
        assert path.exists()
        assert path.name == "watch_test.json"

    def test_record_fields_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        record = _make_record(futile_edges=[["Hall", "north"], ["Hall", "south"]])
        _write_run_record(record)
        on_disk = json.loads((tmp_path / "runs" / "watch_test.json").read_text())
        assert on_disk == record

    def test_runs_dir_created_if_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "runs").exists()
        _write_run_record(_make_record())
        assert (tmp_path / "runs").exists()


# ── load_strategy ─────────────────────────────────────────────────────────────

class TestLoadStrategy:
    def test_returns_empty_defaults_when_file_absent(self, tmp_path):
        result = load_strategy(tmp_path / "nonexistent.json")
        assert result["futile_edges"] == set()
        assert result["run_history"] == []

    def test_loads_futile_edges_as_set_of_tuples(self, tmp_path):
        strategy_file = tmp_path / "strategy.json"
        strategy_file.write_text(json.dumps({
            "futile_edges": [["Hall", "north"], ["Courtyard", "east"]],
            "run_history": [],
        }))
        result = load_strategy(strategy_file)
        assert ("Hall", "north") in result["futile_edges"]
        assert ("Courtyard", "east") in result["futile_edges"]

    def test_loads_run_history(self, tmp_path):
        strategy_file = tmp_path / "strategy.json"
        strategy_file.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [{"run_id": "watch_001", "outcome": "ambiguous", "final_score": None}],
        }))
        result = load_strategy(strategy_file)
        assert len(result["run_history"]) == 1
        assert result["run_history"][0]["run_id"] == "watch_001"


# ── merge_run_record ──────────────────────────────────────────────────────────

class TestMergeRunRecord:
    def test_creates_strategy_file_when_absent(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(), strategy_path)
        assert strategy_path.exists()

    def test_unions_futile_edges_across_runs(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(futile_edges=[["Hall", "north"]]), strategy_path)
        merge_run_record(_make_record(run_id="watch_002", futile_edges=[["Hall", "south"]]), strategy_path)
        result = load_strategy(strategy_path)
        assert ("Hall", "north") in result["futile_edges"]
        assert ("Hall", "south") in result["futile_edges"]

    def test_no_duplicate_futile_edges(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(futile_edges=[["Hall", "north"]]), strategy_path)
        merge_run_record(_make_record(run_id="watch_002", futile_edges=[["Hall", "north"]]), strategy_path)
        result = load_strategy(strategy_path)
        assert len(result["futile_edges"]) == 1

    def test_appends_to_run_history(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(run_id="watch_001"), strategy_path)
        merge_run_record(_make_record(run_id="watch_002", outcome="ambiguous"), strategy_path)
        result = load_strategy(strategy_path)
        ids = [r["run_id"] for r in result["run_history"]]
        assert ids == ["watch_001", "watch_002"]

    def test_only_summary_fields_in_history(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record(), strategy_path)
        result = load_strategy(strategy_path)
        entry = result["run_history"][0]
        assert set(entry.keys()) == {"run_id", "outcome", "final_score"}


# ── make_initial_state startup load ──────────────────────────────────────────

class TestMakeInitialStateLoadsStrategy:
    def test_futile_edges_loaded_from_strategy(self, tmp_path):
        strategy_file = tmp_path / "strategy.json"
        strategy_file.write_text(json.dumps({
            "futile_edges": [["Hall", "north"]],
            "run_history": [],
        }))
        state = make_initial_state(strategy_path=strategy_file)
        assert ("Hall", "north") in state["futile_edges"]

    def test_empty_futile_edges_when_no_strategy(self, tmp_path):
        state = make_initial_state(strategy_path=tmp_path / "nonexistent.json")
        assert state["futile_edges"] == set()


# ── world graph persistence ───────────────────────────────────────────────────

def _make_record_with_graph(run_id="watch_test", nodes=None, edges=None, **kwargs):
    r = _make_record(run_id=run_id, **kwargs)
    r["world_graph"] = {
        "nodes": nodes or [],
        "edges": edges or [],
    }
    return r


class TestWorldGraphPersistence:
    def test_nodes_written_to_strategy(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(
            _make_record_with_graph(nodes=["Hall", "Courtyard"]),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        assert set(result["world_graph"]["nodes"]) == {"Hall", "Courtyard"}

    def test_edges_written_to_strategy(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(
            _make_record_with_graph(
                nodes=["Hall", "Courtyard"],
                edges=[["Hall", "Courtyard", "south"]],
            ),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        assert ["Hall", "Courtyard", "south"] in result["world_graph"]["edges"]

    def test_nodes_unioned_across_runs(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record_with_graph(nodes=["Hall"]), strategy_path)
        merge_run_record(
            _make_record_with_graph(run_id="watch_002", nodes=["Hall", "Courtyard"]),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        assert set(result["world_graph"]["nodes"]) == {"Hall", "Courtyard"}

    def test_no_duplicate_nodes(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(_make_record_with_graph(nodes=["Hall"]), strategy_path)
        merge_run_record(
            _make_record_with_graph(run_id="watch_002", nodes=["Hall"]),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        assert result["world_graph"]["nodes"].count("Hall") == 1

    def test_edge_labels_merged_on_alias(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(
            _make_record_with_graph(
                nodes=["Hall", "Cellar"],
                edges=[["Hall", "Cellar", "south"]],
            ),
            strategy_path,
        )
        merge_run_record(
            _make_record_with_graph(
                run_id="watch_002",
                nodes=["Hall", "Cellar"],
                edges=[["Hall", "Cellar", "down"]],
            ),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        edge_labels = {(u, v): lbl for u, v, lbl in result["world_graph"]["edges"]}
        assert "south" in edge_labels[("Hall", "Cellar")]
        assert "down" in edge_labels[("Hall", "Cellar")]

    def test_unknown_nodes_excluded_from_strategy(self, tmp_path):
        strategy_path = tmp_path / "strategy.json"
        merge_run_record(
            _make_record_with_graph(
                nodes=["Hall", "Unknown (north from Hall)"],
                edges=[["Hall", "Unknown (north from Hall)", "north"]],
            ),
            strategy_path,
        )
        result = load_strategy(strategy_path)
        assert not any(n.startswith("Unknown") for n in result["world_graph"]["nodes"])
        assert not any(
            u.startswith("Unknown") or v.startswith("Unknown")
            for u, v, _ in result["world_graph"]["edges"]
        )

    def test_make_initial_state_seeds_world_graph(self, tmp_path):
        strategy_file = tmp_path / "strategy.json"
        strategy_file.write_text(json.dumps({
            "futile_edges": [],
            "run_history": [],
            "world_graph": {
                "nodes": ["Hall", "Courtyard"],
                "edges": [["Hall", "Courtyard", "south"]],
            },
        }))
        state = make_initial_state(strategy_path=strategy_file)
        g = state["world_graph"]
        assert "Hall" in g.nodes
        assert "Courtyard" in g.nodes
        assert g.has_edge("Hall", "Courtyard")
        assert g["Hall"]["Courtyard"]["label"] == "south"

    def test_make_initial_state_empty_graph_when_no_strategy(self, tmp_path):
        state = make_initial_state(strategy_path=tmp_path / "nonexistent.json")
        assert len(state["world_graph"].nodes) == 0
