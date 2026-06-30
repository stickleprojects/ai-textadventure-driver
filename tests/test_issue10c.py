"""Tests for issue 10c — cross-run learning: runs/ persistence, strategy merge, startup load."""
import json

import pytest

# Import via sys.path manipulation that watch_run itself uses
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scripts.watch_run import _write_run_record, RUNS_DIR
from tests.conftest import make_state


# ── _write_run_record ─────────────────────────────────────────────────────────

class TestWriteRunRecord:
    def test_creates_file_in_runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = make_state(current_score=5, max_score=1000)
        path = _write_run_record("watch_test", "score_improved", state, steps=50)
        assert path.exists()
        assert path.name == "watch_test.json"

    def test_record_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = make_state(current_score=5, max_score=1000)
        state["futile_edges"] = {("Hall", "north"), ("Hall", "south")}
        _write_run_record("watch_test", "score_improved", state, steps=42)
        record = json.loads((tmp_path / "runs" / "watch_test.json").read_text())
        assert record["run_id"] == "watch_test"
        assert record["outcome"] == "score_improved"
        assert record["final_score"] == 5
        assert record["max_score"] == 1000
        assert record["steps"] == 42
        assert sorted(record["futile_edges"]) == [["Hall", "north"], ["Hall", "south"]]

    def test_futile_edges_empty_list_when_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = make_state()
        _write_run_record("watch_test", "ambiguous", state, steps=10)
        record = json.loads((tmp_path / "runs" / "watch_test.json").read_text())
        assert record["futile_edges"] == []

    def test_none_scores_written_as_null(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = make_state()
        _write_run_record("watch_test", "ambiguous", state, steps=10)
        record = json.loads((tmp_path / "runs" / "watch_test.json").read_text())
        assert record["final_score"] is None
        assert record["max_score"] is None

    def test_runs_dir_created_if_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "runs").exists()
        _write_run_record("watch_test", "ambiguous", make_state(), steps=5)
        assert (tmp_path / "runs").exists()
