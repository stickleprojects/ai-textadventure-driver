"""Unit tests for anomaly_detector — pure Python, no LLM, no game process.

Covers only the checks anomaly_detector adds on top of log_analyzer
(utility streaks, regression) plus the report-building/id-assignment glue.
log_analyzer's own detectors are already covered by test_log_analyzer.py.
"""
from anomaly_detector import build_report, detect_anomalies


def _entry(action, response="Nothing happens.", utility="informative", room="Test Room"):
    # extracted must be non-empty and room must vary across entries, or log_analyzer's
    # own empty_llm_extraction/stuck_in_room checks fire and pollute the "clean log"
    # assertions below — this module only tests what it adds on top of log_analyzer.
    return {
        "action": action,
        "response": response,
        "extracted": {"room": room},
        "utility": utility,
    }


class TestUtilityStreaks:
    def test_futile_streak_detected_across_different_actions(self):
        log = (
            [_entry("push gate", utility="futile")] * 0
            + [_entry("push gate", utility="futile"),
               _entry("pull gate", utility="futile"),
               _entry("kick gate", utility="futile"),
               _entry("climb gate", utility="futile")]
        )
        anomalies = detect_anomalies(log, {}, {})
        types = [a["type"] for a in anomalies]
        assert "futile_streak" in types
        streak = next(a for a in anomalies if a["type"] == "futile_streak")
        assert streak["step_range"] == [1, 4]
        assert streak["severity"] == "high"

    def test_short_futile_run_not_flagged(self):
        log = [_entry("push gate", utility="futile"),
               _entry("pull gate", utility="futile"),
               _entry("look", utility="informative")]
        anomalies = detect_anomalies(log, {}, {})
        assert "futile_streak" not in [a["type"] for a in anomalies]

    def test_non_futile_entry_breaks_the_streak(self):
        log = [_entry("push gate", utility="futile"),
               _entry("pull gate", utility="futile"),
               _entry("look", utility="informative"),
               _entry("kick gate", utility="futile"),
               _entry("climb gate", utility="futile")]
        anomalies = detect_anomalies(log, {}, {})
        assert "futile_streak" not in [a["type"] for a in anomalies]

    def test_redundant_streak_detected(self):
        log = [_entry("examine sword", utility="redundant"),
               _entry("examine shield", utility="redundant"),
               _entry("look", utility="redundant"),
               _entry("examine gate", utility="redundant")]
        anomalies = detect_anomalies(log, {}, {})
        types = [a["type"] for a in anomalies]
        assert "redundant_streak" in types
        streak = next(a for a in anomalies if a["type"] == "redundant_streak")
        assert streak["severity"] == "medium"

    def test_clean_log_flags_nothing(self):
        log = [_entry("look", utility="informative", room="Room A"),
               _entry("take sword", utility="productive", room="Room B")]
        anomalies = detect_anomalies(log, {}, {})
        assert anomalies == []


class TestRegression:
    def test_score_below_best_prior_flagged(self):
        run_record = {"final_score": 5, "steps": 50}
        strategy = {"run_history": [{"final_score": 20}, {"final_score": 10}]}
        anomalies = detect_anomalies([], run_record, strategy)
        types = [a["type"] for a in anomalies]
        assert "regression" in types
        reg = next(a for a in anomalies if a["type"] == "regression")
        assert reg["evidence"] == {"final_score": 5, "best_prior_score": 20}

    def test_score_matching_best_prior_not_flagged(self):
        run_record = {"final_score": 20, "steps": 50}
        strategy = {"run_history": [{"final_score": 20}]}
        anomalies = detect_anomalies([], run_record, strategy)
        assert "regression" not in [a["type"] for a in anomalies]

    def test_no_history_not_flagged(self):
        run_record = {"final_score": 5, "steps": 50}
        anomalies = detect_anomalies([], run_record, {"run_history": []})
        assert "regression" not in [a["type"] for a in anomalies]

    def test_no_score_recorded_not_flagged(self):
        run_record = {"final_score": None, "steps": 50}
        strategy = {"run_history": [{"final_score": 20}]}
        anomalies = detect_anomalies([], run_record, strategy)
        assert "regression" not in [a["type"] for a in anomalies]


class TestBuildReport:
    def test_report_shape_and_ids(self):
        log = [_entry("push gate", utility="futile"),
               _entry("pull gate", utility="futile"),
               _entry("kick gate", utility="futile"),
               _entry("climb gate", utility="futile")]
        run_record = {"outcome": "agent_failure", "final_score": None, "steps": 4}
        report = build_report("watch_test", "logs/watch_test.json", "runs/watch_test.json",
                               log, run_record, {})
        assert report["schema"] == "anomaly_report/v1"
        assert report["run_id"] == "watch_test"
        assert report["outcome"] == "agent_failure"
        assert [a["id"] for a in report["anomalies"]] == [f"a{i+1}" for i in range(len(report["anomalies"]))]

    def test_reuses_log_analyzer_loop_detected(self):
        log = [_entry("wait", utility="futile") for _ in range(10)]
        for e in log:
            e["loop_detected"] = "wait"
        report = build_report("watch_test", "l", "r", log, {"outcome": "agent_failure"}, {})
        assert "loop_detected" in [a["type"] for a in report["anomalies"]]
