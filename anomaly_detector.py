"""Shape run data into the anomaly_report/v1 contract (see project memory
"plan-dev-orchestrator" for the full JSON contract and pipeline design).

log_analyzer.analyze_log() already detects loops, hallucinated rooms, timeouts,
misclassified creatures, inspection failures, blocked-verb rate, and stuck-in-room.
This module reuses that wholesale and adds only what it doesn't cover:

- utility streaks: N+ consecutive steps tagged `futile`/`redundant` regardless of
  whether the action text repeats. loop_detected (exact-repeat based) and
  blocked_verb_rate (same-action-repeated based) both miss an agent that bounces
  between different failing verbs/targets without making progress.
- regression: this run's final_score is below the best score in the accumulated
  strategy file's run_history.

Imported by scripts/detect_anomalies.py and tested in tests/test_anomaly_detector.py.
"""
_STREAK_THRESHOLD = 4

_SEVERITY = {
    "loop_detected": "high",
    "command_timeout_or_error": "high",
    "no_steps": "high",
    "hallucinated_room_name": "medium",
    "empty_llm_extraction": "medium",
    "creature_misclassified_as_object": "medium",
    "blocked_verb_rate": "medium",
    "stuck_in_room": "medium",
    "inspection_failure": "low",
}


def _step_range_for_issue(issue, game_log):
    """Best-effort 1-indexed [start, end] step range for a log_analyzer issue."""
    examples = issue.get("examples") or issue.get("looping_actions") or []
    action_names = set()
    for e in examples:
        if isinstance(e, dict) and "action" in e:
            action_names.add(e["action"])
        elif isinstance(e, str):
            action_names.add(e)
    if not action_names:
        return [1, len(game_log)]
    indices = [i + 1 for i, e in enumerate(game_log) if e.get("action") in action_names]
    return [min(indices), max(indices)] if indices else [1, len(game_log)]


def _from_log_analyzer(game_log):
    from log_analyzer import analyze_log

    anomalies = []
    for issue in analyze_log(game_log):
        anomalies.append({
            "type": issue["type"],
            "severity": _SEVERITY.get(issue["type"], "medium"),
            "step_range": _step_range_for_issue(issue, game_log),
            "summary": issue["description"],
            "evidence": {k: v for k, v in issue.items() if k not in ("type", "description")},
        })
    return anomalies


def _streak_anomaly(game_log, kind, start, end):
    slice_ = game_log[start:end + 1]
    return {
        "type": f"{kind}_streak",
        "severity": "high" if kind == "futile" else "medium",
        "step_range": [start + 1, end + 1],
        "summary": f"{end - start + 1} consecutive {kind} actions (steps {start + 1}-{end + 1})",
        "evidence": {
            "actions": [e["action"] for e in slice_],
            "responses": [e["response"][:120] for e in slice_],
        },
    }


def _utility_streaks(game_log, kind, threshold=_STREAK_THRESHOLD):
    anomalies = []
    run_start = None
    for i, entry in enumerate(game_log):
        if entry.get("utility") == kind:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= threshold:
                anomalies.append(_streak_anomaly(game_log, kind, run_start, i - 1))
            run_start = None
    if run_start is not None and len(game_log) - run_start >= threshold:
        anomalies.append(_streak_anomaly(game_log, kind, run_start, len(game_log) - 1))
    return anomalies


def _regression(run_record, strategy):
    scored = [h for h in strategy.get("run_history", []) if h.get("final_score") is not None]
    if not scored or run_record.get("final_score") is None:
        return []
    best_prior = max(h["final_score"] for h in scored)
    if run_record["final_score"] < best_prior:
        return [{
            "type": "regression",
            "severity": "medium",
            "step_range": [1, run_record.get("steps", 0)],
            "summary": (
                f"final_score {run_record['final_score']} is below the best prior "
                f"run's {best_prior} ({len(scored)} scored runs in history)"
            ),
            "evidence": {"final_score": run_record["final_score"], "best_prior_score": best_prior},
        }]
    return []


def detect_anomalies(game_log, run_record, strategy):
    """Return the anomaly list (with ids assigned) for the anomaly_report/v1 contract."""
    anomalies = _from_log_analyzer(game_log)
    anomalies += _utility_streaks(game_log, "futile")
    anomalies += _utility_streaks(game_log, "redundant")
    anomalies += _regression(run_record, strategy)
    for idx, anomaly in enumerate(anomalies, start=1):
        anomaly["id"] = f"a{idx}"
    return anomalies


def build_report(run_id, log_path, run_record_path, game_log, run_record, strategy):
    """Build the full anomaly_report/v1 dict, ready to json.dump()."""
    return {
        "schema": "anomaly_report/v1",
        "run_id": run_id,
        "log_path": str(log_path),
        "run_record_path": str(run_record_path),
        "outcome": run_record.get("outcome"),
        "anomalies": detect_anomalies(game_log, run_record, strategy),
    }
