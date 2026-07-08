"""Shape run data into the anomaly_report/v1 contract (see project memory
"plan-dev-orchestrator" for the full JSON contract and pipeline design).

log_analyzer.analyze_log() already detects loops, hallucinated rooms, timeouts,
misclassified creatures, inspection failures, blocked-verb rate, and stuck-in-room.
This module reuses that wholesale and adds only what it doesn't cover:

- utility streaks: N+ consecutive steps tagged `futile`/`redundant` regardless of
  whether the action text repeats. loop_detected (exact-repeat based) and
  blocked_verb_rate (same-action-repeated based) both miss an agent that bounces
  between different failing verbs/targets without making progress.
- productive_thrash: agent visits only a small set of rooms over a long window,
  all tagged `productive` because the room does change each step. Not caught by
  loop_detected (actions differ) or utility streaks (utility is productive).
- regression: this run's final_score is below the best score in the accumulated
  strategy file's run_history.

Imported by scripts/detect_anomalies.py and tested in tests/test_anomaly_detector.py.
"""
import re

_STREAK_THRESHOLD = 4
_THRASH_WINDOW = 20       # sliding window size in steps
_THRASH_UNIQUE_ROOMS = 4  # ≤ this many distinct rooms in a window = thrash

def _norm_room(entry):
    """Normalise extracted room name for thrash detection.

    Mirrors the bug-57 fix in agent._short_room_name: truncate at the first
    comma/semicolon, strip trailing period, strip leading preposition/article.
    Applied here so the detector works correctly on both old (fragmented) and
    new (already-canonicalised) logs.
    """
    room = (entry.get("extracted") or {}).get("room")
    if not room:
        return None
    room = re.split(r"[,;]", room, maxsplit=1)[0].rstrip(".")
    room = re.sub(r"^(in|on|at)\s+", "", room, flags=re.IGNORECASE)
    room = re.sub(r"^(a|an|the)\s+", "", room, flags=re.IGNORECASE)
    return room.strip().lower() or None


_SEVERITY = {
    "productive_thrash": "medium",
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


def _productive_thrash(game_log):
    """Detect windows where the agent visits only a small set of rooms repeatedly.

    Each step is `productive` (room changes), so utility streaks and loop_detected
    both miss this. Uses a sliding window: if ≤ _THRASH_UNIQUE_ROOMS distinct rooms
    appear across _THRASH_WINDOW consecutive steps, those steps are in thrash.
    Only reports sustained regions (≥ _THRASH_WINDOW steps) so brief transits
    through a small area don't trigger. Skips runs where the whole map is genuinely
    small (total distinct rooms ≤ _THRASH_UNIQUE_ROOMS).
    """
    if len(game_log) < _THRASH_WINDOW:
        return []

    all_rooms = {_norm_room(e) for e in game_log} - {None}
    if len(all_rooms) <= _THRASH_UNIQUE_ROOMS:
        return []  # map is genuinely small — oscillation is unavoidable

    # Mark every step that falls inside a triggering window
    in_thrash = [False] * len(game_log)
    for i in range(_THRASH_WINDOW, len(game_log) + 1):
        window = game_log[i - _THRASH_WINDOW:i]
        rooms = {_norm_room(e) for e in window} - {None}
        if len(rooms) <= _THRASH_UNIQUE_ROOMS:
            for j in range(i - _THRASH_WINDOW, i):
                in_thrash[j] = True

    # Collect contiguous thrash regions that are long enough to report
    anomalies = []
    start = None
    for i, flagged in enumerate(in_thrash):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            if i - start >= _THRASH_WINDOW:
                anomalies.append(_thrash_anomaly(game_log, start, i - 1))
            start = None
    if start is not None and len(game_log) - start >= _THRASH_WINDOW:
        anomalies.append(_thrash_anomaly(game_log, start, len(game_log) - 1))
    return anomalies


def _thrash_anomaly(game_log, start, end):
    slice_ = game_log[start:end + 1]
    rooms = sorted({_norm_room(e) for e in slice_} - {None})
    label = ", ".join(rooms[:3]) + ("…" if len(rooms) > 3 else "")
    return {
        "type": "productive_thrash",
        "severity": "medium",
        "step_range": [start + 1, end + 1],
        "summary": (
            f"{end - start + 1} steps visiting only {len(rooms)} room(s) "
            f"(steps {start + 1}–{end + 1}): {label}"
        ),
        "evidence": {
            "unique_rooms": rooms,
            "room_count": len(rooms),
        },
    }


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


def _unrecognized_failure_responses(game_log):
    """Flag "take X" responses agent.py couldn't classify as a known failure or
    a confirmed success (see agent.py's `unrecognized_failure` game_log field).
    One anomaly per unique (target, response) pair, deduped across repeats in
    this run — evidence only, no hard/soft judgement here (that's
    scripts/classify_failure_patterns.py's job, run separately)."""
    seen = {}
    for i, entry in enumerate(game_log):
        uf = entry.get("unrecognized_failure")
        if not uf:
            continue
        key = (uf["target"], uf["response"])
        seen.setdefault(key, []).append(i + 1)
    return [
        {
            "type": "unrecognized_failure_response",
            "severity": "medium",
            "step_range": [steps[0], steps[-1]],
            "summary": (
                f"'take {target}' -> {response!r} not in any failure pattern, "
                f"item never confirmed added to inventory ({len(steps)}x)"
            ),
            "evidence": {"target": target, "response": response, "occurrences": len(steps)},
        }
        for (target, response), steps in seen.items()
    ]


def detect_anomalies(game_log, run_record, strategy, extra_findings=None):
    """Return the anomaly list (with ids assigned) for the anomaly_report/v1 contract.

    extra_findings — pre-built findings from outside the deterministic
    detectors above (e.g. agent_tools.py's request_capability/
    tool_loop_exhausted, read from runs/orchestrator/<run_id>/
    capability_requests.json by detect_anomalies.py's detect()). Already
    shaped as {"type","severity","step_range","summary"}; appended before id
    assignment so they get real a<N> ids like any other anomaly, and an
    "evidence" stub if missing (same default llm_review.py's own findings
    get) so any future consumer that assumes the key exists doesn't break.
    """
    anomalies = _from_log_analyzer(game_log)
    anomalies += _utility_streaks(game_log, "futile")
    anomalies += _utility_streaks(game_log, "redundant")
    anomalies += _productive_thrash(game_log)
    anomalies += _regression(run_record, strategy)
    anomalies += _unrecognized_failure_responses(game_log)
    for finding in (extra_findings or []):
        finding.setdefault("evidence", {"source": "agent_tools"})
        anomalies.append(finding)
    for idx, anomaly in enumerate(anomalies, start=1):
        anomaly["id"] = f"a{idx}"
    return anomalies


def build_report(run_id, log_path, run_record_path, game_log, run_record, strategy, extra_findings=None):
    """Build the full anomaly_report/v1 dict, ready to json.dump()."""
    return {
        "schema": "anomaly_report/v1",
        "run_id": run_id,
        "log_path": str(log_path),
        "run_record_path": str(run_record_path),
        "outcome": run_record.get("outcome"),
        "anomalies": detect_anomalies(game_log, run_record, strategy, extra_findings=extra_findings),
    }
