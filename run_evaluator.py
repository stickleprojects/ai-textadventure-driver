"""Classify the outcome of a completed agent run and manage the cross-run strategy file.

Outcome categories
------------------
finished        — "congratulations" / "you have finished" matched
game_ended      — other end_state_pattern matched (death, score summary)
score_improved  — final_score > 0 and score was recorded during the run
agent_failure   — loop_detected, crash, or timeout_or_error finding present
ambiguous       — hit max steps, no score change, no error, no end state

Crash sub-classification (classify_finding)
-------------------------------------------
config_fix  — failure response text not matched by any existing pattern in config
python_fix  — finding type is "crash" (exception traceback present)
ambiguous   — neither of the above; needs human review

Strategy file (merge_run_record / load_strategy)
-------------------------------------------------
configs/knight_orc_strategy.json accumulates futile_edges and run_history across
runs. load_strategy() returns empty defaults if the file does not exist.
"""
import json
import re
from pathlib import Path

from game_config import config

_SCORE_RE = re.compile(r"you score\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)


def _scan_for_end_state(game_log):
    """Return (category, matched_text) for the first end-state hit, or (None, None)."""
    for entry in reversed(game_log[-20:]):
        text = entry.get("response", "")
        for category, patterns in config.end_state_patterns.items():
            for pat in patterns:
                if pat.search(text):
                    return category, text
    return None, None


def classify_run(game_log, findings, final_score=None):
    """Return one of the five outcome strings for this run.

    Parameters
    ----------
    game_log : list[dict]   — the agent's game_log list
    findings : list[dict]   — findings list from watch_run.run()
    final_score : int|None  — state["current_score"] at run end, or None
    """
    finding_types = {f.get("type") for f in findings}

    end_category, _ = _scan_for_end_state(game_log)

    if end_category == "finished":
        return "finished"

    if finding_types & {"loop", "timeout_or_error", "crash", "startup_error"}:
        return "agent_failure"

    if end_category in ("death", "score"):
        return "game_ended"

    if final_score is not None and final_score > 0:
        return "score_improved"

    return "ambiguous"


def load_strategy(strategy_path):
    """Load accumulated strategy from file. Returns empty defaults if file absent.

    Returns dict with:
        futile_edges         — set of (room, direction) tuples
        run_history          — list of {run_id, outcome, final_score} dicts
        entity_verb_outcomes — {entity_name: {verb: "invalid"}} persisted hard failures
    """
    path = Path(strategy_path)
    if not path.exists():
        return {"futile_edges": set(), "run_history": [], "entity_verb_outcomes": {}}
    with open(path) as f:
        data = json.load(f)
    return {
        "futile_edges": {tuple(e) for e in data.get("futile_edges", [])},
        "run_history": data.get("run_history", []),
        "entity_verb_outcomes": data.get("entity_verb_outcomes", {}),
    }


def merge_run_record(run_record, strategy_path):
    """Merge a completed run's data into the accumulated strategy file.

    Unions futile_edges across runs; appends a summary entry to run_history.
    Creates the file (and its parent directory) if absent.
    """
    strategy = load_strategy(strategy_path)

    new_edges = {tuple(e) for e in run_record.get("futile_edges", [])}
    strategy["futile_edges"] |= new_edges

    # Union invalid (hard-failure) verb outcomes — blocked outcomes are state-dependent
    # and must not be persisted, as they may succeed in a future run.
    for entity, verbs in run_record.get("entity_verb_outcomes", {}).items():
        existing = strategy["entity_verb_outcomes"].setdefault(entity, {})
        for verb, outcome in verbs.items():
            if outcome == "invalid":
                existing[verb] = "invalid"

    strategy["run_history"].append({
        "run_id": run_record["run_id"],
        "outcome": run_record["outcome"],
        "final_score": run_record.get("final_score"),
    })

    path = Path(strategy_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "futile_edges": [list(e) for e in sorted(strategy["futile_edges"])],
                "run_history": strategy["run_history"],
                "entity_verb_outcomes": strategy["entity_verb_outcomes"],
            },
            f,
            indent=2,
        )
    return strategy


def classify_finding(finding, game_log):
    """Sub-classify a single finding from watch_run for triage.

    Returns 'config_fix', 'python_fix', or 'ambiguous'.
    """
    if finding.get("type") == "crash":
        return "python_fix"

    response = finding.get("response") or finding.get("last_response") or ""
    if response and not config.failure_pattern.search(response):
        return "config_fix"

    return "ambiguous"
