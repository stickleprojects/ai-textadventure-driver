"""Classify the outcome of a completed agent run.

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
"""
import re

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
