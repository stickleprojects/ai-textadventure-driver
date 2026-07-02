#!/usr/bin/env python3
"""Stage 1b of the dev-loop orchestrator: LLM-based log review.

Reads the game log, condenses it into a step-by-step table, and asks Claude
to identify behaviour problems the deterministic detectors in detect_anomalies.py
would miss — e.g. lost position after failed room extraction, navigation that
contradicts stated goals, repetitive patterns not caught by exact-match detectors.

Findings are appended to the existing anomaly_report.json (written by
detect_anomalies.py) so architect.py sees all anomalies in one place.

Usage:
    python scripts/llm_review.py <run_id> [--model MODEL]
    python scripts/llm_review.py watch_20260702_152816

Requires ANTHROPIC_API_KEY in environment.
Requires detect_anomalies.py to have already run (anomaly_report.json must exist).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")
ORCHESTRATOR_DIR = RUNS_DIR / "orchestrator"

_MAX_STEPS_IN_PROMPT = 200  # truncate very long runs to keep token cost reasonable
_DEFAULT_MODEL = "claude-sonnet-4-6"


def _condense_log(game_log):
    """Return a compact markdown table of the log for the LLM prompt."""
    rows = ["| step | action | room | utility | response (first 80 chars) |",
            "|------|--------|------|---------|--------------------------|"]
    for i, entry in enumerate(game_log[:_MAX_STEPS_IN_PROMPT], start=1):
        room = (entry.get("extracted") or {}).get("room") or "—"
        resp = (entry.get("response") or "").replace("\n", " ")[:80]
        rows.append(f"| {i} | {entry.get('action', '')} | {room} | {entry.get('utility', '')} | {resp} |")
    if len(game_log) > _MAX_STEPS_IN_PROMPT:
        rows.append(f"| … | *(truncated — {len(game_log)} steps total)* | | | |")
    return "\n".join(rows)


def _system_prompt():
    return (
        "You are an expert reviewer of an autonomous AI agent that plays the text adventure "
        "game Knight Orc (Level 9 Computing). The agent uses an LLM to extract room names, "
        "exits, and objects from game responses, then navigates using a NetworkX world graph.\n\n"
        "Your job is to identify agent BEHAVIOUR problems from a game log. Focus on:\n"
        "- The agent sending the same action repeatedly without progress\n"
        "- The agent appearing to lose track of its location (room is '—' after a move)\n"
        "- Navigation that contradicts the room sequence (e.g. going north but arriving in "
        "the same or a previously-visited room)\n"
        "- The agent failing to act on visible objects or obvious opportunities\n"
        "- Patterns of futile or redundant actions the deterministic detectors may have missed\n\n"
        "Do NOT flag things the agent is doing correctly. Do NOT flag individual LLM extraction "
        "misses unless they cause a visible downstream problem in the action sequence.\n\n"
        "Respond with a JSON array of findings. Each finding must have:\n"
        '  "type": short snake_case label (e.g. "lost_position", "nav_contradiction")\n'
        '  "severity": "high", "medium", or "low"\n'
        '  "step_range": [first_step, last_step] (1-indexed, inclusive)\n'
        '  "summary": one sentence describing the problem and why it matters\n\n'
        "If you find no problems, return an empty array []. Return ONLY the JSON array, "
        "no prose before or after it."
    )


def _user_message(run_id, log_table, existing_anomaly_types):
    known = ", ".join(existing_anomaly_types) if existing_anomaly_types else "none"
    return (
        f"Run ID: {run_id}\n"
        f"Anomalies already detected by deterministic checks: {known}\n\n"
        f"Game log:\n\n{log_table}\n\n"
        "Identify any behaviour problems NOT already covered by the listed anomalies."
    )


def _parse_findings(raw):
    """Extract a JSON array from the LLM response, handling common wrapping patterns.

    Tries in order:
    1. Direct parse (clean response)
    2. Strip markdown code fences (```json ... ```)
    3. Locate the first '[' and last ']' and parse that substring
    4. If the response is truncated mid-array, close the last complete object and array

    Returns the list on success, None on failure.
    """
    import re as _re

    def _try(text):
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else None
        except json.JSONDecodeError:
            return None

    # 1. Direct
    result = _try(raw)
    if result is not None:
        return result

    # 2. Strip markdown fences
    stripped = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
    stripped = _re.sub(r"```\s*$", "", stripped, flags=_re.MULTILINE).strip()
    result = _try(stripped)
    if result is not None:
        return result

    # 3. Extract the outermost [...] span
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        result = _try(raw[start:end + 1])
        if result is not None:
            return result

    # 4. Truncated mid-array — close the last complete object and the array
    if start != -1:
        candidate = raw[start:]
        # Drop any incomplete trailing object (after the last complete '}')
        last_close = candidate.rfind("}")
        if last_close != -1:
            result = _try(candidate[:last_close + 1] + "]")
            if result is not None:
                return result

    return None


def llm_review(run_id, model=_DEFAULT_MODEL):
    """Run LLM log review for run_id and append findings to anomaly_report.json.

    Returns the list of new findings added (may be empty).
    Raises SystemExit on missing files or API errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    log_path = LOG_DIR / f"{run_id}.json"
    report_path = ORCHESTRATOR_DIR / run_id / "anomaly_report.json"

    for path in (log_path, report_path):
        if not path.exists() or path.stat().st_size == 0:
            print(f"ERROR: {path} is missing or empty — run detect_anomalies.py first.", file=sys.stderr)
            sys.exit(2)

    game_log = json.loads(log_path.read_text())
    report = json.loads(report_path.read_text())
    existing_types = [a["type"] for a in report.get("anomalies", [])]

    log_table = _condense_log(game_log)

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_system_prompt(),
            messages=[{"role": "user", "content": _user_message(run_id, log_table, existing_types)}],
        )
    except anthropic.AuthenticationError:
        print(
            "ERROR: Anthropic API authentication failed. Check ANTHROPIC_API_KEY "
            "(watch for stray characters from terminal paste, e.g. '[[200~' prefix).",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.BadRequestError as exc:
        print(f"ERROR: API rejected the request (400): {exc.message}", file=sys.stderr)
        sys.exit(1)

    if message.stop_reason == "max_tokens":
        print("WARNING: LLM response was truncated (hit max_tokens). Attempting partial parse.", file=sys.stderr)

    raw = message.content[0].text.strip()
    findings = _parse_findings(raw)
    if findings is None:
        print(f"ERROR: LLM returned unparseable output:\n{raw[:500]}", file=sys.stderr)
        sys.exit(1)

    # Assign IDs continuing from where detect_anomalies left off
    next_id = len(report["anomalies"]) + 1
    new_anomalies = []
    for finding in findings:
        finding["id"] = f"a{next_id}"
        finding.setdefault("evidence", {"source": "llm_review"})
        new_anomalies.append(finding)
        next_id += 1

    report["anomalies"].extend(new_anomalies)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"{len(new_anomalies)} LLM finding(s) added to {report_path}", file=sys.stderr)
    for a in new_anomalies:
        print(f"  [{a['severity']}] {a['id']} {a['type']}: {a['summary']}", file=sys.stderr)

    return new_anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID (matches logs/<run_id>.json)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Claude model to use (default: %(default)s)")
    args = parser.parse_args()

    findings = llm_review(args.run_id, model=args.model)
    sys.exit(0 if not findings else 1)


if __name__ == "__main__":
    main()
