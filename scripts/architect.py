#!/usr/bin/env python3
"""Stage 2 of the dev-loop orchestrator: turn one anomaly into a fix plan.

Reads runs/orchestrator/<run_id>/anomaly_report.json, picks the
highest-severity anomaly (or a specific one via --anomaly-id), reads
relevant source files, calls Claude, and writes fix_plan.json alongside
the report.

Usage:
    python scripts/architect.py <run_id> [--anomaly-id a1] [--model MODEL]

Requires ANTHROPIC_API_KEY in environment.

The output fix_plan.json must be reviewed by a human before passing to
the Dev stage — auto-merge is intentionally not wired up yet.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

RUNS_DIR = Path("runs")
ORCHESTRATOR_DIR = RUNS_DIR / "orchestrator"

# Which source files to include per anomaly type. More specific types first;
# the fallback key "" is always appended.
_FILE_ROUTES = {
    "loop_detected":        ["agent.py", "game_config.py"],
    "futile_streak":        ["agent.py", "game_config.py"],
    "redundant_streak":     ["agent.py"],
    "malformed_extraction": ["llm.py", "game_engine.py"],
    "regression":           ["run_evaluator.py", "agent.py"],
    "agent_failure_finding":["agent.py", "game_config.py"],
    "":                     ["agent.py"],  # fallback
}

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

_SYSTEM_PROMPT = """\
You are an expert Python developer working on a text-adventure AI agent called
"Knight Orc". Your job is to analyse a single detected anomaly and produce a
concrete, safe fix plan in JSON.

The repository structure:
- agent.py          — determine_next_action (8-tier priority), process_agent_step
- game_config.py    — GameConfig singleton; failure patterns; prompt_pattern
- game_engine.py    — pexpect subprocess interface; drain_game_buffer
- llm.py            — extract_knowledge; 3-attempt retry; few-shot examples
- run_evaluator.py  — classify_run, load_strategy, merge_run_record
- configs/knight_orc.json         — game-specific config
- configs/knight_orc_strategy.json — cross-run learning: futile_edges, run_history

Key design constraints:
- "invalid" verb outcomes are permanent and may be persisted to strategy JSON.
  "blocked" outcomes are state-dependent and must NEVER be persisted.
- do not propose changes that add LLM calls to the hot path (process_agent_step).
- acceptance_criteria must be concretely observable in a future run or log file,
  not just "tests pass".
"""


def _pick_files(anomaly_type):
    files = list(_FILE_ROUTES.get(anomaly_type, []))
    for f in _FILE_ROUTES[""]:
        if f not in files:
            files.append(f)
    return files


def _read_sources(files):
    parts = []
    for fname in files:
        path = Path(fname)
        if path.exists():
            parts.append(f"### {fname}\n```python\n{path.read_text()}\n```")
        else:
            parts.append(f"### {fname}\n(file not found)")
    return "\n\n".join(parts)


def _pick_anomaly(anomalies, anomaly_id):
    if anomaly_id:
        matches = [a for a in anomalies if a["id"] == anomaly_id]
        if not matches:
            raise ValueError(f"Anomaly id '{anomaly_id}' not found in report")
        return matches[0]
    return min(anomalies, key=lambda a: _SEVERITY_ORDER.get(a["severity"], 99))


def _user_message(anomaly, sources):
    return f"""\
## Anomaly to fix

```json
{json.dumps(anomaly, indent=2)}
```

## Relevant source files

{sources}

## Task

Produce a fix_plan.json object (schema: fix_plan/v1) with these fields:
- schema: "fix_plan/v1"
- anomaly_id: the anomaly's id string
- root_cause_hypothesis: one concise sentence
- target_files: list of files to change
- change_summary: what to change and why (2-4 sentences)
- acceptance_criteria: list of 2-4 strings, each concretely observable in
  a future run log or strategy file — not "tests pass"
- risk_notes: potential regressions or edge cases to watch
- status: "proposed"

Reply with ONLY the raw JSON object, no markdown fences, no commentary.
"""


def architect(run_id, anomaly_id=None, model="claude-opus-4-8"):
    report_path = ORCHESTRATOR_DIR / run_id / "anomaly_report.json"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found — run detect_anomalies.py first", file=sys.stderr)
        sys.exit(1)

    report = json.loads(report_path.read_text())
    if not report["anomalies"]:
        print("No anomalies in report — nothing to plan.", file=sys.stderr)
        sys.exit(0)

    anomaly = _pick_anomaly(report["anomalies"], anomaly_id)
    files = _pick_files(anomaly["type"])
    sources = _read_sources(files)

    client = anthropic.Anthropic()
    print(f"Calling {model} for anomaly {anomaly['id']} ({anomaly['type']}) …")
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(anomaly, sources)}],
    )
    raw = message.content[0].text.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: model returned non-JSON:\n{raw}", file=sys.stderr)
        sys.exit(1)

    out_path = ORCHESTRATOR_DIR / run_id / "fix_plan.json"
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"Wrote {out_path}")
    print(f"Root cause: {plan.get('root_cause_hypothesis', '?')}")
    print(f"Files:      {plan.get('target_files', [])}")
    print()
    print("Review fix_plan.json before passing to the Dev stage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID matching an existing anomaly_report.json")
    parser.add_argument("--anomaly-id", help="Specific anomaly id (default: highest severity)")
    parser.add_argument("--model", default="claude-opus-4-8", help="Claude model to use (default: %(default)s)")
    args = parser.parse_args()
    architect(args.run_id, args.anomaly_id, args.model)


if __name__ == "__main__":
    main()
