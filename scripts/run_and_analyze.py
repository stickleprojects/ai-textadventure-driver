#!/usr/bin/env python3
"""Run the agent headlessly, save a timestamped log, and write an issue analysis.

Usage:
    python scripts/run_and_analyze.py [steps] [--config PATH]

Environment:
    LEVEL9_INTERPRETER  path to glklevel9 binary  (default: ./tools/glklevel9)
    LEVEL9_ROM          path to game ROM           (default: ./gamefiles/knight-orc/GAMEDAT1.DAT)
    EVAL_MODEL_PATH     path to LLM .gguf          (default: ../models/Phi-3.5-mini-instruct-Q3_K_M.gguf)

Output:
    logs/run_TIMESTAMP.json   full game log
    logs/latest_analysis.md   issue report consumed by agent_dev_loop.py

Exit code:
    0  clean run (no issues detected)
    1  issues found, or startup error
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_config import config
from agent import process_agent_step
from game_engine import start_level9
from llm import load_llm

INTERPRETER_PATH = os.environ.get("LEVEL9_INTERPRETER", "./tools/glklevel9")
ROM_PATH = os.environ.get("LEVEL9_ROM", "./gamefiles/knight-orc/GAMEDAT1.DAT")
MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")
LOG_DIR = Path("logs")


def _make_initial_state():
    return {
        "current_room": "Unknown Location",
        "inventory": [],
        "spellbook": [],
        "known_entities": {},
        "world_graph": nx.DiGraph(),
        "uninspected_objects": [],
        "current_inspection": {
            "target": None,
            "sequence": config.inspection_sequence,
            "step_index": 0,
        },
        "known_npcs": {},
        "unresolved_anomalies": {},
        "active_goal": None,
        "game_log": [],
        "is_running": False,
    }


def analyze_log(game_log):
    issues = []
    n = len(game_log)
    if n == 0:
        return [{"type": "no_steps", "description": "No steps were recorded — startup may have failed"}]

    # Empty LLM extractions
    empty = [e for e in game_log if not e["extracted"]]
    if empty:
        issues.append({
            "type": "empty_llm_extraction",
            "count": len(empty),
            "pct": round(100 * len(empty) / n),
            "description": f"LLM returned empty dict on {len(empty)}/{n} steps ({round(100*len(empty)/n)}%)",
            "example_inputs": [e["response"][:120] for e in empty[:3]],
        })

    # Loop detection fires
    loops = [e for e in game_log if e.get("loop_detected")]
    if loops:
        issues.append({
            "type": "loop_detected",
            "count": len(loops),
            "looping_actions": list({e["loop_detected"] for e in loops}),
            "description": "Agent got stuck repeating the same action",
        })

    # Timeouts / critical errors
    timeouts = [e for e in game_log if "WARNING" in e["response"] or "CRITICAL" in e["response"]]
    if timeouts:
        issues.append({
            "type": "command_timeout_or_error",
            "count": len(timeouts),
            "description": "Game commands timed out or produced critical errors",
            "examples": [{"action": e["action"], "snippet": e["response"][:200]} for e in timeouts[:3]],
        })

    # Creature words appearing in the objects list (LLM misclassification)
    misclassified = []
    for e in game_log:
        for obj in e["extracted"].get("objects", []):
            if any(w in obj.lower().split() for w in config.creature_words):
                misclassified.append({
                    "action": e["action"],
                    "object": obj,
                    "response_snippet": e["response"][:80],
                })
    if misclassified:
        issues.append({
            "type": "creature_misclassified_as_object",
            "count": len(misclassified),
            "description": "LLM put a living creature into the 'objects' list",
            "examples": misclassified[:5],
        })

    # Inspection failures (read / examine / look inside returned a refusal)
    inspection_fails = []
    for e in game_log:
        action = e["action"].lower()
        if any(action.startswith(v) for v in ("read ", "look inside ", "examine ")):
            if config.failure_pattern.search(e["response"]):
                inspection_fails.append({"action": e["action"], "response": e["response"][:100]})
    if inspection_fails:
        issues.append({
            "type": "inspection_failure",
            "count": len(inspection_fails),
            "description": "Agent tried to inspect items/creatures that refused the action",
            "examples": inspection_fails[:5],
        })

    # Stuck in a single room for most of the run
    rooms = [e["extracted"].get("room") for e in game_log if e["extracted"].get("room")]
    if rooms:
        most_common_room, count = Counter(rooms).most_common(1)[0]
        if count > n * 0.6:
            issues.append({
                "type": "stuck_in_room",
                "room": most_common_room,
                "steps_in_room": count,
                "total_steps": n,
                "description": f"Agent spent {count}/{n} steps in '{most_common_room}'",
            })

    return issues


def run(steps=50):
    LOG_DIR.mkdir(exist_ok=True)

    child, initial_text = start_level9(INTERPRETER_PATH, ROM_PATH)
    if child is None:
        issues = [{"type": "startup_error", "description": initial_text}]
        _write_report(issues, [], datetime.now().strftime("%Y%m%d_%H%M%S"), None)
        return [], issues

    llm = load_llm(MODEL_PATH)
    state = _make_initial_state()

    for _ in range(steps):
        process_agent_step(state, child, llm)
        if state["game_log"][-1].get("loop_detected"):
            break

    if child.isalive():
        child.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(state["game_log"], f, indent=2)

    issues = analyze_log(state["game_log"])
    _write_report(issues, state["game_log"], ts, log_path)
    return state["game_log"], issues


def _write_report(issues, game_log, ts, log_path):
    lines = [
        f"# Dev Run Analysis — {ts}",
        f"\n**Steps run:** {len(game_log)}  ",
        f"**Issues found:** {len(issues)}  ",
    ]
    if log_path:
        lines.append(f"**Full log:** `{log_path}`\n")

    if not issues:
        lines.append("\nNo issues detected. Agent ran cleanly.\n")
    else:
        for issue in issues:
            lines.append(f"\n## {issue['type']}")
            lines.append(f"{issue['description']}\n")
            for k, v in issue.items():
                if k not in ("type", "description"):
                    lines.append(f"- `{k}`: {json.dumps(v)}")

    lines += [
        "\n---",
        "## Files to edit",
        "- `configs/knight_orc.json` — creature words, failure phrases, inspection sequence",
        "- `llm.py` — LLM prompt and `extract_knowledge()` retry logic",
        "- `agent.py` — `_is_creature()`, `determine_next_action()`, `process_agent_step()`",
        "- `game_engine.py` — pexpect interface, failure detection patterns",
    ]

    report_path = LOG_DIR / "latest_analysis.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {log_path or '(no log)'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="?", type=int, default=50, help="Number of game steps (default: 50)")
    parser.add_argument("--config", metavar="PATH", help="Path to game config JSON (default: built-in Knight Orc values)")
    args = parser.parse_args()

    if args.config:
        config.load_from_file(args.config)

    _, issues = run(steps=args.steps)
    print(json.dumps(issues, indent=2))
    sys.exit(1 if issues else 0)
