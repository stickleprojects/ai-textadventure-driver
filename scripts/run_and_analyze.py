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
from datetime import datetime
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_config import config
from log_analyzer import analyze_log
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
        "world_graph": nx.MultiDiGraph(),
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
        "current_score": None,
        "max_score": None,
        "futile_edges": set(),
        "pending_npc_tasks": [],
        "visited_rooms": set(),
        "recheck_inventory": False,
    }


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
