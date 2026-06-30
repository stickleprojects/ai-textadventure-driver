#!/usr/bin/env python3
"""Run the agent loop headlessly for N steps against the real game + LLM.
Prints a JSON findings report; exits non-zero if a loop or timeout/error was detected.

Usage:
    python scripts/watch_run.py [steps] [--config configs/knight_orc.json]
    python scripts/watch_run.py 100
    python scripts/watch_run.py 50 --config configs/my_game.json

Wire into the /loop skill:
    /loop 30m run python scripts/watch_run.py 50 and summarize any findings —
    if a loop or timeout was detected, tell me which action/step and show the raw response text
"""
import argparse
import json
import os
import sys

import networkx as nx

# Must run from project root so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_config import config
from agent import process_agent_step
from game_engine import start_level9
from llm import load_llm

INTERPRETER_PATH = os.environ.get("LEVEL9_INTERPRETER", "./tools/glklevel9")
ROM_PATH = os.environ.get("LEVEL9_ROM", "./gamefiles/knight-orc/GAMEDAT1.DAT")
MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")


def make_initial_state():
    return {
        "current_room": "Unknown Location",
        "inventory": [],
        "spellbook": [],
        "known_entities": {},
        "world_graph": nx.DiGraph(),
        "uninspected_objects": [],
        "current_inspection": {"target": None, "sequence": config.inspection_sequence, "step_index": 0},
        "known_npcs": {},
        "unresolved_anomalies": {},
        "active_goal": None,
        "game_log": [],
        "is_running": False,
    }


def run(steps=50):
    child, initial_text = start_level9(INTERPRETER_PATH, ROM_PATH)
    if child is None:
        return [{"step": 0, "type": "startup_error", "message": initial_text}]

    llm = load_llm(MODEL_PATH)
    state = make_initial_state()
    findings = []

    for i in range(steps):
        process_agent_step(state, child, llm)
        last = state["game_log"][-1]

        if "WARNING" in last["response"] or "CRITICAL" in last["response"]:
            findings.append({
                "step": i,
                "type": "timeout_or_error",
                "action": last["action"],
                "response": last["response"],
            })

        if last.get("loop_detected"):
            findings.append({
                "step": i,
                "type": "loop",
                "action": last["loop_detected"],
                "recent_actions": [e["action"] for e in state["game_log"][-10:]],
            })
            break

    if child.isalive():
        child.close()

    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="?", type=int, default=50, help="Number of game steps (default: 50)")
    parser.add_argument("--config", metavar="PATH", help="Path to game config JSON (default: built-in Knight Orc values)")
    args = parser.parse_args()

    if args.config:
        config.load_from_file(args.config)

    findings = run(steps=args.steps)
    print(json.dumps(findings, indent=2))
    sys.exit(1 if findings else 0)
