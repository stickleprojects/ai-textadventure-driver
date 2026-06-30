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
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx

# Must run from project root so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_config import config
from agent import process_agent_step
from game_engine import start_level9
from llm import load_llm

# Suppress Streamlit's "missing ScriptRunContext" warning — harmless outside a
# Streamlit session; @st.cache_resource just runs without caching.
# Must be after imports: streamlit resets its logger levels at import time.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

INTERPRETER_PATH = os.environ.get("LEVEL9_INTERPRETER", "./tools/glklevel9")
ROM_PATH = os.environ.get("LEVEL9_ROM", "./gamefiles/knight-orc/GAMEDAT1.DAT")
MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")
LOG_DIR = Path("logs")


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
        "current_score": None,
        "max_score": None,
    }


def _step_summary(entry):
    """One-line summary of what happened in a step, for progress output."""
    ex = entry.get("extracted") or {}
    parts = []
    if ex.get("room"):
        parts.append(ex["room"])
    if ex.get("objects"):
        parts.append(f"objects: {', '.join(ex['objects'][:3])}")
    if ex.get("npcs"):
        parts.append(f"npcs: {', '.join(ex['npcs'][:2])}")
    if ex.get("added_to_inventory"):
        parts.append(f"took: {', '.join(ex['added_to_inventory'])}")
    if not ex:
        parts.append("(no extraction)")
    return " | ".join(parts)


def _write_log(game_log):
    """Persist the full game log to logs/watch_TIMESTAMP.json and return the path."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"watch_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(game_log, f, indent=2)
    return log_path


def run(steps=50, verbose=False):
    child, initial_text = start_level9(INTERPRETER_PATH, ROM_PATH)
    if child is None:
        print(f"ERROR: {initial_text}", file=sys.stderr)
        return [{"step": 0, "type": "startup_error", "message": initial_text}]

    if verbose:
        print(f"Started game. Running {steps} steps...", file=sys.stderr)

    llm = load_llm(MODEL_PATH)
    if verbose and llm is None:
        print("WARNING: LLM not loaded — extraction will return {}.", file=sys.stderr)

    state = make_initial_state()
    findings = []

    try:
        for i in range(steps):
            process_agent_step(state, child, llm)
            last = state["game_log"][-1]

            if verbose:
                flag = ""
                if "WARNING" in last["response"] or "CRITICAL" in last["response"]:
                    flag = "  ⚠ TIMEOUT/ERROR"
                elif last.get("loop_detected"):
                    flag = f"  ✗ LOOP ({last['loop_detected']!r})"
                print(
                    f"[{i+1:>3}/{steps}] {last['action']:<28} {_step_summary(last)}{flag}",
                    file=sys.stderr,
                )

            if "WARNING" in last["response"] or "CRITICAL" in last["response"]:
                findings.append({
                    "step": i + 1,
                    "type": "timeout_or_error",
                    "action": last["action"],
                    "response": last["response"],
                })

            if last.get("loop_detected"):
                recent = state["game_log"][-10:]
                findings.append({
                    "step": i + 1,
                    "type": "loop",
                    "action": last["loop_detected"],
                    "recent_steps": [
                        {"action": e["action"], "response": e["response"]}
                        for e in recent
                    ],
                })
                break
    except Exception as exc:
        findings.append({
            "step": len(state["game_log"]),
            "type": "crash",
            "error": str(exc),
            "last_action": state["game_log"][-1]["action"] if state["game_log"] else None,
            "last_response": state["game_log"][-1]["response"] if state["game_log"] else None,
        })
        print(f"CRASH at step {len(state['game_log'])}: {exc}", file=sys.stderr)
    finally:
        if child.isalive():
            child.close()
        log_path = _write_log(state["game_log"])
        if verbose:
            print(f"\nFull log written to {log_path}", file=sys.stderr)
            print(f"Done. {len(findings)} finding(s).", file=sys.stderr)

    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="?", type=int, default=50, help="Number of game steps (default: 50)")
    parser.add_argument("--config", metavar="PATH", help="Path to game config JSON (default: built-in Knight Orc values)")
    args = parser.parse_args()

    if args.config:
        config.load_from_file(args.config)

    findings = run(steps=args.steps, verbose=True)
    print(json.dumps(findings, indent=2))
    sys.exit(1 if findings else 0)
