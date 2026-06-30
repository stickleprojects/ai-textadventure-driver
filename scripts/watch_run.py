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
import time
from datetime import datetime
from pathlib import Path

import networkx as nx

# Must run from project root so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_config import config
from agent import process_agent_step
from game_engine import start_level9
from llm import load_llm
from run_evaluator import classify_run, load_strategy, merge_run_record

# Suppress Streamlit's "missing ScriptRunContext" warning — harmless outside a
# Streamlit session; @st.cache_resource just runs without caching.
# Must be after imports: streamlit resets its logger levels at import time.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

INTERPRETER_PATH = os.environ.get("LEVEL9_INTERPRETER", "./tools/glklevel9")
ROM_PATH = os.environ.get("LEVEL9_ROM", "./gamefiles/knight-orc/GAMEDAT1.DAT")
MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")
LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")
STRATEGY_PATH = os.environ.get("GAME_STRATEGY", "configs/knight_orc_strategy.json")
# Cost per million tokens — set to forecast cloud API spend; 0 = local/free
LLM_INPUT_PRICE_PER_MILLION = float(os.environ.get("LLM_INPUT_PRICE_PER_MILLION", "0"))
LLM_OUTPUT_PRICE_PER_MILLION = float(os.environ.get("LLM_OUTPUT_PRICE_PER_MILLION", "0"))


def make_initial_state(strategy_path=STRATEGY_PATH):
    strategy = load_strategy(strategy_path)
    known_entities = {
        name: {"status": "discovered", "location": None, "verb_outcomes": dict(verbs)}
        for name, verbs in strategy.get("entity_verb_outcomes", {}).items()
    }
    return {
        "current_room": "Unknown Location",
        "inventory": [],
        "spellbook": [],
        "known_entities": known_entities,
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
        "futile_edges": strategy["futile_edges"],
        "pending_npc_tasks": [],
    }


def _fmt_duration(secs):
    """Format a duration in seconds as a compact human-readable string."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(secs), 3600)
    return f"{h}h{rem // 60:02d}m"


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


def _write_log(game_log, run_id):
    """Persist the full game log to logs/<run_id>.json and return the path."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(game_log, f, indent=2)
    return log_path


def _write_run_record(record):
    """Write a compact per-run summary dict to runs/<run_id>.json."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{record['run_id']}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


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

    run_id = f"watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state = make_initial_state()
    findings = []
    run_start = time.monotonic()
    step_times = []

    try:
        for i in range(steps):
            step_start = time.monotonic()
            process_agent_step(state, child, llm)
            step_times.append(time.monotonic() - step_start)
            last = state["game_log"][-1]

            if verbose:
                flag = ""
                if "WARNING" in last["response"] or "CRITICAL" in last["response"]:
                    flag = "  ⚠ TIMEOUT/ERROR"
                elif last.get("loop_detected"):
                    flag = f"  ✗ LOOP ({last['loop_detected']!r})"
                elapsed = time.monotonic() - run_start
                avg_step = sum(step_times) / len(step_times)
                eta = (steps - (i + 1)) * avg_step
                timing = f"[elapsed {_fmt_duration(elapsed)} | {avg_step:.1f}s/step | remaining {_fmt_duration(eta)}]"
                print(
                    f"[{i+1:>3}/{steps}] {last['action']:<28} {_step_summary(last)}{flag}  {timing}",
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
    except KeyboardInterrupt:
        steps_done = len(state["game_log"])
        print(f"\nInterrupted at step {steps_done} — saving logs...", file=sys.stderr)
        state["game_log"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": "INTERRUPTED",
            "response": "Run interrupted by user (Ctrl+C).",
            "extracted": {},
            "score": state.get("current_score"),
            "utility": "interrupted",
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        })
        findings.append({"step": steps_done, "type": "interrupted"})
    except Exception as exc:
        state["game_log"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": "CRASH",
            "response": str(exc),
            "extracted": {},
            "score": state.get("current_score"),
            "utility": "crash",
        })
        findings.append({
            "step": len(state["game_log"]),
            "type": "crash",
            "error": str(exc),
            "last_action": state["game_log"][-2]["action"] if len(state["game_log"]) > 1 else None,
            "last_response": state["game_log"][-2]["response"] if len(state["game_log"]) > 1 else None,
        })
        print(f"CRASH at step {len(state['game_log'])}: {exc}", file=sys.stderr)
    finally:
        if child.isalive():
            child.close()
        steps_run = len(state["game_log"])
        outcome = classify_run(state["game_log"], findings, state.get("current_score"))
        entity_verb_outcomes = {
            name: {v: o for v, o in data.get("verb_outcomes", {}).items() if o == "invalid"}
            for name, data in state.get("known_entities", {}).items()
            if any(o == "invalid" for o in data.get("verb_outcomes", {}).values())
        }
        run_record = {
            "run_id": run_id,
            "outcome": outcome,
            "final_score": state.get("current_score"),
            "max_score": state.get("max_score"),
            "steps": steps_run,
            "futile_edges": [list(e) for e in sorted(state.get("futile_edges", set()))],
            "entity_verb_outcomes": entity_verb_outcomes,
        }
        total_input = sum(e.get("token_usage", {}).get("input_tokens", 0) for e in state["game_log"])
        total_output = sum(e.get("token_usage", {}).get("output_tokens", 0) for e in state["game_log"])
        run_record["token_usage"] = {"input_tokens": total_input, "output_tokens": total_output}

        log_path = _write_log(state["game_log"], run_id)
        run_path = _write_run_record(run_record)
        merge_run_record(run_record, STRATEGY_PATH)
        if verbose:
            print(f"\nOutcome: {outcome}", file=sys.stderr)
            print(f"Tokens: {total_input:,} in / {total_output:,} out", file=sys.stderr)
            if LLM_INPUT_PRICE_PER_MILLION or LLM_OUTPUT_PRICE_PER_MILLION:
                cost = (total_input * LLM_INPUT_PRICE_PER_MILLION
                        + total_output * LLM_OUTPUT_PRICE_PER_MILLION) / 1_000_000
                print(f"Estimated cost: ${cost:.4f}", file=sys.stderr)
            print(f"Full log written to {log_path}", file=sys.stderr)
            print(f"Run record written to {run_path}", file=sys.stderr)
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
