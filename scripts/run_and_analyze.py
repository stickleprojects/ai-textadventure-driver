#!/usr/bin/env python3
"""Run the agent headlessly via watch_run.py, then write an issue analysis.

Thin wrapper around scripts/watch_run.py's run() — delegates all game-running
mechanics (initial state incl. cross-run strategy load, cloud/local LLM
selection, per-step verbose output, cross-run strategy merge, map images) to
it instead of maintaining a second implementation that drifts out of sync
(see docs/bugs/65.md, docs/bugs/66.md — this script had fallen behind
watch_run.py three separate times before this). Adds the log_analyzer-based
issue report (logs/latest_analysis.md) that agent_dev_loop.py's anomaly-based
loop consumes — that's this script's one remaining distinct job.

Usage:
    python scripts/run_and_analyze.py [steps] [--config PATH]

Environment: same as scripts/watch_run.py (LEVEL9_INTERPRETER, LEVEL9_ROM,
EVAL_MODEL_PATH, LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL/LLM_JSON_MODE,
GAME_STRATEGY) — importing watch_run runs its module-level .env load, so
there's nothing extra to configure here.

Output:
    logs/<run_id>.json         full game log (written by watch_run.py)
    runs/<run_id>.json         run record (written by watch_run.py)
    logs/<run_id>_map.png      map image (written by watch_run.py)
    logs/latest_analysis.md    issue report consumed by agent_dev_loop.py

Every run now also merges into configs/knight_orc_strategy.json's cross-run
learning (futile_edges, entity_verb_outcomes, world_graph, run_history) —
previously this script ran in isolation and neither benefited from nor
contributed to it.

Exit code:
    0  clean run (no issues detected)
    1  issues found, or startup error
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import watch_run
from game_config import config
from log_analyzer import analyze_log

LOG_DIR = watch_run.LOG_DIR


def run(steps=50, verbose=True):
    findings, run_id = watch_run.run(steps=steps, verbose=verbose)

    if run_id is None:
        issues = [{"type": f.get("type", "startup_error"), "description": f.get("message", "")}
                  for f in findings]
        _write_report(issues, [], datetime.now().strftime("%Y%m%d_%H%M%S"), None)
        return [], issues

    log_path = LOG_DIR / f"{run_id}.json"
    game_log = json.loads(log_path.read_text())

    issues = analyze_log(game_log)
    _write_report(issues, game_log, run_id.removeprefix("watch_"), log_path)
    return game_log, issues


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
        "- `agent.py` — `determine_next_action()`, `process_agent_step()`",
        "- `response_classification.py` — `is_creature()`, `is_hard_failure()`, `is_soft_failure()`, `parse_inventory_response()`",
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
