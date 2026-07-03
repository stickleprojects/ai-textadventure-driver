#!/usr/bin/env python3
"""Stage 2 of the dev-loop orchestrator: turn anomalies into fix plans.

Reads runs/orchestrator/<run_id>/anomaly_report.json, calls Claude for each
anomaly, and writes canonical fix plans to plans/P<N>.json.

Deduplication: if the anomaly type already exists among plans/P*.json the run
is recorded against the existing plan and the API is NOT called again.  A
notice is printed so the user can review whether to bump priority.

plans.md is the human-facing index (lean table of plan_id/title/type/severity/
status), regenerated from plans/P*.json on every run. There is no separate
JSON index file — plan_id/anomaly_type act as the lookup keys, derived by
scanning the plan files directly.

Usage:
    python scripts/architect.py <run_id> [--anomaly-id a1] [--all] [--model MODEL]

Requires ANTHROPIC_API_KEY in environment.

Plans must be reviewed by a human before passing to the Dev stage —
auto-merge is intentionally not wired up yet.
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
PLANS_DIR = Path("plans")
PLANS_MD = Path("plans.md")

_STATUS_ORDER = ["open", "in_progress", "deferred", "fixed"]

# Which source files to include per anomaly type. More specific types first;
# the fallback key "" is always appended.
_FILE_ROUTES = {
    "loop_detected":           ["agent.py", "game_config.py"],
    "futile_streak":           ["agent.py", "game_config.py"],
    "redundant_streak":        ["agent.py"],
    "malformed_extraction":    ["llm.py", "game_engine.py"],
    "regression":              ["run_evaluator.py", "agent.py"],
    "agent_failure_finding":   ["agent.py", "game_config.py"],
    "futile_direction_probing":["agent.py", "llm.py"],  # may be extraction or decision layer
    "repetitive_zigzag_navigation": ["agent.py"],
    "redundant_object_inspection":  ["agent.py"],
    "":                        ["agent.py"],  # fallback
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

Before proposing any fix, apply these three checks:

1. CODE AUDIT — read the relevant source files and confirm whether the guard or
   check you are about to propose already exists. State explicitly in root_cause_hypothesis
   whether existing code already enforces this. If it does, your fix must target a
   different layer.

2. PRODUCTIVITY CHECK — if the anomaly involves a repeated or alternating action
   pattern, check whether each step is util=productive (new room, new entity, new
   inventory). If steps are productive, futile-edge marking and loop-break logic
   are contraindicated. The fix must change the *selection policy* (e.g. navigation
   scoring, priority ordering), not the edge state.

3. DATA ORIGIN LAYER — identify which layer produces the bad data:
   (a) game engine response  (b) LLM extraction  (c) graph/state update  (d) decision logic
   The fix must target the *origin layer*. A guard at a downstream consumer is
   fragile and may mask the same data entering via a different code path.
"""


def _plan_id(n):
    return f"P{n:03d}"


def _load_plans():
    """Load every plans/P*.json file, keyed by plan_id.

    This dict *is* the dedup/next-id source of truth — there is no separate
    JSON index to keep in sync.
    """
    return {f.stem: json.loads(f.read_text()) for f in sorted(PLANS_DIR.glob("P*.json"))}


def _next_plan_id(plans):
    nums = [int(pid[1:]) for pid in plans if pid[1:].isdigit()]
    return _plan_id(max(nums, default=0) + 1)


def _find_existing(plans, anomaly_type):
    for pid, plan in plans.items():
        if plan.get("anomaly_type") == anomaly_type:
            return pid, plan
    return None, None


def _write_plans_md(plans):
    """Regenerate plans.md — a lean table serving as the human-facing index.

    Full detail (root cause, acceptance criteria, source runs) lives in the
    linked plans/P<N>.json file, not here.
    """
    def sort_key(item):
        pid, plan = item
        status = plan.get("status", "open")
        status_rank = _STATUS_ORDER.index(status) if status in _STATUS_ORDER else len(_STATUS_ORDER)
        severity_rank = _SEVERITY_ORDER.get(plan.get("severity", "low"), 99)
        return (status_rank, severity_rank, pid)

    rows = sorted(plans.items(), key=sort_key)

    lines = [
        "# Fix Plans",
        "",
        "Architect-generated issue index. Each row is a distinct anomaly type — not a per-run finding.",
        "Full fix plan detail (root cause, acceptance criteria, source runs) lives in the linked plan file.",
        "",
        "| Plan | Title | Type | Severity | Status |",
        "|------|-------|------|----------|--------|",
    ]
    for pid, plan in rows:
        title = plan.get("title", "").replace("|", "\\|")
        atype = plan.get("anomaly_type", "")
        severity = plan.get("severity", "")
        status = plan.get("status", "")
        lines.append(f"| [{pid}](plans/{pid}.json) | {title} | {atype} | {severity} | {status} |")

    PLANS_MD.write_text("\n".join(lines) + "\n")


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


_FIX_PLAN_TOOL = {
    "name": "submit_fix_plan",
    "description": "Submit the completed fix plan for this anomaly.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short human-readable title for this fix, e.g. 'Two-room oscillation loop' (used in plans.md).",
            },
            "data_origin_layer": {
                "type": "string",
                "enum": ["game_engine", "llm_extraction", "graph_state", "decision_logic"],
                "description": "Which layer originates the bad data (see pre-check 3).",
            },
            "productivity_check": {
                "type": "string",
                "enum": ["productive", "non_productive"],
                "description": (
                    "'productive' if the repeated steps are util=productive (fix must not "
                    "mark edges futile), else 'non_productive' (see pre-check 2)."
                ),
            },
            "root_cause_hypothesis": {
                "type": "string",
                "description": (
                    "One concise sentence naming the origin layer and confirming whether "
                    "the guard already exists (see pre-check 1)."
                ),
            },
            "target_files": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Files to change.",
            },
            "change_summary": {
                "type": "string",
                "description": "What to change and why (2-4 sentences).",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
                "description": (
                    "Each concretely observable in a future run log or strategy file — "
                    "not 'tests pass'."
                ),
            },
            "risk_notes": {
                "type": "string",
                "description": "Potential regressions or edge cases to watch.",
            },
        },
        "required": [
            "title",
            "data_origin_layer",
            "productivity_check",
            "root_cause_hypothesis",
            "target_files",
            "change_summary",
            "acceptance_criteria",
            "risk_notes",
        ],
    },
}


def _user_message(anomaly, sources):
    return f"""\
## Anomaly to fix

```json
{json.dumps(anomaly, indent=2)}
```

## Relevant source files

{sources}

## Task

Work through the three pre-checks from your instructions (code audit, productivity
check, data origin layer) before deciding on a fix, then call `submit_fix_plan` with
your conclusions. Fold the pre-check reasoning into `root_cause_hypothesis` and
`productivity_check` rather than writing it out separately.
"""


def _call_api(client, model, anomaly):
    """Call the API for a single anomaly and return the parsed plan dict."""
    files = _pick_files(anomaly["type"])
    sources = _read_sources(files)
    print(f"Calling {model} for anomaly {anomaly['id']} ({anomaly['type']}) …")
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_FIX_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "submit_fix_plan"},
            messages=[{"role": "user", "content": _user_message(anomaly, sources)}],
        )
    except anthropic.AuthenticationError:
        print(
            "ERROR: Anthropic API key was rejected (401).\n"
            "Check ANTHROPIC_API_KEY — watch for stray characters from terminal paste (e.g. '[[200~' prefix).",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.BadRequestError as exc:
        print(f"ERROR: API rejected the request (400): {exc.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as exc:
        print(f"ERROR: API returned {exc.status_code}: {exc.message}", file=sys.stderr)
        sys.exit(1)

    tool_use = next((b for b in message.content if b.type == "tool_use"), None)
    if tool_use is None:
        print(f"ERROR: model did not call submit_fix_plan for {anomaly['id']}", file=sys.stderr)
        sys.exit(1)

    plan = dict(tool_use.input)
    plan["schema"] = "fix_plan/v1"
    plan["anomaly_id"] = anomaly["id"]
    plan["anomaly_type"] = anomaly["type"]
    plan["severity"] = anomaly.get("severity", "low")
    plan["status"] = "open"
    return plan


def architect(run_id, anomaly_id=None, model="claude-opus-4-8"):
    """Produce fix plans for one anomaly (anomaly_id) or all anomalies (anomaly_id=None).

    New anomaly types get a canonical plans/P<N>.json. Duplicate types (same
    anomaly_type already present in an existing plan file) are recorded against
    that plan without calling the API again — a notice is printed to prompt a
    priority review.

    Returns list of plan file paths written (new plans only; duplicates return the
    existing path without rewriting it).
    """
    report_path = ORCHESTRATOR_DIR / run_id / "anomaly_report.json"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found — run detect_anomalies.py first", file=sys.stderr)
        sys.exit(1)

    report = json.loads(report_path.read_text())
    if not report["anomalies"]:
        print("No anomalies in report — nothing to plan.", file=sys.stderr)
        sys.exit(0)

    if anomaly_id is None:
        targets = report["anomalies"]
    elif isinstance(anomaly_id, list):
        targets = [_pick_anomaly(report["anomalies"], aid) for aid in anomaly_id]
    else:
        targets = [_pick_anomaly(report["anomalies"], anomaly_id)]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    PLANS_DIR.mkdir(exist_ok=True)
    plans = _load_plans()

    # Deduplicate targets by type within this batch before hitting the API
    seen_types = {}
    deduped = []
    for anomaly in targets:
        atype = anomaly["type"]
        if atype not in seen_types:
            seen_types[atype] = anomaly["id"]
            deduped.append(anomaly)
        else:
            print(
                f"  BATCH-DUP: {anomaly['id']} ({atype}) same type as {seen_types[atype]} in this run — skipping duplicate"
            )

    out_paths = []
    new_count = 0
    dup_count = 0

    for anomaly in deduped:
        atype = anomaly["type"]
        aid_ref = f"{run_id}/{anomaly['id']}"
        pid, existing = _find_existing(plans, atype)

        if existing:
            dup_count += 1
            plan_path = PLANS_DIR / f"{pid}.json"
            changed = False
            if run_id not in existing.get("source_runs", []):
                existing.setdefault("source_runs", []).append(run_id)
                changed = True
            if aid_ref not in existing.get("anomaly_ids", []):
                existing.setdefault("anomaly_ids", []).append(aid_ref)
                changed = True
            old_sev = existing.get("severity", "low")
            new_sev = anomaly.get("severity", old_sev)
            upgraded = _SEVERITY_ORDER.get(new_sev, 99) < _SEVERITY_ORDER.get(old_sev, 99)
            if upgraded:
                existing["severity"] = new_sev
                changed = True
            if changed:
                plan_path.write_text(json.dumps(existing, indent=2))
            print(
                f"  DUPLICATE: {anomaly['id']} ({atype}) already tracked as {pid} "
                f"[{existing['status']}]"
            )
            print(f"    Added run {run_id} to {pid}.")
            if upgraded:
                print(f"    ⚠ Severity upgraded {old_sev} → {new_sev} — review priority.")
            elif existing["status"] in ("fixed", "deferred"):
                print(f"    ⚠ Status is '{existing['status']}' but issue reappeared — consider reopening.")
            print(f"    Plan: {plan_path}")
            out_paths.append(plan_path)
        else:
            new_count += 1
            plan = _call_api(client, model, anomaly)
            pid = _next_plan_id(plans)
            plan["plan_id"] = pid
            plan["source_runs"] = [run_id]
            plan["anomaly_ids"] = [aid_ref]
            plan_path = PLANS_DIR / f"{pid}.json"
            plan_path.write_text(json.dumps(plan, indent=2))
            plans[pid] = plan
            print(f"  NEW {pid}: {atype}")
            print(f"    Root cause: {plan.get('root_cause_hypothesis', '?')}")
            print(f"    Files:      {plan.get('target_files', [])}")
            print(f"    Plan: {plan_path}")
            out_paths.append(plan_path)

    _write_plans_md(plans)
    print(
        f"\n{new_count} new plan(s), {dup_count} duplicate(s) recorded. "
        f"Review plans.md before passing to the Dev stage."
    )
    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID matching an existing anomaly_report.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--anomaly-id", help="Specific anomaly id, e.g. a1 (default: highest severity)")
    group.add_argument("--all", action="store_true", help="Produce fix plans for all anomalies in the report")
    parser.add_argument("--model", default="claude-opus-4-8", help="Claude model to use (default: %(default)s)")
    args = parser.parse_args()
    anomaly_id = None if args.all else args.anomaly_id
    architect(args.run_id, anomaly_id, args.model)


if __name__ == "__main__":
    main()
