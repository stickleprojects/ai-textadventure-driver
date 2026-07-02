#!/usr/bin/env python3
"""Stage 2 of the dev-loop orchestrator: turn anomalies into fix plans.

Reads runs/orchestrator/<run_id>/anomaly_report.json, calls Claude for each
anomaly, and writes canonical fix plans to plans/P<N>.json.

Deduplication: if the anomaly type already exists in plans/index.json the run
is recorded against the existing entry and the API is NOT called again.  A
notice is printed so the user can review whether to bump priority.

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
PLANS_INDEX = PLANS_DIR / "index.json"
PLANS_MD = Path("plans.md")

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


def _load_index():
    if PLANS_INDEX.exists():
        return json.loads(PLANS_INDEX.read_text())
    return {"next_id": 1, "plans": []}


def _save_index(index):
    PLANS_DIR.mkdir(exist_ok=True)
    PLANS_INDEX.write_text(json.dumps(index, indent=2))


def _plan_id(n):
    return f"P{n:03d}"


def _find_existing(index, anomaly_type):
    for entry in index["plans"]:
        if entry["anomaly_type"] == anomaly_type:
            return entry
    return None


def _write_plans_md(index):
    """Regenerate plans.md from index.json."""
    _STATUS_ORDER = ["open", "deferred", "fixed"]
    by_status = {s: [] for s in _STATUS_ORDER}
    for entry in index["plans"]:
        by_status.setdefault(entry.get("status", "open"), []).append(entry)

    lines = [
        "# Fix Plans",
        "",
        "Architect-generated issue registry. Each entry represents a distinct anomaly type — not a per-run finding.",
        "When a new run surfaces an existing type, the run is added under **Also seen** and priority is reviewed.",
        "Full fix plan JSON lives in `plans/P<N>.json`.",
        "",
        "---",
    ]

    for status in _STATUS_ORDER:
        entries = by_status.get(status, [])
        lines.append("")
        lines.append(f"## {status.capitalize()}")
        lines.append("")
        if not entries:
            lines.append("*(none)*")
        for e in entries:
            runs = e["source_runs"]
            first_run = runs[0] if runs else "unknown"
            also = runs[1:] if len(runs) > 1 else []
            anomaly_refs = ", ".join(
                aid.split("/")[-1] for aid in e.get("anomaly_ids", [])
                if aid.startswith(first_run)
            )
            lines.append(f"### {e['plan_id']} — {e['anomaly_type']} [{e['severity']}]")
            lines.append(e["summary"])
            lines.append(f"- **First seen:** {first_run} ({anomaly_refs})")
            if also:
                also_refs = "; ".join(also)
                lines.append(f"- **Also seen:** {also_refs}")
            else:
                lines.append("- **Also seen:** *(none yet)*")
            if e.get("fixed_in"):
                lines.append(f"- **Fixed in:** {e['fixed_in']}")
            if e.get("notes"):
                lines.append(f"- **Note:** {e['notes']}")
            lines.append(f"- **Plan:** [plans/{e['plan_id']}.json](plans/{e['plan_id']}.json)")
            lines.append("")
        lines.append("---")

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


def _user_message(anomaly, sources):
    return f"""\
## Anomaly to fix

```json
{json.dumps(anomaly, indent=2)}
```

## Relevant source files

{sources}

## Task

Before writing the plan, work through the three pre-checks from your instructions:
1. Does the code already enforce the guard you are about to propose? (Code audit)
2. Are the anomalous steps util=productive? If so, futile-edge marking is off the table. (Productivity check)
3. Which layer — (a) game engine, (b) LLM extraction, (c) graph/state update, (d) decision logic — originates the bad data? (Data origin layer)

Then produce a fix_plan.json object (schema: fix_plan/v1) with these fields:
- schema: "fix_plan/v1"
- anomaly_id: the anomaly's id string
- data_origin_layer: one of "game_engine", "llm_extraction", "graph_state", "decision_logic"
- productivity_check: "productive" if repeated steps are util=productive (fix must not mark edges futile), else "non_productive"
- root_cause_hypothesis: one concise sentence — must name the origin layer and confirm whether the guard already exists
- target_files: list of files to change
- change_summary: what to change and why (2-4 sentences)
- acceptance_criteria: list of 2-4 strings, each concretely observable in
  a future run log or strategy file — not "tests pass"
- risk_notes: potential regressions or edge cases to watch
- status: "proposed"

Reply with ONLY the raw JSON object, no markdown fences, no commentary.
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

    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"ERROR: model returned non-JSON for {anomaly['id']}:\n{raw}", file=sys.stderr)
        sys.exit(1)


def architect(run_id, anomaly_id=None, model="claude-opus-4-8"):
    """Produce fix plans for one anomaly (anomaly_id) or all anomalies (anomaly_id=None).

    New anomaly types get a canonical plans/P<N>.json and an entry in plans/index.json.
    Duplicate types (same anomaly_type already in the index) are recorded against the
    existing entry without calling the API again — a notice is printed to prompt a
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
    index = _load_index()

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
        existing = _find_existing(index, atype)

        if existing:
            dup_count += 1
            plan_path = PLANS_DIR / f"{existing['plan_id']}.json"
            if run_id not in existing["source_runs"]:
                existing["source_runs"].append(run_id)
            if aid_ref not in existing.get("anomaly_ids", []):
                existing.setdefault("anomaly_ids", []).append(aid_ref)
            old_sev = existing["severity"]
            new_sev = anomaly.get("severity", old_sev)
            upgraded = _SEVERITY_ORDER.get(new_sev, 99) < _SEVERITY_ORDER.get(old_sev, 99)
            if upgraded:
                existing["severity"] = new_sev
            print(
                f"  DUPLICATE: {anomaly['id']} ({atype}) already tracked as {existing['plan_id']} "
                f"[{existing['status']}]"
            )
            print(f"    Added run {run_id} to {existing['plan_id']}.")
            if upgraded:
                print(f"    ⚠ Severity upgraded {old_sev} → {new_sev} — review priority.")
            elif existing["status"] in ("fixed", "deferred"):
                print(f"    ⚠ Status is '{existing['status']}' but issue reappeared — consider reopening.")
            print(f"    Plan: {plan_path}")
            out_paths.append(plan_path)
        else:
            new_count += 1
            plan = _call_api(client, model, anomaly)
            pid = _plan_id(index["next_id"])
            index["next_id"] += 1
            plan["plan_id"] = pid
            plan_path = PLANS_DIR / f"{pid}.json"
            plan_path.write_text(json.dumps(plan, indent=2))
            index["plans"].append({
                "plan_id": pid,
                "anomaly_type": atype,
                "summary": plan.get("root_cause_hypothesis", anomaly.get("summary", "")),
                "severity": anomaly.get("severity", "low"),
                "status": "open",
                "source_runs": [run_id],
                "anomaly_ids": [aid_ref],
                "plan_file": str(plan_path),
            })
            print(f"  NEW {pid}: {atype}")
            print(f"    Root cause: {plan.get('root_cause_hypothesis', '?')}")
            print(f"    Files:      {plan.get('target_files', [])}")
            print(f"    Plan: {plan_path}")
            out_paths.append(plan_path)

    _save_index(index)
    _write_plans_md(index)
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
