#!/usr/bin/env python3
"""Agentic dev loop: run game → analyze → Claude fixes code/prompts → test → repeat.

Uses the Anthropic API with tool use. Each iteration:
  1. Run the game headlessly via run_and_analyze.py
  2. Read logs/latest_analysis.md
  3. If no issues: stop (success)
  4. Send the analysis to a Claude fixer agent that edits source files
  5. Run the pytest suite to verify no regressions
  6. Loop

With --spec-target [SCENARIO_ID], the loop instead targets one scenario from
tests/spec_scenarios/scenarios.json (feature 59): it hands the fixer the
relevant docs/agent_behavior_spec.md section plus the fixture, and succeeds
when that scenario's xfail is lifted, its test passes, and the full suite
still passes. Omit SCENARIO_ID to use the first open (xfail'd) scenario in
file order — see scripts/list_scenarios.py / run_dev_loop.sh --list-scenarios
for what's open.

Each --spec-target run gets its own branch (feat/59-scenario-<id>, off
origin/develop) and, on success, its own PR — one scenario fix per PR, so
concurrent fixes never land in the same diff. PR titles are prefixed with
"[fixer-agent]" so they're easy to spot later. Before starting, it checks
for open PRs against develop and refuses to start a new scenario while one
is outstanding, so two scenario branches can't drift out of sync with each
other. Requires a clean working tree and the `gh` CLI to be authenticated.

On success it also runs one real playthrough (scripts/watch_run.py,
--playthrough-steps, default 50) and attaches a "Playthrough evidence"
table to the PR — locations/NPCs/treasure/puzzles discovered and puzzles
solved, compared against the best prior value of each already recorded in
configs/knight_orc_strategy.json's run_history. This is best-effort: if the
playthrough can't run (no LLM/model/interpreter configured on this
machine), the PR still opens with an "evidence unavailable" note instead of
being blocked on it.

Usage:
    python scripts/agent_dev_loop.py [--steps N] [--iterations N] [--model MODEL] [--dry-run]
    python scripts/agent_dev_loop.py --spec-target [SCENARIO_ID] [--iterations N] [--model MODEL] [--dry-run] [--playthrough-steps N]

Environment:
    ANTHROPIC_API_KEY   required (for the fixer agent)
    LEVEL9_INTERPRETER  path to glklevel9 binary  (default: ./tools/glklevel9)
    LEVEL9_ROM          path to game ROM           (default: ./gamefiles/knight-orc/GAMEDAT1.DAT)
    EVAL_MODEL_PATH     path to LLM .gguf          (default: ../models/Phi-3.5-mini-instruct-Q3_K_M.gguf)

Example:
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 50 --iterations 3
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 20 --dry-run
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --spec-target npc_theft_removes_from_inventory
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --spec-target   # first open scenario
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_evaluator import compare_to_history  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
VENV_ACTIVATE = PROJECT_ROOT / ".venv" / "bin" / "activate"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SPEC_DOC = PROJECT_ROOT / "docs" / "agent_behavior_spec.md"
SCENARIOS_FILE = PROJECT_ROOT / "tests" / "spec_scenarios" / "scenarios.json"
STRATEGY_FILE = PROJECT_ROOT / "configs" / "knight_orc_strategy.json"
RUNS_DIR = PROJECT_ROOT / "runs"
BASE_BRANCH = "develop"
PR_TITLE_PREFIX = "[fixer-agent]"

# Cross-run state/fixture files the fixer might touch (directly, or as a side
# effect of running the game/tests), keyed to the schema that defines "valid".
# This catches the checkable slice of "corrupted state" — malformed JSON or
# schema violations. It does not catch runtime corruption like an infinite
# loop; see TEST_TIMEOUT_SECS / SCENARIO_TEST_TIMEOUT_SECS for that.
STATE_FILE_SCHEMAS = {
    "configs/knight_orc.json":          "game_config.schema.json",
    "configs/knight_orc_strategy.json": "strategy.schema.json",
    "configs/knight_orc_rooms.json":    "rooms.schema.json",
    "configs/knight_orc_items.json":    "items.schema.json",
    "tests/spec_scenarios/scenarios.json": "spec_scenario.schema.json",
}
TEST_TIMEOUT_SECS = 180
SCENARIO_TEST_TIMEOUT_SECS = 90
PLAYTHROUGH_TIMEOUT_SECS = 900


# ── Tool implementations ──────────────────────────────────────────────────────

def _bash(command, timeout=120):
    result = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable="/bin/bash",
    )
    return (result.stdout + result.stderr).strip() or "(no output)"


def _read_file(path):
    p = PROJECT_ROOT / path
    try:
        return p.read_text()
    except FileNotFoundError:
        return f"ERROR: not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def _edit_file(path, old_string, new_string):
    p = PROJECT_ROOT / path
    try:
        content = p.read_text()
    except FileNotFoundError:
        return f"ERROR: not found: {path}"
    count = content.count(old_string)
    if count == 0:
        return f"ERROR: old_string not found in {path}"
    if count > 1:
        return f"ERROR: old_string appears {count} times — add more surrounding context to make it unique"
    p.write_text(content.replace(old_string, new_string, 1))
    return f"OK: edited {path}"


def _write_file(path, content):
    p = PROJECT_ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"OK: wrote {path}"


_DISPATCH = {
    "bash":       lambda i: _bash(i["command"]),
    "read_file":  lambda i: _read_file(i["path"]),
    "edit_file":  lambda i: _edit_file(i["path"], i["old_string"], i["new_string"]),
    "write_file": lambda i: _write_file(i["path"], i["content"]),
}

_TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the project root. "
            "Activate the virtualenv with: source .venv/bin/activate && <cmd>. "
            "Returns stdout + stderr."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file. Path is relative to the project root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. "
            "old_string must match verbatim (whitespace and indentation included). "
            "Returns an error if old_string is not found or appears more than once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string", "description": "File path relative to project root"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (or overwrite) a file. Path relative to project root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]

_SYSTEM_PROMPT = """\
You are a software engineer fixing an autonomous text-adventure agent that plays "Knight Orc" (Level 9, 1987).

Project structure:
- configs/knight_orc.json — creature words, failure phrases, prompt pattern, inspection sequence
- game_config.py — GameConfig singleton; edit the JSON file rather than this module
- agent.py       — decision logic, _is_creature() pre-filter, state transitions, process_agent_step()
- llm.py         — LLM prompt and extract_knowledge() with up-to-3 retry logic
- game_engine.py — pexpect subprocess interface, _is_failure_response()
- ui.py          — pyvis map rendering
- tests/         — pytest unit tests (no LLM required)
- tests/spec_scenarios/scenarios.json — spec-driven agent behavior scenarios (feature 59);
  each entry has a "spec_ref" into docs/agent_behavior_spec.md, and an "xfail" key when the
  behavior isn't implemented yet. When your task is to make one of these scenarios pass,
  remove its "xfail" key only once the underlying behavior is actually implemented and the
  scenario's test genuinely passes — do not remove it just to silence the test.

Your job each iteration:
1. Read the analysis report to understand what the agent is doing wrong.
2. Read the relevant source files to understand the current implementation.
3. Apply the minimum change needed to fix the reported issues.
4. Run: source .venv/bin/activate && python -m pytest tests/ -x -q --ignore=tests/test_evals.py
5. Fix any test failures before finishing.
6. Write a brief summary of what you changed and why.

Rules:
- For creature/failure-phrase issues: edit configs/knight_orc.json, not agent.py.
- For LLM extraction issues: prefer prompt edits in llm.py over logic changes in agent.py.
- Do not refactor or add features beyond what the reported issues require.
- Do not add explanatory comments about what you changed.
- If an issue is ambiguous, make a conservative fix and note your uncertainty in the summary.
- configs/*.json and tests/spec_scenarios/scenarios.json are schema-validated
  (schemas/*.schema.json, checked by tests/test_json_schemas.py). If you edit any
  of them, keep them valid JSON matching their schema — the harness independently
  checks this after you finish and reverts the file (discarding your edit to it)
  if it's broken, so a corrupted state file never blocks the rest of your fix.
- Never introduce an unbounded loop (`while True`, unbounded retry/recursion)
  without a guaranteed exit condition. A prior incident (see docs/JOURNAL.md,
  "drain_game_buffer infinite loop in test suite") hung the test suite this way;
  the harness now times out and treats a hang as a failed attempt, but a timeout
  wastes the whole iteration, so avoid it in the first place.
- When a spec section's prose uses a placeholder (a pronoun, a generic name)
  that doesn't appear in that section's own quoted example text, do not invent
  literal matching text for it — match against the quoted examples specifically,
  and where possible verify against actual state instead of parsing narration
  for identity (e.g. "was this item in our inventory?", not "did the text say
  'from you'?"). A prior incident (see docs/JOURNAL.md, "Fixer agent invented a
  literal \"from you\" that the spec never said") hardcoded a word the spec
  never actually used, missing narration the game really produces.\
"""


# ── Fixer agent ───────────────────────────────────────────────────────────────

def run_fixer_agent(analysis_md, model, verbose=True):
    """Run a Claude tool-use agent that reads the analysis and fixes the code."""
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic not installed — run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": f"Here is the latest run analysis. Please fix the issues:\n\n{analysis_md}",
        }
    ]

    if verbose:
        print(f"  [fixer] starting ({model})...")

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8096,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            summary = next((b.text for b in response.content if hasattr(b, "text")), "(no summary)")
            if verbose:
                print(f"  [fixer] done — {summary[:300]}")
            return summary

        if response.stop_reason != "tool_use":
            print(f"  [fixer] unexpected stop_reason: {response.stop_reason}", file=sys.stderr)
            return ""

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  [fixer] {block.name}({json.dumps(block.input)[:100]})")
            result = _DISPATCH[block.name](block.input)
            if verbose:
                print(f"         → {str(result)[:160]}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )

        messages.append({"role": "user", "content": tool_results})


# ── Main loop ─────────────────────────────────────────────────────────────────

def _run_analysis(steps, game_config=None):
    cmd = [str(VENV_PYTHON), "scripts/run_and_analyze.py", str(steps)]
    if game_config:
        cmd += ["--config", game_config]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode


def _run_tests():
    try:
        result = subprocess.run(
            f"source {VENV_ACTIVATE} && python -m pytest tests/ -x -q --ignore=tests/test_evals.py",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            executable="/bin/bash",
            timeout=TEST_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        print(f"Test suite did not finish within {TEST_TIMEOUT_SECS}s — possible infinite "
              "loop in the fixer's change.", file=sys.stderr)
        return False
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


def _state_files_corrupted():
    """Return [(rel_path, problem), ...] for any tracked state/fixture file that
    is no longer valid JSON or no longer matches its schema."""
    problems = []
    for rel_path, schema_name in STATE_FILE_SCHEMAS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            problems.append((rel_path, "file is missing"))
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            problems.append((rel_path, f"invalid JSON: {e}"))
            continue
        schema = json.loads((PROJECT_ROOT / "schemas" / schema_name).read_text())
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            problems.append((rel_path, f"schema violation: {e.message}"))
    return problems


def _revert_state_files(rel_paths):
    _run(["git", "checkout", "--", *rel_paths])


def _report_and_revert_corruption(problems):
    print("\nFixer left state/fixture file(s) invalid — reverting before checking tests:")
    for rel_path, problem in problems:
        print(f"  {rel_path}: {problem}")
    _revert_state_files([p for p, _ in problems])


# ── Spec-target mode (feature 59) ──────────────────────────────────────────────

def _run(cmd, check=True):
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    except FileNotFoundError as e:
        sys.exit(f"command not found: {e.filename} — is it installed and on PATH?")
    if check and result.returncode != 0:
        sys.exit(f"`{' '.join(cmd)}` failed:\n{result.stdout}{result.stderr}")
    return result


def _current_branch():
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _branch_exists(branch):
    return _run(["git", "rev-parse", "--verify", "--quiet", branch], check=False).returncode == 0


def _ensure_clean_worktree():
    status = _run(["git", "status", "--porcelain"]).stdout
    if status.strip():
        sys.exit(
            "Working tree has uncommitted changes — commit or stash them first. "
            "Each --spec-target scenario needs a clean starting point since it "
            f"branches off origin/{BASE_BRANCH}:\n{status}"
        )


def _open_prs(base=BASE_BRANCH):
    result = _run(["gh", "pr", "list", "--base", base, "--state", "open",
                   "--json", "number,title,url,headRefName"])
    return json.loads(result.stdout)


def _create_scenario_branch(scenario_id):
    branch = f"feat/59-scenario-{scenario_id}"
    if _branch_exists(branch):
        sys.exit(
            f"Branch {branch!r} already exists — likely a leftover from a previous "
            "unresolved attempt at this scenario. Inspect/delete it before retrying."
        )
    _run(["git", "fetch", "origin", BASE_BRANCH])
    _run(["git", "checkout", "-b", branch, f"origin/{BASE_BRANCH}"])
    return branch


def _run_playthrough(steps):
    """Run one real playthrough via watch_run.py. Returns (run_record, None) on
    success or (None, error_message) on any failure — never raises, since a
    missing local LLM/model on this machine shouldn't block landing an
    otherwise-good fix, only mean no evidence is attached this time."""
    try:
        result = subprocess.run(
            f"source {VENV_ACTIVATE} && python scripts/watch_run.py {steps}",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            executable="/bin/bash",
            timeout=PLAYTHROUGH_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        return None, f"watch_run.py did not finish within {PLAYTHROUGH_TIMEOUT_SECS}s"

    match = re.search(r"Run ID: (\S+)", result.stderr)
    if not match:
        return None, f"watch_run.py did not report a Run ID:\n{result.stdout}\n{result.stderr}"

    run_id = match.group(1)
    run_record_path = RUNS_DIR / f"{run_id}.json"
    if not run_record_path.exists():
        return None, f"expected {run_record_path} but it wasn't written"

    return json.loads(run_record_path.read_text()), None


def _format_playthrough_evidence(run_record, comparison, steps):
    metric_labels = {
        "final_score": "Score",
        "locations_discovered": "Locations discovered",
        "npcs_discovered": "NPCs discovered",
        "treasure_discovered": "Treasure discovered (silver items)",
        "puzzles_discovered": "Puzzles discovered",
        "puzzles_solved": "Puzzles solved",
    }
    rows = []
    for metric, label in metric_labels.items():
        current = comparison[metric]["current"]
        best_prior = comparison[metric]["best_prior"]
        if best_prior is None:
            delta = "n/a (no prior history)"
        else:
            diff = (current or 0) - best_prior
            delta = f"{diff:+d}"
        rows.append(f"| {label} | {current if current is not None else 'n/a'} | "
                     f"{best_prior if best_prior is not None else 'n/a'} | {delta} |")

    log_path = f"logs/{run_record['run_id']}.json"
    return f"""\
## Playthrough evidence ({steps}-step run after the fix)

Run ID: `{run_record['run_id']}` — outcome: `{run_record.get('outcome')}` — log: `{log_path}`

| Metric | This run | Best prior | Δ |
|---|---|---|---|
{chr(10).join(rows)}

Note: this is a whole-playthrough signal from one real run, not a targeted
check that this scenario's specific mechanic fired — the agent may or may
not encounter the exact situation (e.g. an NPC to steal from) depending on
where this particular run happens to go.
"""


def _commit_push_pr(scenario, branch, summary, evidence_md):
    _run(["git", "add", "-A"])
    commit_msg = (
        f"Resolve spec scenario {scenario['id']} ({scenario['spec_ref']})\n\n"
        f"{summary}\n\n"
        "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
    )
    _run(["git", "commit", "-m", commit_msg])
    _run(["git", "push", "-u", "origin", branch])

    requirement_line = (
        f" (Requirement {scenario['requirement_ref']})" if scenario.get("requirement_ref") else ""
    )
    pr_body = f"""\
## Summary
- Resolves spec scenario `{scenario['id']}` — `{scenario['spec_ref']}`{requirement_line}
- {summary}

## Test plan
- [x] `pytest tests/test_spec_scenarios.py -k {scenario['id']} -q` passes, xfail removed
- [x] `pytest tests/ -q --ignore=tests/test_evals.py` passes (full-suite regression guard)

{evidence_md}
Opened automatically by the fixer agent (`scripts/agent_dev_loop.py --spec-target {scenario['id']}`) — not written by a human.
"""
    result = _run(["gh", "pr", "create", "--base", BASE_BRANCH, "--head", branch,
                   "--title", f"{PR_TITLE_PREFIX} Fix spec scenario: {scenario['id']}",
                   "--body", pr_body])
    return result.stdout.strip()


def _slugify(heading):
    return re.sub(r"[^a-z0-9]+", "-", heading.strip().lower()).strip("-")


def _spec_section_text(spec_ref):
    """Return the docs/agent_behavior_spec.md ### section matching spec_ref, heading included."""
    lines = SPEC_DOC.read_text().splitlines()
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if not line.startswith("### "):
            continue
        if start is not None:
            end = i
            break
        if _slugify(line[4:]) == spec_ref:
            start = i
    if start is None:
        sys.exit(f"No '### ' heading in {SPEC_DOC} slugifies to spec_ref {spec_ref!r}")
    return "\n".join(lines[start:end]).strip()


def _load_scenario(scenario_id):
    scenarios = json.loads(SCENARIOS_FILE.read_text())
    for scenario in scenarios:
        if scenario["id"] == scenario_id:
            return scenario
    sys.exit(f"No scenario with id {scenario_id!r} in {SCENARIOS_FILE}")


def _first_open_scenario_id():
    """Return the id of the first scenario in file order that still has an
    "xfail" key, or None if every scenario is already implemented."""
    scenarios = json.loads(SCENARIOS_FILE.read_text())
    for scenario in scenarios:
        if scenario.get("xfail"):
            return scenario["id"]
    return None


def _scenario_still_xfail(scenario_id):
    scenarios = json.loads(SCENARIOS_FILE.read_text())
    for scenario in scenarios:
        if scenario["id"] == scenario_id:
            return bool(scenario.get("xfail"))
    sys.exit(f"No scenario with id {scenario_id!r} in {SCENARIOS_FILE}")


def _run_scenario_test(scenario_id):
    try:
        result = subprocess.run(
            f"source {VENV_ACTIVATE} && python -m pytest tests/test_spec_scenarios.py -k {scenario_id} -q",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            executable="/bin/bash",
            timeout=SCENARIO_TEST_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        print(f"Scenario test did not finish within {SCENARIO_TEST_TIMEOUT_SECS}s — possible "
              "infinite loop in the fixer's change.", file=sys.stderr)
        return False
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


def _spec_target_prompt(scenario):
    section = _spec_section_text(scenario["spec_ref"])
    return f"""\
Target: make the spec-driven scenario test "{scenario['id']}" pass for real
(not by weakening the assertions or the fixture).

Relevant section of docs/agent_behavior_spec.md ({scenario['spec_ref']}):

{section}

Scenario fixture (tests/spec_scenarios/scenarios.json, id={scenario['id']}):

{json.dumps(scenario, indent=2)}

The fixture's "mode" field means:
- "decision": tests/test_spec_scenarios.py builds state from "initial_state" via
  tests.conftest.make_state(), calls agent.determine_next_action(state), and
  checks the returned action against "expect_action_startswith"/"expect_action_equals".
- "step": same state, but tests/test_spec_scenarios.py patches
  agent.execute_game_command to return "mock_response" and agent.extract_knowledge
  to return "mock_extracted", calls agent.process_agent_step(state, stub_child, None),
  and checks the resulting state against "expect_state_equals"/"expect_contains"/
  "expect_absent".

Implement the behavior described in the spec section so this scenario's test
passes, then remove the "xfail" key for id={scenario['id']} in
tests/spec_scenarios/scenarios.json. Verify with:
  source .venv/bin/activate && python -m pytest tests/test_spec_scenarios.py -k {scenario['id']} -q
"""


def run_spec_target(scenario_id, iterations, model, dry_run, playthrough_steps):
    scenario = _load_scenario(scenario_id)
    prompt = _spec_target_prompt(scenario)
    print(f"Targeting spec scenario {scenario_id!r} (spec_ref={scenario['spec_ref']!r})\n")

    if dry_run:
        print("[dry-run] Prompt that would be sent to the fixer agent:\n")
        print(prompt)
        return

    if not scenario.get("xfail"):
        print(f"Scenario {scenario_id!r} has no 'xfail' key — nothing to build toward.")
        return

    open_prs = _open_prs()
    if open_prs:
        print(f"Open PR(s) against {BASE_BRANCH} — resolve these before starting a new "
              "scenario fix (prevents two scenario branches drifting out of sync):")
        for pr in open_prs:
            print(f"  #{pr['number']} {pr['title']!r} ({pr['headRefName']}) — {pr['url']}")
        sys.exit(1)

    pre_existing_problems = _state_files_corrupted()
    if pre_existing_problems:
        sys.exit(
            "State/fixture file(s) are already invalid before starting — fix these "
            "first, this isn't the fixer's doing:\n"
            + "\n".join(f"  {p}: {msg}" for p, msg in pre_existing_problems)
        )

    _ensure_clean_worktree()
    original_branch = _current_branch()
    branch = _create_scenario_branch(scenario_id)
    print(f"Branched {branch!r} off origin/{BASE_BRANCH}\n")

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Spec-target iteration {iteration}/{iterations}: {scenario_id}")
        print(f"{'='*60}\n")

        summary = run_fixer_agent(prompt, model=model)

        corruption = _state_files_corrupted()
        if corruption:
            _report_and_revert_corruption(corruption)

        print("\nChecking scenario test...")
        scenario_passed = _run_scenario_test(scenario_id)
        still_xfail = _scenario_still_xfail(scenario_id)

        if scenario_passed and not still_xfail:
            print("\nRunning full suite to guard against regressions...")
            if not _run_tests():
                sys.exit(
                    f"Full suite failed after spec-target fix on branch {branch!r} — "
                    "stopping without opening a PR. Review changes above."
                )
            print(f"\nScenario {scenario_id!r} now passes and the full suite is green.")

            print(f"\nRunning a {playthrough_steps}-step playthrough for evidence...")
            prior_history = json.loads(STRATEGY_FILE.read_text()).get("run_history", []) \
                if STRATEGY_FILE.exists() else []
            run_record, playthrough_error = _run_playthrough(playthrough_steps)
            if run_record:
                comparison = compare_to_history(run_record, prior_history)
                evidence_md = _format_playthrough_evidence(run_record, comparison, playthrough_steps)
            else:
                print(f"Playthrough evidence unavailable: {playthrough_error}")
                evidence_md = f"## Playthrough evidence\n\n_Unavailable: {playthrough_error}_\n"

            print("Committing and opening PR...")
            pr_url = _commit_push_pr(scenario, branch, summary, evidence_md)
            print(f"\nOpened PR: {pr_url}")

            _run(["git", "checkout", original_branch])
            print(f"Back on {original_branch!r}.")
            return

        if still_xfail:
            print(f"\nScenario {scenario_id!r} is still marked xfail — behavior not yet implemented.")
        else:
            print(f"\nScenario {scenario_id!r} xfail was removed but the test still fails.")

    print(f"\nReached max iterations ({iterations}) without resolving {scenario_id!r}.")
    print(f"Branch {branch!r} has the unresolved attempt — inspect it or discard it manually "
          f"(git checkout {original_branch!r} first; nothing was committed on {branch!r}).")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps",      type=int, default=50,          help="Game steps per run (default: 50)")
    parser.add_argument("--iterations", type=int, default=5,           help="Max fix iterations (default: 5)")
    parser.add_argument("--model",      default="claude-sonnet-5",     help="Claude model for fixer agent (default: claude-sonnet-5)")
    parser.add_argument("--config",     metavar="PATH",                help="Game config JSON (default: built-in Knight Orc values)")
    parser.add_argument("--dry-run",    action="store_true",           help="Analyze only, skip auto-fix")
    parser.add_argument("--spec-target", nargs="?", const="", metavar="SCENARIO_ID",
                         help="Build toward one tests/spec_scenarios/scenarios.json scenario instead of "
                              "the anomaly-based loop. Omit SCENARIO_ID to use the first open (xfail'd) scenario.")
    parser.add_argument("--playthrough-steps", type=int, default=50,   help="Steps for the post-fix evidence playthrough in --spec-target mode (default: 50)")
    args = parser.parse_args()

    if args.spec_target is not None:
        scenario_id = args.spec_target or _first_open_scenario_id()
        if not scenario_id:
            sys.exit(f"No open (xfail'd) scenarios left in {SCENARIOS_FILE} — nothing to build toward.")
        if not args.spec_target:
            print(f"No SCENARIO_ID given — using first open scenario: {scenario_id!r}\n")
        run_spec_target(scenario_id, args.iterations, args.model, args.dry_run, args.playthrough_steps)
        return

    LOG_DIR.mkdir(exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"{'='*60}\n")

        print(f"[1/4] Running game for {args.steps} steps...")
        exit_code = _run_analysis(args.steps, game_config=args.config)

        analysis_path = LOG_DIR / "latest_analysis.md"
        if not analysis_path.exists():
            sys.exit("run_and_analyze.py produced no analysis file — check for errors above")

        analysis = analysis_path.read_text()
        print(f"\n[2/4] Analysis:\n{analysis}\n")

        if exit_code == 0:
            print("No issues found. Agent is running cleanly — loop complete.")
            return

        if args.dry_run:
            print("[dry-run] Skipping fixer agent.")
            return

        print("[3/4] Fixer agent editing code...")
        run_fixer_agent(analysis, model=args.model)

        corruption = _state_files_corrupted()
        if corruption:
            _report_and_revert_corruption(corruption)

        print("\n[4/4] Running tests...")
        if not _run_tests():
            sys.exit("Tests failed after auto-fix — stopping loop. Review changes above.")
        print("Tests passed.\n")

    print(f"\nReached max iterations ({args.iterations}) without a clean run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
