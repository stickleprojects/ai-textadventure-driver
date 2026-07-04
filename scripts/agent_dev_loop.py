#!/usr/bin/env python3
"""Agentic dev loop: run game → analyze → Claude fixes code/prompts → test → repeat.

Uses the Anthropic API with tool use. Each iteration:
  1. Run the game headlessly via run_and_analyze.py
  2. Read logs/latest_analysis.md
  3. If no issues: stop (success)
  4. Send the analysis to a Claude fixer agent that edits source files
  5. Run the pytest suite to verify no regressions
  6. Loop

With --spec-target SCENARIO_ID, the loop instead targets one scenario from
tests/spec_scenarios/scenarios.json (feature 59): it hands the fixer the
relevant docs/agent_behavior_spec.md section plus the fixture, and succeeds
when that scenario's xfail is lifted, its test passes, and the full suite
still passes.

Usage:
    python scripts/agent_dev_loop.py [--steps N] [--iterations N] [--model MODEL] [--dry-run]
    python scripts/agent_dev_loop.py --spec-target SCENARIO_ID [--iterations N] [--model MODEL] [--dry-run]

Environment:
    ANTHROPIC_API_KEY   required (for the fixer agent)
    LEVEL9_INTERPRETER  path to glklevel9 binary  (default: ./tools/glklevel9)
    LEVEL9_ROM          path to game ROM           (default: ./gamefiles/knight-orc/GAMEDAT1.DAT)
    EVAL_MODEL_PATH     path to LLM .gguf          (default: ../models/Phi-3.5-mini-instruct-Q3_K_M.gguf)

Example:
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 50 --iterations 3
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 20 --dry-run
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --spec-target npc_theft_removes_from_inventory
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
VENV_ACTIVATE = PROJECT_ROOT / ".venv" / "bin" / "activate"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SPEC_DOC = PROJECT_ROOT / "docs" / "agent_behavior_spec.md"
SCENARIOS_FILE = PROJECT_ROOT / "tests" / "spec_scenarios" / "scenarios.json"


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
- If an issue is ambiguous, make a conservative fix and note your uncertainty in the summary.\
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
    result = subprocess.run(
        f"source {VENV_ACTIVATE} && python -m pytest tests/ -x -q --ignore=tests/test_evals.py",
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


# ── Spec-target mode (feature 59) ──────────────────────────────────────────────

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


def _scenario_still_xfail(scenario_id):
    scenarios = json.loads(SCENARIOS_FILE.read_text())
    for scenario in scenarios:
        if scenario["id"] == scenario_id:
            return bool(scenario.get("xfail"))
    sys.exit(f"No scenario with id {scenario_id!r} in {SCENARIOS_FILE}")


def _run_scenario_test(scenario_id):
    result = subprocess.run(
        f"source {VENV_ACTIVATE} && python -m pytest tests/test_spec_scenarios.py -k {scenario_id} -q",
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )
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


def run_spec_target(scenario_id, iterations, model, dry_run):
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

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Spec-target iteration {iteration}/{iterations}: {scenario_id}")
        print(f"{'='*60}\n")

        run_fixer_agent(prompt, model=model)

        print("\nChecking scenario test...")
        scenario_passed = _run_scenario_test(scenario_id)
        still_xfail = _scenario_still_xfail(scenario_id)

        if scenario_passed and not still_xfail:
            print("\nRunning full suite to guard against regressions...")
            if not _run_tests():
                sys.exit("Full suite failed after spec-target fix — stopping. Review changes above.")
            print(f"\nScenario {scenario_id!r} now passes and the full suite is green.")
            return

        if still_xfail:
            print(f"\nScenario {scenario_id!r} is still marked xfail — behavior not yet implemented.")
        else:
            print(f"\nScenario {scenario_id!r} xfail was removed but the test still fails.")

    print(f"\nReached max iterations ({iterations}) without resolving {scenario_id!r}.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps",      type=int, default=50,          help="Game steps per run (default: 50)")
    parser.add_argument("--iterations", type=int, default=5,           help="Max fix iterations (default: 5)")
    parser.add_argument("--model",      default="claude-sonnet-4-6",   help="Claude model for fixer agent")
    parser.add_argument("--config",     metavar="PATH",                help="Game config JSON (default: built-in Knight Orc values)")
    parser.add_argument("--dry-run",    action="store_true",           help="Analyze only, skip auto-fix")
    parser.add_argument("--spec-target", metavar="SCENARIO_ID",        help="Build toward one tests/spec_scenarios/scenarios.json scenario instead of the anomaly-based loop")
    args = parser.parse_args()

    if args.spec_target:
        run_spec_target(args.spec_target, args.iterations, args.model, args.dry_run)
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

        print("\n[4/4] Running tests...")
        if not _run_tests():
            sys.exit("Tests failed after auto-fix — stopping loop. Review changes above.")
        print("Tests passed.\n")

    print(f"\nReached max iterations ({args.iterations}) without a clean run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
