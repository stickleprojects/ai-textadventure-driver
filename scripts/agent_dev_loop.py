#!/usr/bin/env python3
"""Agentic dev loop: run game → analyze → Claude fixes code/prompts → test → repeat.

Uses the Anthropic API with tool use. Each iteration:
  1. Run the game headlessly via run_and_analyze.py
  2. Read logs/latest_analysis.md
  3. If no issues: stop (success)
  4. Send the analysis to a Claude fixer agent that edits source files
  5. Run the pytest suite to verify no regressions
  6. Loop

Usage:
    python scripts/agent_dev_loop.py [--steps N] [--iterations N] [--model MODEL] [--dry-run]

Environment:
    ANTHROPIC_API_KEY   required (for the fixer agent)
    LEVEL9_INTERPRETER  path to glklevel9 binary  (default: ./tools/glklevel9)
    LEVEL9_ROM          path to game ROM           (default: ./gamefiles/knight-orc/GAMEDAT1.DAT)
    EVAL_MODEL_PATH     path to LLM .gguf          (default: ../models/Phi-3.5-mini-instruct-Q3_K_M.gguf)

Example:
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 50 --iterations 3
    ANTHROPIC_API_KEY=sk-... python scripts/agent_dev_loop.py --steps 20 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
VENV_ACTIVATE = PROJECT_ROOT / ".env" / "bin" / "activate"
VENV_PYTHON = PROJECT_ROOT / ".env" / "bin" / "python"


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
            "Activate the virtualenv with: source .env/bin/activate && <cmd>. "
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
- agent.py       — decision logic, _is_creature() pre-filter, state transitions, process_agent_step()
- llm.py         — LLM prompt and extract_knowledge() with up-to-3 retry logic
- game_engine.py — pexpect subprocess interface, _is_failure_response()
- ui.py          — pyvis map rendering
- tests/         — pytest unit tests (no LLM required)

Your job each iteration:
1. Read the analysis report to understand what the agent is doing wrong.
2. Read the relevant source files to understand the current implementation.
3. Apply the minimum change needed to fix the reported issues.
4. Run: source .env/bin/activate && python -m pytest tests/ -x -q --ignore=tests/test_evals.py
5. Fix any test failures before finishing.
6. Write a brief summary of what you changed and why.

Rules:
- Prefer prompt edits in llm.py over logic changes in agent.py where possible.
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

def _run_analysis(steps):
    result = subprocess.run(
        [str(VENV_PYTHON), "scripts/run_and_analyze.py", str(steps)],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps",      type=int, default=50,                help="Game steps per run (default: 50)")
    parser.add_argument("--iterations", type=int, default=5,                 help="Max fix iterations (default: 5)")
    parser.add_argument("--model",      default="claude-sonnet-4-6",         help="Claude model for fixer agent")
    parser.add_argument("--dry-run",    action="store_true",                  help="Analyze only, skip auto-fix")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"{'='*60}\n")

        print(f"[1/4] Running game for {args.steps} steps...")
        exit_code = _run_analysis(args.steps)

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
