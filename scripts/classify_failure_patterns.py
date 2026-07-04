#!/usr/bin/env python3
"""Classify unrecognized take-failure responses and confidence-gated autofix.

agent.py surfaces "take X" responses that matched neither a known failure
pattern nor a confirmed add-to-inventory as `unrecognized_failure_response`
anomalies (see anomaly_detector.py) -- evidence only, no judgement. This
script is the offline "fix agent" that makes the hard/soft judgement call,
never the live game loop:

- High confidence -> autofix: append the new pattern to
  configs/knight_orc.json, run the full test suite, and open a PR (still a
  PR, not a direct commit to develop -- "autofix" skips the interactive
  question, not code review).
- Medium/low confidence, or a proposed pattern that doesn't actually match
  the evidence -> raise to a human: print a NEEDS HUMAN REVIEW block, no
  config write, no PR.

Mirrors scripts/architect.py's structure (tool-use, same error handling).

Usage:
    python scripts/classify_failure_patterns.py <run_id> [--model MODEL]
    python scripts/classify_failure_patterns.py watch_20260704_170848

Requires ANTHROPIC_API_KEY in environment.
Requires scripts/detect_anomalies.py to have already run (anomaly_report.json
must exist).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
import jsonschema

import agent_dev_loop

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")
ORCHESTRATOR_DIR = RUNS_DIR / "orchestrator"
CONFIG_PATH = Path("configs/knight_orc.json")
SCHEMA_PATH = Path("schemas/game_config.schema.json")

_DEFAULT_MODEL = "claude-sonnet-5"
PR_TITLE_PREFIX = "[pattern-classifier]"

_SYSTEM_PROMPT = """\
You are classifying a single unrecognized game response from the text \
adventure Knight Orc (Level 9 Computing), specifically a response to a \
"take <object>" command that matched none of the agent's known failure \
patterns, and where the object was also not confirmed added to inventory.

Your job: decide whether this response describes a PERMANENT, \
object-inherent block ("hard" -- e.g. scenery, too heavy, fixed in place \
-- true every time, safe to treat as a permanent fact) or a TEMPORARY, \
state-dependent block ("soft" -- e.g. carry-capacity, already holding \
something, a condition that could change later -- must never be treated \
as permanent).

Also propose a regex `pattern` fragment (Python re syntax, case-insensitive \
matching will be applied) that matches this response and would generalize \
to similar phrasings, styled like the existing patterns in \
configs/knight_orc.json (e.g. "that'?s probably just scenery", "you can'?t \
do that right now") -- not just an exact literal escape of this one string.

State your `confidence` honestly as "high" only when you're confident this \
classification and pattern will hold for every future occurrence of this \
phrasing. Use "medium" or "low" if there's real ambiguity (e.g. the \
response could plausibly be state-dependent, or you're not sure the \
pattern generalizes correctly) -- a wrong "high" gets auto-committed to a \
shared config file used by every future run, so under-claiming confidence \
is much cheaper than over-claiming it.
"""

_CLASSIFICATION_TOOL = {
    "name": "submit_classification",
    "description": "Submit the classification for this unrecognized failure response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["hard", "soft"],
                "description": "'hard' = permanent/object-inherent, 'soft' = temporary/state-dependent.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How confident this classification and pattern will hold for every future occurrence.",
            },
            "pattern": {
                "type": "string",
                "description": "Python re (case-insensitive) fragment matching this response, styled like existing configs/knight_orc.json patterns.",
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences: why this classification and confidence level.",
            },
        },
        "required": ["classification", "confidence", "pattern", "reasoning"],
    },
}


def _user_message(anomaly):
    evidence = anomaly["evidence"]
    return f"""\
## Unrecognized failure response

- Target: `{evidence['target']}`
- Response: {evidence['response']!r}
- Occurred {evidence['occurrences']}x in this run

Call `submit_classification` with your judgement.
"""


def _call_api(client, model, anomaly):
    try:
        message = client.messages.create(
            model=model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            tools=[_CLASSIFICATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_classification"},
            messages=[{"role": "user", "content": _user_message(anomaly)}],
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
        print(f"ERROR: model did not call submit_classification for {anomaly['id']}", file=sys.stderr)
        sys.exit(1)
    return dict(tool_use.input)


def _pattern_is_valid(pattern, response):
    """A malformed or non-matching pattern is worse than no fix at all —
    reject rather than trust it, regardless of stated confidence."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return False
    return bool(compiled.search(response))


def _slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:40]


def _autofix(anomaly, result):
    evidence = anomaly["evidence"]
    field = "hard_failure_patterns" if result["classification"] == "hard" else "soft_failure_patterns"

    open_prs = agent_dev_loop._open_prs()
    if open_prs:
        print(f"Open PR(s) against {agent_dev_loop.BASE_BRANCH} — resolve these before autofixing "
              "(prevents two auto-branches drifting out of sync):", file=sys.stderr)
        for pr in open_prs:
            print(f"  #{pr['number']} {pr['title']!r} ({pr['headRefName']}) — {pr['url']}", file=sys.stderr)
        sys.exit(1)

    agent_dev_loop._ensure_clean_worktree()
    original_branch = agent_dev_loop._current_branch()

    branch = f"fix/auto-pattern-{_slugify(evidence['target'])}"
    if agent_dev_loop._branch_exists(branch):
        sys.exit(f"Branch {branch!r} already exists — inspect/delete it before retrying.")

    agent_dev_loop._run(["git", "fetch", "origin", agent_dev_loop.BASE_BRANCH])
    agent_dev_loop._run(["git", "checkout", "-b", branch, f"origin/{agent_dev_loop.BASE_BRANCH}"])
    print(f"Branched {branch!r} off origin/{agent_dev_loop.BASE_BRANCH}")

    config_data = json.loads(CONFIG_PATH.read_text())
    config_data.setdefault(field, []).append(result["pattern"])
    jsonschema.validate(config_data, json.loads(SCHEMA_PATH.read_text()))
    CONFIG_PATH.write_text(json.dumps(config_data, indent=4) + "\n")

    print("Running full suite before opening a PR...")
    if not agent_dev_loop._run_tests():
        sys.exit(f"Full suite failed after adding the pattern on branch {branch!r} — "
                 "stopping without opening a PR. Review changes above.")

    agent_dev_loop._run(["git", "add", "-A"])
    commit_msg = (
        f"Auto-add {field.replace('_', ' ')} pattern for {evidence['target']!r}\n\n"
        f"Response: {evidence['response']!r}\n"
        f"Classification: {result['classification']} (confidence: {result['confidence']})\n"
        f"Reasoning: {result['reasoning']}\n\n"
        "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
    )
    agent_dev_loop._run(["git", "commit", "-m", commit_msg])
    agent_dev_loop._run(["git", "push", "-u", "origin", branch])

    pr_body = f"""\
## Summary
- Auto-classified an unrecognized `take {evidence['target']}` failure response as **{result['classification']}** (confidence: **{result['confidence']}**)
- Response: `{evidence['response']}`
- Reasoning: {result['reasoning']}
- Added to `configs/knight_orc.json`'s `{field}`: `{result['pattern']}`

## Test plan
- [x] `pytest tests/ -q` passes with the new pattern added
- [x] Pattern validated to actually match the evidence response before this PR was opened

Opened automatically by `scripts/classify_failure_patterns.py` (confidence-gated autofix — not written by a human). Review the pattern and reasoning before merging.
"""
    result_pr = agent_dev_loop._run(["gh", "pr", "create", "--base", agent_dev_loop.BASE_BRANCH, "--head", branch,
                                      "--title", f"{PR_TITLE_PREFIX} Add {field} pattern for {evidence['target']}",
                                      "--body", pr_body])
    print(f"Opened PR: {result_pr.stdout.strip()}")

    agent_dev_loop._run(["git", "checkout", original_branch])
    print(f"Back on {original_branch!r}.")


def _print_needs_review(anomaly, result):
    evidence = anomaly["evidence"]
    print("\n" + "=" * 60, file=sys.stderr)
    print("NEEDS HUMAN REVIEW", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Target:         {evidence['target']}", file=sys.stderr)
    print(f"Response:       {evidence['response']!r}", file=sys.stderr)
    print(f"Occurrences:    {evidence['occurrences']}", file=sys.stderr)
    print(f"Classification: {result.get('classification', '?')} (confidence: {result.get('confidence', '?')})", file=sys.stderr)
    print(f"Pattern:        {result.get('pattern', '?')}", file=sys.stderr)
    print(f"Reasoning:      {result.get('reasoning', '?')}", file=sys.stderr)
    print("No config change made. Add manually to configs/knight_orc.json if you agree.", file=sys.stderr)


def classify(run_id, model=_DEFAULT_MODEL):
    """Classify every unrecognized_failure_response anomaly for run_id.

    Returns (autofixed_count, needs_review_count).
    """
    report_path = ORCHESTRATOR_DIR / run_id / "anomaly_report.json"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found — run detect_anomalies.py first", file=sys.stderr)
        sys.exit(1)

    report = json.loads(report_path.read_text())
    targets = [a for a in report["anomalies"] if a["type"] == "unrecognized_failure_response"]
    if not targets:
        print("No unrecognized_failure_response anomalies — nothing to classify.", file=sys.stderr)
        return 0, 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Checked once up front, not per-anomaly inside _autofix(): a mid-loop
    # sys.exit on the first high-confidence anomaly would silently drop
    # classification of every anomaly after it. If a PR is already open,
    # still classify (worth knowing what the LLM says) but never autofix.
    open_prs = agent_dev_loop._open_prs()
    if open_prs:
        print(f"Open PR(s) against {agent_dev_loop.BASE_BRANCH} — classifying only, no autofix "
              "this run (prevents two auto-branches drifting out of sync):", file=sys.stderr)
        for pr in open_prs:
            print(f"  #{pr['number']} {pr['title']!r} ({pr['headRefName']}) — {pr['url']}", file=sys.stderr)

    autofixed, needs_review = 0, 0
    for anomaly in targets:
        print(f"Classifying {anomaly['id']} ({anomaly['evidence']['target']!r})...", file=sys.stderr)
        result = _call_api(client, model, anomaly)

        pattern_ok = _pattern_is_valid(result["pattern"], anomaly["evidence"]["response"])
        if not pattern_ok:
            result["reasoning"] += " [pattern failed validation — did not match the evidence response]"

        if result["confidence"] == "high" and pattern_ok and not open_prs:
            _autofix(anomaly, result)
            autofixed += 1
        else:
            _print_needs_review(anomaly, result)
            needs_review += 1

    return autofixed, needs_review


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID matching an existing anomaly_report.json")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Claude model to use (default: %(default)s)")
    args = parser.parse_args()

    autofixed, needs_review = classify(args.run_id, args.model)
    print(f"\n{autofixed} autofixed, {needs_review} needing human review.", file=sys.stderr)
    sys.exit(1 if needs_review else 0)


if __name__ == "__main__":
    main()
