#!/usr/bin/env python3
"""List spec-driven behavior scenarios (tests/spec_scenarios/scenarios.json).

Shows each scenario's id, which docs/agent_behavior_spec.md section it covers
(spec_ref), the requirement it's blocked on (if any), and its status —
"implemented" (has a passing test, no xfail) or "xfail: <reason>" (not yet
built). Use the id with `agent_dev_loop.py --spec-target <id>` /
`./run_dev_loop.sh --spec-target <id>` to build toward one.

Usage:
    python scripts/list_scenarios.py
    python scripts/list_scenarios.py --xfail-only
"""
import argparse
import json
import sys
from pathlib import Path

SCENARIOS_FILE = Path(__file__).parent.parent / "tests" / "spec_scenarios" / "scenarios.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xfail-only", action="store_true",
                         help="Only show scenarios not yet implemented")
    args = parser.parse_args()

    if not SCENARIOS_FILE.exists():
        sys.exit(f"Not found: {SCENARIOS_FILE}")

    scenarios = json.loads(SCENARIOS_FILE.read_text())
    if args.xfail_only:
        scenarios = [s for s in scenarios if s.get("xfail")]

    if not scenarios:
        print("No scenarios match.")
        return

    id_w = max(len(s["id"]) for s in scenarios)
    ref_w = max(len(s["spec_ref"]) for s in scenarios)

    print(f"{'ID':<{id_w}}  {'SPEC_REF':<{ref_w}}  {'REQ':<4}  STATUS")
    for s in scenarios:
        req = str(s.get("requirement_ref") or "-")
        status = f"xfail: {s['xfail']}" if s.get("xfail") else "implemented"
        print(f"{s['id']:<{id_w}}  {s['spec_ref']:<{ref_w}}  {req:<4}  {status}")

    remaining = [s["id"] for s in scenarios if s.get("xfail")]
    if remaining and not args.xfail_only:
        print(f"\n{len(remaining)} scenario(s) not yet implemented. Build toward one with:")
        print("  ./run_dev_loop.sh --spec-target <id>")


if __name__ == "__main__":
    main()
