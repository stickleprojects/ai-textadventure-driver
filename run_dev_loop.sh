#!/usr/bin/env bash
# Run the agentic dev loop via agent_dev_loop.py.
# Usage: ./run_dev_loop.sh --spec-target SCENARIO_ID [--iterations N] [--model MODEL] [--dry-run] [...]
#        ./run_dev_loop.sh [--steps N] [--iterations N] [--model MODEL] [--dry-run]   # anomaly-based loop
#
# --spec-target builds toward one tests/spec_scenarios/scenarios.json scenario:
# own branch off develop, opens a PR (with real-playthrough evidence attached)
# once the scenario's xfail is lifted and the full suite passes. Requires a
# clean working tree, no PR already open against develop, and the `gh` CLI
# authenticated.
#
# Examples:
#   ./run_dev_loop.sh --spec-target npc_interaction_greet
#   ./run_dev_loop.sh --spec-target npc_interaction_greet --dry-run
#   ./run_dev_loop.sh --steps 50 --iterations 3
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/_common.sh
source scripts/_common.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set (needed for the fixer agent)." >&2
    echo "  Set it in .env, or: ANTHROPIC_API_KEY=sk-... ./run_dev_loop.sh ..." >&2
    exit 1
fi

exec python scripts/agent_dev_loop.py "$@"
