#!/usr/bin/env bash
# Run LLM extraction evals.
# Usage: ./run_evals.sh [-k FILTER] [pytest options]
#   -k FILTER   Run only cases matching a substring, e.g. -k taken_no_room
#
# Requires a local llama.cpp model. Set EVAL_MODEL_PATH to override the default:
#   EVAL_MODEL_PATH=../models/my-model.gguf ./run_evals.sh
#
# Examples:
#   ./run_evals.sh
#   ./run_evals.sh -k horse_is_npc
#   EVAL_THRESHOLD=0.8 ./run_evals.sh
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

eval "$(python - <<'EOF'
import os, sys
sys.path.insert(0, ".")
from env_utils import load_env_file
for k in load_env_file():
    v = os.environ[k].replace("'", "'\\''")
    print(f"export {k}='{v}'")
EOF
)"

MODEL="${EVAL_MODEL_PATH:-../models/Phi-3.5-mini-instruct-Q3_K_M.gguf}"
THRESHOLD="${EVAL_THRESHOLD:-0.7}"

echo "Model:     $MODEL" >&2
echo "Threshold: $THRESHOLD" >&2

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "Set EVAL_MODEL_PATH to a valid .gguf file." >&2
    exit 1
fi

exec pytest tests/test_evals.py -m llm -v "$@"
