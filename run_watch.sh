#!/usr/bin/env bash
# Run the headless agent via watch_run.py.
# Usage: ./run_watch.sh [steps] [--detect] [--review] [--architect] [--verbose] [...]
#   steps   Number of game steps (default: 50)
#
# Examples:
#   ./run_watch.sh
#   ./run_watch.sh 100 --detect --review --architect
#   ./run_watch.sh 200 --verbose
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

PROVIDER="${LLM_PROVIDER:-local}"
if [ "$PROVIDER" = "local" ]; then
    echo "LLM: local llama.cpp  model=${LLM_MODEL_PATH:-<not set>}" >&2
else
    echo "LLM: $PROVIDER  model=${LLM_MODEL:-deepseek-chat}" >&2
fi

exec python scripts/watch_run.py "$@"
