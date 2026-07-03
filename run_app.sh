#!/usr/bin/env bash
# Start the Streamlit UI.
# Usage: ./run_app.sh [streamlit options]
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

# Load .env — uses the same loader as the Python app so $VAR expansion works
eval "$(python - <<'EOF'
import os, sys
sys.path.insert(0, ".")
from env_utils import load_env_file
for k in load_env_file():
    v = os.environ[k].replace("'", "'\\''")
    print(f"export {k}='{v}'")
EOF
)"

exec streamlit run app.py "$@"
