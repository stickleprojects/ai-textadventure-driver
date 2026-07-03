#!/usr/bin/env bash
# Sourced by run_*.sh scripts (never executed directly).
# Caller must cd to the repo root before sourcing.

# 1. Require .env
if [ ! -f .env ]; then
    echo "ERROR: .env not found." >&2
    echo "  cp .env.example .env" >&2
    echo "  # then edit .env and fill in your values" >&2
    exit 1
fi

# 2. Find and activate a virtual environment
_find_and_activate_venv() {
    # Try well-known names first
    for d in .venv venv env virtualenv .virtualenv; do
        if [ -f "$d/bin/activate" ]; then
            [ "$d" != ".venv" ] && echo "INFO: using virtual environment at ./$d" >&2
            # shellcheck source=/dev/null
            source "$d/bin/activate"
            return 0
        fi
    done
    # Broader scan (depth 2, looks for pyvenv.cfg which all venvs contain)
    local found
    found=$(find . -maxdepth 2 -name "pyvenv.cfg" -exec dirname {} \; 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "INFO: using virtual environment at $found" >&2
        # shellcheck source=/dev/null
        source "$found/bin/activate"
        return 0
    fi
    echo "ERROR: no virtual environment found." >&2
    echo "  python -m venv .venv" >&2
    echo "  source .venv/bin/activate" >&2
    echo "  pip install -r requirements.txt" >&2
    exit 1
}
_find_and_activate_venv

# 3. Load .env into the shell environment (handles $VAR expansion, inline comments)
eval "$(python - <<'PYEOF'
import os, sys
sys.path.insert(0, ".")
from env_utils import load_env_file
for k in load_env_file():
    v = os.environ[k].replace("'", "'\\''")
    print(f"export {k}='{v}'")
PYEOF
)"
