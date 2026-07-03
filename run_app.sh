#!/usr/bin/env bash
# Start the Streamlit UI.
# Usage: ./run_app.sh [streamlit options]
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/_common.sh
source scripts/_common.sh

exec streamlit run app.py "$@"
