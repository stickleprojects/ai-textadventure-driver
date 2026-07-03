#!/usr/bin/env bash
# Run anomaly detection for a run record.
# Usage: ./run_anomaly.sh [run_filename.json] [--strategy PATH]
#
# If run_filename.json is omitted, the newest JSON file in runs/ is used.
# If provided, the file must exist in runs/.
set -euo pipefail
cd "$(dirname "$0")"

print_help() {
        cat <<'EOF'
Usage: ./run_anomaly.sh [run_filename.json] [--strategy PATH]

Run anomaly detection for a saved run.

Arguments:
    run_filename.json    Optional filename from runs/ (not a path).
                                             If omitted, the newest runs/*.json file is used.

Options:
    --strategy PATH      Strategy JSON passed to detect_anomalies.py.
                                             Default: configs/knight_orc_strategy.json
    -h, --help           Show this help and exit.

Examples:
    ./run_anomaly.sh
    ./run_anomaly.sh watch_20260703_141009.json
    ./run_anomaly.sh watch_20260703_141009.json --strategy configs/knight_orc_strategy.json
EOF
}

for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
                print_help
                exit 0
        fi
done

# shellcheck source=scripts/_common.sh
source scripts/_common.sh

RUNS_DIR="runs"
RUN_FILENAME=""
EXTRA_ARGS=()
STRATEGY_DEFAULT="configs/knight_orc_strategy.json"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    RUN_FILENAME="$1"
    shift
fi

EXTRA_ARGS=("$@")

HAS_STRATEGY_ARG=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--strategy" || "$arg" == --strategy=* ]]; then
        HAS_STRATEGY_ARG=1
        break
    fi
done

if [ "$HAS_STRATEGY_ARG" -eq 0 ]; then
    EXTRA_ARGS+=("--strategy" "$STRATEGY_DEFAULT")
fi

if [ -n "$RUN_FILENAME" ]; then
    if [[ "$RUN_FILENAME" == */* ]]; then
        echo "ERROR: pass only a filename in runs/, not a path: $RUN_FILENAME" >&2
        exit 1
    fi
    if [ ! -f "$RUNS_DIR/$RUN_FILENAME" ]; then
        echo "ERROR: run file not found: $RUNS_DIR/$RUN_FILENAME" >&2
        exit 1
    fi
else
    RUN_FILENAME=$(ls -1t "$RUNS_DIR"/*.json 2>/dev/null | head -1 | xargs -r basename)
    if [ -z "$RUN_FILENAME" ]; then
        echo "ERROR: no run files found in $RUNS_DIR/" >&2
        exit 1
    fi
fi

RUN_ID="${RUN_FILENAME%.json}"

if [ -z "$RUN_ID" ]; then
    echo "ERROR: could not derive run_id from filename: $RUN_FILENAME" >&2
    exit 1
fi

echo "Run file: $RUNS_DIR/$RUN_FILENAME" >&2
echo "Run ID:   $RUN_ID" >&2
if [ "$HAS_STRATEGY_ARG" -eq 0 ]; then
    echo "Strategy: $STRATEGY_DEFAULT (default)" >&2
fi

exec python scripts/detect_anomalies.py "$RUN_ID" "${EXTRA_ARGS[@]}"
