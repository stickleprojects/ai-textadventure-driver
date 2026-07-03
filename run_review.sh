#!/usr/bin/env bash
# Run LLM review for a run record.
# Usage: ./run_review.sh [run_filename.json] [force|--force] [--architect] [--model MODEL]
#
# If run_filename.json is omitted, the newest JSON file in runs/ is used.
# If provided, the file must exist in runs/.
# If anomaly_report.json is missing (or force/--force is set), anomaly detection
# is run first via ./run_anomaly.sh.
set -euo pipefail
cd "$(dirname "$0")"

print_help() {
        cat <<'EOF'
Usage: ./run_review.sh [run_filename.json] [force|--force] [--architect] [--model MODEL]

Run LLM review for a saved run.

Arguments:
    run_filename.json    Optional filename from runs/ (not a path).
                                             If omitted, the newest runs/*.json file is used.
    force, --force       Always run anomaly detection before LLM review.

Behavior:
    If runs/orchestrator/<run_id>/anomaly_report.json is missing or empty,
    anomaly detection is run automatically via ./run_anomaly.sh.
    If --architect is passed, architect is run after review.

Options:
    --model MODEL        Claude model passed to llm_review.py.
    --architect          Run scripts/architect.py after review.
    -h, --help           Show this help and exit.

Examples:
    ./run_review.sh
    ./run_review.sh watch_20260703_141009.json
    ./run_review.sh watch_20260703_141009.json --force --model claude-sonnet-4-6
    ./run_review.sh watch_20260703_141009.json --architect
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
FORCE_ANOMALY=0
RUN_ARCHITECT=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        force|--force)
            FORCE_ANOMALY=1
            shift
            ;;
        --architect)
            RUN_ARCHITECT=1
            shift
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            shift
            if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            if [ -z "$RUN_FILENAME" ]; then
                RUN_FILENAME="$1"
                shift
            else
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

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

REPORT_PATH="$RUNS_DIR/orchestrator/$RUN_ID/anomaly_report.json"
if [ "$FORCE_ANOMALY" -eq 1 ] || [ ! -s "$REPORT_PATH" ]; then
    if [ "$FORCE_ANOMALY" -eq 1 ]; then
        echo "Anomaly detection: forced (force/--force provided)" >&2
    else
        echo "Anomaly detection: report missing or empty at $REPORT_PATH" >&2
    fi
    ANOMALY_STATUS=0
    ./run_anomaly.sh "$RUN_FILENAME" || ANOMALY_STATUS=$?
    if [ "$ANOMALY_STATUS" -ne 0 ] && [ "$ANOMALY_STATUS" -ne 1 ]; then
        exit "$ANOMALY_STATUS"
    fi
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set; required for scripts/llm_review.py" >&2
    exit 1
fi

REVIEW_STATUS=0
python scripts/llm_review.py "$RUN_ID" "${EXTRA_ARGS[@]}" || REVIEW_STATUS=$?
if [ "$REVIEW_STATUS" -ne 0 ] && [ "$REVIEW_STATUS" -ne 1 ]; then
    exit "$REVIEW_STATUS"
fi

if [ "$RUN_ARCHITECT" -eq 1 ]; then
    python scripts/architect.py "$RUN_ID"
fi

exit "$REVIEW_STATUS"
