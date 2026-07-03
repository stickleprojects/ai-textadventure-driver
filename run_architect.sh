#!/usr/bin/env bash
# Run architect planning for a run record.
# Usage: ./run_architect.sh [run_filename.json] [force|--force] [--all|--anomaly-id ID] [--model MODEL]
#
# If run_filename.json is omitted, the newest JSON file in runs/ is used.
# If provided, the file must exist in runs/.
# If anomaly_report.json is missing, empty, or has zero anomalies (or force/--force is set),
# anomaly detection is run first via ./run_anomaly.sh.
# To match watch_run --architect behavior, this script runs llm_review before architect.
set -euo pipefail
cd "$(dirname "$0")"

print_help() {
    cat <<'EOF'
Usage: ./run_architect.sh [run_filename.json] [force|--force] [--all|--anomaly-id ID] [--model MODEL]

Run architect planning for a saved run.

Arguments:
  run_filename.json     Optional filename from runs/ (not a path).
                        If omitted, the newest runs/*.json file is used.
  force, --force        Always run anomaly detection before review/architect.

Behavior:
  If runs/orchestrator/<run_id>/anomaly_report.json is missing, empty, or has
  no anomalies, anomaly detection is run automatically via ./run_anomaly.sh.
  Then llm_review runs, then architect runs.

Options (forwarded to scripts/architect.py):
  --all                 Produce plans for all anomalies.
  --anomaly-id ID       Produce a plan for one anomaly id.
  --model MODEL         Claude model for architect (default from architect.py).
  -h, --help            Show this help and exit.

Examples:
  ./run_architect.sh
  ./run_architect.sh watch_20260703_141009.json
  ./run_architect.sh watch_20260703_141009.json --force --all
  ./run_architect.sh watch_20260703_141009.json --anomaly-id a1 --model claude-opus-4-8
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
FORCE_ANOMALY=0
ARCHITECT_ARGS=()
SUMMARY_NEW_PLANS=0
SUMMARY_DUPLICATES=0

print_summary() {
    local skipped_reason="${1:-none}"
    echo "Summary: new plans written=${SUMMARY_NEW_PLANS}, duplicates only=${SUMMARY_DUPLICATES}, skipped=${skipped_reason}" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        force|--force)
            FORCE_ANOMALY=1
            shift
            ;;
        --all)
            ARCHITECT_ARGS+=("$1")
            shift
            ;;
        --anomaly-id|--model)
            ARCHITECT_ARGS+=("$1")
            shift
            if [ "$#" -gt 0 ]; then
                ARCHITECT_ARGS+=("$1")
                shift
            fi
            ;;
        --anomaly-id=*|--model=*)
            ARCHITECT_ARGS+=("$1")
            shift
            ;;
        --*)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [ -z "$RUN_FILENAME" ]; then
                RUN_FILENAME="$1"
                shift
            else
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
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
anomaly_count_from_report() {
    local report_path="$1"
    python - "$report_path" <<'PYEOF'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print(0)
    raise SystemExit(0)

try:
    data = json.loads(path.read_text())
except Exception:
    print(0)
    raise SystemExit(0)

anomalies = data.get("anomalies", [])
print(len(anomalies) if isinstance(anomalies, list) else 0)
PYEOF
}

ANOMALY_COUNT=0
if [ -s "$REPORT_PATH" ]; then
    ANOMALY_COUNT=$(anomaly_count_from_report "$REPORT_PATH")
fi

if [ "$FORCE_ANOMALY" -eq 1 ] || [ ! -s "$REPORT_PATH" ] || [ "$ANOMALY_COUNT" -eq 0 ]; then
    if [ "$FORCE_ANOMALY" -eq 1 ]; then
        echo "Anomaly detection: forced (force/--force provided)" >&2
    elif [ ! -s "$REPORT_PATH" ]; then
        echo "Anomaly detection: report missing or empty at $REPORT_PATH" >&2
    else
        echo "Anomaly detection: no anomalies found in $REPORT_PATH" >&2
    fi

    ANOMALY_STATUS=0
    ./run_anomaly.sh "$RUN_FILENAME" || ANOMALY_STATUS=$?
    if [ "$ANOMALY_STATUS" -ne 0 ] && [ "$ANOMALY_STATUS" -ne 1 ]; then
        exit "$ANOMALY_STATUS"
    fi

    ANOMALY_COUNT=$(anomaly_count_from_report "$REPORT_PATH")
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set; required for scripts/llm_review.py" >&2
    print_summary "missing ANTHROPIC_API_KEY"
    exit 1
fi

REVIEW_STATUS=0
python scripts/llm_review.py "$RUN_ID" || REVIEW_STATUS=$?
if [ "$REVIEW_STATUS" -ne 0 ] && [ "$REVIEW_STATUS" -ne 1 ]; then
    print_summary "llm_review failed"
    exit "$REVIEW_STATUS"
fi

ANOMALY_COUNT=$(anomaly_count_from_report "$REPORT_PATH")
if [ "$ANOMALY_COUNT" -eq 0 ]; then
    echo "No anomalies found after review; skipping architect." >&2
    print_summary "no anomalies after review"
    exit "$REVIEW_STATUS"
fi

ARCH_OUT=$(mktemp)
ARCH_STATUS=0
python scripts/architect.py "$RUN_ID" "${ARCHITECT_ARGS[@]}" 2>&1 | tee "$ARCH_OUT"
ARCH_STATUS=${PIPESTATUS[0]}

SUMMARY_LINE=$(grep -E '[0-9]+ new plan\(s\), [0-9]+ duplicate\(s\) recorded\.' "$ARCH_OUT" | tail -1 || true)
if [ -n "$SUMMARY_LINE" ]; then
    SUMMARY_NEW_PLANS=$(echo "$SUMMARY_LINE" | sed -E 's/^.*([0-9]+) new plan\(s\), ([0-9]+) duplicate\(s\) recorded\..*$/\1/')
    SUMMARY_DUPLICATES=$(echo "$SUMMARY_LINE" | sed -E 's/^.*([0-9]+) new plan\(s\), ([0-9]+) duplicate\(s\) recorded\..*$/\2/')
fi
rm -f "$ARCH_OUT"

if [ "$ARCH_STATUS" -ne 0 ]; then
    print_summary "architect failed"
    exit "$ARCH_STATUS"
fi

print_summary "none"
exit "$REVIEW_STATUS"
