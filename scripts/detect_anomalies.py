#!/usr/bin/env python3
"""Turn a watch_run.py run into a structured anomaly_report.json.

Detection logic lives in anomaly_detector.py (unit-tested); this script is a
thin CLI that loads a run's log/run-record pair, calls it, and writes the
result. This is stage 1 (Detector) of the dev-loop orchestrator pipeline —
see project memory "plan-dev-orchestrator" for the full JSON contract.

Usage:
    python scripts/detect_anomalies.py <run_id> [--strategy PATH]
    python scripts/detect_anomalies.py watch_20260702_143000

Reads:
    logs/<run_id>.json   — game_log, written by watch_run.py
    runs/<run_id>.json   — run record, written by watch_run.py

Writes:
    runs/orchestrator/<run_id>/anomaly_report.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detector import build_report
from run_evaluator import load_strategy

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")
STRATEGY_PATH = "configs/knight_orc_strategy.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID, e.g. watch_20260702_143000 (matches logs/<run_id>.json)")
    parser.add_argument("--strategy", default=STRATEGY_PATH, help="Path to strategy JSON (default: %(default)s)")
    args = parser.parse_args()

    log_path = LOG_DIR / f"{args.run_id}.json"
    run_path = RUNS_DIR / f"{args.run_id}.json"
    for path in (log_path, run_path):
        if not path.exists() or path.stat().st_size == 0:
            print(f"ERROR: {path} is missing or empty — run was likely interrupted before it could write.", file=sys.stderr)
            sys.exit(2)
    game_log = json.loads(log_path.read_text())
    run_record = json.loads(run_path.read_text())
    strategy = load_strategy(args.strategy)

    report = build_report(args.run_id, log_path, run_path, game_log, run_record, strategy)

    out_dir = RUNS_DIR / "orchestrator" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}")
    print(f"{len(report['anomalies'])} anomaly(ies) found.")
    for a in report["anomalies"]:
        print(f"  [{a['severity']}] {a['id']} {a['type']}: {a['summary']}")

    sys.exit(1 if report["anomalies"] else 0)


if __name__ == "__main__":
    main()
