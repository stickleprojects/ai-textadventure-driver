#!/usr/bin/env python3
"""Reconstruct the world graph from a saved run log and save a PNG map.

Usage:
    python scripts/generate_map.py logs/watch_20260701_085624.json
    python scripts/generate_map.py logs/watch_20260701_085624.json --out my_map.png
    python scripts/generate_map.py logs/watch_20260701_085624.json --no-unknowns
"""
import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import save_graph_image
from world_graph import resolve_room_name, update_graph


def rebuild_graph(game_log):
    """Replay log entries to reconstruct the world graph and last known room."""
    graph = nx.MultiDiGraph()
    state = {"world_graph": graph, "current_room": "Unknown Location"}
    previous_room = None

    for entry in game_log:
        action = entry.get("action", "")
        extracted = entry.get("extracted") or {}
        room = extracted.get("room")
        exits = extracted.get("exits") or []

        if room:
            state["current_room"] = resolve_room_name(state["world_graph"], room, exits)

        update_graph(state, state["current_room"], exits, previous_room, action)
        previous_room = state["current_room"]

    return graph, state["current_room"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="Path to a run log JSON file (logs/<run_id>.json)")
    parser.add_argument("--out", metavar="PATH", help="Output PNG path (default: <log_stem>_map.png)")
    parser.add_argument(
        "--no-unknowns", action="store_true",
        help="Exclude unexplored exit placeholder nodes (? north, ? up, etc.) from the map",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path) as f:
        game_log = json.load(f)

    graph, current_room = rebuild_graph(game_log)

    if args.no_unknowns:
        unknowns = [n for n in list(graph.nodes) if n.startswith("Unknown (")]
        graph.remove_nodes_from(unknowns)
        print(f"Removed {len(unknowns)} unexplored exit nodes", file=sys.stderr)

    print(f"Rebuilt graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges", file=sys.stderr)

    out_path = Path(args.out) if args.out else log_path.with_name(log_path.stem + "_map.png")
    save_graph_image(graph, current_room, out_path)
    print(f"Map written to {out_path}")


if __name__ == "__main__":
    main()
