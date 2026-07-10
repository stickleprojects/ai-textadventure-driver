"""Classify the outcome of a completed agent run and manage the cross-run strategy file.

Outcome categories
------------------
finished        — "congratulations" / "you have finished" matched
game_ended      — other end_state_pattern matched (death, score summary)
score_improved  — final_score > 0 and score was recorded during the run
agent_failure   — loop_detected, crash, or timeout_or_error finding present
interrupted     — user pressed Ctrl+C; logs saved, run ended early
ambiguous       — hit max steps, no score change, no error, no end state

Crash sub-classification (classify_finding)
-------------------------------------------
config_fix  — failure response text not matched by any existing pattern in config
python_fix  — finding type is "crash" (exception traceback present)
ambiguous   — neither of the above; needs human review

Strategy file layout (merge_run_record / load_strategy)
--------------------------------------------------------
The strategy is split across three files derived from the base strategy path:

  knight_orc_strategy.json — futile_edges, run_history
  knight_orc_rooms.json    — world_graph (nodes + edges)
  knight_orc_items.json    — entity_verb_outcomes

Keeping the large, frequently updated sections in separate files makes diffs
readable and lets the user inspect or clear rooms/items independently.

load_strategy() and merge_run_record() handle the split transparently; callers
pass the strategy path as before. Old single-file format (all keys in the main
file) is still read correctly for backward compatibility.
"""
import json
import re
from pathlib import Path

from game_config import config

_SCORE_RE = re.compile(r"you score\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)


def _scan_for_end_state(game_log):
    """Return (category, matched_text) for the first end-state hit, or (None, None)."""
    for entry in reversed(game_log[-20:]):
        text = entry.get("response", "")
        for category, patterns in config.end_state_patterns.items():
            for pat in patterns:
                if pat.search(text):
                    return category, text
    return None, None


def classify_run(game_log, findings, final_score=None):
    """Return one of the six outcome strings for this run.

    Parameters
    ----------
    game_log : list[dict]   — the agent's game_log list
    findings : list[dict]   — findings list from watch_run.run()
    final_score : int|None  — state["current_score"] at run end, or None
    """
    finding_types = {f.get("type") for f in findings}

    if "interrupted" in finding_types:
        return "interrupted"

    end_category, _ = _scan_for_end_state(game_log)

    if end_category == "finished":
        return "finished"

    if finding_types & {"loop", "timeout_or_error", "crash", "startup_error"}:
        return "agent_failure"

    if end_category in ("death", "score"):
        return "game_ended"

    if final_score is not None and final_score > 0:
        return "score_improved"

    return "ambiguous"


def _sidecar(strategy_path, kind):
    """Derive a sidecar path for 'rooms' or 'items' from the strategy path.

    'configs/knight_orc_strategy.json' → 'configs/knight_orc_rooms.json'
    'tmp/strategy.json'                → 'tmp/rooms.json'
    """
    p = Path(strategy_path)
    stem = p.stem[: -len("_strategy")] if p.stem.endswith("_strategy") else p.stem
    return p.with_name(f"{stem}_{kind}.json")


def load_strategy(strategy_path):
    """Load accumulated strategy from file. Returns empty defaults if file absent.

    Reads the main strategy file for futile_edges and run_history, then checks
    for sidecar files (rooms, items) derived from the strategy path. Falls back
    to inline keys in the main file for backward compatibility.

    Returns dict with:
        futile_edges         — set of (room, direction) tuples
        run_history          — list of {run_id, outcome, final_score,
                                locations_discovered, npcs_discovered,
                                treasure_discovered, puzzles_discovered,
                                puzzles_solved} dicts (the metrics fields are
                                optional/None on older entries)
        entity_verb_outcomes — {entity_name: {verb: "invalid"}} persisted hard failures
        world_graph          — {"nodes": [...], "edges": [[src, dst, label], ...]}
    """
    path = Path(strategy_path)
    data = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)

    rooms_path = _sidecar(strategy_path, "rooms")
    if rooms_path.exists():
        with open(rooms_path) as f:
            world_graph = json.load(f)
    else:
        world_graph = data.get("world_graph", {"nodes": [], "edges": []})

    items_path = _sidecar(strategy_path, "items")
    if items_path.exists():
        with open(items_path) as f:
            entity_verb_outcomes = json.load(f)
    else:
        entity_verb_outcomes = data.get("entity_verb_outcomes", {})

    return {
        "futile_edges": {tuple(e) for e in data.get("futile_edges", [])},
        "run_history": data.get("run_history", []),
        "entity_verb_outcomes": entity_verb_outcomes,
        "world_graph": world_graph,
    }


_COMPARISON_METRICS = [
    "final_score",
    "locations_discovered",
    "npcs_discovered",
    "treasure_discovered",
    "puzzles_discovered",
    "puzzles_solved",
]


def compute_playthrough_metrics(state, game_log):
    """Return discovery/progress metrics for a finished run.

    Derived entirely from game_log (per-step "extracted" dicts) plus
    visited_rooms/known_npcs, both of which are fresh per run — unlike
    known_entities, which is pre-seeded from cross-run strategy (bug 64) and
    would double-count history as "discovered this run" if used here.
    Treasure is a name heuristic (Knight Orc treasure objects are all named
    with "silver"), not a separate extraction field.
    """
    treasure, discovered, solved = set(), set(), set()
    for entry in game_log:
        extracted = entry.get("extracted") or {}
        for name in list(extracted.get("objects", [])) + list(extracted.get("added_to_inventory", [])):
            if "silver" in name.lower():
                treasure.add(name.lower())
        for anomaly in extracted.get("anomalies", []):
            target = anomaly.get("target")
            if target:
                discovered.add(target)
        for resolved in extracted.get("resolved_anomalies", []):
            solved.add(resolved)
    return {
        "locations_discovered": len(state.get("visited_rooms", set())),
        "npcs_discovered": len(state.get("known_npcs", {})),
        "treasure_discovered": len(treasure),
        "puzzles_discovered": len(discovered),
        "puzzles_solved": len(solved & discovered),
    }


def compare_to_history(run_record, run_history):
    """Compare one run's metrics against the best prior value of each metric
    already recorded in run_history. Mirrors anomaly_detector._regression()'s
    current-vs-best-prior-score pattern, generalised to all playthrough metrics.

    Returns {metric: {"current": v, "best_prior": v|None}} — best_prior is
    None (not 0) when no prior run recorded that metric, so callers can tell
    "no history yet" apart from "history says 0".
    """
    comparison = {}
    for metric in _COMPARISON_METRICS:
        scored = [h[metric] for h in run_history if h.get(metric) is not None]
        comparison[metric] = {
            "current": run_record.get(metric),
            "best_prior": max(scored) if scored else None,
        }
    return comparison


def build_run_record(state, run_id, findings=None):
    """Assemble the per-run summary dict written to runs/<run_id>.json.

    Shared by scripts/watch_run.py (end of a headless run) and app.py
    (after every GUI step, so a GUI session can be pointed at
    detect_anomalies.py at any time). `findings` feeds classify_run's
    loop/timeout/crash/interrupted detection — pass None (or omit) for
    callers, like the GUI, that don't track findings themselves.
    """
    game_log = state["game_log"]
    outcome = classify_run(game_log, findings or [], state.get("current_score"))
    entity_verb_outcomes = {
        name: {v: o for v, o in data.get("verb_outcomes", {}).items() if o == "invalid"}
        for name, data in state.get("known_entities", {}).items()
        if any(o == "invalid" for o in data.get("verb_outcomes", {}).values())
    }
    g = state["world_graph"]
    world_graph_record = {
        "nodes": [n for n in g.nodes if not n.startswith("Unknown")],
        "edges": [
            [u, v, d.get("label", "")]
            for u, v, d in g.edges(data=True)
            if not u.startswith("Unknown") and not v.startswith("Unknown")
        ],
    }
    run_record = {
        "run_id": run_id,
        "outcome": outcome,
        "final_score": state.get("current_score"),
        "max_score": state.get("max_score"),
        "steps": len(game_log),
        "futile_edges": [list(e) for e in sorted(state.get("futile_edges", set()))],
        "entity_verb_outcomes": entity_verb_outcomes,
        "world_graph": world_graph_record,
        **compute_playthrough_metrics(state, game_log),
    }
    total_input = sum(e.get("token_usage", {}).get("input_tokens", 0) for e in game_log)
    total_output = sum(e.get("token_usage", {}).get("output_tokens", 0) for e in game_log)
    run_record["token_usage"] = {"input_tokens": total_input, "output_tokens": total_output}
    return run_record


def merge_run_record(run_record, strategy_path):
    """Merge a completed run's data into the accumulated strategy file.

    Unions futile_edges across runs; appends a summary entry to run_history.
    Creates the file (and its parent directory) if absent.
    """
    strategy = load_strategy(strategy_path)

    new_edges = {tuple(e) for e in run_record.get("futile_edges", [])}
    strategy["futile_edges"] |= new_edges

    # Union invalid (hard-failure) verb outcomes — blocked outcomes are state-dependent
    # and must not be persisted, as they may succeed in a future run.
    for entity, verbs in run_record.get("entity_verb_outcomes", {}).items():
        existing = strategy["entity_verb_outcomes"].setdefault(entity, {})
        for verb, outcome in verbs.items():
            if outcome == "invalid":
                existing[verb] = "invalid"

    strategy["run_history"].append({
        "run_id": run_record["run_id"],
        "outcome": run_record["outcome"],
        "final_score": run_record.get("final_score"),
        "locations_discovered": run_record.get("locations_discovered"),
        "npcs_discovered": run_record.get("npcs_discovered"),
        "treasure_discovered": run_record.get("treasure_discovered"),
        "puzzles_discovered": run_record.get("puzzles_discovered"),
        "puzzles_solved": run_record.get("puzzles_solved"),
    })

    # Merge world graph — union nodes; each (src, dst, direction) triplet stored
    # separately. Legacy compound labels ("south/out") are split on load.
    # Unknown placeholder nodes are excluded: they are transient and re-discovered
    # naturally each run.
    acc = strategy["world_graph"]
    known_nodes = set(acc["nodes"])
    known_edge_set = set()
    for u, v, label in acc["edges"]:
        for part in label.split("/"):
            if part:
                known_edge_set.add((u, v, part))
    for node in run_record.get("world_graph", {}).get("nodes", []):
        if not node.startswith("Unknown") and node not in known_nodes:
            known_nodes.add(node)
            acc["nodes"].append(node)
    for u, v, label in run_record.get("world_graph", {}).get("edges", []):
        if u.startswith("Unknown") or v.startswith("Unknown"):
            continue
        for part in label.split("/"):
            if part:
                known_edge_set.add((u, v, part))
    acc["edges"] = [[u, v, label] for u, v, label in sorted(known_edge_set)]

    path = Path(strategy_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {
                "futile_edges": [list(e) for e in sorted(strategy["futile_edges"])],
                "run_history": strategy["run_history"],
            },
            f,
            indent=2,
        )

    rooms_path = _sidecar(strategy_path, "rooms")
    with open(rooms_path, "w") as f:
        json.dump(acc, f, indent=2)

    items_path = _sidecar(strategy_path, "items")
    with open(items_path, "w") as f:
        json.dump(strategy["entity_verb_outcomes"], f, indent=2)

    return strategy


def classify_finding(finding, game_log):
    """Sub-classify a single finding from watch_run for triage.

    Returns 'config_fix', 'python_fix', or 'ambiguous'.
    """
    if finding.get("type") == "crash":
        return "python_fix"

    response = finding.get("response") or finding.get("last_response") or ""
    if response and not config.failure_pattern.search(response):
        return "config_fix"

    return "ambiguous"
