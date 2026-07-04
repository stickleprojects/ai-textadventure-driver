from run_evaluator import compare_to_history, compute_playthrough_metrics


def _entry(objects=None, added_to_inventory=None, anomalies=None, resolved_anomalies=None):
    extracted = {}
    if objects is not None:
        extracted["objects"] = objects
    if added_to_inventory is not None:
        extracted["added_to_inventory"] = added_to_inventory
    if anomalies is not None:
        extracted["anomalies"] = anomalies
    if resolved_anomalies is not None:
        extracted["resolved_anomalies"] = resolved_anomalies
    return {"extracted": extracted}


# ── compute_playthrough_metrics ────────────────────────────────────────────────

def test_locations_and_npcs_come_from_state_not_game_log():
    state = {"visited_rooms": {"Courtyard", "Cellar"}, "known_npcs": {"Grendel": {}}}
    metrics = compute_playthrough_metrics(state, [])
    assert metrics["locations_discovered"] == 2
    assert metrics["npcs_discovered"] == 1


def test_treasure_detected_by_silver_name_heuristic():
    game_log = [
        _entry(objects=["silver goblet", "wooden chair"]),
        _entry(added_to_inventory=["silver coin"]),
    ]
    metrics = compute_playthrough_metrics({}, game_log)
    assert metrics["treasure_discovered"] == 2


def test_treasure_deduplicated_case_insensitively():
    game_log = [
        _entry(objects=["Silver Goblet"]),
        _entry(objects=["silver goblet"]),
    ]
    metrics = compute_playthrough_metrics({}, game_log)
    assert metrics["treasure_discovered"] == 1


def test_puzzles_discovered_and_solved():
    game_log = [
        _entry(anomalies=[{"target": "north door", "reason": "locked"}]),
        _entry(anomalies=[{"target": "oak chest", "reason": "locked"}]),
        _entry(resolved_anomalies=["north door"]),
    ]
    metrics = compute_playthrough_metrics({}, game_log)
    assert metrics["puzzles_discovered"] == 2
    assert metrics["puzzles_solved"] == 1


def test_resolved_anomaly_not_previously_discovered_is_not_counted_as_solved():
    game_log = [_entry(resolved_anomalies=["never seen before"])]
    metrics = compute_playthrough_metrics({}, game_log)
    assert metrics["puzzles_discovered"] == 0
    assert metrics["puzzles_solved"] == 0


def test_empty_game_log_and_state_yields_zeros():
    metrics = compute_playthrough_metrics({}, [])
    assert metrics == {
        "locations_discovered": 0,
        "npcs_discovered": 0,
        "treasure_discovered": 0,
        "puzzles_discovered": 0,
        "puzzles_solved": 0,
    }


# ── compare_to_history ─────────────────────────────────────────────────────────

def test_compare_to_history_no_prior_runs_gives_none_not_zero():
    run_record = {"final_score": 5, "locations_discovered": 3}
    comparison = compare_to_history(run_record, [])
    assert comparison["final_score"] == {"current": 5, "best_prior": None}
    assert comparison["locations_discovered"] == {"current": 3, "best_prior": None}


def test_compare_to_history_uses_max_of_prior_runs():
    run_record = {"final_score": 5, "locations_discovered": 3, "npcs_discovered": 1,
                  "treasure_discovered": 0, "puzzles_discovered": 0, "puzzles_solved": 0}
    run_history = [
        {"final_score": 2, "locations_discovered": 7},
        {"final_score": 8, "locations_discovered": 1},
    ]
    comparison = compare_to_history(run_record, run_history)
    assert comparison["final_score"] == {"current": 5, "best_prior": 8}
    assert comparison["locations_discovered"] == {"current": 3, "best_prior": 7}


def test_compare_to_history_ignores_prior_entries_missing_the_metric():
    run_record = {"puzzles_solved": 4}
    run_history = [{"outcome": "ambiguous"}, {"puzzles_solved": 2}]
    comparison = compare_to_history(run_record, run_history)
    assert comparison["puzzles_solved"] == {"current": 4, "best_prior": 2}
