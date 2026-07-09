"""ParseStrategy interface and the shared state-mutation function every
strategy's output is applied through.

apply_parse_result() is the single, shared function both the legacy and
tool-calling decide paths use to turn a ParseStrategy's output into state
mutations — previously duplicated (agent.py's process_agent_step steps
6-24, and agent_tools.py::_apply_parse_game_response, maintained
separately, with the same bug fixed twice at different times: bug 35 then
bug 75).

ParseResult is a superset schema, not a forced migration — every field is
optional, and a strategy populates whichever fields it can. Two aliases are
recognized for the same information (the legacy schema `extract_knowledge`
already produces, and the newer canonical shape `parse_game_response`
introduced), so existing behavior for the default LLMJsonModeParseStrategy
stays exactly what it was:

    action_result: {"succeeded": bool, "reason_if_failed": str|None}
        Optional. If absent, "take" outcomes are disambiguated by whether
        the item shows up in added_to_inventory/inventory_changes (the
        original ambiguity-handling logic, unrecognized_failure and all);
        non-take verbs are classified from hard/soft failure patterns
        alone (same as today's legacy path).
    room_quote / room: str | None
        Either name works; room_quote wins if both are given. Must be a
        verbatim substring of the response text or it's discarded (bug 74)
        — this applies to BOTH aliases, so the legacy LLM-JSON path gets
        the same protection the tool-calling path already had.
    exits: [str] — grounded against the response text, every direction,
        not just up/down (bug 76).
    objects / npcs: [str] — grounded against the response text (bug 77).
    inventory_changes: [{"item", "change": "gained"|"lost", "cause"}]
        Canonical. added_to_inventory / received_from_npc / taken_by_npc
        are the legacy aliases, still accepted. A hard-failure response
        suppresses any "gained" claim from either source (bug 35 / 75).
    learned_spells, anomalies, resolved_anomalies, blocked_by: [...]
        Legacy-only, optional. Only meaningful to determine_next_action's
        goal-seeking tiers — a strategy that doesn't produce them (the
        tool-calling decide loop, a deterministic parser) simply omits
        them, and that tier never fires. Not a crash, not worth forcing
        every strategy to reconstruct this structure.
    notable_events: [str] — new-path only; ignored by the legacy decide
        path, harmless if present regardless of which decide strategy is
        active.
"""
from pathlib import Path
import json

import agent
from game_config import config
import world_graph

_ORCHESTRATOR_DIR = Path("runs") / "orchestrator"


def _capability_requests_path(run_id):
    directory = _ORCHESTRATOR_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "capability_requests.json"


def append_finding(run_id, step_num, finding_type, severity, summary):
    """Append one finding, in the same shape detect_anomalies.py's own findings use,
    to the per-run capability_requests.json side file (written incrementally so it
    survives an interrupted run — see bug 71)."""
    path = _capability_requests_path(run_id)
    findings = json.loads(path.read_text()) if path.exists() else []
    findings.append({
        "type": finding_type,
        "severity": severity,
        "step_range": [step_num, step_num],
        "summary": summary,
    })
    path.write_text(json.dumps(findings, indent=2))


def split_verb_object(command):
    """Return (verb, object) if command starts with a known candidate verb, else (None, None).

    Sufficient for both decide strategies: determine_next_action always
    formats commands as f"{verb} {target}" when it wants a verb outcome
    recorded (tiers 5/6 — both draw verb from config.candidate_verbs by
    construction), and the tool-calling decide loop's execute_game_command
    commands are free-form text matched the same way."""
    if not command:
        return None, None
    lowered = command.strip().lower()
    for verb in sorted(config.candidate_verbs, key=len, reverse=True):
        prefix = f"{verb} "
        if lowered.startswith(prefix):
            return verb, command[len(prefix):].strip()
    return None, None


def _classify_action_result(action_result, response_text):
    """succeeded/blocked/invalid from an explicit action_result — same
    deterministic hard/soft-failure classification used everywhere else,
    applied to the strategy's own reported reason plus the raw response."""
    if action_result.get("succeeded"):
        return "succeeded"
    combined = f"{action_result.get('reason_if_failed') or ''} {response_text}"
    if agent._is_hard_failure(combined):
        return "invalid"
    if agent._is_soft_failure(combined):
        return "blocked"
    # Unrecognized failure phrasing — default to the safer, retryable
    # classification rather than permanently blacklisting an action that
    # might just need different game state (see "Avoiding wasted
    # repetition" in docs/agent_behavior_spec.md).
    return "blocked"


class ParseStrategy:
    """Interface: turn raw game text into a ParseResult dict. No side effects,
    no decision-making — see this module's docstring for the ParseResult schema."""

    def parse(self, response_text, action_taken):
        raise NotImplementedError


def apply_parse_result(state, result, action_taken, response_text, previous_room):
    """The unified state-mutation function — turns a ParseResult (from any
    ParseStrategy) into state updates, shared by agent.py::process_agent_step
    and agent_tools.py's tool-calling decide loop. Returns (extracted,
    is_death, unrecognized_failure) — extracted is the dict that goes into
    the game_log entry; unrecognized_failure is only ever set by the
    legacy-style "take" disambiguation path (see below), None otherwise.
    """
    result = dict(result)
    resp_lower = response_text.lower()
    action_result = result.get("action_result")  # None for legacy-schema strategies

    verb, obj = split_verb_object(action_taken)
    unrecognized_failure = None
    take_failed = False

    if verb and obj:
        if action_result is not None:
            # New-style: the strategy directly reports success/failure.
            outcome = _classify_action_result(action_result, response_text)
            agent._record_verb_outcome(state, obj, verb, outcome)
            entity = state["known_entities"].get(obj)
            if entity is not None:
                entity["last_result_summary"] = (
                    action_result.get("reason_if_failed") if outcome != "succeeded" else None
                )
            take_failed = verb == "take" and outcome != "succeeded"
        elif verb != "take":
            # Legacy-style, non-take verb: classify from hard/soft failure
            # patterns alone (agent.py's original steps 6-24 behavior).
            if agent._is_soft_failure(response_text):
                agent._record_verb_outcome(state, obj, verb, "blocked")
            elif agent._is_hard_failure(response_text):
                agent._record_verb_outcome(state, obj, verb, "invalid")
            else:
                agent._record_verb_outcome(state, obj, verb, "succeeded")
        else:
            # Legacy-style "take": needs added_to_inventory/inventory_changes
            # to tell a confirmed success apart from an unrecognized response
            # (original ambiguity-handling logic, unchanged).
            if agent._is_soft_failure(response_text):
                agent._record_verb_outcome(state, obj, "take", "blocked")
                take_failed = True
            elif agent._is_hard_failure(response_text):
                agent._record_verb_outcome(state, obj, "take", "invalid")
                take_failed = True
            else:
                taken = {i.lower() for i in result.get("added_to_inventory", [])}
                taken |= {
                    c["item"].lower() for c in result.get("inventory_changes", [])
                    if c.get("change") == "gained" and c.get("item")
                }
                if obj.lower() in taken:
                    agent._record_verb_outcome(state, obj, "take", "succeeded")
                else:
                    state["recheck_inventory"] = True
                    take_failed = True
                    unrecognized_failure = {"target": obj, "action": action_taken, "response": response_text}

        # A failed take doesn't mean the object isn't worth examining (bug
        # 31/70) — only abandon the queued inspection when the game says
        # there's nothing there to look at.
        if take_failed and agent._is_scenery_response(response_text):
            state["current_inspection"]["target"] = None
            state["current_inspection"]["step_index"] = 0

    # Deterministic theft cross-check — always runs regardless of strategy;
    # a repeat pop from an already-accurate inventory_changes claim is a
    # harmless no-op.
    for pattern in config.theft_patterns:
        for m in pattern.finditer(response_text):
            stolen = (m.groupdict().get("item") or "").strip().rstrip(".")
            if stolen:
                state["inventory"] = [i for i in state["inventory"] if i.lower() != stolen.lower()]

    for gift in result.get("received_from_npc", []):
        item = (gift.get("item") or "").strip() if isinstance(gift, dict) else None
        if item and item not in state["inventory"]:
            state["inventory"].append(item)
            entity = state["known_entities"].get(item)
            if entity is None:
                state["known_entities"][item] = {"status": "held", "location": None, "verb_outcomes": {}}
            else:
                entity["status"] = "held"
    for theft in result.get("taken_by_npc", []):
        item = (theft.get("item") or "").strip() if isinstance(theft, dict) else None
        if item:
            state["inventory"] = [i for i in state["inventory"] if i.lower() != item.lower()]

    # Bug 76: every direction grounded, not just up/down.
    exits = None
    if "exits" in result:
        exits = [e for e in result["exits"] if e.lower() in resp_lower]

    # Bug 77: objects/npcs grounded — weaker protection than room (short,
    # generic words can coincidentally match), but real coverage.
    objects = [o for o in result.get("objects", []) if o.lower() in resp_lower]
    npcs = [n for n in result.get("npcs", []) if n.lower() in resp_lower]

    # Bug 74: room_quote (or legacy room) must be grounded — applies to
    # both aliases, so the legacy LLM-JSON path gets this protection too.
    room_claim = result.get("room_quote") or result.get("room")
    room_unresolved = False
    if room_claim and room_claim.lower() in resp_lower:
        state["current_room"] = world_graph.resolve_room_name(state["world_graph"], room_claim, exits)
        state.setdefault("visited_rooms", set()).add(state["current_room"])
        state["position_lost"] = False
        state["position_lost_attempts"] = 0
    elif action_taken in world_graph.DIRECTIONS and (
        (action_result is not None and action_result.get("succeeded"))
        or (action_result is None and not agent._is_hard_failure(response_text) and not agent._is_soft_failure(response_text))
    ):
        # Movement appeared to succeed but no room was parsed — position
        # unknown until a future step resolves it (mirrors the original
        # position_lost handling exactly).
        state["position_lost"] = True
        room_unresolved = True
    elif room_claim:
        # Claimed but not grounded in the response — reject rather than
        # trust it, and surface it for review instead of silently
        # swallowing it. No retry-the-parse loop: a bounded retry against a
        # strategy that's already fabricating is just a new way to get stuck.
        run_id = state.get("_run_id")
        if run_id:
            append_finding(
                run_id, len(state["game_log"]) + 1, "room_not_grounded", "medium",
                f"Strategy claimed room={room_claim!r} but it does not appear in "
                f"the response text — discarded rather than trusted.",
            )

    if exits is not None and not room_unresolved:
        world_graph.update_graph(state, state["current_room"], exits, previous_room, action_taken)

    if agent._is_hard_failure(response_text):
        parts = action_taken.split(" on ", 1) if action_taken else []
        if len(parts) == 2 and action_taken.startswith(("use ", "cast ")):
            state["unresolved_anomalies"].pop(parts[1], None)
        if action_taken and action_taken.startswith(("go to ", "run to ")):
            if state["active_goal"]:
                state["unresolved_anomalies"].pop(state["active_goal"].get("target", ""), None)
            state["active_goal"] = None
            prefix = "go to " if action_taken.startswith("go to ") else "run to "
            nav_target = action_taken[len(prefix):]
            state.setdefault("nav_blacklist", set()).add(nav_target)

    for npc in npcs:
        if npc not in state["known_npcs"]:
            state["known_npcs"][npc] = {"location": state["current_room"], "greeted": False}

    for obj_name in objects:
        if obj_name in state["known_npcs"] or agent._is_creature(obj_name):
            if obj_name not in state["known_npcs"]:
                state["known_npcs"][obj_name] = {"location": state["current_room"], "greeted": False}
            continue
        entity = state["known_entities"].get(obj_name)
        permanently_untakeable = entity is not None and entity.get("verb_outcomes", {}).get("take") == "invalid"
        already_pending_or_held = obj_name in state["uninspected_objects"] or obj_name in state["inventory"]
        if not permanently_untakeable and not already_pending_or_held:
            state["uninspected_objects"].append(obj_name)
            if entity is None:
                state["known_entities"][obj_name] = {"status": "discovered", "location": state["current_room"]}
            else:
                entity["location"] = state["current_room"]

    # Bug 35 / 75: a hard-failure response means the attempted action
    # didn't succeed, so nothing should have been gained as a direct result
    # of it, regardless of what's claimed — applies to both schema aliases.
    gained_items, lost_items = [], []
    if not agent._is_hard_failure(response_text):
        gained_items.extend(result.get("added_to_inventory", []))
        gained_items.extend(
            c["item"] for c in result.get("inventory_changes", [])
            if c.get("change") == "gained" and c.get("item")
        )
    lost_items.extend(
        c["item"] for c in result.get("inventory_changes", [])
        if c.get("change") == "lost" and c.get("item")
    )
    for item in gained_items:
        if item not in state["inventory"]:
            state["inventory"].append(item)
        entity = state["known_entities"].get(item)
        if entity is None:
            state["known_entities"][item] = {"status": "held", "location": None, "verb_outcomes": {}}
        else:
            entity["status"] = "held"
    for item in lost_items:
        state["inventory"] = [i for i in state["inventory"] if i.lower() != item.lower()]

    for spell in result.get("learned_spells", []):
        if spell not in state["spellbook"]:
            state["spellbook"].append(spell)

    for anomaly in result.get("anomalies", []):
        target = anomaly.get("target")
        if target and target not in state["unresolved_anomalies"]:
            state["unresolved_anomalies"][target] = {
                "room": state["current_room"],
                "reason": anomaly.get("reason"),
                "potential_solution": anomaly.get("potential_solution") or "",
            }
    for block in result.get("blocked_by", []):
        obstacle = (block.get("obstacle") or "").strip() if isinstance(block, dict) else None
        if obstacle and obstacle not in state["unresolved_anomalies"]:
            state["unresolved_anomalies"][obstacle] = {
                "room": state["current_room"],
                "reason": block.get("blocking") or "route blocked",
                "potential_solution": "",
            }
    for resolved in result.get("resolved_anomalies", []):
        state["unresolved_anomalies"].pop(resolved, None)

    if action_taken == "score":
        m = agent._SCORE_RE.search(response_text)
        if m:
            state["current_score"] = int(m.group(1))
            state["max_score"] = int(m.group(2))

    if action_taken == "inventory":
        # Reconcile against ground truth whenever the real command runs
        # (see "Handling uncertainty" in the behavior spec) — independent
        # of whether inventory_changes was reported for whatever
        # take/theft/gift preceded it (bug 72's parser, ported).
        parsed = agent._parse_inventory_response(response_text)
        if parsed is not None:
            for item in state["inventory"]:
                if item in state["known_entities"]:
                    state["known_entities"][item]["status"] = "discovered"
            state["inventory"] = parsed
            for item in parsed:
                entity = state["known_entities"].get(item)
                if entity is None:
                    state["known_entities"][item] = {"status": "held", "location": None, "verb_outcomes": {}}
                else:
                    entity["status"] = "held"
        state["recheck_inventory"] = False

    is_death = agent._is_death(response_text)

    extracted = {
        "room": state["current_room"] if room_claim and not room_unresolved else result.get("room"),
        "exits": exits or [],
        "objects": objects,
        "npcs": npcs,
        "learned_spells": result.get("learned_spells", []),
        "anomalies": result.get("anomalies", []),
        "resolved_anomalies": result.get("resolved_anomalies", []),
        "received_from_npc": result.get("received_from_npc", []),
        "taken_by_npc": result.get("taken_by_npc", []),
        "blocked_by": result.get("blocked_by", []),
        "notable_events": result.get("notable_events", []),
    }
    # Bug 35/75: on a hard failure, omit the key entirely rather than set it
    # to an empty list — matches the pre-refactor legacy log shape exactly
    # (extract_knowledge's dict had the key popped, not zeroed), which
    # existing consumers/tests treat as "nothing to report" either way but
    # assert on via strict `not in` checks in a few places.
    if gained_items or not agent._is_hard_failure(response_text):
        extracted["added_to_inventory"] = gained_items
    return extracted, is_death, unrecognized_failure
