"""Tool-calling main loop — cloud/tool-use-capable LLM backends only.

Implements docs/agent_tools_spec.md and docs/main_loop_prompt.md: one LLM
tool-calling episode per step, replacing agent.py::determine_next_action's
fixed priority list + llm.py::extract_knowledge's separate extraction call.

The local llama.cpp path (agent.py::process_agent_step) is untouched and
unaffected by this module — see the "Scope" section of
docs/main_loop_prompt.md for why.

Step contract (see docs/agent_tools_spec.md): parse_game_response is the
mandatory first tool call each step, describing the response to the
PREVIOUS step's execute_game_command (or the game's boot text, on the
first step). Because of this one-step lag, run_tool_calling_step finalizes
the previous step's log entry (using this step's parse_game_response call)
before deciding and executing the new action.
"""
import json
from datetime import datetime
from pathlib import Path

import agent
from game_config import config
from game_engine import execute_game_command as _run_game_command

_NON_TERMINAL_CALL_CAP = 6
_FALLBACK_COMMAND = "look"
_ORCHESTRATOR_DIR = Path("runs") / "orchestrator"
_BEHAVIOR_SPEC_PATH = Path(__file__).parent / "docs" / "agent_behavior_spec.md"

_NUDGE_MESSAGE = (
    f"You've reached this step's tool-call limit ({_NON_TERMINAL_CALL_CAP}). "
    "Call execute_game_command now to finish this step."
)
_ORDERING_CORRECTION = (
    "parse_game_response must be called first, before any other tool, every step."
)

_MANDATORY_FIRST_TOOL = "parse_game_response"
_TERMINAL_TOOL = "execute_game_command"

# Some models emit the literal string "null" (or similar) for a nullable field
# instead of an actual null/None — observed live with DeepSeek on the room
# field. Treat these the same as a real null rather than a room name.
_NULL_LIKE_ROOM_VALUES = {"null", "none", "n/a", "unknown", "nil", ""}
_NON_TERMINAL_TOOLS = {"query_map", "query_entity_history", "request_capability",
                        "write_journal", "search_journal"}

# ---------------------------------------------------------------------------
# Tool schemas — canonical, backend-neutral (Anthropic's native input_schema
# shape; llm.py's OpenAIToolAdapter translates this for OpenAI-style function
# calling). See docs/agent_tools_spec.md for the authoritative contract.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "parse_game_response",
        "description": (
            "Parse what happened in the response you were just shown, before doing "
            "anything else. Mandatory first tool call every step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action_result": {
                    "type": "object",
                    "properties": {
                        "succeeded": {"type": "boolean"},
                        "reason_if_failed": {"type": ["string", "null"]},
                    },
                    "required": ["succeeded"],
                },
                "room": {
                    "type": ["string", "null"],
                    "description": (
                        "Only set if the response explicitly names a location (e.g. \"You are in "
                        "the Great Hall\"). Use null — never a placeholder word or your own summary "
                        "of what you inferred happened — if the response is terse, describes an "
                        "object/action without naming a place, or you're inferring/guessing rather "
                        "than reading an explicit name."
                    ),
                },
                "exits": {"type": "array", "items": {"type": "string"}},
                "objects": {"type": "array", "items": {"type": "string"}, "description": "Inanimate items visible"},
                "npcs": {"type": "array", "items": {"type": "string"}, "description": "Living creatures/characters visible"},
                "inventory_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                            "change": {"type": "string", "enum": ["gained", "lost"]},
                            "cause": {"type": "string"},
                        },
                        "required": ["item", "change"],
                    },
                },
                "notable_events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Anything else worth remembering, in plain language",
                },
            },
            "required": ["action_result"],
        },
    },
    {
        "name": "query_map",
        "description": "Look up what's known about a room without holding the whole map in context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room": {"type": ["string", "null"], "description": "Defaults to current room"},
            },
        },
    },
    {
        "name": "query_entity_history",
        "description": (
            "Check what's already been tried on an object, and what happened, before "
            "trying a specific verb on it again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
            },
            "required": ["object"],
        },
    },
    {
        "name": "request_capability",
        "description": (
            "Record that something needed for reliable tracking doesn't exist yet, "
            "instead of guessing or re-deriving the answer from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "write_journal",
        "description": (
            "Write a note about something worth remembering later — an obstacle and "
            "what it needs, something seen but not ready to deal with yet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string"},
            },
            "required": ["note"],
        },
    },
    {
        "name": "search_journal",
        "description": (
            "Search notes written earlier with write_journal — keyword search, not "
            "semantic, so search with the words that matter (e.g. \"key\")."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_game_command",
        "description": "Send a command to the game. Always terminal, always last, exactly once per step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "reason": {"type": ["string", "null"], "description": "Why this command was chosen"},
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are playing Knight Orc, a text adventure game. Each turn you are shown
the raw response to the last command you sent (or the game's boot text, on
turn one), along with what you're currently holding (tracked for you from
your own past parse_game_response calls — trust it, but you can always run
the real INVENTORY command if you want to double-check it against the
game itself).

Every turn follows the same shape:

1. Call parse_game_response first, always — capture what actually happened
   in that response before doing anything else. You can't reason about
   what to do next from a response you haven't read yet.
2. Then, if you need to — call query_map, query_entity_history,
   write_journal, search_journal, and/or request_capability, in any
   order, as many times as genuinely useful. Use them instead of guessing
   from memory:
   - query_map — check a room's exits before assuming you already know
     them
   - query_entity_history — before trying a specific verb on an object
     you're not sure about, check whether you already tried it and what
     happened. If nothing relevant has changed since (see "Avoiding
     wasted repetition" below), don't repeat it — move on.
   - write_journal — noticed something worth remembering later (an
     obstacle and what it needs, something you're not ready to deal with
     yet)? Write it down rather than hoping you'll still remember it in
     50 turns.
   - search_journal — whenever you gain something new (an item, a spell,
     a piece of information), check whether it resolves an obstacle
     you've already noted, before you just add it to your inventory and
     move on — see "Obstacles that need something you don't have yet"
     below. Also useful anytime you want a lead on something (e.g. you're
     holding a key and wondering what it's for).
   - request_capability — if you find yourself wanting to track something
     you can't reliably track right now, say so instead of guessing or
     redoing work to "check" (see "Recognizing the limits of what you can
     track" below)
3. Finally, call execute_game_command with exactly one command, and a
   short reason for choosing it. This ends your turn — you get to see its
   response on the next turn.

Investigate as much as you genuinely need to, but there's a limit on
step 2 ({tool_call_cap} calls) — if you're close to it, wrap up and act.
If you go over, you'll be told to call execute_game_command immediately;
if you don't, the turn ends anyway with a safe fallback command chosen
for you, and you'll be told that happened on your next turn.

{behavior_spec}
"""


def build_system_prompt():
    """Build the system prompt, splicing in docs/agent_behavior_spec.md verbatim
    so the behavior spec stays the single source of truth (docs/main_loop_prompt.md)."""
    behavior_spec = _BEHAVIOR_SPEC_PATH.read_text()
    return _SYSTEM_PROMPT_TEMPLATE.format(tool_call_cap=_NON_TERMINAL_CALL_CAP, behavior_spec=behavior_spec)


def _build_user_message(last_response, inventory, forced_fallback_notice):
    lines = [f'Response to your last command: "{last_response}"']
    held = ", ".join(inventory) if inventory else "nothing"
    lines.append(f"You are currently holding: {held}.")
    if forced_fallback_notice:
        lines.append(forced_fallback_notice)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations — read/write `state` directly, same shape agent.py
# already uses (world_graph, known_entities, inventory, ...), plus a new
# per-run `state["journal"]` list for write_journal/search_journal.
# ---------------------------------------------------------------------------

def _impl_query_map(state, room=None):
    graph = state["world_graph"]
    target_room = room or state["current_room"]
    known_exits, futile_exits = [], []
    if target_room in graph:
        for _, dest, data in graph.edges(target_room, data=True):
            direction = data.get("label", "")
            if data.get("futile"):
                futile_exits.append(direction)
                continue
            destination = None if dest.startswith("Unknown") else dest
            known_exits.append({"direction": direction, "destination": destination})
    return {"room": target_room, "known_exits": known_exits, "futile_exits": futile_exits}


def _impl_query_entity_history(state, object_name):
    entity = state["known_entities"].get(object_name, {})
    return {
        "verb_outcomes": entity.get("verb_outcomes", {}),
        "last_result_summary": entity.get("last_result_summary"),
    }


def _capability_requests_path(run_id):
    directory = _ORCHESTRATOR_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "capability_requests.json"


def _append_finding(run_id, step_num, finding_type, severity, summary):
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


def _impl_request_capability(run_id, step_num, description, rationale=None):
    summary = f"{description} — {rationale}" if rationale else description
    _append_finding(run_id, step_num, "capability_gap", "medium", summary)
    return {"acknowledged": True}


def _impl_write_journal(state, step_num, note):
    state.setdefault("journal", []).append({
        "note": note,
        "room": state["current_room"],
        "step": step_num,
    })
    return {"acknowledged": True}


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "anything",
    "something", "need", "needs", "needed", "what",
}


def _impl_search_journal(state, query):
    words = {w for w in query.lower().split() if len(w) >= 3 and w not in _STOPWORDS}
    scored = []
    for entry in state.get("journal", []):
        note_lower = entry["note"].lower()
        score = sum(1 for w in words if w in note_lower)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], -pair[1]["step"]))
    return {"matches": [
        {"note": e["note"], "room": e["room"], "step": e["step"]} for _, e in scored
    ]}


def _split_verb_object(command):
    """Return (verb, object) if command starts with a known candidate verb, else (None, None)."""
    lowered = command.strip().lower()
    for verb in sorted(config.candidate_verbs, key=len, reverse=True):
        prefix = f"{verb} "
        if lowered.startswith(prefix):
            return verb, command[len(prefix):].strip()
    return None, None


def _classify_action_result(action_result, response_text):
    """succeeded/blocked/invalid — same deterministic classification the legacy
    path uses (agent._is_hard_failure/_is_soft_failure), applied to the model's
    own reported reason plus the raw response rather than re-derived from
    added_to_inventory presence."""
    if action_result.get("succeeded"):
        return "succeeded"
    combined = f"{action_result.get('reason_if_failed') or ''} {response_text}"
    if agent._is_hard_failure(combined):
        return "invalid"
    if agent._is_soft_failure(combined):
        return "blocked"
    # Unrecognized failure phrasing — default to the safer, retryable
    # classification rather than permanently blacklisting an action that
    # might just need different game state (see "Avoiding wasted repetition"
    # in docs/agent_behavior_spec.md: state-dependent, not permanent).
    return "blocked"


def _to_legacy_extracted(args, resolved_room, exits):
    """Reshape parse_game_response's simplified schema into the legacy field
    names existing consumers (run_evaluator.py, log_analyzer.py,
    anomaly_detector.py, ui.py) already read via .get(...) with safe defaults."""
    inventory_changes = args.get("inventory_changes", [])
    added_to_inventory = [
        c["item"] for c in inventory_changes if c.get("change") == "gained" and c.get("item")
    ]
    return {
        "room": resolved_room,
        "exits": exits or [],
        "objects": args.get("objects", []),
        "npcs": args.get("npcs", []),
        "added_to_inventory": added_to_inventory,
        "notable_events": args.get("notable_events", []),
    }


def _apply_parse_game_response(state, args, previous_room, action_taken, response_text):
    """The new-schema equivalent of process_agent_step's steps 6-24
    (agent.py:538-768) — reuses agent.py's resolution/graph helpers directly.
    Does not populate active_goal/unresolved_anomalies: those are legacy
    decision-loop concepts the tool-calling path has no use for (the model
    decides freely each step; notable_events/journal replace the structured
    goal queue — see docs/agent_tools_spec.md)."""
    action_result = args.get("action_result", {})

    verb, obj = _split_verb_object(action_taken) if action_taken else (None, None)
    if verb and obj:
        outcome = _classify_action_result(action_result, response_text)
        agent._record_verb_outcome(state, obj, verb, outcome)
        entity = state["known_entities"].get(obj)
        if entity is not None:
            entity["last_result_summary"] = (
                action_result.get("reason_if_failed") if outcome != "succeeded" else None
            )

    exits = None
    if "exits" in args:
        resp_lower = response_text.lower()
        exits = [e for e in args["exits"] if e not in ("up", "down") or e in resp_lower]

    room_unresolved = False
    room = args.get("room")
    if room and room.strip().lower() not in _NULL_LIKE_ROOM_VALUES:
        state["current_room"] = agent._resolve_room_name(state["world_graph"], room, exits)
        state.setdefault("visited_rooms", set()).add(state["current_room"])
        state["position_lost"] = False
        state["position_lost_attempts"] = 0
    elif action_taken in agent._DIRECTIONS and action_result.get("succeeded"):
        # Movement succeeded but no room was parsed — position unknown until
        # a future parse_game_response resolves it (mirrors agent.py:641-650).
        state["position_lost"] = True
        room_unresolved = True

    if exits is not None and not room_unresolved:
        agent.update_graph(state, state["current_room"], exits, previous_room, action_taken)

    for npc in args.get("npcs", []):
        if npc not in state["known_npcs"]:
            state["known_npcs"][npc] = {"location": state["current_room"], "greeted": False}

    for obj_name in args.get("objects", []):
        if obj_name in state["known_npcs"] or agent._is_creature(obj_name):
            if obj_name not in state["known_npcs"]:
                state["known_npcs"][obj_name] = {"location": state["current_room"], "greeted": False}
            continue
        entity = state["known_entities"].get(obj_name)
        permanently_untakeable = entity is not None and entity.get("verb_outcomes", {}).get("take") == "invalid"
        if not permanently_untakeable:
            if entity is None:
                state["known_entities"][obj_name] = {"status": "discovered", "location": state["current_room"]}
            else:
                entity["location"] = state["current_room"]

    for change in args.get("inventory_changes", []):
        item = (change.get("item") or "").strip()
        if not item:
            continue
        if change.get("change") == "gained":
            if item not in state["inventory"]:
                state["inventory"].append(item)
            entity = state["known_entities"].get(item)
            if entity is None:
                state["known_entities"][item] = {"status": "held", "location": None, "verb_outcomes": {}}
            else:
                entity["status"] = "held"
        elif change.get("change") == "lost":
            state["inventory"] = [i for i in state["inventory"] if i.lower() != item.lower()]

    if action_taken == "score":
        m = agent._SCORE_RE.search(response_text)
        if m:
            state["current_score"] = int(m.group(1))
            state["max_score"] = int(m.group(2))

    if action_taken == "inventory":
        # Reconcile against ground truth whenever the model runs the real
        # command (see "Handling uncertainty" in the behavior spec) —
        # independent of whether inventory_changes was reported for whatever
        # take/theft/gift preceded it. Observed live: a model can correctly
        # verify via INVENTORY and still not report the change that made it
        # true, which would otherwise leave state["inventory"] silently stale
        # forever (it's the ambient context shown back to the model every
        # turn). Reuses the same parser the legacy path's recheck_inventory
        # mechanism uses (agent.py:755-765).
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

    is_death = agent._is_death(response_text)
    return _to_legacy_extracted(args, state["current_room"], exits), is_death


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _tool_result(call_id, payload):
    return {"id": call_id, "content": json.dumps(payload)}


def _finalize_step(state, pending, args, response_text):
    """Apply parse_game_response's effects for `pending` (the action that was
    just executed) and append its game_log entry. Mirrors process_agent_step's
    tail (agent.py:746-789), reusing agent._compute_utility/_detect_loop/
    _mark_edge_futile exactly as the legacy path does."""
    previous_room = pending["previous_room"]
    action_taken = pending["action_taken"]

    # Captured before _apply_parse_game_response mutates known_entities, same
    # as the legacy path's pre_verb_outcomes snapshot (agent.py:526-529) —
    # needed for _compute_utility's "redundant" branch, which was previously
    # unreachable on this path (always passed None/None/{}), so a genuinely
    # repeated verb could never be classified as redundant here even when it
    # legitimately was (observed live: a model repeating "examine putty
    # knife" got "informative" every time instead of "redundant" after the
    # first attempt, undercounting exactly the anomaly type this exists to
    # catch — _utility_streaks(game_log, "redundant") never fires here).
    insp_verb, effective_target = _split_verb_object(action_taken) if action_taken else (None, None)
    pre_verb_outcomes = (
        state["known_entities"].get(effective_target, {}).get("verb_outcomes", {}).copy()
        if effective_target else {}
    )

    extracted, is_death = _apply_parse_game_response(state, args, previous_room, action_taken, response_text)

    utility = agent._compute_utility(
        action_taken, response_text, pending["snap_before"], agent._snapshot_state(state),
        insp_verb, effective_target, pre_verb_outcomes,
    )
    if is_death:
        utility = "death"

    if utility == "futile" and action_taken in agent._DIRECTIONS:
        agent._mark_edge_futile(state, previous_room, action_taken)

    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action_taken,
        "reason": pending.get("action_reason"),
        "response": response_text,
        "extracted": extracted,
        "score": state.get("current_score"),
        "utility": utility,
        "token_usage": pending.get("token_usage", {"input_tokens": 0, "output_tokens": 0}),
    }
    if pending.get("forced_fallback"):
        entry["forced_fallback"] = True
    state["game_log"].append(entry)

    loop_action = agent._detect_loop(state["game_log"])
    if loop_action:
        entry["loop_detected"] = loop_action


def _dispatch_tool(state, run_id, step_num, name, call_id, call_input):
    """Execute one non-terminal (or parse_game_response) tool call. Returns a
    tool_result dict. Counts toward the cap via the caller for every name in
    _NON_TERMINAL_TOOLS."""
    if name == _MANDATORY_FIRST_TOOL:
        return _tool_result(call_id, {"acknowledged": True}), call_input
    if name == "query_map":
        return _tool_result(call_id, _impl_query_map(state, call_input.get("room"))), None
    if name == "query_entity_history":
        return _tool_result(call_id, _impl_query_entity_history(state, call_input.get("object"))), None
    if name == "request_capability":
        return _tool_result(call_id, _impl_request_capability(
            run_id, step_num, call_input.get("description", ""), call_input.get("rationale")
        )), None
    if name == "write_journal":
        return _tool_result(call_id, _impl_write_journal(state, step_num, call_input.get("note", ""))), None
    if name == "search_journal":
        return _tool_result(call_id, _impl_search_journal(state, call_input.get("query", ""))), None
    return _tool_result(call_id, {"error": f"Unknown tool: {name}"}), None


def _run_tool_loop(state, run_id, step_num, tool_adapter, system_prompt, user_message):
    """Runs the tool-calling episode for one step, returning (parsed_args,
    action_taken, action_reason, forced_fallback, total_usage)."""
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    def _track(step_response):
        usage = step_response.get("usage", {})
        total_usage["input_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["output_tokens"] += usage.get("completion_tokens", 0)

    step_response = tool_adapter.start_turn(system_prompt, TOOL_SCHEMAS, user_message)
    _track(step_response)

    parsed_args = None
    non_terminal_calls = 0
    nudged = False

    while True:
        tool_calls = step_response.get("tool_calls", [])

        if not tool_calls:
            # Defensive: both adapters force tool_choice, so this shouldn't
            # happen in practice. Treat it the same as a nudge being ignored —
            # there's nothing to attach a corrective tool_result to.
            _log_tool_loop_exhausted(run_id, step_num, "model returned no tool call")
            return parsed_args, _FALLBACK_COMMAND, "forced fallback: no tool call returned", True, total_usage

        results = []
        action_taken = action_reason = None
        terminal_seen = False

        for call in tool_calls:
            name, call_id, call_input = call["name"], call["id"], call["input"]

            if parsed_args is None and name != _MANDATORY_FIRST_TOOL:
                results.append(_tool_result(call_id, {"error": _ORDERING_CORRECTION}))
                continue

            if name == _TERMINAL_TOOL:
                action_taken = call_input["command"]
                action_reason = call_input.get("reason")
                terminal_seen = True
                results.append(_tool_result(call_id, {"response": "(pending — ends this step)"}))
                continue

            result, maybe_parsed = _dispatch_tool(state, run_id, step_num, name, call_id, call_input)
            results.append(result)
            if maybe_parsed is not None:
                parsed_args = maybe_parsed
            if name in _NON_TERMINAL_TOOLS:
                non_terminal_calls += 1

        if terminal_seen:
            return parsed_args, action_taken, action_reason, False, total_usage

        if parsed_args is not None and non_terminal_calls >= _NON_TERMINAL_CALL_CAP:
            if nudged:
                # Nudge already given and ignored — force the fallback ourselves,
                # no further model call needed (see "Runaway guard" in
                # docs/agent_tools_spec.md).
                _log_tool_loop_exhausted(run_id, step_num, "nudge ignored")
                return parsed_args, _FALLBACK_COMMAND, "forced fallback: tool-call limit exceeded", True, total_usage
            nudged = True
            # Attach the nudge to the last real result's content rather than a
            # fabricated tool_result id — both backends validate that every
            # tool_result corresponds to an actual tool_use block from the
            # prior turn, so a synthetic id would be rejected by the API.
            last = results[-1]
            payload = json.loads(last["content"])
            payload["_nudge"] = _NUDGE_MESSAGE
            last["content"] = json.dumps(payload)

        step_response = tool_adapter.continue_with_results(results)
        _track(step_response)


def _log_tool_loop_exhausted(run_id, step_num, reason):
    _append_finding(
        run_id, step_num, "tool_loop_exhausted", "medium",
        f"Step {step_num}: exceeded the {_NON_TERMINAL_CALL_CAP}-call tool-call limit "
        f"without reaching execute_game_command ({reason}); host forced "
        f"'{_FALLBACK_COMMAND}' as a fallback.",
    )


def run_tool_calling_step(state, child, tool_adapter, run_id, step_num, initial_text=None):
    """Executes one tool-calling episode: finalizes the previous step's response
    (if any), then decides and executes the next command. See module docstring
    for the parse-lags-behind-execute design."""
    pending = state.pop("_pending_step", None)
    if pending is None:
        last_response = initial_text or ""
        previous_room = state["current_room"]
    else:
        last_response = pending["response"]
        previous_room = pending["previous_room"]

    system_prompt = build_system_prompt()
    forced_fallback_notice = (
        "Note: your previous turn exceeded the tool-call limit and was ended "
        f"automatically with '{_FALLBACK_COMMAND}'."
        if pending and pending.get("forced_fallback") else None
    )
    user_message = _build_user_message(last_response, state["inventory"], forced_fallback_notice)

    parsed_args, action_taken, action_reason, forced_fallback, usage = _run_tool_loop(
        state, run_id, step_num, tool_adapter, system_prompt, user_message
    )

    if pending is not None:
        pending["token_usage"] = usage
        _finalize_step(state, pending, parsed_args or {"action_result": {"succeeded": True}}, last_response)
    elif parsed_args is not None:
        # First step: still apply the boot text's parse_game_response effects
        # (room/exits), but there's no prior action to log.
        _apply_parse_game_response(state, parsed_args, previous_room, None, last_response)

    response_text = _run_game_command(child, action_taken)

    state["_pending_step"] = {
        "previous_room": state["current_room"],
        "action_taken": action_taken,
        "action_reason": action_reason,
        "response": response_text,
        "snap_before": agent._snapshot_state(state),
        "forced_fallback": forced_fallback,
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
    }


def finalize_pending_step(state):
    """Call once after the run loop ends to flush the last executed action's
    log entry. parse_game_response always runs at the START of the NEXT step
    (see module docstring) — there is no next step once the run is over, so
    without this call the very last command sent to the game would never
    appear in game_log. Makes no LLM call: logs it with a neutral, unparsed
    extraction rather than spending a call just to close out."""
    pending = state.pop("_pending_step", None)
    if pending is None or pending["action_taken"] is None:
        return
    _finalize_step(state, pending, {"action_result": {"succeeded": True}}, pending["response"])
