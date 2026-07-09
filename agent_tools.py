"""Tool-calling main loop — cloud/tool-use-capable LLM backends only.

Implements docs/agent_tools_spec.md and docs/main_loop_prompt.md: one LLM
tool-calling episode per step to decide the next command, replacing
agent.py::determine_next_action's fixed priority list. Parsing the game's
response is a separate concern — see parse_strategies.py — handled by a
swappable ParseStrategy immediately after execute_game_command, not bundled
into this decide episode. That split is what lets both this module and
agent.py::process_agent_step share one apply_parse_result implementation,
and removes the one-step lag an earlier version of this loop had (deciding
and parsing used to be the same tool-calling episode, so parsing the
response to step N's command could only happen on step N+1's episode).

The local llama.cpp path (agent.py::process_agent_step) is untouched and
unaffected by this module — see the "Scope" section of
docs/main_loop_prompt.md for why.
"""
import json
from datetime import datetime
from pathlib import Path

import agent
import parse_strategies
from game_engine import execute_game_command as _run_game_command

_NON_TERMINAL_CALL_CAP = 6
_FALLBACK_COMMAND = "look"
_BEHAVIOR_SPEC_PATH = Path(__file__).parent / "docs" / "agent_behavior_spec.md"

_NUDGE_MESSAGE = (
    f"You've reached this step's tool-call limit ({_NON_TERMINAL_CALL_CAP}). "
    "Call execute_game_command now to finish this step."
)

_TERMINAL_TOOL = "execute_game_command"
_NON_TERMINAL_TOOLS = {"query_map", "query_entity_history", "request_capability",
                        "write_journal", "search_journal"}

# ---------------------------------------------------------------------------
# Tool schemas — canonical, backend-neutral (Anthropic's native input_schema
# shape; llm.py's OpenAIToolAdapter translates this for OpenAI-style function
# calling). See docs/agent_tools_spec.md for the authoritative contract.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
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
turn one), along with what you're currently holding (tracked for you
automatically from your own past commands — trust it, but you can always
run the real INVENTORY command if you want to double-check it against the
game itself).

Every turn follows the same shape:

1. If you need to — call query_map, query_entity_history, write_journal,
   search_journal, and/or request_capability, in any order, as many times
   as genuinely useful. Use them instead of guessing from memory:
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
2. Finally, call execute_game_command with exactly one command, and a
   short reason for choosing it. This ends your turn — you get to see its
   response on the next turn.

Investigate as much as you genuinely need to, but there's a limit on
step 1 ({tool_call_cap} calls) — if you're close to it, wrap up and act.
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


def _impl_request_capability(run_id, step_num, description, rationale=None):
    summary = f"{description} — {rationale}" if rationale else description
    parse_strategies.append_finding(run_id, step_num, "capability_gap", "medium", summary)
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


def _tool_result(call_id, payload):
    return {"id": call_id, "content": json.dumps(payload)}


def _dispatch_tool(state, run_id, step_num, name, call_id, call_input):
    """Execute one non-terminal tool call. Returns a tool_result dict. Counts
    toward the cap via the caller for every name in _NON_TERMINAL_TOOLS."""
    if name == "query_map":
        return _tool_result(call_id, _impl_query_map(state, call_input.get("room")))
    if name == "query_entity_history":
        return _tool_result(call_id, _impl_query_entity_history(state, call_input.get("object")))
    if name == "request_capability":
        return _tool_result(call_id, _impl_request_capability(
            run_id, step_num, call_input.get("description", ""), call_input.get("rationale")
        ))
    if name == "write_journal":
        return _tool_result(call_id, _impl_write_journal(state, step_num, call_input.get("note", "")))
    if name == "search_journal":
        return _tool_result(call_id, _impl_search_journal(state, call_input.get("query", "")))
    return _tool_result(call_id, {"error": f"Unknown tool: {name}"})


def _run_decide_loop(state, run_id, step_num, tool_adapter, system_prompt, user_message):
    """Runs the decide-only tool-calling episode for one step, returning
    (action_taken, action_reason, forced_fallback, total_usage)."""
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    def _track(step_response):
        usage = step_response.get("usage", {})
        total_usage["input_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["output_tokens"] += usage.get("completion_tokens", 0)

    step_response = tool_adapter.start_turn(system_prompt, TOOL_SCHEMAS, user_message)
    _track(step_response)

    non_terminal_calls = 0
    nudged = False

    while True:
        tool_calls = step_response.get("tool_calls", [])

        if not tool_calls:
            # Defensive: both adapters force tool_choice, so this shouldn't
            # happen in practice. Treat it the same as a nudge being ignored —
            # there's nothing to attach a corrective tool_result to.
            _log_tool_loop_exhausted(run_id, step_num, "model returned no tool call")
            return _FALLBACK_COMMAND, "forced fallback: no tool call returned", True, total_usage

        results = []
        action_taken = action_reason = None
        terminal_seen = False

        for call in tool_calls:
            name, call_id, call_input = call["name"], call["id"], call["input"]

            if name == _TERMINAL_TOOL:
                action_taken = call_input["command"]
                action_reason = call_input.get("reason")
                terminal_seen = True
                results.append(_tool_result(call_id, {"response": "(pending — ends this step)"}))
                continue

            results.append(_dispatch_tool(state, run_id, step_num, name, call_id, call_input))
            if name in _NON_TERMINAL_TOOLS:
                non_terminal_calls += 1

        if terminal_seen:
            return action_taken, action_reason, False, total_usage

        if non_terminal_calls >= _NON_TERMINAL_CALL_CAP:
            if nudged:
                # Nudge already given and ignored — force the fallback ourselves,
                # no further model call needed (see "Runaway guard" in
                # docs/agent_tools_spec.md).
                _log_tool_loop_exhausted(run_id, step_num, "nudge ignored")
                return _FALLBACK_COMMAND, "forced fallback: tool-call limit exceeded", True, total_usage
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
    parse_strategies.append_finding(
        run_id, step_num, "tool_loop_exhausted", "medium",
        f"Step {step_num}: exceeded the {_NON_TERMINAL_CALL_CAP}-call tool-call limit "
        f"without reaching execute_game_command ({reason}); host forced "
        f"'{_FALLBACK_COMMAND}' as a fallback.",
    )


def run_tool_calling_step(state, child, tool_adapter, parse_strategy, run_id, step_num, initial_text=None):
    """Executes one step: decide (tool-calling episode) → execute → parse →
    apply → log. One game_log entry per call — no cross-call lag (see module
    docstring). `state["_last_response"]`/`state["_forced_fallback_notice"]`
    carry context from the previous call; absent on the first call, where
    `initial_text` (the game's boot text) is used instead.
    """
    state["_run_id"] = run_id  # lets apply_parse_result log findings (e.g. bug 74) without a signature change
    previous_room = state["current_room"]
    last_response = state.pop("_last_response", None)
    if last_response is None:
        last_response = initial_text or ""
    forced_fallback_notice = state.pop("_forced_fallback_notice", None)

    system_prompt = build_system_prompt()
    user_message = _build_user_message(last_response, state["inventory"], forced_fallback_notice)

    action_taken, action_reason, forced_fallback, usage = _run_decide_loop(
        state, run_id, step_num, tool_adapter, system_prompt, user_message
    )

    snap_before = agent._snapshot_state(state)
    insp_verb, effective_target = parse_strategies.split_verb_object(action_taken)
    pre_verb_outcomes = (
        state["known_entities"].get(effective_target, {}).get("verb_outcomes", {}).copy()
        if effective_target else {}
    )

    response_text = _run_game_command(child, action_taken)

    result = parse_strategy.parse(response_text, action_taken)
    parse_usage = result.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
    token_usage = {
        "input_tokens": usage["input_tokens"] + parse_usage["input_tokens"],
        "output_tokens": usage["output_tokens"] + parse_usage["output_tokens"],
    }
    extracted, is_death, _ = parse_strategies.apply_parse_result(
        state, result, action_taken, response_text, previous_room
    )

    utility = agent._compute_utility(
        action_taken, response_text, snap_before, agent._snapshot_state(state),
        insp_verb, effective_target, pre_verb_outcomes,
    )
    if is_death:
        utility = "death"

    if utility == "futile" and action_taken in agent._DIRECTIONS:
        agent._mark_edge_futile(state, previous_room, action_taken)

    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action_taken,
        "reason": action_reason,
        "response": response_text,
        "extracted": extracted,
        "score": state.get("current_score"),
        "utility": utility,
        "token_usage": token_usage,
    }
    if forced_fallback:
        entry["forced_fallback"] = True
    state["game_log"].append(entry)

    loop_action = agent._detect_loop(state["game_log"])
    if loop_action:
        entry["loop_detected"] = loop_action

    state["_last_response"] = response_text
    if forced_fallback:
        state["_forced_fallback_notice"] = (
            "Note: your previous turn exceeded the tool-call limit and was ended "
            f"automatically with '{_FALLBACK_COMMAND}'."
        )
