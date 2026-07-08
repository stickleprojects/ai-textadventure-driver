# Agent Tools Specification

**Status:** Draft, under discussion — companion to `docs/main_loop_prompt.md`.

## Purpose

This document specifies the tool surface a text-adventure-playing agent
needs to act on `docs/agent_behavior_spec.md` without holding unbounded
state (a growing map, a full history of every verb tried on every object)
in its own context. Like the behavior spec, it's written generically —
nothing here names Knight Orc, `glklevel9`, or any other game-specific
detail. A different game config should be able to reuse this tool surface
unchanged, the same way `configs/*.json` already parameterises verbs,
failure patterns, and nav commands per game.

Where `docs/agent_behavior_spec.md` says *what a good player would do*,
this document says *what the agent needs to be able to ask or do* in order
to do it — the interaction surface, not the strategy.

**A tool here means a free lookup against records the agent (or its host)
already built — no game turn spent.** Anything that queries the game's own
ground truth — `inventory`, `score`, `look`, `examine`, or any other verb
the game itself recognises — is not a tool, it's a normal command sent via
`execute_game_command`, and it costs a turn like any other. `inventory` in
particular doesn't need a dedicated tool: what's currently held is tracked
automatically from `parse_game_response`'s `inventory_changes` field and
can be handed back as small ambient context each turn without a lookup at
all — the actual `inventory` command stays available for whenever the
agent wants to verify that belief against the game's ground truth (see
"Handling uncertainty" in the behavior spec), which is a real command, not
a tool call.

## Step contract

One step is exactly one real command sent to the game. This is a hard
invariant every consumer of the step log (run history, anomaly detectors,
futile-edge/loop-detection) already assumes.

A step's tool calls follow a fixed shape:

1. **`parse_game_response` first, always.** The agent must parse what
   happened in the previous response (or the game's boot text, on the very
   first step) before doing anything else — you can't reason about what to
   do next from a response you haven't read.
2. **Any number of other non-terminal tools, in any order**, up to a cap
   (default 6 calls) — `query_map`, `query_entity_history`,
   `request_capability`, as needed.
3. **`execute_game_command`, exactly once.** Ends the step.

If the cap in step 2 is reached without the agent calling
`execute_game_command`, the host application does not force a fallback
immediately — it sends one nudge (a synthetic tool result, e.g. "You've
reached this step's tool-call limit. Call execute_game_command now to
finish this step.") instead of executing whatever non-terminal tool call
triggered the cap. This gives the agent one more turn to comply:

- **If it calls `execute_game_command` in response** — the step ends
  normally with whatever real command it chose. No finding is logged; the
  nudge did its job.
- **If it calls another non-terminal tool instead (ignoring the nudge)**
  — the host forces the fallback itself: injects a synthetic
  `execute_game_command` call with a safe default command (e.g. `look` —
  always valid, side-effect-free, the same final fallback
  `determine_next_action` already uses today), executes it for real, and
  ends the step without asking the model again. This step's log entry
  should carry a `forced_fallback: true` marker, and a finding is
  recorded in the same shape `request_capability` produces —
  `type: "tool_loop_exhausted"`, `severity: "medium"` — so it's visible
  for review without stopping the run. The agent should be told about
  this on its *next* turn (see `docs/main_loop_prompt.md`), since
  otherwise it has no way to know the `look` response it's shown wasn't
  its own choice.

This bounds worst-case cost cleanly: at most `cap + 2` real model calls
per step (the cap, plus the mandatory `parse_game_response`, plus the one
nudge exchange) — the forced fallback itself, if it comes to that, is
host-injected and needs no further model call.

| Tool | Terminal? | When |
|------|-----------|------|
| `parse_game_response` | No | First, every step |
| `query_map` | No | Anytime after parsing |
| `query_entity_history` | No | Anytime after parsing |
| `request_capability` | No | Anytime after parsing |
| `execute_game_command` | Yes — ends the step | Last, every step |

## Tools

### `parse_game_response`
Non-terminal. Mandatory first call each step. Deliberately simple:
distinguishes the direct outcome of the action just taken from everything
else that happened around it (an NPC entering, something being given or
stolen, ambient chatter) — most of which is just narrative color the agent
should register but doesn't need pre-classified into a typed slot.
```
input: {
  "action_result": {
    "succeeded": "boolean",
    "reason_if_failed": "string | null — short reason, e.g. \"too heavy\", \"you can't go that way\""
  },
  "room": "string | null — only if explicitly named",
  "exits": ["string"],
  "objects": ["string — inanimate items visible"],
  "npcs": ["string — living creatures/characters visible"],
  "inventory_changes": [
    {"item": "string", "change": "gained | lost", "cause": "string — e.g. \"took it\", \"orc stole it\", \"denzyl gave it\""}
  ],
  "notable_events": ["string — anything else worth remembering, in plain language"]
}
output: { "acknowledged": true }
```
`action_result` is what the model directly observed, not a classification
— the host still applies the existing config-driven
`hard_failure_patterns`/`soft_failure_patterns` matching against
`reason_if_failed` and the raw response to classify a verb's outcome as
`succeeded`/`blocked`/`invalid` for `query_entity_history`, unchanged from
today. The model reports what happened; the host still owns deciding
whether that's permanent or state-dependent — that boundary has been
deterministic and config-driven since requirement 18, and stays that way.

`notable_events` deliberately has no further structure and isn't persisted
beyond ambient recent-turn context — what used to be separate
`anomalies`/`resolved_anomalies`/`blocked_by`/`learned_spells` fields fold
in here as plain sentences. The agent reasons about them itself on a later
turn, the same way it reasons about anything else it remembers; if a given
run shows something written here needs to survive longer than ambient
context allows, that's exactly what `request_capability` is for — see
"Recognizing the limits of what you can track" in the behavior spec.

### `execute_game_command`
Send a command to the game. Always terminal, always last.
```
input:  { "command": "string — the raw command to send" }
output: { "response": "string — raw game output" }
```

### `query_map`
Look up what's known about a room without holding the whole map in
context.
```
input:  { "room": "string | null — defaults to current room" }
output: {
  "room": "string",
  "known_exits": [{"direction": "string", "destination": "string | null (null = unexplored)"}],
  "futile_exits": ["string — directions confirmed not to work from here"]
}
```

### `query_entity_history`
A check before trying a specific verb on an object, not a completeness
tracker — most objects will only ever have `examine` (and maybe `take`) in
here, because most verbs are never tried unless something gives a reason
to (see "Object interaction" in the behavior spec: examine and take-if-
useful by default, everything else only on actual need). Call this right
before trying a verb you're not certain about, not proactively for every
object you see.
```
input:  { "object": "string" }
output: {
  "verb_outcomes": { "<verb>": "succeeded | blocked | invalid" },
  "last_result_summary": "string | null — e.g. \"too heavy\" from the most recent attempt"
}
```
Only verbs actually tried appear in `verb_outcomes` — an untried verb is
simply absent, not listed with a placeholder status. `blocked` means
state-dependent (may become worth retrying if something relevant changes
— see "Avoiding wasted repetition" in the behavior spec). `invalid` means
permanent, and persists across runs.

### `request_capability`
Non-terminal. Record that something needed for reliable tracking doesn't
exist yet, instead of guessing or re-deriving the answer from memory.
```
input:  { "description": "string — what capability is missing",
           "rationale": "string — why it would help" }
output: { "acknowledged": true }
```
This is the mechanism behind "Recognizing the limits of what you can
track" in the behavior spec: a structured way to say "I need X" that a
later process can pick up, rather than the agent silently working around
the gap. How a given host application routes these findings (a backlog, an
existing anomaly pipeline, a human inbox) is host-specific and belongs in
that application's own docs, not here — see `docs/main_loop_prompt.md` for
how this project wires it up.

## Adding a tool later

A tool earns a place in this spec by being generic to text-adventure play,
not to Knight Orc specifically. Something Knight-Orc-specific (e.g. a
verb only that game recognises) belongs in the game config
(`configs/*.json`), not here.
