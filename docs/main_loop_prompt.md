# Main Loop Prompt (design draft — not yet implemented)

**Status:** Draft, under discussion. Nothing here is wired into `agent.py` /
`llm.py` yet — this is the design pass that precedes it, per
`docs/agent_behavior_spec.md`'s own stated purpose (behavior first,
implementation derived from it second).

## Why this exists

Triggered by a real thrash: `watch_20260708_140045` spent 66 of its 100
steps re-inspecting a single already-fully-inspected object (`flagpole`),
because `determine_next_action`'s priority list has no concept of "I've
already learned everything I can about this, and nothing has changed." See
`docs/JOURNAL.md` for the full trace (TODO once this lands).

This replaces two things `determine_next_action` currently does with a
single LLM call per step:

1. **Decide the next action** — currently a fixed Python priority list
   (`agent.py::determine_next_action`).
2. **Extract structured knowledge** from the response to the action just
   taken — currently a separate call (`llm.py::extract_knowledge`).

Folding both into one reasoning episode means the *decision* can be
informed by the same LLM judgment that reads the response, instead of a
second, separate deterministic pass over already-extracted fields. It also
gives the agent behavior spec's new "Avoiding wasted repetition" and
"Recognizing the limits of what you can track" sections somewhere to
actually apply: instead of Python heuristics guessing whether an object is
"done", the model can check the real record and decide.

**Scope:** this applies only to LLM backends with reliable tool-use
(Anthropic, OpenAI-compatible with function calling, DeepSeek). Local
`llama.cpp` (`.gguf`) models have inconsistent tool-use support, so they
keep using the current `agent.py::determine_next_action` +
`llm.py::extract_knowledge` path unchanged — two code paths, deliberately,
rather than a fallback that has to simulate tool-calling by hand. This
line item may be worth revisiting once local tool-use support is more
consistent, but isn't blocking now.

## Tools

The tool surface (`execute_game_command`, `query_map`,
`query_entity_history`, `request_capability`, `write_journal`,
`search_journal`) is specified in `docs/agent_tools_spec.md`, not here —
those contracts are generic to any text-adventure agent built on this
codebase, not specific to Knight Orc, so they live separately and this
document just uses them.

Two things worth calling out about how *this project* uses that generic
surface:

- **`query_entity_history` is the direct fix for the flagpole thrash**: the
  model can see `take: blocked` from a prior attempt and apply "Avoiding
  wasted repetition" (behavior spec) itself, rather than Python
  re-deciding it from a queue every time the object is re-mentioned. It
  also would have been a smaller thrash regardless — the old fixed
  inspection sequence ran all ~6 candidate verbs on flagpole whether or
  not any of them mattered; "Object interaction" now says examine, take
  if useful, and stop unless there's an actual reason to try more.
- **`request_capability` findings route into the existing anomaly
  pipeline**: appended in the same shape `detect_anomalies.py` /
  `llm_review.py` already produce (`type`, `severity`, `summary`,
  `step_range`), with `type: "capability_gap"`. `scripts/architect.py`
  needs **no new code** to pick these up — they flow into `plans/P*.json`
  through the pipeline that already exists, get reviewed by a human like
  any other plan, and once implemented, the new tool becomes part of next
  run's toolset. This is the "define the map-file spec, we implement it,
  it's available next run" loop from the original idea, reusing
  infrastructure instead of inventing a parallel one.

**Persistence:** `request_capability` (and `tool_loop_exhausted`, from the
runaway guard in `docs/agent_tools_spec.md`'s step contract) write
incrementally to a per-run side file —
`runs/orchestrator/<run_id>/capability_requests.json` — as they happen,
rather than being buffered in memory and flushed only at a clean run end.
Today `anomaly_report.json` is written post-run by `detect_anomalies.py`;
this side file gets merged into it at the next pipeline stage (either by
`detect_anomalies.py` itself, or a small merge step before
`architect.py` runs — implementation detail, not decided here). Writing
incrementally means these findings survive a crash or an interrupted run,
the same lesson bug 71 flagged for `agent_dev_loop.py`: state worth
keeping should be written as it's produced, not held until a clean exit
that isn't guaranteed to happen.

## Draft system prompt

```
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

{behavior_spec_block}
```

`{behavior_spec_block}` is the full prose of `docs/agent_behavior_spec.md`
— it stays the single source of truth for *how to play*; this prompt only
adds the mechanics of *how to act on that judgment through tools* on top.
The three sections called out explicitly above ("Avoiding wasted
repetition", "Recognizing the limits of what you can track", "Obstacles
that need something you don't have yet") are the ones this redesign was
written to operationalize — named directly rather than left for the model
to notice on its own in the middle of a longer prose block.

Tool schemas (`parse_game_response`, `query_map`, `query_entity_history`,
`request_capability`, `write_journal`, `search_journal`,
`execute_game_command`) are registered with the API
call per `docs/agent_tools_spec.md`, not inlined into this prompt text —
the model sees them as actual tool definitions, this prompt just tells it
the order to use them in.

## Resolved design questions

- **Turn/message sequencing:** `parse_game_response` is a mandatory tool
  call, first every step; non-terminal tools are capped at 6 calls. Hitting
  the cap triggers one nudge (not an immediate override) — a compliant
  response ends the step normally with no finding logged; an ignored
  nudge forces a `look` fallback, marks the step `forced_fallback: true`,
  logs a `medium`-severity `tool_loop_exhausted` finding, and tells the
  agent what happened on its next turn. Worst case: `cap + 2` = 8 real
  model calls per step. See the Step contract in
  `docs/agent_tools_spec.md`.
- **Where `request_capability` findings live mid-run:** written
  incrementally to a per-run side file, not buffered in memory — see
  Persistence above.
- **Local model support:** out of scope for this redesign; local
  `llama.cpp` models keep the existing priority-list path — see Scope
  above.
- **Extraction schema shape:** simplified from 11 top-level fields to ~7
  (`action_result`, `room`, `exits`, `objects`, `npcs`,
  `inventory_changes`, `notable_events`) — see `parse_game_response` in
  `docs/agent_tools_spec.md`. `inventory_changes` unifies the old
  `added_to_inventory`/`taken_by_npc`/`received_from_npc` three-way split;
  `notable_events` is unstructured plain text and isn't persisted beyond
  ambient recent-turn context, deliberately accepting that a distant clue
  could fade from context on a long run — mitigated by `write_journal`/
  `search_journal` (below) for anything worth keeping, rather than
  pre-building durable *structured* anomaly-tracking on a guess.
- **Durable per-run memory:** built proactively as `write_journal`/
  `search_journal` (keyword search over host-timestamped notes, per-run
  not cross-run) rather than only ever waiting on a `request_capability`
  round-trip for something this predictable — locked-obstacle-needs-item
  is exactly the pattern the behavior spec's "Locked exits and keys"/
  "Locked containers" sections already describe. `request_capability`
  still exists for anything *this* doesn't cover.
- **Log parity:** `execute_game_command` carries an optional `reason`
  input (why this command was chosen) — found missing during the
  implementation pass. Doesn't affect game behavior; exists purely so
  `game_log` entries on this path stay as readable as today's
  `determine_next_action`-produced entries, which already have a `reason`
  field.
- **Grounding, not prompting, against hallucination:** live-testing turned
  up a real case (bug 74) of the model fabricating a plausible-sounding
  room and narrative for a response that supported neither — `room`
  became `room_quote`, required to be a verbatim substring of the response
  rather than a name the model constructs, so the host can verify it
  deterministically instead of trusting compliance with an instruction.
  The same principle was applied to `exits`/`objects`/`npcs` (bugs 76/77)
  and `inventory_changes` (bug 75, reusing the hard-failure suppression
  already proven for the legacy path's `added_to_inventory`). A rejected
  claim is discarded, not retried — see `docs/agent_tools_spec.md`'s
  `parse_game_response` section for the full reasoning.

## Explicitly not decided yet (implementation-review pass)

- Model choice / cost impact of the now-variable per-step call count —
  deferred until there's a real implementation to measure against
  (`token_usage` is already tracked per run, so this is a "look at the
  numbers after the first real run" question, not a design question to
  resolve on paper).
- Exact merge mechanics for `capability_requests.json` →
  `anomaly_report.json` (which script owns the merge step).
