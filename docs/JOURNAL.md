# Development Journal

Running notes on findings, decisions, and things that surprised us during development.

---

## 2026-07-08 (implementation of the tool-calling main loop)

### A verification mechanism that never verified anything (bug 72)

Live-testing the new tool-calling loop turned up something the redesign
didn't cause but exposed: the model ran `inventory` to double-check its
holdings — exactly the "Handling uncertainty" behavior the spec asks for —
and `state["inventory"]` stayed empty anyway. Chased it to
`agent._parse_inventory_response`, which only recognizes `"carrying[:\s]"`
phrasing. Checked every historical run log with an `inventory` action
(`logs/watch_20260701_*.json` through `watch_20260704_*.json`): Knight
Orc's real response is consistently `"You own X[. You are wearing Y]."` —
`"carrying"` never appears once.

That means the *legacy* path's `recheck_inventory` mechanism
(agent.py:755-765) — the thing this exact regex exists to serve — has
silently never resynced against ground truth either, for the whole
project's history. It never surfaced as a visible failure because the
symptom is silence, not an error: state just quietly stays stale instead
of throwing anything. Nothing was ever watching for "did the resync
actually happen," only for "did something crash."

Fixed by adding patterns for the confirmed real phrasing, tried before the
existing broad `"nothing"` fallback scan (a positive match should win over
a keyword scan that could false-trigger on unrelated trailing narration,
e.g. an NPC's shouted line). Full writeup in bug 72.

Lesson: a "verify against the game" instruction is only as good as the
parser reading the verification response — and a parser that fails
*silently* (returns `None`, caller just skips the resync) can hide a total
loss of function for a long time, since the failure mode looks identical
to "nothing needed resyncing" from the outside.

### P010 was P008 wearing a different anomaly-type label

P010 hypothesised that `extract_knowledge` under-reports the dingy stable's exits — only `down`/`out` ever get wired as edges, so cardinal directions and `in` never become candidates and the agent re-cycles the two it knows. Re-reading the source run against the raw extraction disproved this: both times the stable was actually `look`ed at (steps 0 and 14 of `watch_20260707_133153`), `extract_knowledge` returned exactly `['east']` — its one real exit, correctly and completely. The `down`/`out` the agent kept retrying at steps 15–23 were never extracted from the stable at all; they were the phantom edges from the P008 bug (exits reported without a room getting wired onto the stale `current_room` instead of the room just entered), already identified and fixed under a different anomaly label in the 2026-07-07 P008 entry above.

Confirmed rather than just inferred: ran a fresh 40-step watch (`watch_20260708_135628`) against the post-P008 code. Step 0 still extracts `['east']` for the stable (extraction was never the problem), but step 1's `east` move this time resolved a room directly, and the final `world_graph` shows the stable with only `east` edges — no `down`/`out`, no retry streak. Closed P010 as `fixed`, `fixed_in` pointing at PR #82, with no changes to `llm.py` or `configs/knight_orc.json` (the layer P010 targeted was never broken).

Lesson, same shape as the P005/P006 entry: an anomaly's root-cause hypothesis is a starting point, not a diagnosis. Here the hint was in the fix note we'd already written for P008 — "very likely the true explanation for P010 ... worth re-checking after this fix lands" — worth treating a fix's own side-effect notes as a checklist for the rest of the open backlog before writing new code against it.

---

## 2026-07-07

### A "productive" navigation loop can still be a loop (P004)

The architect pipeline flagged a run where the agent shuttled between "dingy stable" and "jousting field" for dozens of steps. It never tripped `_detect_loop` or got an edge marked futile, because both of those safeguards are built around *futility*: `_detect_loop` explicitly excludes direction actions that changed the room (bug 14's fix), and `_compute_utility` only tags an action `futile` on a hard-failure response. Every step of the shuttle changed the room and succeeded, so by every existing metric it was "productive" — just productive at going nowhere new.

The actual cause lived in two places in `determine_next_action`: the immediate current-room Unknown-exit scan just took the first unresolved edge it found, and the diversity-scored `best_target` search (which picks the nearest room with an Unknown exit when the current room has none left) only penalised overused *directions*, not overused *rooms*. Two adjacent rooms that both still had an unexplored exit of their own were nearest-neighbours of each other, so the two selection points kept volleying the agent back and forth.

Fixed both spots with the same idea — penalise recency, don't forbid it. The current-room scan now defers (not skips) the Unknown exit whose direction reverses the move that just got us here, falling back to it only if it's the sole remaining option. The `best_target` score gained a `recent_room_visits` term (a `Counter` over the last 10 log entries' extracted rooms), so a farther but genuinely fresh frontier outscores a nearby room the agent has already been bouncing off of. Neither change touches `futile_edges` or the loop detector — this was a decision-logic problem, not a detection-logic one.

---

### `_DIRECTIONS` missing "in"/"out" broke futile-edge marking for two anomaly types (P005/P006)

The very next watch run got stuck again, this time repeating `out` from "dingy stable" indefinitely. The architect produced two *separate* high-severity plans for it: P005 hypothesised the P004 fresh-exit guard was insufficient for the in/out pair, and P006 hypothesised `"You can't go that way."` was missing from `hard_failure_patterns`. Both hypotheses were wrong, and P006's own risk_notes said so — it explicitly flagged that the default pattern already looked like it should match and asked to confirm against the loaded config before assuming a code change was needed.

Checked: `config.hard_failure_pattern.search("You can't go that way.")` returns `True`, and the run log shows `utility: "futile"` computed correctly on nearly every `out` attempt. So `_compute_utility` was doing its job. The actual break was one line away: `process_agent_step`'s futile-marking guard is `if utility == "futile" and action_taken in _DIRECTIONS`, and `_DIRECTIONS` — unlike `_REVERSE` and `_DIRECTION_NORMALIZE`, which both already treat "in"/"out" as real directions — never included them. So `down` (a cardinal direction, tried once in the same log) got marked futile correctly and was never retried, while `out` kept sailing past the guard and being re-offered forever. One-line fix (add "in"/"out" to `_DIRECTIONS`) resolved both plans at once, since they were the same root cause wearing two different anomaly-type labels.

Lesson: an anomaly report's root-cause hypothesis is a starting point, not a diagnosis — verify against the actual config/state before writing a fix for the layer the architect blamed, especially when the plan's own risk_notes already flag doubt.

### A false-positive anomaly: "redundant" verb retries vs. deliberate attempt-and-learn (P009)

The same run also produced P009 ("verb sequence runs redundant inspection verbs after first non-informative response"), proposing to abort the inspection sequence early once `_compute_utility` classifies a step as `redundant`. This is not a bug — it's requirement 18's design working as intended. `docs/requirements/18.md`/`27.md` established attempt-and-learn deliberately: every verb in `candidate_verbs` is tried once per object so the *game* classifies it (`succeeded`/`blocked`/`invalid`), and only `invalid` is ever excluded, permanently, cross-run. `redundant` is a diff-based utility label for anomaly *detection*, not a verb-exclusion signal — it was never wired to gate future attempts, and a textually-similar response from one verb doesn't prove a different verb (e.g. `push`) wouldn't change game state in a way the diff check can't see from text alone (see the 2026-06-30 entry below).

Deferred rather than fixed, with the reasoning written into `plans/P009.json` so the next architect run recognises it as already-considered instead of re-flagging it. It's effectively a duplicate of P001 (deferred earlier for the same underlying reason, under a different anomaly_type label) — the two should be re-evaluated together, not separately, if a future run shows this costing real progress rather than just step count.

---

### An llm_review anomaly can misdiagnose its own evidence, but still stumble onto a real bug (P007)

P007 claimed the agent lost track of its location for 13 consecutive steps (2–14) after a combat-forced move, never issuing a `look` to recover. Replaying the log against the actual state-transition rules showed this was wrong: steps 4–14 are `take`/`examine`/`read`/`wear`/`push` responses on two objects, which legitimately have no `room` field — that's normal for non-movement verbs (requirement 18's fixed inspection sequence), not evidence of a stale `current_room`. `position_lost` actually fired once at step 2 and was correctly resolved by the forced `look` at step 3 — one step of blindness, not thirteen.

But the hypothesis the reviewer wrote to explain its (mistaken) evidence turned out to describe a real, separate defect: `determine_next_action` unconditionally cleared `position_lost` after issuing exactly one `look`, and the re-arm branch in `process_agent_step` only triggers on directional moves — so if that recovery `look` itself failed to yield a room (which didn't happen in this log, but easily could in a different one), the flag would clear anyway and the agent would carry on blind indefinitely with no further recovery attempt. Fixed by only clearing `position_lost` once a room is actually resolved, with a capped retry count (`_POSITION_LOST_MAX_ATTEMPTS = 3`) so a game state that genuinely never confirms a room can't loop forever.

Lesson, paired with the P005/P006 entry above: treat an anomaly's cited evidence and its root-cause hypothesis as separable claims. The evidence here was wrong; the hypothesis, on independent inspection of the code, wasn't — worth fixing anyway, just not as "13 steps blind in this run."

---

### Exits reported without a room got wired onto the wrong room (P008), and it explains where P005/P006's phantom edges came from

P008 proposed adding a "re-look after a combat-forced relocation" trigger, on the theory that `determine_next_action` never re-examines a room's exits after an unexpected relocation. That trigger already exists — it's `position_lost`, and the log shows it worked: the "east" move from "dingy stable" lost the room (combat text ate it), and the very next step was a forced `look` that correctly reported "huge pile of garbage" and its exits.

The real bug was in the step *before* that recovery, and much narrower. When the "east" move's response included `exits: ["down", "outside"]` but no `room` (the extraction the LLM gave for the combat-interlude text), `process_agent_step` still called `update_graph(state, state["current_room"], exits, ...)` — but `state["current_room"]` hadn't changed yet (nothing updates it when `extracted["room"]` is falsy), so it was still `"dingy stable"`. Those exits actually belonged to "huge pile of garbage", the room just entered, not "dingy stable" — which, per its own `look` responses (both before and after this incident), has exactly one real exit, `east`. The mis-attribution wired phantom `Unknown (down from dingy stable)` / `Unknown (out from dingy stable)` edges onto a room that doesn't have them.

This is the same phantom-edge mechanism that fed P005/P006 — "dingy stable" never really had a `down` or `out` exit to begin with; it was corrupted graph state from this exact bug, one step earlier in the same incident. It's also very likely the true explanation for P010 ("stable exits never listed"), which blamed the LLM for under-reporting cardinal directions for the stable — worth re-checking after this fix lands, since there may be no missing-exit-extraction problem at all once phantom edges stop being created.

Fixed by skipping `update_graph` for the one step where a directional move sets `position_lost` — those exits describe the not-yet-confirmed destination room, not the stale `current_room`, and the forced follow-up `look` already re-reports the same exits correctly attributed once the room resolves, so nothing is lost.

---

## 2026-07-01

### Prompt rule blocking "outside" caused room name paraphrasing and duplicates

When investigating a duplicate map node ("cave in juniper scrubland" vs "cave in a juniper scrubland"), we discovered the game text is perfectly consistent: every visit returns `"you go north and are outside a cave in a juniper scrubland"`. The LLM was stripping `"outside a"` on every visit, then inconsistently dropping the inner article `"a"`, producing two different node IDs for the same room.

The root cause was a prompt rule that listed `"outside"` as a forbidden bare descriptor (alongside `"dark"`). The intent was to prevent responses like `"It is dark"` from being extracted as `room: "dark"`. The LLM correctly applied this rule but too broadly — it stripped `"outside"` even when it was part of a compound description like `"outside a cave in a juniper scrubland"`.

The fix was two-part: (1) revise the prompt to distinguish bare descriptors (`"dark"`) from compound descriptions (`"outside a cave"` or `"top of the hill"`), and add an explicit verbatim-copy instruction with an example; (2) add `_resolve_room_name` as a safety net that collapses article variations into the first-seen node name, protecting the graph against future LLM paraphrasing of this kind.

The lesson: prompt rules that name specific words to exclude (rather than naming a structural pattern) are fragile — the LLM applies them by surface match rather than by intent.

---

## 2026-06-30

### Attempt-and-learn beats pre-classification for verb-object compatibility (issue 18)

**Origin:** while adding `wear` to `inspection_sequence`, Copilot autocompleted a long list of other verbs — `open`, `close`, `lock`, `unlock`, `eat`, etc. That made it obvious that each new verb would need its own "attempt, evaluate response, record outcome" cycle, and that the right solution wasn't to hard-code which verbs apply to which objects but to let the game teach the agent.

Initial plan was to classify every object into a type (container, clothing, food, etc.) using the LLM at extraction time, then look up allowed verbs from a config table. Rejected for two reasons:

1. **The game is a better oracle than the LLM.** Knight Orc will always tell you when a verb doesn't apply — and crucially, *how* it fails tells you something the LLM can't: whether the failure is permanent ("you can't wear an apple") or state-dependent ("you can't wear that right now, you're already wearing armour"). A static type table can't capture that distinction.

2. **Wearable state is compositional.** You can wear a hat and gloves at the same time but not a cloak over armour. This depends on what the agent is currently wearing — not on the object's type. No pre-classification scheme handles this; only the game's response does.

**Decision:** replace the planned type-classification approach with attempt-and-learn. Try a broad candidate verb list on every object; record outcomes (`succeeded`, `blocked`, `invalid`) in `known_entities`; skip `invalid` verbs permanently and defer `blocked` ones until state changes. Splits `failure_pattern` in config into `hard_failure_patterns` and `soft_failure_patterns`.

**Why this is lower risk:** pattern matching on game responses is already proven (`_is_failure_response`). No dependency on LLM classification reliability.

---

### Room name hallucination from terse LLM responses (issue observed in watch_run)

**Origin:** during a 10-step `watch_run`, the location field visibly reset to `"current location"` on step 3 and then switched to `"putty knife room"` on step 4 — a name that had never appeared in the actual game output. We suspected LLM hallucination but didn't know the exact trigger.

Root cause turned out to be the prompt requiring a `room` field on *every* extraction. When the game sends a terse confirmation (`"Taken."`) with no location text, the LLM has nothing to fill in and either uses a placeholder (`"current location"`) or derives a name from the action being performed (`"putty knife room"` from `take putty knife`). It was faithfully following the schema even when there was no valid value.

Fix: explicit prompt instruction to omit `room` entirely when the response is terse and contains no location text. Detection added to `log_analyzer.py` (`hallucinated_room_name` issue type) so any future regressions are caught automatically rather than requiring someone to watch a run and notice the value looks wrong.

---

### Issue 10 design: "wait" strategy should be emergent, not hardcoded

**Origin:** designing cross-run learning for issue 10. Initial instinct was to hardcode a preamble list (`["wait", "wait", "wait", "wait"]`) because we know the first few turns of Knight Orc always end with the agent dumped on the rubbish pile regardless of what it does. The user rejected this.

The core objection: if we hardcode "wait 4 times at the start", we're encoding domain knowledge that the agent should be able to *discover* itself. There may be other locations in the game with the same structure (movement is blocked, some event will fire if you wait) — and we'd have to hardcode each one manually rather than the agent generalising the pattern.

**Agreed design:** action utility tagging. Every game log entry gets a `utility` value — `productive`, `informative`, `redundant`, or `futile` — computed from state diff after each step (no LLM needed). A new `room_policies` dict in agent state tracks which movement directions have been tried from each room. When N consecutive direction attempts from a room all come back `futile` (hard failure, no room change), those directions are marked futile and `preferred_when_stuck` is set to `"wait"`. The agent then issues `wait` naturally when it has nothing else to do in that room.

**Why this is better:** the agent discovers the wait strategy in one or two runs and records it in `configs/knight_orc_strategy.json`. On subsequent runs it skips the futile movement attempts entirely. The same logic applies to any room where movement is blocked — no manual config per location. The human-readable strategy JSON also makes it easy to inspect what the agent has learned and correct it if wrong.

**Key invariant:** a direction attempt is only `futile` if *both* the room didn't change *and* a hard failure pattern matched. A direction that yields a new description without moving is `informative`, not futile.

---

### Agentic dev loop architecture

**Origin:** hand-iterating Knight Orc runs was getting tedious — restart the process, watch it play, spot what went wrong, tweak the config or prompt, repeat. There's also a longer-term goal: once the loop is solid on Knight Orc, swap in a different game's config and get a self-improving agent without touching the core code. Doing that by hand every time wouldn't scale.

Built `scripts/agent_dev_loop.py`: run game headlessly → analyze log for issue patterns → feed analysis to a Claude fixer agent (Anthropic API, tool use: bash/read_file/edit_file) → run pytest → repeat. The fixer agent is directed to `configs/knight_orc.json` first for creature/failure-phrase issues, then `llm.py` for prompt issues, then `agent.py` for logic issues. This ordering matters: most issues can be fixed in config without touching code, keeping the core game-agnostic.

`log_analyzer.py` is the detection layer — pure Python, no LLM, unit-tested — so the agentic fixer gets a structured list of issue types rather than raw logs to interpret. Pattern for adding detectors: write the check, write a test that fires on a synthetic bad log entry and a test that passes on a clean one.

---

## 2026-07-02

### `productive_thrash` is a blind spot for every prior detector

When reviewing `watch_20260702_124917`, the agent spent 116 of 150 steps bouncing between stable → jousting field → fairground → stable, yet no anomaly was detected. Every step was tagged `productive` (the room did change each step), so utility streaks didn't fire. The action text varied each step (directions changed because the agent tried different nav strategies each time), so `loop_detected` didn't fire either.

The core issue is that "productive" only means the room changed — it says nothing about whether the agent is making forward progress in the map. A pure oscillation between N rooms is indistinguishable from exploration until you look at the *density* of unique rooms visited over a window of time.

Added `_productive_thrash()` to `anomaly_detector.py`: sliding 20-step window; if ≤ 4 distinct rooms appear in a window (and the total run has visited more than that, ruling out genuinely small maps), all steps in the window are flagged. Contiguous flagged regions of ≥ 20 steps are reported as one anomaly. This caught the real run's thrash from step 35 onwards. Room names are normalised inline (split at first comma/semicolon, strip leading preposition/article) so the detector works correctly on both old fragmented logs and new canonicalised ones.

### `drain_game_buffer` infinite loop in test suite

`drain_game_buffer()` runs `while True: child.expect(...)` and exits only when `pexpect.TIMEOUT` is raised. A `MagicMock` child's `expect()` never raises `TIMEOUT` by default, so the test suite hung — on a developer's machine this triggered the OS "process not responding" popup. Fixed with an `autouse` fixture in `test_game_engine.py` that patches `drain_game_buffer` to a no-op. Pattern for future net I/O helpers: any function with a `while True` loop must either (a) have a dedicated test with the TIMEOUT path exercised, or (b) be patched out at the test module level.

### Fast-nav to undiscovered rooms caused engine-level failures

The `run to <room>` template was being emitted for rooms that hadn't been physically entered yet. Knight Orc rejects this immediately with an error message ("You don't know where that is" or similar). The first draft fix plan from the architect proposed retrying a few times before giving up and marking the room as `blocked`; a human reviewer caught that this was wrong on two counts: (1) the game's rejection is immediate and deterministic, not a transient state issue, so retrying accomplishes nothing; (2) marking the room `blocked` in the strategy store would be harmful, because the constraint is structural (agent hasn't been there yet) not permanent. The actual fix was to gate the fast-nav template on `target in state["visited_rooms"]` — a one-line structural filter. Lesson: the architect's first draft should be reviewed by a human before implementation starts; prompt it to distinguish structural constraints from state-dependent ones.

### Architect misdiagnoses root cause when code already implements the proposed guard (a4)

For anomaly a4 (agent probing "up" and "down" in flat rooms), the architect produced: "Unknown edges are generated for directions the LLM never listed." The fix plan said to only create Unknown placeholder edges for directions in the extracted exits list. That guard already exists — `update_graph` has always iterated over `exits` and only created Unknown nodes for listed directions.

The actual root cause was one level up: the LLM itself infers "up" and "down" from the phrase "Exits lead in all directions", inserting them into the exits list even when they don't exist in the game. The fix belonged in `process_agent_step`, not `update_graph`: filter "up"/"down" from extracted exits unless those words appear literally in the response text.

This pattern — architect diagnosing a symptom in code that already has the guard — suggests the architect prompt should explicitly ask "does the code already enforce this?" and "at what layer does the bad data originate?" before proposing a fix location.

### Architect proposes harmful fix for productive diagonal exploration (a3)

For anomaly a3 (sw/east zigzag), the architect proposed: detect A/B alternation, mark the oscillating edges futile. This would have broken the agent: steps 56–75 of the same run show sw/east alternating with `util=productive` on every step (each leg visits a new room). Marking those edges futile would have cut off valid corridors.

The real cause is structural: the forest world has a diagonal grid topology. Each sw room has an east Unknown exit; each east room has a sw Unknown exit. The agent follows these chains greedily because it always picks the nearest room with an Unknown exit, and "nearest" always resolves to the next diagonal step.

The correct fix: add a direction-diversity penalty to navigation target scoring. When picking which room to navigate to for Unknown exit exploration, score by `path_len + recent_dir_frequency(unknown_exit_direction)`. This steers the agent toward rooms with underused exit directions without ever marking productive corridors as futile.

Lesson: when an anomaly type is "repetitive pattern", the architect must first check whether each repetition is productive (new rooms, new entities, new inventory). If steps are productive, the edges are never candidates for futile marking — the fix must instead change the *selection policy*, not the edge state.

---

## 2026-07-03

### Cross-run object memory silently blocked re-picking-up known items (bug 64)

A live run showed the agent walking straight past objects in a room on step 2 without ever attempting to take them. Neither the deterministic anomaly detectors nor the LLM review flagged it — there was no failure, no repetition, just an absent action, which log-diffing detectors have no way to notice without domain knowledge of what *should* have happened.

The cause: `known_entities` is pre-seeded at session start from the cross-run strategy sidecar, so it accumulates every object name ever seen across *every* prior run. The object-queueing gate in `process_agent_step` was `obj not in state["known_entities"]` — a check that conflates "have I ever seen this object" (permanent, cross-run) with "do I currently possess it" (per-run; `state["inventory"]` and `state["uninspected_objects"]` both reset every run). Once an object had been seen in any run, it could never be queued for `take` again.

The fix required separating two kinds of memory that had been collapsed into one field:
- **Cross-run, correctly persistent**: verb outcomes on non-`take` inspection verbs (`smell`, `read`, `examine`, ...) — if we've already learned an object has no smell, don't smell it again. This was already implemented correctly via `verb_outcomes` filtering in the post-take inspection sequence and did not need to change.
- **Cross-run, but only for one specific fact**: `take` outcome `"invalid"` (permanently un-takeable, e.g. scenery) is a real fact about the object and should keep blocking re-queueing forever.
- **Per-run, must reset**: whether the object is currently held or already pending pickup this run. This is what should gate re-queueing for `take` — not blanket `known_entities` membership.

A second, related bug surfaced during the fix: the `take` outcome classifier used the broad `_is_failure_response` (hard OR soft) and recorded *any* failure as `"invalid"`, unlike every other verb which already split hard (permanent) from soft (state-dependent, "blocked", never persisted). A state-dependent take failure — e.g. "you're already carrying that", or a future carry-capacity limit — would have been wrongly persisted as a permanent fact and blacklisted the object cross-run. Fixed by giving `take` the same hard/soft split already used elsewhere.

Lesson: when a single dict (`known_entities`) is used both as "things I remember learning" (cross-run) and "things relevant to my current state" (per-run), any check against bare membership in that dict is a latent bug. The fix is never "clear it every run" (that would also lose the legitimately-permanent verb-outcome memory) — it's identifying exactly which sub-fact is permanent and which is run-scoped, then gating on the right one.

---

## 2026-07-04

### Feature 59's own cross-reference list understated how much of the spec was already implemented

Before writing spec-driven fixtures, we checked how much of `docs/agent_behavior_spec.md` already had code behind it. Feature 59's doc lists open requirements (2, 21, 23, 25, 40) against several spec sections, implying most of the doc describes unimplemented behavior. In fact the generic `unresolved_anomalies` → `active_goal` mechanism in `agent.py` (`determine_next_action`) already handles three separate spec sections at once — locked exits, locked containers, and spells — because all three reduce to the same shape (a target with a `potential_solution` string matched against inventory/spellbook; `cast` vs `use` is picked by spellbook membership). Death/pearl-room handling (`recheck_inventory`, `position_lost`) is also real, not aspirational.

The genuinely unimplemented sections are narrower than the doc's framing suggests: NPC interaction verbs (SAY/GIVE/TAKE FROM/FOLLOW — Requirement 2/21/40), NPC theft detection (Requirement 23), and landmark/`GO TO`-partial-block handling (Requirement 25). Decided to scope feature 59's first phase around exactly this gap — spec-traceable fixtures for the sections that already work (regression coverage) plus `xfail`'d fixtures for the three that don't (the actual punch list) — rather than starting with the code/spec split feature 59 describes first. The split is real future work, but doing it before there's any spec-traceable test coverage would mean refactoring blind.

### Agent-level spec fixtures didn't need any new test infrastructure

The natural worry going in was that testing *agent behavior* (as opposed to LLM extraction, which `tests/evals/` already covers) would require pulling a pure `apply_extraction(state, ...)` function out of the monolithic `process_agent_step` (which currently also calls `execute_game_command` and `extract_knowledge` directly). It didn't: `tests/test_agent.py` already tests `process_agent_step` end-to-end by patching `agent.execute_game_command` and `agent.extract_knowledge` with `unittest.mock.patch`, then asserting on the mutated `state`. Reusing that exact pattern, data-driven instead of hand-written per case, was enough — no `agent.py` refactor needed to get spec-scenario fixtures working (`tests/spec_scenarios/`).

### Fixer agent invented a literal "from you" that the spec never said (PR #62)

The first `--spec-target` run against `npc_theft_removes_from_inventory` (Requirement 23) resolved the scenario's mocked test on iteration 1, but the fix it wrote — `_NPC_THEFT_RE` matching `"stole/takes <item> from you"` — hardcoded a word the spec doc never actually uses. `docs/agent_behavior_spec.md`'s "NPCs stealing items" section gives two concrete example narrations, neither containing "you": `"<npc> just stole <object> from a stinking orc"` and `"<npc> takes the <object>"`. The prose above them talks about "your item" and "your inventory", but the fixer (and, on first pass, a human reviewing it) collapsed that into literally matching the pronoun "you" in game text — which Knight Orc apparently doesn't always use, since it can narrate the player's own character in third person (by creature description) rather than address them directly.

The fix: stop trying to identify the "victim" from narration text at all. Match the item being stolen (any theft phrasing, any or no named victim) and remove it from inventory *only if currently held* — using the agent's own state as ground truth instead of parsing prose for identity. This is also more robust in general: it correctly ignores theft between two other NPCs (item never in our inventory, so no-op) without needing to know who's who. Moved the patterns into `configs/knight_orc.json` (`theft_patterns`, schema-validated) rather than leaving them hardcoded in `agent.py`, so new phrasings are a config edit, not a code change.

Lesson for future fixer-agent runs (and human review of them): when a spec section's prose uses a placeholder pronoun/example that doesn't appear in its own concrete quoted examples, don't invent literal matching text for it — check it against the quoted examples specifically, and prefer verifying against actual state (inventory, in this case) over parsing narration for who a pronoun refers to.

## 2026-07-06

### "That's too heavy." was misclassified as a soft (retryable) failure, causing endless re-inspection of heavy scenery

User reported the agent looking "stupid" — repeatedly trying to take/wear/push/examine the same heap-of-garbage-type objects. Root cause: `configs/knight_orc.json` had `"too heavy"` in `soft_failure_patterns`, meaning a failed `take` on a heavy object was recorded as `"blocked"` (state-dependent, may succeed later) rather than `"invalid"` (permanent). Since `determine_next_action`'s re-queueing guard only checks for `verb_outcomes["take"] == "invalid"`, a `"blocked"` outcome never stops the object from being re-added to `uninspected_objects` — and it gets re-added every time the room is revisited and the LLM re-reports the object as visible, replaying the *entire* inspection sequence (take, examine, read, look inside, wear, push...) from scratch each time.

Checked every response containing "too heavy" across all of `logs/*.json`: zero instances of a `take`/`push`/`wear` ever succeeding on the same object after a "too heavy" response, for any object, in any run. The message is a fixed trait of the object ("this is scenery-heavy"), not a transient player-encumbrance state — Level 9's actual carry-limit message is different text entirely. Moved `"too heavy"` from `soft_failure_patterns` to `hard_failure_patterns` so it's recorded as `"invalid"` and permanently blocks re-queueing, same as other scenery objects.

Lesson: when adding a failure pattern, check empirically (grep the log corpus) whether it ever actually resolves into success on a retry, rather than guessing from the wording alone — "too heavy" *sounds* like a temporary carrying-capacity complaint, but in this game's text it never is.

## 2026-07-07

### Bug 45 fix: disjoint exit sets, not literal fingerprints, distinguish same-named maze rooms

The bug's original write-up proposed a literal `room_name + sorted(exits)` node ID. Building it, that turned out to be the wrong granularity: exits aren't always fully known on every visit (a terse response, or the LLM under-reporting one exit) — a literal fingerprint would treat any partial exit list as a different room from a fuller one seen earlier, fragmenting a single real room into several nodes.

What actually distinguishes "same room revisited with slightly different exit info" from "different room reusing the same display name" is whether the two exit sets share *anything in common*. Reused: overlapping or subset exits (or no exits recorded yet for the existing node) merge into the existing node. Disjoint (non-empty exit sets, zero overlap): treated as a genuinely different physical room and given a disambiguating suffix (`"Alder Clump #2"`). This is a much better fit for how mazes are actually authored — each room in a maze typically has its own distinct exit configuration even when it shares a generic name with others — and tolerates the LLM's normal exit-reporting noise without fragmenting legitimate revisits.

Also deviated from the write-up's suggestion to strip the disambiguation suffix from the map display label. Left it visible instead: two different physical rooms showing identical text on the map, distinguished only by graph position, seemed like a worse outcome for a human trying to read the map than just showing `"Alder Clump #2"`.

### Verifying a graph-identity fix needs real extraction, not a mocked dict

The existing single-step test patterns (`tests/spec_scenarios`, most of `tests/test_agent.py`) mock `extract_knowledge` to return a hand-written dict — fine for testing decision logic, but it can't catch whether the fix actually holds up against how a real LLM phrases and sequences room/exit data across several consecutive turns. Added `tests/test_simulations.py`: a scripted 5-response maze walk fed through `process_agent_step` in a loop with a real DeepSeek call each step (`@pytest.mark.llm`, mirroring the existing gating for `tests/evals`), only mocking the game engine's response queue. Kept the fixture format plain Python rather than JSON+schema (unlike `tests/evals`/`tests/spec_scenarios`) since this is a fixed per-bug regression scenario, not a growing library that needs schema governance.

## 2026-07-09

### One casing bug turned out to be four independent bugs wearing the same disguise

Bug 73 was filed after a single live run showed two symptoms of model-casing inconsistency (`LOOK`/`look`, `EXAMINE putty knife`/`examine putty knife`): loop-detection undercounting, and a theoretical `known_entities` fragmentation risk. Picking it up for implementation and grepping every call site of `DIRECTIONS`/`REVERSE` (the same case-sensitive-set-membership mistake, applied literally) turned up three more instances that had nothing to do with each other structurally: `world_graph.update_graph`'s `REVERSE.get(action, "")` silently no-ops on capitalized directions (skips reverse-edge wiring and placeholder cleanup); `agent._detect_loop`'s own "exclude productive moves" exemption check was *backwards* under the same bug — a genuinely productive capitalized move got counted toward the stuck-threshold instead of being exempted; and the anti-oscillation/direction-diversity scoring in `determine_next_action` silently undercounted capitalized moves too.

None of these four shared code — each was an independent reinvention (or copy-paste) of the identical case-sensitive membership check against `DIRECTIONS`/`REVERSE`. Lesson: once one case-sensitivity bug is found in a comparison against a fixed-casing set or dict, grep every other call site of that same set/dict before considering the bug closed — the identical mistake tends to recur independently across a codebase's history rather than existing in exactly one place.

### A "should already be covered" trace turned out to have a real gap

While scoping the `known_entities`/`known_npcs` case-insensitive-keying fix, an initial pass concluded `determine_next_action`'s `uninspected_objects.pop(0)` → `known_entities.get(new_target)` lookup didn't need a separate fix, reasoning that a second differently-cased mention of an already-tracked object never gets re-queued once insertion sites are fixed. That reasoning holds *within* a single run, but missed the cross-run case: `known_entities` can be pre-seeded from the strategy sidecar (feature 64) under one casing (`"Putty Knife"`), while this run's own extraction produces a different casing (`"putty knife"`) that resolves fine on *insertion* (finds the existing key, updates it) but still queues the *raw* casing into `uninspected_objects` — so the later pop-and-lookup misses the pre-seeded key and silently loses that object's entire cross-run `verb_outcomes` history. Only re-checking against the cross-run-seeding journal entry from 2026-07-03 (bug 64) surfaced this. Lesson: "already covered by the general fix" needs to be checked against every place state can originate, not just the code path being edited — a naive single-run trace missed a state source (sidecar pre-seeding) introduced by an entirely different feature.

### Shared helpers prevented a fifth recurrence

The fix intentionally centralized normalization at two boundaries instead of adding more inline `.lower()` calls at each bug site: `world_graph.normalize_direction` for direction comparisons/storage, and `parse_strategies.resolve_entity_key` for case-insensitive dict key lookups that preserve first-seen casing. That made every known `DIRECTIONS`/`REVERSE` and `known_entities`/`known_npcs` call site consistently correct, and removed the need for each module to remember its own ad-hoc normalization rules.

After wiring the helpers through both runtime paths (`agent.py` and `agent_tools.py`) and parser application, the most important regression tests were the mixed-case loop-thrashing case (case-folded repeat count now trips loop detection), uppercase-direction productive-move exemption, uppercase graph edge wiring/placeholder cleanup, and cross-case entity-history lookup (including pre-seeded cross-run state). The main lesson is that this class of bug is easiest to prevent at API boundaries, not with local patches.
