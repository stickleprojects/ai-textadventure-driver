# Requirements

Agent gameplay capabilities — things the agent must be able to do to play Knight Orc effectively.

## Open

2. the engine needs to know it can speak to npcs, so it should politely greet them and ask them for help or offer to help them, once it finds out what they need they can help it in the future
   - Detection: add `npcs` field to LLM extraction schema alongside `objects`
   - Conversation: fixed single greeting (`greet X`) for now; deeper dialogue later
   - State: new `known_npcs` dict `{name: {location, greeted, needs, can_help_with}}`
   - **Effort: High | Risk: High** — new priority tier in `determine_next_action`, multi-turn dialogue state, LLM must parse free-form NPC responses reliably. Knight Orc NPCs use non-standard phrasing; high chance of LLM misreads. Scope can creep into full quest tracking.

10. (c) cross-run learning — **Effort: Very High | Risk: Very High** — utility tagging per step, `room_policies` / futility threshold, `runs/` persistence, `knight_orc_strategy.json` merge, startup load. See [[plan-issue10]] for full design.
    - See also requirement 21 (timed multi-step strategies) for a concrete example of what cross-run learning needs to produce

20. ~~the agent has no mechanism for user-authored strategy hints — short natural-language tips that the user has learned across runs and wants to feed back in (e.g. "NPCs only attack orcs they recognise — wear a disguise such as a hooded cloak to avoid attacks").~~ — fixed: `hints` list added to `knight_orc.json` and `GameConfig`; non-empty hints are injected into the `extract_knowledge` prompt under a "Strategy hints from the user" heading so the LLM can match game events to known solutions.
    - **Limitation — only inventory-solution hints are actionable today.** The anomaly→goal pipeline fires only when `potential_solution` matches something in `inventory + spellbook`. Three hint archetypes and their status:
      - ✅ **NPC gift** ("the hermit likes shiny things") — LLM creates `{"target": "hermit", "potential_solution": "gold plate"}`; once agent holds gold plate it tries `use gold plate on hermit`. Works if the game accepts `use X on NPC`.
      - ⚠️ **Discovery** ("the marrow has the cold spell") — no inventory solution; the pipeline can't navigate to a location and examine an object as a goal. Needs requirement 43.
      - ❌ **NPC following** ("follow the ghost to find its lair") — requires multi-turn NPC tracking. Needs requirement 21.

21. the agent cannot learn or execute timed multi-step strategies that require observing NPC behaviour across multiple turns. Canonical example: entering the troll lair requires observing the troll leave, learning that rushing in immediately fails, discovering that following the troll and dropping gold to distract it is the only approach that works.
    - **New capabilities required:**
      - NPC position and movement history tracking (`known_npcs` currently stores only `{location, greeted}`)
      - Event-triggered opportunistic actions ("troll left lair" fires a behaviour sequence)
      - Follow-NPC as a first-class action type in `determine_next_action`
      - Timed/sequenced multi-step plans with awareness that window may close
      - Cross-run strategy store: records what was tried, what was observed, what worked
    - **Relationship to requirement 10:** this is the concrete motivating example for cross-run learning. Resolve requirement 10 first.
    - **Effort: Very High | Risk: Very High**

22. the LLM extraction schema has no way to detect NPC movement events — NPCs wander independently each game tick and may appear in or leave the current room at any time. Without this, `known_npcs` locations go stale immediately.
    - Add `npc_arrivals: ["Denzyl"]` — NPC appeared in current room
    - Add `npc_departures: [{"npc": "Denzyl", "direction": "north", "destination": "inside"}]` — NPC left; direction and destination both optional
    - Agent effects: arrivals update `known_npcs[npc].location = current_room`; departures mark location stale
    - **Effort: Low | Risk: Low** — schema addition + two new branches in `process_agent_step`

23. the LLM extraction schema cannot detect inventory transfers initiated by NPCs — items given by an NPC are not added to inventory, and items stolen are not removed. Both corrupt agent state silently.
    - Add `received_from_npc: [{"item": "spear", "npc": "Denzyl"}]` — item enters inventory via NPC
    - Add `taken_by_npc: [{"item": "gold plate", "npc": "troll"}]` — item leaves inventory involuntarily; agent removes from `inventory`, optionally adds anomaly
    - **Dependency on requirement 20:** implement this first, then add the hint "NPCs will steal treasure from you if they see you carrying it — put it inside a sack or box" via the hint mechanism
    - **Effort: Low | Risk: Low** — schema addition + inventory mutation in `process_agent_step`

25. the agent has no way to detect or respond to route blockages reported by the game ("You are blocked by the drawbridge"). These are distinct from hard failures on objects — the verb is valid but a specific obstacle is preventing movement.
    - Add `blocked_by: [{"obstacle": "drawbridge", "blocking": "route to castle"}]` to extraction schema
    - Agent effect: mark the relevant graph edge as futile OR add as anomaly with `potential_solution: null`
    - **Effort: Low | Risk: Low** — schema addition + one new branch in `process_agent_step`

40. add support for npcs talking to you — Denzyl and others sometimes initiate dialogue ("hi"); agent currently has no way to detect or respond to unprompted NPC speech

43. hint system only supports inventory-solution anomalies — discovery hints ("the marrow has the cold spell") and NPC greeting hints ("say hello to Denzyl") cannot be acted on by the current pipeline.
    - **Discovery hints** need a new goal type: "navigate to a room where this object has been seen and examine/take it". The LLM could emit `{"target": "marrow", "potential_solution": "examine", "hint_driven": true}` but the agent needs a tier that routes `examine`-type solutions to the inspection queue rather than the inventory check.
    - **NPC greeting hints** need the ungreeted-NPC priority tier from requirement 2; until then the LLM can see the hint but the agent has no action to execute.
    - **Effort: Medium | Risk: Low** — extends the anomaly pipeline with a new solution category; does not change LLM prompt or extraction schema

## Closed

6. ~~the engine needs to know it can wear things and should wear disguises (the hood for example)~~ — superseded by requirement 18: `wear` is now in `candidate_verbs` and is attempted on every discovered object
10. ~~(a) death/end detection~~ — fixed: `end_state_patterns` in `knight_orc.json`; `run_evaluator.classify_run()` returns one of six outcomes
10. ~~(b) score parsing~~ — fixed: agent issues `score` command every 20 steps; response parsed into `state["current_score"]` / `state["max_score"]`
18. ~~the agent applies the same inspection sequence to every object regardless of type~~ — fixed: replaced fixed `inspection_sequence` with `candidate_verbs` (15 verbs); records `succeeded`, `blocked`, or `invalid` per verb per entity
27. ~~the agent re-learns verb failures from scratch on every run~~ — fixed: `entity_verb_outcomes` persisted in `knight_orc_strategy.json`; pre-loaded at startup so subsequent runs skip permanently-invalid verbs
