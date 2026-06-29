# issues

1. ~~the engine is reading the horse, and looking inside the horse, it should recognize that a horse cannot be read~~ — fixed: LLM schema now separates `npcs` (living creatures) from `objects` (inanimate items); NPCs are routed to `known_npcs` state and never enter the inspection queue
2. the engine needs to know it can speak to npcs, so it should politely greet them and ask them for help or offer to help them, once it finds out what they need they can help it in the future
   - Detection: add `npcs` field to LLM extraction schema alongside `objects`
   - Conversation: fixed single greeting (`greet X`) for now; deeper dialogue later
   - State: new `known_npcs` dict `{name: {location, greeted, needs, can_help_with}}`
3. ~~the boxes for the graph dont scale and there's no zoom so you cant read the text (the text is too large to fit into the boxes)~~ — fixed: replaced matplotlib with pyvis interactive graph (zoom, pan, drag; current room highlighted in amber)
4. the engine needs to understand "you can see" and interpret that as things in the current room
5. the engine appears to assume that if it can see it; it is in the inventory - it needs to take things to be in the inventory
   - Fix: tighten LLM prompt so `added_to_inventory` only fires on explicit take confirmations ("Taken.", "You pick up the...")
   - State: `known_entities` already stores `location`; update status from `discovered` → `held` when taken
6. the engine needs to know it can wear things and should wear disguises (the hood for example)
7. ~~the engine doesnt understand "you cant do that", "you cant see the huge knight"~~ — fixed: `_is_failure_response()` detects Level 9 refusal phrases and immediately clears the current inspection target so the agent moves on rather than continuing the take/examine/read/look-inside sequence
8. ~~`anomaly_data.get("potential_solution", "").lower()` throws `NoneType has no attribute lower` when the LLM returns `null` for `potential_solution`~~ — fixed: normalise to `""` at storage time in `process_agent_step` so `None` never enters state
9. ~~the visualisation of the map doesnt draw the boxes in cardinal direction, update it so "east of XXX" appears to the right of its target~~ — fixed: BFS from first discovered room assigns pixel positions from edge direction labels (`_compute_cardinal_positions`); physics disabled so nodes stay pinned; zoom/pan/drag still work
10. we need to add persistent memory so that the engine can learn successful or failed ways of playing the game and improve, this also means we need to understand the game scoring mechanism and "you died" or "the game ended" states
11. the warning command timed out message is annoying, i cant use it to debug what it actually received, improve the console logging or display the raw text so i can update the rules
12. the engine is still trying to read the horse
13. eval suite (`tests/test_evals.py`) found the quantized Phi-3.5-mini model non-deterministically returns `{}` for short inputs ("Taken.", terse room descriptions) — extraction reliability needs improvement, possibly via retry logic, prompt tuning, or a larger/better-quantized model
14. the test loop detection is suspicious - we can look at lots of items in the inventory but this would fail the loop check, we might need to use an llm to check for this instead of raw compares

