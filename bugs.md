# Bugs

Incorrect or broken behaviour observed during runs. Original issue numbers preserved for git/PR reference.

## Open

45. maze rooms with non-unique names cause the world graph to collapse distinct locations into a single node, triggering a loop. Observed in the "alder clump" area — multiple distinct rooms share the same name but have different exits; the agent loops because it thinks it has already visited and explored the single "alder clump" node.
    - **Root cause:** `update_graph` uses the raw room name as the node ID; `state["current_room"]` is set from `extracted["room"]` before exits are known, so there is no opportunity to fingerprint at assignment time
    - **Fix:** use `room_name + sorted(exits)` as the canonical node ID (e.g. `"alder clump {E,S,SW,W}"`); resolve the fingerprinted ID after exits are extracted and use it for both `state["current_room"]` and the edge from the previous room; display label strips the suffix for readability
    - **Edge cases:** LLM returns empty exits on first visit (defer fingerprinting until exits are known); exits discovered incrementally across visits (merge into existing fingerprinted node or create new one)
    - **Effort: Medium | Risk: Medium** — touches `update_graph`, `process_agent_step`, and `state["current_room"]` assignment; needs careful handling of the empty-exits case

38. we see a location of "propet_northeast" — LLM is hallucinating a room name from a direction or property string; prompt needs a rule that room names must be proper nouns from the game narrative, not directions or adjectives
39. we see a location of "Denzyl" not listed as an npc — LLM is placing an NPC name in the `room` field; prompt should clarify that room must be a place, not a character name; `Denzyl` should appear in `npcs` instead

## Closed

1. ~~the engine is reading the horse, and looking inside the horse, it should recognize that a horse cannot be read~~ — fixed: LLM schema now separates `npcs` (living creatures) from `objects` (inanimate items); NPCs are routed to `known_npcs` state and never enter the inspection queue
4. ~~the engine needs to understand "you can see" and interpret that as things in the current room~~ — fixed: prompt now explicitly states "you can see X" means X is in the current room (objects/npcs), not inventory
5. ~~the engine appears to assume that if it can see it; it is in the inventory - it needs to take things to be in the inventory~~ — fixed: `known_entities` status updated from `discovered` → `held` when `added_to_inventory` fires; overlapping prompt fix landed in issue 4
7. ~~the engine doesnt understand "you cant do that", "you cant see the huge knight"~~ — fixed: `_is_failure_response()` detects Level 9 refusal phrases and immediately clears the current inspection target so the agent moves on rather than continuing the take/examine/read/look-inside sequence
8. ~~`anomaly_data.get("potential_solution", "").lower()` throws `NoneType has no attribute lower` when the LLM returns `null` for `potential_solution`~~ — fixed: normalise to `""` at storage time in `process_agent_step` so `None` never enters state
11. ~~the warning command timed out message is annoying, i cant use it to debug what it actually received, improve the console logging or display the raw text so i can update the rules~~ — fixed: timeout now includes the raw partial output received before pexpect gave up; Streamlit file watcher disabled to prevent coredumps; action_taken wired into LLM extraction prompt
12. ~~the engine is still trying to read the horse~~ — fixed: `_is_creature()` pre-filter in `process_agent_step` blocks creature-named items from entering `uninspected_objects`; any creature-word match is routed to `known_npcs` instead
13. ~~eval suite (`tests/test_evals.py`) found the quantized Phi-3.5-mini model non-deterministically returns `{}` for short inputs~~ — fixed: `extract_knowledge` retries up to 3 times on empty result; prompt updated with two few-shot examples and explicit rules for terse take confirmation and exits extraction
14. ~~the test loop detection is suspicious - we can look at lots of items in the inventory but this would fail the loop check~~ — fixed: direction actions that produce a room change are excluded from the loop count (productive navigation); object inspection never false-positives because each action includes the object name as a unique string
17. ~~after a few successful commands, knightorc will respond with "we wont bother with what now from now on" and the prompt changes to ">" on its own~~ — fixed: `prompt_pattern` in `game_config.py` and `configs/knight_orc.json` updated to `What now\?|\r?\n>` so pexpect matches both prompts
19. ~~after `use putty knife on rubbish` the agent crashed with `ValueError: None cannot be a node` and then repeated the same command in a loop~~ — fixed: `extracted.get("room")` replaces `"room" in extracted` so null values don't corrupt `state["current_room"]`; `update_graph` gains an early-return guard for None room; goal hard-failure removes anomaly from queue
29. ~~command echo appearing in game responses — pexpect includes a leading `\r\n` before the echoed command in some terminal modes~~ — fixed: `lstrip('\r\n ')` before the startswith check in `execute_game_command`; regression test added
31. ~~LLM does not extract sub-objects revealed when examining another object — "examine flagpole" returned "Fastened to it is a halyard" but only `flagpole` was extracted, not `halyard`~~ — fixed: added prompt rule "if examining an object reveals another distinct item (fastened to it, inside, attached), include it in objects"
32. ~~crash details not visible in log file~~ — fixed: crash appended to `state["game_log"]` as a `utility: "crash"` entry before the log is written
33. ~~the prompt needs to understand "Exits lead in all directions."~~ — fixed: prompt rule expands this to `["north", "south", "east", "west", "up", "down"]`; eval cases added
35. ~~LLM hallucinates `added_to_inventory` on hard-failure responses such as "You don't need to use the word X"~~ — fixed: `extracted.pop("added_to_inventory", None)` on hard failure before log entry is written; eval case and regression tests added
36. ~~after issuing "wear flagpole" we get a location of "current location not explicitly named"~~ — fixed: prompt now explicitly instructs the LLM to set `room` to null if the response does not name a location
37. ~~after issuing "examine flagpole" we get a location of "flagpole" which is not correct~~ — fixed: same prompt rule as 36; `room` must be null when no place is named
41. ~~add negative failure for "Don't be silly"~~ — fixed by user
