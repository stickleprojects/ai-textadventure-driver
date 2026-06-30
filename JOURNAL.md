# Development Journal

Running notes on findings, decisions, and things that surprised us during development.

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
