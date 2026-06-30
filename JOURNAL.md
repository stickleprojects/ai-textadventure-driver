# Development Journal

Running notes on findings, decisions, and things that surprised us during development.

---

## 2026-06-30

### Attempt-and-learn beats pre-classification for verb-object compatibility (issue 18)

Initial plan for issue 18 was to classify every object into a type (container, clothing, food, etc.) using the LLM at extraction time, then look up allowed verbs from a config table. Rejected for two reasons:

1. **The game is a better oracle than the LLM.** Knight Orc will always tell you when a verb doesn't apply — and crucially, *how* it fails tells you something the LLM can't: whether the failure is permanent ("you can't wear an apple") or state-dependent ("you can't wear that right now, you're already wearing armour"). A static type table can't capture that distinction.

2. **Wearable state is compositional.** You can wear a hat and gloves at the same time but not a cloak over armour. This depends on what the agent is currently wearing — not on the object's type. No pre-classification scheme handles this; only the game's response does.

**Decision:** replace the planned type-classification approach with attempt-and-learn. Try a broad candidate verb list on every object; record outcomes (`succeeded`, `blocked`, `invalid`) in `known_entities`; skip `invalid` verbs permanently and defer `blocked` ones until state changes. Splits `failure_pattern` in config into `hard_failure_patterns` and `soft_failure_patterns`.

**Why this is lower risk:** pattern matching on game responses is already proven (`_is_failure_response`). No dependency on LLM classification reliability.

---

### Room name hallucination from terse LLM responses (issue observed in watch_run)

Observed in a 10-step run: after `take putty knife` (response: `"Taken."`), the LLM returned `"room": "current location"`. After `examine putty knife`, it returned `"room": "putty knife room"` — a room name invented from the action text.

Root cause: the prompt required a `room` field on every extraction. When the game response contains no room description, the LLM fills in a placeholder or derives one from the action.

Fix: explicit prompt instruction to omit `room` entirely on terse responses. Detection added to `log_analyzer.py` (`hallucinated_room_name` issue type) to catch this pattern in future runs.

---

### Agentic dev loop architecture

Built `scripts/agent_dev_loop.py`: run game headlessly → analyze log for issue patterns → feed analysis to a Claude fixer agent (Anthropic API, tool use: bash/read_file/edit_file) → run pytest → repeat. The fixer agent is directed to `configs/knight_orc.json` first for creature/failure-phrase issues, then `llm.py` for prompt issues, then `agent.py` for logic issues.

`log_analyzer.py` is the detection layer — pure Python, no LLM, unit-tested. Pattern for adding detectors: write the check, write a test that fires on a synthetic bad log entry and a test that passes on a clean one.
