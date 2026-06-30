---
name: project-status
description: "Current branch, open issues and recommended next steps as of 2026-06-30"
metadata:
  node_type: memory
  type: project
  originSessionId: b35cd896-b84c-49d9-9ba7-c8ff443373d4
---

## State as of 2026-06-30

**Branch:** `develop` — up to date with origin. No feature branch open.

**Last merged PRs (this session):**
- #11 — JOURNAL.md context updates
- #12 — Issue 18: attempt-and-learn verb-object compatibility (candidate_verbs, hard/soft failure split, verb_outcomes in known_entities)
- #13 — issues 6 and 18 marked fixed
- #14 — Issue 19: null room crash fix + stuck anomaly loop fix + watch_run.py full log persistence
- #15 — Issues 20 and 21 documented (hints and timed strategy learning)
- #16 — 'don't need to use the word' added as hard failure pattern

---

## Open issues (from issues.md)

| # | Summary | Effort | Risk | Notes |
|---|---|---|---|---|
| 2 | NPC interaction (greet, ask, quest tracking) | High | High | New priority tier in determine_next_action; multi-turn dialogue |
| 10 | Persistent memory — death/end detection (a), score parsing (b), cross-run learning (c) | Very High | Very High | Do a+b first; c is the hard part. See issue 21 as target for c. |
| 20 | User-authored hints injected into LLM prompt | Low | Low | Config field + one prompt change in llm.py; no new state |
| 21 | Timed multi-step strategies learned across runs (troll lair example) | Very High | Very High | Needs NPC tracking, follow action, timed sequences, strategy store |

---

## Architecture quick-reference

| File | Role |
|---|---|
| `agent.py` | `_is_hard_failure`, `_is_soft_failure`, `_record_verb_outcome`, `determine_next_action`, `process_agent_step` |
| `llm.py` | `extract_knowledge()` — 3-attempt retry; prompt omits `room` on terse responses |
| `game_engine.py` | pexpect; matches both `What now?` and `>` prompts via `config.prompt_pattern` |
| `game_config.py` | `GameConfig` singleton; `candidate_verbs`, `hard_failure_pattern`, `soft_failure_pattern`, `failure_pattern` (union); `inspection_sequence` property alias |
| `configs/knight_orc.json` | candidate_verbs (15 verbs), creature_words, hard_failure_patterns, soft_failure_patterns |
| `log_analyzer.py` | `analyze_log(game_log)` — detectors: empty_llm_extraction, hallucinated_room_name, loop_detected, command_timeout_or_error, creature_misclassified_as_object, inspection_failure, stuck_in_room, blocked_verb_rate |
| `ui.py` | pyvis graph, log exports |
| `app.py` | Streamlit entry point; loads config from `GAME_CONFIG` env var |
| `scripts/watch_run.py` | Headless runner; writes `logs/watch_TIMESTAMP.json` on finish or crash; loop/timeout findings include raw responses |
| `scripts/run_and_analyze.py` | Run + analyze; writes `logs/run_TIMESTAMP.json` and `logs/latest_analysis.md` |
| `scripts/agent_dev_loop.py` | Agentic loop: run → analyze → Claude fixer (tool use) → pytest → repeat |
| `tests/` | 93 unit tests (61 agent, 21 log_analyzer, evals) |

## Key state shape (known_entities)
```json
{
  "sword": {
    "status": "discovered | held",
    "location": "Hall",
    "verb_outcomes": {
      "take": "succeeded",
      "read": "invalid",
      "wear": "blocked"
    }
  }
}
```

## Adding new log detectors
1. Add check + issue dict to `log_analyzer.py`
2. Add unit tests in `tests/test_log_analyzer.py` (one fires, one clean)
3. Optionally add eval fixture if the fix involves the LLM prompt

## Adding new failure phrases
Edit `configs/knight_orc.json` → `hard_failure_patterns` (permanent) or `soft_failure_patterns` (state-dependent).
If general Level 9 behaviour, also add to `game_config.py` defaults so tests cover it without loading JSON.

## Issue 10 design — see [[plan-issue10]] for full detail

**Full design agreed 2026-06-30. Key points:**

- Score: agent issues `score` command every ~20 steps; parse `"you score X out of 1000"`
- Death/end: new `end_state_patterns` in `knight_orc.json`; `run_evaluator.py` classifies run into 5 outcome categories
- Utility tagging: every `game_log` entry gets `utility` = productive / informative / redundant / futile — derived from state diff, no LLM
- **"Wait" strategy is emergent, not hardcoded**: after N consecutive `futile` direction attempts from a room, agent adds those directions to `room_policies[room]["futile_directions"]` and sets `preferred_when_stuck = "wait"`. Discovered in one or two runs, persisted to `configs/knight_orc_strategy.json`.
- Cross-run store: `runs/<timestamp>.json` per run; strategy file accumulates `room_policies` across runs; loaded at startup
- Crash classifier in `run_evaluator.py`: config_fix (unmatched failure phrase) vs python_fix (exception) vs ambiguous

**Build order:** run_evaluator → score tracking → utility tagging → room_policies/futility → runs/ persistence → strategy merge → startup load → tests
