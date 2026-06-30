---
name: plan-issue10
description: "Implementation plan for issue 10 — persistent memory, score tracking, action utility learning, emergent wait strategy, cross-run strategy store"
metadata: 
  node_type: memory
  type: project
  originSessionId: c2522a1b-99cd-443d-ab77-1944c6ff3507
---

# Issue 10 — Persistent Memory & Cross-Run Learning

Designed 2026-06-30. See [[project-status]] for open issues context.

---

## Scope (3 sub-problems)

**(a) Death/end detection** — pattern match on game output (Low effort)
**(b) Score tracking** — agent periodically issues `score` command (Low effort)
**(c) Cross-run learning** — action utility tagging, per-room policy, strategy store (High effort)

Tackle a+b first; c depends on them.

---

## Part (a) — Death / end detection

Add patterns to `configs/knight_orc.json` under a new `"end_state_patterns"` key:
- Death: `"you have died"`, `"you are dead"`, `"killed"`
- Game end: `"the end"`, `"congratulations"`, `"you have finished"`
- Score summary: `"you score \d+ out of \d+"` (also used by part b)

Detect in `run_evaluator.py` (new file) by scanning the last N log entries.

Run outcome categories (5):
- `agent_failure` — loop_detected, crash, or timeout in findings
- `game_ended` — end_state_pattern matched
- `ambiguous` — hit max steps, no score change, no error
- `score_improved` — final score > initial score
- `finished` — "congratulations" / "finished" matched

---

## Part (b) — Score tracking

- Add `score` as a periodic agent action: issued every 20 steps in `determine_next_action` when no higher-priority action exists.
- Add `"score"` to game_config so it is not treated as exploration.
- Parse response with regex `you score (\d+) out of (\d+)` → record `state["current_score"]` and `state["max_score"]`.
- Include score in each `game_log` entry and in the run metadata JSON.

---

## Part (c) — Action utility & emergent wait strategy

### Action utility tagging (per step in game_log)

After each `process_agent_step`, compute `utility` from diff of state before/after:

| Value | Signal |
|---|---|
| `productive` | Room changed, new object/NPC found, item added to inventory |
| `informative` | Look/examine returned unseen description (text not previously seen for this target) |
| `redundant` | Look/examine of already-fully-known entity |
| `futile` | Hard failure response, zero state change |

Stored as `last["utility"]` in game_log entry. Deterministic — no LLM needed.

### Per-room movement policy (where "wait" emerges)

Add `room_policies` to agent state: `{room_name: {futile_directions: [...], preferred_when_stuck: "wait"|None}}`

Logic in `determine_next_action`:
1. Before attempting a direction, check `room_policies[current_room]["futile_directions"]` — skip if listed
2. If last N (config: default 3) direction attempts from this room were all `futile`, add all tried directions to `futile_directions` and set `preferred_when_stuck = "wait"`
3. When stuck (no uninspected objects, no unknown exits, no anomalies) and `preferred_when_stuck == "wait"`, issue `wait` instead of a direction

The "look at everything first, try to move, fail, then wait" sequence emerges naturally:
- Phase 1: uninspected objects → `informative` (useful)
- Phase 2: try directions → `futile` (knight prevents movement)
- Phase 3: threshold hit → `wait` added to policy

### Cross-run persistence

**Per-run output:** `runs/<watch_TIMESTAMP>.json`
```json
{
  "run_id": "watch_20260630_115024",
  "outcome": "score_improved",
  "final_score": 5,
  "max_score": 1000,
  "steps": 50,
  "room_policies": {...}
}
```

**Accumulated strategy:** `configs/knight_orc_strategy.json`
```json
{
  "room_policies": {
    "Starting Location": {
      "futile_directions": ["north", "south", "east", "west"],
      "preferred_when_stuck": "wait"
    }
  },
  "run_history": [
    {"run_id": "...", "outcome": "score_improved", "final_score": 5}
  ]
}
```

After each run, `run_evaluator.py` merges newly discovered `room_policies` into the strategy file (union of futile_directions, keep preferred_when_stuck if set).

On startup, `make_initial_state()` in `watch_run.py` loads `room_policies` from strategy file so the agent skips already-known-futile moves immediately.

### Crash classification

`run_evaluator.classify_finding(finding, game_log)` returns:
- `config_fix` — failure response text not matched by any existing pattern in `knight_orc.json`
- `python_fix` — finding type is `crash` (exception traceback present)
- `ambiguous` — neither above; needs human review

---

## Build order

1. `run_evaluator.py` — outcome classifier + crash classifier (no agent changes)
2. Score: `score` periodic action + `current_score` in state + parser
3. Utility tagging on each game_log entry
4. `room_policies` in state + futility threshold in `determine_next_action`
5. `runs/` directory + per-run JSON write from `watch_run.py`
6. `knight_orc_strategy.json` merge step in `run_evaluator.py`
7. Load `room_policies` from strategy at `make_initial_state()` startup
8. Tests for each layer

---

## Key decisions

- **"wait" is never hardcoded** — it emerges from the futility threshold after one or two runs
- **Utility is state-diff based, not LLM** — fast, deterministic, no extra LLM calls
- **Score via `score` command** — simpler than parsing status bar; pexpect sees it fine
- **Strategy file is human-readable JSON** — user can inspect/edit what the agent has learned
