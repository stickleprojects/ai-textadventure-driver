# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
always branch off develop, never commit direct on develop or main


## Project Overview

A Streamlit-based autonomous agent that plays the classic text adventure "Knight Orc" (Level 9 Computing, 1987). The agent uses an LLM for natural language understanding and NetworkX for spatial mapping. Supports both local llama.cpp models and cloud LLM providers (DeepSeek, OpenAI-compatible) via env vars.

## Running the App

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app defaults to:
- Interpreter: `./tools/glklevel9`
- ROM: `./gamefiles/knight-orc/GAMEDAT1.DAT`
- LLM model: `../models/Phi-3.5-mini-instruct-Q3_K_M.gguf` (outside this repo)

`llama-cpp-python` is optional — the app runs without it, but `determine_next_action()` will have no LLM-extracted knowledge to act on.

## Architecture

### Core modules

| File | Streamlit dependency | Responsibility |
|------|---------------------|----------------|
| `game_engine.py` | None | pexpect subprocess interface |
| `llm.py` | `@st.cache_resource` only | Model loading + knowledge extraction (local or cloud) |
| `agent.py` | None | State mutation and decision logic |
| `run_evaluator.py` | None | Run classification, strategy load/merge, sidecar persistence |
| `game_config.py` | None | `GameConfig` singleton loaded from `configs/knight_orc.json` |
| `env_utils.py` | None | `.env` file loader with indirect variable expansion |
| `anomaly_detector.py` | None | Deterministic detector helpers (used by `detect_anomalies.py`) |
| `log_analyzer.py` | None | Log analysis helpers (suspicious rooms, empty extractions, etc.) |
| `ui.py` | Yes | pyvis graph rendering, log export helpers |
| `app.py` | Yes | Session state init, layout, auto-run loop |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/watch_run.py` | Headless runner; `--detect/--review/--architect` flags; cloud LLM via env vars |
| `scripts/detect_anomalies.py` | Stage 1: deterministic anomaly detectors → `anomaly_report.json` |
| `scripts/llm_review.py` | Stage 1b: LLM review of the run log, appends findings to report |
| `scripts/architect.py` | Stage 2: anomaly → fix plan; deduplicates via `plans/index.json` |
| `scripts/generate_evals.py` | Generate eval fixtures from a saved game log (requires `ANTHROPIC_API_KEY`) |
| `scripts/generate_map.py` | Regenerate a map PNG from a saved run log |
| `scripts/agent_dev_loop.py` | Dev orchestrator: detect → architect → dev → verify (feature 56) |

### Key data files

| File | Contents |
|------|---------|
| `configs/knight_orc.json` | Game config (verbs, creature words, failure patterns, nav commands) |
| `configs/knight_orc_strategy.json` | `futile_edges` + `run_history` (cross-run persistence) |
| `configs/knight_orc_rooms.json` | World graph sidecar (nodes + edges) |
| `configs/knight_orc_items.json` | Entity verb outcomes sidecar |
| `plans/index.json` | Central fix-plan registry; dedup by anomaly type |
| `schemas/` | JSON Schema Draft-7 files validating all of the above |

**`game_engine.py`** — `start_level9` spawns `glklevel9` via `pexpect` and returns `(child, initial_text)`. `execute_game_command(child, command)` sends a line and reads until the `What now?` prompt. ANSI escape codes are stripped before output is passed to the LLM.

**`llm.py`** — `extract_knowledge(text, action, llm)` sends raw game output to the LLM with a fixed JSON schema prompt. Extracts: current room, exits, visible objects, items added to inventory, learned spells, new anomalies (`{target, reason, potential_solution}`), and resolved anomalies. Falls back to `{}` on parse failure. `load_cloud_llm(provider, model, api_key)` returns an OpenAI-compatible adapter for cloud providers.

**`agent.py`** — All functions take `state` (the `st.session_state.system_state` dict) as an explicit parameter.

- `update_graph(state, room, exits, previous_room, action)` — maintains the NetworkX `DiGraph`; nodes are room names, edge labels are movement directions; unseen exits are added as `Unknown (direction from room)` nodes.
- `determine_next_action(state)` — priority order:
  1. `position_lost` flag → `look` to re-establish position
  2. Active goal → navigate to target room, then apply solution (`use X on Y` or `cast X on Y`)
  3. Check `unresolved_anomalies` — if inventory/spellbook can solve one, set it as `active_goal`
  4. Continue `current_inspection` multi-step sequence (dynamic: skips verbs already marked `invalid`)
  5. Pop next item from `uninspected_objects`, begin inspection
  6. Follow Unknown exit from current room (up/down only if those words appear in response text)
  7. Navigate to nearest room with Unknown exits (diversity-scored: `path_len + min_recent_dir_frequency`)
  8. Navigate to nearest unvisited known room (from pre-seeded world graph)
  9. `score` every 20 steps
  10. `wait for <npc>` if `pending_npc_tasks` is non-empty
  11. Fallback: `look`
- `process_agent_step(state, child, llm)` — calls decide → act → extract → update state → append to `game_log`.

**`ui.py`** — `render_graph(state)` draws the interactive world graph via pyvis (zoom, pan, drag; current room highlighted). `generate_markdown_log` / `generate_json_log` serialise `game_log` for download.

**`app.py`** — Initialises `st.session_state.system_state` and `level9_process`, renders the sidebar and two-column dashboard, and drives the auto-run loop (`st.rerun()` each cycle).

## Known Issues

Issues are tracked in three files:
- `bugs.md` — incorrect or broken behaviour
- `requirements.md` — agent gameplay capabilities not yet implemented
- `features.md` — enhancements, tooling, infrastructure

Current open issues of note:
- Agent doesn't distinguish NPCs from objects (tries to `read` or `look inside` horses, knights)
- NPC interaction (greet, ask for help) is not implemented
- `"you can see X"` is not reliably parsed as room contents vs. inventory
- Object visibility ≠ possession; agent sometimes assumes seen items are in inventory
- Wearable items / disguises (e.g. hood) not handled

## Planning

When designing or documenting an implementation plan (in memory, issues.md, or a design session), always include a **"How will we know it worked?"** section. This must describe what you would actually observe during a live runthrough of the game — not test results, not log file contents alone, but visible agent behaviour:

- Which action(s) should appear (or stop appearing) in the step log
- What the agent should do differently compared to before the change
- Which files or state fields to inspect and what to expect in them
- If the change only takes effect across multiple runs, describe what to watch on run 1 vs run 2

This section is required for any plan that touches agent behaviour, the LLM extraction schema, the config, or the strategy store.

## Journal

`JOURNAL.md` in the project root records non-obvious findings, rejected approaches, and design decisions. Add an entry whenever:
- An approach was considered and rejected (and why)
- A surprising behaviour was observed in the game or LLM
- A non-obvious architectural decision was made

Format: `## YYYY-MM-DD` heading, short subheading, 2–4 paragraphs covering what was observed, why it matters, and what was decided. Do not journal routine fixes — only things that would surprise a future reader.

## Quality Gates

**Linting** — `ruff check .` (config in `pyproject.toml`). Rules: E/F/W, E501 deferred. Per-file E402 ignores for files that call `load_env_file()` before project imports.

**JSON validation** — `pytest tests/test_json_schemas.py`. Validates all config, plan, and eval fixture files against `schemas/*.schema.json`. Also checks fixture ID uniqueness and plan index/file consistency.

**CI** — `.github/workflows/ci.yml` runs both jobs on every push/PR to `main` or `develop`. Tests run with `-m "not llm and not integration and not slow"`.

## Game Files

`gamefiles/knight-orc/` contains the original Level 9 ROM files. `GAMEDAT1.DAT` is the primary entry point. `tools/glklevel9` is the dumb-GLK Level 9 interpreter binary (Linux x86-64).
