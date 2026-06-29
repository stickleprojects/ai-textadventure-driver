# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
always branch off develop, never commit direct on develop or main


## Project Overview

A Streamlit-based autonomous agent that plays the classic text adventure "Knight Orc" (Level 9 Computing, 1987). The agent uses a local llama.cpp LLM for natural language understanding and NetworkX for spatial mapping. It runs entirely offline.

## Running the App

```bash
# Activate the virtual environment first
source .env/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app defaults to:
- Interpreter: `./tools/glklevel9`
- ROM: `./gamefiles/knight-orc/GAMEDAT1.DAT`
- LLM model: `../models/Phi-3.5-mini-instruct-Q3_K_M.gguf` (outside this repo)

`llama-cpp-python` is optional — the app runs without it, but `determine_next_action()` will have no LLM-extracted knowledge to act on.

## Architecture

The app is split into four modules plus a thin Streamlit entry point:

| File | Streamlit dependency | Responsibility |
|------|---------------------|----------------|
| `game_engine.py` | None | pexpect subprocess interface |
| `llm.py` | `@st.cache_resource` only | Model loading + knowledge extraction |
| `agent.py` | None | State mutation and decision logic |
| `ui.py` | Yes | Rendering helpers |
| `app.py` | Yes | Session state init, layout, wiring |

**`game_engine.py`** — `start_level9` spawns `glklevel9` via `pexpect` and returns `(child, initial_text)`. `execute_game_command(child, command)` sends a line and reads until the `What now?` prompt. ANSI escape codes are stripped before output is passed to the LLM.

**`llm.py`** — `extract_knowledge(text, action, llm)` sends raw game output to the local LLM with a fixed JSON schema prompt. Extracts: current room, exits, visible objects, items added to inventory, learned spells, new anomalies (`{target, reason, potential_solution}`), and resolved anomalies. Falls back to `{}` on parse failure.

**`agent.py`** — All functions take `state` (the `st.session_state.system_state` dict) as an explicit parameter.

- `update_graph(state, room, exits, previous_room, action)` — maintains the NetworkX `DiGraph`; nodes are room names, edge labels are movement directions; unseen exits are added as `Unknown (direction from room)` nodes.
- `determine_next_action(state)` — priority order:
  1. Navigate to `active_goal.room`, then apply solution (`use X on Y` or `cast X on Y`)
  2. Check `unresolved_anomalies` — if inventory/spellbook can solve one, set it as `active_goal`
  3. Continue `current_inspection` multi-step sequence (`take → examine → read → look inside`)
  4. Pop next item from `uninspected_objects`, begin inspection
  5. Follow any edge leading to an `Unknown (...)` node
  6. Fallback: `look`
- `process_agent_step(state, child, llm)` — calls decide → act → extract → update state → append to `game_log`.

**`ui.py`** — `render_graph(state)` draws the world graph via matplotlib/NetworkX. `generate_markdown_log(state)` serialises `game_log` to a markdown string for download.

**`app.py`** — Initialises `st.session_state.system_state` and `level9_process`, renders the sidebar and two-column dashboard, and drives the auto-run loop (`st.rerun()` each cycle).

## Known Issues (from issues.md)

- Agent doesn't distinguish NPCs from objects (tries to `read` or `look inside` horses, knights)
- NPC interaction (greet, ask for help) is not implemented
- World graph node boxes don't scale with text — long room names overflow
- `"you can see X"` is not reliably parsed as room contents vs. inventory
- Object visibility ≠ possession; agent sometimes assumes seen items are in inventory
- Wearable items / disguises (e.g. hood) not handled
- "You can't do that" responses are not fed back to correct the inspection queue

## Game Files

`gamefiles/knight-orc/` contains the original Level 9 ROM files. `GAMEDAT1.DAT` is the primary entry point. `tools/glklevel9` is the dumb-GLK Level 9 interpreter binary (Linux x86-64).
