# Features

Enhancements, new functionality, tooling, and infrastructure improvements.

## Open

24. game-specific navigation and interaction commands are hardcoded in `determine_next_action` rather than declared in config, making the agent Knight Orc-specific.
    - Add `navigation` block to `game_config.json`: `fast_nav_command` (`"run to {target}"`) and `full_nav_command` (`"go to {target}"`) — both optional; absent means use graph-based step-by-step fallback
    - Add `npc_commands` block: `follow_command`, `wait_for_command`
    - Add `item_commands` block: `drop_command`, `give_command`
    - **Effort: Low | Risk: Low**

26. add pipeline and quality gates to ensure the docs are good standard and the python tests pass

34. add support for cloud LLM instead of llama.cpp
    - ~~**Step 1:** instrument token usage — accumulate per-step in `watch_run.py`, print final summary with configurable price env vars~~ — done
    - **Step 2:** swap `llm.py` to call a cloud API (OpenAI-compatible endpoint, Anthropic, DeepSeek) — gated behind an env var so local llama.cpp still works
    - **Effort: Medium | Risk: Low**

43. the world map does not represent elevation (up/down connections) or interior zones — all rooms are rendered flat on a single cardinal plane
    - **Option A (do first):** layered single map — extend `_compute_cardinal_positions` BFS in `ui.py` to track a Z-level per room (`up` edge = +1, `down` edge = -1, ground = 0). Map Z to a vertical Y-band in the pyvis view; colour-code nodes by level. No zone detection needed; low risk; immediately useful.
    - **Option B (later):** zone clustering — detect interiors as room clusters reachable from the main graph only via a single chokepoint node. Render each zone as a separate Streamlit tab with its own layered map. The Z-level data from Option A feeds directly into this.
    - **Effort: Low (A) / High (B) | Risk: Low (A) / Medium (B)**

44. generate thumbnail images for each map location based on accumulated room descriptions
    - The game log already captures raw `response` text and `extracted.room` per step; all the source material exists
    - **Step 1:** post-process the game log to aggregate all responses per room name into a single description string
    - **Step 2:** send each description to an image generation API (DALL-E, Stable Diffusion, etc.) with a style prompt (e.g. "pixel art top-down RPG view of: {description}")
    - **Step 3:** display thumbnails on the map node or in a hover tooltip in the pyvis graph
    - **Dependency:** requires a cloud image API — new external dependency not currently in the project
    - **Effort: Medium | Risk: Low**

## Closed

3. ~~the boxes for the graph dont scale and there's no zoom so you cant read the text~~ — fixed: replaced matplotlib with pyvis interactive graph (zoom, pan, drag; current room highlighted in amber)
9. ~~the visualisation of the map doesnt draw the boxes in cardinal direction~~ — fixed: BFS from first discovered room assigns pixel positions from edge direction labels; physics disabled so nodes stay pinned
15. ~~the map visualization doesnt word wrap the room descriptions~~ — fixed: `_display_label()` abbreviates `Unknown (X from Y)` to `? X` and word-wraps long names at 20 chars
16. ~~move all knightorc-specific code into a config file~~ — fixed: `game_config.py` introduces a `GameConfig` singleton; `configs/knight_orc.json` holds creature words, failure phrases, prompt pattern and inspection sequence
28. ~~watch_run.py takes ages for each step, add metrics to each line~~ — fixed: each verbose line now shows `[elapsed Xm00s | X.Xs/step | remaining Xm00s]`
30. ~~add a way to break out of watch_run.py but ensure it closes cleanly and writes the logs~~ — fixed: `except KeyboardInterrupt` saves logs, appends `INTERRUPTED` entry, classifies run as `"interrupted"`
34. ~~Step 1: instrument token usage~~ — fixed: per-step token usage tracked via `llama_cpp` response `usage` field; total and estimated cost printed at end of run
42. ~~`watch_run.py` does not print the run ID at startup~~ — fixed: run ID printed to stderr immediately after game start
