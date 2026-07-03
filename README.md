# Knight Orc Autonomous OS

An AI agent that plays the 1987 Level 9 text adventure **Knight Orc** autonomously. It uses an LLM (local llama.cpp or a cloud provider) to understand the game's natural language output, builds a spatial map of the world as it explores, tracks items and spells, and resolves blocked paths once it has the right tool or magic.

Built with Streamlit so you can watch it think in real time. Also runs headless via `scripts/watch_run.py`.

![Dashboard showing spatial graph, I-O terminal, and state schema]()

---

## How it works

The agent runs a continuous decide → act → extract → update loop:

1. **Game engine** — `glklevel9` runs as a subprocess; the agent sends text commands and reads the response.
2. **Knowledge extraction** — raw game text is sent to an LLM which returns structured JSON (room, exits, objects seen, items picked up, spells learned, anomalies encountered).
3. **Decision logic** — a priority stack determines the next command:
   - Re-establish position if lost
   - Pursue an active goal (navigate to a blocked object, then apply the known solution)
   - Resolve a known anomaly if inventory/spellbook now allows it
   - Inspect the next queued object (dynamic verb sequence; skips already-invalidated verbs)
   - Explore an unknown exit
   - Navigate toward unexplored areas (diversity-scored)
   - Fallback: `look`
4. **World graph** — rooms are nodes, movement directions are edges; shortest-path navigation is used when backtracking.
5. **Cross-run learning** — `futile_edges` and entity verb outcomes persist across runs in `configs/`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.14 |
| `glklevel9` binary | Included at `tools/glklevel9` (Linux x86-64). Other platforms need to compile from source: [glk-implementations](https://github.com/DavidGriffith/glk-implementations) |
| Knight Orc game files | Level 9 released their games as freeware. Obtain `GAMEDAT1.DAT`, `GAMEDAT2.DAT`, `GAMEDAT3.DAT` from the [Level 9 Memorial](http://www.if-legends.org/~l9memorial/html/home.html) or Abandonware archives and place them in `gamefiles/knight-orc/` |
| LLM | Either a GGUF model for local inference (default: `../models/Phi-3.5-mini-instruct-Q3_K_M.gguf`) **or** a cloud provider via env vars (see below) |

---

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd <repo-dir>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env — set LLM_API_KEY if using a cloud provider

# Place game files
mkdir -p gamefiles/knight-orc
cp /path/to/your/GAMEDAT*.DAT gamefiles/knight-orc/

# Make the interpreter executable (Linux)
chmod +x tools/glklevel9
```

---

## Running

### Option A: Local llama.cpp model (Streamlit UI)

```bash
source .venv/bin/activate
streamlit run app.py
```

In the sidebar: **Boot Engine** → set model path → **Start Auto-Run**.

### Option B: Cloud LLM (headless)

Set env vars in `.env` (or export directly), then run headlessly:

```bash
export LLM_PROVIDER=deepseek      # or "openai"
export LLM_API_KEY=sk-...
export LLM_MODEL=deepseek-chat    # or "gpt-4o-mini" etc.

python scripts/watch_run.py 100
```

`LLM_PROVIDER=local` (the default) uses llama.cpp instead.

### Option C: Local llama.cpp (headless)

```bash
python scripts/watch_run.py 50
```

The model path is read from the `LLM_MODEL_PATH` env var (or sidebar for Streamlit).

---

## Testing

### Lint

```bash
ruff check .
```

Config is in `pyproject.toml`. CI runs this on every PR.

### Unit tests (no LLM or game binary required)

```bash
pytest -m "not llm and not integration and not slow" -v
```

### JSON schema validation

```bash
pytest tests/test_json_schemas.py -v
```

Validates all `configs/`, `plans/`, and `tests/evals/` JSON files against schemas in `schemas/`.

### LLM extraction evals (requires model)

```bash
pytest tests/test_evals.py -m llm -v
# Override model path or pass threshold:
EVAL_MODEL_PATH=../models/my-model.gguf EVAL_THRESHOLD=0.8 pytest tests/test_evals.py -m llm -v
# Run a single eval case by ID:
pytest "tests/test_evals.py::test_extraction_case[taken_no_room_hallucination]" -v -m llm
```

### Anomaly detection pipeline

After a run, trigger the full detect → review → architect pipeline:

```bash
python scripts/watch_run.py 100 --detect --review --architect
```

- `--detect` runs `scripts/detect_anomalies.py` → `anomaly_report.json`
- `--review` appends LLM findings to the report (requires `ANTHROPIC_API_KEY`)
- `--architect` produces fix plans in `plans/` (deduped by type via `plans/index.json`)

### Generating a map from a saved run log

`watch_run.py` saves PNG maps alongside the log automatically. To regenerate:

```bash
python scripts/generate_map.py logs/watch_20260701_085624.json
python scripts/generate_map.py $(ls -t logs/watch_*.json | head -1)  # latest
```

### Growing the eval suite from a saved game log

```bash
# Preview what would be added (no writes)
python scripts/generate_evals.py --log game_log.json --out tests/evals/fixtures.json --dry-run

# Append new fixtures (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... python scripts/generate_evals.py --log game_log.json --out tests/evals/fixtures.json
```

### Writing tests for new scenarios

`extract_knowledge(response, action, llm)` is the boundary between the LLM and the agent:

```
game text  →  extract_knowledge()  →  agent state / determine_next_action()
              ↑ eval tests here         ↑ unit tests here
```

**Eval case** — add an entry to `tests/evals/fixtures.json`:

```json
{
    "id": "revealed_object_from_take",
    "action": "take welcome mat",
    "game_output": "You pick up the welcome mat. Underneath it you find a key!",
    "expected": {"objects": ["key"]},
    "must_not": {"added_to_inventory": ["key"]}
}
```

**Agent unit test** — add to `tests/test_agent.py`, mock `execute_game_command` + `extract_knowledge`, assert on state or `determine_next_action()` output. `stub_child` is a pytest fixture in `conftest.py`.

| What you're testing | File | What to mock |
|---|---|---|
| LLM extracts the right fields | `tests/evals/fixtures.json` | Nothing — uses real model |
| Agent state updates after a step | `tests/test_agent.py` → `process_agent_step` | `execute_game_command` + `extract_knowledge` |
| Next action is correct | `tests/test_agent.py` → `determine_next_action` | Just set up `state` |

---

## Project structure

```
app.py                      Streamlit entry point
agent.py                    Decision logic and state mutation
game_engine.py              glklevel9 subprocess interface via pexpect
llm.py                      LLM loading + knowledge extraction (local or cloud)
run_evaluator.py            Run classification, strategy merge, sidecar persistence
game_config.py              GameConfig singleton (loaded from configs/knight_orc.json)
env_utils.py                .env file loader with variable expansion
anomaly_detector.py         Deterministic detector helpers
log_analyzer.py             Log analysis helpers
ui.py                       pyvis graph rendering, log export

scripts/
  watch_run.py              Headless runner with --detect/--review/--architect pipeline
  detect_anomalies.py       Stage 1: deterministic anomaly detection
  llm_review.py             Stage 1b: LLM log review
  architect.py              Stage 2: anomaly → fix plan
  generate_evals.py         Generate eval fixtures from game logs
  generate_map.py           Regenerate map PNG from a run log
  agent_dev_loop.py         Dev orchestrator (detect → architect → dev → verify)

configs/
  knight_orc.json           Game config (verbs, failure patterns, nav commands)
  knight_orc_strategy.json  Cross-run persistence: futile edges + run history
  knight_orc_rooms.json     World graph sidecar
  knight_orc_items.json     Entity verb outcomes sidecar

schemas/                    JSON Schema Draft-7 files for all tracked JSON files
plans/                      Fix plan documents + index.json registry
tests/                      pytest suite; evals/fixtures.json for LLM evals
.github/workflows/ci.yml    GitHub Actions CI (lint + test on every PR)
pyproject.toml              Ruff lint config
gamefiles/                  Knight Orc ROM files (user-supplied)
tools/glklevel9             Level 9 interpreter binary (Linux x86-64)
```

---

## Known issues / roadmap

- Agent does not distinguish NPCs from objects — tries to `examine` or `read` characters
- NPC interaction (greet, ask for help, trade) not yet implemented
- `"you can see X"` not reliably parsed as room contents vs. inventory
- Wearable items and disguises (e.g. the hood) not handled

Full issue tracking: [`docs/bugs.md`](docs/bugs.md), [`docs/requirements.md`](docs/requirements.md), [`docs/features.md`](docs/features.md)
