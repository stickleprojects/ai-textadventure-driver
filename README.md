# Knight Orc Autonomous OS

A local, fully-offline AI agent that plays the 1987 Level 9 text adventure **Knight Orc** autonomously. It uses a local llama.cpp model to understand the game's natural language output, builds a spatial map of the world as it explores, tracks items and spells, and resolves blocked paths once it has the right tool or magic.

Built with Streamlit so you can watch it think in real time.

![Dashboard showing spatial graph, I-O terminal, and state schema]()

---

## How it works

The agent runs a continuous decide → act → extract → update loop:

1. **Game engine** — `glklevel9` runs as a subprocess; the agent sends text commands and reads the response.
2. **Knowledge extraction** — raw game text is sent to a local LLM which returns structured JSON (room, exits, objects seen, items picked up, spells learned, anomalies encountered).
3. **Decision logic** — a priority queue determines the next command:
   - Pursue an active goal (navigate to a blocked object, then apply the known solution)
   - Resolve a known anomaly if inventory/spellbook now allows it
   - Inspect the next queued object (`take → examine → read → look inside`)
   - Explore an unknown exit
   - Fallback: `look`
4. **World graph** — rooms are nodes, movement directions are edges; shortest-path navigation is used when backtracking to resolve anomalies.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.14 (see `.env/`) |
| `glklevel9` binary | Included at `tools/glklevel9` (Linux x86-64). Other platforms need to compile from source: [glk-implementations](https://github.com/DavidGriffith/glk-implementations) or equivalent dumb-glk Level 9 build |
| Knight Orc game files | Level 9 released their games as freeware. Obtain `GAMEDAT1.DAT`, `GAMEDAT2.DAT`, `GAMEDAT3.DAT` from the [Level 9 Memorial](http://www.if-legends.org/~l9memorial/html/home.html) or Abandonware archives and place them in `gamefiles/knight-orc/` |
| llama.cpp model | A GGUF-format instruction model. The default config expects `../models/Phi-3.5-mini-instruct-Q3_K_M.gguf`. Download from [Hugging Face](https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF) or substitute any compatible GGUF |

> **The local LLM is required.** Without it, knowledge extraction returns empty results and the agent has nothing to act on.

---

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd <repo-dir>

# Create and activate a virtual environment
python -m venv .env
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place game files
mkdir -p gamefiles/knight-orc
cp /path/to/your/GAMEDAT*.DAT gamefiles/knight-orc/

# Make the interpreter executable (Linux)
chmod +x tools/glklevel9
```
---

## Running

### 1. Start llama.cpp server

The app loads the model via `llama-cpp-python` directly (no separate server needed), but you can also run `llama-server` standalone to test or inspect model output:

```bash
llama-server \
  -m ../models/Phi-3.5-mini-instruct-Q3_K_M.gguf \
  --ctx-size 2048 \
  -t 4 \
  --no-mmap \
  --port 8080
```

Settings tuned for i7-8565U (4 physical cores, 16 GB RAM, CPU-only):

| Flag | Value | Reason |
|---|---|---|
| `--ctx-size` | `2048` | Matches `n_ctx` in `llm.py`; fits comfortably in 16 GB |
| `-t` | `4` | Physical core count; avoids hyperthreading overhead on inference |
| `--no-mmap` | — | More stable memory behaviour on Linux without a dedicated GPU |

> If you only intend to run the Streamlit app, skip this step — the model is loaded in-process automatically once you set the model path in the sidebar.

### 2. Run the app

```bash
source .env/bin/activate
streamlit run app.py
```

The app opens in your browser. In the sidebar:

1. **Boot Engine** — starts the `glklevel9` subprocess and reads the opening room.
2. Set **Local llama.cpp Model Path** to your GGUF file.
3. Use **Step Once** to advance one agent cycle, or **Start Auto-Run** to let it run continuously. Adjust **Step Delay** to control pace.
4. **Export Run Log** downloads the full action history as Markdown.

---

## Testing

### Unit tests (no LLM or game binary required)

```bash
source .env/bin/activate
pytest tests/test_game_engine.py tests/test_agent.py -v
```

### LLM extraction evals (requires local model)

```bash
pytest tests/test_evals.py -m llm -v
# Override model path or pass threshold:
EVAL_MODEL_PATH=../models/my-model.gguf EVAL_THRESHOLD=0.8 pytest tests/test_evals.py -m llm -v
# Run a single eval case by ID:
pytest "tests/test_evals.py::test_extraction_case[handles_you_dont_need_to_use_the_word]" -v -m llm
```

### Full suite minus LLM evals

```bash
pytest -m "not llm" -v
```

### Headless watch run (real ROM + real model, no Streamlit)

Runs N agent steps and exits non-zero if a loop or timeout is detected:

```bash
python scripts/watch_run.py 50
```

Wire to the `/loop` skill to automate manual watching:

```
/loop 30m run python scripts/watch_run.py 50 and summarize any findings
```

### Generating a map from a saved run log

`watch_run.py` saves a PNG map alongside the log automatically. To regenerate it (or produce one from an older log):

```bash
# Writes logs/watch_<run_id>_map.png by default
python scripts/generate_map.py logs/watch_20260701_085624.json

# Override the output path
python scripts/generate_map.py logs/watch_20260701_085624.json --out map.png
```

To use the latest log without typing the filename:

```bash
python scripts/generate_map.py $(ls -t logs/watch_*.json | head -1)
```

### Growing the eval suite from a saved game log

Export a run log via the **Export Run Log (JSON)** button in the sidebar, then:

```bash
# Preview what would be added (no writes)
python scripts/generate_evals.py --log game_log.json --out tests/evals/fixtures.py --dry-run

# Append new fixtures (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... python scripts/generate_evals.py --log game_log.json --out tests/evals/fixtures.py
```

### Writing tests for new scenarios

Every game behaviour worth testing falls into one of two categories, each needing a different kind of test. You almost always need both.

#### The boundary

`extract_knowledge(response, action, llm)` is the boundary between the LLM and the agent. Tests on the left side verify the LLM extracts the right fields from game text. Tests on the right side verify the agent reacts correctly to whatever `extract_knowledge` returns — without involving the LLM at all.

```
game text  →  extract_knowledge()  →  agent state / determine_next_action()
              ↑ eval tests here         ↑ unit tests here
```

#### Test 1: Eval case (does the LLM extract X?)

Add an entry to `tests/evals/fixtures.py`. Each case needs:

- `"id"` — unique snake_case string, used to run the test by name
- `"action"` — the command the agent sent (LLM uses it as context)
- `"game_output"` — the raw game response text
- `"expected"` — fields and values that must be present (scored with Jaccard similarity, threshold 0.7)
- `"absent"` — field names that must be null/missing (e.g. `room` should not appear after a terse "Taken.")
- `"must_not"` — field values that must not appear (e.g. key must not be in `added_to_inventory` if it was only revealed, not taken)

Example — a take action reveals a hidden object:
```python
{
    "id": "revealed_object_from_take",
    "action": "take welcome mat",
    "game_output": "You pick up the welcome mat. Underneath it you find a key!",
    "expected": {"objects": ["key"]},
    "must_not": {"added_to_inventory": ["key"]},  # revealed ≠ taken yet
},
```

Run it:
```bash
pytest tests/test_evals.py -m llm -k revealed_object -v
```

> **You don't need to know the exact game wording upfront.** Write the eval with expected wording, run the agent on the real ROM, then update `"game_output"` with what the game actually produces.

#### Test 2: Agent unit test (does state update correctly?)

Add to `tests/test_agent.py`. These are fast and deterministic — mock the game and the LLM, assert on state.

The three things you set up:
1. **`state`** via `make_state(...)` from `tests/conftest.py` — only include keys relevant to the scenario
2. **`patch("agent.execute_game_command", return_value="...")`** — the raw game response
3. **`patch("agent.extract_knowledge", return_value={...})`** — what the LLM would return

Then assert on state fields (`uninspected_objects`, `inventory`, `known_entities`, etc.) or on the next `determine_next_action()` call.

Example — key revealed by taking mat is queued, then taken next:
```python
class TestRevealedObject:
    def test_key_queued_when_revealed_by_taking_mat(self, stub_child):
        state = make_state(
            current_room="Doorstep",
            current_inspection={
                "target": "welcome mat",
                "sequence": ["take"],
                "step_index": 0,
            },
            known_entities={
                "welcome mat": {"status": "discovered", "location": "Doorstep", "verb_outcomes": {}},
            },
        )
        with patch("agent.execute_game_command",
                   return_value="You pick up the welcome mat. Underneath it you find a key!"), \
             patch("agent.extract_knowledge",
                   return_value={"added_to_inventory": ["welcome mat"], "objects": ["key"]}):
            process_agent_step(state, stub_child, None)

        assert "key" in state["uninspected_objects"]

    def test_key_taken_on_next_step(self, stub_child):
        state = make_state(
            current_room="Doorstep",
            uninspected_objects=["key"],
            known_entities={
                "key": {"status": "discovered", "location": "Doorstep", "verb_outcomes": {}},
            },
        )
        action, _ = determine_next_action(state)
        assert action == "take key"
```

Run it:
```bash
pytest tests/test_agent.py -k TestRevealedObject -v
```

#### Quick reference

| What you're testing | File | What to mock |
|---|---|---|
| LLM extracts the right fields from game text | `tests/evals/fixtures.py` | Nothing — uses real model |
| Agent state updates correctly after a step | `tests/test_agent.py`, call `process_agent_step` | `execute_game_command` + `extract_knowledge` |
| `determine_next_action` returns the right command | `tests/test_agent.py`, call `determine_next_action` | Just set up `state` — no mocks needed |

`stub_child` (the mock game process) is a pytest fixture defined in `tests/conftest.py` and available in all agent tests automatically.

---

## Project structure

```
app.py            Streamlit entry point — session state, layout, auto-run loop
agent.py          Decision logic and state mutation (no Streamlit dependency)
game_engine.py    glklevel9 subprocess interface via pexpect
llm.py            Local model loading and knowledge extraction
ui.py             Graph rendering and log export
gamefiles/        Knight Orc ROM files (user-supplied)
tools/glklevel9   Level 9 interpreter binary (Linux x86-64)
requirements.txt
```

---

## Known issues / roadmap

- Agent does not distinguish NPCs from objects — it tries to `examine` or `read` characters
- NPC interaction (greet, ask for help, trade) is not yet implemented
- Room name labels overflow their graph nodes — zoom/scaling not yet available
- `"you can see X"` is not reliably parsed as room contents rather than inventory
- Wearable items and disguises (e.g. the hood) are not handled
- `"You can't do that"` responses are not fed back to skip the current inspection step
