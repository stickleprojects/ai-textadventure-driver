# Simulations

Multi-step, end-to-end regression tests that drive `process_agent_step()`
across several consecutive turns with **real LLM extraction** — not a mocked
dict. They exist to catch bugs that only show up once state has built up
across steps (e.g. two rooms with the same display name getting collapsed
into one graph node), which a single-step test can't see.

## How they differ from the other two test tiers

| Tier | Steps | Extraction | Fixture format | Tests |
|---|---|---|---|---|
| `tests/evals` | 1 | Real LLM | `fixtures.json` (schema-governed) | Extraction quality (Jaccard score vs. expected fields) |
| `tests/spec_scenarios` | 1 | Mocked (`extract_knowledge` patched) | `scenarios.json` (schema-governed) | Decision logic / state update in isolation |
| `tests/simulations` (this folder) | N (several `process_agent_step` calls in a loop) | Real LLM | Plain Python in `fixtures.py` | Agent-level behavior emerging across a sequence of turns |

Only `game_engine.execute_game_command` is mocked here (via
`side_effect=<list of scripted responses>`), so each step still runs the real
`extract_knowledge()` call against a configured cloud LLM. Everything else —
graph updates, room-identity resolution, loop detection — runs unmodified.

## How a simulation works mechanically

1. `fixtures.py` defines a plain Python list of strings, one scripted game
   engine response per step (e.g. `MAZE_WALK_RESPONSES`). Each string is
   phrased in the same plain "You are in the `<Room>`. Exits: `<a>`, `<b>`."
   style already proven reliable in `tests/evals/fixtures.json` — the point
   is to test the *agent's* room-identity/state logic, not the LLM's ability
   to parse unusual phrasing (that's `tests/evals`' job).
2. `test_simulations.py` patches `agent.execute_game_command` with
   `side_effect=<the list>`, then calls `process_agent_step(state, stub_child,
   llm)` once per scripted response, in order. Each call still performs a real
   LLM extraction call.
3. Assertions run both mid-loop (e.g. `_detect_loop` must never fire) and
   after the full sequence completes, inspecting the resulting `state` (e.g.
   `state["world_graph"].nodes`, `len(state["game_log"])`).

Fixtures are plain Python rather than JSON+schema like the other two tiers
because each one is a fixed, one-off regression scenario for a specific bug —
there's no growing library here that needs schema governance.

## Running

Simulations are real LLM calls, so they're marked `@pytest.mark.llm` and
`@pytest.mark.slow` and are excluded from the default CI test run.

```bash
# needs LLM_API_KEY in .env (or exported), same as tests/evals
export LLM_PROVIDER=deepseek LLM_MODEL=deepseek-chat LLM_API_KEY=sk-...
pytest tests/test_simulations.py -v -m llm
```

Each test session builds one `CloudLLMAdapter` (session-scoped fixture) and
reuses it across all simulation tests. Tests skip (not fail) if the `openai`
package isn't installed or `LLM_API_KEY` isn't set.

## Adding a new simulation

1. Add a new list of scripted responses to `fixtures.py`, one string per
   step, with a comment explaining what physical situation each step
   represents and why the sequence proves the fix.
2. Add a test function to `test_simulations.py` that patches
   `execute_game_command` with that list, loops `process_agent_step` over it,
   and asserts on the resulting state. Reuse the `deepseek_llm` and
   `stub_child` fixtures already defined there.
3. Keep assertions about *end state* (graph nodes, log length, loop
   detection) rather than the LLM's exact wording — the LLM's phrasing is
   allowed to vary between runs; the agent's state handling should not.
