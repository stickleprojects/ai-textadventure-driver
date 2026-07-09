"""Multi-step simulations driving real LLM extraction (not a mocked dict) over a
scripted sequence of engine responses. See tests/simulations/fixtures.py.

Distinct from tests/evals (single-step extraction scoring) and
tests/spec_scenarios (single-step decisions with mocked extraction) — these
tests exercise several consecutive process_agent_step() calls end-to-end
against the real, configured LLM, to verify agent-level behavior (state built
up across steps) rather than one extraction or one decision in isolation.
"""
import os
from unittest.mock import patch

import pytest

from agent import _detect_loop, process_agent_step
from env_utils import load_env_file
from llm import OPENAI_AVAILABLE, CloudLLMAdapter
from tests.conftest import make_state
from tests.simulations.fixtures import MAZE_WALK_RESPONSES
from world_graph import base_room_name, normalize_room

pytestmark = [pytest.mark.llm, pytest.mark.slow]

load_env_file()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


@pytest.fixture(scope="session")
def deepseek_llm():
    """Real DeepSeek-backed extractor for this project's configured cloud LLM.

    Instantiates CloudLLMAdapter directly rather than going through
    llm.load_cloud_llm(), which is wrapped in @st.cache_resource (a Streamlit
    caching decorator expecting a running app context) — the same reason
    tests/evals/test_evals.py builds its local-model fixture by calling
    llama_cpp.Llama() directly instead of going through llm.load_llm().
    """
    if not OPENAI_AVAILABLE:
        pytest.skip("openai package not installed (required for the DeepSeek adapter)")
    if not LLM_API_KEY:
        pytest.skip(
            "LLM_API_KEY not set; export LLM_PROVIDER=deepseek LLM_API_KEY=sk-... "
            "LLM_MODEL=deepseek-chat (see scripts/watch_run.py)"
        )
    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")
    return CloudLLMAdapter(client, LLM_MODEL, json_mode=True)


def test_maze_walk_keeps_repeated_room_name_distinct(deepseek_llm, stub_child):
    """Bug 45 regression, end-to-end with real extraction.

    A maze that reuses "Alder Clump" for physically distinct rooms (disjoint
    exits) must not collapse them into one graph node — the pre-fix behavior
    that made the agent believe an unexplored area was already fully mapped —
    and the run must never trip the loop detector.
    """
    state = make_state(current_room=None)

    with patch("agent.execute_game_command", side_effect=MAZE_WALK_RESPONSES):
        for _ in MAZE_WALK_RESPONSES:
            process_agent_step(state, stub_child, deepseek_llm)
            loop = _detect_loop(state["game_log"])
            assert loop is None, f"loop detected: {loop!r} after {len(state['game_log'])} steps"

    assert len(state["game_log"]) == len(MAZE_WALK_RESPONSES)

    alder_clump_nodes = [
        node for node in state["world_graph"].nodes
        if not node.startswith("Unknown")
        and normalize_room(base_room_name(node)) == "alder clump"
    ]
    # >=2 is the actual regression proof: bug 45 would collapse every "Alder
    # Clump" visit into exactly 1 node regardless of exits.
    assert len(alder_clump_nodes) >= 2, (
        f"expected at least 2 distinct 'Alder Clump' nodes, got: {alder_clump_nodes}"
    )
    # <=3 checks the fix doesn't over-fragment: step 4 revisits step 1's room
    # with the same exits and should merge back rather than spawning a 4th node.
    assert len(alder_clump_nodes) <= 3, (
        f"expected at most 3 distinct 'Alder Clump' nodes (revisit should merge), "
        f"got: {alder_clump_nodes}"
    )
