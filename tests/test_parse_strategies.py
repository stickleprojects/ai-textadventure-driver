"""Shared, strategy-agnostic tests: known game responses, run through each
ParseStrategy + apply_parse_result, asserting the SAME resulting state
regardless of which strategy produced the raw ParseResult.

Two tiers:
- test_llm_json_mode_strategy / test_llm_tool_call_strategy — fast, free,
  run by default. Exercise a faked backend (no real LLM/API call) returning
  a known-good extraction for each case, in that strategy's own schema.
  This tests the strategy + shared apply_parse_result plumbing, not LLM
  extraction quality.
- test_llm_json_mode_strategy_real / test_llm_tool_call_strategy_real —
  marked `cloud` (excluded by default, run with `-m cloud`), make real
  network calls against every provider we have credentials for (loaded
  from .env via env_utils.load_env_file(), same as scripts/watch_run.py).
  Parametrized across whatever's actually configured — anthropic
  (ANTHROPIC_API_KEY), LLM_PROVIDER's own key (LLM_API_KEY, when not
  "local"), and the local llama.cpp model if its file exists — each
  skipped individually if its credentials/file aren't present, so this
  degrades gracefully in an environment with no .env at all.

KNOWN_CASES' response text is real, verbatim Knight Orc output sampled from
saved run logs (logs/*.json) — not hand-invented phrasing — so the same
cases that exercise the two LLM-backed strategies (via a faked backend
reporting the expected extraction) also genuinely exercise
DeterministicParseStrategy's regex patterns against the exact wording they
were derived from.
"""
import os
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from env_utils import load_env_file
load_env_file()  # populate os.environ from .env before any os.environ.get calls below

import world_graph
from parse_strategies import (
    DeterministicParseStrategy,
    LLMJsonModeParseStrategy,
    LLMToolCallParseStrategy,
    apply_parse_result,
)
from tests.conftest import make_state
from llm import load_anthropic_tool_llm, load_cloud_llm, load_llm, load_openai_tool_llm


@dataclass
class KnownCase:
    id: str
    action: str
    response: str
    room: str | None = None
    exits: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    npcs: list = field(default_factory=list)
    gained: list = field(default_factory=list)
    expect_gain: bool = True  # whether `gained` items should actually land in inventory


KNOWN_CASES = [
    # Real Knight Orc output, verbatim from logs/watch_20260708_*.json.
    KnownCase(
        id="room_and_exits",
        action="south",
        response="You go south and are in an alder ghostwood. Exits lead north, south, southwest, west and northwest.",
        room="alder ghostwood",
        exits=["north", "south", "southwest", "west", "northwest"],
    ),
    KnownCase(
        id="objects_and_npcs",
        action="look",
        response=(
            "You are on a trampled field, pock-marked by the feet of humans and their horses. "
            "An exit leads west. You can see Denzyl and a pile of garbage."
        ),
        room="trampled field",
        exits=["west"],
        objects=["pile of garbage"],
        npcs=["Denzyl"],
    ),
    KnownCase(
        id="take_success",
        action="take putty knife",
        response="You take the putty knife.",
        gained=["putty knife"],
    ),
    KnownCase(
        id="hard_failure_suppresses_gain",
        action="take castle",
        response="You can't take that.",
        gained=["castle"],
        expect_gain=False,
    ),
]


class _FakeToolAdapter:
    """Always returns one canned tool call carrying a known canonical result."""

    def __init__(self, canonical_result):
        self._result = canonical_result

    def start_turn(self, system_prompt, tool_schemas, user_message):
        return {
            "tool_calls": [{"name": "parse_game_response", "id": "c1", "input": dict(self._result)}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _legacy_extraction(case):
    """The shape LLMJsonModeParseStrategy (agent.extract_knowledge) returns."""
    return {
        "room": case.room,
        "exits": case.exits,
        "objects": case.objects,
        "npcs": case.npcs,
        "added_to_inventory": case.gained,
    }


def _canonical_extraction(case):
    """The shape LLMToolCallParseStrategy's parse_game_response tool call returns."""
    succeeded = case.expect_gain or not case.gained
    return {
        "action_result": {"succeeded": succeeded, "reason_if_failed": None},
        "room_quote": case.room,
        "exits": case.exits,
        "objects": case.objects,
        "npcs": case.npcs,
        "inventory_changes": [{"item": g, "change": "gained"} for g in case.gained],
    }


def _assert_case_applied(state, case):
    if case.room:
        assert state["current_room"] == case.room
        for direction in case.exits:
            # update_graph normalizes compound directions before naming the
            # placeholder node (southwest -> sw, etc. — world_graph.DIRECTION_NORMALIZE).
            normalized = world_graph.DIRECTION_NORMALIZE.get(direction, direction)
            assert state["world_graph"].has_node(f"Unknown ({normalized} from {case.room})")
    for obj in case.objects:
        assert obj in state["uninspected_objects"]
        assert obj in state["known_entities"]
    for npc in case.npcs:
        assert npc in state["known_npcs"]
    for item in case.gained:
        if case.expect_gain:
            assert item in state["inventory"]
        else:
            assert item not in state["inventory"]


@pytest.mark.parametrize("case", KNOWN_CASES, ids=[c.id for c in KNOWN_CASES])
def test_llm_json_mode_strategy(case):
    strategy = LLMJsonModeParseStrategy(llm_instance=object())
    with patch("agent.extract_knowledge", return_value=_legacy_extraction(case)):
        result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)


@pytest.mark.parametrize("case", KNOWN_CASES, ids=[c.id for c in KNOWN_CASES])
def test_llm_tool_call_strategy(case):
    strategy = LLMToolCallParseStrategy(_FakeToolAdapter(_canonical_extraction(case)))
    result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)


def test_llm_tool_call_strategy_verbose_room_quote_resolves_to_existing_node():
    # Bug 78 end-to-end: a (faked) LLM quotes a whole sentence instead of just
    # the room name. It's still grounded (bug 74's check passes — it IS a
    # literal substring), so apply_parse_result must not discard it, but
    # _resolve_room_name must canonicalize/merge it into the existing node
    # rather than fragmenting the graph. Not extending KNOWN_CASES across all
    # three strategies for this shape — DeterministicParseStrategy can't
    # produce a verbose narrator-prefixed room_quote by construction (see
    # parse_strategies/deterministic.py), so that would test an unrealistic
    # input for two of the three strategies.
    canonical = {
        "action_result": {"succeeded": True, "reason_if_failed": None},
        "room_quote": "You are in the dingy stable",
        "exits": ["north", "east"],
    }
    strategy = LLMToolCallParseStrategy(_FakeToolAdapter(canonical))
    response = "You go north. You are in the dingy stable. Exits lead north and east."
    state = make_state()
    state["world_graph"].add_node("dingy stable")  # already-known short-form node
    result = strategy.parse(response, "north")
    apply_parse_result(state, result, "north", response, previous_room=None)
    assert state["current_room"] == "dingy stable"
    assert not state["world_graph"].has_node("You are in the dingy stable")


@pytest.mark.parametrize("case", KNOWN_CASES, ids=[c.id for c in KNOWN_CASES])
def test_deterministic_strategy(case):
    strategy = DeterministicParseStrategy()
    result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)


# ---------------------------------------------------------------------------
# Real-backend tests — actually call configured cloud/local LLMs, no fakes.
# Excluded by default (pytest.ini: `-m "not cloud"`); run with `-m cloud`.
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LOCAL_MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")


def _real_tool_call_backends():
    """(id, tool_adapter) pairs for every tool-calling backend we have real
    credentials for — built once at collection time (constructs API clients,
    makes no network call yet; the actual request happens in strategy.parse())."""
    backends = []
    if ANTHROPIC_API_KEY:
        backends.append(("anthropic", load_anthropic_tool_llm(ANTHROPIC_MODEL, ANTHROPIC_API_KEY)))
    if LLM_PROVIDER != "local" and LLM_API_KEY:
        backends.append((LLM_PROVIDER, load_openai_tool_llm(LLM_PROVIDER, LLM_MODEL, LLM_API_KEY)))
    return [(bid, adapter) for bid, adapter in backends if adapter is not None]


def _real_json_mode_backends():
    """(id, llm_instance) pairs for every JSON-mode-capable backend we have
    real credentials/files for — same collection-time-safe construction."""
    backends = []
    if LLM_PROVIDER != "local" and LLM_API_KEY:
        backends.append((f"{LLM_PROVIDER}-json", load_cloud_llm(LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, json_mode=True)))
    if os.path.isfile(LOCAL_MODEL_PATH):
        backends.append(("local", load_llm(LOCAL_MODEL_PATH)))
    return [(bid, inst) for bid, inst in backends if inst is not None]


def _backend_params(backends, no_creds_reason):
    """Parametrize values for a list of (id, backend) pairs, or a single
    skipped placeholder if the list is empty — so the suite reports
    "skipped: no credentials" instead of silently collecting zero tests.
    Carries (id, backend) as the param value (not just the object) so tests
    can look a specific (case, backend id) combination up in
    _KNOWN_REAL_MODEL_FAILURES."""
    if not backends:
        return [pytest.param((None, None), id="none", marks=pytest.mark.skip(reason=no_creds_reason))]
    return [pytest.param((bid, backend), id=bid) for bid, backend in backends]


# Known, real-model quality gaps — not a bug in the strategy/apply_parse_result
# plumbing (that's what these tests exist to check), so these get xfail rather
# than a loosened assertion that would silently stop catching a regression in
# the plumbing itself. Empty for now — bug 78's entry was removed once fixed
# (agent.py's narrator-prefix stripping + fuzzy containment, plus prompt/schema
# tightening in both LLM-backed strategies); kept as reusable scaffolding for
# any future real-model quality gap.
_KNOWN_REAL_MODEL_FAILURES = {}


def _xfail_if_known_real_failure(case_id, backend_id):
    reason = _KNOWN_REAL_MODEL_FAILURES.get((case_id, backend_id))
    if reason:
        pytest.xfail(reason)


@pytest.mark.cloud
@pytest.mark.parametrize(
    "tool_call_backend",
    _backend_params(_real_tool_call_backends(), "no cloud tool-calling credentials configured (ANTHROPIC_API_KEY / LLM_API_KEY)"),
)
@pytest.mark.parametrize("case", KNOWN_CASES, ids=[c.id for c in KNOWN_CASES])
def test_llm_tool_call_strategy_real(case, tool_call_backend):
    backend_id, tool_adapter = tool_call_backend
    _xfail_if_known_real_failure(case.id, backend_id)
    strategy = LLMToolCallParseStrategy(tool_adapter)
    result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)


@pytest.mark.cloud
@pytest.mark.parametrize(
    "json_mode_backend",
    _backend_params(_real_json_mode_backends(), "no JSON-mode credentials/local model configured (LLM_API_KEY / EVAL_MODEL_PATH)"),
)
@pytest.mark.parametrize("case", KNOWN_CASES, ids=[c.id for c in KNOWN_CASES])
def test_llm_json_mode_strategy_real(case, json_mode_backend):
    backend_id, llm_instance = json_mode_backend
    _xfail_if_known_real_failure(case.id, backend_id)
    strategy = LLMJsonModeParseStrategy(llm_instance)
    result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)
