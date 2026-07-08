"""Shared, strategy-agnostic tests: known game responses, run through each
ParseStrategy + apply_parse_result, asserting the SAME resulting state
regardless of which strategy produced the raw ParseResult.

LLMJsonModeParseStrategy / LLMToolCallParseStrategy are exercised with a
faked backend (no real LLM/API call — that's what tests/test_evals.py's
`pytest.mark.llm` suite is for) returning a known-good extraction for each
case, in that strategy's own schema. This tests the strategy + shared
apply_parse_result plumbing, not LLM extraction quality.

DeterministicParseStrategy is a skeleton (parse() raises NotImplementedError,
see parse_strategies/deterministic.py) — its cases are xfail until a real
implementation exists. Once implemented, these same cases start exercising
it for free; pytest will report XPASS as a nudge to remove the xfail marks.
"""
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from parse_strategies import (
    DeterministicParseStrategy,
    LLMJsonModeParseStrategy,
    LLMToolCallParseStrategy,
    apply_parse_result,
)
from tests.conftest import make_state


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
    KnownCase(
        id="room_and_exits",
        action="look",
        response="You are in the dingy stable. Exits lead north and east.",
        room="dingy stable",
        exits=["north", "east"],
    ),
    KnownCase(
        id="objects_and_npcs",
        action="look",
        response="A putty knife lies here. A huge knight blocks the doorway.",
        objects=["putty knife"],
        npcs=["huge knight"],
    ),
    KnownCase(
        id="take_success",
        action="take putty knife",
        response="Taken.",
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
            assert state["world_graph"].has_node(f"Unknown ({direction} from {case.room})")
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


def _xfail_deterministic(case):
    return pytest.param(
        case,
        marks=pytest.mark.xfail(
            raises=NotImplementedError,
            reason="DeterministicParseStrategy not yet implemented — feature 62 step 5",
        ),
    )


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


@pytest.mark.parametrize("case", [_xfail_deterministic(c) for c in KNOWN_CASES], ids=[c.id for c in KNOWN_CASES])
def test_deterministic_strategy(case):
    strategy = DeterministicParseStrategy()
    result = strategy.parse(case.response, case.action)
    state = make_state()
    extracted, is_death, _ = apply_parse_result(state, result, case.action, case.response, previous_room=None)
    _assert_case_applied(state, case)
