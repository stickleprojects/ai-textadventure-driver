"""Spec-driven agent behavior tests — see docs/agent_behavior_spec.md and
tests/spec_scenarios/fixtures.py for the scenario format."""
from unittest.mock import patch

import pytest

from agent import determine_next_action, process_agent_step
from tests.conftest import make_state
from tests.spec_scenarios.fixtures import SCENARIOS


def _check_membership(state, expectations, should_contain):
    for key, values in expectations.items():
        container = state.get(key, [])
        for value in values:
            is_present = value in container
            if should_contain:
                assert is_present, f"expected {value!r} in state[{key!r}] ({container!r})"
            else:
                assert not is_present, f"expected {value!r} NOT in state[{key!r}] ({container!r})"


@pytest.mark.parametrize("case", SCENARIOS, ids=[c.id for c in SCENARIOS])
def test_spec_scenario(case, stub_child):
    if case.xfail:
        pytest.xfail(case.xfail)

    state = make_state(**case.initial_state)

    if case.mode == "decision":
        action, _reason = determine_next_action(state)
        if case.expect_action_startswith:
            assert action.startswith(case.expect_action_startswith), (
                f"{case.id}: action {action!r} does not start with "
                f"{case.expect_action_startswith!r}"
            )
        if case.expect_action_equals:
            assert action == case.expect_action_equals, (
                f"{case.id}: action {action!r} != {case.expect_action_equals!r}"
            )
        return

    # mode == "step"
    with patch("agent.execute_game_command", return_value=case.mock_response), \
         patch("agent.extract_knowledge", return_value=dict(case.mock_extracted)):
        process_agent_step(state, stub_child, None)

    for key, value in case.expect_state_equals.items():
        assert state.get(key) == value, (
            f"{case.id}: state[{key!r}] = {state.get(key)!r}, expected {value!r}"
        )
    _check_membership(state, case.expect_contains, should_contain=True)
    _check_membership(state, case.expect_absent, should_contain=False)
