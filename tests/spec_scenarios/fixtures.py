"""
Spec-driven agent behavior scenario loader.

Scenarios are defined in scenarios.json alongside this file.  Each entry
covers one section of docs/agent_behavior_spec.md (see `spec_ref`) and is
run one of two ways:

  mode="decision"  Build state via tests.conftest.make_state(**initial_state),
                    call agent.determine_next_action(state), and assert on
                    the returned action (expect_action_startswith /
                    expect_action_equals). Use for priority-stack decisions
                    that don't require a game response.

  mode="step"       Same state, but patch agent.execute_game_command to
                    return mock_response and agent.extract_knowledge to
                    return mock_extracted, then call
                    agent.process_agent_step(state, stub_child, None) and
                    assert on the resulting state (expect_state_equals /
                    expect_contains / expect_absent).

  id                str            Unique snake_case identifier used as the
                                   pytest test ID.  Required.

  spec_ref          str            kebab-case slug of the docs/agent_behavior_spec.md
                                   ### heading this scenario covers.  Required.

  requirement_ref   int | null     docs/requirements/<n>.md this scenario is
                                   blocked on, if any behavior gap remains.

  mode              str            "decision" or "step".  Required.

  initial_state     object         Overrides merged onto
                                   tests.conftest.make_state().  Required.

  mock_response     str            mode=step only: raw text returned by the
                                   (mocked) game engine.

  mock_extracted    object         mode=step only: dict returned by the
                                   (mocked) LLM extraction.

  expect_action_startswith / expect_action_equals   str
                    mode=decision only: assertion on the chosen action.

  expect_state_equals / expect_contains / expect_absent   object
                    mode=step only: assertions on the resulting state.
                    expect_contains / expect_absent check membership in a
                    list-or-dict-valued state field; expect_state_equals
                    checks equality.

  xfail             str (optional) If present, the scenario is marked
                                   pytest.xfail with this string as the
                                   reason instead of being run — use for
                                   spec behavior that isn't implemented yet.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScenarioCase:
    """One spec-driven agent behavior scenario."""

    id: str
    spec_ref: str
    mode: str
    initial_state: dict
    requirement_ref: int | None = None
    mock_response: str = ""
    mock_extracted: dict = field(default_factory=dict)
    expect_action_startswith: str | None = None
    expect_action_equals: str | None = None
    expect_state_equals: dict = field(default_factory=dict)
    expect_contains: dict = field(default_factory=dict)
    expect_absent: dict = field(default_factory=dict)
    xfail: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioCase":
        return cls(
            id=data["id"],
            spec_ref=data["spec_ref"],
            mode=data["mode"],
            initial_state=data.get("initial_state", {}),
            requirement_ref=data.get("requirement_ref"),
            mock_response=data.get("mock_response", ""),
            mock_extracted=data.get("mock_extracted", {}),
            expect_action_startswith=data.get("expect_action_startswith"),
            expect_action_equals=data.get("expect_action_equals"),
            expect_state_equals=data.get("expect_state_equals", {}),
            expect_contains=data.get("expect_contains", {}),
            expect_absent=data.get("expect_absent", {}),
            xfail=data.get("xfail"),
        )


def _load() -> list[ScenarioCase]:
    path = Path(__file__).parent / "scenarios.json"
    raw = json.loads(path.read_text())
    return [ScenarioCase.from_dict(entry) for entry in raw]


SCENARIOS: list[ScenarioCase] = _load()
