"""
Eval case loader for LLM extraction tests.

Cases are defined in fixtures.json alongside this file.  The JSON schema
for each case is:

  id           str            Unique snake_case identifier used as the pytest
                              test ID.  Required.

  action       str            The command the agent sent to the game engine.
                              The LLM receives this as context when extracting
                              knowledge from game_output.  Required.

  game_output  str            The raw text the game engine returned.
                              This is the text the LLM must interpret.  Required.

  expected     object         Fields and values that must appear in the
                              extraction result.  Each field is scored with
                              Jaccard similarity against the actual extraction;
                              all field scores are averaged and must meet
                              EVAL_THRESHOLD (default 0.7).  Omit or use {}
                              when the test is purely about absence.

  absent       list[str]      Field names whose value must be null, empty, or
                              missing entirely in the result.  Use this to
                              guard against hallucination (e.g. "room" must
                              not appear after a terse "Taken." response).

  must_not     object         Fields whose values must NOT appear in the
                              result.  Use this when the field may legitimately
                              contain other values, but specific ones are
                              forbidden (e.g. objects must not contain "horse"
                              when horse should be classified as an NPC).

  xfail        str (optional) If present, the test is marked xfail with this
                              string as the reason.  Use for known LLM
                              weaknesses you want to track without blocking CI.

Example case:

  {
    "id": "revealed_object_from_take",
    "action": "take welcome mat",
    "game_output": "You pick up the welcome mat. Underneath it you find a key!",
    "expected": { "objects": ["key"], "added_to_inventory": ["welcome mat"] },
    "must_not": { "added_to_inventory": ["key"] }
  }
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """One extraction eval scenario."""

    id: str
    action: str
    game_output: str
    expected: dict = field(default_factory=dict)
    absent: list = field(default_factory=list)
    must_not: dict = field(default_factory=dict)
    xfail: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        return cls(
            id=data["id"],
            action=data["action"],
            game_output=data["game_output"],
            expected=data.get("expected", {}),
            absent=data.get("absent", []),
            must_not=data.get("must_not", {}),
            xfail=data.get("xfail"),
        )


def _load() -> list[EvalCase]:
    path = Path(__file__).with_suffix(".json")
    raw = json.loads(path.read_text())
    return [EvalCase.from_dict(entry) for entry in raw]


EVAL_CASES: list[EvalCase] = _load()
