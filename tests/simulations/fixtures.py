"""Scripted engine-response sequences for multi-step, real-LLM simulations.

Unlike tests/evals/fixtures.json (single-step extraction cases) and
tests/spec_scenarios/scenarios.json (single-step agent decisions), these
fixtures drive several consecutive process_agent_step() calls with real LLM
extraction, so plain Python is used instead of a JSON+schema fixture format —
there's no need for a growing schema-validated library here, just a fixed
regression scenario per bug.

Response phrasing deliberately mirrors the plain "You are in the <Room>.
Exits: <a>, <b>." style already proven reliable in tests/evals/fixtures.json,
so the assertions are testing the room-identity fix (agent.py's
_resolve_room_name), not the LLM's ability to parse unusual phrasing —
that's tests/evals' job.
"""

# Bug 45: a maze reusing the display name "Alder Clump" for three physically
# distinct rooms. Steps 1-3 walk into three genuinely different rooms sharing
# the name; step 4 revisits room 1 (same exits) to confirm it merges back
# instead of fragmenting further; step 5 moves on to an unrelated room to
# confirm ordinary traversal still works after the maze.
MAZE_WALK_RESPONSES = [
    "You are in the Alder Clump. Exits: north, east.",
    "You are in the Alder Clump. Exits: south, west.",
    "You are in the Alder Clump. Exits: north, south, east.",
    "You are in the Alder Clump. Exits: north, east.",
    "You are in the Mossy Clearing. Exits: south.",
]
