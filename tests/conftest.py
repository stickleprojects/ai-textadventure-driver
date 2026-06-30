from unittest.mock import MagicMock

import networkx as nx
import pytest

from game_config import config


def make_state(**overrides):
    state = {
        "current_room": "Unknown Location",
        "inventory": [],
        "spellbook": [],
        "known_entities": {},
        "known_npcs": {},
        "unresolved_anomalies": {},
        "world_graph": nx.DiGraph(),
        "uninspected_objects": [],
        "current_inspection": {"target": None, "sequence": config.inspection_sequence, "step_index": 0},
        "active_goal": None,
        "game_log": [],
        "is_running": False,
        "current_score": None,
        "max_score": None,
        "futile_edges": set(),
    }
    state.update(overrides)
    return state


@pytest.fixture
def stub_child():
    child = MagicMock()
    child.isalive.return_value = True
    child.before = ""
    return child
