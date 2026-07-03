"""Validate all tracked JSON files against their schemas."""
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).parent.parent
SCHEMAS = ROOT / "schemas"
CONFIGS = ROOT / "configs"
PLANS = ROOT / "plans"
EVALS = ROOT / "tests" / "evals"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _schema(name: str):
    return _load(SCHEMAS / name)


# ── config files ──────────────────────────────────────────────────────────────

def test_game_config_valid():
    jsonschema.validate(_load(CONFIGS / "knight_orc.json"), _schema("game_config.schema.json"))


def test_strategy_valid():
    jsonschema.validate(_load(CONFIGS / "knight_orc_strategy.json"), _schema("strategy.schema.json"))


def test_rooms_valid():
    jsonschema.validate(_load(CONFIGS / "knight_orc_rooms.json"), _schema("rooms.schema.json"))


def test_items_valid():
    jsonschema.validate(_load(CONFIGS / "knight_orc_items.json"), _schema("items.schema.json"))


# ── eval fixtures ─────────────────────────────────────────────────────────────

def test_eval_fixtures_valid():
    jsonschema.validate(_load(EVALS / "fixtures.json"), _schema("eval_fixture.schema.json"))


def test_eval_fixture_ids_unique():
    fixtures = _load(EVALS / "fixtures.json")
    ids = [f["id"] for f in fixtures]
    assert len(ids) == len(set(ids)), f"Duplicate fixture IDs: {[i for i in ids if ids.count(i) > 1]}"


# ── plan files ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plan_file", sorted(PLANS.glob("P*.json")))
def test_plan_files_valid(plan_file):
    jsonschema.validate(_load(plan_file), _schema("plan.schema.json"))


def test_plan_ids_match_filenames():
    for plan_file in PLANS.glob("P*.json"):
        assert _load(plan_file)["plan_id"] == plan_file.stem, f"plan_id mismatch in {plan_file}"


# ── all JSON files parse ──────────────────────────────────────────────────────

_MERGE_SUFFIXES = ("_BACKUP_", "_BASE_", "_LOCAL_", "_REMOTE_")

_ALL_JSON = [
    f
    for f in (
        list(CONFIGS.glob("*.json"))
        + list(PLANS.glob("*.json"))
        + list(EVALS.glob("*.json"))
    )
    if not any(s in f.name for s in _MERGE_SUFFIXES)
]


@pytest.mark.parametrize("json_file", _ALL_JSON, ids=lambda f: f.name)
def test_all_json_parseable(json_file):
    with open(json_file) as fh:
        json.load(fh)
