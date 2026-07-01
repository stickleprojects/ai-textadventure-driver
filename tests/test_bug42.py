"""Tests for bug 42 — NoneType crash when LLM returns null for list fields."""
from llm import extract_knowledge

_LIST_FIELDS = ("exits", "objects", "npcs", "added_to_inventory",
                "learned_spells", "anomalies", "resolved_anomalies")


def _make_fake_llm(json_text):
    """Return a callable that mimics llama_cpp returning json_text."""
    def fake_llm(system_prompt, user_message):
        return {"choices": [{"text": json_text}], "usage": {}}
    return fake_llm


class TestNullListFieldsNormalized:
    def test_null_exits_becomes_empty_list(self):
        llm = _make_fake_llm('{"exits": null}')
        result = extract_knowledge("You are in the forest.", "look", llm)
        assert result.get("exits") == []

    def test_null_objects_becomes_empty_list(self):
        llm = _make_fake_llm('{"objects": null, "npcs": null}')
        result = extract_knowledge("You are in the forest.", "look", llm)
        assert result.get("objects") == []
        assert result.get("npcs") == []

    def test_null_added_to_inventory_becomes_empty_list(self):
        llm = _make_fake_llm('{"added_to_inventory": null}')
        result = extract_knowledge("Taken.", "take sword", llm)
        assert result.get("added_to_inventory") == []

    def test_null_anomalies_and_resolved_become_empty_list(self):
        llm = _make_fake_llm('{"anomalies": null, "resolved_anomalies": null}')
        result = extract_knowledge("Nothing happens.", "push door", llm)
        assert result.get("anomalies") == []
        assert result.get("resolved_anomalies") == []

    def test_valid_list_fields_unchanged(self):
        llm = _make_fake_llm('{"exits": ["north", "south"], "objects": ["sword"]}')
        result = extract_knowledge("You are in the hall.", "look", llm)
        assert result["exits"] == ["north", "south"]
        assert result["objects"] == ["sword"]

    def test_all_null_fields_do_not_raise(self):
        payload = (
            '{"room": null, "exits": null, "objects": null, "npcs": null,'
            ' "added_to_inventory": null, "learned_spells": null,'
            ' "anomalies": null, "resolved_anomalies": null}'
        )
        llm = _make_fake_llm(payload)
        result = extract_knowledge("You don't need to use the word rodents.", "take rodents", llm)
        for field in _LIST_FIELDS:
            assert result.get(field) == [], f"{field} should be [] not {result.get(field)}"
