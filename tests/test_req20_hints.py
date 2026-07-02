"""Tests for requirement 20 — user-authored strategy hints."""
from game_config import GameConfig
from llm import extract_knowledge


class TestHintsConfig:
    def test_default_hints_empty(self):
        cfg = GameConfig()
        assert cfg.hints == []

    def test_load_hints_from_json(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text('{"hints": ["wear a disguise to avoid orc attacks", "put treasure in a sack"]}')
        cfg = GameConfig()
        cfg.load_from_file(str(cfg_file))
        assert cfg.hints == ["wear a disguise to avoid orc attacks", "put treasure in a sack"]

    def test_missing_hints_key_leaves_default(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text('{"candidate_verbs": ["take"]}')
        cfg = GameConfig()
        cfg.load_from_file(str(cfg_file))
        assert cfg.hints == []


class TestHintsInjectedIntoPrompt:
    def _capture_prompt(self, monkeypatch, hints):
        """Patch config.hints and capture the prompt passed to the LLM."""
        import game_config
        monkeypatch.setattr(game_config.config, "hints", hints)

        captured = {}

        def fake_llm(system_prompt, user_message):
            captured["system"] = system_prompt
            return {"choices": [{"text": "{}"}], "usage": {}}

        extract_knowledge("You are in the Forest.", "look", fake_llm)
        return captured.get("system", "")

    def test_no_hints_block_when_empty(self, monkeypatch):
        prompt = self._capture_prompt(monkeypatch, [])
        assert "Strategy hints" not in prompt

    def test_hints_appear_in_prompt(self, monkeypatch):
        prompt = self._capture_prompt(monkeypatch, ["wear a disguise to avoid attacks"])
        assert "Strategy hints" in prompt
        assert "wear a disguise to avoid attacks" in prompt

    def test_multiple_hints_all_appear(self, monkeypatch):
        hints = ["put treasure in a sack", "follow the troll before entering the lair"]
        prompt = self._capture_prompt(monkeypatch, hints)
        for h in hints:
            assert h in prompt
