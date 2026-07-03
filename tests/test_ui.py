"""
UI smoke tests using streamlit.testing.v1.AppTest (no browser needed).

Covers:
  - App boots without errors under local and cloud LLM_PROVIDER
  - Sidebar shows the correct fields for each provider mode
  - Session state is initialised with the expected keys
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.ui

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _make_mock_llm():
    llm = MagicMock()
    llm.return_value = {"choices": [{"text": "{}"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    return llm


def _make_mock_child():
    child = MagicMock()
    child.isalive.return_value = True
    return child


def _patches_local():
    return [
        patch("llm.LLAMA_AVAILABLE", True),
        patch("llm.load_llm", return_value=_make_mock_llm()),
        patch("game_engine.start_level9", return_value=(_make_mock_child(), "You are in a dark room.")),
        patch("agent.process_agent_step", return_value=None),
    ]


def _patches_cloud():
    return [
        patch("llm.OPENAI_AVAILABLE", True),
        patch("llm.load_cloud_llm", return_value=_make_mock_llm()),
        patch("game_engine.start_level9", return_value=(_make_mock_child(), "You are in a dark room.")),
        patch("agent.process_agent_step", return_value=None),
    ]


class TestLocalMode:
    def _run(self, env_overrides=None):
        from streamlit.testing.v1 import AppTest
        env = {"LLM_PROVIDER": "local", "LLM_API_KEY": "", "LLM_MODEL": "", "LLM_BASE_URL": ""}
        if env_overrides:
            env.update(env_overrides)
        with patch.dict(os.environ, env, clear=False):
            with patch("llm.LLAMA_AVAILABLE", True), patch("llm.load_llm", return_value=_make_mock_llm()):
                at = AppTest.from_file(APP_PATH, default_timeout=10)
                at.run()
        return at

    def test_no_exception_on_boot(self):
        at = self._run()
        assert not at.exception, f"App raised: {at.exception}"

    def test_session_state_keys(self):
        at = self._run()
        assert at.session_state["system_state"] is not None
        assert at.session_state["level9_process"] is None  # not booted yet

    def test_system_state_has_required_fields(self):
        at = self._run()
        state = at.session_state["system_state"]
        for field in ("current_room", "inventory", "spellbook", "world_graph",
                      "game_log", "is_running", "unresolved_anomalies"):
            assert field in state, f"Missing state field: {field}"

    def test_model_path_input_visible(self):
        at = self._run()
        labels = [w.label for w in at.text_input]
        assert any("model path" in label.lower() for label in labels), (
            f"Expected model path input in local mode, got: {labels}"
        )

    def test_api_key_input_absent(self):
        at = self._run()
        labels = [w.label for w in at.text_input]
        assert not any("api key" in label.lower() for label in labels), (
            f"API Key input should not appear in local mode, got: {labels}"
        )


class TestCloudMode:
    def _run(self, provider="deepseek"):
        from streamlit.testing.v1 import AppTest
        env = {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": "deepseek-chat",
            "LLM_API_KEY": "sk-test",
            "LLM_BASE_URL": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("llm.OPENAI_AVAILABLE", True), patch("llm.load_cloud_llm", return_value=_make_mock_llm()):
                at = AppTest.from_file(APP_PATH, default_timeout=10)
                at.run()
        return at

    def test_no_exception_on_boot(self):
        at = self._run()
        assert not at.exception, f"App raised: {at.exception}"

    def test_api_key_input_visible(self):
        at = self._run()
        labels = [w.label for w in at.text_input]
        assert any("api key" in label.lower() for label in labels), (
            f"Expected API Key input in cloud mode, got: {labels}"
        )

    def test_model_path_input_absent(self):
        at = self._run()
        labels = [w.label for w in at.text_input]
        assert not any("llama.cpp model path" in label.lower() for label in labels), (
            f"Local model path input should not appear in cloud mode, got: {labels}"
        )

    def test_session_state_keys(self):
        at = self._run()
        assert at.session_state["system_state"] is not None
        assert at.session_state["level9_process"] is None  # not booted yet
