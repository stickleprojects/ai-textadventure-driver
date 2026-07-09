"""
UI smoke tests using streamlit.testing.v1.AppTest (no browser needed).

Covers:
  - App boots without errors under local and cloud LLM_PROVIDER
  - Sidebar shows the correct fields for each provider mode
  - Session state is initialised with the expected keys
"""
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
        patch("llm.load_openai_tool_llm", return_value=_make_mock_llm()),
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
            # app.py calls env_utils.load_env_file() at its own module scope,
            # re-run fresh on every AppTest .run(). load_env_file() now
            # deliberately overwrites os.environ whenever a real .env value
            # differs (so a keyring: reference can refresh) — on a machine
            # with a real .env configured, that would silently clobber the
            # env dict set above back to whatever .env actually contains.
            # No-op it here so this test's env is authoritative regardless
            # of what .env exists on the machine running it.
            with patch("env_utils.load_env_file", return_value=set()), \
                 patch("llm.LLAMA_AVAILABLE", True), patch("llm.load_llm", return_value=_make_mock_llm()):
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
            # See TestLocalMode._run's comment: no-op the real .env load so
            # this test's env (including the fake "sk-test" key) can't be
            # silently overwritten by a real .env on the machine running it.
            with patch("env_utils.load_env_file", return_value=set()), \
                 patch("llm.OPENAI_AVAILABLE", True), patch("llm.load_openai_tool_llm", return_value=_make_mock_llm()):
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


class TestToolCallingMode:
    """Verify that a non-local LLM_PROVIDER loads tool adapters (not CloudLLMAdapter)."""

    def _run(self, provider="anthropic"):
        from streamlit.testing.v1 import AppTest
        env = {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": "claude-3-haiku-20240307",
            "LLM_API_KEY": "sk-ant-test",
            "LLM_BASE_URL": "",
        }
        mock_tool_adapter = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            with patch("env_utils.load_env_file", return_value=set()), \
                 patch("llm.ANTHROPIC_AVAILABLE", True), \
                 patch("llm.load_anthropic_tool_llm", return_value=mock_tool_adapter) as mock_load:
                at = AppTest.from_file(APP_PATH, default_timeout=10)
                at.run()
        return at, mock_load

    def test_no_exception_on_boot(self):
        at, _ = self._run()
        assert not at.exception, f"App raised: {at.exception}"

    def test_tool_adapter_loaded_not_cloud_llm(self):
        """load_anthropic_tool_llm should be called; load_cloud_llm should not."""
        env = {
            "LLM_PROVIDER": "anthropic",
            "LLM_MODEL": "claude-3-haiku-20240307",
            "LLM_API_KEY": "sk-ant-test",
            "LLM_BASE_URL": "",
        }
        mock_tool_adapter = MagicMock()
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, env, clear=False):
            with patch("env_utils.load_env_file", return_value=set()), \
                 patch("llm.ANTHROPIC_AVAILABLE", True), \
                 patch("llm.load_anthropic_tool_llm", return_value=mock_tool_adapter) as mock_anthropic, \
                 patch("llm.load_cloud_llm") as mock_cloud:
                at = AppTest.from_file(APP_PATH, default_timeout=10)
                at.run()
        assert not at.exception, f"App raised: {at.exception}"
        mock_anthropic.assert_called_once()
        mock_cloud.assert_not_called()

    def test_openai_provider_loads_openai_tool_adapter(self):
        """For a non-anthropic provider, load_openai_tool_llm should be called."""
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_MODEL": "deepseek-chat",
            "LLM_API_KEY": "sk-ds-test",
            "LLM_BASE_URL": "",
        }
        mock_tool_adapter = MagicMock()
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, env, clear=False):
            with patch("env_utils.load_env_file", return_value=set()), \
                 patch("llm.OPENAI_AVAILABLE", True), \
                 patch("llm.load_openai_tool_llm", return_value=mock_tool_adapter) as mock_openai, \
                 patch("llm.load_cloud_llm") as mock_cloud:
                at = AppTest.from_file(APP_PATH, default_timeout=10)
                at.run()
        assert not at.exception, f"App raised: {at.exception}"
        mock_openai.assert_called_once()
        mock_cloud.assert_not_called()

    def test_run_id_initialised_in_session_state(self):
        at, _ = self._run()
        assert "run_id" in at.session_state

    def test_api_key_input_visible(self):
        at, _ = self._run()
        labels = [w.label for w in at.text_input]
        assert any("api key" in label.lower() for label in labels), (
            f"Expected API Key input in tool-calling mode, got: {labels}"
        )

    def test_local_model_path_absent(self):
        at, _ = self._run()
        labels = [w.label for w in at.text_input]
        assert not any("llama.cpp model path" in label.lower() for label in labels), (
            f"Local model path should not appear in tool-calling mode, got: {labels}"
        )
