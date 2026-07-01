"""Tests for env_utils.load_env_file."""
import os
import pytest
from env_utils import load_env_file


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove test keys from environment before and after each test."""
    for key in ("TEST_KEY", "TEST_INDIRECT", "TEST_QUOTED", "TEST_COMMENT",
                "TEST_EXISTING", "REAL_SECRET"):
        monkeypatch.delenv(key, raising=False)


def _write_env(tmp_path, content):
    p = tmp_path / ".env"
    p.write_text(content)
    return str(p)


class TestBasicLoading:
    def test_simple_key_value(self, tmp_path):
        path = _write_env(tmp_path, "TEST_KEY=hello\n")
        load_env_file(path)
        assert os.environ["TEST_KEY"] == "hello"

    def test_double_quoted_value(self, tmp_path):
        path = _write_env(tmp_path, 'TEST_QUOTED="hello world"\n')
        load_env_file(path)
        assert os.environ["TEST_QUOTED"] == "hello world"

    def test_single_quoted_value(self, tmp_path):
        path = _write_env(tmp_path, "TEST_QUOTED='hello world'\n")
        load_env_file(path)
        assert os.environ["TEST_QUOTED"] == "hello world"

    def test_comment_lines_ignored(self, tmp_path):
        path = _write_env(tmp_path, "# this is a comment\nTEST_KEY=value\n")
        load_env_file(path)
        assert os.environ["TEST_KEY"] == "value"

    def test_inline_comment_stripped(self, tmp_path):
        path = _write_env(tmp_path, "TEST_COMMENT=value  # inline comment\n")
        load_env_file(path)
        assert os.environ["TEST_COMMENT"] == "value"

    def test_missing_file_does_nothing(self, tmp_path):
        load_env_file(str(tmp_path / "nonexistent.env"))
        assert "TEST_KEY" not in os.environ

    def test_returns_set_keys(self, tmp_path):
        path = _write_env(tmp_path, "TEST_KEY=a\nTEST_COMMENT=b\n")
        keys = load_env_file(path)
        assert "TEST_KEY" in keys
        assert "TEST_COMMENT" in keys


class TestIndirectExpansion:
    def test_dollar_var_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REAL_SECRET", "actual_value")
        path = _write_env(tmp_path, "TEST_INDIRECT=$REAL_SECRET\n")
        load_env_file(path)
        assert os.environ["TEST_INDIRECT"] == "actual_value"

    def test_brace_var_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REAL_SECRET", "actual_value")
        path = _write_env(tmp_path, "TEST_INDIRECT=${REAL_SECRET}\n")
        load_env_file(path)
        assert os.environ["TEST_INDIRECT"] == "actual_value"

    def test_unresolvable_ref_left_as_is(self, tmp_path):
        path = _write_env(tmp_path, "TEST_INDIRECT=$UNDEFINED_VAR_XYZ\n")
        load_env_file(path)
        assert os.environ["TEST_INDIRECT"] == "$UNDEFINED_VAR_XYZ"

    def test_api_key_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-realkey")
        path = _write_env(tmp_path, "LLM_API_KEY=$DEEPSEEK_API_KEY\n")
        load_env_file(path)
        assert os.environ["LLM_API_KEY"] == "sk-realkey"
        monkeypatch.delenv("LLM_API_KEY", raising=False)


class TestPrecedence:
    def test_existing_env_not_overwritten(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_EXISTING", "from_shell")
        path = _write_env(tmp_path, "TEST_EXISTING=from_file\n")
        load_env_file(path)
        assert os.environ["TEST_EXISTING"] == "from_shell"

    def test_not_set_key_is_set(self, tmp_path):
        path = _write_env(tmp_path, "TEST_KEY=from_file\n")
        load_env_file(path)
        assert os.environ["TEST_KEY"] == "from_file"
