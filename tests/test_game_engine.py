import pexpect
import pytest

from game_engine import clean_level9_output, execute_game_command


@pytest.fixture(autouse=True)
def _no_drain(monkeypatch):
    # drain_game_buffer loops until pexpect.TIMEOUT; stub_child never raises it.
    # Patch it out here — drain behaviour is tested separately if needed.
    monkeypatch.setattr("game_engine.drain_game_buffer", lambda child, timeout=0.3: None)


def test_timeout_includes_partial_output(stub_child):
    stub_child.before = "You are in a dark cave.\nA troll blocks the way"
    stub_child.expect.side_effect = pexpect.TIMEOUT("timeout")
    result = execute_game_command(stub_child, "north")
    assert "WARNING" in result
    assert "dark cave" in result
    assert "troll blocks the way" in result


def test_timeout_empty_before(stub_child):
    stub_child.before = ""
    stub_child.expect.side_effect = pexpect.TIMEOUT("timeout")
    result = execute_game_command(stub_child, "north")
    assert "WARNING" in result
    assert "(nothing received)" in result


def test_clean_output_strips_ansi():
    raw = "\x1b[1mYou are in a forest.\x1b[0m\r\n\r\nExits: north"
    cleaned = clean_level9_output(raw)
    assert "\x1b" not in cleaned
    assert cleaned == "You are in a forest.\nExits: north"


def test_dead_process_returns_critical(stub_child):
    stub_child.isalive.return_value = False
    result = execute_game_command(stub_child, "look")
    assert "CRITICAL ERROR" in result


def test_command_strips_echo(stub_child):
    stub_child.before = "look\nYou are in a forest."
    result = execute_game_command(stub_child, "look")
    assert result == "You are in a forest."


def test_command_strips_echo_with_leading_crlf(stub_child):
    # pexpect may include \r\n before the echo in some terminal modes
    stub_child.before = "\r\nwear putty knife\r\nYou can't wear that."
    result = execute_game_command(stub_child, "wear putty knife")
    assert result == "You can't wear that."
