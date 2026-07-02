import re
import pexpect

from game_config import config


def clean_level9_output(raw_text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', raw_text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def start_level9(interpreter_path, rom_path):
    """Spawns glklevel9 and reads the initial room description.

    Returns (child, text) — child is None on failure.
    """
    try:
        child = pexpect.spawn(f'{interpreter_path} "{rom_path}"', encoding='utf-8', timeout=5)
        child.expect(config.prompt_pattern)
        return child, clean_level9_output(child.before)
    except Exception as e:
        return None, f"Error starting interpreter: {str(e)}"


def drain_game_buffer(child, timeout=0.3):
    """Consume any auto-advance prompts left over from game state transitions.

    The death holding room runs automatically for several ticks, each printing a
    'What now?' prompt without waiting for player input.  If we don't drain these
    before the next command, execute_game_command will mis-pair actions with responses.

    Reads until no further prompt arrives within `timeout` seconds.
    """
    try:
        while True:
            child.expect(config.prompt_pattern, timeout=timeout)
    except pexpect.TIMEOUT:
        pass


def execute_game_command(child, command):
    """Sends a command to the active process and returns the response text."""
    if not child or not child.isalive():
        return "CRITICAL ERROR: Game process is not running. Please start the ROM."
    drain_game_buffer(child)
    try:
        child.sendline(command)
        child.expect(config.prompt_pattern)
        raw_output = child.before
        trimmed = raw_output.lstrip('\r\n ')
        if trimmed.startswith(command):
            raw_output = trimmed[len(command):]
        return clean_level9_output(raw_output)
    except pexpect.TIMEOUT:
        partial = clean_level9_output(child.before) if child.before else "(nothing received)"
        return f"WARNING: Command timed out waiting for game prompt\nRaw output received:\n{partial}"
    except Exception as e:
        return f"CRITICAL ERROR: {str(e)}"
