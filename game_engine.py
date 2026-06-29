import re
import pexpect


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
        child.expect(r'What now\?')
        return child, clean_level9_output(child.before)
    except Exception as e:
        return None, f"Error starting interpreter: {str(e)}"


def execute_game_command(child, command):
    """Sends a command to the active process and returns the response text."""
    if not child or not child.isalive():
        return "CRITICAL ERROR: Game process is not running. Please start the ROM."
    try:
        child.sendline(command)
        child.expect(r'What now\?')
        raw_output = child.before
        if raw_output.startswith(command):
            raw_output = raw_output[len(command):]
        return clean_level9_output(raw_output)
    except pexpect.TIMEOUT:
        return "WARNING: Command timed out. The game might be waiting for non-standard input."
    except Exception as e:
        return f"CRITICAL ERROR: {str(e)}"
