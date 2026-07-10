#!/usr/bin/env python3
"""REPL driver for the Knight Orc Streamlit UI (app.py).

Drives it via Playwright against the system Chrome binary (not
Playwright's own bundled Chromium — none is installed, and none is
needed since /usr/bin/google-chrome is already present). Designed for
an agent to wrap in tmux and send-keys commands, same shape as the
Claude Code "run" skill's Electron driver pattern, adapted for a plain
browser app instead of a desktop app.

Usage:
    python3 .claude/skills/run-app/driver.py
    (then type commands at the "driver> " prompt, or pipe them in)

Commands:
    start-server [port]     launch `streamlit run app.py` headless, wait for health check
    launch [url]            open a page against the running server (default http://localhost:<port>/)
    ss [name]               screenshot -> $SCREENSHOT_DIR/<name>.png (default /tmp/shots)
    click-text <text>       click a button/element whose text matches (exact, else substring)
    fill <label> <text>     fill a text_input/number_input by its visible label
    wait-text <text>        wait (10s) until <text> appears anywhere on the page
    text                    print the page's visible body text
    eval <js>                evaluate JS in the page, print JSON result
    stop-server             stop the streamlit subprocess
    quit                    stop server (if running), close browser, exit
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

SHOT_DIR = os.environ.get("SCREENSHOT_DIR", "/tmp/shots")
os.makedirs(SHOT_DIR, exist_ok=True)
CHROME_PATH = shutil.which("google-chrome") or "/usr/bin/google-chrome"

_pw = sync_playwright().start()
browser = None
page = None
server_proc = None
server_port = None


def _log(*a):
    print(*a)
    sys.stdout.flush()


def cmd_start_server(port="8765"):
    global server_proc, server_port
    if server_proc is not None:
        _log("already running on port", server_port)
        return
    server_port = port
    log_path = f"/tmp/streamlit_driver_{port}.log"
    server_proc = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(port)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}/_stcore/health"
    for _ in range(30):
        try:
            if urllib.request.urlopen(url, timeout=1).read() == b"ok":
                _log("server healthy on port", port, "log:", log_path)
                return
        except Exception:
            time.sleep(1)
    _log("TIMEOUT waiting for server health check — see", log_path)


def cmd_launch(url=None):
    global browser, page
    if page is not None:
        _log("already launched")
        return
    browser = _pw.chromium.launch(executable_path=CHROME_PATH, headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    target = url or f"http://localhost:{server_port or 8765}/"
    page.goto(target, timeout=15000)
    # Streamlit's first real render happens after its websocket connects;
    # wait for a stable piece of the sidebar rather than a blind sleep.
    try:
        page.wait_for_selector("text=Engine Configuration", timeout=15000)
    except Exception:
        _log("WARNING: sidebar text not seen within 15s — page may still be loading")
    _log("launched:", target)


def cmd_ss(name=None):
    if page is None:
        _log("ERROR: launch first")
        return
    fname = os.path.join(SHOT_DIR, (name or f"ss-{int(time.time())}") + ".png")
    page.screenshot(path=fname, full_page=True)
    _log("screenshot:", fname)


def cmd_click_text(*text_parts):
    text = " ".join(text_parts)
    if page is None:
        _log("ERROR: launch first")
        return
    try:
        page.get_by_text(text, exact=True).first.click(timeout=5000)
        _log("click-text (exact)", repr(text), "-> OK")
        return
    except Exception:
        pass
    try:
        page.get_by_text(text, exact=False).first.click(timeout=5000)
        _log("click-text (substring)", repr(text), "-> OK")
    except Exception as e:
        _log("click-text", repr(text), "-> NOT_FOUND", str(e)[:200])


def cmd_fill(label, *text_parts):
    if page is None:
        _log("ERROR: launch first")
        return
    text = " ".join(text_parts)
    try:
        page.get_by_label(label).fill(text, timeout=5000)
        _log("fill", repr(label), "->", repr(text))
    except Exception as e:
        _log("fill", repr(label), "-> FAILED", str(e)[:200])


def cmd_wait_text(*text_parts):
    text = " ".join(text_parts)
    if page is None:
        _log("ERROR: launch first")
        return
    try:
        page.wait_for_selector(f"text={text}", timeout=10000)
        _log("found:", text)
    except Exception:
        _log("TIMEOUT:", text)


def cmd_text():
    if page is None:
        _log("ERROR: launch first")
        return
    _log(page.inner_text("body")[:4000])


def cmd_eval(*expr_parts):
    if page is None:
        _log("ERROR: launch first")
        return
    expr = " ".join(expr_parts)
    try:
        _log(json.dumps(page.evaluate(expr)))
    except Exception as e:
        _log("ERROR:", str(e))


def cmd_sleep(secs):
    time.sleep(float(secs))


def cmd_stop_server():
    global server_proc, server_port
    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        server_proc = None
        server_port = None
        _log("server stopped")
    else:
        _log("no server running")


def cmd_quit():
    global browser, page
    cmd_stop_server()
    if browser is not None:
        browser.close()
        browser = None
        page = None
    _pw.stop()
    _log("bye")


COMMANDS = {
    "start-server": cmd_start_server,
    "launch": cmd_launch,
    "ss": cmd_ss,
    "click-text": cmd_click_text,
    "fill": cmd_fill,
    "wait-text": cmd_wait_text,
    "text": cmd_text,
    "eval": cmd_eval,
    "sleep": cmd_sleep,
    "stop-server": cmd_stop_server,
    "quit": cmd_quit,
}


def main():
    _log("Knight Orc UI driver — commands:", ", ".join(COMMANDS))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        name, args = parts[0], parts[1:]
        fn = COMMANDS.get(name)
        if fn is None:
            _log("unknown command:", name)
            continue
        try:
            fn(*args)
        except TypeError as e:
            _log("bad args for", name, "-", e)
        if name == "quit":
            return


if __name__ == "__main__":
    main()
