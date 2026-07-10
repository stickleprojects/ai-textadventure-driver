---
name: run-app
description: Launch and drive the Knight Orc Streamlit UI (app.py) headlessly via Playwright, for verifying UI changes end-to-end. Use when asked to run, screenshot, or interactively verify app.py.
---

`app.py` is a Streamlit web app — there is no `chromium-cli` in this
environment, so it's driven via a small Playwright REPL driver at
`.claude/skills/run-app/driver.py`, pointed at the system Chrome binary
(`/usr/bin/google-chrome`) rather than a Playwright-downloaded Chromium
(none is installed, and none is needed).

## Prerequisites (one-time, already satisfied on this machine)

Headless Chrome needs a handful of shared libs. Most were already
present here; only accessibility packages needed installing:

```bash
sudo apt-get install -y libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 \
  libgtk-3-0t64 libgbm1 libasound2t64 libxss1 libxshmfence1
```

**Always ask the user before running this `sudo` line and explain why**
— that's a standing rule for this project, not a one-time check, even
though most machines will already have these installed (confirm with
`ldconfig -p | grep -c libnss3` first — if the count is 0, they're
missing).

Python side: `pip install playwright` (no root needed — it's reusing the
system Chrome, not downloading Playwright's own browser, so no
`playwright install` step).

## Run

```bash
cd /mnt/NTFSData/code/ai/text-adventure-runner
source .venv/bin/activate
python3 .claude/skills/run-app/driver.py
```

It reads commands from stdin, one per line — pipe a heredoc for a full
scripted session, or run interactively:

```bash
cat <<'EOF' | python3 .claude/skills/run-app/driver.py
start-server 8765
launch
click-text Boot Engine (Start Game)
wait-text Run ID
ss 01-booted
quit
EOF
```

Screenshots land in `$SCREENSHOT_DIR` (default `/tmp/shots`).

### Commands

| command | what it does |
|---|---|
| `start-server [port]` | launch `streamlit run app.py --server.headless true`, poll `/_stcore/health` until ready |
| `launch [url]` | open a page (default `http://localhost:<port>/`), wait for the sidebar to render |
| `ss [name]` | full-page screenshot → `$SCREENSHOT_DIR/<name>.png` |
| `click-text <text>` | click a button/element by visible text (exact match first, falls back to substring) |
| `fill <label> <text>` | fill a `text_input`/`number_input` by its visible label |
| `wait-text <text>` | wait up to 10s for `<text>` to appear anywhere on the page |
| `sleep <secs>` | plain wait — needed for anything that triggers a real LLM call + Streamlit rerun loop (Step/Auto-Run), since those take several seconds and there's no clean "done" selector to wait on |
| `text` | print the page's visible body text (first 4000 chars) |
| `eval <js>` | evaluate JS in the page, print JSON result |
| `stop-server` | stop the streamlit subprocess |
| `quit` | stop server (if running), close browser, exit |

## Gotchas

- **`.env` always overwrites shell env vars.** `env_utils.load_env_file()`
  is designed to refresh keyring-resolved values on every load, so it
  unconditionally overwrites `os.environ` from `.env` — setting
  `LLM_PROVIDER=local` in the shell before `start-server` does **not**
  override a `LLM_PROVIDER=deepseek` (or similar) already in `.env`. If
  you need to verify local-model behavior specifically and `.env` is
  pointed at a cloud provider, ask the user first — either they
  temporarily point `.env` elsewhere, or you accept spending a small
  amount of real API cost driving the cloud path instead. Don't edit
  their `.env` without asking.
- **Clicking Step/Start Auto-Run costs real money if `.env` has a cloud
  provider configured.** Each step is a real LLM API call. Ask the user
  before triggering more than a couple of steps this way — see the note
  above about `.env` overwriting local-mode overrides.
- **No fixed "done" signal for a multi-step batch.** Streamlit's
  rerun-driven loop (Auto-Run / N-step batches, feature 69) has no DOM
  marker for "batch finished" — poll `logs/<run_id>.json` on disk (it's
  rewritten after every step, feature 66) or just `sleep` long enough
  (steps × `step_delay` × a safety margin, plus real API latency for
  cloud providers).
- **Streamlit widgets need real DOM events**, not JS value assignment —
  `fill()`/`click()` go through Playwright's real input pipeline
  already; don't `eval` a manual `.value = …` set, React won't see it.
- **Clean up test artifacts.** Any `logs/<run_id>.json` /
  `runs/<run_id>.json` / `runs/orchestrator/<run_id>/` created by a
  verification run (`run_id` prefix `ui_YYYYMMDD_HHMMSS`) is throwaway —
  delete it afterward so it doesn't clutter real run history.

## Troubleshooting

- **`sudo: a terminal is required to authenticate`** when installing the
  prerequisite libs from a non-interactive Bash tool call: use
  `SUDO_ASKPASS=/usr/bin/ssh-askpass sudo -A <cmd>` instead of bare
  `sudo` — this machine has `ssh-askpass` installed, which pops a GUI
  password prompt the user can answer directly. Still ask first.
- **Health check times out:** check the `/tmp/streamlit_driver_<port>.log`
  path printed by `start-server` for a Python traceback — usually an
  import error or a bad `_config_path`/`_strategy_path` env var.
- **`click-text` reports `NOT_FOUND`:** Streamlit sometimes hasn't
  finished its first real render yet even after the sidebar text is
  visible (widgets mount slightly after static text) — add a short
  `sleep 1` before the click, or `wait-text` on something closer to the
  target element first.
