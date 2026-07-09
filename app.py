import os
import time
from datetime import datetime

import streamlit.components.v1 as components

import networkx as nx
import streamlit as st

from env_utils import load_env_file
load_env_file()  # populate os.environ from .env before config or LLM setup

from game_config import config
from agent import process_agent_step
from agent_tools import run_tool_calling_step
from game_engine import start_level9
from llm import (
    ANTHROPIC_AVAILABLE,
    LLAMA_AVAILABLE,
    OPENAI_AVAILABLE,
    extract_knowledge,
    load_anthropic_tool_llm,
    load_llm,
    load_openai_tool_llm,
)
from parse_strategies import LLMToolCallParseStrategy
from run_evaluator import load_strategy
from ui import generate_json_log, generate_markdown_log, render_graph
from world_graph import update_graph

_config_path = os.environ.get("GAME_CONFIG", "configs/knight_orc.json")
_strategy_path = os.environ.get("STRATEGY_PATH", "configs/knight_orc_strategy.json")
if _config_path and os.path.isfile(_config_path):
    config.load_from_file(_config_path)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")

st.set_page_config(page_title="Text Adventure Autonomous OS", layout="wide", initial_sidebar_state="expanded")


def _make_clean_state():
    """Build a fresh system_state seeded with strategy data from disk."""
    strategy = load_strategy(_strategy_path)
    known_entities = {
        name: {"status": "discovered", "location": None, "verb_outcomes": dict(verbs)}
        for name, verbs in strategy.get("entity_verb_outcomes", {}).items()
    }
    wg_data = strategy.get("world_graph", {"nodes": [], "edges": []})
    world_graph = nx.MultiDiGraph()
    world_graph.add_nodes_from(wg_data.get("nodes", []))
    for u, v, label in wg_data.get("edges", []):
        # Split legacy compound labels ("south/out") into separate edges on load.
        for part in label.split("/"):
            if part:
                existing = {d["label"] for d in (world_graph.get_edge_data(u, v) or {}).values()}
                if part not in existing:
                    world_graph.add_edge(u, v, label=part)
    return {
        "current_room": "Unknown Location",
        "inventory": [],
        "spellbook": [],
        "known_entities": known_entities,
        "world_graph": world_graph,
        "uninspected_objects": [],
        "current_inspection": {
            "target": None,
            "sequence": config.inspection_sequence,
            "step_index": 0,
        },
        "known_npcs": {},
        "unresolved_anomalies": {},
        "active_goal": None,
        "game_log": [],
        "is_running": False,
        "current_score": None,
        "max_score": None,
        "futile_edges": strategy["futile_edges"],
        "pending_npc_tasks": [],
    }


def init_state():
    if 'system_state' not in st.session_state:
        st.session_state.system_state = _make_clean_state()
    if 'level9_process' not in st.session_state:
        st.session_state.level9_process = None
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
    if 'initial_text' not in st.session_state:
        st.session_state.initial_text = None


init_state()
state = st.session_state.system_state

st.title("🛡️ Knight Orc Autonomous OS")
st.markdown("Local LLM-driven text adventure agent featuring autonomous spatial backtracking and dynamic anomaly resolution.")


def _run_step(child):
    """Dispatch one agent step to the appropriate loop based on LLM_PROVIDER."""
    if LLM_PROVIDER == "local":
        process_agent_step(state, child, llm)
    else:
        step_num = len(state["game_log"]) + 1
        run_tool_calling_step(
            state, child, llm, parse_strategy, st.session_state.run_id, step_num,
            initial_text=st.session_state.get("initial_text"),
        )


# --- Sidebar ---
with st.sidebar:
    st.header("Engine Configuration")

    interpreter_path = st.text_input("Interpreter Path", value="./tools/glklevel9")
    rom_path = st.text_input("Level 9 ROM Path", value="./gamefiles/knight-orc/GAMEDAT1.DAT")

    if LLM_PROVIDER == "local":
        model_path = st.text_input("Local llama.cpp Model Path", value="../models/Phi-3.5-mini-instruct-Q3_K_M.gguf")
        llm = load_llm(model_path) if LLAMA_AVAILABLE else None
        parse_strategy = None
        if not LLAMA_AVAILABLE:
            st.warning("llama-cpp-python offline. Logic will fail without a model.")
    else:
        st.info(f"Cloud LLM: **{LLM_PROVIDER}** / {LLM_MODEL}")
        cloud_model = st.text_input("Model", value=LLM_MODEL)
        cloud_key = st.text_input("API Key", value=LLM_API_KEY, type="password")
        cloud_url = st.text_input("Base URL (optional)", value=LLM_BASE_URL)
        if LLM_PROVIDER == "anthropic":
            llm = load_anthropic_tool_llm(cloud_model, cloud_key) if ANTHROPIC_AVAILABLE else None
            available = ANTHROPIC_AVAILABLE
        else:
            llm = load_openai_tool_llm(LLM_PROVIDER, cloud_model, cloud_key, cloud_url or None) if OPENAI_AVAILABLE else None
            available = OPENAI_AVAILABLE
        if not available:
            st.warning("Required cloud SDK not installed. Tool-calling LLM unavailable.")
        parse_strategy = LLMToolCallParseStrategy(llm) if llm is not None else None

    _proc = st.session_state.level9_process
    engine_running = _proc is not None and _proc.isalive()

    if st.button("Boot Engine (Start Game)", type="primary", use_container_width=True, disabled=engine_running):
        child, init_response = start_level9(interpreter_path, rom_path)
        st.session_state.level9_process = child
        st.session_state.run_id = f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if LLM_PROVIDER == "local":
            extracted_init = extract_knowledge(init_response, "look", llm)
            if extracted_init.get("room"):
                state["current_room"] = extracted_init["room"]
            if extracted_init.get("exits"):
                update_graph(state, state["current_room"], extracted_init["exits"], None, None)
            log_extracted = extracted_init
        else:
            log_extracted = {}
        st.session_state.initial_text = init_response

        state["game_log"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": "[SYSTEM BOOT]",
            "response": init_response,
            "extracted": log_extracted,
        })

        if child:
            st.success("glklevel9 instance running.")
        else:
            st.error(init_response)

    if st.button("Reset Game & State", use_container_width=True, disabled=not engine_running):
        if st.session_state.level9_process is not None:
            st.session_state.level9_process.terminate(force=True)
            st.session_state.level9_process = None
        if _config_path and os.path.isfile(_config_path):
            config.load_from_file(_config_path)
        st.session_state.system_state = _make_clean_state()
        st.session_state.run_id = None
        st.session_state.initial_text = None
        state = st.session_state.system_state
        st.success("State reset. Config and strategy reloaded from disk.")

    st.header("Agent Operations")
    step_delay = st.slider("Step Delay (seconds)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    col1, col2 = st.columns(2)
    with col1:
        label = "Stop Auto-Run" if state["is_running"] else "Start Auto-Run"
        if st.button(label, use_container_width=True):
            state["is_running"] = not state["is_running"]
    with col2:
        if st.button("Step Once", use_container_width=True):
            child = st.session_state.level9_process
            if child and child.isalive():
                _run_step(child)
            else:
                st.error("Engine offline. Boot engine first.")

    st.divider()
    st.download_button(
        label="Export Run Log (Markdown)",
        data=generate_markdown_log(state),
        file_name=f"knight_orc_run_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        label="Export Run Log (JSON)",
        data=generate_json_log(state),
        file_name=f"knight_orc_run_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )

# --- Main Dashboard ---
col_viz, col_state = st.columns([3, 2])

with col_viz:
    st.subheader("Spatial Matrix")
    render_graph(state)

    st.subheader("I-O Terminal")
    with st.container(height=350):
        for entry in state["game_log"][-8:]:
            st.markdown(f"**> `{entry['action']}`**")
            st.text(entry['response'])
            trace = entry.get("tool_trace") or entry.get("llm_trace")
            if trace:
                with st.expander("Decision detail", expanded=False):
                    st.json(trace)
            st.divider()
        st.markdown('<div class="io-terminal-bottom"></div>', unsafe_allow_html=True)
    components.html(
        """<script>
        (function() {
            function scroll() {
                try {
                    var markers = window.parent.document.getElementsByClassName('io-terminal-bottom');
                    if (!markers.length) return;
                    markers[markers.length - 1].scrollIntoView({block: 'end', behavior: 'instant'});
                } catch(e) {}
            }
            scroll();
            setTimeout(scroll, 150);
        })();
        </script>""",
        height=0,
    )

with col_state:
    st.subheader("Dynamic Schema")
    st.json({
        "Location": state["current_room"],
        "Active Goal": state["active_goal"] or "Exploring",
        "Inventory": state["inventory"],
        "Spellbook": state["spellbook"],
    })

    st.subheader("Unresolved Anomalies")
    if state["unresolved_anomalies"]:
        for target, data in state["unresolved_anomalies"].items():
            st.warning(
                f"**{target.upper()}** in {data['room']}\n\n"
                f"*Reason:* {data['reason']} | *Needs:* {data['potential_solution']}"
            )
    else:
        st.success("No active blockers detected.")

# --- Auto-run loop ---
if state["is_running"]:
    child = st.session_state.level9_process
    if child and child.isalive():
        _run_step(child)
        time.sleep(step_delay)
        st.rerun()
    else:
        state["is_running"] = False
        st.sidebar.error("Auto-Run stopped: Engine offline.")
        st.rerun()
