import json

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


def render_graph(state):
    g = state["world_graph"]

    if not g.nodes:
        st.markdown(
            "<p style='color:#7c3aed;font-family:monospace;text-align:center'>"
            "INITIALIZING MAPPING MATRIX...</p>",
            unsafe_allow_html=True,
        )
        return

    net = Network(height="430px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)

    current_room = state["current_room"]
    for node in g.nodes:
        is_current = node == current_room
        is_unknown = node.startswith("Unknown")
        net.add_node(
            node,
            label=node,
            shape="box",
            color={
                "background": "#f59e0b" if is_current else ("#2a2a4a" if is_unknown else "#7c3aed"),
                "border": "#f59e0b" if is_current else "#7c3aed",
            },
            font={"size": 12, "color": "white"},
            borderWidth=3 if is_current else 1,
        )

    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, label=data.get("label", ""), color="#4a4a6a",
                     font={"size": 10, "color": "#a78bfa", "strokeWidth": 0})

    net.set_options("""{
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": { "springLength": 120, "gravitationalConstant": -60 },
        "stabilization": { "iterations": 150 }
      },
      "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 } }
    }""")

    components.html(net.generate_html(), height=450, scrolling=False)


def generate_markdown_log(state):
    lines = ["# Autonomous Run Log\n"]
    for entry in state["game_log"]:
        lines.append(f"### [{entry['timestamp']}] Action: `{entry['action']}`")
        lines.append(f"**Response:**\n> {entry['response']}\n")
        lines.append(f"**Extracted Knowledge:**\n```json\n{json.dumps(entry['extracted'], indent=2)}\n```\n\n---")
    return "\n".join(lines)
