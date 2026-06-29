import json

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from datetime import datetime


def render_graph(state):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    g = state["world_graph"]
    if g.nodes:
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, ax=ax, with_labels=True,
                node_color='#7c3aed', edge_color='#4a4a6a',
                font_color='white', node_size=1800, font_size=9,
                font_weight="bold", width=1.5, node_shape="s")
        edge_labels = nx.get_edge_attributes(g, 'label')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='#7c3aed', font_size=8,
                                     bbox=dict(facecolor='#1a1a2e', edgecolor='none'))
    else:
        ax.text(0.5, 0.5, "INITIALIZING MAPPING MATRIX...",
                color="#7c3aed", ha="center", fontfamily="monospace")

    st.pyplot(fig)


def generate_markdown_log(state):
    lines = ["# Autonomous Run Log\n"]
    for entry in state["game_log"]:
        lines.append(f"### [{entry['timestamp']}] Action: `{entry['action']}`")
        lines.append(f"**Response:**\n> {entry['response']}\n")
        lines.append(f"**Extracted Knowledge:**\n```json\n{json.dumps(entry['extracted'], indent=2)}\n```\n\n---")
    return "\n".join(lines)
