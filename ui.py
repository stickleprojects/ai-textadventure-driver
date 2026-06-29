import json
from collections import deque

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

_DIRECTION_VECTORS = {
    "north": (0, -1), "south": (0, 1),
    "east":  (1,  0), "west":  (-1, 0),
    "ne": (1, -1), "nw": (-1, -1),
    "se": (1,  1), "sw": (-1,  1),
    "up": (0, -1), "down": (0, 1),
}
_SPACING = 220  # pixels between grid cells


def _compute_cardinal_positions(graph):
    """BFS from the first discovered node, assigning pixel positions from edge direction labels."""
    if not graph.nodes:
        return {}

    grid = {}  # node -> (grid_x, grid_y)
    start = next(iter(graph.nodes))
    grid[start] = (0, 0)
    queue = deque([start])

    while queue:
        node = queue.popleft()
        gx, gy = grid[node]
        for _, neighbor, data in graph.edges(node, data=True):
            if neighbor in grid:
                continue
            dx, dy = _DIRECTION_VECTORS.get(data.get("label", "").lower(), (0, 0))
            cx, cy = gx + dx, gy + dy
            # nudge right if the target cell is already occupied
            occupied = set(grid.values())
            while (cx, cy) in occupied:
                cx += 1
            grid[neighbor] = (cx, cy)
            queue.append(neighbor)

    return {node: (gx * _SPACING, gy * _SPACING) for node, (gx, gy) in grid.items()}


def render_graph(state):
    g = state["world_graph"]

    if not g.nodes:
        st.markdown(
            "<p style='color:#7c3aed;font-family:monospace;text-align:center'>"
            "INITIALIZING MAPPING MATRIX...</p>",
            unsafe_allow_html=True,
        )
        return

    positions = _compute_cardinal_positions(g)
    net = Network(height="430px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)

    current_room = state["current_room"]
    for node in g.nodes:
        is_current = node == current_room
        is_unknown = node.startswith("Unknown")
        px, py = positions.get(node, (0, 0))
        net.add_node(
            node,
            label=node,
            shape="box",
            x=px, y=py,
            physics=False,
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
      "physics": { "enabled": false },
      "interaction": { "dragNodes": true, "zoomView": true, "dragView": true },
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
