import json
from collections import deque

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

_CARDINAL_VECTORS = {
    "north": (0, -1), "south": (0, 1),
    "east":  (1,  0), "west":  (-1, 0),
    "ne": (1, -1), "nw": (-1, -1),
    "se": (1,  1), "sw": (-1,  1),
}
_SPACING = 220       # pixels between grid cells within a level
_Z_BAND = 600        # pixels between elevation bands

# PNG export tweaks — adjust after inspecting the first output image
_MAP_FIG_SIZE = (16, 10)   # minimum figure size in inches (width, height)
_MAP_PX_PER_CELL = 120     # pixels allocated per grid cell; drives auto figure sizing
_MAP_NODE_SIZE = 1800       # matplotlib scatter area units; increase if labels clip
_MAP_FONT_SIZE = 7          # node label font size in points
_MAP_EDGE_FONT_SIZE = 6     # edge label font size in points
_MAP_DPI = 120              # output resolution


def _compute_positions_and_levels(graph):
    """BFS assigning (grid_x, grid_y) for cardinal moves and z_level for up/down.

    Runs a separate BFS pass for each weakly-connected component so disconnected
    subgraphs (e.g. rooms only seen in responses but never traversed) still get
    positions. Components are laid out side-by-side with a gap between them.
    """
    if not graph.nodes:
        return {}, {}

    grid = {}    # node -> (grid_x, grid_y)
    levels = {}  # node -> z_level

    # x-offset applied to each new component so they don't overlap
    component_x_offset = 0

    for component in nx.weakly_connected_components(graph):
        start = next(iter(component))
        grid[start] = (component_x_offset, 0)
        levels[start] = 0
        queue = deque([start])
        component_max_x = component_x_offset

        while queue:
            node = queue.popleft()
            gx, gy = grid[node]
            gz = levels[node]
            # Follow outgoing edges (with direction label for positioning)
            for _, neighbor, data in graph.edges(node, data=True):
                if neighbor in grid:
                    continue
                label = data.get("label", "").lower()
                if label == "up":
                    grid[neighbor] = (gx, gy)
                    levels[neighbor] = gz + 1
                elif label == "down":
                    grid[neighbor] = (gx, gy)
                    levels[neighbor] = gz - 1
                else:
                    dx, dy = _CARDINAL_VECTORS.get(label, (0, 0))
                    cx, cy = gx + dx, gy + dy
                    occupied = {grid[n] for n in grid if levels.get(n, 0) == gz}
                    while (cx, cy) in occupied:
                        cx += 1
                    grid[neighbor] = (cx, cy)
                    levels[neighbor] = gz
                component_max_x = max(component_max_x, grid[neighbor][0])
                queue.append(neighbor)
            # Follow incoming edges so nodes only reachable via reverse edges get positions
            for predecessor, _, data in graph.in_edges(node, data=True):
                if predecessor in grid:
                    continue
                label = data.get("label", "").lower()
                reverse = {"north": "south", "south": "north", "east": "west", "west": "east",
                           "up": "down", "down": "up", "ne": "sw", "sw": "ne", "nw": "se", "se": "nw"}
                rev_label = reverse.get(label, "")
                if rev_label == "up":
                    grid[predecessor] = (gx, gy)
                    levels[predecessor] = gz + 1
                elif rev_label == "down":
                    grid[predecessor] = (gx, gy)
                    levels[predecessor] = gz - 1
                else:
                    dx, dy = _CARDINAL_VECTORS.get(rev_label, (0, 0))
                    cx, cy = gx + dx, gy + dy
                    occupied = {grid[n] for n in grid if levels.get(n, 0) == gz}
                    while (cx, cy) in occupied:
                        cx += 1
                    grid[predecessor] = (cx, cy)
                    levels[predecessor] = gz
                component_max_x = max(component_max_x, grid[predecessor][0])
                queue.append(predecessor)

        component_x_offset = component_max_x + 4  # gap between components

    return grid, levels


def _pixel_positions(grid, levels):
    """Convert grid coordinates + z-level to pixel positions, banding by elevation."""
    return {
        node: (gx * _SPACING, gy * _SPACING + levels.get(node, 0) * _Z_BAND)
        for node, (gx, gy) in grid.items()
    }


def _node_color(z, is_current, is_unknown):
    if is_current:
        return {"background": "#f59e0b", "border": "#f59e0b"}
    if is_unknown:
        return {"background": "#2a2a4a", "border": "#7c3aed"}
    if z > 0:
        return {"background": "#1d4ed8", "border": "#3b82f6"}  # upper floor — blue
    if z < 0:
        return {"background": "#78350f", "border": "#92400e"}  # basement — brown
    return {"background": "#7c3aed", "border": "#7c3aed"}      # ground — purple


def _display_label(node_id):
    """Short, word-wrapped label for graph nodes (node IDs are unchanged)."""
    if node_id.startswith("Unknown (") and node_id.endswith(")"):
        inner = node_id[len("Unknown ("):-1]
        direction = inner.split(" from ")[0] if " from " in inner else inner[:20]
        return f"? {direction}"
    words = node_id.split()
    lines, line, length = [], [], 0
    for w in words:
        if length + len(w) + (1 if line else 0) > 20:
            lines.append(" ".join(line))
            line, length = [w], len(w)
        else:
            line.append(w)
            length += len(w) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def render_graph(state):
    g = state["world_graph"]

    if not g.nodes:
        st.markdown(
            "<p style='color:#7c3aed;font-family:monospace;text-align:center'>"
            "INITIALIZING MAPPING MATRIX...</p>",
            unsafe_allow_html=True,
        )
        return

    grid, levels = _compute_positions_and_levels(g)
    positions = _pixel_positions(grid, levels)
    net = Network(height="430px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)

    current_room = state["current_room"]
    for node in g.nodes:
        z = levels.get(node, 0)
        px, py = positions.get(node, (0, 0))
        color = _node_color(z, node == current_room, node.startswith("Unknown"))
        net.add_node(
            node,
            label=_display_label(node),
            shape="box",
            x=px, y=py,
            physics=False,
            color=color,
            font={"size": 12, "color": "white"},
            borderWidth=3 if node == current_room else 1,
        )

    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, label=data.get("label", ""), color="#4a4a6a",
                     font={"size": 10, "color": "#a78bfa", "strokeWidth": 0})

    net.set_options("""{
      "physics": { "enabled": false },
      "interaction": { "dragNodes": true, "zoomView": true, "dragView": true },
      "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 } },
      "nodes": { "widthConstraint": { "maximum": 150 } }
    }""")

    components.html(net.generate_html(), height=450, scrolling=False)


def save_graph_image(graph, current_room, output_path):
    """Render the world graph to a PNG using matplotlib. Safe to call headlessly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    if not graph.nodes:
        return

    grid, levels = _compute_positions_and_levels(graph)
    # matplotlib uses (x, y) with y increasing upward — negate y so north is up
    pos = {node: (gx * 2, -(gy * 2 + levels.get(node, 0) * 6))
           for node, (gx, gy) in grid.items()}

    node_colors = []
    for node in graph.nodes:
        z = levels.get(node, 0)
        if node == current_room:
            node_colors.append("#f59e0b")
        elif node.startswith("Unknown"):
            node_colors.append("#2a2a4a")
        elif z > 0:
            node_colors.append("#1d4ed8")
        elif z < 0:
            node_colors.append("#78350f")
        else:
            node_colors.append("#7c3aed")

    labels = {n: _display_label(n) for n in graph.nodes}
    edge_labels = {(u, v): d.get("label", "") for u, v, d in graph.edges(data=True)}

    # Size the figure from the actual grid bounding box so the layout fills
    # the image regardless of how many rooms there are.  Each grid cell gets
    # _MAP_PX_PER_CELL pixels; figure size is clamped to _MAP_FIG_SIZE minimum.
    if grid:
        xs = [gx for gx, gy in grid.values()]
        ys = [gy + levels.get(n, 0) * 3 for n, (gx, gy) in grid.items()]
        x_span = max(max(xs) - min(xs) + 2, 1)
        y_span = max(max(ys) - min(ys) + 2, 1)
    else:
        x_span = y_span = 1
    fig_w = max(_MAP_FIG_SIZE[0], x_span * _MAP_PX_PER_CELL / _MAP_DPI)
    fig_h = max(_MAP_FIG_SIZE[1], y_span * _MAP_PX_PER_CELL / _MAP_DPI)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx(
        graph, pos=pos, ax=ax,
        labels=labels,
        node_color=node_colors,
        node_size=_MAP_NODE_SIZE,
        node_shape="s",
        font_color="white",
        font_size=_MAP_FONT_SIZE,
        edge_color="#4a4a6a",
        arrows=True,
        arrowsize=12,
    )
    nx.draw_networkx_edge_labels(
        graph, pos=pos, edge_labels=edge_labels, ax=ax,
        font_color="#a78bfa", font_size=_MAP_EDGE_FONT_SIZE,
    )

    ax.set_title(f"World Map — {len(graph.nodes)} rooms", color="white", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=_MAP_DPI, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def generate_markdown_log(state):
    lines = ["# Autonomous Run Log\n"]
    for entry in state["game_log"]:
        lines.append(f"### [{entry['timestamp']}] Action: `{entry['action']}`")
        lines.append(f"**Response:**\n> {entry['response']}\n")
        lines.append(f"**Extracted Knowledge:**\n```json\n{json.dumps(entry['extracted'], indent=2)}\n```\n\n---")
    return "\n".join(lines)


def generate_json_log(state):
    return json.dumps(state["game_log"], indent=2)
