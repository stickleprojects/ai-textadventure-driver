import json
import textwrap
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
_MAP_NODE_SIZE = 1800       # matplotlib scatter area units for real rooms
_MAP_UNKNOWN_SIZE = 300     # smaller circle for Unknown placeholder nodes
_MAP_FONT_SIZE = 7          # node label font size in points
_MAP_EDGE_FONT_SIZE = 6     # edge label font size in points
_MAP_DPI = 120              # output resolution
_LABEL_WRAP_WIDTH = 12      # chars per line; matches ~11-char node box at _MAP_NODE_SIZE=1800


_BFS_REVERSE = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "up": "down", "down": "up", "ne": "sw", "sw": "ne", "nw": "se", "se": "nw",
    "in": "out", "out": "in",
}


def _free_cell(gx, gy, dx, dy, grid, levels, gz):
    """Return the nearest unoccupied grid cell starting at (gx+dx, gy+dy).

    For diagonals (dx≠0 and dy≠0) collision shifts along the diagonal so the
    45° geometry is preserved.  For cardinal moves or unknown directions the
    fallback shift is always +x (east) to keep the row tidy.
    """
    cx, cy = gx + dx, gy + dy
    occupied = {grid[n] for n in grid if levels.get(n, 0) == gz}
    sx = dx if (dx != 0 and dy != 0) else 1
    sy = dy if (dx != 0 and dy != 0) else 0
    while (cx, cy) in occupied:
        cx += sx
        cy += sy
    return cx, cy


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
                # Merged labels (e.g. "south/down") — use first token for positioning
                label = data.get("label", "").lower().split("/")[0]
                if label in ("up", "out"):
                    grid[neighbor] = (gx, gy)
                    levels[neighbor] = gz + 1
                elif label in ("down", "in"):
                    grid[neighbor] = (gx, gy)
                    levels[neighbor] = gz - 1
                else:
                    dx, dy = _CARDINAL_VECTORS.get(label, (0, 0))
                    grid[neighbor] = _free_cell(gx, gy, dx, dy, grid, levels, gz)
                    levels[neighbor] = gz
                component_max_x = max(component_max_x, grid[neighbor][0])
                queue.append(neighbor)
            # Follow incoming edges so nodes only reachable via reverse edges get positions
            for predecessor, _, data in graph.in_edges(node, data=True):
                if predecessor in grid:
                    continue
                label = data.get("label", "").lower().split("/")[0]
                rev_label = _BFS_REVERSE.get(label, "")
                if rev_label in ("up", "out"):
                    grid[predecessor] = (gx, gy)
                    levels[predecessor] = gz + 1
                elif rev_label in ("down", "in"):
                    grid[predecessor] = (gx, gy)
                    levels[predecessor] = gz - 1
                else:
                    dx, dy = _CARDINAL_VECTORS.get(rev_label, (0, 0))
                    grid[predecessor] = _free_cell(gx, gy, dx, dy, grid, levels, gz)
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
    """Word-wrapped label for graph nodes. Unknown placeholders show as '?'."""
    if node_id.startswith("Unknown ("):
        return "?"
    return "\n".join(textwrap.wrap(node_id, width=_LABEL_WRAP_WIDTH) or [node_id])


def _iter_display_edges(g):
    """Yield (u, v, label, bidirectional) for rendering.

    When both (u, v) and (v, u) exist, yield them once as a single bidirectional
    entry with a combined label (e.g. "north/south"). One-way edges are yielded
    as-is with bidirectional=False.
    """
    seen = set()
    for u, v, data in g.edges(data=True):
        if (u, v) in seen:
            continue
        label = data.get("label", "")
        rev = g.get_edge_data(v, u)
        if rev is not None:
            rev_label = rev.get("label", "")
            combined = f"{label}/{rev_label}" if label != rev_label else label
            yield u, v, combined, True
            seen.add((v, u))
        else:
            yield u, v, label, False
        seen.add((u, v))


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

    for u, v, label, bidir in _iter_display_edges(g):
        net.add_edge(u, v, label=label, color="#4a4a6a",
                     arrows="to, from" if bidir else "to",
                     font={"size": 10, "color": "#a78bfa", "strokeWidth": 0})

    net.set_options("""{
      "physics": { "enabled": false },
      "interaction": { "dragNodes": true, "zoomView": true, "dragView": true },
      "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 } },
      "nodes": { "widthConstraint": { "maximum": 150 } }
    }""")

    components.html(net.generate_html(), height=450, scrolling=False)


def save_graph_image(graph, current_room, output_path, draw_unknowns=True):
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

    real_nodes = [n for n in graph.nodes if not n.startswith("Unknown")]
    if not draw_unknowns:
        unknown_nodes = []
    else:
        unknown_nodes = [n for n in graph.nodes if n.startswith("Unknown")]

    real_colors = []
    for node in real_nodes:
        z = levels.get(node, 0)
        if node == current_room:
            real_colors.append("#f59e0b")
        elif z > 0:
            real_colors.append("#1d4ed8")
        elif z < 0:
            real_colors.append("#78350f")
        else:
            real_colors.append("#7c3aed")

    # Size figure from actual grid bounding box
    if grid:
        xs = [gx for gx, gy in grid.values()]
        ys = [gy + levels.get(n, 0) * 3 for n, (gx, gy) in grid.items()]
        x_span = max(max(xs) - min(xs) + 2, 1)
        y_span = max(max(ys) - min(ys) + 2, 1)
    else:
        x_span = y_span = 1
    fig_w = max(_MAP_FIG_SIZE[0], x_span * _MAP_PX_PER_CELL / _MAP_DPI)
    fig_h = max(_MAP_FIG_SIZE[1], y_span * _MAP_PX_PER_CELL / _MAP_DPI)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    # Real rooms — coloured squares with white labels
    if real_nodes:
        nx.draw_networkx_nodes(
            graph, pos=pos, ax=ax, nodelist=real_nodes,
            node_color=real_colors, node_size=_MAP_NODE_SIZE, node_shape="s",
        )
        nx.draw_networkx_labels(
            graph, pos=pos, ax=ax,
            labels={n: _display_label(n) for n in real_nodes},
            font_color="white", font_size=_MAP_FONT_SIZE,
        )

    # Unknown placeholders — small grey circles with "?" label
    if unknown_nodes:
        nx.draw_networkx_nodes(
            graph, pos=pos, ax=ax, nodelist=unknown_nodes,
            node_color="#aaaaaa", node_size=_MAP_UNKNOWN_SIZE, node_shape="o",
        )
        nx.draw_networkx_labels(
            graph, pos=pos, ax=ax,
            labels={n: "?" for n in unknown_nodes},
            font_color="black", font_size=_MAP_FONT_SIZE - 1,
        )

    # Edges — black lines, black labels on transparent background
    visible_nodes = set(real_nodes) | set(unknown_nodes)
    visible_edges = [(u, v) for u, v in graph.edges() if u in visible_nodes and v in visible_nodes]
    nx.draw_networkx_edges(
        graph, pos=pos, ax=ax, edgelist=visible_edges,
        edge_color="black", arrows=True, arrowsize=12,
    )
    edge_labels = {(u, v): d.get("label", "") for u, v, d in graph.edges(data=True)
                   if u in visible_nodes and v in visible_nodes}
    nx.draw_networkx_edge_labels(
        graph, pos=pos, edge_labels=edge_labels, ax=ax,
        font_color="black", font_size=_MAP_EDGE_FONT_SIZE,
        bbox={"facecolor": "none", "edgecolor": "none"},
    )

    ax.set_title(f"World Map — {len(graph.nodes)} rooms", color="black", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=_MAP_DPI, bbox_inches="tight", facecolor="white")
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
