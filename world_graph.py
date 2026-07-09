"""Shared room-resolution, world-graph mutation, and navigation helpers.

Feature 63 extracts this logic out of agent.py so every caller uses one
public module rather than reaching into private agent internals.
"""

import re

import networkx as nx

from game_config import config

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b\s*", re.IGNORECASE)
# Strip leading positional prepositions — LLM says "in an alder ghostwood",
# "on a jousting field" etc. Two passes needed: preposition first, then article.
# "inside"/"outside" are NOT stripped: they denote distinct rooms.
_LEADING_PREP_RE = re.compile(r"^(in|on|at)\s+", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
# Bug 78: a grounded (bug 74) but overly-verbose room quote — "You are in the
# dingy stable" instead of just "the dingy stable" — defeats _LEADING_PREP_RE
# above, since that regex is anchored at the very start of the string and
# "You are in ..." starts with "You", not "in". Strip the narrator's own
# framing before the existing preposition/article logic runs.
_NARRATOR_PREFIX_RE = re.compile(
    r"^you\s+(?:go\s+\w+\s+and\s+)?are\s+(?:in|on|at|beside)\s+", re.IGNORECASE
)
_NARRATOR_PREFIX_OUTSIDE_RE = re.compile(
    r"^you\s+(?:go\s+\w+\s+and\s+)?are\s+(?=(?:outside|inside)\b)", re.IGNORECASE
)
# Bug 45 disambiguation suffix appended to a node id when the same display name
# is reused for a physically distinct room (e.g. "Alder Clump #2").
_DISAMBIGUATION_RE = re.compile(r" #(\d+)$")
# Bug 78: floor on the existing-node side of resolve_room_name's fuzzy
# containment match.
_MIN_FUZZY_MATCH_CHARS = 4

# P005/P006: "in"/"out" are real navigable directions.
DIRECTIONS = {"north", "south", "east", "west", "up", "down", "ne", "nw", "se", "sw", "in", "out"}

REVERSE = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "up": "down", "down": "up",
    "ne": "sw", "sw": "ne",
    "nw": "se", "se": "nw",
    "in": "out", "out": "in",
}

# LLM sometimes returns full direction names; normalize to the abbreviated form
# used throughout the codebase so BFS vectors and placeholder keys stay consistent.
DIRECTION_NORMALIZE = {
    "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    "inside": "in", "outside": "out",
    "downwards": "down", "upwards": "up",
}


def normalize_direction(word):
    """Return a lowercased, normalized direction alias for comparisons."""
    word = (word or "").strip().lower()
    return DIRECTION_NORMALIZE.get(word, word)


def _require_multidigraph(graph):
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("world_graph must be a networkx.MultiDiGraph")


def short_room_name(name):
    """Truncate at the first comma or semicolon and strip trailing period."""
    return re.split(r"[,;]", name, maxsplit=1)[0].rstrip(".")


def _strip_narrator_prefix(name):
    """Strip a leading "You (go north and) are ..." narrator preamble (bug 78)."""
    name = _NARRATOR_PREFIX_RE.sub("", name)
    return _NARRATOR_PREFIX_OUTSIDE_RE.sub("", name)


def normalize_room(name):
    name = short_room_name(name)
    name = _strip_narrator_prefix(name)
    name = _LEADING_PREP_RE.sub("", name)
    return _ARTICLE_RE.sub("", name).strip().lower()


def canonicalize_room(name):
    """Return a clean short node name for storage."""
    name = short_room_name(name)
    name = _strip_narrator_prefix(name)
    name = _LEADING_PREP_RE.sub("", name)
    name = _LEADING_ARTICLE_RE.sub("", name)
    return name.strip()


def base_room_name(node):
    """Strip a bug-45 disambiguation suffix (' #2') to recover the base name."""
    return _DISAMBIGUATION_RE.sub("", node)


def known_exits(graph, node):
    """Return the set of exit directions already recorded as edges from node."""
    _require_multidigraph(graph)
    exits = set()
    for _, _, data in graph.edges(node, data=True):
        label = data.get("label", "")
        if label:
            exits.update(label.split("/"))
    return frozenset(exits)


def resolve_room_name(graph, room_name, exits=None):
    """Return the graph node room_name should resolve to."""
    _require_multidigraph(graph)
    target = normalize_room(room_name)
    candidates = [
        node for node in graph.nodes
        if not node.startswith("Unknown") and normalize_room(base_room_name(node)) == target
    ]
    if not candidates:
        target_words = set(target.split())
        candidates = [
            node for node in graph.nodes
            if not node.startswith("Unknown")
            and len((base_norm := normalize_room(base_room_name(node))).replace(" ", "")) >= _MIN_FUZZY_MATCH_CHARS
            and set(base_norm.split()) <= target_words
        ]
    if not candidates:
        return canonicalize_room(room_name)

    if not exits:
        return candidates[0]

    exits_set = frozenset(DIRECTION_NORMALIZE.get(d.lower(), d.lower()) for d in exits)
    compatible = [
        node for node in candidates
        if not known_exits(graph, node) or not exits_set.isdisjoint(known_exits(graph, node))
    ]
    if compatible:
        return compatible[0]

    base = base_room_name(candidates[0])
    existing_suffixes = [
        int(m.group(1)) for c in candidates if (m := _DISAMBIGUATION_RE.search(c))
    ]
    return f"{base} #{max(existing_suffixes, default=1) + 1}"


def update_graph(state, room_name, exits, previous_room, action):
    """Add the current room and its exits to the world graph."""
    if not room_name:
        return
    graph = state["world_graph"]
    _require_multidigraph(graph)
    exits = [normalize_direction(d) for d in exits]
    if room_name not in graph:
        graph.add_node(room_name)

    normalized_action = normalize_direction(action) if action else ""
    reverse_action = REVERSE.get(normalized_action, "")
    if previous_room and previous_room != room_name and reverse_action:
        for placeholder in (
            f"Unknown ({normalized_action} from {previous_room})",
            f"Unknown ({reverse_action} from {room_name})",
        ):
            if graph.has_node(placeholder):
                graph.remove_node(placeholder)
        existing_labels = {
            d.get("label")
            for d in (graph.get_edge_data(previous_room, room_name) or {}).values()
        }
        if normalized_action not in existing_labels:
            graph.add_edge(previous_room, room_name, label=normalized_action)

    for direction in exits:
        existing_labels = {d.get("label", "") for _, _, d in graph.edges(room_name, data=True)}
        if direction in existing_labels:
            continue
        if direction == reverse_action and previous_room and previous_room != room_name:
            graph.add_edge(room_name, previous_room, label=direction)
            continue
        target_node = f"Unknown ({direction} from {room_name})"
        if not graph.has_edge(room_name, target_node):
            graph.add_edge(room_name, target_node, label=direction)


def get_next_move_to_target(state, target_room):
    """Return the next movement command toward target_room by shortest path."""
    graph = state["world_graph"]
    _require_multidigraph(graph)
    if target_room == state["current_room"]:
        return None
    try:
        path = nx.shortest_path(graph, source=state["current_room"], target=target_room)
        if len(path) > 1:
            edge_data = graph.get_edge_data(state["current_room"], path[1])
            if not edge_data:
                return None
            return next(iter(edge_data.values())).get("label")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def nav_command(state, target, fast=True):
    """Return native-nav command if configured/eligible, else graph step nav."""
    template = config.fast_nav_command if fast else config.full_nav_command
    if (
        template
        and target in state.get("visited_rooms", set())
        and target not in state.get("nav_blacklist", set())
    ):
        return template.format(target=target)
    return get_next_move_to_target(state, target)


def mark_edge_futile(state, from_room, direction):
    """Mark a direction from a room as permanently futile."""
    graph = state["world_graph"]
    _require_multidigraph(graph)
    direction = normalize_direction(direction)
    state["futile_edges"].add((from_room, direction))
    for _, _, data in graph.edges(from_room, data=True):
        if data.get("label") == direction:
            data["futile"] = True
            break
