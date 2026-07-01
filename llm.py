import json
import re

import streamlit as st

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


@st.cache_resource
def load_llm(model_path):
    if not LLAMA_AVAILABLE:
        return None
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4, use_mlock=False, verbose=False)


def extract_knowledge(text, action_taken, llm_instance):
    """Uses the local LLM to extract structured state changes from raw game output."""
    if not llm_instance:
        return {}

    prompt = f"""
    Analyze the text adventure game output and extract environment data in strict JSON.
    Track inventory additions, learned magic, and physical/magical blockers (anomalies).

    Useful notes:
    - If the response has "Exits lead in all directions" or "Exits lead in every direction", treat it as a special case and set exits to ["north", "south", "east", "west", "up", "down"].

    IMPORTANT distinctions:
    - "objects" are inanimate items only (sword, key, stone, pool, door, chest...)
    - "npcs" are living creatures or characters (horse, knight, orc, guard, man, woman...)
    - Never put a living creature in "objects". Never put an inanimate item in "npcs".
    - "you can see X" means X is in the current room — add to "objects" if inanimate, "npcs" if living. It does NOT mean X is in your inventory.
    - Only add to "added_to_inventory" if the game explicitly confirms the item was taken
      (e.g. "Taken.", "You pick up the...", "You take the..."). Seeing an item does not mean it is held.
    - When the action is "take <item>" and the response is a terse confirmation ("Taken.", "OK."),
      set added_to_inventory to ["<item>"] — use the item name from the action, not the response.
    - Always extract exits from phrases like "Exits: north, east" or "you can go north".
    - Only set "room" if the game output explicitly names a location (e.g. "You are in the Great Hall",
      "Dungeon Entrance"). If the response is terse ("Taken.", "OK.", "You can't do that.") or describes
      an object/action without naming a place, set "room" to null. Never use an item name, NPC name,
      direction, or vague phrase ("current location", "unknown") as the room value.
    - Copy the room description VERBATIM from the game text — do not paraphrase, drop articles, or
      shorten it. For example: "you go north and are outside a cave in a juniper scrubland" →
      room = "outside a cave in a juniper scrubland" (not "cave in juniper scrubland").
    - A room must be a named place: "Alder Clump", "Castle Entrance", "Dark Corridor". It must NOT be:
      a direction ("northeast", "propet_northeast"), a character name ("Denzyl", "troll"),
      an object name ("flagpole", "rubbish"), or a bare descriptor ("dark"). If in doubt, set null.
      Note: compound descriptions like "outside a cave" or "top of the hill" ARE valid room names.
    - If examining an object reveals another distinct item (e.g. "fastened to it is a halyard",
      "inside is a key", "a note is attached"), include that item in "objects" too.

    Schema required:
    {{
        "room": "string (current location, only if explicitly named in output)",
        "exits": ["list of directions"],
        "objects": ["list of inanimate items seen (not creatures)"],
        "npcs": ["list of living creatures or characters seen"],
        "added_to_inventory": ["list of items explicitly confirmed as taken"],
        "learned_spells": ["list of spells learned"],
        "anomalies": [
            {{"target": "object name", "reason": "why it's blocked", "potential_solution": "item/spell needed"}}
        ],
        "resolved_anomalies": ["list of targets that are no longer blocked"]
    }}

    Examples:
    Action: "take sword"  Output: "Taken."  → {{"added_to_inventory": ["sword"]}}
    Action: "look"  Output: "Exits: north, east."  → {{"exits": ["north", "east"]}}

    Action taken: "{action_taken}"
    Game Output: "{text}"
    JSON:
    """

    total_input = 0
    total_output = 0
    for _ in range(3):
        response = llm_instance(prompt, max_tokens=250, stop=["\n\n"], echo=False)
        usage = response.get("usage", {})
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)
        output_text = response['choices'][0]['text'].strip()
        try:
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                if result:
                    for field in ("exits", "objects", "npcs", "added_to_inventory",
                                  "learned_spells", "anomalies", "resolved_anomalies"):
                        if field in result and not isinstance(result[field], list):
                            result[field] = []
                    result["_usage"] = {"input_tokens": total_input, "output_tokens": total_output}
                    return result
        except json.JSONDecodeError:
            pass
    return {}
