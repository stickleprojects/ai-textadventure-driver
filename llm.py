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
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4)


def extract_knowledge(text, action_taken, llm_instance):
    """Uses the local LLM to extract structured state changes from raw game output."""
    if not llm_instance:
        return {}

    prompt = f"""
    Analyze the text adventure game output and extract environment data in strict JSON.
    Track inventory additions, learned magic, and physical/magical blockers (anomalies).

    Schema required:
    {{
        "room": "string (current location)",
        "exits": ["list of directions"],
        "objects": ["list of items seen"],
        "added_to_inventory": ["list of items successfully taken"],
        "learned_spells": ["list of spells learned"],
        "anomalies": [
            {{"target": "object name", "reason": "why it's blocked", "potential_solution": "item/spell needed"}}
        ],
        "resolved_anomalies": ["list of targets that are no longer blocked"]
    }}

    Game Output: "{text}"
    JSON:
    """

    response = llm_instance(prompt, max_tokens=250, stop=["\n\n"], echo=False)
    output_text = response['choices'][0]['text'].strip()

    try:
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}
    except json.JSONDecodeError:
        return {}
