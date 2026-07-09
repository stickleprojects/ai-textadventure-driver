import json
import re
from functools import lru_cache

from game_config import config

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt — split into a static system part (cacheable by cloud providers) and
# a tiny dynamic user part (action + game output only). DeepSeek and other
# providers with prefix caching save tokens on every call after the first.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
Analyze the text adventure game output and extract environment data in strict JSON.
Track inventory additions, learned magic, and physical/magical blockers (anomalies).

Useful notes:
- If the response has "Exits lead in all directions" or "Exits lead in every direction", treat it as a special case and set exits to ["north", "south", "east", "west", "up", "down"].
{hints_block}
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
- If an NPC gives, hands, or offers you an item ("Denzyl gives you a spear"), put it in
  "received_from_npc" — do NOT also put it in "added_to_inventory".
- If an NPC takes, steals, snatches, or grabs an item away from you ("The troll snatches the
  gold plate from you"), put it in "taken_by_npc". Only report items you were actually holding —
  not items an NPC takes from someone/something else, or from the room.
- If the game says a specific obstacle is preventing movement ("You are blocked by the
  drawbridge", "The gate bars your way"), put it in "blocked_by". This is distinct from a plain
  "you can't do that" — only use it when a specific obstacle is named as the reason.

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
    "resolved_anomalies": ["list of targets that are no longer blocked"],
    "received_from_npc": [{{"item": "item name", "npc": "npc name"}}],
    "taken_by_npc": [{{"item": "item name", "npc": "npc name"}}],
    "blocked_by": [{{"obstacle": "obstacle name", "blocking": "what it's blocking, e.g. 'route to castle'"}}]
}}

Examples:
Action: "take sword"  Output: "Taken."  → {{"added_to_inventory": ["sword"]}}
Action: "look"  Output: "Exits: north, east."  → {{"exits": ["north", "east"]}}
Action: "ask denzyl for help"  Output: "Denzyl gives you a spear."  → {{"received_from_npc": [{{"item": "spear", "npc": "Denzyl"}}]}}
Action: "north"  Output: "You are blocked by the drawbridge."  → {{"blocked_by": [{{"obstacle": "drawbridge", "blocking": "route north"}}]}}
"""


def _build_system_prompt():
    """Build the static system prompt, injecting any configured hints."""
    hints_block = ""
    if config.hints:
        lines = "\n".join(f"- {h}" for h in config.hints)
        hints_block = f"\nStrategy hints from the user (apply when relevant):\n{lines}\n"
    return _SYSTEM_PROMPT_TEMPLATE.format(hints_block=hints_block)


def _build_user_message(action_taken, text):
    return f'Action taken: "{action_taken}"\nGame Output: "{text}"\nJSON:'


# ---------------------------------------------------------------------------
# Adapters — both expose __call__(system_prompt, user_message) and return:
#   {"choices": [{"text": "..."}], "usage": {"prompt_tokens": N, "completion_tokens": N}}
# ---------------------------------------------------------------------------

class LocalLLMAdapter:
    """Wraps llama_cpp.Llama. Concatenates system+user into a single completion prompt."""

    def __init__(self, model):
        self._model = model

    def __call__(self, system_prompt, user_message):
        full_prompt = system_prompt + "\n" + user_message
        response = self._model(full_prompt, max_tokens=250, stop=["\n\n"], echo=False)
        usage = response.get("usage", {})
        return {
            "choices": [{"text": response["choices"][0]["text"]}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }


class CloudLLMAdapter:
    """Wraps any OpenAI-compatible client (DeepSeek, OpenAI, etc.) as a chat model.

    Set json_mode=True for providers that support response_format=json_object
    (DeepSeek-chat and OpenAI gpt-4o+ both do). This guarantees valid JSON output
    and eliminates parse failures without needing the retry loop.
    """

    def __init__(self, client, model, json_mode=False):
        self._client = client
        self._model = model
        self._json_mode = json_mode

    def __call__(self, system_prompt, user_message):
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
        )
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return {
            "choices": [{"text": response.choices[0].message.content}],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }


# ---------------------------------------------------------------------------
# Tool-calling adapters — used by agent_tools.py's tool-calling main loop
# (cloud/tool-use-capable providers only; see docs/main_loop_prompt.md).
# Unlike LocalLLMAdapter/CloudLLMAdapter's one-shot __call__, these run a
# multi-turn tool-use episode and normalize both backends' wire formats into
# the same shape:
#   start_turn(system_prompt, tools, user_message) -> StepResponse
#   continue_with_results(tool_results) -> StepResponse
#   StepResponse = {"tool_calls": [{"id","name","input"}], "text": str|None,
#                    "usage": {"prompt_tokens": N, "completion_tokens": N}}
# `tools` is the canonical, backend-neutral schema list defined in
# agent_tools.py::TOOL_SCHEMAS (name/description/input_schema — Anthropic's
# native shape; each adapter translates for its own wire format).
# One instance = one step's conversation — a fresh instance per step, no
# history carried across steps (see "small ambient context each turn" in
# docs/main_loop_prompt.md).
# ---------------------------------------------------------------------------

class AnthropicToolAdapter:
    """Wraps anthropic.Anthropic's Messages API for one step's tool-use episode."""

    _MAX_TOKENS = 1024

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._system_prompt = None
        self._tools = None
        self._messages = []

    def start_turn(self, system_prompt, tools, user_message):
        self._system_prompt = system_prompt
        self._tools = tools
        self._messages = [{"role": "user", "content": user_message}]
        return self._call()

    def continue_with_results(self, tool_results):
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                for r in tool_results
            ],
        })
        return self._call()

    def _call(self):
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._MAX_TOKENS,
            system=self._system_prompt,
            tools=self._tools,
            tool_choice={"type": "any"},
            messages=self._messages,
        )
        self._messages.append({
            "role": "assistant",
            "content": [self._block_to_dict(b) for b in response.content],
        })
        tool_calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]
        text = next((b.text for b in response.content if b.type == "text"), None)
        return {
            "tool_calls": tool_calls,
            "text": text,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        }

    @staticmethod
    def _block_to_dict(block):
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        return {"type": "text", "text": block.text}


class OpenAIToolAdapter:
    """Wraps an OpenAI-compatible chat.completions client for one step's tool-use episode."""

    _MAX_TOKENS = 1024

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._tools = None
        self._messages = []

    def start_turn(self, system_prompt, tools, user_message):
        self._tools = [self._to_openai_tool(t) for t in tools]
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._call()

    def continue_with_results(self, tool_results):
        for r in tool_results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": r["id"],
                "content": r["content"],
            })
        return self._call()

    def _call(self):
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._MAX_TOKENS,
            messages=self._messages,
            tools=self._tools,
            tool_choice="required",
        )
        message = response.choices[0].message
        self._messages.append(message.model_dump(exclude_none=True))
        tool_calls = []
        for tc in (message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        return {
            "tool_calls": tool_calls,
            "text": message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    @staticmethod
    def _to_openai_tool(tool):
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }


# ---------------------------------------------------------------------------
# Factory functions — cached so app.py's Streamlit reruns (and repeated calls
# from anywhere else) reuse the same client/model instead of reconnecting or
# reloading every time. Plain functools.lru_cache rather than
# @st.cache_resource: this module has no other Streamlit dependency, and a
# script/test process calling these doesn't need Streamlit's caching UI
# (cache-clear button, cross-session sharing) — just memoization by args.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_llm(model_path):
    """Load the local llama.cpp model wrapped in LocalLLMAdapter. Returns None if unavailable."""
    if not LLAMA_AVAILABLE:
        return None
    return LocalLLMAdapter(
        Llama(model_path=model_path, n_ctx=2048, n_threads=4, use_mlock=False, verbose=False)
    )


@lru_cache(maxsize=None)
def load_cloud_llm(provider, model, api_key, base_url=None, json_mode=True):
    """Load a cloud LLM adapter.

    provider  — label only (e.g. "deepseek", "openai"); selects sensible defaults
    model     — model ID (e.g. "deepseek-chat", "gpt-4o-mini")
    api_key   — provider API key
    base_url  — override endpoint; defaults: deepseek→api.deepseek.com, openai→default
    json_mode — request JSON output format (supported by deepseek-chat and gpt-4o+)
    """
    if not OPENAI_AVAILABLE:
        return None
    _DEFAULT_URLS = {
        "deepseek": "https://api.deepseek.com",
    }
    resolved_url = base_url or _DEFAULT_URLS.get(provider)
    kwargs = {"api_key": api_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url
    client = OpenAI(**kwargs)
    return CloudLLMAdapter(client, model, json_mode=json_mode)


@lru_cache(maxsize=None)
def load_anthropic_tool_llm(model, api_key):
    """Load an AnthropicToolAdapter for the tool-calling main loop. Returns None if unavailable."""
    if not ANTHROPIC_AVAILABLE:
        return None
    return AnthropicToolAdapter(anthropic.Anthropic(api_key=api_key), model)


@lru_cache(maxsize=None)
def load_openai_tool_llm(provider, model, api_key, base_url=None):
    """Load an OpenAIToolAdapter for the tool-calling main loop (DeepSeek, OpenAI, etc.).

    provider  — label only; selects sensible defaults (same map as load_cloud_llm)
    model     — model ID (e.g. "deepseek-chat", "gpt-4o-mini")
    api_key   — provider API key
    base_url  — override endpoint; defaults: deepseek→api.deepseek.com, openai→default
    """
    if not OPENAI_AVAILABLE:
        return None
    _DEFAULT_URLS = {
        "deepseek": "https://api.deepseek.com",
    }
    resolved_url = base_url or _DEFAULT_URLS.get(provider)
    kwargs = {"api_key": api_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url
    return OpenAIToolAdapter(OpenAI(**kwargs), model)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_knowledge(text, action_taken, llm_instance):
    """Extract structured state changes from raw game output using the configured LLM."""
    if not llm_instance:
        return {}

    system_prompt = _build_system_prompt()
    user_message = _build_user_message(action_taken, text)
    trace = {"system": system_prompt, "user": user_message, "attempts": []}

    total_input = 0
    total_output = 0
    for _ in range(3):
        response = llm_instance(system_prompt, user_message)
        usage = response.get("usage", {})
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)
        output_text = response["choices"][0]["text"].strip()
        attempt = {"output": output_text, "parsed": False}
        try:
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                if result:
                    for field in ("exits", "objects", "npcs", "added_to_inventory",
                                  "learned_spells", "anomalies", "resolved_anomalies",
                                  "received_from_npc", "taken_by_npc", "blocked_by"):
                        if field in result and not isinstance(result[field], list):
                            result[field] = []
                    attempt["parsed"] = True
                    trace["attempts"].append(attempt)
                    result["_usage"] = {"input_tokens": total_input, "output_tokens": total_output}
                    result["_trace"] = trace
                    return result
        except json.JSONDecodeError:
            pass
        trace["attempts"].append(attempt)
    return {}
