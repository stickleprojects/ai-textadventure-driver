"""LLMToolCallParseStrategy — the tool-calling-backend parse strategy."""
from .base import ParseStrategy

_PARSE_SYSTEM_PROMPT = (
    "Extract structured information from a text adventure game's response. "
    "Call parse_game_response with what actually happened — see the tool's "
    "schema for what each field means and how strictly it's checked."
)

PARSE_TOOL_SCHEMA = {
    "name": "parse_game_response",
    "description": "Parse what happened in the game response you were just shown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action_result": {
                "type": "object",
                "properties": {
                    "succeeded": {"type": "boolean"},
                    "reason_if_failed": {"type": ["string", "null"]},
                },
                "required": ["succeeded"],
            },
            "room_quote": {
                "type": ["string", "null"],
                "description": (
                    "A VERBATIM substring of the response naming the location — copy it exactly, "
                    "don't summarize or paraphrase it. Quote ONLY the room's own name/description, "
                    "not the surrounding narration: if the response says \"You are in the dingy "
                    "stable\", quote \"the dingy stable\" or \"dingy stable\" (either is fine — "
                    "with or without its leading article), not the whole sentence. Use null if the "
                    "response is terse, describes an object/action without naming a place, or "
                    "you'd have to infer or guess the location rather than read it directly. Bug 74: "
                    "a model once reported a specific, plausible-sounding room for a response that "
                    "only said \"You own nothing at all!\" — nothing it wrote was actually in the "
                    "text. A value here that isn't a literal substring of the response is discarded."
                ),
            },
            "exits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only directions literally mentioned in the response — not inferred from \"exits lead in all directions\" or similar.",
            },
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Inanimate items visible — only ones actually named in the response, not ones you'd plausibly expect to be there.",
            },
            "npcs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Living creatures/characters visible — only ones actually named in the response.",
            },
            "inventory_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "change": {"type": "string", "enum": ["gained", "lost"]},
                        "cause": {"type": "string"},
                    },
                    "required": ["item", "change"],
                },
            },
            "notable_events": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Anything else worth remembering, in plain language",
            },
        },
        "required": ["action_result"],
    },
}


class LLMToolCallParseStrategy(ParseStrategy):
    """For tool-calling-capable backends (Anthropic/OpenAI adapters, see
    llm.py). A single tool-call episode — parse_game_response is the only
    schema registered, so the forced tool_choice both adapters already set
    (llm.py) guarantees exactly one call, no dispatch loop needed."""

    def __init__(self, tool_adapter):
        self._tool_adapter = tool_adapter

    def parse(self, response_text, action_taken):
        user_message = f'Action taken: "{action_taken}"\nGame response: "{response_text}"'
        step_response = self._tool_adapter.start_turn(_PARSE_SYSTEM_PROMPT, [PARSE_TOOL_SCHEMA], user_message)
        tool_calls = step_response.get("tool_calls", [])
        result = dict(tool_calls[0]["input"]) if tool_calls else {}
        usage = step_response.get("usage", {})
        result["_usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        return result
