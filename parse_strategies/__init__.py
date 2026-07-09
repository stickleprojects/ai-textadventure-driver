"""Pluggable game-response parsing, decoupled from deciding.

A ParseStrategy's only job is turning raw game text into structured data —
it never chooses the next command (that's agent.py::determine_next_action
or agent_tools.py's tool-calling decide loop, both unaffected by this
package). apply_parse_result() is the single, shared function both the
legacy and tool-calling paths use to turn a ParseStrategy's output into
state mutations — see base.py's docstring for the full ParseResult schema.

One file per strategy:
    base.py           — ParseStrategy interface + apply_parse_result (shared)
    llm_json_mode.py  — LLMJsonModeParseStrategy (legacy local-model default)
    llm_tool_call.py  — LLMToolCallParseStrategy (cloud tool-calling backends)
    deterministic.py  — DeterministicParseStrategy (plain-Python stub, no LLM)
"""
from .base import (
    ParseStrategy,
    apply_parse_result,
    append_finding,
    resolve_entity_key,
    split_verb_object,
)
from .llm_json_mode import LLMJsonModeParseStrategy
from .llm_tool_call import LLMToolCallParseStrategy
from .deterministic import DeterministicParseStrategy

__all__ = [
    "ParseStrategy",
    "apply_parse_result",
    "append_finding",
    "resolve_entity_key",
    "split_verb_object",
    "LLMJsonModeParseStrategy",
    "LLMToolCallParseStrategy",
    "DeterministicParseStrategy",
]
