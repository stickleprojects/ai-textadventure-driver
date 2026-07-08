"""LLMJsonModeParseStrategy — the legacy local-model parse strategy."""
import agent

from .base import ParseStrategy


class LLMJsonModeParseStrategy(ParseStrategy):
    """Wraps llm.extract_knowledge unchanged — the legacy path's default,
    exact same prompt/behavior as before this package existed.

    Calls through agent.extract_knowledge (module-attribute lookup, not a
    direct import) rather than importing extract_knowledge itself, so that
    the many existing tests patching "agent.extract_knowledge" keep working
    unchanged — agent.py imports the same name into its own namespace."""

    def __init__(self, llm_instance):
        self._llm_instance = llm_instance

    def parse(self, response_text, action_taken):
        return agent.extract_knowledge(response_text, action_taken, self._llm_instance)
