"""Unit tests for the deterministic logic in classify_failure_patterns.py.

Matches this codebase's existing convention for Claude-calling scripts
(architect.py, llm_review.py have no tests): the API-calling shell is left
untested; only the pure functions around it are covered here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from classify_failure_patterns import _pattern_is_valid, _slugify


class TestPatternIsValid:
    def test_valid_pattern_matching_response(self):
        assert _pattern_is_valid(r"that'?s too heavy", "That's too heavy.")

    def test_malformed_regex_rejected(self):
        assert not _pattern_is_valid(r"(unclosed", "That's too heavy.")

    def test_valid_regex_not_matching_response_rejected(self):
        assert not _pattern_is_valid(r"nonsense", "That's too heavy.")

    def test_case_insensitive(self):
        assert _pattern_is_valid(r"TOO HEAVY", "that's too heavy.")


class TestSlugify:
    def test_spaces_become_hyphens(self):
        assert _slugify("pile of garbage") == "pile-of-garbage"

    def test_punctuation_stripped(self):
        assert _slugify("Grendel's Lair!!") == "grendel-s-lair"

    def test_truncated_to_40_chars(self):
        long_name = "a" * 100
        assert len(_slugify(long_name)) == 40
