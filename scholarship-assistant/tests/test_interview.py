"""
Tests for scholarship-assistant/server/interview.py

Tests only the pure helper functions that don't require voice I/O:
  _is_skip, _is_list_field, _parse_list_response
"""

import sys
import types
from pathlib import Path

# ── Stub only the hardware/API modules that are unavailable in test env ──
# We do NOT stub the real modules.jarvis_quotes — let Python find the real
# package (the repo root is on sys.path when running pytest from the repo).

_el = types.ModuleType("elevenlabs")
_el.play = lambda *a, **kw: None
sys.modules["elevenlabs"] = _el

_elc = types.ModuleType("elevenlabs.client")
_elc.ElevenLabs = object
sys.modules["elevenlabs.client"] = _elc

_rtstt = types.ModuleType("RealtimeSTT")
_rtstt.AudioToTextRecorder = object
sys.modules["RealtimeSTT"] = _rtstt

# modules.tts_cache uses elevenlabs at import time — stub it so generate_with_cache
# doesn't try to connect to the API when voice.py imports it.
_tc = types.ModuleType("modules.tts_cache")
_tc.generate_with_cache = lambda *a, **kw: b""
sys.modules["modules.tts_cache"] = _tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.interview import (
    _is_skip,
    _is_list_field,
    _parse_list_response,
    SKIP_PHRASES,
    INTERVIEW_QUESTIONS,
)


# ===========================================================================
# _is_skip
# ===========================================================================

class TestIsSkip:
    def test_skip_literal(self):
        assert _is_skip("skip") is True

    def test_pass_literal(self):
        assert _is_skip("pass") is True

    def test_next_literal(self):
        assert _is_skip("next") is True

    def test_no_literal(self):
        assert _is_skip("no") is True

    def test_nope_literal(self):
        assert _is_skip("nope") is True

    def test_skip_it(self):
        assert _is_skip("skip it") is True

    def test_pass_on_that(self):
        assert _is_skip("pass on that") is True

    def test_id_rather_not(self):
        assert _is_skip("i'd rather not") is True

    def test_case_insensitive_skip(self):
        assert _is_skip("SKIP") is True

    def test_case_insensitive_pass(self):
        assert _is_skip("Pass") is True

    def test_trailing_period_stripped(self):
        assert _is_skip("skip.") is True

    def test_leading_trailing_whitespace(self):
        assert _is_skip("  skip  ") is True

    def test_not_a_skip_phrase_real_answer(self):
        assert _is_skip("John Smith") is False

    def test_not_a_skip_phrase_sentence(self):
        assert _is_skip("I would like to answer this") is False

    def test_not_a_skip_phrase_empty_string(self):
        assert _is_skip("") is False

    def test_all_defined_skip_phrases(self):
        for phrase in SKIP_PHRASES:
            assert _is_skip(phrase), f"Expected '{phrase}' to be a skip phrase"


# ===========================================================================
# _is_list_field
# ===========================================================================

class TestIsListField:
    LIST_FIELDS = [
        "education_current.majors",
        "education_current.minors",
        "disability.disability_types",
        "disability.specific_conditions",
        "professional.skills",
        "professional.programming_languages",
        "professional.technologies",
        "professional.professional_memberships",
        "extracurricular.volunteer_work",
        "extracurricular.leadership_roles",
        "extracurricular.awards_honors",
        "extracurricular.organizational_memberships",
        "financial.current_aid",
    ]

    def test_all_list_fields_recognized(self):
        for field in self.LIST_FIELDS:
            assert _is_list_field(field), f"Expected '{field}' to be a list field"

    def test_scalar_fields_not_list(self):
        scalar_fields = [
            "personal.full_legal_name",
            "personal.email",
            "personal.phone",
            "education_current.university_name",
            "education_current.current_gpa",
            "financial.fafsa_filed",
        ]
        for field in scalar_fields:
            assert not _is_list_field(field), f"Expected '{field}' NOT to be a list field"

    def test_unknown_key_returns_false(self):
        assert _is_list_field("nonexistent.key") is False

    def test_empty_key_returns_false(self):
        assert _is_list_field("") is False

    def test_partial_match_not_list(self):
        # A prefix that partially matches a list key should NOT match
        assert _is_list_field("professional") is False


# ===========================================================================
# _parse_list_response
# ===========================================================================

class TestParseListResponse:
    def test_comma_separated(self):
        result = _parse_list_response("math, physics, chemistry")
        assert result == ["math", "physics", "chemistry"]

    def test_and_separated(self):
        result = _parse_list_response("math and physics and chemistry")
        assert "math" in result
        assert "physics" in result
        assert "chemistry" in result

    def test_mixed_comma_and_and(self):
        result = _parse_list_response("Python, Java and C++")
        assert "Python" in result
        assert "Java" in result
        assert "C++" in result

    def test_single_item(self):
        result = _parse_list_response("Computer Science")
        assert result == ["Computer Science"]

    def test_strips_trailing_period(self):
        result = _parse_list_response("Python, Java.")
        for item in result:
            assert not item.endswith(".")

    def test_strips_whitespace(self):
        result = _parse_list_response("  math ,  physics  ")
        assert all(item == item.strip() for item in result)

    def test_empty_string(self):
        result = _parse_list_response("")
        assert result == []

    def test_only_commas(self):
        result = _parse_list_response(",,,")
        assert result == []

    def test_preserves_multi_word_items(self):
        result = _parse_list_response("machine learning, data science")
        assert "machine learning" in result
        assert "data science" in result


# ===========================================================================
# INTERVIEW_QUESTIONS structure
# ===========================================================================

class TestInterviewQuestionsStructure:
    """Sanity-check the static data to catch typos early."""

    def test_all_questions_are_tuples_of_4(self):
        for q in INTERVIEW_QUESTIONS:
            assert len(q) == 4, f"Question entry has wrong length: {q}"

    def test_all_dot_keys_have_at_least_two_parts(self):
        for dot_key, *_ in INTERVIEW_QUESTIONS:
            assert "." in dot_key, f"Dot key '{dot_key}' has no dot"

    def test_all_questions_are_strings(self):
        for dot_key, question, optional_flag, sensitive_flag in INTERVIEW_QUESTIONS:
            assert isinstance(question, str) and question

    def test_optional_and_sensitive_flags_are_bools(self):
        for dot_key, question, optional_flag, sensitive_flag in INTERVIEW_QUESTIONS:
            assert isinstance(optional_flag, bool)
            assert isinstance(sensitive_flag, bool)

    def test_no_duplicate_dot_keys(self):
        keys = [q[0] for q in INTERVIEW_QUESTIONS]
        assert len(keys) == len(set(keys)), "Duplicate dot keys in INTERVIEW_QUESTIONS"
