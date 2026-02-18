"""
Tests for modules/jarvis_quotes.py

All functions are pure randomisation from static lists — no external deps.
"""

import pytest
from modules.jarvis_quotes import (
    ACKNOWLEDGMENTS,
    STARTUP_LINES,
    PROCESSING_LINES,
    COMPLETION_LINES,
    ERROR_LINES,
    INTERVIEW_FILLERS,
    CURIOSITY_LINES,
    SHUTDOWN_LINES,
    DYNAMIC_TEMPLATES,
    ALL_QUOTES,
    get_acknowledgment,
    get_startup_line,
    get_processing_line,
    get_completion_line,
    get_error_line,
    get_interview_filler,
    get_random_quote,
    get_curiosity_line,
    get_shutdown_line,
    get_estimated_completion,
    get_power_level,
    get_awake_reminder,
    get_incoming_call,
    get_version_not_ready,
    get_all_cacheable_lines,
)


# ===========================================================================
# List contents
# ===========================================================================

class TestListContents:
    def test_acknowledgments_non_empty(self):
        assert len(ACKNOWLEDGMENTS) > 0

    def test_startup_lines_non_empty(self):
        assert len(STARTUP_LINES) > 0

    def test_processing_lines_non_empty(self):
        assert len(PROCESSING_LINES) > 0

    def test_completion_lines_non_empty(self):
        assert len(COMPLETION_LINES) > 0

    def test_error_lines_non_empty(self):
        assert len(ERROR_LINES) > 0

    def test_interview_fillers_non_empty(self):
        assert len(INTERVIEW_FILLERS) > 0

    def test_curiosity_lines_non_empty(self):
        assert len(CURIOSITY_LINES) > 0

    def test_shutdown_lines_non_empty(self):
        assert len(SHUTDOWN_LINES) > 0

    def test_all_quotes_non_empty(self):
        assert len(ALL_QUOTES) > 0

    def test_all_acknowledgments_are_strings(self):
        for line in ACKNOWLEDGMENTS:
            assert isinstance(line, str) and line.strip()

    def test_all_startup_lines_are_strings(self):
        for line in STARTUP_LINES:
            assert isinstance(line, str) and line.strip()

    def test_dynamic_templates_has_expected_keys(self):
        for key in ["estimated_completion", "power_level", "awake_reminder",
                    "incoming_call", "version_not_ready"]:
            assert key in DYNAMIC_TEMPLATES


# ===========================================================================
# Getter functions
# ===========================================================================

class TestGetters:
    def test_get_acknowledgment_returns_string(self):
        assert isinstance(get_acknowledgment(), str)

    def test_get_acknowledgment_in_list(self):
        assert get_acknowledgment() in ACKNOWLEDGMENTS

    def test_get_startup_line_in_list(self):
        assert get_startup_line() in STARTUP_LINES

    def test_get_processing_line_in_list(self):
        assert get_processing_line() in PROCESSING_LINES

    def test_get_completion_line_in_list(self):
        assert get_completion_line() in COMPLETION_LINES

    def test_get_error_line_in_list(self):
        assert get_error_line() in ERROR_LINES

    def test_get_interview_filler_in_list(self):
        assert get_interview_filler() in INTERVIEW_FILLERS

    def test_get_random_quote_in_all_quotes(self):
        assert get_random_quote() in ALL_QUOTES

    def test_get_curiosity_line_in_list(self):
        assert get_curiosity_line() in CURIOSITY_LINES

    def test_get_shutdown_line_in_list(self):
        assert get_shutdown_line() in SHUTDOWN_LINES

    def test_getters_return_non_empty(self):
        for fn in [get_acknowledgment, get_startup_line, get_processing_line,
                   get_completion_line, get_error_line, get_interview_filler,
                   get_random_quote, get_curiosity_line, get_shutdown_line]:
            result = fn()
            assert result and result.strip(), f"{fn.__name__} returned empty string"


# ===========================================================================
# Dynamic template helpers
# ===========================================================================

class TestDynamicTemplates:
    def test_get_estimated_completion_includes_hours(self):
        result = get_estimated_completion(3)
        assert "3" in result

    def test_get_estimated_completion_returns_string(self):
        assert isinstance(get_estimated_completion(5), str)

    def test_get_power_level_includes_percent(self):
        result = get_power_level(75)
        assert "75" in result

    def test_get_power_level_returns_string(self):
        assert isinstance(get_power_level(100), str)

    def test_get_awake_reminder_includes_hours(self):
        result = get_awake_reminder(48)
        assert "48" in result

    def test_get_incoming_call_includes_contact(self):
        result = get_incoming_call("Miss Potts")
        assert "Miss Potts" in result

    def test_get_version_not_ready_includes_version(self):
        result = get_version_not_ready("2.0")
        assert "2.0" in result

    def test_template_zero_hours(self):
        result = get_estimated_completion(0)
        assert "0" in result


# ===========================================================================
# get_all_cacheable_lines
# ===========================================================================

class TestGetAllCacheableLines:
    def test_returns_list(self):
        assert isinstance(get_all_cacheable_lines(), list)

    def test_non_empty(self):
        assert len(get_all_cacheable_lines()) > 0

    def test_all_items_are_strings(self):
        for line in get_all_cacheable_lines():
            assert isinstance(line, str)

    def test_all_acknowledgments_included(self):
        cacheable = set(get_all_cacheable_lines())
        for line in ACKNOWLEDGMENTS:
            assert line in cacheable, f"Acknowledgment missing from cacheable: {line!r}"

    def test_all_interview_fillers_included(self):
        cacheable = set(get_all_cacheable_lines())
        for line in INTERVIEW_FILLERS:
            assert line in cacheable

    def test_long_movie_quotes_excluded(self):
        cacheable = get_all_cacheable_lines()
        for line in cacheable:
            if line in ALL_QUOTES:
                assert len(line) < 60, f"Long movie quote in cache: {line!r}"

    def test_short_movie_quotes_included(self):
        short_quotes = [q for q in ALL_QUOTES if len(q) < 60]
        cacheable = set(get_all_cacheable_lines())
        for q in short_quotes:
            assert q in cacheable, f"Short quote missing from cacheable: {q!r}"

    def test_result_is_sorted(self):
        lines = get_all_cacheable_lines()
        assert lines == sorted(lines)

    def test_no_duplicates(self):
        lines = get_all_cacheable_lines()
        assert len(lines) == len(set(lines))
