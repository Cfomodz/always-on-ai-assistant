"""
Tests for scholarship-assistant/server/field_matcher.py

categorize_matches() is pure logic — no API calls required.
match_fields() is tested only via its error-handling path (DeepSeek stubbed).
"""

import sys
import types
from pathlib import Path

# Stub modules.deepseek before importing field_matcher so the OpenAI client
# is never initialised during tests.  Do NOT stub the parent `modules` package
# — the real package must remain importable for other tests in the same run.
_ds = types.ModuleType("modules.deepseek")
_ds.json_prompt = lambda *a, **kw: {}
sys.modules["modules.deepseek"] = _ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from server.field_matcher import categorize_matches
from server.config import AUTO_FILL_THRESHOLD, CONFIRM_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(field_id: str, confidence: float) -> dict:
    return {"field_id": field_id, "profile_key": "personal.email",
            "value": "test@test.com", "confidence": confidence, "reasoning": "test"}


def _make_unmatched(field_id: str, is_essay: bool = False) -> dict:
    return {"field_id": field_id, "label": "Some label",
            "type": "textarea" if is_essay else "text", "is_essay": is_essay}


# ===========================================================================
# categorize_matches
# ===========================================================================

class TestCategorizeMatches:
    def test_high_confidence_goes_to_auto_fill(self):
        result = categorize_matches({"matches": [_make_match("f1", 0.95)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["auto_fill"])
        assert result["confirm"] == []
        assert result["ask"] == []

    def test_at_threshold_auto_fill(self):
        """Exactly at AUTO_FILL_THRESHOLD should go into auto_fill."""
        result = categorize_matches({"matches": [_make_match("f1", AUTO_FILL_THRESHOLD)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["auto_fill"])

    def test_medium_confidence_goes_to_confirm(self):
        confidence = (CONFIRM_THRESHOLD + AUTO_FILL_THRESHOLD) / 2  # e.g. 0.70
        result = categorize_matches({"matches": [_make_match("f1", confidence)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["confirm"])
        assert result["auto_fill"] == []
        assert result["ask"] == []

    def test_at_confirm_threshold(self):
        """Exactly at CONFIRM_THRESHOLD should go into confirm, not ask."""
        result = categorize_matches({"matches": [_make_match("f1", CONFIRM_THRESHOLD)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["confirm"])

    def test_low_confidence_goes_to_ask(self):
        result = categorize_matches({"matches": [_make_match("f1", 0.3)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["ask"])
        assert result["auto_fill"] == []
        assert result["confirm"] == []

    def test_zero_confidence_goes_to_ask(self):
        result = categorize_matches({"matches": [_make_match("f1", 0.0)], "unmatched": []})
        assert result["ask"][0]["field_id"] == "f1"

    def test_essay_goes_to_essay(self):
        result = categorize_matches({"matches": [], "unmatched": [_make_unmatched("essay1", is_essay=True)]})
        assert any(u["field_id"] == "essay1" for u in result["essay"])
        assert result["skip"] == []

    def test_non_essay_unmatched_goes_to_skip(self):
        result = categorize_matches({"matches": [], "unmatched": [_make_unmatched("text1", is_essay=False)]})
        assert any(u["field_id"] == "text1" for u in result["skip"])
        assert result["essay"] == []

    def test_empty_input_returns_empty_buckets(self):
        result = categorize_matches({"matches": [], "unmatched": []})
        assert result == {"auto_fill": [], "confirm": [], "ask": [], "essay": [], "skip": []}

    def test_missing_matches_key_handled(self):
        result = categorize_matches({"unmatched": []})
        assert result["auto_fill"] == []

    def test_missing_unmatched_key_handled(self):
        result = categorize_matches({"matches": []})
        assert result["essay"] == []

    def test_mixed_confidence_split_correctly(self):
        matches = [
            _make_match("high", 0.95),
            _make_match("medium", 0.70),
            _make_match("low", 0.30),
        ]
        result = categorize_matches({"matches": matches, "unmatched": []})
        assert result["auto_fill"][0]["field_id"] == "high"
        assert result["confirm"][0]["field_id"] == "medium"
        assert result["ask"][0]["field_id"] == "low"

    def test_multiple_essays(self):
        unmatched = [_make_unmatched(f"essay{i}", is_essay=True) for i in range(3)]
        result = categorize_matches({"matches": [], "unmatched": unmatched})
        assert len(result["essay"]) == 3

    def test_result_has_all_five_keys(self):
        result = categorize_matches({"matches": [], "unmatched": []})
        for key in ["auto_fill", "confirm", "ask", "essay", "skip"]:
            assert key in result

    def test_confidence_just_below_auto_fill_goes_to_confirm(self):
        confidence = AUTO_FILL_THRESHOLD - 0.01
        result = categorize_matches({"matches": [_make_match("f1", confidence)], "unmatched": []})
        # Should land in confirm (if above CONFIRM_THRESHOLD) or ask
        assert "f1" not in [m["field_id"] for m in result["auto_fill"]]

    def test_confidence_just_below_confirm_goes_to_ask(self):
        confidence = CONFIRM_THRESHOLD - 0.01
        result = categorize_matches({"matches": [_make_match("f1", confidence)], "unmatched": []})
        assert any(m["field_id"] == "f1" for m in result["ask"])


# ===========================================================================
# match_fields — error path (DeepSeek raises, all fields become unmatched)
# ===========================================================================

class TestMatchFieldsErrorPath:
    def test_error_returns_all_unmatched(self):
        import server.field_matcher as fm
        from unittest.mock import patch

        fields = [
            {"id": "email", "label": "Email", "type": "text", "required": True},
            {"id": "essay", "label": "Tell us about yourself", "type": "textarea", "required": False},
        ]

        with patch("server.field_matcher.json_prompt", side_effect=RuntimeError("API down")):
            with patch("server.profile_manager.PROFILE_PATH", Path("/nonexistent/profile.json")):
                result = fm.match_fields(fields, profile={})

        assert result["matches"] == []
        assert len(result["unmatched"]) == 2
        field_ids = {u["field_id"] for u in result["unmatched"]}
        assert "email" in field_ids
        assert "essay" in field_ids

    def test_error_path_textarea_flagged_as_essay(self):
        import server.field_matcher as fm
        from unittest.mock import patch

        fields = [{"id": "essay_box", "label": "Essay", "type": "textarea"}]
        with patch("server.field_matcher.json_prompt", side_effect=RuntimeError("API down")):
            result = fm.match_fields(fields, profile={})

        essay_unmatched = next(u for u in result["unmatched"] if u["field_id"] == "essay_box")
        assert essay_unmatched["is_essay"] is True

    def test_error_path_text_field_not_flagged_as_essay(self):
        import server.field_matcher as fm
        from unittest.mock import patch

        fields = [{"id": "first_name", "label": "First name", "type": "text"}]
        with patch("server.field_matcher.json_prompt", side_effect=RuntimeError("API down")):
            result = fm.match_fields(fields, profile={})

        assert result["unmatched"][0]["is_essay"] is False
