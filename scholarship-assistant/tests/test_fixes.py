"""
Tests covering the three bug fixes addressed in PR review comments.

1. sys.path fix in main.py for direct CLI startup
2. Dedup scoring in history.py for partial data (one identifier missing)
3. Non-string value coercion in the userscript setFieldValue (tested via
   documented behaviour since the userscript runs in a browser context)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Resolve project root and add scholarship-assistant/ to sys.path so that
# "from server.X import ..." works in tests, mirroring the corrected fix.
# ---------------------------------------------------------------------------
SCHOLARSHIP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCHOLARSHIP_ROOT))


# ===========================================================================
# Fix 1 – sys.path points to scholarship-assistant/, not repo root
# ===========================================================================

class TestSysPathFix(unittest.TestCase):
    """Verify that main.py inserts the scholarship-assistant/ directory
    (i.e. the parent of the server/ package) rather than the repo root."""

    def test_path_insert_points_to_scholarship_root(self):
        """The path inserted in main.py must be the parent of server/__init__.py."""
        main_py = SCHOLARSHIP_ROOT / "server" / "main.py"
        source = main_py.read_text()

        # The corrected insertion uses one ".." (parent of server/ = scholarship-assistant/)
        self.assertIn(
            'os.path.join(os.path.dirname(__file__), "..")',
            source,
            "main.py should insert the scholarship-assistant/ directory (one level up "
            "from server/) so that 'from server.X import ...' resolves correctly.",
        )

        # The old, broken insertion used two ".." and should not be present anymore
        self.assertNotIn(
            'os.path.join(os.path.dirname(__file__), "..", "..")',
            source,
            "main.py must NOT insert the repo root; that leaves 'server' unresolvable "
            "when the file is run directly.",
        )

    def test_scholarship_root_contains_server_package(self):
        """SCHOLARSHIP_ROOT must contain a 'server' package (has __init__.py)."""
        server_pkg = SCHOLARSHIP_ROOT / "server" / "__init__.py"
        self.assertTrue(
            server_pkg.exists(),
            f"Expected a server/__init__.py at {server_pkg} so that "
            "'from server.X import ...' resolves when SCHOLARSHIP_ROOT is on sys.path.",
        )

    def test_repo_root_does_not_contain_server_package(self):
        """The repo root must NOT contain a top-level 'server' package.
        This confirms that inserting the repo root (the old bug) would fail."""
        repo_root = SCHOLARSHIP_ROOT.parent
        server_at_repo_root = repo_root / "server" / "__init__.py"
        self.assertFalse(
            server_at_repo_root.exists(),
            "There must be no top-level 'server' package at the repo root; "
            "inserting the repo root (the old path) would NOT resolve server imports.",
        )


# ===========================================================================
# Fix 2 – Dedup handles partial data (one identifier missing)
# ===========================================================================

def _check_duplicate_logic(
    scholarship_name: str,
    organization: str,
    history: list,
    threshold: float = 0.85,
):
    """
    Pure-Python re-implementation of the corrected check_duplicate logic,
    isolated from file I/O so we can test the scoring without touching disk.
    """
    from thefuzz import fuzz

    if not scholarship_name and not organization:
        return None

    for record in history:
        name_score = fuzz.token_sort_ratio(
            scholarship_name.lower(), record.get("scholarship_name", "").lower()
        )
        org_score = fuzz.token_sort_ratio(
            organization.lower(), record.get("organization", "").lower()
        )

        has_name = bool(scholarship_name and record.get("scholarship_name"))
        has_org = bool(organization and record.get("organization"))

        if has_name and has_org:
            combined_score = (name_score * 0.6 + org_score * 0.4) / 100.0
        elif has_name:
            combined_score = name_score / 100.0
        elif has_org:
            combined_score = org_score / 100.0
        else:
            combined_score = 0.0

        if combined_score >= threshold:
            return record

    return None


class TestDedupPartialData(unittest.TestCase):
    """check_duplicate() must still detect duplicates when only one of
    scholarship_name / organization is available."""

    def test_both_identifiers_present_detects_duplicate(self):
        history = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford University"}]
        result = _check_duplicate_logic("Rhodes Scholarship", "Oxford University", history)
        self.assertIsNotNone(result, "Should detect duplicate when both fields match.")

    def test_name_only_detects_duplicate(self):
        """When organization is empty, use only scholarship_name for scoring."""
        history = [{"scholarship_name": "Rhodes Scholarship", "organization": ""}]
        result = _check_duplicate_logic("Rhodes Scholarship", "", history)
        self.assertIsNotNone(result, "Should detect duplicate using name only when org is missing.")

    def test_org_only_detects_duplicate(self):
        """When scholarship_name is empty, use only organization for scoring."""
        history = [{"scholarship_name": "", "organization": "Gates Foundation"}]
        result = _check_duplicate_logic("", "Gates Foundation", history)
        self.assertIsNotNone(result, "Should detect duplicate using org only when name is missing.")

    def test_old_formula_would_fail_name_only(self):
        """Demonstrate that the old fixed-weight formula caused false negatives."""
        from thefuzz import fuzz
        # name matches perfectly; org is absent in the history record
        name_score = fuzz.token_sort_ratio("Rhodes Scholarship", "Rhodes Scholarship")
        org_score = fuzz.token_sort_ratio("Oxford University", "")
        old_combined = (name_score * 0.6 + org_score * 0.4) / 100.0
        self.assertLess(
            old_combined, 0.85,
            "Confirms the old formula misses duplicates when org is absent in history."
        )

    def test_new_formula_succeeds_where_old_failed(self):
        """Verify the new logic succeeds on the case where the old formula failed."""
        # History has a record with org absent; query supplies org
        history = [{"scholarship_name": "Rhodes Scholarship", "organization": ""}]
        result = _check_duplicate_logic("Rhodes Scholarship", "Oxford University", history)
        # Only scholarship_name is compared (has_org is False because record.org is empty)
        self.assertIsNotNone(result, "New formula should catch the name-only duplicate.")

    def test_no_identifiers_returns_none(self):
        """When both identifiers are empty, return None immediately."""
        history = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford"}]
        result = _check_duplicate_logic("", "", history)
        self.assertIsNone(result, "Should return None with no identifiers to match on.")

    def test_no_match_returns_none(self):
        """Different scholarships should not be flagged as duplicates."""
        history = [{"scholarship_name": "Fulbright Scholarship", "organization": "US Department of State"}]
        result = _check_duplicate_logic("Rhodes Scholarship", "Oxford University", history)
        self.assertIsNone(result, "Different scholarships should not match.")

    def test_history_py_source_has_partial_data_fix(self):
        """Confirm history.py source contains the partial-data branching logic."""
        history_py = SCHOLARSHIP_ROOT / "server" / "history.py"
        source = history_py.read_text()
        self.assertIn("has_name", source, "history.py should have partial-data branching logic.")
        self.assertIn("has_org", source, "history.py should have partial-data branching logic.")
        self.assertIn("elif has_name:", source)
        self.assertIn("elif has_org:", source)


# ===========================================================================
# Fix 3 – setFieldValue coerces value to string for select / radio matching
# ===========================================================================

class TestSetFieldValueStringCoercion(unittest.TestCase):
    """Verify that the userscript source coerces value to String() before
    calling .toLowerCase() in the select and radio branches.

    The userscript runs in a browser; these tests inspect the source code
    to confirm the coercion is in place, and also unit-test the logic in a
    simulated Python equivalent of the matching algorithm.
    """

    USERSCRIPT_PATH = (
        SCHOLARSHIP_ROOT / "userscript" / "scholarship-assistant.user.js"
    )

    def _get_fn_body(self):
        source = self.USERSCRIPT_PATH.read_text()
        fn_start = source.index("function setFieldValue(")
        fn_end = source.index("\n  }", fn_start) + 4
        return source[fn_start:fn_end]

    def test_select_branch_coerces_to_string(self):
        """The select branch must call String(value) before .toLowerCase()."""
        fn_body = self._get_fn_body()
        self.assertIn(
            "String(value)",
            fn_body,
            "setFieldValue must coerce value to string in the select branch.",
        )

    def test_radio_branch_coerces_to_string(self):
        """The radio branch must also coerce the value to string."""
        fn_body = self._get_fn_body()
        radio_start = fn_body.index('} else if (type === "radio")')
        radio_body = fn_body[radio_start:]
        self.assertIn(
            "String(value)",
            radio_body,
            "setFieldValue must coerce value to string in the radio branch.",
        )

    def test_select_lower_uses_str_value_not_value(self):
        """After String(value), comparisons must use strValue.toLowerCase(), not value.toLowerCase()."""
        fn_body = self._get_fn_body()
        # strValue should be used in the select block comparisons
        select_start = fn_body.index('if (type === "select")')
        select_end = fn_body.index('} else if (type === "radio")')
        select_block = fn_body[select_start:select_end]
        self.assertIn("strValue.toLowerCase()", select_block)

    def test_radio_lower_uses_str_value_not_value(self):
        """Radio block must use strValue.toLowerCase() consistently."""
        fn_body = self._get_fn_body()
        radio_start = fn_body.index('} else if (type === "radio")')
        # Radio block ends before } else if (type === "checkbox")
        radio_end = fn_body.index('} else if (type === "checkbox")')
        radio_block = fn_body[radio_start:radio_end]
        self.assertIn("strValue.toLowerCase()", radio_block)

    def test_select_matching_logic_with_numeric_value(self):
        """Simulate the corrected select-matching logic with a numeric value."""
        options = [{"text": "1", "value": "1"}, {"text": "2", "value": "2"}]
        value = 2  # integer from LLM output
        str_value = str(value)
        match = next(
            (o for o in options if o["text"].lower() == str_value.lower()), None
        ) or next(
            (o for o in options if o["value"].lower() == str_value.lower()), None
        )
        self.assertIsNotNone(match, "Numeric 2 should match option '2' after coercion.")
        self.assertEqual(match["value"], "2")

    def test_select_matching_logic_with_boolean_value(self):
        """Simulate the corrected select-matching logic with a boolean value."""
        options = [{"text": "True", "value": "true"}, {"text": "False", "value": "false"}]
        value = True
        str_value = str(value)  # "True"
        match = next(
            (o for o in options if o["text"].lower() == str_value.lower()), None
        ) or next(
            (o for o in options if o["value"].lower() == str_value.lower()), None
        )
        self.assertIsNotNone(match, "Boolean True should match option 'True' after coercion.")

    def test_no_coercion_raises_on_number(self):
        """Confirm that calling .lower() on a non-string raises AttributeError,
        which is the bug the fix prevents."""
        value = 42
        with self.assertRaises(AttributeError):
            _ = value.lower()  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
