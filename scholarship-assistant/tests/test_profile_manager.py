"""
Tests for scholarship-assistant/server/profile_manager.py

All tests use temporary directories so the real ~/.scholarship-assistant
profile is never touched.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make scholarship-assistant/ the package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.profile_manager import (
    _empty_profile,
    get_field,
    set_field,
    get_flat_profile_for_matching,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_with(updates: dict) -> dict:
    """Return a fresh empty profile with the given dot-key updates applied."""
    profile = _empty_profile()
    for k, v in updates.items():
        set_field(profile, k, v)
    return profile


# ===========================================================================
# _empty_profile
# ===========================================================================

class TestEmptyProfile(unittest.TestCase):
    def test_returns_dict(self):
        self.assertIsInstance(_empty_profile(), dict)

    def test_has_version(self):
        self.assertIn("_version", _empty_profile())

    def test_has_all_top_level_sections(self):
        profile = _empty_profile()
        for section in ["personal", "disability", "education_current",
                        "education_history", "professional", "financial",
                        "extracurricular", "essays"]:
            self.assertIn(section, profile, f"Missing section: {section}")

    def test_personal_has_email(self):
        self.assertIn("email", _empty_profile()["personal"])

    def test_essays_is_empty_dict(self):
        self.assertEqual(_empty_profile()["essays"], {})

    def test_list_fields_are_empty_lists(self):
        profile = _empty_profile()
        self.assertIsInstance(profile["education_current"]["majors"], list)
        self.assertIsInstance(profile["disability"]["disability_types"], list)
        self.assertIsInstance(profile["professional"]["skills"], list)

    def test_multiple_calls_return_independent_copies(self):
        p1 = _empty_profile()
        p2 = _empty_profile()
        p1["personal"]["email"] = "test@example.com"
        self.assertEqual(p2["personal"]["email"], "")


# ===========================================================================
# load_profile / save_profile / profile_exists (via disk)
# ===========================================================================

class TestProfileDiskOperations(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._profile_path = Path(self._tmpdir.name) / "profile.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _patched_import(self):
        """Context manager that patches PROFILE_PATH and ensure_data_dir."""
        return patch.multiple(
            "server.profile_manager",
            PROFILE_PATH=self._profile_path,
            ensure_data_dir=lambda: None,
        )

    def test_load_profile_returns_empty_when_no_file(self):
        with self._patched_import():
            from server.profile_manager import load_profile
            profile = load_profile()
        self.assertIn("personal", profile)
        self.assertEqual(profile["personal"]["email"], "")

    def test_save_and_reload(self):
        with self._patched_import():
            from server.profile_manager import load_profile, save_profile
            profile = load_profile()
            profile["personal"]["email"] = "test@example.com"
            save_profile(profile)
            reloaded = load_profile()
        self.assertEqual(reloaded["personal"]["email"], "test@example.com")

    def test_save_is_atomic_via_tmp(self):
        """save_profile writes a .tmp file first and renames it — verify."""
        with self._patched_import():
            from server.profile_manager import load_profile, save_profile
            profile = load_profile()
            save_profile(profile)
        # After save the .tmp file should no longer exist
        tmp_path = self._profile_path.with_suffix(".tmp")
        self.assertFalse(tmp_path.exists())
        self.assertTrue(self._profile_path.exists())

    def test_profile_exists_false_before_save(self):
        with self._patched_import():
            from server.profile_manager import profile_exists
            self.assertFalse(profile_exists())

    def test_profile_exists_true_after_save(self):
        with self._patched_import():
            from server.profile_manager import load_profile, save_profile, profile_exists
            save_profile(load_profile())
            self.assertTrue(profile_exists())

    def test_update_profile_persists(self):
        with self._patched_import():
            from server.profile_manager import update_profile, load_profile
            update_profile({"personal.phone": "555-1234"})
            reloaded = load_profile()
        self.assertEqual(reloaded["personal"]["phone"], "555-1234")

    def test_add_essay_persists(self):
        with self._patched_import():
            from server.profile_manager import add_essay, load_profile
            add_essay("leadership prompt", "I led a team of five.")
            reloaded = load_profile()
        self.assertIn("leadership prompt", reloaded["essays"])
        self.assertEqual(reloaded["essays"]["leadership prompt"], "I led a team of five.")

    def test_add_essay_overwrites_existing(self):
        with self._patched_import():
            from server.profile_manager import add_essay, load_profile
            add_essay("prompt", "first version")
            add_essay("prompt", "updated version")
            reloaded = load_profile()
        self.assertEqual(reloaded["essays"]["prompt"], "updated version")


# ===========================================================================
# get_field
# ===========================================================================

class TestGetField(unittest.TestCase):
    def setUp(self):
        self.profile = _profile_with({"personal.email": "alice@example.com"})

    def test_simple_two_level_key(self):
        self.assertEqual(get_field(self.profile, "personal.email"), "alice@example.com")

    def test_three_level_key(self):
        profile = _profile_with({"education_history.high_school.name": "Lincoln High"})
        self.assertEqual(
            get_field(profile, "education_history.high_school.name"), "Lincoln High"
        )

    def test_missing_top_level_key_returns_none(self):
        self.assertIsNone(get_field(self.profile, "nonexistent.key"))

    def test_missing_nested_key_returns_none(self):
        self.assertIsNone(get_field(self.profile, "personal.nonexistent"))

    def test_returns_list(self):
        profile = _profile_with({"education_current.majors": ["CS", "Math"]})
        self.assertEqual(get_field(profile, "education_current.majors"), ["CS", "Math"])

    def test_returns_empty_string_for_unset_scalar(self):
        self.assertEqual(get_field(self.profile, "personal.phone"), "")

    def test_single_level_key(self):
        self.assertIsInstance(get_field(self.profile, "personal"), dict)

    def test_version_key(self):
        self.assertIsNotNone(get_field(self.profile, "_version"))


# ===========================================================================
# set_field
# ===========================================================================

class TestSetField(unittest.TestCase):
    def setUp(self):
        self.profile = _empty_profile()

    def test_set_existing_scalar(self):
        set_field(self.profile, "personal.email", "bob@example.com")
        self.assertEqual(self.profile["personal"]["email"], "bob@example.com")

    def test_set_creates_intermediate_dict(self):
        set_field(self.profile, "new_section.subsection.field", "value")
        self.assertEqual(self.profile["new_section"]["subsection"]["field"], "value")

    def test_set_list_value(self):
        set_field(self.profile, "education_current.majors", ["CS", "Math"])
        self.assertEqual(self.profile["education_current"]["majors"], ["CS", "Math"])

    def test_set_returns_profile(self):
        result = set_field(self.profile, "personal.phone", "555-0000")
        self.assertIs(result, self.profile)

    def test_set_overwrites_existing(self):
        set_field(self.profile, "personal.email", "first@example.com")
        set_field(self.profile, "personal.email", "second@example.com")
        self.assertEqual(self.profile["personal"]["email"], "second@example.com")

    def test_set_three_level_nested(self):
        set_field(self.profile, "education_history.high_school.name", "East High")
        self.assertEqual(self.profile["education_history"]["high_school"]["name"], "East High")


# ===========================================================================
# get_flat_profile_for_matching
# ===========================================================================

class TestGetFlatProfile(unittest.TestCase):
    def test_empty_profile_gives_empty_flat(self):
        profile = _empty_profile()
        flat = get_flat_profile_for_matching(profile)
        self.assertEqual(flat, {})

    def test_populated_string_field_appears(self):
        profile = _profile_with({"personal.email": "alice@example.com"})
        flat = get_flat_profile_for_matching(profile)
        self.assertEqual(flat["personal.email"], "alice@example.com")

    def test_empty_string_excluded(self):
        profile = _profile_with({"personal.phone": ""})
        flat = get_flat_profile_for_matching(profile)
        self.assertNotIn("personal.phone", flat)

    def test_non_empty_list_appears(self):
        profile = _profile_with({"education_current.majors": ["Computer Science"]})
        flat = get_flat_profile_for_matching(profile)
        self.assertIn("education_current.majors", flat)
        self.assertEqual(flat["education_current.majors"], ["Computer Science"])

    def test_empty_list_excluded(self):
        profile = _empty_profile()
        # All list fields start empty
        flat = get_flat_profile_for_matching(profile)
        self.assertNotIn("education_current.majors", flat)

    def test_version_key_excluded(self):
        profile = _empty_profile()
        flat = get_flat_profile_for_matching(profile)
        for key in flat:
            self.assertFalse(key.startswith("_"), f"Private key '{key}' leaked into flat profile")

    def test_multiple_fields_populated(self):
        profile = _profile_with({
            "personal.email": "test@test.com",
            "personal.phone": "555-9999",
            "education_current.university_name": "State U",
        })
        flat = get_flat_profile_for_matching(profile)
        self.assertIn("personal.email", flat)
        self.assertIn("personal.phone", flat)
        self.assertIn("education_current.university_name", flat)

    def test_none_excluded(self):
        profile = _empty_profile()
        profile["personal"]["age"] = None
        flat = get_flat_profile_for_matching(profile)
        self.assertNotIn("personal.age", flat)

    def test_nested_three_levels(self):
        profile = _profile_with({"education_history.high_school.name": "Lincoln High"})
        flat = get_flat_profile_for_matching(profile)
        self.assertIn("education_history.high_school.name", flat)


if __name__ == "__main__":
    unittest.main()
