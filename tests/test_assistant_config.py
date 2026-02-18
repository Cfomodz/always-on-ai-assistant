"""
Tests for modules/assistant_config.py

Uses temporary YAML files so the real assistant_config.yml is not required.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from modules.assistant_config import get_config, get_config_file


# ---------------------------------------------------------------------------
# Helper: write a temporary YAML file and return its absolute path
# ---------------------------------------------------------------------------

def _write_yaml(tmp_dir: str, data: dict, name: str = "test_config.yml") -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ===========================================================================
# get_config
# ===========================================================================

class TestGetConfig(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_cwd = os.getcwd()
        # get_config uses os.getcwd() to resolve the config path
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmpdir.cleanup()

    def _config(self, data: dict, name: str = "test.yml") -> str:
        return _write_yaml(self._tmpdir.name, data, name)

    # ── Basic lookups ─────────────────────────────────────────────────────

    def test_simple_top_level_key(self):
        path = self._config({"greeting": "hello"})
        self.assertEqual(get_config("greeting", path), "hello")

    def test_nested_two_levels(self):
        path = self._config({"parent": {"child": "value"}})
        self.assertEqual(get_config("parent.child", path), "value")

    def test_nested_three_levels(self):
        path = self._config({"a": {"b": {"c": 42}}})
        self.assertEqual(get_config("a.b.c", path), 42)

    def test_integer_value(self):
        path = self._config({"port": 8741})
        self.assertEqual(get_config("port", path), 8741)

    def test_float_value(self):
        path = self._config({"threshold": 0.85})
        self.assertAlmostEqual(get_config("threshold", path), 0.85)

    def test_list_value(self):
        path = self._config({"items": [1, 2, 3]})
        self.assertEqual(get_config("items", path), [1, 2, 3])

    def test_boolean_value(self):
        path = self._config({"enabled": True})
        self.assertTrue(get_config("enabled", path))

    # ── Error cases ───────────────────────────────────────────────────────

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_config("key", "nonexistent_config.yml")

    def test_missing_key_raises_key_error(self):
        path = self._config({"present": "yes"})
        with self.assertRaises(KeyError):
            get_config("absent", path)

    def test_missing_nested_key_raises_key_error(self):
        path = self._config({"parent": {"child": "value"}})
        with self.assertRaises(KeyError):
            get_config("parent.missing_child", path)

    # ── Real config (if present) ──────────────────────────────────────────

    def test_real_config_scholarship_assistant_brain(self):
        """Integration test against the real config file, if it exists."""
        os.chdir(self._original_cwd)
        real_config = os.path.join(self._original_cwd, "assistant_config.yml")
        if not os.path.exists(real_config):
            self.skipTest("assistant_config.yml not found — skipping integration test")
        result = get_config("scholarship_assistant.brain", "assistant_config.yml")
        self.assertIsInstance(result, str)


# ===========================================================================
# get_config_file
# ===========================================================================

class TestGetConfigFile(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmpdir.cleanup()

    def test_returns_string(self):
        path = _write_yaml(self._tmpdir.name, {"key": "value"})
        result = get_config_file(path)
        self.assertIsInstance(result, str)

    def test_content_matches_file(self):
        path = _write_yaml(self._tmpdir.name, {"greeting": "hello"})
        result = get_config_file(path)
        self.assertIn("greeting", result)
        self.assertIn("hello", result)

    def test_multiline_config_preserved(self):
        data = {"section": {"a": 1, "b": 2, "c": 3}}
        path = _write_yaml(self._tmpdir.name, data)
        result = get_config_file(path)
        self.assertIn("section", result)

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            get_config_file("definitely_does_not_exist.yml")


if __name__ == "__main__":
    unittest.main()
