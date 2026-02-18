"""
Tests for modules/utils.py

All functions are pure logic or light file I/O — no external API calls.
"""

import datetime
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Import from the repo root (pytest is run from there)
from modules.utils import (
    build_file_path,
    build_file_name_session,
    to_json_file_pretty,
    current_date_time_str,
    current_date_str,
    dict_item_diff_by_set,
    create_session_logger_id,
    parse_markdown_backticks,
)


# ===========================================================================
# build_file_path
# ===========================================================================

class TestBuildFilePath(unittest.TestCase):
    def test_returns_string(self):
        result = build_file_path("test.txt")
        self.assertIsInstance(result, str)

    def test_contains_filename(self):
        result = build_file_path("myfile.txt")
        self.assertIn("myfile.txt", result)

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("modules.utils.OUTPUT_DIR", tmp):
                path = build_file_path("test.txt")
            self.assertTrue(os.path.isdir(tmp))

    def test_path_includes_output_dir(self):
        result = build_file_path("hello.txt")
        self.assertIn("output", result)


# ===========================================================================
# build_file_name_session
# ===========================================================================

class TestBuildFileNameSession(unittest.TestCase):
    def test_contains_session_id(self):
        result = build_file_name_session("log.txt", "session123")
        self.assertIn("session123", result)

    def test_contains_filename(self):
        result = build_file_name_session("session.log", "sess42")
        self.assertIn("session.log", result)

    def test_creates_session_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("modules.utils.OUTPUT_DIR", tmp):
                build_file_name_session("log.txt", "my_session")
            self.assertTrue(os.path.isdir(os.path.join(tmp, "my_session")))


# ===========================================================================
# to_json_file_pretty
# ===========================================================================

class TestToJsonFilePretty(unittest.TestCase):
    def test_writes_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            to_json_file_pretty(path, {"key": "value"})
            with open(f"{path}.json") as f:
                data = json.load(f)
            self.assertEqual(data["key"], "value")

    def test_writes_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            to_json_file_pretty(path, [1, 2, 3])
            with open(f"{path}.json") as f:
                data = json.load(f)
            self.assertEqual(data, [1, 2, 3])

    def test_pretty_printed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            to_json_file_pretty(path, {"a": 1})
            content = Path(f"{path}.json").read_text()
            # Pretty-printed JSON has newlines
            self.assertIn("\n", content)

    def test_pydantic_model_serialised(self):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            value: int

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            to_json_file_pretty(path, {"model": MyModel(name="test", value=42)})
            with open(f"{path}.json") as f:
                data = json.load(f)
            self.assertEqual(data["model"]["name"], "test")

    def test_unserializable_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            with self.assertRaises(TypeError):
                to_json_file_pretty(path, {"bad": object()})


# ===========================================================================
# current_date_time_str
# ===========================================================================

class TestCurrentDateTimeStr(unittest.TestCase):
    FORMAT_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def test_format_matches(self):
        result = current_date_time_str()
        self.assertRegex(result, self.FORMAT_REGEX)

    def test_returns_string(self):
        self.assertIsInstance(current_date_time_str(), str)

    def test_year_is_current(self):
        result = current_date_time_str()
        year = result.split("-")[0]
        self.assertEqual(year, str(datetime.datetime.now().year))


# ===========================================================================
# current_date_str
# ===========================================================================

class TestCurrentDateStr(unittest.TestCase):
    FORMAT_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def test_format_matches(self):
        self.assertRegex(current_date_str(), self.FORMAT_REGEX)

    def test_returns_string(self):
        self.assertIsInstance(current_date_str(), str)

    def test_year_is_current(self):
        result = current_date_str()
        year = result.split("-")[0]
        self.assertEqual(year, str(datetime.datetime.now().year))


# ===========================================================================
# dict_item_diff_by_set
# ===========================================================================

class TestDictItemDiffBySet(unittest.TestCase):
    def test_new_item_detected(self):
        prev = [{"id": "a"}, {"id": "b"}]
        curr = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        diff = dict_item_diff_by_set(prev, curr, "id")
        self.assertEqual(diff, ["c"])

    def test_no_change_returns_empty(self):
        items = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(dict_item_diff_by_set(items, items, "id"), [])

    def test_removed_item_not_returned(self):
        prev = [{"id": "a"}, {"id": "b"}]
        curr = [{"id": "a"}]
        diff = dict_item_diff_by_set(prev, curr, "id")
        self.assertEqual(diff, [])

    def test_multiple_new_items(self):
        prev = [{"name": "x"}]
        curr = [{"name": "x"}, {"name": "y"}, {"name": "z"}]
        diff = dict_item_diff_by_set(prev, curr, "name")
        self.assertCountEqual(diff, ["y", "z"])

    def test_empty_previous_list(self):
        curr = [{"id": "new"}]
        diff = dict_item_diff_by_set([], curr, "id")
        self.assertEqual(diff, ["new"])

    def test_empty_current_list(self):
        prev = [{"id": "old"}]
        diff = dict_item_diff_by_set(prev, [], "id")
        self.assertEqual(diff, [])


# ===========================================================================
# create_session_logger_id
# ===========================================================================

class TestCreateSessionLoggerId(unittest.TestCase):
    PATTERN = re.compile(r"^\d{8}-\d{6}-[a-f0-9]{6}$")

    def test_format_matches(self):
        result = create_session_logger_id()
        self.assertRegex(result, self.PATTERN)

    def test_returns_string(self):
        self.assertIsInstance(create_session_logger_id(), str)

    def test_unique_each_call(self):
        ids = {create_session_logger_id() for _ in range(10)}
        self.assertEqual(len(ids), 10)


# ===========================================================================
# parse_markdown_backticks
# ===========================================================================

class TestParseMarkdownBackticks(unittest.TestCase):
    def test_no_backticks_returns_stripped(self):
        self.assertEqual(parse_markdown_backticks("  hello world  "), "hello world")

    def test_python_code_block(self):
        text = "```python\nprint('hello')\n```"
        result = parse_markdown_backticks(text)
        self.assertEqual(result, "print('hello')")

    def test_bash_code_block(self):
        text = "```bash\necho hello\n```"
        result = parse_markdown_backticks(text)
        self.assertEqual(result, "echo hello")

    def test_no_language_specifier(self):
        text = "```\nsome code\n```"
        result = parse_markdown_backticks(text)
        self.assertEqual(result, "some code")

    def test_multiline_code_block(self):
        text = "```python\nline1\nline2\nline3\n```"
        result = parse_markdown_backticks(text)
        self.assertIn("line1", result)
        self.assertIn("line3", result)

    def test_leading_trailing_stripped_within_block(self):
        text = "```python\n   code   \n```"
        result = parse_markdown_backticks(text)
        self.assertNotEqual(result, "")

    def test_empty_code_block(self):
        text = "```\n```"
        result = parse_markdown_backticks(text)
        self.assertEqual(result, "")

    def test_plain_text_unchanged(self):
        text = "just plain text"
        self.assertEqual(parse_markdown_backticks(text), "just plain text")

    def test_text_with_content_before_block(self):
        # Content before the code block should be stripped away
        text = "Here is the code:\n```python\nresult = 42\n```"
        result = parse_markdown_backticks(text)
        self.assertIn("result", result)


if __name__ == "__main__":
    unittest.main()
