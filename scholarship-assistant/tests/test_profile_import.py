"""
Tests for scholarship-assistant/server/profile_import.py
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub modules.deepseek before importing profile_import (avoids openai dependency)
_ds = types.ModuleType("modules.deepseek")
_ds.json_prompt = lambda *a, **kw: {}
sys.modules["modules.deepseek"] = _ds

import unittest

from server.profile_import import _detect_format, _parse_qa_rows, import_into_profile


class TestProfileImport(unittest.TestCase):
    def test_parse_qa_rows_detects_format(self):
        qa_content = """Question	Answer(s)	Last Answered
What is your citizenship?	US Citizen	6/13/2025
Which college?	WGU	2/13/2026"""
        self.assertEqual(_detect_format(qa_content), "qa")
        rows = _parse_qa_rows(qa_content)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "What is your citizenship?")
        self.assertEqual(rows[0]["answer"], "US Citizen")
        self.assertEqual(rows[0]["last_answered"], "6/13/2025")

    def test_detect_raw_format(self):
        raw = "I am a US citizen. I attend WGU. I live in Montana."
        self.assertEqual(_detect_format(raw), "raw")

    def test_import_dry_run_returns_updates_without_applying(self):
        mock_result = {
            "updates": {"personal.citizenship": "US Citizen", "personal.race_ethnicity": "White or Caucasian"},
            "summary": "Imported 2 fields.",
            "skipped": [],
        }

        with patch("server.profile_import.json_prompt", return_value=mock_result):
            with patch("server.profile_import.load_profile") as mock_load:
                mock_load.return_value = {"personal": {"citizenship": "", "race_ethnicity": ""}}
                result = import_into_profile(
                    "What is your citizenship?\tUS Citizen\t6/13/2025",
                    dry_run=True,
                )
        self.assertFalse(result["applied"])
        self.assertEqual(len(result["updates"]), 2)
        self.assertIn("personal.citizenship", result["updates"])
