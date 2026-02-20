"""
Tests for scholarship-assistant/server/profile_import.py
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub modules.deepseek before importing profile_import (avoids openai dependency)
_ds = types.ModuleType("modules.deepseek")
_ds.json_prompt = lambda *a, **kw: {}
sys.modules["modules.deepseek"] = _ds

import unittest

from server.profile_import import (
    _detect_format,
    _parse_qa_rows,
    import_into_profile,
    load_content_from_path,
    review_profile,
)

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


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

    def test_load_content_from_path_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello, scholarship data.")
            path = Path(f.name)
        try:
            self.assertEqual(load_content_from_path(path), "Hello, scholarship data.")
        finally:
            path.unlink()

    @unittest.skipUnless(HAS_PYMUPDF, "pymupdf not installed")
    def test_load_content_from_path_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = Path(f.name)
        try:
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Resume content: CS degree from WGU")
            doc.save(path)
            doc.close()
            content = load_content_from_path(path)
            self.assertIn("Resume content", content)
            self.assertIn("WGU", content)
        finally:
            path.unlink()

    def test_load_content_from_path_unsupported_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_content_from_path(path)
            self.assertIn(".docx", str(ctx.exception))
            self.assertIn(".txt", str(ctx.exception))
        finally:
            path.unlink()

    def test_review_profile_dry_run_returns_updates_without_applying(self):
        mock_result = {
            "updates": {
                "education_current.degree_type": ["Bachelor of Science in Computer Science"],
            },
            "summary": "Consolidated 3 degree variants into 1.",
            "skipped": [],
        }
        mock_profile = {
            "education_current": {
                "degree_type": ["BS", "Bachelor of Science", "Bachelor of Science in Computer Science"],
                "majors": ["CS"],
            },
        }
        with patch("server.profile_import.json_prompt", return_value=mock_result):
            with patch("server.profile_import.load_profile", return_value=mock_profile):
                result = review_profile(dry_run=True)
        self.assertFalse(result["applied"])
        self.assertEqual(result["updates"]["education_current.degree_type"], ["Bachelor of Science in Computer Science"])
        self.assertIn("Consolidated", result["summary"])
