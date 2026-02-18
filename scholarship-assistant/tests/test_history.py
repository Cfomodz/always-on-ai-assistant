"""
Tests for scholarship-assistant/server/history.py

File I/O is redirected to temporary directories; no real API calls are made.
"""

import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

# Make scholarship-assistant/ the package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_history_file(tmp_dir: str, records: list) -> Path:
    p = Path(tmp_dir) / "history.json"
    p.write_text(json.dumps(records))
    return p


def _ctx(history_path: Path):
    """Patch HISTORY_PATH and ensure_data_dir for the duration of a with-block."""
    return patch.multiple(
        "server.history",
        HISTORY_PATH=history_path,
        ensure_data_dir=lambda: None,
    )


# ===========================================================================
# load_history / save_history
# ===========================================================================

class TestLoadSaveHistory(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_returns_empty_list_when_no_file(self):
        empty_path = Path(self._tmpdir.name) / "missing.json"
        with patch("server.history.HISTORY_PATH", empty_path):
            from server.history import load_history
            self.assertEqual(load_history(), [])

    def test_load_returns_existing_records(self):
        records = [{"scholarship_name": "Test", "organization": "Org"}]
        p = _make_history_file(self._tmpdir.name, records)
        with patch("server.history.HISTORY_PATH", p):
            from server.history import load_history
            self.assertEqual(load_history(), records)

    def test_save_then_load_round_trips(self):
        p = Path(self._tmpdir.name) / "history.json"
        records = [{"scholarship_name": "X", "organization": "Y"}]
        with _ctx(p):
            from server.history import save_history, load_history
            save_history(records)
            self.assertEqual(load_history(), records)

    def test_save_is_atomic_via_tmp(self):
        p = Path(self._tmpdir.name) / "history.json"
        with _ctx(p):
            from server.history import save_history
            save_history([])
        tmp_path = p.with_suffix(".tmp")
        self.assertFalse(tmp_path.exists())
        self.assertTrue(p.exists())


# ===========================================================================
# add_record
# ===========================================================================

class TestAddRecord(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "history.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _add(self, **kwargs):
        defaults = dict(
            url="https://example.com/scholarship",
            scholarship_name="Rhodes Scholarship",
            organization="Oxford University",
        )
        defaults.update(kwargs)
        with _ctx(self._path):
            from server.history import add_record
            return add_record(**defaults)

    def test_returns_record_dict(self):
        record = self._add()
        self.assertIsInstance(record, dict)

    def test_record_has_required_keys(self):
        record = self._add()
        for key in ["id", "timestamp", "url", "scholarship_name", "organization",
                    "status", "fields_filled", "fields_manual", "essays", "notes"]:
            self.assertIn(key, record, f"Missing key: {key}")

    def test_record_id_is_valid_uuid(self):
        record = self._add()
        uuid.UUID(record["id"])  # raises ValueError if not a valid UUID

    def test_record_persisted(self):
        self._add(scholarship_name="Test Scholarship")
        with patch("server.history.HISTORY_PATH", self._path):
            from server.history import load_history
            history = load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["scholarship_name"], "Test Scholarship")

    def test_multiple_records_accumulate(self):
        self._add(scholarship_name="Scholarship A")
        self._add(scholarship_name="Scholarship B")
        with patch("server.history.HISTORY_PATH", self._path):
            from server.history import load_history
            history = load_history()
        self.assertEqual(len(history), 2)

    def test_default_status_is_submitted(self):
        record = self._add()
        self.assertEqual(record["status"], "submitted")

    def test_custom_status(self):
        record = self._add(status="pending")
        self.assertEqual(record["status"], "pending")

    def test_essays_defaults_to_empty_list(self):
        record = self._add()
        self.assertEqual(record["essays"], [])

    def test_essays_provided(self):
        record = self._add(essays=["Essay 1"])
        self.assertEqual(record["essays"], ["Essay 1"])


# ===========================================================================
# check_duplicate
# ===========================================================================

class TestCheckDuplicate(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run(self, name: str, org: str, records: list, threshold: float = 0.85):
        p = _make_history_file(self._tmpdir.name, records)
        with patch("server.history.HISTORY_PATH", p):
            from server.history import check_duplicate
            return check_duplicate(name, org, threshold)

    # ── No identifiers ──────────────────────────────────────────────────────
    def test_empty_both_returns_none(self):
        self.assertIsNone(self._run("", "", [{"scholarship_name": "X", "organization": "Y"}]))

    # ── Both identifiers match ───────────────────────────────────────────────
    def test_exact_match_both_fields(self):
        records = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford University"}]
        result = self._run("Rhodes Scholarship", "Oxford University", records)
        self.assertIsNotNone(result)

    def test_no_match_returns_none(self):
        records = [{"scholarship_name": "Fulbright", "organization": "US State Dept"}]
        result = self._run("Gates Scholarship", "Gates Foundation", records)
        self.assertIsNone(result)

    def test_returns_matching_record(self):
        records = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford University", "id": "abc"}]
        result = self._run("Rhodes Scholarship", "Oxford University", records)
        self.assertEqual(result["id"], "abc")

    # ── Partial data: name only ─────────────────────────────────────────────
    def test_name_only_match_when_org_missing_in_history(self):
        records = [{"scholarship_name": "Rhodes Scholarship", "organization": ""}]
        result = self._run("Rhodes Scholarship", "", records)
        self.assertIsNotNone(result)

    def test_name_only_no_false_positive(self):
        records = [{"scholarship_name": "Fulbright Scholarship", "organization": ""}]
        result = self._run("Rhodes Scholarship", "", records)
        self.assertIsNone(result)

    # ── Partial data: org only ──────────────────────────────────────────────
    def test_org_only_match_when_name_missing_in_history(self):
        records = [{"scholarship_name": "", "organization": "Gates Foundation"}]
        result = self._run("", "Gates Foundation", records)
        self.assertIsNotNone(result)

    def test_org_only_no_false_positive(self):
        records = [{"scholarship_name": "", "organization": "Oxford University"}]
        result = self._run("", "Gates Foundation", records)
        self.assertIsNone(result)

    # ── Threshold edge cases ────────────────────────────────────────────────
    def test_threshold_1_0_requires_exact_match(self):
        records = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford University"}]
        result = self._run("Rhodes", "Oxford", records, threshold=1.0)
        self.assertIsNone(result)

    def test_threshold_0_requires_any_match(self):
        records = [{"scholarship_name": "Rhodes", "organization": "Oxford"}]
        result = self._run("Completely Different", "Totally Unrelated", records, threshold=0.0)
        self.assertIsNotNone(result)

    def test_empty_history_returns_none(self):
        result = self._run("Rhodes Scholarship", "Oxford", [])
        self.assertIsNone(result)

    # ── Fuzzy matching ──────────────────────────────────────────────────────
    def test_fuzzy_name_match_with_minor_typo(self):
        records = [{"scholarship_name": "Rhodes Scholarship", "organization": "Oxford University"}]
        # "Rhods" is close enough to "Rhodes"
        result = self._run("Rhods Scholarship", "Oxford University", records, threshold=0.80)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
