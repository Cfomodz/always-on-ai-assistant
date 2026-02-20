"""
Tests for scholarship-assistant/server/voice.py
Tests only the pure helper functions that don't require audio I/O.
"""

import sys
import types
from pathlib import Path

# Stub hardware/API modules before importing voice
_el = types.ModuleType("elevenlabs")
_el.play = lambda *a, **kw: None
sys.modules["elevenlabs"] = _el

_elc = types.ModuleType("elevenlabs.client")
_elc.ElevenLabs = object
sys.modules["elevenlabs.client"] = _elc

_astt = types.ModuleType("alternative_stt")
_astt.SimpleAudioRecorder = type("SimpleAudioRecorder", (), {"text_blocking": lambda self: ""})
sys.modules["alternative_stt"] = _astt

_tc = types.ModuleType("modules.tts_cache")
_tc.generate_with_cache = lambda *a, **kw: b""
sys.modules["modules.tts_cache"] = _tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.voice import _normalize_confirmation_response


class TestNormalizeConfirmationResponse:
    """Test letter-by-letter and spelled yes/no handling."""

    def test_yes_normal(self):
        assert _normalize_confirmation_response("yes") == "yes"

    def test_no_normal(self):
        assert _normalize_confirmation_response("no") == "no"

    def test_letter_by_letter_no(self):
        assert _normalize_confirmation_response("n o") == "no"

    def test_letter_by_letter_yes(self):
        assert _normalize_confirmation_response("y e s") == "yes"

    def test_strip_trailing_period(self):
        assert _normalize_confirmation_response("no.") == "no"

    def test_empty_returns_empty(self):
        assert _normalize_confirmation_response("") == ""

    def test_yeah_returns_raw(self):
        assert _normalize_confirmation_response("yeah") == "yeah"

    def test_y_normalizes_to_yes(self):
        assert _normalize_confirmation_response("y") == "yes"

    def test_n_normalizes_to_no(self):
        assert _normalize_confirmation_response("n") == "no"
