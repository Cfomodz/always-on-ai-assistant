"""
Tests for modules/tts_cache.py

Cache directory is redirected to a temporary folder so the real
~/.assistant-cache/tts directory is never touched.
ElevenLabs API calls are stubbed.
"""

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub elevenlabs before importing tts_cache so the import-time `from elevenlabs
# import play` line doesn't fail in the test environment.
_el = types.ModuleType("elevenlabs")
_el.play = lambda *a, **kw: None
sys.modules.setdefault("elevenlabs", _el)
_elc = types.ModuleType("elevenlabs.client")
_elc.ElevenLabs = object
sys.modules.setdefault("elevenlabs.client", _elc)

import modules.tts_cache as tts_cache


def _patch_cache_dir(tmp_dir: str):
    """Context manager: redirect CACHE_DIR inside tts_cache to tmp_dir."""
    new_dir = Path(tmp_dir) / "tts"
    return patch.object(tts_cache, "CACHE_DIR", new_dir)


def _expected_key(text: str, voice: str, model: str) -> str:
    payload = f"{text}|{voice}|{model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ===========================================================================
# _cache_key (internal, but critical — test via public interface)
# ===========================================================================

class TestCacheKey(unittest.TestCase):
    def test_deterministic(self):
        key1 = tts_cache._cache_key("hello", "voice1", "model1")
        key2 = tts_cache._cache_key("hello", "voice1", "model1")
        self.assertEqual(key1, key2)

    def test_different_text_different_key(self):
        k1 = tts_cache._cache_key("hello", "v", "m")
        k2 = tts_cache._cache_key("world", "v", "m")
        self.assertNotEqual(k1, k2)

    def test_different_voice_different_key(self):
        k1 = tts_cache._cache_key("text", "voice_a", "m")
        k2 = tts_cache._cache_key("text", "voice_b", "m")
        self.assertNotEqual(k1, k2)

    def test_different_model_different_key(self):
        k1 = tts_cache._cache_key("text", "v", "model_a")
        k2 = tts_cache._cache_key("text", "v", "model_b")
        self.assertNotEqual(k1, k2)

    def test_returns_hex_string(self):
        key = tts_cache._cache_key("text", "voice", "model")
        int(key, 16)  # raises if not valid hex

    def test_sha256_length(self):
        key = tts_cache._cache_key("text", "voice", "model")
        self.assertEqual(len(key), 64)  # SHA-256 = 32 bytes = 64 hex chars

    def test_matches_expected_hash(self):
        text, voice, model = "Hello world", "v123", "eleven_flash"
        self.assertEqual(tts_cache._cache_key(text, voice, model),
                         _expected_key(text, voice, model))


# ===========================================================================
# _cache_path
# ===========================================================================

class TestCachePath(unittest.TestCase):
    def test_returns_path_with_mp3_suffix(self):
        p = tts_cache._cache_path("abc123")
        self.assertEqual(p.suffix, ".mp3")

    def test_filename_is_the_key(self):
        key = "deadbeef"
        p = tts_cache._cache_path(key)
        self.assertEqual(p.stem, key)


# ===========================================================================
# is_cached / get_cached_audio / store_audio
# ===========================================================================

class TestCacheOperations(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_is_cached_false_before_store(self):
        with _patch_cache_dir(self._tmpdir.name):
            result = tts_cache.is_cached("text", "voice", "model")
        self.assertFalse(result)

    def test_is_cached_true_after_store(self):
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("text", "voice", "model", b"audio_bytes")
            result = tts_cache.is_cached("text", "voice", "model")
        self.assertTrue(result)

    def test_get_cached_audio_returns_none_on_miss(self):
        with _patch_cache_dir(self._tmpdir.name):
            result = tts_cache.get_cached_audio("text", "voice", "model")
        self.assertIsNone(result)

    def test_get_cached_audio_returns_bytes_on_hit(self):
        audio = b"\xff\xfb\x90\x00" * 100  # fake MP3 bytes
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("text", "voice", "model", audio)
            result = tts_cache.get_cached_audio("text", "voice", "model")
        self.assertEqual(result, audio)

    def test_store_creates_file_with_correct_name(self):
        with _patch_cache_dir(self._tmpdir.name):
            path = tts_cache.store_audio("text", "voice", "model", b"data")
        self.assertTrue(path.exists())
        expected_key = _expected_key("text", "voice", "model")
        self.assertEqual(path.stem, expected_key)

    def test_store_returns_path(self):
        with _patch_cache_dir(self._tmpdir.name):
            result = tts_cache.store_audio("text", "voice", "model", b"data")
        self.assertIsInstance(result, Path)

    def test_cache_hit_returns_exact_bytes(self):
        audio = b"exact_audio_data_12345"
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("t", "v", "m", audio)
            result = tts_cache.get_cached_audio("t", "v", "m")
        self.assertEqual(result, audio)


# ===========================================================================
# cache_stats
# ===========================================================================

class TestCacheStats(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_empty_cache_returns_zeros(self):
        with _patch_cache_dir(self._tmpdir.name):
            stats = tts_cache.cache_stats()
        self.assertEqual(stats, {"files": 0, "total_bytes": 0})

    def test_no_cache_dir_returns_zeros(self):
        non_existent = Path(self._tmpdir.name) / "does_not_exist" / "tts"
        with patch.object(tts_cache, "CACHE_DIR", non_existent):
            stats = tts_cache.cache_stats()
        self.assertEqual(stats, {"files": 0, "total_bytes": 0})

    def test_file_count_matches(self):
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("a", "v", "m", b"x" * 100)
            tts_cache.store_audio("b", "v", "m", b"x" * 200)
            stats = tts_cache.cache_stats()
        self.assertEqual(stats["files"], 2)

    def test_total_bytes_correct(self):
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("a", "v", "m", b"A" * 100)
            tts_cache.store_audio("b", "v", "m", b"B" * 300)
            stats = tts_cache.cache_stats()
        self.assertEqual(stats["total_bytes"], 400)


# ===========================================================================
# clear_cache
# ===========================================================================

class TestClearCache(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_clear_on_non_existent_dir_returns_zero(self):
        non_existent = Path(self._tmpdir.name) / "no_cache" / "tts"
        with patch.object(tts_cache, "CACHE_DIR", non_existent):
            count = tts_cache.clear_cache()
        self.assertEqual(count, 0)

    def test_clear_removes_all_mp3_files(self):
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("a", "v", "m", b"data")
            tts_cache.store_audio("b", "v", "m", b"data")
            count = tts_cache.clear_cache()
            stats = tts_cache.cache_stats()
        self.assertEqual(count, 2)
        self.assertEqual(stats["files"], 0)

    def test_clear_returns_count(self):
        with _patch_cache_dir(self._tmpdir.name):
            for i in range(4):
                tts_cache.store_audio(str(i), "v", "m", b"data")
            count = tts_cache.clear_cache()
        self.assertEqual(count, 4)

    def test_double_clear_returns_zero_second_time(self):
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.store_audio("x", "v", "m", b"data")
            tts_cache.clear_cache()
            count = tts_cache.clear_cache()
        self.assertEqual(count, 0)


# ===========================================================================
# generate_with_cache (cache=True path; API stubbed)
# ===========================================================================

class TestGenerateWithCache(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _fake_client(self, audio: bytes):
        client = MagicMock()
        client.generate.return_value = iter([audio])
        return client

    def test_cache_miss_calls_generate(self):
        audio = b"fresh_audio"
        client = self._fake_client(audio)
        with _patch_cache_dir(self._tmpdir.name):
            result = tts_cache.generate_with_cache(client, "text", "voice", "model", cache=True)
        client.generate.assert_called_once()
        self.assertEqual(result, audio)

    def test_cache_hit_skips_generate(self):
        audio = b"cached_audio"
        client = self._fake_client(audio)
        with _patch_cache_dir(self._tmpdir.name):
            # Warm the cache
            tts_cache.store_audio("text", "voice", "model", audio)
            result = tts_cache.generate_with_cache(client, "text", "voice", "model", cache=True)
        client.generate.assert_not_called()
        self.assertEqual(result, audio)

    def test_cache_false_always_calls_generate(self):
        audio = b"always_fresh"
        client = self._fake_client(audio)
        with _patch_cache_dir(self._tmpdir.name):
            # Pre-populate cache
            tts_cache.store_audio("text", "voice", "model", audio)
            # cache=False should bypass cache
            result = tts_cache.generate_with_cache(client, "text", "voice", "model", cache=False)
        client.generate.assert_called_once()

    def test_generate_result_is_stored_in_cache(self):
        audio = b"will_be_cached"
        client = self._fake_client(audio)
        with _patch_cache_dir(self._tmpdir.name):
            tts_cache.generate_with_cache(client, "text", "voice", "model", cache=True)
            cached = tts_cache.get_cached_audio("text", "voice", "model")
        self.assertEqual(cached, audio)


if __name__ == "__main__":
    unittest.main()
