"""
TTS Audio Cache — Caches ElevenLabs generated audio to disk.

When the exact same (text, voice_id, model) tuple is requested again,
returns the cached audio bytes instead of making a new API call.

Cache directory: ~/.assistant-cache/tts/
File naming: SHA-256 hash of (text + voice_id + model) → .mp3
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from elevenlabs import play
from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

# --- Cache directory ---
CACHE_DIR = Path.home() / ".assistant-cache" / "tts"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(text: str, voice: str, model: str) -> str:
    """Deterministic hash for a (text, voice, model) tuple."""
    payload = f"{text}|{voice}|{model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.mp3"


def get_cached_audio(text: str, voice: str, model: str) -> Optional[bytes]:
    """Return cached audio bytes if they exist, else None."""
    key = _cache_key(text, voice, model)
    path = _cache_path(key)
    if path.exists():
        logger.debug(f"TTS cache HIT: {path.name} ({text[:40]}...)")
        return path.read_bytes()
    return None


def store_audio(text: str, voice: str, model: str, audio_bytes: bytes) -> Path:
    """Write audio bytes to the cache. Returns the file path."""
    _ensure_cache_dir()
    key = _cache_key(text, voice, model)
    path = _cache_path(key)
    path.write_bytes(audio_bytes)
    logger.debug(f"TTS cache STORE: {path.name} ({len(audio_bytes)} bytes)")
    return path


def is_cached(text: str, voice: str, model: str) -> bool:
    """Check if audio for this text/voice/model combo exists in cache."""
    key = _cache_key(text, voice, model)
    return _cache_path(key).exists()


def generate_with_cache(
    client: ElevenLabs,
    text: str,
    voice: str,
    model: str,
    cache: bool = True,
) -> bytes:
    """
    Generate TTS audio, using cache when available.

    Args:
        client: ElevenLabs client instance.
        text: The text to speak.
        voice: ElevenLabs voice ID.
        model: ElevenLabs model ID.
        cache: If True, check/store in cache. If False, always generate fresh.

    Returns:
        Audio bytes (MP3).
    """
    if cache:
        cached = get_cached_audio(text, voice, model)
        if cached is not None:
            return cached

    # Generate fresh audio
    audio_generator = client.generate(
        text=text,
        voice=voice,
        model=model,
        stream=False,
    )
    audio_bytes = b"".join(list(audio_generator))

    if cache:
        store_audio(text, voice, model, audio_bytes)

    return audio_bytes


def clear_cache() -> int:
    """Remove all cached audio files. Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.mp3"):
        f.unlink()
        count += 1
    logger.info(f"TTS cache cleared: {count} files removed")
    return count


def cache_stats() -> dict:
    """Return basic cache statistics."""
    if not CACHE_DIR.exists():
        return {"files": 0, "total_bytes": 0}
    files = list(CACHE_DIR.glob("*.mp3"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"files": len(files), "total_bytes": total_bytes}
