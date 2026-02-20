"""
Voice module — ElevenLabs TTS + openai-whisper for speech recognition.
Uses alternative_stt (openai-whisper + PyAudio) instead of RealtimeSTT.
Supports disk-based audio caching and pre-generated filler responses.
"""

import logging
import os
import sys
import threading
from typing import Optional

from elevenlabs import play
from elevenlabs.client import ElevenLabs

# Ensure parent project modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.tts_cache import generate_with_cache
from modules.jarvis_quotes import (
    get_acknowledgment,
    get_processing_line,
    get_completion_line,
    get_error_line,
    get_interview_filler,
    get_startup_line,
    get_all_cacheable_lines,
)

from server.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    WHISPER_MODEL,
    TEXT_INPUT_MODE,
)
from server.cleanup import clean_transcription

logger = logging.getLogger("scholarship-assistant")

# --- TTS (ElevenLabs) ---

_elevenlabs_client: Optional[ElevenLabs] = None


def _get_elevenlabs_client() -> ElevenLabs:
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _elevenlabs_client


def speak(text: str, cache: bool = False) -> None:
    """
    Speak text aloud via ElevenLabs TTS.

    Args:
        text: The text to speak.
        cache: If True, cache the audio on disk. Use for fixed/repeatable
               segments (intro lines, filler responses, etc.).
    With TEXT_INPUT_MODE=1, prints to console instead (avoids audio stack).
    """
    if not text.strip():
        return
    if TEXT_INPUT_MODE:
        print(f"[VOICE] {text}")
        return
    try:
        client = _get_elevenlabs_client()
        audio_bytes = generate_with_cache(
            client=client,
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
            cache=cache,
        )
        play(audio_bytes)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Fallback: print to console so the user still sees the message
        print(f"[VOICE] {text}")


def speak_async(text: str, cache: bool = False) -> threading.Thread:
    """Speak text in a background thread (non-blocking)."""
    t = threading.Thread(target=speak, args=(text, cache), daemon=True)
    t.start()
    return t


# --- Filler / acknowledgment helpers ---


def speak_acknowledgment() -> None:
    """Play a random short acknowledgment (cached) so the user knows we heard them."""
    speak(get_acknowledgment(), cache=True)


def speak_acknowledgment_async() -> threading.Thread:
    """Play a random acknowledgment in the background (non-blocking)."""
    return speak_async(get_acknowledgment(), cache=True)


def speak_processing() -> None:
    """Play a random 'processing/thinking' line (cached)."""
    speak(get_processing_line(), cache=True)


def speak_completion() -> None:
    """Play a random task-completion line (cached)."""
    speak(get_completion_line(), cache=True)


def speak_error() -> None:
    """Play a random error/warning line (cached)."""
    speak(get_error_line(), cache=True)


def speak_interview_filler() -> None:
    """Play a random interview transition filler (cached)."""
    speak(get_interview_filler(), cache=True)


def speak_startup() -> None:
    """Play a random startup/boot line (cached)."""
    speak(get_startup_line(), cache=True)


def warmup_cache() -> int:
    """
    Pre-generate and cache all filler/acknowledgment audio.
    Call once at startup to eliminate latency on first use.
    Returns the number of lines generated.
    """
    client = _get_elevenlabs_client()
    lines = get_all_cacheable_lines()
    generated = 0
    for line in lines:
        try:
            generate_with_cache(
                client=client,
                text=line,
                voice=ELEVENLABS_VOICE_ID,
                model=ELEVENLABS_MODEL,
                cache=True,
            )
            generated += 1
        except Exception as e:
            logger.warning(f"Cache warmup failed for '{line[:30]}...': {e}")
    logger.info(f"TTS cache warmup complete: {generated}/{len(lines)} lines cached")
    return generated


# --- STT (openai-whisper via alternative_stt) ---

_recorder = None
_recorder_lock = threading.Lock()


def _get_recorder():
    """Lazy-init the whisper-based recorder (heavy import)."""
    global _recorder
    if _recorder is None:
        with _recorder_lock:
            if _recorder is None:
                from server.audio_utils import suppress_alsa_stderr, _quiet_stderr
                suppress_alsa_stderr()
                with _quiet_stderr():
                    from alternative_stt import SimpleAudioRecorder
                    _recorder = SimpleAudioRecorder(
                        model_name=WHISPER_MODEL,
                        silence_duration=1.5,
                        print_transcription_time=True,
                    )
    return _recorder


def listen(cleanup: bool = True) -> str:
    """
    Listen for speech and return the transcribed text.
    Blocks until the user stops speaking (silence-based VAD).
    If cleanup=True, runs filler removal and punctuation fixes.
    With TEXT_INPUT_MODE=1, uses stdin instead of mic (for testing without audio).
    """
    if TEXT_INPUT_MODE:
        text = input("> ").strip()
        if text:
            logger.info(f"Heard: {text}")
            print(f"Heard: {text}")
        if cleanup and text:
            text = clean_transcription(text)
        return text
    recorder = _get_recorder()
    text = recorder.text_blocking()
    if text:
        logger.info(f"Heard: {text}")
        print(f"Heard: {text}")
    if cleanup and text:
        text = clean_transcription(text)
    return text


def listen_raw() -> str:
    """Listen and return raw transcription without cleanup."""
    return listen(cleanup=False)


# --- Combined voice interaction helpers ---


def ask_and_listen(question: str, cleanup: bool = True) -> str:
    """Speak a question, then listen for the user's response."""
    speak(question)
    return listen(cleanup=cleanup)


# Confirmation result: "confirmed" | "correction" | "skip"
CONFIRM_SKIP = "skip"
CONFIRM_CONFIRMED = "confirmed"
CONFIRM_CORRECTION = "correction"

CONFIRM_SKIP_PHRASES = {"skip", "pass", "next", "skip it", "pass on that"}


def _normalize_confirmation_response(response: str) -> str:
    """
    Normalize confirmation response for yes/no matching.
    Handles y/n, letter-by-letter spelling (n o, y e s), and typed variations.
    """
    if not response:
        return ""
    raw = response.lower().strip().rstrip(".")
    collapsed = "".join(raw.split())
    if collapsed in ("no", "n"):
        return "no"
    if collapsed in ("yes", "y"):
        return "yes"
    return raw


def _is_confirm_skip(response: str) -> bool:
    """Check if response is a skip phrase during confirmation."""
    normalized = response.lower().strip()
    return normalized in CONFIRM_SKIP_PHRASES or "".join(normalized.split()) == "skip"


def _parse_confirm_response(response: str) -> str:
    """
    Parse confirmation response into "yes", "no", "skip", or "" (ambiguous).
    """
    if not response:
        return ""
    if _is_confirm_skip(response):
        return "skip"
    normalized = _normalize_confirmation_response(response)
    affirmatives = {
        "yes", "yeah", "yep", "correct", "right", "that's right",
        "sure", "yup", "uh huh", "that's correct", "sounds right",
    }
    negatives = {
        "no", "nope", "wrong", "incorrect", "not right", "nah",
        "that's wrong", "not correct",
    }
    if normalized in affirmatives:
        return "yes"
    if normalized in negatives:
        return "no"
    return ""


def confirm(question: str, proposed_value: str) -> tuple[str, str]:
    """
    Voice-confirm a value. Loops until user says yes, no, or skip.

    Speaks: "{question} I have {proposed_value}. Is that right?"
    Returns: (CONFIRM_CONFIRMED, "") or (CONFIRM_CORRECTION, correction) or (CONFIRM_SKIP, "")

    Accepts y/n, yes/no, skip, and re-asks on ambiguous responses.
    """
    prompt = f"{question} I have: {proposed_value}. Is that right?"

    while True:
        response = ask_and_listen(prompt)
        parsed = _parse_confirm_response(response)
        if parsed == "yes":
            return (CONFIRM_CONFIRMED, "")
        if parsed == "skip":
            return (CONFIRM_SKIP, "")
        if parsed == "no":
            correction = ask_and_listen("What should it be instead? You can say skip to leave this blank.")
            if _is_confirm_skip(correction) or not correction.strip():
                return (CONFIRM_SKIP, "")
            return (CONFIRM_CORRECTION, correction)
        speak("Please say yes, no, or skip.", cache=False)


def _format_phone_for_readback(value: str) -> str:
    """Format a phone number for clear spoken readback (digit groups)."""
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) == 10:
        return f"{digits[:3]}, {digits[3:6]}, {digits[6:]}"
    if len(digits) == 11 and digits[0] == "1":
        return f"1, {digits[1:4]}, {digits[4:7]}, {digits[7:]}"
    return value


def confirm_with_explicit_readback(
    field_label: str,
    proposed_value: str,
    *,
    readback_style: str = "normal",
) -> tuple[bool, str]:
    """
    Like confirm(), but always does an explicit readback first so the user
    can verify we heard correctly. Use for critical fields (phone, SSN, etc.).

    readback_style: "normal" | "phone" | "digits"
      - "phone": format as digit groups (555, 123, 4567)
      - "digits": spell digit-by-digit for max clarity
      - "normal": same as confirm()
    """
    if readback_style == "phone":
        display = _format_phone_for_readback(proposed_value)
    elif readback_style == "digits":
        digits = "".join(c for c in proposed_value if c.isdigit())
        display = ", ".join(digits) if digits else proposed_value
    else:
        display = proposed_value
    return confirm(f"For {field_label}:", display)
