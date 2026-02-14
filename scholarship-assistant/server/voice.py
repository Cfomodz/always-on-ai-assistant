"""
Voice module — ElevenLabs TTS + RealtimeSTT for speech recognition.
Reuses the parent project's ElevenLabs patterns and RealtimeSTT dependency.
"""

import logging
import os
import threading
from typing import Optional

from elevenlabs import play
from elevenlabs.client import ElevenLabs

from server.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    WHISPER_MODEL,
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


def speak(text: str) -> None:
    """Speak text aloud via ElevenLabs TTS."""
    if not text.strip():
        return
    try:
        client = _get_elevenlabs_client()
        audio = client.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
            stream=False,
        )
        play(audio)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Fallback: print to console so the user still sees the message
        print(f"[VOICE] {text}")


def speak_async(text: str) -> threading.Thread:
    """Speak text in a background thread (non-blocking)."""
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()
    return t


# --- STT (RealtimeSTT / whisper) ---

_recorder = None
_recorder_lock = threading.Lock()


def _get_recorder():
    """Lazy-init the AudioToTextRecorder (heavy import)."""
    global _recorder
    if _recorder is None:
        with _recorder_lock:
            if _recorder is None:
                from RealtimeSTT import AudioToTextRecorder

                _recorder = AudioToTextRecorder(
                    spinner=False,
                    post_speech_silence_duration=1.5,
                    compute_type="float32",
                    model=WHISPER_MODEL,
                    beam_size=8,
                    batch_size=25,
                    language="en",
                    print_transcription_time=True,
                )
    return _recorder


def listen(cleanup: bool = True) -> str:
    """
    Listen for speech and return the transcribed text.
    Blocks until the user stops speaking (silence-based VAD).
    If cleanup=True, runs filler removal and punctuation fixes.
    """
    recorder = _get_recorder()
    text = recorder.text()
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


def confirm(question: str, proposed_value: str) -> tuple[bool, str]:
    """
    Voice-confirm a value.
    Speaks: "{question} I have {proposed_value}. Is that right?"
    Returns (confirmed: bool, correction_or_empty: str)
    """
    prompt = f"{question} I have: {proposed_value}. Is that right?"
    response = ask_and_listen(prompt)

    response_lower = response.lower().strip().rstrip(".")
    affirmatives = {"yes", "yeah", "yep", "correct", "right", "that's right", "sure", "yup", "uh huh"}
    negatives = {"no", "nope", "wrong", "incorrect", "not right", "nah"}

    if response_lower in affirmatives:
        return True, ""
    elif response_lower in negatives:
        correction = ask_and_listen("What should it be instead?")
        return False, correction
    else:
        # Treat any other response as a correction
        return False, response
