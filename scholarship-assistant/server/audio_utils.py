"""
Audio utilities — ALSA stderr suppression and diagnostics.

Suppresses noisy ALSA/JACK error messages that often appear on Linux
when no physical audio device is configured (e.g. headless, SSH, or
PulseAudio routing). Audio typically still works; the messages are cosmetic.
"""

import contextlib
import logging
import os
import platform
import subprocess
import sys

logger = logging.getLogger("scholarship-assistant")


@contextlib.contextmanager
def _quiet_stderr():
    """Temporarily redirect stderr to /dev/null (for noisy audio lib init)."""
    if platform.system() != "Linux":
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
    except Exception:
        yield


def _set_playback_env() -> None:
    """Prefer PulseAudio for ffplay/SDL when on Linux (reduces ALSA probing)."""
    if platform.system() == "Linux" and "SDL_AUDIODRIVER" not in os.environ:
        os.environ.setdefault("SDL_AUDIODRIVER", "pulse")

_SUPPRESSED = False


def suppress_alsa_stderr() -> bool:
    """
    Suppress ALSA lib error messages to stderr.
    Must be called before any audio library (PyAudio, sounddevice, etc.) is loaded.

    Returns True if suppression was applied, False if skipped (non-Linux or lib not found).
    """
    global _SUPPRESSED
    if _SUPPRESSED:
        return True

    if platform.system() != "Linux":
        return False

    _set_playback_env()

    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

        # ALSA error handler: args are (file, line, function, err, fmt)
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

        def _py_error_handler(file, line, function, err, fmt):
            pass  # Silently ignore

        _c_handler = ERROR_HANDLER_FUNC(_py_error_handler)
        asound = cdll.LoadLibrary("libasound.so.2")
        asound.snd_lib_error_set_handler(_c_handler)
        _SUPPRESSED = True
        logger.debug("ALSA stderr suppression enabled")
        return True
    except OSError:
        # libasound.so.2 not found (e.g. non-ALSA system)
        return False
    except Exception as e:
        logger.debug(f"Could not suppress ALSA stderr: {e}")
        return False


def run_audio_diagnostics() -> dict:
    """Run basic audio diagnostics. Returns a dict of checks."""
    suppress_alsa_stderr()  # Suppress before touching PyAudio/sounddevice
    result = {"platform": platform.system(), "checks": {}}

    # 1. ffplay (for ElevenLabs play)
    try:
        p = subprocess.run(["which", "ffplay"], capture_output=True, text=True)
        result["checks"]["ffplay"] = "ok" if p.returncode == 0 else "not found"
    except Exception as e:
        result["checks"]["ffplay"] = str(e)

    # 2. PulseAudio
    try:
        p = subprocess.run(["pactl", "info"], capture_output=True, text=True)
        result["checks"]["pulseaudio"] = "running" if p.returncode == 0 else "not running"
    except FileNotFoundError:
        result["checks"]["pulseaudio"] = "pactl not installed"
    except Exception as e:
        result["checks"]["pulseaudio"] = str(e)

    # 3. PyAudio (if importable)
    try:
        with _quiet_stderr():
            import pyaudio
            pa = pyaudio.PyAudio()
            dev_count = pa.get_device_count()
            pa.terminate()
        result["checks"]["pyaudio"] = f"ok ({dev_count} devices)"
    except Exception as e:
        result["checks"]["pyaudio"] = str(e)

    # 4. sounddevice (if importable)
    try:
        with _quiet_stderr():
            import sounddevice as sd
            _ = sd.query_devices()
            default_dev = sd.default.device
        result["checks"]["sounddevice"] = f"ok (default: {default_dev})"
    except Exception as e:
        result["checks"]["sounddevice"] = str(e)

    return result


if __name__ == "__main__":
    import json
    d = run_audio_diagnostics()
    print(json.dumps(d, indent=2))
