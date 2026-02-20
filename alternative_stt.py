#!/usr/bin/env python3
"""
Alternative STT implementation using openai-whisper instead of RealtimeSTT.
Uses sounddevice for recording (avoids PyAudio segfaults on some Linux setups).
"""

# Suppress ALSA stderr before audio libs load (Linux only)
def _suppress_alsa():
    import platform
    if platform.system() == "Linux":
        try:
            from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
            _handler = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)(lambda *a: None)
            cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_handler)
        except Exception:
            pass
_suppress_alsa()

import tempfile
import os
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


def _rms(chunk: np.ndarray) -> float:
    """RMS of audio chunk (float32)."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


class SimpleAudioRecorder:
    def __init__(
        self,
        model_name: str = "tiny.en",
        chunk_size: int = 1024,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        print_transcription_time: bool = True,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.print_transcription_time = print_transcription_time

        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")

        self.is_recording = False
        self.recording_thread = None

    def _record_audio(self, callback: Callable[[str], None]):
        """Record audio with sounddevice (avoids PyAudio segfaults)."""
        q: queue.Queue = queue.Queue()
        frames_list: list[np.ndarray] = []
        speech_detected = False
        silence_chunks = 0
        chunks_per_silence = int(self.silence_duration * self.sample_rate / self.chunk_size)

        def stream_callback(indata: np.ndarray, _frames: int, _time, status):
            if status:
                print(status, end="", flush=True)
            q.put(indata.copy())

        print("🎤 Listening for speech...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=stream_callback,
        ):
            while self.is_recording:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms_val = _rms(chunk)
                is_silence = rms_val < self.silence_threshold

                if not is_silence:
                    if not speech_detected:
                        print("🎤 Speech detected, recording...")
                        speech_detected = True
                    frames_list.append(chunk)
                    silence_chunks = 0
                elif speech_detected:
                    frames_list.append(chunk)
                    silence_chunks += 1
                    if silence_chunks >= chunks_per_silence:
                        # End of utterance
                        if frames_list:
                            audio_data = np.concatenate(frames_list, axis=0)
                            with tempfile.NamedTemporaryFile(
                                suffix=".wav", delete=False
                            ) as tmp:
                                tmp_path = tmp.name
                            sf.write(tmp_path, audio_data, self.sample_rate)
                            try:
                                start = time.time()
                                result = self.model.transcribe(tmp_path, language="en")
                                elapsed = time.time() - start
                                text = result["text"].strip()
                                if text:
                                    if self.print_transcription_time:
                                        print(f"🎤 Transcription took {elapsed:.2f}s")
                                    callback(text)
                            except Exception as e:
                                print(f"❌ Transcription error: {e}")
                            finally:
                                try:
                                    os.unlink(tmp_path)
                                except OSError:
                                    pass
                        frames_list = []
                        speech_detected = False
                        silence_chunks = 0
                        print("🎤 Listening for speech...")
    
    def start(self):
        """Start recording"""
        self.is_recording = True
    
    def stop(self):
        """Stop recording"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
    
    def text(self, callback: Callable[[str], None]):
        """Start listening and call callback with transcribed text"""
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop()
        
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._record_audio, 
            args=(callback,),
            daemon=True
        )
        self.recording_thread.start()
        
        try:
            # Keep the main thread alive
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            self.stop()

    def text_blocking(self) -> str:
        """Block until one utterance is transcribed, then return it. For single-shot listen()."""
        result: list[str] = []
        def on_text(t: str):
            result.append(t)
            self.stop()
        self.text(on_text)
        return result[0] if result else ""

    def __del__(self):
        """Cleanup (sounddevice has no explicit terminate)."""
        pass


# Compatibility alias
AudioToTextRecorder = SimpleAudioRecorder
