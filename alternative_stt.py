#!/usr/bin/env python3
"""
Alternative STT implementation using openai-whisper instead of RealtimeSTT
This avoids the ctranslate2 executable stack issue.
"""

import pyaudio
import wave
import tempfile
import os
import whisper
import numpy as np
from typing import Callable, Optional
import threading
import time


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
        
        # Load whisper model
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")
        
        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.recording_thread = None
        
    def _is_silence(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains silence"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        # Calculate RMS (root mean square) to measure volume
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        # Normalize to 0-1 range
        normalized_rms = rms / 32768.0
        return normalized_rms < self.silence_threshold
    
    def _record_audio(self, callback: Callable[[str], None]):
        """Record audio and transcribe when speech is detected"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 Listening for speech...")
        
        while self.is_recording:
            frames = []
            silence_count = 0
            speech_detected = False
            
            # Listen for speech
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                if not self._is_silence(data):
                    if not speech_detected:
                        print("🎤 Speech detected, recording...")
                        speech_detected = True
                    frames.append(data)
                    silence_count = 0
                elif speech_detected:
                    frames.append(data)
                    silence_count += 1
                    # If we've had enough silence, stop recording
                    if silence_count > (self.silence_duration * self.sample_rate / self.chunk_size):
                        break
            
            if frames and speech_detected:
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # Write WAV file
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(frames))
                
                # Transcribe audio
                try:
                    start_time = time.time()
                    result = self.model.transcribe(temp_filename, language="en")
                    transcription_time = time.time() - start_time
                    
                    text = result["text"].strip()
                    if text:
                        if self.print_transcription_time:
                            print(f"🎤 Transcription took {transcription_time:.2f}s")
                        callback(text)
                    
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                
                print("🎤 Listening for speech...")
        
        stream.stop_stream()
        stream.close()
    
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
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


# Compatibility alias
AudioToTextRecorder = SimpleAudioRecorder
