from typing import List, Dict
import logging
import os
from modules.deepseek import conversational_prompt as deepseek_conversational_prompt
from modules.ollama import conversational_prompt as ollama_conversational_prompt
from modules.utils import build_file_name_session
from modules.tts_cache import generate_with_cache
from RealtimeTTS import TextToAudioStream, SystemEngine
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import pyttsx3
import time
from modules.assistant_config import get_config


class PlainAssistant:
    def __init__(self, logger: logging.Logger, session_id: str):
        self.logger = logger
        self.session_id = session_id
        self.conversation_history = []

        # Get voice configuration
        self.voice_type = get_config("base_assistant.voice")
        self.elevenlabs_voice = get_config("base_assistant.elevenlabs_voice")
        self.brain = get_config("base_assistant.brain")

        # Initialize appropriate TTS engine
        if self.voice_type == "local":
            self.logger.info("🔊 Initializing local TTS engine")
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)  # Speed of speech
            self.engine.setProperty("volume", 1.0)  # Volume level
        elif self.voice_type == "realtime-tts":
            self.logger.info("🔊 Initializing RealtimeTTS engine")
            self.engine = SystemEngine()
            self.stream = TextToAudioStream(
                self.engine, frames_per_buffer=256, playout_chunk_size=1024
            )
        elif self.voice_type == "elevenlabs":
            self.logger.info("🔊 Initializing ElevenLabs TTS engine")
            self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        else:
            raise ValueError(f"Unsupported voice type: {self.voice_type}")

    def process_text(self, text: str) -> str:
        """Process text input and generate response"""
        try:
            # Check if text matches our last response
            if (
                self.conversation_history
                and text.strip().lower()
                in self.conversation_history[-1]["content"].lower()
            ):
                self.logger.info("🤖 Ignoring own speech input")
                return ""

            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})

            # Generate response using configured brain
            self.logger.info(f"🤖 Processing text with {self.brain}...")
            if self.brain.startswith("ollama:"):
                model_no_prefix = ":".join(self.brain.split(":")[1:])
                response = ollama_conversational_prompt(
                    self.conversation_history, model=model_no_prefix
                )
            else:
                response = deepseek_conversational_prompt(self.conversation_history)

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})

            # Speak the response
            self.speak(response)

            return response

        except Exception as e:
            self.logger.error(f"❌ Error occurred: {str(e)}")
            raise

    def speak(self, text: str, cache: bool = False):
        """
        Convert text to speech using configured engine.

        Args:
            text: The text to speak.
            cache: If True and using ElevenLabs, cache audio to disk.
                   Use for fixed/repeatable segments.
        """
        try:
            self.logger.info(f"🔊 Speaking: {text}")

            if self.voice_type == "local":
                self.engine.say(text)
                self.engine.runAndWait()

            elif self.voice_type == "realtime-tts":
                self.stream.feed(text)
                self.stream.play()

            elif self.voice_type == "elevenlabs":
                audio_bytes = generate_with_cache(
                    client=self.elevenlabs_client,
                    text=text,
                    voice=self.elevenlabs_voice,
                    model="eleven_turbo_v2",
                    cache=cache,
                )
                play(audio_bytes)

            self.logger.info(f"🔊 Spoken: {text}")

        except Exception as e:
            self.logger.error(f"❌ Error in speech synthesis: {str(e)}")
            raise
