"""
Configuration and environment loading for the scholarship assistant.
Reuses the parent project's .env for API keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent project root
_parent_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_parent_root / ".env")

# --- API Keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVEN_API_KEY", "")

# --- Paths ---
DATA_DIR = Path.home() / ".scholarship-assistant"
PROFILE_PATH = DATA_DIR / "profile.json"
HISTORY_PATH = DATA_DIR / "history.json"

# --- Server ---
HOST = "127.0.0.1"
PORT = 8741

# --- DeepSeek ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"
DEEPSEEK_MODEL = "deepseek-chat"

# --- ElevenLabs ---
ELEVENLABS_VOICE_ID = "UQoLnPXvf18gaKpLzfb8"
ELEVENLABS_MODEL = "eleven_flash_v2_5"

# --- STT ---
WHISPER_MODEL = "base.en"

# --- Confidence Thresholds ---
AUTO_FILL_THRESHOLD = 0.8
CONFIRM_THRESHOLD = 0.6

# --- Dedup ---
DEDUP_SIMILARITY_THRESHOLD = 0.85

# --- Profile version ---
PROFILE_VERSION = 1


def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
