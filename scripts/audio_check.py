#!/usr/bin/env python3
"""Audio diagnostics for the voice assistant. Run from project root."""

import json
import os
import sys

# Add project root
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# Try scholarship-assistant path for audio_utils
sa_path = os.path.join(root, "scholarship-assistant")
if os.path.isdir(sa_path) and sa_path not in sys.path:
    sys.path.insert(0, sa_path)

def main():
    try:
        from server.audio_utils import run_audio_diagnostics
    except ImportError:
        # Fallback if not in scholarship-assistant
        import platform
        import subprocess
        result = {"platform": platform.system(), "checks": {}}
        for cmd, key in [(["which", "ffplay"], "ffplay"), (["pactl", "info"], "pulseaudio")]:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True)
                result["checks"][key] = "ok" if p.returncode == 0 else "not found"
            except FileNotFoundError:
                result["checks"][key] = "command not found"
            except Exception as e:
                result["checks"][key] = str(e)
    else:
        result = run_audio_diagnostics()

    print("Audio diagnostics:")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
