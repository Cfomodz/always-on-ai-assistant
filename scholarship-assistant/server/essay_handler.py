"""
Essay / Open-Ended Flow — Handles transcription of essay responses.
No AI writing. Voice transcription only with readback confirmation.
"""

import logging
from typing import Optional

from server.profile_manager import load_profile, add_essay
from server.voice import speak, ask_and_listen, listen, speak_acknowledgment
from server.cleanup import clean_transcription

logger = logging.getLogger("scholarship-assistant")

# Similarity threshold for reusing a previous essay
ESSAY_REUSE_THRESHOLD = 0.7


def _find_similar_essay(prompt_text: str, essays: dict) -> Optional[tuple[str, str]]:
    """
    Check if the profile has a previously transcribed essay that matches this prompt.
    Returns (prompt_key, essay_text) or None.
    """
    if not essays:
        return None

    from thefuzz import fuzz

    prompt_lower = prompt_text.lower()
    best_match = None
    best_score = 0

    for key, text in essays.items():
        score = fuzz.token_sort_ratio(prompt_lower, key.lower()) / 100.0
        if score > best_score:
            best_score = score
            best_match = (key, text)

    if best_match and best_score >= ESSAY_REUSE_THRESHOLD:
        return best_match
    return None


def _summarize_prompt(prompt_text: str) -> str:
    """Create a short key for storing this essay prompt in the profile."""
    # Truncate to first 80 chars and clean up
    summary = prompt_text.strip()[:80]
    if len(prompt_text) > 80:
        summary = summary.rsplit(" ", 1)[0] + "..."
    return summary


def handle_essay(field_id: str, label: str) -> Optional[str]:
    """
    Handle an essay/open-ended field via voice interaction.

    Flow:
    1. Check for similar previous essay → offer reuse
    2. If new: read prompt → record → cleanup → readback → confirm
    3. Save to profile for future reuse
    4. Return the final text to fill

    Args:
        field_id: The DOM field ID
        label: The essay prompt text

    Returns:
        The essay text to fill, or None if skipped.
    """
    profile = load_profile()
    essays = profile.get("essays", {})

    # Check for similar previous essay
    similar = _find_similar_essay(label, essays)

    if similar:
        prev_key, prev_text = similar
        speak(
            f"I found a similar essay you wrote before for the prompt: {prev_key}. "
            "Let me read it to you."
        )

        # Read back the previous essay
        speak(prev_text)

        response = ask_and_listen(
            "Would you like to reuse this essay, modify it, or start fresh?"
        )
        response_lower = response.lower().strip().rstrip(".")

        if "reuse" in response_lower or "use" in response_lower or "same" in response_lower:
            speak_acknowledgment()
            logger.info(f"Essay: reusing previous for field {field_id}")
            return prev_text
        elif "fresh" in response_lower or "new" in response_lower or "start over" in response_lower:
            pass  # Fall through to new essay flow
        else:
            # Treat as modification intent — but we don't AI-modify, so re-record
            speak("Got it. Let's record a new response.")

    # New essay flow
    speak(f"Here's the essay prompt: {label}")
    speak("Go ahead and speak your response. Take your time — I'll wait for you to finish.")

    # Listen with no cleanup for essay (preserve natural speech more)
    raw_text = listen(cleanup=False)

    if not raw_text:
        speak("I didn't catch anything. Would you like to try again, or skip this one?")
        retry = ask_and_listen("Try again or skip?")
        if "skip" in retry.lower():
            return None
        raw_text = listen(cleanup=False)
        if not raw_text:
            speak("Still nothing. I'll skip this one for now.")
            return None

    # Light cleanup only
    cleaned = clean_transcription(raw_text)

    # Readback
    speak("Here's what I got:")
    speak(cleaned)

    response = ask_and_listen("Sound good, or would you like to redo it?")
    response_lower = response.lower().strip().rstrip(".")

    if "redo" in response_lower or "again" in response_lower or "no" in response_lower:
        speak("Let's try again. Go ahead.")
        raw_text = listen(cleanup=False)
        if raw_text:
            cleaned = clean_transcription(raw_text)
            speak("Here's the new version:")
            speak(cleaned)
            final_ok = ask_and_listen("Good to go?")
            if "no" in final_ok.lower():
                speak("I'll use this version for now. You can edit it manually on the page.")

    # Save to profile for future reuse
    prompt_key = _summarize_prompt(label)
    add_essay(prompt_key, cleaned)
    logger.info(f"Essay: saved new essay for '{prompt_key}' ({len(cleaned)} chars)")

    return cleaned
