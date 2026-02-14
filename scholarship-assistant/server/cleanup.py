"""
Transcription cleanup — filler removal and basic punctuation fixes.
No AI rewriting. No rephrasing. Just hygiene.
"""

import re

FILLER_WORDS = [
    r"\bum\b",
    r"\buh\b",
    r"\blike\b(?=\s*,)",  # "like," but not "I like cats"
    r"\byou know\b",
    r"\bI mean\b",
    r"\bso\b(?=\s*,)",  # leading "so," filler
    r"\bbasically\b(?=\s*,)",
    r"\bactually\b(?=\s*,)",
    r"\bkind of\b",
    r"\bsort of\b",
]

FILLER_PATTERN = re.compile("|".join(FILLER_WORDS), re.IGNORECASE)


def remove_fillers(text: str) -> str:
    """Remove common filler words/phrases."""
    text = FILLER_PATTERN.sub("", text)
    # Clean up leftover double spaces and orphan commas
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"^\s*,\s*", "", text)
    return text.strip()


def fix_punctuation(text: str) -> str:
    """
    Basic punctuation heuristics:
    - Capitalize first letter of sentences
    - Ensure text ends with a period if no terminal punctuation
    - Fix spacing around punctuation
    """
    if not text:
        return text

    # Fix spacing before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Fix spacing after punctuation (ensure single space)
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Capitalize after sentence-ending punctuation
    text = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )

    # Add period at end if missing terminal punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


def clean_transcription(text: str) -> str:
    """Full cleanup pipeline: fillers → punctuation."""
    text = remove_fillers(text)
    text = fix_punctuation(text)
    return text
