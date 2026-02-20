"""
Optional Deepseek sanity check for ambiguous interview responses.
When enabled, runs transcribed responses through Deepseek to disambiguate
e.g. "no" (decline vs literal value) or nonsensical transcriptions.

Enable via env: SCHOLARSHIP_ASSISTANT_SANITY_CHECK=1
"""

import logging
import os

logger = logging.getLogger("scholarship-assistant")

SANITY_CHECK_ENABLED = os.environ.get("SCHOLARSHIP_ASSISTANT_SANITY_CHECK", "").lower() in ("1", "true", "yes")


def interpret_response(question: str, response: str, is_optional: bool) -> str:
    """
    Use Deepseek to interpret ambiguous responses.
    Returns: "skip" | "use" | "reask"

    - "skip": user meant to decline (no value)
    - "use": treat response as the literal value to store
    - "reask": unclear, suggest re-asking
    """
    if not SANITY_CHECK_ENABLED or not response or not question:
        return "use"

    response_lower = response.lower().strip()
    # Short ambiguous responses that might mean "no value" vs "literal"
    ambiguous = response_lower in ("no", "nope", "none", "nothing", "nah")

    if not ambiguous:
        return "use"

    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from modules.deepseek import json_prompt

        result = json_prompt(
            f"""The user was asked: "{question}"
They responded: "{response}"

For optional questions, "no"/"nope" often means "I don't have one" (skip).
For confirmation steps, "no" means "that's wrong" (correction needed).

Is the question optional? {is_optional}

Interpret the user's intent. Respond with JSON only:
{{"interpretation": "skip" | "use" | "reask", "reason": "brief explanation"}}

- skip: user meant to decline/skip (no value to store)
- use: the literal value "{response}" should be stored
- reask: unclear, should ask again
"""
        )
        interp = result.get("interpretation", "use")
        logger.info(f"Sanity check: {response!r} -> {interp} ({result.get('reason', '')})")
        return interp if interp in ("skip", "use", "reask") else "use"
    except Exception as e:
        logger.warning(f"Sanity check failed: {e}")
        return "use"
