"""
Field Matcher — Uses DeepSeek JSON mode to match form fields against the user profile.
Produces confidence-scored match instructions for each field.
"""

import json
import logging
import sys
import os
from typing import Any

# Add parent project root to path so we can import modules/deepseek
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.deepseek import json_prompt
from server.profile_manager import load_profile, get_flat_profile_for_matching
from server.config import AUTO_FILL_THRESHOLD, CONFIRM_THRESHOLD

logger = logging.getLogger("scholarship-assistant")

MATCH_SYSTEM_PROMPT = """You are a scholarship form field matcher. You receive a list of form fields scraped from a scholarship application page and a user profile with their personal information.

For each form field, determine:
1. Whether it matches a profile field (and which one)
2. The correct value to fill in
3. Your confidence (0.0 to 1.0) that the match is correct
4. Whether the field is an essay/open-ended question

Rules:
- For select/dropdown fields, the value MUST be one of the provided options (exact match)
- For radio/checkbox fields, the value MUST be one of the provided option values
- For date fields, infer the expected format from the field and format accordingly
- If a field asks for something not in the profile, mark it as unmatched
- If a field is a textarea with an essay-like prompt (more than a short answer), mark is_essay: true
- Confidence should reflect how certain you are of the mapping AND the value:
  - 0.9-1.0: Direct, unambiguous match (e.g., "Email" → profile.email)
  - 0.7-0.85: Likely match but phrasing differs (e.g., "Cultural background" → race_ethnicity)
  - 0.4-0.65: Possible match but uncertain
  - 0.0-0.3: Very unlikely or no match

Respond with valid JSON matching this exact schema:
{
  "matches": [
    {
      "field_id": "string (the DOM id or name of the field)",
      "profile_key": "string (dot-notation key in profile)",
      "value": "the value to fill (must match field type constraints)",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation of the match"
    }
  ],
  "unmatched": [
    {
      "field_id": "string",
      "label": "string (the field label text)",
      "type": "string (text/select/textarea/etc.)",
      "is_essay": true/false
    }
  ]
}"""


def match_fields(fields: list[dict], profile: dict | None = None) -> dict:
    """
    Send form fields + profile to DeepSeek and get match results.

    Args:
        fields: List of field dicts from the userscript, each with:
            id, label, type, options (if select), required
        profile: Optional profile dict. If None, loads from disk.

    Returns:
        Dict with "matches" and "unmatched" lists.
    """
    if profile is None:
        profile = load_profile()

    flat_profile = get_flat_profile_for_matching(profile)

    prompt_payload = json.dumps({
        "fields": fields,
        "profile": flat_profile,
    }, indent=2)

    full_prompt = f"""{MATCH_SYSTEM_PROMPT}

Here are the form fields and user profile:

{prompt_payload}

Respond with JSON only."""

    logger.info(f"Field matcher: sending {len(fields)} fields to DeepSeek")

    try:
        result = json_prompt(full_prompt)
    except Exception as e:
        logger.error(f"Field matcher DeepSeek error: {e}")
        # Return everything as unmatched on failure
        return {
            "matches": [],
            "unmatched": [
                {
                    "field_id": f.get("id", ""),
                    "label": f.get("label", ""),
                    "type": f.get("type", "text"),
                    "is_essay": f.get("type") == "textarea",
                }
                for f in fields
            ],
        }

    # Validate and normalize the result
    matches = result.get("matches", [])
    unmatched = result.get("unmatched", [])

    logger.info(
        f"Field matcher: {len(matches)} matches, {len(unmatched)} unmatched"
    )

    return {"matches": matches, "unmatched": unmatched}


def categorize_matches(match_result: dict) -> dict:
    """
    Split matches into confidence tiers for the fill pipeline.

    Returns:
        {
            "auto_fill": [...],     # confidence >= 0.8
            "confirm": [...],       # 0.6 <= confidence < 0.8
            "ask": [...],           # confidence < 0.6 and not essay
            "essay": [...],         # unmatched essays
            "skip": [...],          # unmatched non-essays
        }
    """
    auto_fill = []
    needs_confirm = []
    needs_ask = []
    essays = []
    skip = []

    for match in match_result.get("matches", []):
        confidence = match.get("confidence", 0)
        if confidence >= AUTO_FILL_THRESHOLD:
            auto_fill.append(match)
        elif confidence >= CONFIRM_THRESHOLD:
            needs_confirm.append(match)
        else:
            needs_ask.append(match)

    for unmatched in match_result.get("unmatched", []):
        if unmatched.get("is_essay", False):
            essays.append(unmatched)
        else:
            skip.append(unmatched)

    return {
        "auto_fill": auto_fill,
        "confirm": needs_confirm,
        "ask": needs_ask,
        "essay": essays,
        "skip": skip,
    }
