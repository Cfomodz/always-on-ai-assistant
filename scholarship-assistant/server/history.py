"""
Application History & Dedup Engine.
Tracks which scholarships have been applied to and prevents duplicates.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from thefuzz import fuzz

from server.config import HISTORY_PATH, DEDUP_SIMILARITY_THRESHOLD, ensure_data_dir

logger = logging.getLogger("scholarship-assistant")


def load_history() -> list[dict]:
    """Load application history from disk."""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def save_history(history: list[dict]) -> None:
    """Write history to disk."""
    ensure_data_dir()
    tmp_path = HISTORY_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(history, f, indent=2)
    tmp_path.replace(HISTORY_PATH)


def add_record(
    url: str,
    scholarship_name: str,
    organization: str,
    amount: str = "",
    deadline: str = "",
    status: str = "submitted",
    fields_filled: int = 0,
    fields_manual: int = 0,
    essays: Optional[list[str]] = None,
    notes: str = "",
) -> dict:
    """Create and persist a new application history record."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "scholarship_name": scholarship_name,
        "organization": organization,
        "amount": amount,
        "deadline": deadline,
        "status": status,
        "fields_filled": fields_filled,
        "fields_manual": fields_manual,
        "essays": essays or [],
        "notes": notes,
    }

    history = load_history()
    history.append(record)
    save_history(history)

    logger.info(f"History: recorded application to {scholarship_name} ({organization})")
    return record


def check_duplicate(
    scholarship_name: str,
    organization: str,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> Optional[dict]:
    """
    Check if a scholarship has already been applied to.
    Uses fuzzy matching on normalized name + organization.

    Returns the matching history record if found, else None.
    """
    if not scholarship_name and not organization:
        return None

    history = load_history()

    for record in history:
        name_score = fuzz.token_sort_ratio(
            scholarship_name.lower(), record.get("scholarship_name", "").lower()
        )
        org_score = fuzz.token_sort_ratio(
            organization.lower(), record.get("organization", "").lower()
        )

        # Weight: name matters more than org
        combined_score = (name_score * 0.6 + org_score * 0.4) / 100.0

        if combined_score >= threshold:
            logger.info(
                f"Dedup: matched '{scholarship_name}' to existing "
                f"'{record['scholarship_name']}' (score={combined_score:.2f})"
            )
            return record

    return None


def extract_scholarship_info(page_context: dict) -> dict:
    """
    Use DeepSeek to extract scholarship name + organization from page metadata.
    page_context should contain: title, headings, url, visible_text (snippet).
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from modules.deepseek import json_prompt

    prompt = f"""Extract the scholarship name and organization from this page metadata.
If you can't determine one or both, use empty string.

Page title: {page_context.get('title', '')}
URL: {page_context.get('url', '')}
Headings: {', '.join(page_context.get('headings', []))}
Visible text snippet: {page_context.get('visible_text', '')[:1000]}

Respond with JSON:
{{"scholarship_name": "string", "organization": "string", "amount": "string", "deadline": "string"}}"""

    try:
        return json_prompt(prompt)
    except Exception as e:
        logger.error(f"Failed to extract scholarship info: {e}")
        return {
            "scholarship_name": "",
            "organization": "",
            "amount": "",
            "deadline": "",
        }
