"""
Profile Import — Digest Q&A pairs or raw text into the user profile via DeepSeek.
Accepts tab-separated Question/Answer/LastAnswered or free-form text.
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.deepseek import json_prompt
from server.profile_manager import load_profile, get_flat_profile_for_matching, update_profile

logger = logging.getLogger("scholarship-assistant")

IMPORT_SYSTEM_PROMPT = """You are a scholarship profile data extractor. You receive:
1. The current user profile (flattened, dot-notation keys)
2. Imported content: either structured Q&A pairs (Question / Answer(s) / Last Answered) or raw text

Your job: Extract ALL facts from the imported content and produce profile updates. The profile supports DYNAMIC keys—you are NOT limited to the schema below. Add new keys whenever the data doesn't fit an existing field.

Standard schema (prefer these when they fit):
- personal: full_legal_name, preferred_name, pronouns, date_of_birth, age, ssn_last4, phone, email, mailing_address, citizenship, residency_status, state_of_residence, race_ethnicity, gender_identity, marital_status, veteran_status
- disability: disability_status, disability_types (array), specific_conditions (array), accommodations, documentation_status
- education_current: university_name, campus, degree_type, majors (array), minors (array), concentration, expected_enrollment_date, expected_graduation_date, enrollment_status, current_gpa, student_id
- education_history: prior_institutions (array), high_school ({name, graduation_year, gpa, class_rank, state, county, city, country}), standardized_tests (array), certifications (array), relevant_coursework (array), credits_transferred
- professional: years_of_experience, current_employer, current_title, current_dates, prior_positions (array), skills (array), programming_languages (array), technologies (array), professional_memberships (array)
- financial: fafsa_filed, efc, household_income_range, dependents, current_aid (array), employment_status_during_school
- extracurricular: volunteer_work (array), leadership_roles (array), awards_honors (array), publications (array), organizational_memberships (array)
- extended: dynamic key-value store for ANY data that doesn't fit above. Use descriptive snake_case keys.

CRITICAL: For data with no standard mapping, ADD IT UNDER extended. Examples:
- "Family members in Armed Forces?" → extended.family_military_branches (array, e.g. ["Army"])
- "Medical specialty you plan to pursue?" → extended.medical_specialties_planned (array)
- "Licensures you hold?" → extended.licensures_held (array) or education_history.certifications
- "Employers of family members?" → extended.employers_family (array)
- "Religious affiliation?" → extended.religious_affiliation or personal.religious_affiliation (add if missing)
- "Sports/interests?" → extended.sports_interests, extended.art_interests, extended.industry_interests, etc.
- "Greek/social orgs?" → extended.greek_memberships
- Any other scholarship-specific question → extended.{descriptive_key}

Rules:
- For list/array fields: if the answer contains multiple items (comma-separated), split into an array.
- Skip answers that are "N/A", "None", "—", "-", or clearly indicate "not applicable"
- Use the most recent answer when duplicates exist (prefer later Last Answered date)
- Normalize values: trim whitespace, use consistent casing where appropriate
- You MAY create new keys in any category. Use extended.* for ad-hoc/schema-less data.
- Don't invent data. Only extract what is explicitly stated.
- Merge array values with existing profile data when updating; deduplicate.

Respond with valid JSON:
{
  "updates": {
    "dot.key": "value or [array]",
    "extended.new_key": ["item1", "item2"]
  },
  "summary": "Brief human-readable summary of what was imported (1-2 sentences)",
  "skipped": ["Reason for skipping ambiguous items, if any"]
}"""


def _parse_qa_rows(content: str) -> list[dict]:
    """Parse tab-separated Q&A format. Returns list of {question, answer, last_answered}."""
    rows = []
    lines = content.strip().splitlines()
    if not lines:
        return []

    # Check for header row (Question, Answer(s), Last Answered)
    first = lines[0].lower()
    start_idx = 0
    if "question" in first and ("answer" in first or "answer(s)" in first):
        start_idx = 1

    for line in lines[start_idx:]:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            rows.append({
                "question": parts[0],
                "answer": parts[1],
                "last_answered": parts[2] if len(parts) > 2 else "",
            })
    return rows


def _detect_format(content: str) -> str:
    """Return 'qa' if content looks like Q&A table, else 'raw'."""
    lines = content.strip().splitlines()
    if not lines:
        return "raw"

    first = lines[0].lower()
    if "question" in first and "answer" in first:
        return "qa"

    # If we have tab-separated rows with at least 2 columns
    for line in lines[:5]:
        if "\t" in line and len(line.split("\t")) >= 2:
            return "qa"

    return "raw"


def import_into_profile(content: str, dry_run: bool = False) -> dict:
    """
    Process imported content (Q&A pairs or raw text) via DeepSeek and merge into profile.

    Args:
        content: Raw pasted text. Can be tab-separated Q&A (Question, Answer(s), Last Answered) or free-form.
        dry_run: If True, return proposed updates without applying.

    Returns:
        {
            "applied": bool,
            "updates": {dot_key: value, ...},
            "summary": str,
            "skipped": list[str],
            "error": str | None
        }
    """
    content = content.strip()
    if not content:
        return {"applied": False, "updates": {}, "summary": "No content to import.", "skipped": [], "error": None}

    profile = load_profile()
    flat_profile = get_flat_profile_for_matching(profile)

    format_type = _detect_format(content)
    if format_type == "qa":
        rows = _parse_qa_rows(content)
        if not rows:
            return {"applied": False, "updates": {}, "summary": "Could not parse Q&A rows.", "skipped": [], "error": None}
        import_payload = json.dumps({"type": "qa_pairs", "pairs": rows}, indent=2)
    else:
        import_payload = json.dumps({"type": "raw_text", "content": content}, indent=2)

    prompt = f"""{IMPORT_SYSTEM_PROMPT}

Current profile (non-empty fields only):
{json.dumps(flat_profile, indent=2)}

Imported content:
{import_payload}

Extract all applicable facts and produce the updates object. Respond with JSON only."""

    try:
        result = json_prompt(prompt)
    except Exception as e:
        logger.error(f"Profile import DeepSeek error: {e}")
        return {
            "applied": False,
            "updates": {},
            "summary": "",
            "skipped": [],
            "error": str(e),
        }

    updates = result.get("updates", {})
    if not isinstance(updates, dict):
        updates = {}

    summary = result.get("summary", "Import completed.")
    skipped = result.get("skipped", [])
    if not isinstance(skipped, list):
        skipped = [str(skipped)]

    if dry_run:
        return {
            "applied": False,
            "updates": updates,
            "summary": summary,
            "skipped": skipped,
            "error": None,
        }

    if updates:
        try:
            update_profile(updates)
            logger.info(f"Profile import: applied {len(updates)} updates")
        except Exception as e:
            logger.error(f"Profile import save error: {e}")
            return {
                "applied": False,
                "updates": updates,
                "summary": summary,
                "skipped": skipped,
                "error": str(e),
            }

    return {
        "applied": bool(updates),
        "updates": updates,
        "summary": summary,
        "skipped": skipped,
        "error": None,
    }
