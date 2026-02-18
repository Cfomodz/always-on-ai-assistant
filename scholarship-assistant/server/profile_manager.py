"""
Profile Manager — CRUD operations on the user's scholarship profile JSON.
Stored at ~/.scholarship-assistant/profile.json
"""

import json
import copy
from pathlib import Path
from typing import Any, Optional

from server.config import PROFILE_PATH, PROFILE_VERSION, ensure_data_dir


def _empty_profile() -> dict:
    """Return a blank profile skeleton with all categories."""
    return {
        "_version": PROFILE_VERSION,
        "personal": {
            "full_legal_name": "",
            "preferred_name": "",
            "pronouns": "",
            "date_of_birth": "",
            "age": "",
            "ssn_last4": "",
            "phone": "",
            "email": "",
            "mailing_address": "",
            "citizenship": "",
            "residency_status": "",
            "state_of_residence": "",
            "race_ethnicity": "",
            "gender_identity": "",
            "marital_status": "",
            "veteran_status": "",
        },
        "disability": {
            "disability_status": "",
            "disability_types": [],
            "specific_conditions": [],
            "accommodations": "",
            "documentation_status": "",
        },
        "education_current": {
            "university_name": "",
            "campus": "",
            "degree_type": "",
            "majors": [],
            "minors": [],
            "concentration": "",
            "expected_enrollment_date": "",
            "expected_graduation_date": "",
            "enrollment_status": "",
            "current_gpa": "",
            "student_id": "",
        },
        "education_history": {
            "prior_institutions": [],
            "high_school": {
                "name": "",
                "graduation_year": "",
                "gpa": "",
                "class_rank": "",
            },
            "standardized_tests": [],
            "certifications": [],
            "relevant_coursework": [],
            "credits_transferred": "",
        },
        "professional": {
            "years_of_experience": "",
            "current_employer": "",
            "current_title": "",
            "current_dates": "",
            "prior_positions": [],
            "skills": [],
            "programming_languages": [],
            "technologies": [],
            "professional_memberships": [],
        },
        "financial": {
            "fafsa_filed": "",
            "efc": "",
            "household_income_range": "",
            "dependents": "",
            "current_aid": [],
            "employment_status_during_school": "",
        },
        "extracurricular": {
            "volunteer_work": [],
            "leadership_roles": [],
            "awards_honors": [],
            "publications": [],
            "organizational_memberships": [],
        },
        "essays": {},
    }


def load_profile() -> dict:
    """Load profile from disk, or return empty profile if none exists."""
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return _empty_profile()


def save_profile(profile: dict) -> None:
    """Write profile to disk (atomic-ish via write-then-rename)."""
    ensure_data_dir()
    tmp_path = PROFILE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(profile, f, indent=2)
    tmp_path.replace(PROFILE_PATH)


def profile_exists() -> bool:
    """Check whether a profile file exists on disk."""
    return PROFILE_PATH.exists()


def get_field(profile: dict, dot_key: str) -> Any:
    """
    Get a value from the profile using dot-notation.
    e.g. get_field(profile, "personal.full_legal_name")
    """
    keys = dot_key.split(".")
    current = profile
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def set_field(profile: dict, dot_key: str, value: Any) -> dict:
    """
    Set a value in the profile using dot-notation.
    Creates intermediate dicts if needed. Returns the modified profile.
    """
    keys = dot_key.split(".")
    current = profile
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return profile


def update_profile(updates: dict) -> dict:
    """
    Load profile, apply a dict of dot-key: value updates, save and return.
    """
    profile = load_profile()
    for dot_key, value in updates.items():
        set_field(profile, dot_key, value)
    save_profile(profile)
    return profile


def add_essay(prompt_summary: str, transcription: str) -> dict:
    """Add or overwrite an essay entry in the profile."""
    profile = load_profile()
    profile["essays"][prompt_summary] = transcription
    save_profile(profile)
    return profile


def get_flat_profile_for_matching(profile: Optional[dict] = None) -> dict:
    """
    Return a flattened view of the profile suitable for sending to DeepSeek.
    Strips empty values so the LLM only sees what's populated.
    """
    if profile is None:
        profile = load_profile()

    flat = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("_"):
                    continue
                new_key = f"{prefix}.{k}" if prefix else k
                _flatten(v, new_key)
        elif isinstance(obj, list):
            if obj:
                flat[prefix] = obj
        elif obj not in (None, "", []):
            flat[prefix] = obj

    _flatten(profile)
    return flat
