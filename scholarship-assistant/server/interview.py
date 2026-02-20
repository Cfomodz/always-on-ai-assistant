"""
Init Interview — Voice-driven walkthrough that builds the user's profile.
Triggered on first run (no profile.json) or via --init / POST /init.
"""

import logging
from typing import Callable, Optional

from server.profile_manager import load_profile, save_profile, profile_exists, set_field, get_field
from server.voice import (
    speak,
    ask_and_listen,
    confirm,
    confirm_with_explicit_readback,
    speak_interview_filler,
    CONFIRM_CONFIRMED,
    CONFIRM_CORRECTION,
    CONFIRM_SKIP,
)
from server.sanity_check import interpret_response, SANITY_CHECK_ENABLED

logger = logging.getLogger("scholarship-assistant")

# Fields with expected/valid options — if transcription doesn't match, re-ask for clarification
FIELD_EXPECTED_OPTIONS: dict[str, list[str]] = {
    "personal.pronouns": [
        "he/him",
        "she/her",
        "they/them",
        "he/him or they/them",
        "she/her or they/them",
        "any",
        "prefer not to say",
    ],
}

# Each question: (dot_key, spoken_question, optional_flag, sensitive_flag)
# optional_flag: if True, explain why and respect "skip"
# sensitive_flag: if True, give extra context about why it's collected

INTERVIEW_QUESTIONS = [
    # --- Personal ---
    ("personal.full_legal_name", "What is your full legal name?", False, False),
    ("personal.preferred_name", "Do you go by a different name or nickname?", True, False),
    ("personal.pronouns", "What are your pronouns?", True, False),
    ("personal.date_of_birth", "What is your date of birth?", False, False),
    ("personal.phone", "What's the best phone number to reach you at?", False, False),
    ("personal.email", "What's your email address?", False, False),
    ("personal.mailing_address", "What is your current mailing address?", False, False),
    ("personal.citizenship", "What is your country of citizenship?", False, False),
    ("personal.residency_status", "What is your residency status? For example, U.S. citizen, permanent resident, or visa holder.", False, False),
    ("personal.state_of_residence", "What state do you currently live in?", False, False),
    ("personal.race_ethnicity", "How do you identify your race or ethnicity? Many scholarships ask this for eligibility purposes.", False, True),
    ("personal.gender_identity", "How do you identify your gender?", True, False),
    ("personal.marital_status", "What is your marital status?", True, False),
    ("personal.veteran_status", "Are you a veteran or active-duty military?", False, False),
    ("personal.ssn_last4", "Some applications ask for the last four digits of your Social Security Number for identification. This is completely optional — would you like to provide it, or skip?", True, True),

    # --- Disability ---
    ("disability.disability_status", "Do you have a disability? Some scholarships are specifically for students with disabilities, so this can help match you. You can skip if you prefer.", True, True),
    ("disability.disability_types", "What type of disability or disabilities do you have? For example, physical, learning, visual, hearing, or other.", True, True),
    ("disability.specific_conditions", "Are there specific conditions you'd like to note? For example, ADHD, dyslexia, mobility impairment, etc.", True, True),
    ("disability.accommodations", "Do you currently receive or plan to request any accommodations at your school?", True, True),
    ("disability.documentation_status", "Do you have documentation of your disability on file with your school?", True, True),

    # --- Education Current ---
    ("education_current.university_name", "What university or college are you attending or planning to attend?", False, False),
    ("education_current.campus", "Which campus, if the school has multiple?", True, False),
    ("education_current.degree_type", "What degree are you pursuing? For example, bachelor's, master's, associate's.", False, False),
    ("education_current.majors", "What is your major or majors?", False, False),
    ("education_current.minors", "Do you have any minors?", True, False),
    ("education_current.concentration", "Do you have a concentration or specialization within your major?", True, False),
    ("education_current.expected_enrollment_date", "When do you expect to start, or when did you start?", False, False),
    ("education_current.expected_graduation_date", "When do you expect to graduate?", False, False),
    ("education_current.enrollment_status", "Will you be full-time or part-time?", False, False),
    ("education_current.current_gpa", "What is your current GPA, if you have one yet?", True, False),
    ("education_current.student_id", "Do you know your student ID number?", True, False),

    # --- Education History ---
    ("education_history.high_school.name", "What high school did you attend?", False, False),
    ("education_history.high_school.graduation_year", "What year did you graduate high school?", False, False),
    ("education_history.high_school.gpa", "What was your high school GPA?", True, False),
    ("education_history.high_school.class_rank", "Do you know your class rank?", True, False),

    # --- Professional ---
    ("professional.years_of_experience", "How many years of work experience do you have?", True, False),
    ("professional.current_employer", "Who is your current or most recent employer?", True, False),
    ("professional.current_title", "What is or was your job title?", True, False),
    ("professional.skills", "What are your key skills? List as many as you'd like.", True, False),

    # --- Financial ---
    ("financial.fafsa_filed", "Have you filed the FAFSA?", False, False),
    ("financial.efc", "Do you know your Expected Family Contribution or Student Aid Index from FAFSA?", True, True),
    ("financial.household_income_range", "What is your approximate household income range? This helps match income-based scholarships. You can give a range like 30 to 50 thousand, or skip.", True, True),
    ("financial.dependents", "How many dependents do you have, if any?", True, False),
    ("financial.employment_status_during_school", "Will you be working while in school? Full-time, part-time, or not at all?", False, False),

    # --- Extracurricular ---
    ("extracurricular.volunteer_work", "Do you have any volunteer or community service experience you'd like to note?", True, False),
    ("extracurricular.leadership_roles", "Have you held any leadership roles in organizations, clubs, or at work?", True, False),
    ("extracurricular.awards_honors", "Do you have any awards, honors, or recognitions?", True, False),
    ("extracurricular.organizational_memberships", "Are you a member of any organizations or associations?", True, False),
]

SKIP_PHRASES = {"skip", "pass", "next", "skip it", "pass on that", "i'd rather not"}
# For optional "do you have X?" questions, "no"/"nope" means "I don't have one" = skip
SKIP_PHRASES_OPTIONAL = SKIP_PHRASES | {"no", "nope", "i don't", "i do not", "none", "nothing"}


def _is_field_filled(profile: dict, dot_key: str) -> bool:
    """Return True if the field already has a meaningful value (skip on resume)."""
    val = get_field(profile, dot_key)
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0
    if isinstance(val, str):
        return len(val.strip()) > 0
    return True  # other types (int, bool, etc.) count as filled


def _count_filled_fields(profile: dict) -> int:
    """Return how many interview fields are already filled."""
    return sum(1 for (dk, _, _, _) in INTERVIEW_QUESTIONS if _is_field_filled(profile, dk))


def _is_skip(response: str, is_optional: bool = False) -> bool:
    normalized = response.lower().strip().rstrip(".")
    if is_optional and normalized in SKIP_PHRASES_OPTIONAL:
        return True
    return normalized in SKIP_PHRASES


def _is_list_field(dot_key: str) -> bool:
    """Check if this field expects a list value (majors, skills, etc.)."""
    list_fields = {
        "education_current.majors",
        "education_current.minors",
        "disability.disability_types",
        "disability.specific_conditions",
        "professional.skills",
        "professional.programming_languages",
        "professional.technologies",
        "professional.professional_memberships",
        "extracurricular.volunteer_work",
        "extracurricular.leadership_roles",
        "extracurricular.awards_honors",
        "extracurricular.organizational_memberships",
        "financial.current_aid",
    }
    return dot_key in list_fields


def _parse_list_response(response: str) -> list:
    """Split a spoken list into items. Handles 'and', commas, etc."""
    # Replace " and " with comma for splitting
    text = response.replace(" and ", ", ")
    items = [item.strip().strip(".") for item in text.split(",") if item.strip()]
    return items


def _match_to_expected_options(value: str, options: list[str], threshold: int = 60) -> str | None:
    """
    Try to match a transcribed value to one of the expected options.
    Returns the matched option string or None if no good match.
    Handles phonetically-misspelled variants like "hemridge" -> "he/him" via fuzzy match.
    """
    value_lower = value.lower().strip()
    if not value_lower:
        return None
    for opt in options:
        opt_lower = opt.lower()
        # Exact or contained
        if value_lower == opt_lower or opt_lower in value_lower:
            return opt
        # Normalize: "he him", "he/him", "he-him" -> compare stems
        value_norm = value_lower.replace("/", " ").replace("-", " ").replace(".", " ")
        opt_norm = opt_lower.replace("/", " ").replace("-", " ")
        if value_norm.split() == opt_norm.split():
            return opt
    # Fuzzy match for phonetic variants (e.g. "hemridge" -> "he/him")
    try:
        from thefuzz import fuzz

        best_score = 0
        best_opt = None
        for opt in options:
            opt_lower = opt.lower()
            score = max(
                fuzz.ratio(value_lower, opt_lower),
                fuzz.partial_ratio(value_lower, opt_lower),
                fuzz.token_set_ratio(value_lower, opt_lower),
            )
            if score > best_score and score >= threshold:
                best_score = score
                best_opt = opt
        return best_opt
    except ImportError:
        return None


def run_interview(
    on_field_complete: Optional[Callable[[str, object], None]] = None,
) -> dict:
    """
    Run the full init interview. Speaks questions, listens for answers,
    confirms, and saves profile incrementally.

    Args:
        on_field_complete: Optional callback(dot_key, value) after each field is saved.

    Returns:
        The completed profile dict.
    """
    profile = load_profile()
    filled_count = _count_filled_fields(profile)
    total_count = len(INTERVIEW_QUESTIONS)

    if filled_count >= total_count:
        speak(
            "Your profile is already complete. All questions have been answered. "
            "If you'd like to update anything, use the profile import page or "
            "re-run the interview and answer the questions you want to change.",
            cache=True,
        )
        return profile

    if filled_count > 0:
        speak(
            f"Welcome back. I'll pick up where we left off. "
            f"You've answered {filled_count} of {total_count} questions so far. "
            "You can say skip or pass on any question you'd rather not answer.",
            cache=True,
        )
    else:
        speak(
            "Hi! I'm your scholarship assistant. I'm going to ask you a series of "
            "questions to build your profile. This will save you a ton of time when "
            "filling out scholarship applications later. You can say 'skip' or 'pass' "
            "on any question you'd rather not answer right now. Let's get started.",
            cache=True,
        )

    # Keys that should not be cleaned up (factual data)
    NO_CLEANUP_KEYS = {
        "personal.email",
        "personal.phone",
        "personal.ssn_last4",
        "education_current.current_gpa",
        "education_current.student_id",
        "personal.date_of_birth",
        "education_current.expected_enrollment_date",
        "education_current.expected_graduation_date",
        "education_history.high_school.graduation_year",
        "education_history.high_school.gpa",
        "financial.efc",
        "financial.household_income_range",
    }

    for dot_key, question, is_optional, is_sensitive in INTERVIEW_QUESTIONS:
        if _is_field_filled(profile, dot_key):
            logger.info(f"Interview: skipping {dot_key} (already answered)")
            continue
        logger.info(f"Interview: asking {dot_key}")

        cleanup = dot_key not in NO_CLEANUP_KEYS

        if is_sensitive and is_optional:
            # Give extra context for sensitive optional fields
            response = ask_and_listen(question, cleanup=cleanup)
        elif is_optional:
            response = ask_and_listen(question + " You can say skip if you'd rather not answer.", cleanup=cleanup)
        else:
            response = ask_and_listen(question, cleanup=cleanup)

        # Treat skip phrases as skip. For ambiguous responses (no, nope, etc.) on
        # optional questions: when SANITY_CHECK_ENABLED, Deepseek decides; else
        # we treat no/nope as skip via _is_skip.
        if not response:
            logger.info(f"Interview: skipped {dot_key} (empty)")
            continue
        ambiguous = response.lower().strip() in ("no", "nope", "none", "nothing")
        if ambiguous and is_optional and SANITY_CHECK_ENABLED:
            interp = interpret_response(question, response, is_optional)
            if interp == "skip":
                logger.info(f"Interview: skipped {dot_key} (sanity check: skip)")
                continue
            # "use" or "reask": fall through; confirm() gives user a chance to correct
        elif _is_skip(response, is_optional=is_optional):
            logger.info(f"Interview: skipped {dot_key}")
            continue

        # Parse list fields
        if _is_list_field(dot_key):
            value = _parse_list_response(response)
            display_value = ", ".join(value)
        else:
            value = response
            display_value = response

        # For fields with expected options (e.g. pronouns): validate transcription
        if dot_key in FIELD_EXPECTED_OPTIONS:
            matched = _match_to_expected_options(
                value, FIELD_EXPECTED_OPTIONS[dot_key]
            )
            if matched:
                value = matched
                display_value = matched
            else:
                # Transcription doesn't match any known option; ask for clarification
                options_str = ", ".join(FIELD_EXPECTED_OPTIONS[dot_key][:5])
                clarification = ask_and_listen(
                    f"I heard '{value}' but I'm not sure which you meant. "
                    f"Did you mean one of these: {options_str}? Or say yours and I'll remember it.",
                    cleanup=cleanup,
                )
                if clarification and not _is_skip(clarification, is_optional=True):
                    matched = _match_to_expected_options(
                        clarification, FIELD_EXPECTED_OPTIONS[dot_key]
                    )
                    value = matched if matched else clarification
                    display_value = value
                else:
                    logger.info(f"Interview: skipped {dot_key} (clarification declined)")
                    continue

        # Confirm — always read back and get explicit confirmation. Loop until yes/no/skip.
        # Critical fields (phone, SSN) get extra clear readback.
        field_label = dot_key.split(".")[-1].replace("_", " ")
        if dot_key == "personal.phone":
            result, correction = confirm_with_explicit_readback(
                field_label, display_value, readback_style="phone"
            )
        elif dot_key == "personal.ssn_last4":
            result, correction = confirm_with_explicit_readback(
                field_label, display_value, readback_style="digits"
            )
        else:
            result, correction = confirm(
                f"For {field_label}:", display_value
            )

        if result == CONFIRM_SKIP:
            logger.info(f"Interview: skipped {dot_key} (user skipped during confirm)")
            speak_interview_filler()
            continue

        if result == CONFIRM_CORRECTION and correction:
            if _is_skip(correction, is_optional=True):
                logger.info(f"Interview: skipped {dot_key} (skip during correction)")
                speak_interview_filler()
                continue
            if _is_list_field(dot_key):
                value = _parse_list_response(correction)
            else:
                value = correction

        # Save incrementally (never store literal "skip")
        if isinstance(value, str) and value.strip().lower() == "skip":
            logger.info(f"Interview: skipped {dot_key} (would have stored literal skip)")
            speak_interview_filler()
            continue
        set_field(profile, dot_key, value)
        save_profile(profile)
        logger.info(f"Interview: saved {dot_key} = {value}")

        # Play a short filler so the user knows we're moving on
        speak_interview_filler()

        if on_field_complete:
            on_field_complete(dot_key, value)

    # Summary
    speak(
        "That's everything! Your profile is saved and ready to go. "
        "When you click the scholarship assistant button on an application page, "
        "I'll use this information to fill in forms for you automatically. "
        "If I ever encounter a question I don't have an answer for, I'll ask you "
        "and remember your answer for next time.",
        cache=True,
    )

    return profile
