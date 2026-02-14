"""
Init Interview — Voice-driven walkthrough that builds the user's profile.
Triggered on first run (no profile.json) or via --init / POST /init.
"""

import logging
from typing import Callable, Optional

from server.profile_manager import load_profile, save_profile, profile_exists, set_field
from server.voice import speak, ask_and_listen, confirm, speak_interview_filler

logger = logging.getLogger("scholarship-assistant")

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

SKIP_PHRASES = {"skip", "pass", "next", "no", "nope", "skip it", "pass on that", "i'd rather not"}


def _is_skip(response: str) -> bool:
    return response.lower().strip().rstrip(".") in SKIP_PHRASES


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

    speak(
        "Hi! I'm your scholarship assistant. I'm going to ask you a series of "
        "questions to build your profile. This will save you a ton of time when "
        "filling out scholarship applications later. You can say 'skip' or 'pass' "
        "on any question you'd rather not answer right now. Let's get started.",
        cache=True,
    )

    for dot_key, question, is_optional, is_sensitive in INTERVIEW_QUESTIONS:
        logger.info(f"Interview: asking {dot_key}")

        if is_sensitive and is_optional:
            # Give extra context for sensitive optional fields
            response = ask_and_listen(question)
        elif is_optional:
            response = ask_and_listen(question + " You can say skip if you'd rather not answer.")
        else:
            response = ask_and_listen(question)

        if not response or _is_skip(response):
            logger.info(f"Interview: skipped {dot_key}")
            continue

        # Parse list fields
        if _is_list_field(dot_key):
            value = _parse_list_response(response)
            display_value = ", ".join(value)
        else:
            value = response
            display_value = response

        # Confirm
        confirmed, correction = confirm(
            f"For {dot_key.split('.')[-1].replace('_', ' ')}:",
            display_value,
        )

        if not confirmed and correction:
            if _is_list_field(dot_key):
                value = _parse_list_response(correction)
            else:
                value = correction

        # Save incrementally
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
