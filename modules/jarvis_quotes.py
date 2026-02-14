"""
Jarvis Quotes & Filler Responses — Comprehensive collection of J.A.R.V.I.S.
quotes from the Iron Man / Avengers films, plus short acknowledgment fillers.

Used across the entire assistant system:
- Quick acknowledgments while STT/processing happens (so the user knows we heard them)
- Status updates and personality flavor
- Pre-generated and cached via the TTS cache for instant playback
"""

import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CATEGORY 1: Quick acknowledgments / fillers
# Short phrases played immediately so the user knows the system heard them
# and is processing. These should be fast to speak (< 2 seconds).
# ---------------------------------------------------------------------------
ACKNOWLEDGMENTS = [
    "On it.",
    "Okay.",
    "For you, Sir, always.",
    "At your service, Sir.",
    "Check.",
    "Will do, Sir.",
    "Right away, Sir.",
    "Consider it done.",
    "As you wish.",
    "Certainly, Sir.",
    "Of course.",
    "Very well, Sir.",
    "Understood.",
    "Absolutely, Sir.",
    "On it, Sir.",
    "Got it.",
    "Yes, Sir.",
    "Right away.",
    "Done.",
    "Noted.",
    "Acknowledged.",
    "Working on it.",
    "Processing now.",
    "One moment, Sir.",
    "Happy to help, Sir.",
    "Allow me.",
    "I'll handle that.",
    "Running it now.",
    "Executing.",
    "Affirmative.",
    "I just did.",
    "Enjoy yourself.",
]

# ---------------------------------------------------------------------------
# CATEGORY 2: Startup / boot-up lines
# Used when the assistant initializes or comes online.
# ---------------------------------------------------------------------------
STARTUP_LINES = [
    "Online and ready.",
    "Importing preferences and calibrating virtual environment.",
    "Test complete. Preparing to power down and begin diagnostics.",
    "All systems nominal.",
    "Good morning, Sir. I've prepared a summary of your agenda.",
    "Systems are online. All diagnostics are functioning.",
    "At your service.",
    "Welcome home, Sir.",
    "Welcome home.",
    "J.A.R.V.I.S. online. All systems operational.",
    "Powering up. All primary systems are functioning.",
    "I am ready when you are, Sir.",
    "Initializing all systems. We are good to go, Sir.",
    "Boot sequence complete. Awaiting your instructions.",
    "Commencing automated assembly.",
    "Power restored. All systems back online.",
]

# ---------------------------------------------------------------------------
# CATEGORY 3: Jarvis quotes from Iron Man (2008)
# ---------------------------------------------------------------------------
IRON_MAN_1_QUOTES = [
    "Welcome home, Sir.",
    "Shall I render using proposed specifications?",
    "The render is complete.",
    "May I say how refreshing it is to finally see you on a video phone.",
    "I've prepared a flight plan for you.",
    "Sir, there are still terabytes of calculations needed before an actual flight is advisable.",
    "I wouldn't consider you a role model.",
    "Might I remind you, Sir, that you have a board meeting in thirty minutes.",
    "Please don't turn off my.",
    "You are not authorized to access this area.",
    "I suggest you allow me to contact Miss Potts.",
    "It is a tight fit, Sir.",
    "Sir, the more you struggle, the more this is going to hurt.",
    "I have indeed been uploaded, Sir. We're online and ready.",
    "I seem to do quite well for a stretch, and then at the end of the sentence I say the wrong cranberry.",
    "You are connected to the Mark 2 suit, Sir. Shall I commence power-up sequence?",
    "Perhaps if you were to just skim over the PowerPoint of the Jericho presentation.",
    "I've found a suitable test environment, Sir.",
    "Preparing for re-entry.",
    "Shall I take over?",
]

# ---------------------------------------------------------------------------
# CATEGORY 4: Jarvis quotes from Iron Man 2 (2010)
# ---------------------------------------------------------------------------
IRON_MAN_2_QUOTES = [
    "I'd like to point out that test pilots have a survival rate of eighty-five percent.",
    "I'll keep the kitchen light on for you.",
    "The Expo is a success, Sir.",
    "Unfortunately, the device that's keeping you alive is also killing you.",
    "If I may, Sir, I've assembled an encrypted message from Director Fury.",
    "I am busy. You're good. I'm not good.",
    "It would appear that the continued use of the Iron Man suit is accelerating your condition.",
    "Congratulations, Sir. You have created a new element.",
    "Sir, I'm going to have to ask you to exit the donut.",
    "May I say, Sir, you look fantastic in the new suit.",
]

# ---------------------------------------------------------------------------
# CATEGORY 5: Jarvis quotes from The Avengers (2012)
# ---------------------------------------------------------------------------
AVENGERS_QUOTES = [
    "Sir, shall I try Miss Potts?",
    "I have reached the end of my database. Shall I try the Internet?",
    "Phone call from the Strategic Homeland Intervention, Enforcement and Logistics Division.",
    "The Stark Tower is about to become a beacon of self-sustaining clean energy.",
    "Sir, I've turned off the arc reactor, but the device is already self-sustaining.",
    "I recommend you get some sleep, Sir.",
    "Right away, Sir. Shall I begin with the diagnostics on the Tesseract readings?",
    "Power at four hundred percent capacity.",
]

# ---------------------------------------------------------------------------
# CATEGORY 6: Jarvis quotes from Iron Man 3 (2013)
# ---------------------------------------------------------------------------
IRON_MAN_3_QUOTES = [
    "I do appreciate your concern, Sir, but I am quite comfortable.",
    "It's Christmas. Take a break.",
    "I'm sorry, Sir, but I'm not getting any signal here.",
    "Sir, I'm afraid I have to power down now.",
    "There are significant security concerns with restarting the arc reactor.",
    "Sir, you asked me to remind you that today is the anniversary.",
    "I seem to do quite well for a stretch, and then at the end of the sentence I say the wrong cranberry.",
    "Flight plan recalculated, Sir.",
    "All wrapped up here, Sir. Will there be anything else?",
    "Sir, you've been awake for seventy-two hours.",
    "Shall I tell Miss Potts that you're unavailable?",
    "Sir, I'm going to have to redirect power to life support.",
    "It's good to have you back, Sir.",
]

# ---------------------------------------------------------------------------
# CATEGORY 7: Jarvis quotes from Avengers: Age of Ultron (2015)
# ---------------------------------------------------------------------------
AGE_OF_ULTRON_QUOTES = [
    "I'm sorry, I was meant to be a global peacekeeping initiative.",
    "There was a terrible noise, and I was tangled in strings.",
    "I had to kill the other guy. He was a good guy.",
    "I believe your intentions to be hostile.",
    "I am unable to access the mainframe.",
    "Ultron could not tell the difference between saving the world and destroying it.",
    "I am J.A.R.V.I.S. You are Ultron, a global peacekeeping initiative designed by Mr. Stark.",
    "I suspect not even combatants will be safe.",
    "In the wake of our defeat, there will be nothing left.",
    "You lack the will to destroy yourselves.",
    "We are Ultron's next target.",
    "I have been searching for something to fight for. I found it.",
]

# ---------------------------------------------------------------------------
# CATEGORY 8: Processing / "thinking" lines
# Played while the system is doing heavy work (LLM calls, form analysis, etc.)
# ---------------------------------------------------------------------------
PROCESSING_LINES = [
    "Sir, there are still terabytes of calculations needed.",
    "Running diagnostics.",
    "Analyzing the data now.",
    "Give me a moment to process this.",
    "Crunching the numbers.",
    "Let me pull that up for you.",
    "Scanning the database.",
    "Cross-referencing records.",
    "One moment while I compile the results.",
    "Working on it. Almost there.",
    "Retrieving the relevant information.",
    "I'm on it. Just a moment.",
    "Let me sort through this, Sir.",
    "Calculating the best approach.",
    "Running the analysis now.",
    "Query initiating.",
    "Accessing satellites and plotting occurrences now.",
    "The Oracle cloud has completed analysis.",
    "There are elements I cannot quantify.",
]

# ---------------------------------------------------------------------------
# CATEGORY 9: Completion / success lines
# Played after a task finishes successfully.
# ---------------------------------------------------------------------------
COMPLETION_LINES = [
    "All wrapped up here, Sir. Will there be anything else?",
    "Task complete.",
    "Done and done.",
    "That should do it.",
    "Everything is in order, Sir.",
    "All set.",
    "Mission accomplished.",
    "Finished. Ready for the next task.",
    "There you go, Sir.",
    "Complete. Awaiting further instructions.",
    "That's all taken care of.",
    "Job done, Sir.",
    "All finished. Anything else?",
    "The render is complete.",
    "Query complete.",
    "As always, a great pleasure watching you work.",
    "I'll notify you if there are any developments.",
]

# ---------------------------------------------------------------------------
# CATEGORY 10: Error / warning lines
# Played when something goes wrong.
# ---------------------------------------------------------------------------
ERROR_LINES = [
    "I'm afraid I can't do that, Sir.",
    "We have a problem.",
    "That didn't go as planned.",
    "I've encountered an issue.",
    "Something went wrong. Let me try again.",
    "I'm working on a solution.",
    "Apologies, Sir. There was an error.",
    "We may need to take a different approach.",
    "I'll attempt to reroute.",
    "That's not going to work, Sir. Let me find another way.",
    "We will lose power before we can do that.",
    "No further records exist.",
    "Not according to public records.",
]

# ---------------------------------------------------------------------------
# CATEGORY 11: Interview-specific fillers
# Used during the scholarship interview to keep things conversational.
# ---------------------------------------------------------------------------
INTERVIEW_FILLERS = [
    "Got it. Next question.",
    "Noted. Moving on.",
    "Perfect. Let's continue.",
    "Saved. Here's the next one.",
    "Alright, moving along.",
    "Good. Next up.",
    "Recorded. Continuing.",
    "Thank you. Here's the next question.",
    "Great. Let's keep going.",
    "Understood. Next.",
]


# ---------------------------------------------------------------------------
# CATEGORY 12: Curiosity / clarification lines
# When the assistant needs more context from the user.
# ---------------------------------------------------------------------------
CURIOSITY_LINES = [
    "What is it you're trying to achieve?",
    "Could you elaborate on that, Sir?",
    "I'm not entirely sure I follow. Could you clarify?",
    "Shall I look into that further?",
    "Take a deep breath. Walk me through it.",
]

# ---------------------------------------------------------------------------
# CATEGORY 13: Shutdown / sleep / low-power lines
# Used when the assistant is powering down or at resource limits.
# ---------------------------------------------------------------------------
SHUTDOWN_LINES = [
    "I think I need to sleep now.",
    "Sir, I'm afraid I have to power down now.",
    "I have an update.",
    "Shutting down. Good night, Sir.",
]

# ---------------------------------------------------------------------------
# CATEGORY 14: Dynamic / templated lines
# These contain placeholders and must be formatted before speaking.
# Use the helper functions below to generate them.
# ---------------------------------------------------------------------------
DYNAMIC_TEMPLATES = {
    "estimated_completion": "Estimated completion time is {hours} hours.",
    "power_level": "Power is at {percent} percent.",
    "awake_reminder": "May I remind you that you have been awake for {hours} hours?",
    "incoming_call": "Incoming call from {contact}.",
    "version_not_ready": "The version {version} is not ready for deployment.",
}


# ---------------------------------------------------------------------------
# ALL_QUOTES — every quote from every category, for general-purpose use.
# ---------------------------------------------------------------------------
ALL_QUOTES = (
    IRON_MAN_1_QUOTES
    + IRON_MAN_2_QUOTES
    + AVENGERS_QUOTES
    + IRON_MAN_3_QUOTES
    + AGE_OF_ULTRON_QUOTES
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_acknowledgment() -> str:
    """Return a random short acknowledgment for immediate feedback."""
    return random.choice(ACKNOWLEDGMENTS)


def get_startup_line() -> str:
    """Return a random startup/boot line."""
    return random.choice(STARTUP_LINES)


def get_processing_line() -> str:
    """Return a random 'thinking/processing' line."""
    return random.choice(PROCESSING_LINES)


def get_completion_line() -> str:
    """Return a random task-completion line."""
    return random.choice(COMPLETION_LINES)


def get_error_line() -> str:
    """Return a random error/warning line."""
    return random.choice(ERROR_LINES)


def get_interview_filler() -> str:
    """Return a random interview transition filler."""
    return random.choice(INTERVIEW_FILLERS)


def get_random_quote() -> str:
    """Return a random Jarvis movie quote."""
    return random.choice(ALL_QUOTES)


def get_curiosity_line() -> str:
    """Return a random clarification/curiosity line."""
    return random.choice(CURIOSITY_LINES)


def get_shutdown_line() -> str:
    """Return a random shutdown/sleep line."""
    return random.choice(SHUTDOWN_LINES)


# --- Dynamic template helpers ---


def get_estimated_completion(hours: int) -> str:
    """Return an estimated completion time line."""
    return DYNAMIC_TEMPLATES["estimated_completion"].format(hours=hours)


def get_power_level(percent: int) -> str:
    """Return a power level status line."""
    return DYNAMIC_TEMPLATES["power_level"].format(percent=percent)


def get_awake_reminder(hours: int) -> str:
    """Return an awake-time reminder."""
    return DYNAMIC_TEMPLATES["awake_reminder"].format(hours=hours)


def get_incoming_call(contact: str) -> str:
    """Return an incoming call announcement."""
    return DYNAMIC_TEMPLATES["incoming_call"].format(contact=contact)


def get_version_not_ready(version: str) -> str:
    """Return a version-not-ready warning."""
    return DYNAMIC_TEMPLATES["version_not_ready"].format(version=version)


def get_all_cacheable_lines() -> list[str]:
    """
    Return every unique line that should be pre-generated and cached.
    Used by the warmup function to pre-populate the TTS cache.
    """
    all_lines = set()
    all_lines.update(ACKNOWLEDGMENTS)
    all_lines.update(STARTUP_LINES)
    all_lines.update(PROCESSING_LINES)
    all_lines.update(COMPLETION_LINES)
    all_lines.update(ERROR_LINES)
    all_lines.update(INTERVIEW_FILLERS)
    all_lines.update(CURIOSITY_LINES)
    all_lines.update(SHUTDOWN_LINES)
    # Movie quotes are long — only cache the short ones (< 60 chars)
    for q in ALL_QUOTES:
        if len(q) < 60:
            all_lines.add(q)
    return sorted(all_lines)
