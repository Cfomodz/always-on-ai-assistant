import logging
from modules.deepseek import prefix_prompt

logger = logging.getLogger("main")

# The router prompt is intentionally minimal to stay fast (~200ms).
# It reads the available command names and decides if the utterance
# is asking to run one of them, or is just conversation.
ROUTER_PROMPT_TEMPLATE = """<purpose>
Classify the user's spoken utterance as either a "command" request or a "conversation" request.
</purpose>

<instructions>
    <instruction>A "command" utterance is one that asks to execute, run, list, create, delete, search, ping, backup, restore, queue, filter, compare, encrypt, decrypt, inspect, generate, download, upload, migrate, or otherwise operate on data using one of the available CLI commands below.</instruction>
    <instruction>A "conversation" utterance is anything else: greetings, questions, opinions, chitchat, or requests that don't map to a CLI command.</instruction>
    <instruction>Respond with exactly one word: command OR conversation</instruction>
</instructions>

<available-commands>
{{available_commands}}
</available-commands>

<utterance>
{{utterance}}
</utterance>
"""


def classify(utterance: str, typer_commands_source: str) -> str:
    """Classify an utterance as 'command' or 'conversation'.

    Uses a lightweight DeepSeek prefix-constrained call to force a
    single-token classification.

    Args:
        utterance: The user's spoken text (trigger word already stripped).
        typer_commands_source: The raw source of the Typer commands file(s),
            so the model can see what commands are available.

    Returns:
        "command" or "conversation"
    """
    prompt_text = (
        ROUTER_PROMPT_TEMPLATE
        .replace("{{available_commands}}", typer_commands_source)
        .replace("{{utterance}}", utterance)
    )

    try:
        result = prefix_prompt(
            prompt=prompt_text,
            prefix="Classification: ",
            no_prefix=True,
        )
        classification = result.strip().lower()

        # Normalize to one of the two valid values
        if "command" in classification:
            classification = "command"
        else:
            classification = "conversation"

        logger.info(f"Router classified '{utterance[:50]}...' as: {classification}")
        return classification

    except Exception as e:
        logger.error(f"Router classification failed: {e} — defaulting to conversation")
        return "conversation"
