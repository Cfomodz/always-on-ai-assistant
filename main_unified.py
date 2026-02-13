from RealtimeSTT import AudioToTextRecorder
from typing import List
from modules.assistant_config import get_config
from modules.base_assistant import PlainAssistant
from modules.typer_agent import TyperAgent
from modules.router import classify
from modules.utils import create_session_logger_id, setup_logging
import typer
import os

app = typer.Typer()


@app.command()
def ping():
    print("pong")


@app.command()
def awaken(
    typer_file: str = typer.Option(
        "commands/template.py",
        "--typer-file",
        "-f",
        help="Path to typer commands file",
    ),
    scratchpad: str = typer.Option(
        "scratchpad.md", "--scratchpad", "-s", help="Path to scratchpad file"
    ),
    context_files: List[str] = typer.Option(
        [], "--context", "-c", help="List of context files"
    ),
    mode: str = typer.Option(
        "execute",
        "--mode",
        "-m",
        help="Command execution mode: default (no exec), execute (exec + scratch), execute-no-scratch",
    ),
):
    """Unified voice assistant — routes speech to commands or conversation."""
    config_key = "unified_assistant"

    # Session setup
    session_id = create_session_logger_id()
    logger = setup_logging(session_id)
    logger.info(f"Starting unified session {session_id}")

    assistant_name = get_config(f"{config_key}.assistant_name")
    logger.info(f"Assistant name: {assistant_name}")

    # -- Build both agents --
    # TyperAgent for command execution
    typer_agent, typer_file_resolved, scratchpad_resolved = TyperAgent.build_agent(
        typer_file, [scratchpad]
    )
    typer_agent.logger = logger

    # PlainAssistant for conversation
    conversation_agent = PlainAssistant(logger, session_id)

    # -- Load typer commands source for the router --
    typer_commands_source = ""
    for tf in [typer_file]:
        if os.path.exists(tf):
            with open(tf, "r") as f:
                typer_commands_source += f.read() + "\n"

    # -- STT setup --
    whisper_model = get_config(f"{config_key}.whisper_model") or "tiny.en"
    logger.info(f"Whisper model: {whisper_model}")

    recorder = AudioToTextRecorder(
        spinner=False,
        post_speech_silence_duration=1.5,
        compute_type="float32",
        model=whisper_model,
        beam_size=8,
        batch_size=25,
        language="en",
        print_transcription_time=True,
    )

    def process_text(text):
        print(f"\n  Heard: {text}")

        # -- Trigger word check --
        if assistant_name.lower() not in text.lower():
            logger.info(f"Not {assistant_name} — ignoring")
            return

        # -- Self-hearing check (from PlainAssistant) --
        if (
            conversation_agent.conversation_history
            and text.strip().lower()
            in conversation_agent.conversation_history[-1]["content"].lower()
        ):
            logger.info("Ignoring own speech input")
            return

        recorder.stop()

        try:
            # -- Route --
            intent = classify(text, typer_commands_source)
            logger.info(f"Intent: {intent}")

            if intent == "command":
                output = typer_agent.process_text(
                    text,
                    typer_file_resolved,
                    scratchpad_resolved,
                    context_files,
                    mode,
                )
                print(f"  Command output:\n{output}")
            else:
                response = conversation_agent.process_text(text)
                logger.info(f"Conversation response: {response}")

        except Exception as e:
            logger.error(f"Error: {e}")

        recorder.start()

    # -- Main loop --
    try:
        print(f"  {assistant_name} is listening... (Ctrl+C to exit)")
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        logger.info("Session ended by user")


if __name__ == "__main__":
    app()
