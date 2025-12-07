from alternative_stt import AudioToTextRecorder
from modules.assistant_config import get_config
from modules.typer_agent import TyperAgent
from modules.utils import create_session_logger_id, setup_logging
import logging
import typer
from typing import List
import os

app = typer.Typer()


@app.command()
def ping():
    print("pong")


@app.command()
def awaken(
    typer_file: str = typer.Option(
        ..., "--typer-file", "-f", help="Path to typer commands file"
    ),
    scratchpad: str = typer.Option(
        ..., "--scratchpad", "-s", help="Path to scratchpad file"
    ),
    context_files: List[str] = typer.Option(
        [], "--context", "-c", help="List of context files"
    ),
    mode: str = typer.Option(
        "default",
        "--mode",
        "-m",
        help="Options: ('default', 'execute', 'execute-no-scratch'). Execution mode: default (no exec), execute (exec + scratch), execute-no-scratch (exec only)",
    ),
):
    """Run STT interface that processes speech into typer commands"""
    # Remove the list concatenation - pass scratchpad as a single string
    assistant, typer_file, _ = TyperAgent.build_agent(typer_file, [scratchpad])

    print("🎤 Speak now... (press Ctrl+C to exit)")

    recorder = AudioToTextRecorder(
        model_name="tiny.en",  # VERY fast (.5s), but not accurate
        # model_name="small.en",  # decent speed (1.5s), improved accuracy
        # model_name="base.en",  # balanced speed and accuracy
        # model_name="large",  # very slow, but accurate
        silence_threshold=0.01,  # Adjust based on your microphone sensitivity
        silence_duration=1.5,  # how long to wait after speech ends before processing
        print_transcription_time=True,
    )

    def process_text(text):
        print(f"\n🎤 Heard: {text}")
        try:
            assistant_name = get_config("typer_assistant.assistant_name")
            if assistant_name.lower() not in text.lower():
                print(f"🤖 Not {assistant_name} - ignoring")
                return

            recorder.stop()
            output = assistant.process_text(
                text, typer_file, scratchpad, context_files, mode
            )
            print(f"🤖 Response:\n{output}")
            recorder.start()
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    while True:
        recorder.text(process_text)


if __name__ == "__main__":
    app()
