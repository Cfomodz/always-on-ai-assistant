from RealtimeSTT import AudioToTextRecorder
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
        spinner=False,
        # wake_words="deep" # specific wake words to trigger the assistant using the realtime-stt library. we do this manually below so we can use any word.
        # realtime_processing_pause=0.3,
        post_speech_silence_duration=1.5,  # how long to wait after speech ends before processing
        # compute_type="int8",
        compute_type="float32",
        model="tiny.en",  # VERY fast (.5s), but not accurate
        # model="small.en",  # decent speed (1.5s), improved accuracy
        # Beam size controls how many alternative transcription paths are explored
        # Higher values = more accurate but slower, lower values = faster but less accurate
        # beam_size=3,
        # beam_size=5,
        beam_size=8,
        # Batch size controls how many audio chunks are processed together
        # Higher values = faster processing but uses more memory, lower values = slower processing but uses less memory
        batch_size=25,
        # model="large-v3",  # very slow, but accurate
        # model="distil-large-v3", # very slow (but faster than large-v3) but accurate
        # realtime_model_type="tiny.en", # realtime models are used for the on_realtime_transcription_update() callback
        # realtime_model_type="large-v3",
        language="en",
        print_transcription_time=True,
        # enable_realtime_transcription=True,
        # on_realtime_transcription_update=lambda text: print(
        #     f"🎤 on_realtime_transcription_update(): {text}"
        # ),
        # on_realtime_transcription_stabilized=lambda text: print(
        #     f"🎤 on_realtime_transcription_stabilized(): {text}"
        # ),
        # on_recorded_chunk=lambda chunk: print(f"🎤 on_recorded_chunk(): {chunk}"),
        # on_transcription_start=lambda: print("🎤 on_transcription_start()"),
        # on_recording_stop=lambda: print("🎤 on_transcription_stop()"),
        # on_recording_start=lambda: print("🎤 on_recording_start()"),
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
