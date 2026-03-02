#!/bin/bash
uv run python main_typer_assistant_alt.py awaken \
  --typer-file commands/template.py \
  --scratchpad scratchpad.md \
  --mode execute
