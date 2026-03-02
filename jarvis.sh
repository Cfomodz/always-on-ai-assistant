#!/bin/bash
export LD_PRELOAD=""
export CT2_VERBOSE=1
setarch $(uname -m) -R uv run python main_unified.py awaken \
  --typer-file commands/template.py \
  --scratchpad scratchpad.md \
  --mode execute
