#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"
MAX_PER_CLASS="${1:-1000}"

"$PYTHON" -m emotion_detection.app download-fer2013 \
  --output data/fer2013_images \
  --max-per-class "$MAX_PER_CLASS"
