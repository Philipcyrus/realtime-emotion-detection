#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"

"$PYTHON" -m pip check
"$PYTHON" -m unittest discover -s tests
"$PYTHON" -m emotion_detection.app --help
"$PYTHON" -m emotion_detection.app doctor --model assets/models/emotion_model.keras --camera-max-index 1
