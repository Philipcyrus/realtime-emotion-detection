#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"

"$PYTHON" -m emotion_detection.app evaluate \
  --model assets/models/emotion_model.keras \
  --data-dir data/fer2013_images/validation \
  --metadata assets/models/emotion_model.metadata.json \
  --output reports/evaluation.json
