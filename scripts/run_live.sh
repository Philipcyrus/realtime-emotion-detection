#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"
CAMERA_INDEX="${1:-0}"
MODEL_PATH="${2:-assets/models/emotion_model.keras}"

"$PYTHON" -m emotion_detection.app run \
  --camera-index "$CAMERA_INDEX" \
  --model "$MODEL_PATH" \
  --report-dir reports \
  --session-name live-demo
