#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

CAMERA_INDEX="${1:-0}"

"./.venv/Scripts/python.exe" -m emotion_detection.app run \
  --camera-index "$CAMERA_INDEX" \
  --model assets/models/emotion_model.keras \
  --report-dir reports \
  --session-name live-demo
