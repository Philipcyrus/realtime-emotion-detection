#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"
EPOCHS="${1:-30}"
BATCH_SIZE="${2:-64}"
PATIENCE="${3:-10}"

"$PYTHON" -m emotion_detection.app train \
  --train-dir data/fer2013_images/train \
  --validation-dir data/fer2013_images/validation \
  --model-out assets/models/emotion_model.keras \
  --metadata-out assets/models/emotion_model.metadata.json \
  --labels angry,disgust,fear,happy,sad,surprise,neutral \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate 0.0003 \
  --patience "$PATIENCE"
