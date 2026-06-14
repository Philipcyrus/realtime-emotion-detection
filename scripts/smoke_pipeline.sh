#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."

PYTHON="./.venv/Scripts/python.exe"
CAMERA_INDEX="${1:-0}"

"$PYTHON" -m emotion_detection.app make-demo-data \
  --output data/smoke_emotions \
  --labels happy,neutral \
  --samples-per-label 3

"$PYTHON" -m emotion_detection.app train \
  --train-dir data/smoke_emotions/train \
  --validation-dir data/smoke_emotions/validation \
  --model-out assets/models/smoke_emotion_model.keras \
  --metadata-out assets/models/smoke_emotion_model.metadata.json \
  --labels happy,neutral \
  --epochs 1 \
  --batch-size 2

"$PYTHON" -m emotion_detection.app evaluate \
  --model assets/models/smoke_emotion_model.keras \
  --data-dir data/smoke_emotions/validation \
  --metadata assets/models/smoke_emotion_model.metadata.json \
  --output reports/smoke_evaluation.json \
  --batch-size 2

"$PYTHON" -m emotion_detection.app \
  --camera-index "$CAMERA_INDEX" \
  --model assets/models/smoke_emotion_model.keras \
  --labels happy,neutral \
  --headless \
  --max-frames 3 \
  --report-dir reports \
  --session-name smoke-webcam
