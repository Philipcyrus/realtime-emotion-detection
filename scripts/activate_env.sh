#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR/.."
PROJECT_ROOT="$PWD"
VENV_DIR="$PROJECT_ROOT/.venv"

if [[ ! -x "$VENV_DIR/Scripts/python.exe" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  echo "Create it with: python -m venv .venv"
  return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export _OLD_EMOTION_DETECTION_PATH="$PATH"
export PATH="$VENV_DIR/Scripts:$PATH"
export PYTHONNOUSERSITE=1

echo "Activated Emotion Detection venv"
echo "Python: $("$VENV_DIR/Scripts/python.exe" --version)"
echo "Use: deactivate_env"

deactivate_env() {
  export PATH="$_OLD_EMOTION_DETECTION_PATH"
  unset VIRTUAL_ENV
  unset PYTHONNOUSERSITE
  unset _OLD_EMOTION_DETECTION_PATH
  unset -f deactivate_env
  echo "Deactivated Emotion Detection venv"
}
