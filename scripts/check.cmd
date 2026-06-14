@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m pip check
".venv\Scripts\python.exe" -m unittest discover -s tests
".venv\Scripts\python.exe" -m emotion_detection.app --help
".venv\Scripts\python.exe" -m emotion_detection.app doctor --model assets/models/emotion_model.keras --camera-max-index 1
