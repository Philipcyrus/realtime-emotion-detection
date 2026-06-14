@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m emotion_detection.app evaluate --model assets/models/emotion_model.keras --data-dir data/fer2013_images/validation --metadata assets/models/emotion_model.metadata.json --output reports/evaluation.json
