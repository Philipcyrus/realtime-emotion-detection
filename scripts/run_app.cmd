@echo off
setlocal
cd /d "%~dp0\.."
set CAMERA_INDEX=%~1
if "%CAMERA_INDEX%"=="" set CAMERA_INDEX=0
".venv\Scripts\python.exe" -m emotion_detection.app run --camera-index %CAMERA_INDEX% --model assets/models/emotion_model.keras --report-dir reports --session-name live-demo
