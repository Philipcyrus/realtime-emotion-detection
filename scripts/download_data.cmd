@echo off
setlocal
cd /d "%~dp0\.."
set MAX_PER_CLASS=%~1
if "%MAX_PER_CLASS%"=="" set MAX_PER_CLASS=1000
".venv\Scripts\python.exe" -m emotion_detection.app download-fer2013 --output data/fer2013_images --max-per-class %MAX_PER_CLASS%
