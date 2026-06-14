@echo off
setlocal
cd /d "%~dp0\.."
set EPOCHS=%~1
set BATCH_SIZE=%~2
set PATIENCE=%~3
if "%EPOCHS%"=="" set EPOCHS=60
if "%BATCH_SIZE%"=="" set BATCH_SIZE=64
if "%PATIENCE%"=="" set PATIENCE=10
".venv\Scripts\python.exe" -m emotion_detection.app train --train-dir data/fer2013_images/train --validation-dir data/fer2013_images/validation --model-out assets/models/emotion_model.keras --metadata-out assets/models/emotion_model.metadata.json --labels angry,disgust,fear,happy,sad,surprise,neutral --epochs %EPOCHS% --batch-size %BATCH_SIZE% --learning-rate 0.0003 --patience %PATIENCE%
