@echo off
setlocal
cd /d "%~dp0\.."
set MAX_PER_CLASS=%~1
set EPOCHS=%~2
set BATCH_SIZE=%~3
if "%MAX_PER_CLASS%"=="" set MAX_PER_CLASS=4000
if "%EPOCHS%"=="" set EPOCHS=60
if "%BATCH_SIZE%"=="" set BATCH_SIZE=64
call scripts\download_data.cmd %MAX_PER_CLASS%
call scripts\train_model.cmd %EPOCHS% %BATCH_SIZE% 10
call scripts\evaluate_model.cmd
