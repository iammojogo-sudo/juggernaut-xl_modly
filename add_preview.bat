@echo off
setlocal
set "HERE=%~dp0"
set "PY=%HERE%venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Could not find the extension Python at:
  echo   %PY%
  echo Reinstall the extension (its setup creates this).
  pause
  exit /b 1
)
"%PY%" "%HERE%add_preview.py"
pause
