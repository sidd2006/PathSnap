@echo off
setlocal
cd /d "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt using the README quick start.
    exit /b 1
)
".venv\Scripts\python.exe" scripts/show_results.py all
if errorlevel 1 exit /b 1
pause
