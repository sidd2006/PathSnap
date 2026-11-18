@echo off
setlocal
if not "%~1"=="" set "PATHSNAP_EVAL_DIR=%~f1"
cd /d "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt using the README quick start.
    exit /b 1
)
if not defined PATHSNAP_EVAL_DIR set "PATHSNAP_EVAL_DIR=%CD%\data\evaluator_test"
if not exist "%PATHSNAP_EVAL_DIR%\." (
    echo Trip directory missing: "%PATHSNAP_EVAL_DIR%"
    exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%
