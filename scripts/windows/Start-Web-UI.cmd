@echo off
rem PathSnap macOS-style web UI (Laptop 2). Usage: Start-Web-UI.cmd [http://<laptop-1-ip>:8000]
setlocal
cd /d "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt using the README quick start.
    exit /b 1
)
if not "%~1"=="" set "PATHSNAP_API=%~1"
if not defined PATHSNAP_API set "PATHSNAP_API=http://127.0.0.1:8000"
".venv\Scripts\python.exe" ui\serve.py --api "%PATHSNAP_API%"
exit /b %errorlevel%
