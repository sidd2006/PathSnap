@echo off
setlocal
cd /d "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt using the README quick start.
    exit /b 1
)
if not defined PATHSNAP_API set "PATHSNAP_API=http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m streamlit run ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
exit /b %errorlevel%
