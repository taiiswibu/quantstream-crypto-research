@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call INSTALL_WINDOWS.cmd
    if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe run_local.py
if errorlevel 1 pause
