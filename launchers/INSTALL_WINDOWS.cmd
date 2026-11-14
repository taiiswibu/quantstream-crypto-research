@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-local.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --no-deps wheels\quantstream_local-0.2.0-py3-none-any.whl
if errorlevel 1 goto fail
echo.
echo Installation complete. Open START_WINDOWS.cmd
pause
exit /b 0
:fail
echo.
echo Installation failed. Install Python 3.12 64-bit from python.org, then retry.
pause
exit /b 1
