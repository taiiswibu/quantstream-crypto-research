@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install -r launchers\requirements-local.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --no-deps -e .
if errorlevel 1 goto fail
if not exist "workspace\artifacts\local_registry.json" (
    echo Downloading 90 days of public candles and fitting CPU baselines...
    .venv\Scripts\python.exe -m quantstream.terminal --root workspace setup --days 90
    if errorlevel 1 goto fail
)
echo.
echo Ready. Open START_WINDOWS.cmd
exit /b 0
:fail
echo.
echo Setup failed. Check the error above. Python 3.12 64-bit and Internet are required.
echo Run INSTALL_WINDOWS.cmd again after correcting the error.
pause
exit /b 1
