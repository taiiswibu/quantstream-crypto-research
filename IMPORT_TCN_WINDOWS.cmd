@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run INSTALL_WINDOWS.cmd first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -c "import torch; assert tuple(int(v) for v in torch.__version__.split('.')[:2]) >= (2,6)" >nul 2>&1
if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto fail
)
set "checkpoint_folder=%~1"
if not defined checkpoint_folder set "checkpoint_folder=artifacts\tcn"
.venv\Scripts\python.exe -m quantstream.terminal --root workspace import-tcn --folder "%checkpoint_folder%"
if errorlevel 1 goto fail
echo.
echo Imported without training. Open the app and select the imported TCN.
pause
exit /b 0
:fail
echo Import failed. Keep the original files and read the error above.
pause
exit /b 1
