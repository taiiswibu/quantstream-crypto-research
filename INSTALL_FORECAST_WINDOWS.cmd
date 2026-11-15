@echo off
setlocal
cd /d "%~dp0"
call INSTALL_WINDOWS.cmd --inference-only
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install "tirex-2==0.5.0" "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m quantstream.modern_models --model Timestral-5M --folder artifacts\forecast\Timestral-5M
if errorlevel 1 goto fail
echo.
echo Timestral-5M is ready. Open START_WINDOWS.cmd and the forecast page.
exit /b 0
:fail
echo.
echo Forecast extension setup failed. The core application can still run.
pause
exit /b 1
