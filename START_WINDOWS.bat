@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 py -3.11 -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Install Python 3.11 or 3.12, then run this file again.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 (
  pause
  exit /b 1
)
if not exist artifacts\champion.json (
  .venv\Scripts\python.exe -m quantstream demo
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -c "import json,sys; from pathlib import Path; p=Path('artifacts/champion.json'); sys.exit(1 if p.exists() and json.loads(p.read_text())['type']=='tcn' else 0)"
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
