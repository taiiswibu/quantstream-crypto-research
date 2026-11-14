"""Start the installed terminal on this computer only."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
if root.name == "launchers" and (root.parent / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
os.environ['QUANTSTREAM_ROOT'] = str(root / 'workspace')
subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(root/'app.py'),
                '--server.address', '127.0.0.1', '--server.port', '8501',
                '--browser.gatherUsageStats', 'false'], check=True)
