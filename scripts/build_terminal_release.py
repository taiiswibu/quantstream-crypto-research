"""Build an installable local terminal with its current trained baseline artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from filelock import FileLock
from quantstream.common import git_provenance

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / 'workspace'
    if (workspace / 'paper/state.json').exists():
        raise ValueError('Release requires a fresh paper workspace; archive the session first')
    with tempfile.TemporaryDirectory(prefix='quantstream-release-') as tmp:
        stage = Path(tmp) / 'QuantStream_Local_Terminal'
        stage.mkdir()
        for file in (ROOT / 'launchers').iterdir():
            shutil.copy2(file, stage / file.name)
        shutil.copy2(ROOT / 'app.py', stage / 'app.py')
        shutil.copy2(ROOT / 'docs/LOCAL_TERMINAL_VI.md', stage / 'HUONG_DAN.md')
        shutil.copytree(ROOT / '.streamlit', stage / '.streamlit')
        shutil.copytree(ROOT / 'notebooks', stage / 'notebooks', ignore=shutil.ignore_patterns('.ipynb_checkpoints'))
        for name in ('LICENSE',):
            if (ROOT / name).exists():
                shutil.copy2(ROOT / name, stage / name)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps', '--no-build-isolation', '-w', str(stage / 'wheels')], check=True)
        with FileLock(str(workspace / 'terminal.lock'), timeout=5):
            active_run = json.loads((workspace/'artifacts/latest_run.json').read_text())['run']
            for name in ('config.json', 'health.json', 'quotes.json', 'latest_forecasts.json'):
                if (workspace/name).exists():
                    target=stage/'workspace'/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(workspace/name, target)
            for folder in ('artifacts', 'data', active_run):
                shutil.copytree(workspace/folder, stage/'workspace'/folder)
            database=workspace/'terminal.sqlite'
            if database.exists():
                with sqlite3.connect(database) as source, sqlite3.connect(stage/'workspace/terminal.sqlite') as target:
                    source.backup(target)
        (stage/'workspace/controls.json').write_text('{"paper_enabled": false}\n')
        validation = {'source': git_provenance(ROOT), 'mode': 'local_paper_only',
                      'includes_live_exchange_orders': False, 'includes_llm': False,
                      'windows_execution_verified': False,
                      'kaggle_training_rerun_for_notebook_style': False,
                      'local_review_notebook': 'executed top-to-bottom using in-process IPython; saved figures inspected',
                      'app_validation': 'Streamlit AppTest all five routes; direct browser review',
                      'files': {str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(stage.rglob('*')) if p.is_file()}}
        (stage/'RELEASE.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
        zip_path=destination/'QuantStream_Local_Terminal.zip'
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent))
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.testzip() is None
        print(zip_path, zip_path.stat().st_size, 'bytes')
        return zip_path


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv)>1 else ROOT.parent/'deliverables')
