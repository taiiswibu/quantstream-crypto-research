"""Source and release launchers must address the intended local workspace."""
import os
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('source_checkout', [True, False])
def test_launcher_uses_project_root_for_both_layouts(tmp_path, monkeypatch, source_checkout):
    original = Path(__file__).resolve().parents[1]/'launchers/run_local.py'
    root = tmp_path/'terminal'
    folder = root/'launchers' if source_checkout else root
    folder.mkdir(parents=True)
    script = folder/'run_local.py'
    shutil.copy2(original, script)
    (root/'app.py').write_text('pass\n')
    if source_checkout:
        (root/'pyproject.toml').write_text('[project]\nname="test-terminal"\n')
    calls = []
    def capture(command, **kwargs):
        calls.append((command, kwargs, Path.cwd(), os.environ['QUANTSTREAM_ROOT']))
    monkeypatch.setattr(subprocess, 'run', capture)
    monkeypatch.setenv('QUANTSTREAM_ROOT', 'previous-workspace')
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name='__main__')
    command, options, cwd, workspace = calls[0]
    assert cwd == root
    assert Path(workspace) == root/'workspace'
    assert command[command.index('run')+1] == str(root/'app.py')
    assert command[command.index('--server.address')+1] == '127.0.0.1'
    assert options['check'] is True
