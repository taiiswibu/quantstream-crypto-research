"""Execute the four study notebooks, each in a fresh IPython process.

Uses in-process rich output capture rather than a TCP Jupyter kernel. No cells
are skipped. Errors abort the sequence; image exports retain their exact bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['01_Data.ipynb', '02_Training.ipynb', '03_Evaluation.ipynb', '04_Inference.ipynb']


def execute(name):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline', force=True)
    from matplotlib_inline.backend_inline import configure_inline_support
    from scripts.export_notebook_images import export

    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    path = ROOT / 'notebooks' / name
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs, cell.execution_count = [], None
    started = time.monotonic()
    notebook.metadata['execution'] = {'completed': False, 'engine': 'IPython fresh process, sequential rich capture'}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        with capture_output(display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = result.execution_count
        for stream in ('stdout', 'stderr'):
            value = getattr(captured, stream)
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=stream, text=value))
        cell.outputs.extend(nbformat.v4.new_output('display_data', data=o.data, metadata=o.metadata)
                            for o in captured.outputs)
        if result.error_before_exec or result.error_in_exec:
            error = result.error_before_exec or result.error_in_exec
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__,
                                                       evalue=str(error), traceback=[]))
            nbformat.write(notebook, path)
            result.raise_error()
        print(f'{name} cell {index} completed ({time.monotonic() - started:.1f}s)', flush=True)
    notebook.metadata['execution'].update(completed=True, runtime_seconds=round(time.monotonic() - started, 3),
        completed_at=datetime.now(timezone.utc).isoformat())
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    manifest = export(path, ROOT / 'reports/final_study/figures')
    print(f"COMPLETED {name}: {len(manifest['images'])} real output images", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', choices=NAMES, help='Internal child-process entry point')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.one:
        execute(args.one)
    else:
        for name in NAMES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--one', name], cwd=ROOT, check=True)
