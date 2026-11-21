"""Notebook image evidence must retain the executed output bytes and source cell."""
import base64
import hashlib
import json

import nbformat
import pytest

from scripts.export_notebook_images import export


def notebook_file(tmp_path, executed=True, failure=False):
    payload = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0u8AAAAASUVORK5CYII=')
    output = nbformat.v4.new_output('display_data', data={
        'image/png': base64.b64encode(payload).decode()})
    cell = nbformat.v4.new_code_cell('print("fixture")',
        execution_count=1 if executed else None, outputs=[output])
    if failure:
        cell.outputs.append(nbformat.v4.new_output('error', ename='ValueError',
            evalue='fixture', traceback=['fixture']))
    path = tmp_path / 'evidence.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
    return path, payload


def test_export_preserves_bytes_and_cell_identity(tmp_path):
    path, payload = notebook_file(tmp_path)
    manifest = export(path, tmp_path / 'images')
    record = manifest['images'][0]
    assert (tmp_path / 'images' / record['file']).read_bytes() == payload
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['cell_index_zero_based'] == 0
    assert record['execution_count'] == 1
    saved = json.loads((tmp_path / 'images/evidence-images.json').read_text())
    assert saved['notebook_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('executed,failure', [(False, False), (True, True)])
def test_rejects_unexecuted_images_or_failed_notebook(tmp_path, executed, failure):
    path, _ = notebook_file(tmp_path, executed, failure)
    with pytest.raises(ValueError):
        export(path, tmp_path / 'images')
