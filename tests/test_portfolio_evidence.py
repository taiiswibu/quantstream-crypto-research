"""Published plots must be the exact outputs of the retained executed notebooks."""
import base64
import hashlib
import json
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_every_retained_notebook_completed_and_images_match_its_outputs():
    paths = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    assert [p.name for p in paths] == [
        '01_Data.ipynb', '02_Training.ipynb', '03_Evaluation.ipynb', '04_Inference.ipynb']
    total_images = 0
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        assert nb.metadata.execution.completed is True
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
        manifest = json.loads((ROOT / 'reports/final_study/figures' / (path.stem + '-images.json')).read_text())
        assert manifest['notebook_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        for record in manifest['images']:
            cell = nb.cells[record['cell_index_zero_based']]
            output = cell.outputs[record['output_index']]
            payload = base64.b64decode(output.data[record['mime']])
            image_path = ROOT / 'reports/final_study/figures' / record['file']
            assert image_path.read_bytes() == payload
            assert hashlib.sha256(payload).hexdigest() == record['sha256']
            assert cell.execution_count == record['execution_count']
        total_images += len(manifest['images'])
    assert total_images == 17


def test_readme_images_and_local_links_exist_and_attribution_is_preserved():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if target.startswith(('http:', 'https:', '#')):
            continue
        assert (ROOT / target.split('#')[0]).exists(), target
    images = re.findall(r'!\[.*?\]\(([^)]+)\)', readme)
    manifest = json.loads((ROOT / 'reports/app/screenshots.json').read_text(encoding='utf-8'))
    expected_app = {'reports/app/' + entry['file'] for entry in manifest['images']}
    assert {path for path in images if path.startswith('reports/app/')} == expected_app
    assert len(images) == 17 + len(expected_app)
    for entry in manifest['images']:
        payload = (ROOT / 'reports/app' / entry['file']).read_bytes()
        assert payload.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(payload) == entry['bytes']
        assert hashlib.sha256(payload).hexdigest() == entry['sha256']
    assert 'Apache-2.0' in readme and 'Binance public data terms' in readme
    assert (ROOT / 'LICENSE').is_file()


def test_execution_source_hashes_and_fold_boundaries_are_auditable():
    record = json.loads((ROOT / 'reports/final_study/execution.json').read_text())
    for path, expected in record['source_files_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert record['independent_final_test'] is False
    import csv
    rows = list(csv.DictReader((ROOT / 'reports/final_study/splits.csv').open()))
    assert len(rows) == 3
    assert all(row['last_train_label'] < row['validation_start'] for row in rows)
