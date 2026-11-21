"""Export image output bytes from an executed notebook, retaining cell provenance."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import nbformat


def export(notebook_path, output_dir):
    notebook_path, output_dir = Path(notebook_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    records = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        if any(output.output_type == 'error' for output in cell.outputs):
            raise ValueError(f'Notebook có lỗi tại cell {index}')
        for output_index, output in enumerate(cell.outputs):
            for mime, suffix in [('image/png', '.png'), ('image/jpeg', '.jpg')]:
                encoded = output.get('data', {}).get(mime)
                if encoded is None:
                    continue
                if cell.execution_count is None:
                    raise ValueError(f'Ảnh chưa có execution_count tại cell {index}')
                if isinstance(encoded, list):
                    encoded = ''.join(encoded)
                payload = base64.b64decode(encoded, validate=True)
                signature = b'\x89PNG\r\n\x1a\n' if mime == 'image/png' else b'\xff\xd8'
                if not payload.startswith(signature):
                    raise ValueError(f'Image MIME không khớp bytes tại cell {index}')
                stem = cell.metadata.get('evidence_name', f'cell-{index:02d}')
                stem = ''.join(c for c in stem if c.isalnum() or c in '-_')
                filename = f'{notebook_path.stem}-{stem}-{output_index}{suffix}'
                (output_dir / filename).write_bytes(payload)
                records.append({'file': filename, 'cell_index_zero_based': index,
                                'output_index': output_index, 'execution_count': cell.execution_count,
                                'mime': mime, 'bytes': len(payload),
                                'sha256': hashlib.sha256(payload).hexdigest()})
    manifest = {'notebook': notebook_path.name,
                'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                'images': records}
    (output_dir / f'{notebook_path.stem}-images.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.notebook, args.output)
    print(f"Đã xuất {len(result['images'])} ảnh từ {result['notebook']}")
