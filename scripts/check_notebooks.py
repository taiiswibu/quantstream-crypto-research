"""Check portable notebook structure and Python syntax without running training."""
import ast
from pathlib import Path

import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "notebooks").glob("*.ipynb"))
    for path in files:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path.name}:cell{index}")
        print(f"OK: {path.name}")
    if not files:
        raise RuntimeError("No notebooks found")


if __name__ == "__main__":
    main()
