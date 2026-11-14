"""Generate Notebook 10 without altering the executed Run 006 notebook."""

import argparse
from pathlib import Path

import nbformat as nbf

from build_workflow_notebook import build

ROOT = Path(__file__).resolve().parents[1]


def extended(source):
    notebook = build(source, write=False)
    for cell in notebook.cells:
        cell.source = cell.source.replace("Run006", "Run007").replace(
            "Run 006", "Run 007"
        )
    first = notebook.cells[0]
    first.source = first.source.replace("N_TRIALS = 24", "N_TRIALS = 48")
    first.source = first.source.replace("MAX_EPOCHS = 12", "MAX_EPOCHS = 60")
    first.source = first.source.replace("JOB_SECONDS = 7200", "JOB_SECONDS = 14400")
    first.source = first.source.replace(
        "assert 4 <= N_TRIALS <= 24 and 1 <= MAX_EPOCHS <= 12 and JOB_SECONDS <= 21600",
        "assert 14 <= N_TRIALS <= 48 and 1 <= MAX_EPOCHS <= 60 and 0 < JOB_SECONDS <= 14400",
    )
    first.source = first.source.replace(
        "from quantstream.workflow import prepare_dataset",
        "from quantstream.extended_training import prepare_dataset, find_parent_snapshot",
    )
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "manifest = prepare_dataset(WORK)",
        "parent = find_parent_snapshot(Path('/kaggle/input'), BASE/'run006-dataset-cache')\n"
        "print('Snapshot Run 006:',str(parent) if parent else 'Không có input; thu thập lại cùng khoảng ngày đã khóa')\n"
        "manifest = prepare_dataset(WORK, parent)",
    )
    notebook.cells[
        3
    ].source += (
        "\nprint('Run 007 dùng ba fold validation; test cũ chỉ là chẩn đoán hồi cứu.')"
    )
    train = notebook.cells[7]
    train.source = train.source.replace("2 finalist", "3 finalist")
    train.source = train.source.replace(
        "'--device','cuda'", "'--device','cuda','--protocol','run007'"
    )
    train.source = train.source.replace(
        "old['source']['commit']==SOURCE_COMMIT",
        "old['protocol']=='run007-depth-v1' and old['source']['commit']==SOURCE_COMMIT",
    )
    notebook.cells[8].source = notebook.cells[8].source.replace("head(24)", "head(48)")
    notebook.cells[
        8
    ].source += "\nfrom quantstream.extended_training import figures\nfigures(WORK)"
    notebook.cells[
        9
    ].source += "\ndisplay(pd.DataFrame([selection['validation_gate']]))\nprint('Gate được khóa bằng validation; không phải quyền giao dịch tiền thật.')"
    notebook.cells[
        11
    ].source += (
        "\nprint('Training last label UTC:', exported.get('training_last_label'))"
    )
    nbf.validate(notebook)
    destination = ROOT / "notebooks/10_Kaggle_Extended_Training.ipynb"
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    extended(parser.parse_args().source)
