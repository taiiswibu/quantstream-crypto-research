"""Avoid changing installed packages underneath cached notebook imports."""
import sys
import types
import zipfile
from pathlib import Path

import nbformat
import pytest


@pytest.mark.parametrize("module", ["peft", "torchao", "chronos", "transformers.utils"])
def test_setup_requires_fresh_session_before_running_pip(module, monkeypatch):
    notebook = nbformat.read(Path(__file__).resolve().parents[1] /
                             "notebooks/05_Kaggle_LoRA_Lab.ipynb", as_version=4)
    setup = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    calls = []
    monkeypatch.setattr("subprocess.check_call", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    with pytest.raises(RuntimeError, match="Restart Session"):
        exec(compile(setup, "lora-setup", "exec"), {})
    assert calls == []


def test_failed_fit_keeps_logs_without_overwriting_original_zip(tmp_path):
    notebook = nbformat.read(Path(__file__).resolve().parents[1] /
                             "notebooks/05_Kaggle_LoRA_Lab.ipynb", as_version=4)
    fit_cell = next(cell.source for cell in notebook.cells
                    if cell.cell_type == "code" and cell.source.startswith("reports=[]"))
    work = tmp_path / "working"
    output = work / "run-003-lora-lab-rerun"
    output.mkdir(parents=True)
    original = tmp_path / "QuantStream_LoRA_Output.zip"
    original.write_bytes(b"original run 003 evidence")

    def failed_fit(snapshot, folder, **kwargs):
        checkpoint = folder / "trainer/checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
        (checkpoint / "optimizer.pt").write_bytes(b"large optimizer fixture")
        raise RuntimeError("fit interrupted")

    namespace = {"WORK": work, "OUTPUT": output, "SNAPSHOT": tmp_path,
                 "CONFIGS": [{"id": "A", "steps": 100, "learning_rate": 1e-5}],
                 "BASE_REVISION": "test", "chronos_experiment": failed_fit,
                 "zipfile": zipfile, "display": lambda value: None, "FileLink": str}
    with pytest.raises(RuntimeError, match="fit interrupted"):
        exec(compile(fit_cell, "lora-fit", "exec"), namespace)
    with zipfile.ZipFile(tmp_path / "QuantStream_LoRA_Rerun_Output.zip") as archive:
        assert archive.namelist() == ["A/trainer/checkpoint-1/trainer_state.json"]
    assert original.read_bytes() == b"original run 003 evidence"
    assert not (output / "comparison.json").exists()
