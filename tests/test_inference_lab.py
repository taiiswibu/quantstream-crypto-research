"""Fixed-weight mode/RNG controls and artifact discovery for Run 005."""
import ast
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import pytest

from quantstream import inference_lab as lab
from quantstream.common import Config


def test_experiment_has_no_fit_backward_optimizer_or_merge_calls():
    path = Path(lab.__file__)
    tree = ast.parse(path.read_text())
    prohibited = {"fit", "backward", "step", "merge_and_unload", "save_pretrained"}
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)]
    assert not prohibited.intersection(calls)
    assert len(lab.CASES) == 8


def test_prepare_inputs_accepts_expanded_kaggle_upload(tmp_path):
    inputs, work = tmp_path / "inputs", tmp_path / "work"
    snapshot = inputs / "dataset"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}")
    (snapshot / "development.parquet").write_bytes(b"fixture")
    artifact = inputs / "run004"
    (artifact / "seed-42/adapter").mkdir(parents=True)
    (artifact / "plan.json").write_text("{}")
    (artifact / "seed-42/chronos_validation.parquet").write_bytes(b"fixture")
    (artifact / "seed-42/adapter/adapter_model.safetensors").write_bytes(b"fixture")
    work.mkdir()
    actual_snapshot, actual_artifact = lab.prepare_inputs(inputs, work)
    assert actual_snapshot == work / "snapshot" and actual_artifact == artifact
    assert (actual_snapshot / "development.parquet").read_bytes() == b"fixture"


def test_prepare_inputs_rejects_ambiguous_or_unsafe_archives(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(FileNotFoundError, match="Run 004"):
        lab.prepare_inputs(tmp_path, work)
    (tmp_path / "QuantStream_Run004_Output.zip").write_bytes(b"not read in discovery")
    with zipfile.ZipFile(tmp_path / "QuantStream_Kaggle_Input.zip", "w") as archive:
        archive.writestr("../outside.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        lab.prepare_inputs(tmp_path, work)
    assert not (tmp_path.parent / "outside.json").exists()


def test_reference_rejects_modified_zip_before_extracting_weights(tmp_path):
    path = tmp_path / "QuantStream_Run004_Output.zip"
    path.write_bytes(b"modified archive")
    with pytest.raises(ValueError, match="reviewed completed"):
        lab.read_reference(path, tmp_path, tmp_path)
    assert not list(tmp_path.glob("adapter-*"))


def test_notebook_has_one_path_and_setup_failure_keeps_zero_step_evidence(tmp_path, monkeypatch):
    import nbformat
    import platform
    root = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(root / "scripts/build_inference_notebook.py"))
    notebook_path = tmp_path / "07.ipynb"
    builder["build"]("a" * 40, notebook_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for cell in cells:
        ast.parse(cell)
    assert not any("MODE=" in cell or "MODE =" in cell for cell in cells)
    assert any("report=run_inference_lab(" in cell for cell in cells)
    setup = next(cell for cell in cells if cell.startswith("import os, importlib.util"))
    code = tmp_path / "source"
    lock = code / "reports/experiments/run-004-seed-stability/kaggle-preflight-lock-rerun-001.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"environment": {"python": "not-this-python",
        "packages": {"peft": "0.21.2", "numpy": "1.26.4"}}}))
    output, archive = tmp_path / "output", tmp_path / "evidence.zip"
    calls = []
    monkeypatch.setattr(subprocess, "check_call", lambda *args, **kwargs: calls.append(args))
    namespace = {"CODE": code, "WORK": tmp_path, "OUTPUT": output, "ARCHIVE": archive,
                 "json": json, "sys": sys, "platform": platform, "subprocess": subprocess,
                 "traceback": traceback, "zipfile": zipfile, "display": lambda value: None, "FileLink": str}
    with pytest.raises(RuntimeError, match="Python base image"):
        exec(compile(setup, "inference-setup", "exec"), namespace)
    assert calls == []
    assert (tmp_path / "run004-constraints.txt").read_text() == "peft==0.21.2\nnumpy==1.26.4\n"
    with zipfile.ZipFile(archive) as saved:
        assert json.loads(saved.read("status.json"))["optimizer_steps"] == 0
        assert "setup-error.txt" in saved.namelist()


def test_real_peft_reload_and_chronos_mode_controls_without_training(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("chronos")
    from chronos import Chronos2Pipeline
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Chronos2CoreConfig(d_model=32, d_kv=16, d_ff=64, num_layers=1, num_heads=2,
        dropout_rate=0.3, chronos_config={"context_length": 128, "output_patch_size": 16,
            "input_patch_size": 16, "input_patch_stride": 16,
            "quantiles": [index / 10 for index in range(1, 10)], "max_output_patches": 1, "use_reg_token": True})
    base = Chronos2Model(config)
    base_state = copy.deepcopy(base.state_dict())
    adapted = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["self_attention.q", "self_attention.v"]))
    # Synthetic fixture initialization, not optimizer updates.
    with torch.no_grad():
        for name, value in adapted.named_parameters():
            if "lora_B" in name:
                value.fill_(0.01)
    adapter = tmp_path / "adapter"
    adapted.save_pretrained(adapter)
    contexts = [np.sin(np.arange(128, dtype=np.float32) / scale) for scale in (9, 13, 17, 23)]
    scored = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        "decision_time": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
        "target_return": [0.01, -0.02, 0.03, -0.01]})
    original = Chronos2Pipeline(adapted)
    original.model.eval()
    expected = lab.forecast_returns(original, contexts, Config())
    reloaded_base = Chronos2Model(copy.deepcopy(config))
    reloaded_base.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter, is_trainable=False)
    pipe = Chronos2Pipeline(reloaded)
    pipe.model.eval()
    assert lab.difference(expected, lab.forecast_returns(pipe, contexts, Config()))["within_tolerance"]
    monkeypatch.setattr(Chronos2Pipeline, "fit", lambda *args, **kwargs: pytest.fail("Inference must not fit"))
    before = lab.state_fingerprint(pipe.model)
    result = lab.controlled_passes(pipe, contexts, scored, Config(), tmp_path / "diagnostic",
                                  scored.assign(finetuned_return=expected))
    assert result["controls_passed"] and result["completed_passes"] == 8
    assert result["mode_effect_observed"] and result["rng_effect_with_dropout_observed"]
    assert result["checks"]["run004_eval_reference"]["within_tolerance"]
    assert lab.state_fingerprint(pipe.model) == before and not pipe.model.training
    assert len(result["metrics"]) == 24
    assert all(row["trainable_parameters"] == row["optimizer_steps"] == 0 for row in result["passes"])
    assert all(not row["grad_enabled_during_prediction"] for row in result["passes"])
    assert all(row["merged_lora_layers"] == 0 for row in result["passes"])
    assert len(list((tmp_path / "diagnostic/passes").glob("*.parquet"))) == 8


def test_state_mutation_fails_and_preserves_partial_evidence(tmp_path):
    torch = pytest.importorskip("torch")
    class MutatingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))
    class Pipeline:
        model = MutatingModel()
        def predict_quantiles(self, contexts, **kwargs):
            self.model.counter.add_(1)
            return None, [torch.zeros(1, 5) for _ in contexts]
    frame = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"],
                          "decision_time": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
                          "target_return": [0.01, -0.01]})
    pipe = Pipeline()
    with pytest.raises(RuntimeError, match="parameters/buffers changed"):
        lab.controlled_passes(pipe, [np.zeros(128)] * 2, frame, Config(), tmp_path,
                              frame.assign(finetuned_return=0))
    evidence = json.loads((tmp_path / "passes/eval_42_a.json").read_text())
    assert evidence["state_unchanged"] is False
    assert not (tmp_path / "comparison.json").exists() and not pipe.model.training
