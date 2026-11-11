"""Run 004 isolation, shared populations, actual trainer evidence and interrupted runs."""
import json
from pathlib import Path
import runpy
import zipfile

import nbformat
import numpy as np
import pandas as pd
import pytest

from quantstream import seed_lab as lab
from quantstream.common import Config, sha256, write_json
from quantstream.finetune import chronos_inputs


@pytest.fixture
def predictions():
    # Gapped, unequal calendar-week blocks; BTC/ETH at shared timestamps.
    times = pd.to_datetime(["2025-01-05T23:00Z", "2025-01-06T00:00Z", "2025-01-07T12:00Z"])
    truth = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])
    base = pd.DataFrame({"symbol": ["BTCUSDT"] * 3 + ["ETHUSDT"] * 3,
                         "decision_time": list(times) * 2, "target_return": truth,
                         "zero_shot_return": truth * 0.5, "finetuned_return": truth * 0.25})
    return [base.copy() for _ in lab.SEEDS]


@pytest.mark.parametrize("damage", ["order", "duplicate", "nan", "baseline", "target", "missing"])
def test_alignment_rejects_different_seed_populations(predictions, damage):
    bad = predictions[-1]
    if damage == "order":
        predictions[-1] = bad.iloc[::-1]
    elif damage == "duplicate":
        bad.loc[1, ["symbol", "decision_time"]] = bad.loc[0, ["symbol", "decision_time"]]
    elif damage == "nan":
        bad.loc[0, "finetuned_return"] = np.nan
    elif damage == "baseline":
        bad.loc[0, "zero_shot_return"] += 0.01
    elif damage == "target":
        bad.loc[0, "target_return"] += 0.01
    else:
        predictions[-1] = bad.iloc[:-1]
    with pytest.raises(ValueError):
        lab.validate_alignment(predictions)


def test_summary_reports_all_seeds_assets_baselines_and_no_promotion(predictions):
    result = lab.summarize(predictions)
    assert len(result["metrics"]) == 15  # 3 scopes x (2 baselines + 3 seeds).
    assert len(result["seed_summary"]) == 3
    assert result["diagnostic_decision"] == "reject"
    assert result["deployment_status"] == "research_only_no_promotion"
    assert result["paired_mae_bootstrap"]["blocks"] == 2
    for row in result["metrics"]:
        if row["model"] == "lora_B":
            assert row["mae_delta_vs_zero_shot"] > 0
        if row["model"] == "zero_return":
            assert row["mae_ratio_to_zero"] == row["mse_ratio_to_zero"] == 1
    with pytest.raises(ValueError, match="all three"):
        lab.summarize(predictions[:2])


def test_bootstrap_keeps_utc_weeks_assets_and_boundary_blocks():
    # The Sunday row stays in its short boundary block; Monday BTC/ETH share a block.
    times = ["2025-01-05T23:00Z", "2025-01-06T00:00Z", "2025-01-06T00:00Z"]
    result = lab.weekly_bootstrap(times, np.array([2.0, 0.0, 0.0]))
    assert result["blocks"] == 2
    assert result["estimate"] == pytest.approx(2 / 3)  # Weighted rows, not mean block means.
    assert result["percentile_95"] == [0.0, 2.0]
    assert result == lab.weekly_bootstrap(times, np.array([2.0, 0.0, 0.0]))


def test_lock_rejects_cpu_or_changed_runtime_before_any_research_fit(monkeypatch):
    lock = {"protocol": lab.PROTOCOL, "model_id": lab.MODEL, "revision": lab.REVISION,
            "steps": 200, "seeds": list(lab.SEEDS), "context_length": 128,
            "batch_size": 8, "max_windows": 512, "learning_rate": 1e-5,
            "inference_mode": "eval", "min_past": 64, "forecast_batch_size": 16,
            "forecast_quantiles": [0.1, 0.5, 0.9],
            "smoke": {"device": "cpu", "global_step": 1, "synthetic_only": True}}
    with pytest.raises(ValueError, match="CUDA"):
        lab.validate_lock(lock, Path("unused"))
    lock["smoke"]["device"] = "cuda"
    lock.update({"environment": {"torch": "old"}, "inputs": {}, "source": {}, "lora": {}})
    monkeypatch.setattr(lab, "runtime", lambda: {"torch": "new"})
    monkeypatch.setattr(lab, "inputs_identity", lambda snapshot: {})
    monkeypatch.setattr(lab, "source_identity", lambda: {})
    monkeypatch.setattr(lab, "resolved_lora", lambda: {})
    with pytest.raises(ValueError, match="environment changed"):
        lab.validate_lock(lock, Path("unused"))


def test_sampling_preserves_run003_order_and_excludes_future_from_training(tmp_path):
    times = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    bars = pd.concat([pd.DataFrame({"symbol": symbol, "timestamp": times,
                                   "close": 100 + np.arange(len(times), dtype=float)})
                      for symbol in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
    val = pd.concat([pd.DataFrame({"symbol": symbol, "decision_time": times[200:300],
                                  "label_end_time": times[200:300] + pd.Timedelta(hours=5),
                                  "target_return": np.arange(100) / 1000})
                     for symbol in ("ETHUSDT", "BTCUSDT")], ignore_index=True)
    bars.to_parquet(tmp_path / "bars.parquet", index=False)
    val.to_parquet(tmp_path / "validation.parquet", index=False)
    manifest = {"contains_final_holdout": False, "config": Config().to_dict(),
                "split": {"test_start": str(times[-1]), "folds": [{"validation_start": str(times[200])}]},
                "files": {p.name: sha256(p) for p in tmp_path.glob("*.parquet")}}
    write_json(tmp_path / "manifest.json", manifest)
    _, _, training, contexts, scored = chronos_inputs(tmp_path, max_windows=4)
    assert scored.symbol.tolist() == ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"]
    assert scored.target_return.tolist() == [0, 0.099, 0, 0.099]
    assert [len(series) for series in training] == [199, 199]
    assert np.exp(training[0][-1]) == pytest.approx(298, rel=1e-6)
    # Last close in first evaluation context is strictly before decision time.
    assert np.exp(contexts[0][-1]) == pytest.approx(299, rel=1e-6)


def test_seed_runner_uses_fresh_fits_and_preserves_failure(tmp_path, predictions, monkeypatch):
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"symbol": ["BTCUSDT"] * 256 + ["ETHUSDT"] * 256,
        "decision_time": list(pd.date_range("2025-01-01", periods=256, freq="h", tz="UTC")) * 2,
        "target_return": np.tile([0.01, -0.02], 256),
        "zero_shot_return": 0.0, "finetuned_return": 0.001})
    predictions = [frame.copy() for _ in lab.SEEDS]
    lock_path = tmp_path / "lock.json"
    write_json(lock_path, {"lora": {"r": 8}, "inputs": {"sample_sha256": lab.sample_sha256(frame)}})
    monkeypatch.setattr(lab, "validate_lock", lambda *args: None)
    monkeypatch.setattr(lab, "trainer_evidence", lambda folder: None)
    monkeypatch.setattr(lab, "adapter_identity", lambda folder: {"files": {}})
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def fit(snapshot, folder, **kwargs):
        assert not folder.joinpath("chronos_report.json").exists()
        assert kwargs["revision"] == lab.REVISION and kwargs["steps"] == 200
        assert kwargs["lora_config"] == {"r": 8} and kwargs["logging_steps"] == 1
        calls.append(kwargs["seed"])
        write_json(folder / "trainer_state.json", {"global_step": 200, "log_history": [{"loss": 0.2}]})
        predictions[len(calls) - 1].to_parquet(folder / "chronos_validation.parquet", index=False)
        return {"evaluation_windows": 512, "seed": kwargs["seed"]}

    monkeypatch.setattr(lab, "chronos_experiment", fit)
    output = tmp_path / "complete"
    lab.run_seed_lab(tmp_path, output, lock_path, "https://www.kaggle.com/code/test/run004")
    assert calls == [42, 123, 2026]
    assert json.loads((output / "status.json").read_text())["total_optimizer_steps"] == 600
    with pytest.raises(FileExistsError):
        lab.run_seed_lab(tmp_path, output, lock_path, "https://www.kaggle.com/code/test/run004")
    assert len(calls) == 3

    def failed_fit(snapshot, folder, **kwargs):
        write_json(folder / "trainer_state.json", {"global_step": 17})
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(lab, "chronos_experiment", failed_fit)
    failed = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interruption"):
        lab.run_seed_lab(tmp_path, failed, lock_path, "https://www.kaggle.com/code/test/run004")
    assert not (failed / "comparison.json").exists()
    assert not (failed / "seed-123").exists()
    assert json.loads((failed / "seed-42/trainer_state.json").read_text())["global_step"] == 17
    assert json.loads((failed / "status.json").read_text())["status"] == "failed_or_interrupted"
    original = tmp_path / "QuantStream_LoRA_Output.zip"
    original.write_bytes(b"run003")
    checkpoint = failed / "seed-42/trainer/checkpoint-17"
    write_json(checkpoint / "trainer_state.json", {"global_step": 17})
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    lab.archive_evidence(failed, tmp_path / "run004.zip")
    with zipfile.ZipFile(tmp_path / "run004.zip") as archive:
        assert "seed-42/trainer/checkpoint-17/trainer_state.json" in archive.namelist()
        assert not any(name.endswith("optimizer.pt") for name in archive.namelist())
    assert original.read_bytes() == b"run003"


def test_notebook_defaults_to_preflight_and_pinned_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(root / "scripts/build_seed_notebook.py"))
    path = tmp_path / "06.ipynb"
    builder["build"]("a" * 40, path)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "MODE='preflight'" in cells and f"SOURCE_COMMIT='{'a' * 40}'" in cells
    assert "CODE=WORK/('source-'+SOURCE_COMMIT[:12])" in cells
    assert "verified=verify_run003_snapshot(SNAPSHOT)" in cells
    assert "archive_evidence(target,archive)" in cells
    assert "if target.exists() or archive.exists():" in cells
    with pytest.raises(ValueError, match="immutable"):
        builder["build"]("main", path)


def test_actual_tiny_chronos_lora_trainer_records_step_and_adapter(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("chronos")
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    torch.set_num_threads(2)
    config = Chronos2CoreConfig(d_model=32, d_kv=16, d_ff=64, num_layers=1, num_heads=2,
        chronos_config={"context_length": 128, "output_patch_size": 16, "input_patch_size": 16,
                        "input_patch_stride": 16, "quantiles": [q / 10 for q in range(1, 10)], "max_output_patches": 1,
                        "use_reg_token": True})
    pipe = Chronos2Pipeline(Chronos2Model(config))
    evidence = lab.smoke_pipeline(pipe, tmp_path, lab.resolved_lora(), "cpu")
    assert evidence["global_step"] == 1 and evidence["forecast_finite"]
    assert evidence["inference_model_training"] is False
    assert evidence["adapter"]["tensors"] > 0
    assert (tmp_path / "training_args.json").exists()
    state = json.loads((tmp_path / "trainer_state.json").read_text())
    assert any("loss" in row for row in state["log_history"])
