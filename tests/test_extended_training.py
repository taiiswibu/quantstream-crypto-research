"""Real protocol invariants; synthetic fixtures never measure market performance."""

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantstream import extended_training, workflow
from quantstream.common import Config
from quantstream.features import build_features, FEATURES, fold_data
from test_integrity import fixture_bars
from test_workflow import synthetic_dataset


def test_snapshot_reuse_changes_only_validation_split_and_preserves_parent(tmp_path):
    parent, root = tmp_path / "parent", tmp_path / "run007"
    synthetic_dataset(parent)
    original = (parent / "dataset_manifest.json").read_bytes()
    manifest = extended_training.prepare_dataset(root, parent)
    assert (parent / "dataset_manifest.json").read_bytes() == original
    assert len(manifest["split"]["tuning_folds"]) == 3
    assert not (root / "deployment").exists()
    data = pd.read_parquet(root / "data/processed/features.parquet")
    for bounds in manifest["split"]["tuning_folds"]:
        train, val = fold_data(
            data, bounds["validation_start"], bounds["validation_end"]
        )
        assert train.label_end_time.max() < val.decision_time.min()
    assert extended_training.prepare_dataset(root, parent) == manifest
    with (root / "data/processed/features.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        extended_training.prepare_dataset(root, parent)


def test_validation_gate_cannot_hide_a_losing_fold_behind_the_mean():
    reports = [
        {
            "folds": [
                {"prediction": {"mse_ratio_to_zero": ratio}}
                for ratio in [0.9, 0.92, 1.01]
            ]
        }
        for _ in range(3)
    ]
    result = extended_training.validation_gate(reports)
    assert result["mean_seed_fold_mse_ratio"] < 1
    assert not result["beats_zero_on_mean_and_every_fold"]


def test_attached_snapshot_zip_extracts_only_dataset_members(tmp_path):
    import zipfile
    from quantstream.common import write_json

    parent = tmp_path / "parent"
    synthetic_dataset(parent)
    manifest = json.loads((parent / "dataset_manifest.json").read_text())
    # Discovery flags simulate a real attachment; values remain fixture-only.
    manifest.update(synthetic=False, frozen_end="2026-11-14 00:00:00+00:00")
    write_json(parent / "dataset_manifest.json", manifest)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with zipfile.ZipFile(inputs / "QuantStream_Run006_Output.zip", "w") as zipped:
        for name in ["dataset_manifest.json", *manifest["files"]]:
            zipped.write(parent / name, "nested/" + name)
        zipped.writestr("nested/deployment/model.pt", "do not extract")
        zipped.writestr("../../outside.txt", "do not extract")
    found = extended_training.find_parent_snapshot(inputs, tmp_path / "cache")
    assert found and (found / "data/processed/features.parquet").exists()
    assert not (found / "deployment/model.pt").exists()
    assert not (tmp_path / "outside.txt").exists()
    extended_training.prepare_dataset(tmp_path / "reused", found)


def test_longer_training_replays_scheduler_and_uses_recent_train_only(tmp_path):
    pytest.importorskip("torch")
    cfg = replace(Config(), symbols=("BTCUSDT",), threads=1)
    data = build_features(fixture_bars(600), cfg).dropna(subset=["target_return"])
    params = {
        "family": "tcn",
        "width": 16,
        "dropout": 0.0,
        "activation": "silu",
        "lookback": 48,
        "batch_size": 64,
        "lr": 0.0005,
        "weight_decay": 0.001,
        "loss": "mse",
        "lr_schedule": "cosine",
        "schedule_epochs": 60,
        "epoch_cap": 60,
        "patience": 8,
        "train_window_days": 5,
    }
    fitted = workflow.fit_model(
        data, params, 42, 4, cfg, "cpu", workflow.Budget(30), tmp_path
    )
    early = pd.read_csv(tmp_path / "early-stop/history.csv")
    refit = pd.read_csv(tmp_path / "refit/history.csv")
    np.testing.assert_allclose(
        refit.learning_rate, early.learning_rate.iloc[: len(refit)]
    )
    recent = data[
        data.decision_time.ge(data.decision_time.max() - pd.Timedelta(days=5))
    ]
    assert fitted["training_rows"] == len(recent) < len(data)
    np.testing.assert_allclose(
        fitted["mean"], recent[FEATURES].mean().to_numpy(np.float32)
    )
    assert "train_huber_scaled" not in refit
    assert set(refit.loss_name) == {"mse"}
    assert json.loads((tmp_path / "fit.json").read_text())["rows"] == len(recent)


def test_run007_full_pipeline_and_mse_model_ui(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    from streamlit.testing.v1 import AppTest
    from quantstream import ui_app, terminal, paper
    from quantstream.tcn_import import import_bundle

    parent, root = tmp_path / "parent", tmp_path / "run007"
    cfg = synthetic_dataset(parent)
    extended_training.prepare_dataset(root, parent)
    # A one-epoch fixture checks every anchor, three folds and export without GPU research.
    workflow.run(
        root, n_trials=14, max_epochs=1, seconds=180, device="cpu", protocol="run007"
    )
    meta = json.loads((root / "deployment/metadata.json").read_text())
    assert meta["research_run"] == "run007" and meta["training_last_label"]
    assert "validation_gate" in meta
    assert (
        json.loads((root / "training_plan.json").read_text())["protocol"]
        == "run007-depth-v1"
    )
    assert len(pd.read_csv(root / "trials.csv")) == 14
    # Force a separate MSE neural export to cover its app loss-history contract.
    data = build_features(fixture_bars(600), cfg).dropna(subset=["target_return"])
    params = {
        "family": "tcn",
        "width": 16,
        "dropout": 0.0,
        "activation": "silu",
        "lookback": 48,
        "batch_size": 64,
        "lr": 0.0005,
        "weight_decay": 0.001,
        "loss": "mse",
        "epoch_cap": 1,
    }
    bundle_root = tmp_path / "mse-model"
    fitted = workflow.fit_model(
        data, params, 42, 1, cfg, "cpu", workflow.Budget(30), bundle_root / "final-fit"
    )
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    bundle = workflow.export_model(
        bundle_root,
        fitted,
        cfg,
        manifest,
        {"validation": None},
        None,
        data,
        run_name="run007",
    )
    live = tmp_path / "live"
    from quantstream.common import write_json, write_parquet

    write_json(live / "config.json", cfg.to_dict())
    imported = import_bundle(live, bundle)
    assert imported["origin"]["label"] == "Notebook 10 · TCN"
    write_parquet(live / "data/processed/bars.parquet", fixture_bars(600))
    terminal.activate(live, imported["model"])
    terminal.predict_latest(live)
    paper.paper_step(
        live,
        cfg,
        refresh_data=False,
        price_provider=lambda _: 100,
        allow_new_entries=False,
    )
    monkeypatch.setattr(ui_app, "ROOT", live)
    app = AppTest.from_file(
        str(Path(ui_app.__file__).parents[1] / "app.py"), default_timeout=30
    ).run()
    app.radio[0].set_value("Kiểm định").run()
    assert not app.exception, [error.message for error in app.exception]
    # Execute the generated notebook's export cell against actual fixture artifacts.
    monkeypatch.syspath_prepend(str(Path(ui_app.__file__).parents[1] / "scripts"))
    import build_extended_notebook
    import nbformat
    from quantstream.common import sha256

    notebook_root = tmp_path / "generated"
    (notebook_root / "notebooks").mkdir(parents=True)
    monkeypatch.setattr(build_extended_notebook, "ROOT", notebook_root)
    build_extended_notebook.extended("0" * 40)
    notebook = nbformat.read(
        notebook_root / "notebooks/10_Kaggle_Extended_Training.ipynb", as_version=4
    )
    assert "'--protocol','run007'" in notebook.cells[7].source
    exec(
        notebook.cells[11].source,
        {
            "WORK": root,
            "json": json,
            "pd": pd,
            "sha256": sha256,
            "display": lambda _: None,
        },
    )
