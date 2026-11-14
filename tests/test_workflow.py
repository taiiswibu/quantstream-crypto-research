"""Chronological boundaries, whole-job budget and artifact export contracts."""

from pathlib import Path
from dataclasses import replace
import pytest
import numpy as np
import pandas as pd

from quantstream import workflow
from quantstream.common import Config, sha256, write_json, write_parquet
from quantstream.features import build_features, FEATURES, fold_data
from test_integrity import fixture_bars


def test_workflow_split_purges_every_boundary():
    data = build_features(fixture_bars(), Config())
    split = workflow.make_split(data.dropna(subset=["target_return"]))
    for bounds in split["tuning_folds"] + [
        {"validation_start": split["test_start"], "validation_end": split["end"]}
    ]:
        train, val = fold_data(
            data, bounds["validation_start"], bounds["validation_end"]
        )
        assert train.label_end_time.max() < val.decision_time.min()
    assert split["test_is_independent"] is False


def test_budget_expires_without_extra_fit(monkeypatch):
    clock = iter([0, 2])
    monkeypatch.setattr(workflow.time, "monotonic", lambda: next(clock))
    budget = workflow.Budget(1)
    with pytest.raises(workflow.BudgetExpired):
        budget.check()


def synthetic_dataset(root):
    cfg = replace(
        Config(), symbols=("BTCUSDT",), position_fraction=0.1, drawdown_halt=0.1
    )
    bars = fixture_bars(1800)
    data = build_features(bars, cfg)
    split = workflow.make_split(data.dropna(subset=["target_return"]))
    write_parquet(root / "data/processed/bars.parquet", bars)
    write_parquet(root / "data/processed/features.parquet", data)
    write_json(root / "split.json", split)
    manifest = {
        "config": cfg.to_dict(),
        "features": FEATURES,
        "split": split,
        "feature_code_sha256": sha256(Path(workflow.__file__).with_name("features.py")),
        "files": {
            str(p.relative_to(root)): sha256(p)
            for p in [
                root / "data/processed/bars.parquet",
                root / "data/processed/features.parquet",
                root / "split.json",
            ]
        },
    }
    write_json(root / "dataset_manifest.json", manifest)
    return cfg


def test_small_end_to_end_job_exports_reproducible_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    cfg = synthetic_dataset(tmp_path)
    workflow.run(tmp_path, n_trials=4, max_epochs=1, seconds=180, device="cpu")
    import json

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "completed"
    assert (tmp_path / "deployment/bundle_manifest.json").exists()
    assert (tmp_path / "retrospective-test.started.json").exists()
    from quantstream.tcn_import import import_bundle

    runtime = tmp_path / "live"
    write_json(runtime / "config.json", cfg.to_dict())
    meta = import_bundle(runtime, tmp_path / "deployment")
    assert meta["parity_max_abs_delta"] < 1e-5 and meta["import_optimizer_steps"] == 0
    with pytest.raises(ValueError, match="already started"):
        workflow.run(tmp_path, n_trials=4, max_epochs=1, seconds=180, device="cpu")


@pytest.mark.parametrize("family", ["tcn", "gru", "xgboost"])
def test_portable_neural_and_booster_roundtrip(tmp_path, family):
    pytest.importorskip("torch")
    from quantstream.search import SequenceRegressor
    from quantstream.tcn_import import import_bundle
    from quantstream import terminal

    cfg = synthetic_dataset(tmp_path)
    data = pd.read_parquet(tmp_path / "data/processed/features.parquet")
    manifest = __import__("json").loads(
        (tmp_path / "dataset_manifest.json").read_text()
    )
    if family == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)
        model.fit(data[FEATURES], data.target_return.fillna(0))
        params = {
            "family": family,
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.1,
        }
        fitted = {
            "model": model,
            "params": params,
            "epochs": None,
            "training_rows": len(data),
        }
    else:
        params = {
            "family": family,
            "width": 16,
            "dropout": 0.15,
            "lookback": 48,
            "activation": "relu",
        }
        model = SequenceRegressor(params)
        fitted = {
            "model": model,
            "params": params,
            "mean": np.zeros(18),
            "std": np.ones(18),
            "device": "cpu",
            "epochs": 1,
            "training_rows": len(data),
        }
    folder = workflow.export_model(
        tmp_path, fitted, cfg, manifest, {"validation": None}, None, data
    )
    runtime = tmp_path / "live"
    write_json(runtime / "config.json", cfg.to_dict())
    meta = import_bundle(runtime, folder)
    assert meta["parity_max_abs_delta"] < 1e-5
    terminal.activate(runtime, meta["model"])
    write_parquet(runtime / "data/processed/bars.parquet", fixture_bars(300))
    forecasts = terminal.predict_latest(runtime)
    assert len(forecasts) == 1 and np.isfinite(forecasts[0]["prediction"])
