"""Synthetic checkpoint fixtures test import/inference, never model performance."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from quantstream import terminal, tcn_import
from quantstream.common import sha256, write_json
from quantstream.features import FEATURES, build_features
from test_integrity import fixture_bars

torch = pytest.importorskip("torch")
deep = pytest.importorskip("quantstream.deep")


def checkpoint(folder, cfg):
    folder.mkdir(parents=True)
    torch.manual_seed(42)
    model = deep.TCN()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": [0.0] * 18,
            "std": [1.0] * 18,
            "lookback": 48,
        },
        folder / "model.pt",
    )
    write_json(
        folder / "training_snapshot_manifest.json",
        {
            "full_features_sha256": "fixture",
            "config": cfg.to_dict(),
            "code_files": {
                "quantstream/features.py": sha256(
                    Path(deep.__file__).with_name("features.py")
                )
            },
        },
    )
    metadata = {
        "created_at": "2024-01-01T00:00:00Z",
        "type": "tcn",
        "model": "tcn",
        "features": FEATURES,
        "config": cfg.to_dict(),
        "threshold": 0.006,
        "contains_final_holdout": False,
        "features_sha256": "fixture",
        "snapshot_manifest_sha256": sha256(folder / "training_snapshot_manifest.json"),
        "sha256": sha256(folder / "model.pt"),
        "training_rows": 100,
        "history": [],
    }
    write_json(folder / "tcn_metadata.json", metadata)
    return folder


def test_tcn_import_preserves_weights_and_uses_full_context(tmp_path, monkeypatch):
    root = tmp_path / "live"
    cfg = terminal.config(root)
    write_json(
        root / "artifacts/champion.json", {"model": "before", "sha256": "before"}
    )
    source = checkpoint(tmp_path / "source", cfg)
    before = (source / "model.pt").read_bytes()
    monkeypatch.setattr(
        deep, "train_loop", lambda *a, **k: pytest.fail("Import must not train")
    )
    meta = tcn_import.import_checkpoint(root, source)
    assert (source / "model.pt").read_bytes() == before
    assert terminal.read_json(root / "artifacts/champion.json")["model"] == "before"
    assert (
        tcn_import.import_checkpoint(root, source)["imported_at"] == meta["imported_at"]
    )
    bars = pd.concat(
        [fixture_bars(300, symbol=s) for s in cfg.symbols], ignore_index=True
    )
    delta = (
        pd.Timestamp.now(tz="UTC").floor("h")
        - pd.Timedelta(hours=1)
        - bars.timestamp.max()
    )
    for key in ("timestamp", "source_close_time"):
        bars[key] += delta
    path = root / "data/processed/bars.parquet"
    path.parent.mkdir(parents=True)
    bars.to_parquet(path, index=False)
    terminal.activate(root, meta["model"])
    actual = terminal.predict_latest(root)
    context = build_features(bars, cfg).groupby("symbol").tail(48)
    expected = deep.predict_tcn_file(source / "model.pt", context)
    final = context.groupby("symbol").tail(1)
    expected = pd.Series(expected, index=context.index).loc[final.index].to_numpy()
    assert len(actual) == 2 and all(np.isfinite(row["prediction"]) for row in actual)
    assert np.allclose([r["prediction"] for r in actual], expected)
    assert (source / "model.pt").read_bytes() == before
    assert len(tcn_import.normalized_window(root, meta, "BTCUSDT")) == 48


def test_tcn_refuses_bad_scaler_even_with_matching_checksum(tmp_path):
    cfg = terminal.config(tmp_path / "live")
    folder = checkpoint(tmp_path / "source", cfg)
    payload = torch.load(folder / "model.pt", weights_only=True)
    payload["std"][0] = 0
    torch.save(payload, folder / "model.pt")
    metadata = json.loads((folder / "tcn_metadata.json").read_text())
    metadata["sha256"] = sha256(folder / "model.pt")
    write_json(folder / "tcn_metadata.json", metadata)
    with pytest.raises(ValueError, match="Scaler"):
        tcn_import.import_checkpoint(tmp_path / "live", folder)


def test_registered_reference_identity_is_exact():
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "reports/reference/tcn_metadata.json",
        "reports/experiments/run-001-kaggle-tcn/tcn_metadata.json",
    ]:
        metadata = json.loads((root / relative).read_text())
        assert (
            tcn_import.identity(metadata)
            == tcn_import.KNOWN[metadata["sha256"]]["metadata_identity"]
        )
    cpu = json.loads((root / "reports/reference/tcn_metadata.json").read_text())
    assert tcn_import.KNOWN[cpu["sha256"]]["kind"] == "reference_cpu"
