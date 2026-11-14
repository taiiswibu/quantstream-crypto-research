"""Imported notebook bundles must preserve identity and reproduce saved predictions."""

from pathlib import Path
import json
from dataclasses import replace
import numpy as np
import pytest

from quantstream.common import sha256, write_json
from quantstream.features import FEATURES
from quantstream.tcn_import import import_bundle
from quantstream import terminal


def ridge_bundle(folder, cfg):
    folder.mkdir(parents=True)
    path = folder / "model.json"
    write_json(
        path,
        {
            "mean": [0.0] * 18,
            "scale": [1.0] * 18,
            "coef": [0.01] * 18,
            "intercept": 0.001,
        },
    )
    write_json(
        folder / "metadata.json",
        {
            "format": "quantstream-notebook-v1",
            "type": "portable",
            "model": "run006_ridge",
            "hyperparameters": {"family": "ridge", "alpha": 10},
            "features": FEATURES,
            "threshold": 0.006,
            "config": cfg.to_dict(),
            "feature_code_sha256": sha256(
                Path(terminal.__file__).with_name("features.py")
            ),
            "created_at": "2026-01-01T00:00:00Z",
            "model_file": "model.json",
            "sha256": sha256(path),
            "training_rows": 100,
            "validation": None,
            "test": None,
            "test_is_independent": False,
        },
    )
    windows = np.ones((2, 48, 18)).tolist()
    write_json(
        folder / "parity.json", {"windows": windows, "predictions": [0.181, 0.181]}
    )
    seal_bundle(folder)
    return folder


def seal_bundle(folder):
    write_json(
        folder / "bundle_manifest.json",
        {
            "format": "quantstream-notebook-v1",
            "files": {
                p.name: sha256(p)
                for p in folder.iterdir()
                if p.name != "bundle_manifest.json"
            },
        },
    )


def test_portable_import_leaves_current_model_and_paper_intact(tmp_path):
    root = tmp_path / "live"
    cfg = terminal.config(root)
    write_json(
        root / "artifacts/champion.json", {"model": "current", "sha256": "current"}
    )
    write_json(
        root / "paper/state.json",
        {"model_sha256": "current", "assets": {"BTCUSDT": {"qty": 1}}},
    )
    bundle = ridge_bundle(tmp_path / "bundle", cfg)
    before = (root / "paper/state.json").read_bytes()
    meta = import_bundle(root, bundle)
    assert meta["type"] == "portable" and meta["import_optimizer_steps"] == 0
    assert meta["parity_max_abs_delta"] < 1e-6
    assert terminal.read_json(root / "artifacts/champion.json")["model"] == "current"
    assert (root / "paper/state.json").read_bytes() == before
    assert import_bundle(root, bundle)["imported_at"] == meta["imported_at"]
    with pytest.raises(ValueError, match="still open"):
        terminal.activate(root, meta["model"], archive_empty=True)
    assert (root / "paper/state.json").read_bytes() == before


def test_portable_prediction_mismatch_rejected_before_registration(tmp_path):
    root = tmp_path / "live"
    cfg = terminal.config(root)
    bundle = ridge_bundle(tmp_path / "bundle", cfg)
    parity = json.loads((bundle / "parity.json").read_text())
    parity["predictions"][0] += 1
    write_json(bundle / "parity.json", parity)
    seal_bundle(bundle)
    with pytest.raises(ValueError, match="parity failed"):
        import_bundle(root, bundle)
    assert not (root / "artifacts/local_registry.json").exists()


def test_import_rejects_contract_change(tmp_path):
    root = tmp_path / "live"
    cfg = terminal.config(root)
    bundle = ridge_bundle(tmp_path / "bundle", replace(cfg, horizon=8))
    with pytest.raises(ValueError, match="horizon"):
        import_bundle(root, bundle)
    assert not (root / "artifacts/local_registry.json").exists()


def test_imported_ridge_renders_all_app_routes(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_integrity import fixture_bars
    import pandas as pd
    from quantstream import ui_app, paper

    root = tmp_path / "live"
    cfg = terminal.config(root)
    bundle = ridge_bundle(tmp_path / "bundle", cfg)
    meta = import_bundle(root, bundle)
    bars = pd.concat(
        [fixture_bars(300, symbol=s) for s in cfg.symbols], ignore_index=True
    )
    path = root / "data/processed/bars.parquet"
    path.parent.mkdir(parents=True)
    bars.to_parquet(path, index=False)
    terminal.activate(root, meta["model"])
    terminal.predict_latest(root)
    paper.paper_step(
        root,
        cfg,
        refresh_data=False,
        price_provider=lambda s: 100,
        allow_new_entries=False,
    )
    monkeypatch.setattr(ui_app, "ROOT", root)
    app = AppTest.from_file(
        str(Path(ui_app.__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    for route in ["Thị trường", "Dự báo", "Kiểm định", "Danh mục", "Dữ liệu"]:
        app.radio[0].set_value(route).run()
        assert not app.exception, [e.message for e in app.exception]
