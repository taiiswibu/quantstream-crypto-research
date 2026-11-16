from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from quantstream.common import Config, write_json
from quantstream.history_import import import_history
from quantstream import terminal
from test_integrity import fixture_bars


def bundle(bars, wrong_hash=False):
    payload = io.BytesIO()
    bars.to_parquet(payload, index=False)
    raw = payload.getvalue()
    manifest = {
        "format": "quantstream-history-v1",
        "synthetic": False,
        "bars_sha256": "bad" if wrong_hash else hashlib.sha256(raw).hexdigest(),
        "symbols": ["BTCUSDT"],
        "interval": "1h",
        "rows": len(bars),
    }
    result = io.BytesIO()
    with ZipFile(result, "w") as z:
        z.writestr("bars.parquet", raw)
        z.writestr("history_manifest.json", json.dumps(manifest))
    result.seek(0)
    return result


def setup(root, bars):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", replace(Config(), symbols=("BTCUSDT",)).to_dict())
    path = root / "data/processed/bars.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path, index=False)
    return path


def test_history_merge_retains_newer_data_and_models_and_is_idempotent(tmp_path):
    bars = fixture_bars(100)
    path = setup(tmp_path, bars.tail(30))
    registry = {"imported": {"sha256": "unchanged"}}
    write_json(tmp_path / "artifacts/local_registry.json", registry)
    write_json(tmp_path / "paper/state.json", {"assets": {"BTCUSDT": {"qty": 1}}})
    before = (tmp_path / "paper/state.json").read_bytes()
    result = import_history(tmp_path, bundle(bars.head(80)))
    assert result["added_rows"] == 70
    assert pd.read_parquet(path).timestamp.max() == bars.timestamp.max()
    assert import_history(tmp_path, bundle(bars.head(80)))["added_rows"] == 0
    assert (tmp_path / "paper/state.json").read_bytes() == before
    assert json.loads((tmp_path / "artifacts/local_registry.json").read_text()) == registry


@pytest.mark.parametrize("failure", ["hash", "conflict", "layout"])
def test_invalid_import_does_not_modify_candles(tmp_path, failure):
    bars = fixture_bars(100)
    path = setup(tmp_path, bars.tail(30))
    before = path.read_bytes()
    if failure == "conflict":
        bars.loc[75, "volume"] += 1
    archive = bundle(bars.head(80), wrong_hash=failure == "hash")
    if failure == "layout":
        archive = io.BytesIO()
        with ZipFile(archive, "w") as z:
            z.writestr("../bars.parquet", b"bad")
        archive.seek(0)
    with pytest.raises(ValueError):
        import_history(tmp_path, archive)
    assert path.read_bytes() == before


def test_earlier_history_increases_training_without_changing_recent_validation(tmp_path, monkeypatch):
    bars = fixture_bars(5000)
    original_build = terminal.build_features
    def indexed_build(data, cfg):
        result = original_build(data, cfg)
        result.index = result.decision_time.astype("int64")
        return result
    monkeypatch.setattr(terminal, "build_features", indexed_build)
    records = []
    class Estimator:
        def fit(self, features, target):
            self.last_train = int(features.index.max()) + 5 * 3600 * 10**9
            self.rows = len(features)
            return self
        def predict(self, features):
            assert self.last_train < int(features.index.min())
            records.append((self.rows, tuple(features.index)))
            return np.zeros(len(features))
    monkeypatch.setattr(terminal, "make_estimator", lambda name, cfg: Estimator())
    monkeypatch.setattr(terminal.joblib, "dump", lambda model, path: Path(path).write_bytes(b"test-model"))
    for name, data in [("short", bars.tail(2160)), ("full", bars)]:
        root = tmp_path / name
        setup(root, data)
        write_json(root / "artifacts/local_registry.json", {"old_import": {"type": "portable", "marker": "keep"}})
        models = terminal.fit_local(root)
        kept = json.loads((root / "artifacts/local_registry.json").read_text())
        assert kept["old_import"]["marker"] == "keep"
        assert all(not m["validation_is_independent_final_test"] for m in models.values())
        assert all(len(m["validation_fold_mse_ratios"]) == 3 for m in models.values())
        assert models["ridge"]["validation_window_days"] == 90
    short, full = records[:6], records[6:]
    assert len(short) == len(full) == 6
    assert [keys for count, keys in short] == [keys for count, keys in full]
    assert all(b[0] > a[0] for a, b in zip(short, full))
