"""Regression coverage for configuration and split drift before Kaggle export."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from quantstream.common import Config
from quantstream.contracts import verify_feature_snapshot
from quantstream.features import prepare
from quantstream.research import export_kaggle
from test_integrity import fixture_bars


@pytest.fixture
def prepared(tmp_path):
    cfg = replace(Config(), symbols=("BTCUSDT",))
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    fixture_bars().to_parquet(folder / "bars.parquet", index=False)
    prepare(tmp_path, cfg)
    return tmp_path, cfg


def test_prepared_snapshot_passes(prepared):
    root, cfg = prepared
    manifest, plan = verify_feature_snapshot(root, cfg)
    assert plan == manifest["split"]


def test_export_rejects_changed_horizon_before_writing(prepared):
    root, cfg = prepared
    with pytest.raises(ValueError, match="Config changed"):
        export_kaggle(root, replace(cfg, horizon=8))
    assert not (root / "kaggle_export").exists()


def test_training_data_tampering_is_detected(prepared):
    root, cfg = prepared
    path = root / "data/processed/features.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "ret_1"] = 999
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="feature data changed"):
        verify_feature_snapshot(root, cfg)


def test_changed_split_is_detected(prepared):
    root, cfg = prepared
    path = root / "data/processed/split_plan.json"
    plan = json.loads(path.read_text())
    plan["test_start"] = "2026-01-01T00:00:00+00:00"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Temporal split changed"):
        verify_feature_snapshot(root, cfg)
