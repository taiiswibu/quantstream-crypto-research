from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from quantstream.common import Config
from quantstream.features import FEATURES
from quantstream.monitor import drift_reference, population_stability
from quantstream import paper
from test_integrity import fixture_bars


def test_psi_identical_and_shifted_populations():
    data = pd.DataFrame({name: np.linspace(-1, 1, 200) for name in FEATURES})
    reference = drift_reference(data)
    assert max(r["psi"] for r in population_stability(data, reference)) == pytest.approx(0)
    assert min(r["psi"] for r in population_stability(data + 100, reference)) > 0.25


def setup_paper(tmp_path, monkeypatch):
    cfg = replace(Config(), symbols=("BTCUSDT",))
    bars = fixture_bars(300)
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    bars.to_parquet(folder / "bars.parquet", index=False)
    meta = {"config": cfg.to_dict(), "sha256": "fixed", "model": "ridge", "threshold": 0.006}
    monkeypatch.setattr(paper, "load_metadata", lambda root: (meta, None))
    monkeypatch.setattr(paper, "predict_champion", lambda root, frame: np.full(len(frame), 0.01))
    now = bars.timestamp.iloc[-1] + pd.Timedelta(hours=1, minutes=30)
    return cfg, now


def test_paper_repeat_does_not_duplicate_orders(tmp_path, monkeypatch):
    cfg, now = setup_paper(tmp_path, monkeypatch)
    one = paper.paper_step(tmp_path, cfg, now=now, refresh_data=False, price_provider=lambda s: 100)
    due = pd.Timestamp(one["assets"]["BTCUSDT"]["pending"]["due"])
    two = paper.paper_step(tmp_path, cfg, now=due + pd.Timedelta(seconds=30), refresh_data=False, price_provider=lambda s: 100)
    three = paper.paper_step(tmp_path, cfg, now=due + pd.Timedelta(seconds=40), refresh_data=False, price_provider=lambda s: 100)
    assert len([e for e in two["events"] if e["side"] == "BUY"]) == 1
    assert len([e for e in three["events"] if e["side"] == "BUY"]) == 1


def test_paper_skips_missed_entry(tmp_path, monkeypatch):
    cfg, now = setup_paper(tmp_path, monkeypatch)
    one = paper.paper_step(tmp_path, cfg, now=now, refresh_data=False, price_provider=lambda s: 100)
    due = pd.Timestamp(one["assets"]["BTCUSDT"]["pending"]["due"])
    two = paper.paper_step(tmp_path, cfg, now=due + pd.Timedelta(minutes=3), refresh_data=False, price_provider=lambda s: 100)
    assert two["assets"]["BTCUSDT"]["qty"] == 0
    assert two["events"][-1]["reason"] == "missed_entry_window"


def test_final_holdout_refuses_different_model(tmp_path, monkeypatch):
    import json
    from quantstream import research
    cfg = Config()
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "latest_run.json").write_text(json.dumps({"run": "runs/test"}))
    (folder / "final_test_lock.json").write_text(json.dumps({"dataset": "dataset1", "model": "model1"}))
    meta = {"config": cfg.to_dict(), "features_sha256": "dataset1", "sha256": "model2"}
    monkeypatch.setattr(research, "load_metadata", lambda root: (meta, None))
    with pytest.raises(ValueError, match="already opened"):
        research.final_evaluate(tmp_path, cfg)
