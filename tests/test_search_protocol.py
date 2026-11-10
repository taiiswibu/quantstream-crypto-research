"""Temporal leakage and evidence-preservation checks for budgeted research."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from quantstream import search  # noqa: E402
from quantstream.common import Config, sha256  # noqa: E402
from quantstream.features import FEATURES  # noqa: E402


@pytest.fixture
def panel():
    times = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    frames = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frame = pd.DataFrame({"symbol": symbol, "decision_time": times,
                              "label_end_time": times + pd.Timedelta(hours=5),
                              "target_return": np.sin(np.arange(len(times))) / 100,
                              "segment": 1})
        for i, name in enumerate(FEATURES):
            frame[name] = np.cos(np.arange(len(times)) / (i + 1))
        frames.append(frame)
    return pd.concat(frames).sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def test_inner_purge_uses_shared_asset_cut(panel):
    fit, stop, cut = search.inner_split(panel)
    assert fit.label_end_time.max() < cut
    assert stop.decision_time.min() == cut
    assert set(fit.symbol) == set(stop.symbol) == {"BTCUSDT", "ETHUSDT"}
    assert not fit.decision_time.isin(stop.decision_time).any()


def test_context_hides_past_labels_and_restarts_after_gap(panel):
    # A missing observation must invalidate sequence coverage for the next 47h.
    missing = panel.decision_time.unique()[310]
    gapped = panel[~((panel.symbol == "BTCUSDT") & (panel.decision_time == missing))]
    cut = panel.decision_time.unique()[300]
    val = gapped[gapped.decision_time >= cut]
    context = search.with_context(gapped, val, cut)
    assert context.loc[~context.is_validation, "target_return"].isna().all()
    affected = context[(context.symbol == "BTCUSDT") & (context.decision_time > missing)
                       & (context.decision_time < missing + pd.Timedelta(hours=48))]
    assert not affected.eligible_window.any()
    assert context[(context.symbol == "ETHUSDT") & context.is_validation].eligible_window.all()


def test_outer_labels_never_select_neural_epoch(panel, monkeypatch):
    start = panel.decision_time.unique()[300]
    bounds = {"validation_start": str(start), "validation_end": str(panel.decision_time.max())}
    calls = []

    def fake_fit(train, params, seed, epochs, device, threads, validation=None):
        calls.append((train.copy(), epochs, validation))
        if validation is not None:
            assert validation.decision_time.max() < start
            assert train.label_end_time.max() < validation.loc[validation.is_validation, "decision_time"].min()
        return object(), np.zeros(len(FEATURES)), np.ones(len(FEATURES)), 2, []

    monkeypatch.setattr(search, "neural_fit", fake_fit)
    monkeypatch.setattr(search, "predict_tcn", lambda model, frame, *args: np.zeros(len(frame)))
    part, trace = search.fit_predict(panel, bounds, {"family": "tcn", "lookback": 48},
                                      42, 8, Config(), "cpu")
    assert len(calls) == 2
    assert calls[1][1] == 2 and calls[1][2] is None
    assert calls[1][0].label_end_time.max() < start
    assert trace["best_epoch"] == 2 and part.decision_time.min() == start


def test_sequence_forward_is_causal_and_dropout_eval_is_stable():
    torch.manual_seed(2)
    for family in ("tcn", "gru"):
        model = search.SequenceRegressor({"family": family, "width": 16, "dropout": 0.3}).eval()
        x = torch.randn(2, 48, len(FEATURES))
        assert model(x).shape == (2,)
        assert torch.equal(model(x), model(x))
        # A prefix forward does not require or ingest the future tail.
        before = model(x[:, :24]).detach()
        x[:, 24:] = 1000
        assert torch.equal(model(x[:, :24]), before)


def test_snapshot_rejects_holdout_even_with_matching_file_hashes(panel, tmp_path):
    end = panel.decision_time.unique()[300]
    files = {}
    for name in ("development", "train", "validation"):
        path = tmp_path / f"{name}.parquet"
        panel.to_parquet(path, index=False)
        files[path.name] = sha256(path)
    bars = tmp_path / "bars.parquet"
    pd.DataFrame({"timestamp": panel.decision_time}).to_parquet(bars)
    files[bars.name] = sha256(bars)
    manifest = {"contains_final_holdout": False, "feature_columns": FEATURES,
                "files": files, "split": {"test_start": str(end)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holdout leaked"):
        search.load_snapshot(tmp_path)


def test_completed_diagnostic_cannot_be_silently_retried(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    manifest = {"source_control": {"commit": "test"}}
    monkeypatch.setattr(search, "load_snapshot", lambda p: (manifest, None, None, Config()))
    (tmp_path / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    (output / "diagnostic.started.json").write_text("{}")
    with pytest.raises(ValueError, match="already started"):
        search.run_search(tmp_path, output, n_trials=4, max_epochs=1, device="cpu")
