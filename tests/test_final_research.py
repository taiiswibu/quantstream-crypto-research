"""Final protocol invariants; fixture tests never assert market performance."""

from pathlib import Path
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantstream.forecast_contract import target_scale, allows_paper_entries
from quantstream import final_research, workflow, terminal
from quantstream.common import Config, write_json, write_parquet
from quantstream.features import build_features
from test_integrity import fixture_bars
from test_workflow import synthetic_dataset


def test_target_scale_is_causal_and_forecast_inverse_matches_train_units():
    frame = pd.DataFrame({"vol_48": [0.001, 0.01, 0.02]})
    params = {"target_transform": "volatility", "forecast_horizon": 4}
    scale = target_scale(frame, params)
    np.testing.assert_allclose(scale, [0.2, 2.0, 4.0])
    y = np.array([0.01, -0.02, 0.03])
    np.testing.assert_allclose(y / scale * scale, y)
    assert allows_paper_entries({"signal_approved": False}) is False
    assert (
        allows_paper_entries(
            {"validation_gate": {"beats_zero_on_mean_and_every_fold": False}}
        )
        is False
    )


def test_week_clusters_preserve_datetime_resolution_and_positive_loss_gap():
    dates = pd.date_range("2023-01-01", periods=24 * 90, freq="h", tz="UTC")
    p = pd.DataFrame(
        {"decision_time": dates, "fold": 0, "target_return": 0.01, "prediction": -0.01}
    )
    ns = final_research.weekly_interval(p, repeats=100)
    p["decision_time"] = p.decision_time.dt.as_unit("us")
    us = final_research.weekly_interval(p, repeats=100)
    assert ns == us and ns["lower"] > 0


@pytest.mark.parametrize("family", ["ridge", "tcn"])
def test_normalized_model_export_cpu_parity_and_actual_return_units(tmp_path, family):
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    from quantstream.tcn_import import import_bundle

    cfg = replace(Config(), symbols=("BTCUSDT",), threads=1)
    data = build_features(fixture_bars(700), cfg).dropna(subset=["target_return"])
    params = final_research.conditions(1, 1)[1 if family == "ridge" else 5]
    fitted = workflow.fit_model(
        data, params, 42, 1, cfg, "cpu", workflow.Budget(30), tmp_path / "final-fit"
    )
    manifest = {
        "feature_code_sha256": __import__(
            "quantstream.common", fromlist=["sha256"]
        ).sha256(Path(workflow.__file__).with_name("features.py")),
        "files": {"data/processed/features.parquet": "fixture"},
    }
    selection = {"validation": None, "signal_approved": False}
    folder = final_research._export(
        tmp_path, fitted, cfg, manifest, selection, None, data, "fixture"
    )
    live = tmp_path / "live"
    write_json(live / "config.json", cfg.to_dict())
    meta = import_bundle(live, folder)
    assert meta["parity_max_abs_delta"] < 1e-5 and meta["signal_approved"] is False
    terminal.activate(live, meta["model"])
    write_parquet(live / "data/processed/bars.parquet", fixture_bars(700))
    values = terminal.predict_latest(live)
    assert len(values) == 1 and np.isfinite(values[0]["prediction"])
    if family == "ridge":
        # Zero-intercept normalized Ridge contributions must sum in original return units.
        assert terminal.feature_contributions(
            live, "BTCUSDT"
        ).value.sum() == pytest.approx(values[0]["prediction"])


def test_final_full_pipeline_exports_distinct_audit_and_latest_refit(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    from quantstream.extended_training import prepare_dataset
    from quantstream.tcn_import import import_bundle
    from streamlit.testing.v1 import AppTest
    from quantstream import ui_app, paper

    parent, root = tmp_path / "parent", tmp_path / "final"
    cfg = synthetic_dataset(parent)
    prepare_dataset(root, parent)
    final_research.run(root, epochs=1, minimum_epochs=1, seconds=240, device="cpu")
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed" and status["conditions_completed"] == 8
    serving = json.loads((root / "deployment/metadata.json").read_text())
    audit = json.loads((root / "audit_model/deployment/metadata.json").read_text())
    assert serving["test"] is None and audit["test"] is not None
    assert pd.Timestamp(serving["training_last_label"]) > pd.Timestamp(
        audit["training_last_label"]
    )
    zero = json.loads((root / "zero_reference/deployment/metadata.json").read_text())
    assert zero["training_rows"] == 0 and zero["signal_approved"] is False
    assert zero["validation"]["mse_ratio_to_zero"] == 1.0
    live = tmp_path / "live"
    write_json(live / "config.json", cfg.to_dict())
    meta = import_bundle(live, root / "deployment")
    terminal.activate(live, meta["model"])
    write_parquet(live / "data/processed/bars.parquet", fixture_bars(1800))
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
    for route in ["Thị trường", "Dự báo", "Kiểm định", "Danh mục", "Dữ liệu"]:
        app.radio[0].set_value(route).run()
        assert not app.exception, [e.message for e in app.exception]
    baseline = import_bundle(live, root / "zero_reference/deployment")
    assert baseline["hyperparameters"]["family"] == "zero_return"
    terminal.activate(live, baseline["model"], archive_empty=True)
    terminal.predict_latest(live)
    for route in ["Thị trường", "Dự báo", "Kiểm định", "Danh mục", "Dữ liệu"]:
        app.radio[0].set_value(route).run()
        assert not app.exception, [e.message for e in app.exception]


def test_failed_acceptance_cancels_pending_entry_but_preserves_exit(
    tmp_path, monkeypatch
):
    from test_paper_monitor import setup_paper
    from quantstream import paper

    cfg, now = setup_paper(tmp_path, monkeypatch)
    state = paper.paper_step(
        tmp_path, cfg, now=now, refresh_data=False, price_provider=lambda _: 100
    )
    due = pd.Timestamp(state["assets"]["BTCUSDT"]["pending"]["due"])
    meta, _ = paper.load_metadata(tmp_path)
    meta["signal_approved"] = False
    state = paper.paper_step(
        tmp_path, cfg, now=due, refresh_data=False, price_provider=lambda _: 100
    )
    assert state["assets"]["BTCUSDT"]["pending"] is None
    assert not any(e["side"] == "BUY" for e in state["events"])
    # Already open positions can still close after validation has failed.
    asset = state["assets"]["BTCUSDT"]
    asset.update(qty=1.0, exit_due=str(due))
    write_json(tmp_path / "paper/state.json", state)
    state = paper.paper_step(
        tmp_path, cfg, now=due, refresh_data=False, price_provider=lambda _: 100
    )
    assert state["assets"]["BTCUSDT"]["qty"] == 0
    assert state["events"][-1]["side"] == "SELL"
