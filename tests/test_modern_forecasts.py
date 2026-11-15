from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from quantstream.common import Config
from quantstream.features import build_features
from quantstream.modern_models import prepare_window
from quantstream.risk_reference import (
    empirical_interval,
    save_forward_interval,
    forward_interval_scores,
)


def samples():
    from test_integrity import fixture_bars

    btc = fixture_bars(1800)
    eth = btc.copy().assign(symbol="ETHUSDT")
    eth[["open", "high", "low", "close"]] *= 0.05
    return build_features(pd.concat([btc, eth], ignore_index=True), Config())


def test_model_window_cannot_read_future_or_unmatured_labels():
    frame = samples()
    origin = frame.decision_time.max() - pd.Timedelta(hours=20)
    before = prepare_window(frame, origin, ["ret_1", "vol_48"])
    changed = frame.copy()
    changed.loc[changed.label_end_time.ge(origin), "target_return"] = 99999.0
    changed.loc[changed.decision_time.gt(origin), ["ret_1", "vol_48"]] = 99999.0
    after = prepare_window(changed, origin, ["ret_1", "vol_48"])
    np.testing.assert_array_equal(before["target"], after["target"])
    np.testing.assert_array_equal(
        before["future_covariates"], after["future_covariates"]
    )
    assert before["target"].shape == (2, 128)
    assert before["future_covariates"].shape == (4, 134)
    assert pd.Timestamp(before["last_target_origin"]) + pd.Timedelta(hours=5) < origin


def test_interval_ignores_unmatured_targets_and_rejects_wrong_horizon():
    frame = samples()
    origin = frame.decision_time.max()
    before = empirical_interval(frame, "BTCUSDT")
    changed = frame.copy()
    changed.loc[changed.label_end_time.ge(origin), "target_return"] = 99999.0
    after = empirical_interval(changed, "BTCUSDT")
    assert before == after
    assert before["q10"] <= before["q50"] <= before["q90"]
    assert pd.Timestamp(before["latest_label_used"]) < origin
    with pytest.raises(ValueError, match="Cần thêm"):
        empirical_interval(frame.groupby("symbol").tail(20), "BTCUSDT")
    changed["label_end_time"] = changed.decision_time + pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="4 giờ"):
        empirical_interval(changed, "BTCUSDT")


def test_only_prospective_intervals_are_logged_and_scored_when_mature(tmp_path):
    frame = samples()
    origin = frame.decision_time.max() - pd.Timedelta(hours=20)
    historical = frame[frame.decision_time.le(origin)]
    record = empirical_interval(historical, "BTCUSDT")
    assert not save_forward_interval(
        tmp_path, record, now=origin + pd.Timedelta(days=1)
    )
    assert save_forward_interval(tmp_path, record, now=origin + pd.Timedelta(minutes=3))
    assert not save_forward_interval(
        tmp_path, record, now=origin + pd.Timedelta(minutes=4)
    )
    assert forward_interval_scores(
        tmp_path, frame, now=origin + pd.Timedelta(hours=4)
    ).empty
    scored = forward_interval_scores(
        tmp_path, frame, now=origin + pd.Timedelta(hours=6)
    )
    assert scored.rows.sum() == 1
    with sqlite3.connect(Path(tmp_path) / "terminal.sqlite") as db:
        assert db.execute("SELECT count(*) FROM interval_forecasts").fetchone()[0] == 1
