"""Synthetic test fixtures verify causality/accounting; they are never training data."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from quantstream.common import Config
from quantstream.data import timestamp_series, validate_bars
from quantstream.features import build_features, FEATURES, temporal_plan, fold_data
from quantstream.backtest import simulate


def fixture_bars(n=1800, symbol="BTCUSDT", flat=False):
    t = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.full(n, 100.0) if flat else 100 * np.exp(np.arange(n) * 0.00005 + 0.005 * np.sin(np.arange(n) / 13))
    return pd.DataFrame({"symbol": symbol, "timestamp": t, "source_close_time": t + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                         "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 10 + np.arange(n) % 7, "quote_volume": 1000,
                         "trades": 10, "taker_buy_volume": 5})


def test_mixed_millisecond_microsecond_timestamps():
    result = timestamp_series(pd.Series([1735689600000, 1735689600000000]))
    assert result.iloc[0] == result.iloc[1] == pd.Timestamp("2025-01-01", tz="UTC")


def test_quality_quarantine_dedup_conflict():
    bars = fixture_bars(5)
    bars.loc[1, "high"] = 1
    combined = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    clean, bad, report = validate_bars(combined)
    assert len(bad) == 1 and report["duplicate_rows"] == 1 and len(clean) == 4
    assert report["input_rows"] == report["valid_rows"] + report["quarantine_rows"] + report["duplicate_rows"]
    conflict = fixture_bars(5).iloc[[0]].copy()
    conflict["volume"] = 99
    with pytest.raises(ValueError, match="Conflicting"):
        validate_bars(pd.concat([fixture_bars(5), conflict], ignore_index=True))


def test_features_do_not_change_when_future_changes():
    cfg = Config()
    bars = fixture_bars()
    a = build_features(bars, cfg)
    modified = bars.copy()
    modified.loc[1200:, ["open", "high", "low", "close", "volume"]] *= 2
    b = build_features(modified, cfg)
    pd.testing.assert_frame_equal(a[a.timestamp < bars.iloc[1200].timestamp][FEATURES],
                                  b[b.timestamp < bars.iloc[1200].timestamp][FEATURES])


def test_label_matches_delayed_execution_prices_and_purged_split():
    cfg = Config()
    bars = fixture_bars()
    samples = build_features(bars, cfg)
    row = samples.iloc[100]
    i = bars.index[bars.timestamp.eq(row.timestamp)][0]
    expected = bars.iloc[i + cfg.horizon + cfg.execution_lag_bars].close / bars.iloc[i + 1 + cfg.execution_lag_bars].open - 1
    assert row.target_return == pytest.approx(expected)
    plan = temporal_plan(samples, cfg)
    for bounds in plan["folds"]:
        train, val = fold_data(samples, bounds["validation_start"], bounds["validation_end"])
        assert train.label_end_time.max() < val.decision_time.min()
        assert val.label_end_time.max() < pd.Timestamp(bounds["validation_end"])


def test_flat_market_loses_exact_roundtrip_costs():
    cfg = replace(Config(), symbols=("BTCUSDT",), position_fraction=1.0, drawdown_halt=0.99)
    bars = fixture_bars(12, flat=True)
    signals = pd.DataFrame({"symbol": ["BTCUSDT"], "decision_time": [bars.timestamp.iloc[0]], "go_long": [True]})
    curve, trades, summary = simulate(bars, signals, bars.timestamp.iloc[0], bars.timestamp.iloc[-1] + pd.Timedelta(hours=1), cfg)
    fee, slip = cfg.fee_bps / 1e4, cfg.slippage_bps / 1e4
    expected = ((1 - slip) * (1 - fee)) / ((1 + slip) * (1 + fee))
    assert curve.iloc[-1] == pytest.approx(expected)
    assert len(trades) == 1
    assert pd.Timestamp(trades.iloc[0].entry_time) == bars.timestamp.iloc[1]
    assert pd.Timestamp(trades.iloc[0].exit_time) == bars.timestamp.iloc[5]


def test_non_overlapping_positions_and_idle_cash():
    cfg = replace(Config(), symbols=("BTCUSDT",), fee_bps=0, slippage_bps=0)
    bars = fixture_bars(24, flat=True)
    signals = pd.DataFrame({"symbol": "BTCUSDT", "decision_time": bars.timestamp, "go_long": True})
    curve, trades, _ = simulate(bars, signals, bars.timestamp.iloc[0], bars.timestamp.iloc[-1] + pd.Timedelta(hours=1), cfg)
    assert (curve == 1).all()
    for previous, following in zip(trades.itertuples(), list(trades.itertuples())[1:]):
        assert pd.Timestamp(following.entry_time) >= pd.Timestamp(previous.exit_time)


def test_gap_resets_features_and_prevents_cross_gap_targets():
    bars = fixture_bars().drop(index=[1000]).reset_index(drop=True)
    out = build_features(bars, Config())
    around_gap = out[(out.timestamp > pd.Timestamp("2024-02-11 16:00", tz="UTC")) & (out.timestamp < pd.Timestamp("2024-02-12 16:00", tz="UTC"))]
    assert around_gap.empty


def test_tcn_block_is_causal():
    torch = pytest.importorskip("torch")
    from quantstream.deep import CausalBlock
    torch.manual_seed(1)
    block = CausalBlock(2, 3, 2).eval()
    a = torch.randn(1, 2, 30)
    b = a.clone()
    b[:, :, 20:] += 100
    assert torch.allclose(block(a)[:, :, :20], block(b)[:, :, :20])
