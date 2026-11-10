"""Causal features shared by research, Kaggle and live inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import Config, write_json, write_parquet, sha256, utcnow
from pathlib import Path

FEATURES = ["ret_1", "ret_4", "ret_12", "ret_24", "vol_12", "vol_48",
            "rsi_14", "macd", "atr_14", "volume_z", "range_pct", "body_pct",
            "taker_ratio", "ema_gap", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def build_features(bars, cfg: Config):
    pieces = []
    for symbol, data in bars.groupby("symbol", sort=True):
        data = data.sort_values("timestamp").reset_index(drop=True).copy()
        # Restart indicators at gaps: a rolling window must mean consecutive hourly bars.
        data["segment"] = data.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _, g in data.groupby("segment", sort=True):
            g = g.copy()
            close, previous = g.close, g.close.shift(1)
            logp = np.log(close)
            for lag in (1, 4, 12, 24):
                g[f"ret_{lag}"] = logp.diff(lag)
            for window in (12, 48):
                g[f"vol_{window}"] = g.ret_1.rolling(window).std()
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            rs = gain / loss.replace(0, np.nan)
            g["rsi_14"] = (100 - 100 / (1 + rs)).where(loss.ne(0), 100).where((gain + loss).ne(0), 50) / 100
            fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
            slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
            g["macd"] = (fast - slow) / close
            tr = pd.concat([g.high - g.low, (g.high - previous).abs(), (g.low - previous).abs()], axis=1).max(axis=1)
            g["atr_14"] = tr.rolling(14).mean() / close
            logv = np.log1p(g.volume)
            g["volume_z"] = (logv - logv.rolling(48).mean()) / logv.rolling(48).std().replace(0, 1)
            g["range_pct"] = (g.high - g.low) / close
            g["body_pct"] = (close - g.open) / g.open
            g["taker_ratio"] = g.taker_buy_volume / g.volume.replace(0, np.nan)
            g["taker_ratio"] = g.taker_ratio.fillna(0.5)
            g["ema_gap"] = (close.ewm(span=12, adjust=False).mean() - close.ewm(span=48, adjust=False).mean()) / close
            # The candle becomes observable only at its close, one hour after open_time.
            g["decision_time"] = g.timestamp + pd.Timedelta(hours=1)
            hour, dow = g.decision_time.dt.hour, g.decision_time.dt.dayofweek
            g["hour_sin"], g["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
            g["dow_sin"], g["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
            # Conservative one FULL hourly-bar delay for execution. No same-boundary fill.
            entry_shift = 1 + cfg.execution_lag_bars
            exit_shift = cfg.horizon + cfg.execution_lag_bars
            g["entry_time"] = g.timestamp.shift(-entry_shift)
            g["label_end_time"] = g.timestamp.shift(-exit_shift) + pd.Timedelta(hours=1)
            g["target_return"] = g.close.shift(-exit_shift) / g.open.shift(-entry_shift) - 1
            expected = g.decision_time + pd.Timedelta(hours=cfg.horizon + cfg.execution_lag_bars)
            g.loc[g.label_end_time.ne(expected), "target_return"] = np.nan
            pieces.append(g)
    frame = pd.concat(pieces, ignore_index=True)
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    # NaN targets at the right edge remain usable for live inference.
    return frame.dropna(subset=FEATURES).sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def temporal_plan(samples, cfg):
    eligible = samples.dropna(subset=["target_return", "label_end_time"])
    times = np.sort(eligible.decision_time.unique())
    if len(times) < 1200:
        raise ValueError("At least 1200 hourly observations required for temporal evaluation")
    test_index = int(len(times) * (1 - cfg.test_fraction))
    first_validation = int(test_index * 0.6)
    edges = np.linspace(first_validation, test_index, cfg.walk_forward_folds + 1, dtype=int)
    return {"test_start": pd.Timestamp(times[test_index]).isoformat(),
            "end": (pd.Timestamp(times[-1]) + pd.Timedelta(hours=cfg.horizon + cfg.execution_lag_bars)).isoformat(),
            "folds": [{"validation_start": pd.Timestamp(times[a]).isoformat(),
                       "validation_end": pd.Timestamp(times[b]).isoformat()}
                      for a, b in zip(edges[:-1], edges[1:])],
            "rule": "train.label_end_time < validation_start; validation.label_end_time < validation_end",
            "multi_asset_split": "all symbols share UTC boundaries"}


def fold_data(samples, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = samples[(samples.label_end_time < start) & samples.target_return.notna()].copy()
    val = samples[(samples.decision_time >= start) & (samples.label_end_time < end) & samples.target_return.notna()].copy()
    if train.empty or val.empty:
        raise ValueError("Empty temporal split")
    assert train.label_end_time.max() < val.decision_time.min()
    return train, val


def prepare(root, cfg):
    root = Path(root)
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    samples = build_features(bars, cfg)
    plan = temporal_plan(samples, cfg)
    path = root / "data/processed/features.parquet"
    write_parquet(path, samples)
    write_json(root / "data/processed/split_plan.json", plan)
    write_json(root / "data/processed/feature_manifest.json", {
        "created_at": utcnow(), "bars_sha256": sha256(root / "data/processed/bars.parquet"),
        "features_sha256": sha256(path), "feature_code_sha256": sha256(Path(__file__)),
        "feature_columns": FEATURES, "rows": len(samples), "config": cfg.to_dict(), "split": plan})
    return samples, plan
