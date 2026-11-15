"""Four-hour return intervals from strictly matured observations."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


def empirical_interval(frame, symbol, days=28, minimum=168):
    group = frame[frame.symbol.eq(symbol)].sort_values("decision_time")
    if group.empty:
        raise ValueError("Chưa có dữ liệu cho " + symbol)
    latest = group.iloc[-1]
    origin = latest.decision_time
    history = group[
        group.label_end_time.lt(origin)
        & group.decision_time.ge(origin - pd.Timedelta(days=days))
        & group.target_return.notna()
    ]
    if len(history) < minimum:
        raise ValueError("Cần thêm nhãn đã đủ thời gian để tính khoảng tham chiếu")
    if (
        not (history.label_end_time - history.decision_time)
        .eq(pd.Timedelta(hours=5))
        .all()
    ):
        raise ValueError("Khoảng tham chiếu này dùng return 4 giờ, chờ vào 1 giờ")
    scales = np.clip(history.vol_48.to_numpy() * 2, 0.001, 0.05)
    normalized = history.target_return.to_numpy() / scales
    scale_now = float(np.clip(latest.vol_48 * 2, 0.001, 0.05))
    quantiles = np.quantile(normalized, [0.1, 0.5, 0.9]) * scale_now
    if not np.isfinite(quantiles).all():
        raise ValueError("Khoảng tham chiếu không hợp lệ")
    return {
        "symbol": symbol,
        "decision_time": origin.isoformat(),
        "entry_time": (origin + pd.Timedelta(hours=1)).isoformat(),
        "horizon_end": (origin + pd.Timedelta(hours=5)).isoformat(),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
        "source": "rolling_empirical",
        "history_rows": len(history),
        "latest_label_used": history.label_end_time.max().isoformat(),
        "nominal_coverage": 0.8,
        "coverage_is_guaranteed": False,
        "parameters": {"days": days, "minimum": minimum, "volatility_hours": 48},
    }


def save_forward_interval(root, record, now=None):
    """Never turn archived predictions into prospective evidence."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    origin = pd.Timestamp(record["decision_time"])
    if not 0 <= (now - origin).total_seconds() <= 3720:
        return False
    identity = {
        "method": record["source"],
        "parameters": record.get("parameters"),
        "weights_sha256": record.get("weights_sha256"),
        "forecaster_code_sha256": record.get("forecaster_code_sha256"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        **record,
        "forecast_signature": signature,
        "recorded_at": now.isoformat(),
    }
    with sqlite3.connect(root / "terminal.sqlite", timeout=10) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS interval_forecasts "
            "(signature TEXT,symbol TEXT,decision_time TEXT,payload TEXT,"
            "PRIMARY KEY(signature,symbol,decision_time))"
        )
        cursor = db.execute(
            "INSERT OR IGNORE INTO interval_forecasts VALUES (?,?,?,?)",
            (signature, record["symbol"], record["decision_time"], json.dumps(payload)),
        )
        return cursor.rowcount == 1


def forward_interval_scores(root, frame, now=None):
    path = Path(root) / "terminal.sqlite"
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='interval_forecasts'"
        ).fetchone():
            return pd.DataFrame()
        records = [
            json.loads(r[0])
            for r in db.execute("SELECT payload FROM interval_forecasts")
        ]
    if not records:
        return pd.DataFrame()
    logged = pd.DataFrame(records)
    logged["decision_time"] = pd.to_datetime(logged.decision_time, utc=True)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    matured = frame[frame.label_end_time.le(now)]
    joined = logged.merge(
        matured[["symbol", "decision_time", "target_return"]],
        on=["symbol", "decision_time"],
        how="left",
        validate="many_to_one",
    ).dropna(subset=["target_return"])
    if joined.empty:
        return pd.DataFrame()
    joined["covered"] = (
        (joined.target_return >= joined.q10) & (joined.target_return <= joined.q90)
    ).astype(float)
    joined["width"] = joined.q90 - joined.q10
    return joined.groupby(["source", "symbol"], as_index=False).agg(
        rows=("covered", "size"), coverage=("covered", "mean"), width=("width", "mean")
    )
