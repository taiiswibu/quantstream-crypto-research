"""Public market data only. This module cannot submit real exchange orders."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import write_json, utcnow
from .data import refresh, latest_price
from .features import build_features
from .research import load_metadata, predict_champion


def paper_step(root, cfg, now=None, refresh_data=True, price_provider=latest_price):
    root = Path(root)
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    meta, _ = load_metadata(root)
    if json.dumps(meta["config"], sort_keys=True) != json.dumps(cfg.to_dict(), sort_keys=True):
        raise ValueError("Paper config must match the selected model's frozen configuration.")
    path = root / "paper/state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["model_sha256"] != meta["sha256"]:
            raise ValueError("Paper state belongs to another model. Archive it with 'paper-reset' first.")
    else:
        state = {"created_at": utcnow(), "model_sha256": meta["sha256"], "initial_capital": 10000,
                 "assets": {s: {"cash": 10000 / len(cfg.symbols), "qty": 0.0, "peak": 10000 / len(cfg.symbols),
                                "halted": False, "last_signal": None, "pending": None} for s in cfg.symbols},
                 "events": [], "predictions": [], "equity_history": []}
    bars = refresh(root, cfg) if refresh_data else pd.read_parquet(root / "data/processed/bars.parquet")
    full_samples = build_features(bars, cfg)
    samples = full_samples.groupby("symbol", group_keys=False).tail(cfg.lookback)
    predictions = predict_champion(root, samples)
    samples = samples.assign(prediction=predictions)
    fee, slip = cfg.fee_bps / 10000, cfg.slippage_bps / 10000
    equity = 0.0
    for symbol in cfg.symbols:
        asset = state["assets"][symbol]
        price = float(price_provider(symbol))
        mark = asset["cash"] + asset["qty"] * price
        asset["peak"] = max(asset["peak"], mark)
        stop = mark / asset["peak"] - 1 <= -cfg.drawdown_halt
        due_exit = asset["qty"] and current >= pd.Timestamp(asset["exit_due"])
        if asset["qty"] and (due_exit or stop):
            fill = price * (1 - slip)
            proceeds = asset["qty"] * fill
            asset["cash"] += proceeds * (1 - fee)
            state["events"].append({"time": str(current), "symbol": symbol, "side": "SELL", "price": fill,
                                    "qty": asset["qty"], "fee": proceeds * fee,
                                    "reason": "drawdown_halt" if stop else "horizon",
                                    "execution_delay_seconds": max(0, (current - pd.Timestamp(asset["exit_due"])).total_seconds())})
            asset["qty"] = 0.0
        if stop:
            asset["halted"] = True
            asset["pending"] = None
        pending = asset["pending"]
        if pending is not None:
            due = pd.Timestamp(pending["due"])
            if current > due + pd.Timedelta(minutes=2):
                state["events"].append({"time": str(current), "symbol": symbol, "side": "SKIP", "reason": "missed_entry_window"})
                asset["pending"] = None
            elif due <= current and not asset["qty"] and not asset["halted"]:
                fill = price * (1 + slip)
                budget = asset["cash"] * cfg.position_fraction
                qty = budget / (fill * (1 + fee))
                asset["cash"] -= qty * fill * (1 + fee)
                asset["qty"] = qty
                asset["exit_due"] = (current + pd.Timedelta(hours=cfg.horizon)).isoformat()
                state["events"].append({"time": str(current), "symbol": symbol, "side": "BUY", "price": fill,
                                        "qty": qty, "fee": qty * fill * fee,
                                        "signal_time": pending["signal_time"]})
                asset["pending"] = None
        eligible = samples[(samples.symbol == symbol) & samples.decision_time.le(current)]
        if not eligible.empty:
            row = eligible.iloc[-1]
            decision = row.decision_time.isoformat()
            asset["last_prediction"] = float(row.prediction) if np.isfinite(row.prediction) else None
            asset["data_age_seconds"] = float((current - row.decision_time).total_seconds())
            if asset["last_signal"] != decision:
                asset["last_signal"] = decision
                fresh = current - row.decision_time < pd.Timedelta(hours=1, minutes=2)
                if fresh and np.isfinite(row.prediction):
                    state.setdefault("predictions", []).append({"symbol": symbol, "decision_time": decision,
                                                                "prediction": float(row.prediction)})
                if fresh and np.isfinite(row.prediction) and row.prediction > meta["threshold"] and not asset["qty"] and not asset["halted"]:
                    asset["pending"] = {"due": (row.decision_time + pd.Timedelta(hours=cfg.execution_lag_bars)).isoformat(),
                                        "signal_time": decision, "prediction": float(row.prediction)}
        asset["last_price"] = price
        asset["equity"] = asset["cash"] + asset["qty"] * price
        equity += asset["equity"]
    state["updated_at"] = str(current)
    state["equity"] = equity
    state["equity_history"].append({"time": str(current), "equity": equity})
    reference_path = root / "artifacts/drift_reference.json"
    if reference_path.exists():
        from .monitor import population_stability, prediction_quality
        recent_features = full_samples.groupby("symbol", group_keys=False).tail(168)
        state["monitoring"] = {"feature_drift_last_7_days": population_stability(recent_features, json.loads(reference_path.read_text())),
                               "prediction_quality": prediction_quality(full_samples, state.get("predictions", []), meta["model"]),
                               "note": "PSI >= 0.25 is a review heuristic, not an automatic trading or retraining rule."}
    write_json(path, state)
    return state
