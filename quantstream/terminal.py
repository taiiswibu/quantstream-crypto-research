"""Local market data, small forecasting models and paper-only operation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request

from filelock import FileLock
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    Config,
    git_provenance,
    sha256,
    stamp,
    utcnow,
    write_json,
    write_parquet,
)
from .data import REST, normalise, validate_bars
from .features import FEATURES, build_features
from .research import load_metadata, predict_champion

PROJECT = Path(__file__).resolve().parents[1]


def config(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if not path.exists():
        cfg = replace(Config(), position_fraction=0.10, drawdown_halt=0.10)
        write_json(path, cfg.to_dict())
    return Config.load(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def request_json(endpoint, **params):
    query = urllib.parse.urlencode(params)
    url = REST + endpoint + ("?" + query if query else "")
    req = urllib.request.Request(
        url, headers={"User-Agent": "QuantStream-Terminal/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Market data HTTP {error.code}; retry later.") from error


def fetch_symbol(symbol, start, now):
    cursor, pieces = pd.Timestamp(start), []
    for _ in range(8):
        raw = request_json(
            "/klines",
            symbol=symbol,
            interval="1h",
            limit=1000,
            startTime=int(cursor.timestamp() * 1000),
            endTime=int(now.timestamp() * 1000),
        )
        if not isinstance(raw, list):
            raise ValueError("Unexpected candles response")
        if not raw:
            break
        batch = normalise(raw, symbol)
        pieces.append(batch)
        next_cursor = batch.timestamp.max() + pd.Timedelta(hours=1)
        if next_cursor <= cursor:
            raise ValueError("Market-data pagination did not advance")
        cursor = next_cursor
        if len(batch) < 1000 or cursor >= now:
            break
    else:
        raise ValueError("Recent-history request exceeded its page budget")
    if not pieces:
        raise ValueError("No candles received: " + symbol)
    return pd.concat(pieces, ignore_index=True)


def update_market(root, days=90, reset_history=False):
    root = Path(root)
    cfg = config(root)
    now = pd.Timestamp.now(tz="UTC")
    path = root / "data/processed/bars.parquet"
    previous = (
        pd.read_parquet(path) if path.exists() and not reset_history else pd.DataFrame()
    )
    starts = {
        symbol: now.floor("h") - pd.Timedelta(days=days) for symbol in cfg.symbols
    }
    if not previous.empty:
        for symbol in cfg.symbols:
            existing = previous.loc[previous.symbol.eq(symbol), "timestamp"]
            if len(existing):
                starts[symbol] = existing.max()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            symbol: pool.submit(fetch_symbol, symbol, starts[symbol], now)
            for symbol in cfg.symbols
        }
        received = [future.result() for future in futures.values()]
    combined = pd.concat(
        ([previous] if not previous.empty else []) + received, ignore_index=True
    )
    combined = combined[combined.timestamp + pd.Timedelta(hours=1) <= now]
    clean, bad, quality = validate_bars(combined, now=now)
    if set(clean.symbol.unique()) != set(cfg.symbols):
        raise ValueError("Missing market data for one of the configured symbols")
    write_parquet(path, clean)
    if not bad.empty:
        write_parquet(root / "data/quarantine/live_bars.parquet", bad)
    write_json(
        root / "data/processed/live_quality.json", {"updated_at": utcnow(), **quality}
    )
    manifest = read_json(root / "data/processed/source_manifest.json", {})
    write_json(
        root / "data/processed/source_manifest.json",
        {
            "provider": "Binance Spot public REST",
            "data_is_synthetic": False,
            "first_download_at": manifest.get("first_download_at", utcnow()),
            "updated_at": utcnow(),
            "endpoint": REST + "/klines",
            "bars_sha256": sha256(path),
            "rows": len(clean),
            "first_candle": clean.timestamp.min(),
            "last_candle": clean.timestamp.max(),
        },
    )
    quotes, errors = {}, {}
    for symbol in cfg.symbols:
        try:
            quote = request_json("/ticker/24hr", symbol=symbol)
            value = {
                key: float(quote[key])
                for key in (
                    "lastPrice",
                    "priceChangePercent",
                    "highPrice",
                    "lowPrice",
                    "quoteVolume",
                )
            }
            if not np.isfinite(list(value.values())).all() or value["lastPrice"] <= 0:
                raise ValueError("Invalid quote values")
            quotes[symbol] = {
                **value,
                "received_at": utcnow(),
                "close_time": int(quote["closeTime"]),
            }
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            errors[symbol] = str(error)
    write_json(
        root / "quotes.json",
        {"quotes": quotes, "errors": errors, "updated_at": utcnow()},
    )
    return clean


def forecast_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return {
        "rows": len(truth),
        "mae": float(np.mean(abs(prediction - truth))),
        "mse": float(np.mean((prediction - truth) ** 2)),
        "mse_ratio_to_zero": float(
            np.mean((prediction - truth) ** 2) / np.mean(truth**2)
        ),
        "directional_accuracy": float(np.mean((prediction > 0) == (truth > 0))),
    }


def make_estimator(name, cfg):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "xgboost_d3":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=10,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=cfg.threads,
            random_state=cfg.seed,
        )
    raise ValueError("Unknown local model")


def fit_local(root):
    root = Path(root)
    cfg = config(root)
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    if (root / "paper/state.json").exists():
        raise ValueError("Archive the paper session before refitting models")
    samples = build_features(bars, cfg)
    labelled = samples.dropna(subset=["target_return", "label_end_time"])
    times = np.sort(labelled.decision_time.unique())
    if len(times) < 1200:
        raise ValueError(
            "Need at least 1,200 hourly observations; load 90 days of history"
        )
    edges = [
        pd.Timestamp(times[int(len(times) * fraction)])
        for fraction in (0.55, 0.70, 0.85)
    ]
    edges.append(pd.Timestamp(labelled.label_end_time.max()) + pd.Timedelta(hours=1))
    run = root / "runs" / ("local-" + stamp())
    run.mkdir(parents=True, exist_ok=False)
    registry, metric_rows, predictions = {}, [], []
    for name in ("ridge", "xgboost_d3"):
        started, folds = time.monotonic(), []
        for fold, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            train = labelled[labelled.label_end_time < start]
            validation = labelled[
                (labelled.decision_time >= start) & (labelled.label_end_time < end)
            ].copy()
            assert not train.empty and not validation.empty
            assert train.label_end_time.max() < validation.decision_time.min()
            estimator = make_estimator(name, cfg)
            estimator.fit(train[FEATURES], train.target_return)
            validation["prediction"] = estimator.predict(validation[FEATURES])
            validation["fold"], validation["model"] = fold, name
            folds.append(validation)
        oof = pd.concat(folds, ignore_index=True)
        if predictions:
            pd.testing.assert_frame_equal(
                oof[["symbol", "decision_time", "target_return"]].reset_index(
                    drop=True
                ),
                predictions[0][
                    ["symbol", "decision_time", "target_return"]
                ].reset_index(drop=True),
            )
        estimator = make_estimator(name, cfg)
        estimator.fit(labelled[FEATURES], labelled.target_return)
        path = run / (name + ".joblib")
        joblib.dump(
            {
                "model": estimator,
                "name": name,
                "features": FEATURES,
                "threshold": 0.006,
                "config": cfg.to_dict(),
            },
            path,
        )
        metrics = forecast_metrics(oof.target_return, oof.prediction)
        import sklearn

        meta = {
            "created_at": utcnow(),
            "type": "sklearn",
            "model": name,
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "threshold": 0.006,
            "features": FEATURES,
            "config": cfg.to_dict(),
            "training_rows": len(labelled),
            "feature_code_sha256": sha256(Path(__file__).parent / "features.py"),
            "bars_sha256": sha256(root / "data/processed/bars.parquet"),
            "source_control": git_provenance(PROJECT),
            "sklearn_version": sklearn.__version__,
            "deployment_status": "paper_only_baseline",
            "validation_is_independent_final_test": False,
            "validation": metrics,
            "residual_abs_q80": float(
                np.quantile(abs(oof.prediction - oof.target_return), 0.8)
            ),
            "fit_seconds": time.monotonic() - started,
            "artifact_bytes": path.stat().st_size,
            "validation_start": str(oof.decision_time.min()),
            "validation_end": str(oof.label_end_time.max()),
            "run": str(run.relative_to(root)),
        }
        registry[name] = meta
        from .backtest import evaluate_strategy, simulate

        start, end = oof.decision_time.min(), oof.label_end_time.max()
        curve, trades, strategy = evaluate_strategy(
            bars, oof, oof.prediction, 0.006, start, end, cfg
        )
        empty = pd.DataFrame(columns=["symbol", "decision_time", "go_long"])
        hold, _, hold_metrics = simulate(bars, empty, start, end, cfg, buy_hold=True)
        ema, _, ema_metrics = evaluate_strategy(
            bars, oof, oof.ema_gap, 0, start, end, cfg
        )
        write_parquet(
            run / (name + "_equity.parquet"),
            pd.DataFrame(
                {
                    "timestamp": curve.index,
                    "model": curve.values,
                    "buy_hold": hold.values,
                    "ema": ema.values,
                }
            ),
        )
        write_parquet(run / (name + "_trades.parquet"), trades)
        write_json(
            run / (name + "_backtest.json"),
            {"strategy": strategy, "buy_hold": hold_metrics, "ema": ema_metrics},
        )
        metric_rows.append({"model": name, **metrics})
        predictions.append(
            oof[
                [
                    "symbol",
                    "decision_time",
                    "label_end_time",
                    "target_return",
                    "prediction",
                    "fold",
                    "model",
                ]
            ]
        )
    reference = predictions[0]
    for name, forecast in (
        ("zero_return", np.zeros(len(reference))),
        ("naive_ret4", np.expm1(pd.concat(folds).ret_4.to_numpy())),
    ):
        metric_rows.append(
            {"model": name, **forecast_metrics(reference.target_return, forecast)}
        )
    write_parquet(
        run / "validation_predictions.parquet",
        pd.concat(predictions, ignore_index=True),
    )
    write_json(run / "validation_metrics.json", metric_rows)
    write_json(run / "registry.json", registry)
    write_parquet(run / "training_bars.parquet", bars)
    previous = read_json(root / "artifacts/local_registry.json", {})
    combined = {
        name: item
        for name, item in previous.items()
        if item.get("type") in ("tcn", "portable")
    }
    combined.update(registry)
    previous_active = read_json(root / "artifacts/champion.json", {}).get(
        "model", "ridge"
    )
    write_json(root / "artifacts/local_registry.json", combined)
    write_json(root / "artifacts/latest_run.json", {"run": str(run.relative_to(root))})
    # A fixed baseline serves the app; no automatic promotion to real-money execution.
    activate(root, previous_active if previous_active in combined else "ridge")
    return registry


def activate(root, model, archive_empty=False):
    root = Path(root)
    registry = read_json(root / "artifacts/local_registry.json", {})
    if model not in registry:
        raise ValueError("Model not found in local registry")
    target = registry[model]
    if target.get("path") and sha256(root / target["path"]) != target["sha256"]:
        raise ValueError("Model artifact hash mismatch")
    if target.get("type") == "tcn":
        import torch

        torch.set_num_threads(config(root).threads)
        from .tcn_import import load_checked_payload

        load_checked_payload(root / target["path"], target)
    if target.get("type") == "portable":
        from .workflow_model import check_parity

        path = root / target["path"]
        check_parity(path, target, read_json(path.parent / "parity.json"))
    state = read_json(root / "paper/state.json", {})
    if state and state.get("model_sha256") != registry[model]["sha256"]:
        if not archive_empty:
            raise ValueError("Archive the current paper session before changing models")
        _archive_paper(root)
    write_json(root / "artifacts/champion.json", registry[model])
    return registry[model]


def predict_latest(root):
    root = Path(root)
    cfg = config(root)
    meta, _ = load_metadata(root)
    if meta.get("type", "sklearn") == "sklearn":
        import sklearn

        if meta.get("sklearn_version") != sklearn.__version__:
            raise ValueError(
                "Model library differs; refit the baseline from saved candles"
            )
    elif meta.get("type") == "tcn" or meta.get("hyperparameters", {}).get("family") in (
        "tcn",
        "gru",
    ):
        import torch

        torch.set_num_threads(cfg.threads)
    if meta.get("feature_code_sha256") != sha256(Path(__file__).parent / "features.py"):
        raise ValueError("Feature code changed; retrain the local baseline")
    for key in ("symbols", "interval", "horizon", "execution_lag_bars", "lookback"):
        expected = meta.get("config", {}).get(key)
        if expected is not None and json.dumps(expected) != json.dumps(
            cfg.to_dict()[key]
        ):
            raise ValueError("Cấu hình inference khác model: " + key)
    samples = build_features(pd.read_parquet(root / "data/processed/bars.parquet"), cfg)
    context = samples.groupby("symbol", sort=True).tail(cfg.lookback).copy()
    forecasts = pd.Series(predict_champion(root, context), index=context.index)
    latest = context.groupby("symbol", sort=True).tail(1).copy()
    latest["prediction"] = forecasts.loc[latest.index]
    now = pd.Timestamp.now(tz="UTC")
    records = []
    for row in latest.itertuples():
        age = float((now - row.decision_time).total_seconds())
        record = {
            "symbol": row.symbol,
            "decision_time": row.decision_time.isoformat(),
            "entry_time": (
                row.decision_time + pd.Timedelta(hours=cfg.execution_lag_bars)
            ).isoformat(),
            "horizon_end": (
                row.decision_time
                + pd.Timedelta(hours=cfg.execution_lag_bars + cfg.horizon)
            ).isoformat(),
            "prediction": float(row.prediction),
            "last_close": float(row.close),
            "model": meta["model"],
            "model_sha256": meta["sha256"],
            "data_age_seconds": age,
            "fresh": 0 <= age <= 3720,
            "recorded_at": utcnow(),
            "usage": "paper_only",
        }
        if not np.isfinite(record["prediction"]):
            raise ValueError(
                "Dự báo không hữu hạn; TCN cần đủ nến liên tục theo lookback đã train"
            )
        records.append(record)
    with sqlite3.connect(root / "terminal.sqlite", timeout=10) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS forecasts (model_sha TEXT, symbol TEXT, decision_time TEXT, payload TEXT, PRIMARY KEY(model_sha,symbol,decision_time))"
        )
        for row in records:
            if row["fresh"]:
                db.execute(
                    "INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?)",
                    (
                        row["model_sha256"],
                        row["symbol"],
                        row["decision_time"],
                        json.dumps(row),
                    ),
                )
    write_json(
        root / "latest_forecasts.json", {"updated_at": utcnow(), "forecasts": records}
    )
    return records


def forecast_history(root, limit=1000):
    path = Path(root) / "terminal.sqlite"
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT payload FROM forecasts ORDER BY decision_time DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    frame = pd.DataFrame([json.loads(row[0]) for row in rows])
    if not frame.empty:
        frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    return frame


def local_step(root, refresh=True):
    root = Path(root)
    cfg = config(root)
    with FileLock(str(root / "terminal.lock"), timeout=2):
        try:
            if refresh:
                update_market(root)
            forecasts = predict_latest(root)
            controls = read_json(root / "controls.json", {"paper_enabled": False})
            state = read_json(root / "paper/state.json", {})
            if controls.get("paper_enabled") or state:
                from .paper import paper_step

                quotes = read_json(root / "quotes.json", {}).get("quotes", {})

                def price(symbol):
                    quote = quotes[symbol]
                    if pd.Timestamp.now(tz="UTC") - pd.Timestamp(
                        quote["received_at"]
                    ) > pd.Timedelta(minutes=2):
                        raise ValueError("Quote too old for paper fill")
                    return quote["lastPrice"]

                paper_step(
                    root,
                    cfg,
                    refresh_data=False,
                    price_provider=price,
                    allow_new_entries=bool(controls.get("paper_enabled")),
                )
            write_json(
                root / "health.json",
                {"status": "ok", "updated_at": utcnow(), "error": None},
            )
            return forecasts
        except Exception as error:
            write_json(
                root / "health.json",
                {"status": "error", "updated_at": utcnow(), "error": str(error)},
            )
            raise


def initialize(root, days=90):
    root = Path(root)
    config(root)
    if (root / "artifacts/local_registry.json").exists():
        raise ValueError(
            "Local models already exist; use refresh or refit from saved candles"
        )
    with FileLock(str(root / "terminal.lock"), timeout=2):
        update_market(root, days=days, reset_history=True)
        fit_local(root)
    return local_step(root, refresh=False)


def archive_paper(root):
    root = Path(root)
    with FileLock(str(root / "terminal.lock"), timeout=2):
        _archive_paper(root)


def _archive_paper(root):
    """Caller holds terminal.lock, including model activation transactions."""
    root = Path(root)
    path = root / "paper/state.json"
    if path.exists():
        state = read_json(path)
        if any(asset.get("qty") for asset in state.get("assets", {}).values()):
            raise ValueError(
                "Paper positions are still open; dừng lệnh mới và chờ vị thế thoát"
            )
        path.rename(path.with_name("state_archived_" + stamp() + ".json"))
    write_json(root / "controls.json", {"paper_enabled": False})


def feature_contributions(root, symbol):
    root = Path(root)
    meta, path = load_metadata(root)
    if meta.get("type") == "portable":
        family = meta["hyperparameters"]["family"]
        if family == "xgboost":
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(path)
            score = model.get_score(importance_type="gain")
            return pd.DataFrame(
                {"feature": FEATURES, "value": [score.get(f, 0) for f in FEATURES]}
            ).sort_values("value", ascending=False)
        if family == "ridge":
            data = read_json(path)
            samples = build_features(
                pd.read_parquet(root / "data/processed/bars.parquet"), config(root)
            )
            row = samples[samples.symbol.eq(symbol)].tail(1)
            x = (row[FEATURES].to_numpy()[0] - np.asarray(data["mean"])) / np.asarray(
                data["scale"]
            )
            return (
                pd.DataFrame(
                    {"feature": FEATURES, "value": x * np.asarray(data["coef"])}
                )
                .assign(magnitude=lambda f: f.value.abs())
                .sort_values("magnitude", ascending=False)
            )
        raise ValueError("Neural model has no linear feature contributions")
    model = joblib.load(path)["model"]
    if meta["model"] != "ridge":
        return pd.DataFrame(
            {"feature": FEATURES, "value": model.feature_importances_}
        ).sort_values("value", ascending=False)
    samples = build_features(
        pd.read_parquet(root / "data/processed/bars.parquet"), config(root)
    )
    row = samples[samples.symbol.eq(symbol)].tail(1)
    x = model.named_steps["standardscaler"].transform(row[FEATURES])[0]
    return (
        pd.DataFrame(
            {"feature": FEATURES, "value": x * model.named_steps["ridge"].coef_}
        )
        .assign(magnitude=lambda f: f.value.abs())
        .sort_values("magnitude", ascending=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "workspace")
    parser.add_argument(
        "command",
        choices=(
            "setup",
            "refresh",
            "watch",
            "import-tcn",
            "import-model",
            "inspect-tcn",
            "activate",
            "archive-paper",
        ),
    )
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--archive-empty-paper", action="store_true")
    args = parser.parse_args()
    if args.command in ("import-tcn", "inspect-tcn", "import-model"):
        if not args.folder:
            parser.error("--folder is required")
        from .tcn_import import import_checkpoint, inspect_checkpoint

        if args.command == "import-model":
            from .tcn_import import import_bundle

            result = import_bundle(args.root, args.folder)
        elif args.command == "import-tcn":
            result = import_checkpoint(args.root, args.folder)
        else:
            metadata, origin, _ = inspect_checkpoint(args.folder, config(args.root))
            result = {
                "sha256": metadata["sha256"],
                "origin": origin,
                "created_at": metadata["created_at"],
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "activate":
        if not args.model:
            parser.error("--model is required")
        config(args.root)
        with FileLock(str(args.root / "terminal.lock"), timeout=5):
            activate(args.root, args.model, archive_empty=args.archive_empty_paper)
        print(json.dumps(local_step(args.root, refresh=False), indent=2))
    elif args.command == "archive-paper":
        archive_paper(args.root)
        print("Đã lưu phiên paper; model không đổi")
    elif args.command == "setup":
        print(json.dumps(initialize(args.root, args.days), indent=2))
    elif args.command == "refresh":
        print(json.dumps(local_step(args.root), indent=2))
    else:
        while True:
            try:
                local_step(args.root)
                print(utcnow(), "updated", flush=True)
            except Exception as error:
                print(utcnow(), type(error).__name__, str(error), flush=True)
            time.sleep(60)
