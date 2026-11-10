"""Walk-forward selection on development data; one locked final holdout."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error
from .common import sha256, write_json, write_parquet, stamp, utcnow
from .features import FEATURES, fold_data
from .backtest import evaluate_strategy, simulate
from .contracts import verify_feature_snapshot


def make_model(name, cfg):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=cfg.seed))
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=180, max_depth=int(name[-1]), learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85, min_child_weight=30,
                        reg_lambda=10, objective="reg:squarederror", tree_method="hist",
                        random_state=cfg.seed, n_jobs=cfg.threads)


def fit_model(model, name, train):
    y = (train.target_return > 0).astype(int) if name == "logistic" else train.target_return
    model.fit(train[FEATURES], y)
    return model


def predict_model(model, name, samples):
    if name == "logistic":
        return model.predict_proba(samples[FEATURES])[:, 1]
    return model.predict(samples[FEATURES])


def score_summary(summary, minimum_trades):
    # No amount of tuning can make a strategy eligible without enough observations.
    if summary["trades"] < minimum_trades:
        return -1e6 + summary["trades"]
    return summary["sharpe_hourly_estimate"]


def research(root, cfg):
    root = Path(root)
    _, plan = verify_feature_snapshot(root, cfg)
    samples = pd.read_parquet(root / "data/processed/features.parquet")
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    run = root / "runs" / stamp()
    run.mkdir(parents=True)
    names = ["ridge", "logistic", "xgboost_d3", "xgboost_d5"]
    comparisons, all_oof = [], []
    for name in names:
        frames = []
        for fold, bounds in enumerate(plan["folds"]):
            train, val = fold_data(samples, bounds["validation_start"], bounds["validation_end"])
            model = fit_model(make_model(name, cfg), name, train)
            val = val.copy()
            val["prediction"] = predict_model(model, name, val)
            val["fold"] = fold
            frames.append(val)
        oof = pd.concat(frames, ignore_index=True)
        thresholds = [0.50, 0.53, 0.56, 0.60] if name == "logistic" else [0.0, 0.0015, 0.003, 0.006]
        for threshold in thresholds:
            fold_metrics = []
            for i, bounds in enumerate(plan["folds"]):
                data = oof[oof.fold == i]
                _, _, m = evaluate_strategy(bars, data, data.prediction, threshold,
                                           bounds["validation_start"], bounds["validation_end"], cfg)
                fold_metrics.append(m)
            total_trades = sum(m["trades"] for m in fold_metrics)
            eligible = total_trades >= cfg.minimum_trades
            mean_sharpe = float(np.mean([m["sharpe_hourly_estimate"] for m in fold_metrics]))
            comparisons.append({"model": name, "threshold": threshold, "eligible": eligible,
                                "mean_validation_sharpe": mean_sharpe,
                                "mean_validation_return": float(np.mean([m["total_return"] for m in fold_metrics])),
                                "worst_validation_drawdown": min(m["max_drawdown"] for m in fold_metrics),
                                "validation_trades": total_trades,
                                "last_fold": fold_metrics[-1],
                                "selection_score": mean_sharpe if eligible else -1e6 + total_trades})
        oof["model"] = name
        all_oof.append(oof)
        print(f"Walk-forward complete: {name}", flush=True)
    comparisons.sort(key=lambda x: (x["selection_score"], x["mean_validation_return"]), reverse=True)
    winner = comparisons[0]
    dev = samples[(samples.label_end_time < pd.Timestamp(plan["test_start"])) & samples.target_return.notna()]
    final = fit_model(make_model(winner["model"], cfg), winner["model"], dev)
    from .monitor import drift_reference
    write_json(root / "artifacts/drift_reference.json", drift_reference(dev))
    bundle = {"model": final, "name": winner["model"], "features": FEATURES,
              "threshold": winner["threshold"], "config": cfg.to_dict()}
    joblib.dump(bundle, run / "model.joblib")
    oof_output = pd.concat(all_oof, ignore_index=True)
    write_parquet(run / "validation_predictions.parquet", oof_output[["symbol", "decision_time", "target_return", "prediction", "fold", "model"]])
    metadata = {"created_at": utcnow(), "type": "sklearn", "model": winner["model"],
                "path": str((run / "model.joblib").relative_to(root)),
                "sha256": sha256(run / "model.joblib"), "threshold": winner["threshold"],
                "features": FEATURES, "config": cfg.to_dict(), "split": plan,
                "features_sha256": sha256(root / "data/processed/features.parquet"),
                "training_rows": len(dev), "selection": winner,
                "deployment_status": "research_only"}
    write_json(run / "model_metadata.json", metadata)
    write_json(run / "comparison.json", comparisons)
    write_json(root / "artifacts/champion.json", metadata)
    write_json(root / "artifacts/latest_run.json", {"run": str(run.relative_to(root))})
    return run


def load_metadata(root):
    root = Path(root)
    meta = json.loads((root / "artifacts/champion.json").read_text(encoding="utf-8"))
    path = root / meta["path"]
    if sha256(path) != meta["sha256"]:
        raise ValueError("Model artifact hash mismatch")
    return meta, path


def predict_champion(root, samples):
    meta, path = load_metadata(root)
    if meta["type"] == "sklearn":
        bundle = joblib.load(path)  # Only load artifacts generated by your own trusted training run.
        return predict_model(bundle["model"], bundle["name"], samples)
    if meta["type"] == "tcn":
        from .deep import predict_tcn_file
        return predict_tcn_file(path, samples)
    raise ValueError("Unsupported serving model")


def final_evaluate(root, cfg):
    root = Path(root)
    meta, _ = load_metadata(root)
    if json.dumps(meta["config"], sort_keys=True) != json.dumps(cfg.to_dict(), sort_keys=True):
        raise ValueError("Evaluation config must match the frozen training configuration.")
    run = root / json.loads((root / "artifacts/latest_run.json").read_text())["run"]
    lock = root / "artifacts/final_test_lock.json"
    fingerprint = meta["features_sha256"]
    if lock.exists():
        used = json.loads(lock.read_text())
        if used["dataset"] == fingerprint and used["model"] == meta["sha256"]:
            return root / used["report"]
        raise ValueError("Final holdout already opened. Use NEW future holdout data for another model; see README.")
    if fingerprint != sha256(root / "data/processed/features.parquet"):
        raise ValueError("Dataset changed after training. Prepare and train a new research run.")
    samples = pd.read_parquet(root / "data/processed/features.parquet")
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    start, end = pd.Timestamp(meta["split"]["test_start"]), pd.Timestamp(meta["split"]["end"])
    test = samples[(samples.decision_time >= start) & (samples.label_end_time < end) & samples.target_return.notna()].copy()
    preds = predict_champion(root, samples)
    prediction = pd.Series(preds, index=samples.index).loc[test.index].to_numpy()
    valid = np.isfinite(prediction)
    test, prediction = test.loc[valid], prediction[valid]
    curve, trades, summary = evaluate_strategy(bars, test, prediction, meta["threshold"], start, end, cfg)
    no_signals = pd.DataFrame(columns=["symbol", "decision_time", "go_long"])
    hold_curve, hold_trades, hold_metrics = simulate(bars, no_signals, start, end, cfg, buy_hold=True)
    ema_curve, ema_trades, ema_metrics = evaluate_strategy(bars, test, test.ema_gap.to_numpy(), 0.0, start, end, cfg)
    write_parquet(run / "test_equity.parquet", pd.DataFrame({"timestamp": curve.index, "model": curve.values,
                                                            "buy_hold": hold_curve.values, "ema": ema_curve.values}))
    write_parquet(run / "test_trades.parquet", trades)
    write_parquet(run / "test_predictions.parquet", test[["symbol", "decision_time", "target_return"]].assign(prediction=prediction))
    if meta["model"] == "logistic":
        directional = float(np.mean((prediction > 0.5) == (test.target_return.to_numpy() > 0)))
        prediction_metrics = {"directional_accuracy": directional, "mae_return": None}
    else:
        prediction_metrics = {"directional_accuracy": float(np.mean((prediction > 0) == (test.target_return.to_numpy() > 0))),
                              "mae_return": float(mean_absolute_error(test.target_return, prediction))}
    report = {"evaluated_at": utcnow(), "model": meta["model"], "model_sha256": meta["sha256"],
              "threshold": meta["threshold"], "period": {"start": str(start), "end": str(end)},
              "prediction": prediction_metrics, "strategy": summary, "buy_hold": hold_metrics, "ema": ema_metrics,
              "assumptions": {"fee_bps_each_side": cfg.fee_bps, "slippage_bps_each_side": cfg.slippage_bps,
                              "execution_lag_hours": cfg.execution_lag_bars, "holding_hours": cfg.horizon,
                              "drawdown_halt_per_asset": cfg.drawdown_halt, "position_fraction_per_asset_sleeve": cfg.position_fraction},
              "limitations": ["Hourly OHLCV fills are an approximation, not order-book execution.",
                              "Two chosen liquid assets have selection bias; this is not an all-crypto universe study.",
                              "Trading rule and model were selected only on development folds.",
                              "No profitability guarantee. Sharpe is an unadjusted hourly estimate."]}
    write_json(run / "final_report.json", report)
    write_json(lock, {"dataset": fingerprint, "model": meta["sha256"], "opened_at": utcnow(),
                      "report": str((run / "final_report.json").relative_to(root))})
    return run / "final_report.json"


def export_kaggle(root, cfg):
    """Export development-only data. The final holdout is not in this ZIP."""
    import zipfile
    root = Path(root)
    _, plan = verify_feature_snapshot(root, cfg)
    all_features = pd.read_parquet(root / "data/processed/features.parquet")
    bounds = plan["folds"][-1]
    train, val = fold_data(all_features, bounds["validation_start"], bounds["validation_end"])
    dev = all_features[(all_features.label_end_time < pd.Timestamp(plan["test_start"])) & all_features.target_return.notna()]
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    bars = bars[bars.timestamp + pd.Timedelta(hours=1) < pd.Timestamp(plan["test_start"])]
    folder = root / "kaggle_export"
    folder.mkdir(exist_ok=True)
    for name, df in [("train", train), ("validation", val), ("development", dev), ("bars", bars)]:
        write_parquet(folder / f"{name}.parquet", df)
    manifest = {"created_at": utcnow(), "config": cfg.to_dict(), "split": plan, "feature_columns": FEATURES,
                "full_features_sha256": sha256(root / "data/processed/features.parquet"),
                "files": {p.name: sha256(p) for p in folder.glob("*.parquet")},
                "contains_final_holdout": False}
    write_json(folder / "manifest.json", manifest)
    archive = root / "QuantStream_Kaggle_Input.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.iterdir()):
            z.write(p, "snapshot/" + p.name)
        for p in sorted((root / "quantstream").glob("*.py")):
            z.write(p, "code/quantstream/" + p.name)
        z.write(root / "pyproject.toml", "code/pyproject.toml")
    return archive
