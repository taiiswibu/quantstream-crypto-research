"""Budgeted, chronological model search. Never calls final_evaluate or registers a model.

Outer folds 0/1 rank candidates; inner chronological tails choose neural epochs.
Fold 2 is a single post-selection development diagnostic, historically already seen.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .backtest import evaluate_strategy
from .common import Config, sha256, utcnow, write_json, write_parquet
from .deep import CausalBlock, Windows, predict_tcn
from .features import FEATURES, fold_data

PROTOCOL = "budget-search-v1"
THRESHOLD = 0.006  # Fixed from run 001; never searched in this experiment.
SEEDS = (42, 123, 2026)
MAX_LOOKBACK = 48


class SequenceRegressor(nn.Module):
    def __init__(self, params):
        super().__init__()
        width = params["width"]
        self.family = params["family"]
        if self.family == "tcn":
            self.encoder = nn.Sequential(CausalBlock(len(FEATURES), width, 1),
                                         CausalBlock(width, width, 2),
                                         CausalBlock(width, width, 4),
                                         CausalBlock(width, width, 8))
        elif self.family == "gru":
            self.encoder = nn.GRU(len(FEATURES), width, batch_first=True)
        else:
            raise ValueError("Expected tcn or gru")
        self.head = nn.Sequential(nn.Dropout(params["dropout"]), nn.Linear(width, 1))

    def forward(self, x):
        if self.family == "tcn":
            hidden = self.encoder(x.transpose(1, 2))[:, :, -1]
        else:
            hidden = self.encoder(x)[0][:, -1]
        return self.head(hidden).squeeze(-1)


def inner_split(train, fraction=0.2):
    """Same UTC cut for every asset, purge labels crossing that cut."""
    times = np.sort(train.decision_time.unique())
    cut = pd.Timestamp(times[int(len(times) * (1 - fraction))])
    fit = train[train.label_end_time < cut].copy()
    stop = train[(train.decision_time >= cut) & train.target_return.notna()].copy()
    if fit.empty or stop.empty or fit.label_end_time.max() >= stop.decision_time.min():
        raise ValueError("Invalid inner chronological split")
    return fit, stop, cut


def with_context(dev, val, start):
    past = dev[dev.decision_time < pd.Timestamp(start)].groupby("symbol").tail(MAX_LOOKBACK)
    context = pd.concat([past, val]).drop_duplicates(["symbol", "decision_time"])
    context = context.sort_values(["decision_time", "symbol"]).reset_index(drop=True)
    context["is_validation"] = context.decision_time.ge(pd.Timestamp(start))
    context.loc[~context.is_validation, "target_return"] = np.nan
    # Common 48-hour contiguous coverage for tabular and sequence families.
    windows = Windows(context, np.zeros(len(FEATURES)), np.ones(len(FEATURES)),
                      MAX_LOOKBACK, labelled=False)
    context["eligible_window"] = context.index.isin([item[2] for item in windows.items])
    return context


def neural_fit(train, params, seed, epochs, device, threads, validation=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    mean = train[FEATURES].mean().to_numpy(np.float32)
    std = train[FEATURES].std().clip(lower=1e-6).to_numpy(np.float32)
    model = SequenceRegressor(params).to(device)
    data = Windows(train.reset_index(drop=True), mean, std, params["lookback"])
    if not len(data):
        raise ValueError("No contiguous training windows")
    loader = DataLoader(data, batch_size=params["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                 weight_decay=params["weight_decay"])
    loss_fn = nn.HuberLoss()
    history, best, best_loss, best_epoch, stale = [], None, float("inf"), epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
        row = {"epoch": epoch, "train_huber": total / seen}
        if validation is not None:
            prediction = predict_tcn(model, validation, mean, std, params["lookback"], device)
            mask = (validation.is_validation & validation.eligible_window).to_numpy() & np.isfinite(prediction)
            if not mask.any():
                raise ValueError("No eligible inner validation windows")
            mse = float(np.mean((prediction[mask] - validation.target_return.to_numpy()[mask]) ** 2))
            row["inner_validation_mse"] = mse
            if mse < best_loss:
                best_loss, best_epoch, stale = mse, epoch, 0
                best = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= 2:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, mean, std, best_epoch, history


def fit_predict(dev, bounds, params, seed, max_epochs, cfg, device):
    train, val = fold_data(dev, bounds["validation_start"], bounds["validation_end"])
    context = with_context(dev, val, bounds["validation_start"])
    trace = {}
    if params["family"] in ("tcn", "gru"):
        fit, stop, cut = inner_split(train)
        inner_context = with_context(train, stop, cut)
        pilot, _, _, epoch, history = neural_fit(fit, params, seed, max_epochs, device,
                                                cfg.threads, inner_context)
        del pilot
        # Retrain on outer TRAIN only with a frozen epoch count. No outer labels
        # are passed to early stopping, normalization or model fitting.
        model, mean, std, _, _ = neural_fit(train, params, seed, epoch, device, cfg.threads)
        prediction = predict_tcn(model, context, mean, std, params["lookback"], device)
        trace = {"best_epoch": epoch, "inner_cut": str(cut), "history": history}
        del model
    else:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        if params["family"] == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
        else:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=120, max_depth=params["max_depth"],
                                 learning_rate=params["learning_rate"], subsample=0.85,
                                 colsample_bytree=0.85, min_child_weight=30, reg_lambda=10,
                                 tree_method="hist", random_state=seed, n_jobs=cfg.threads)
        model.fit(train[FEATURES], train.target_return)
        prediction = model.predict(context[FEATURES])
    mask = (context.is_validation & context.eligible_window).to_numpy()
    if not np.isfinite(prediction[mask]).all() or not mask.any():
        raise ValueError("Missing predictions on shared evaluation population")
    part = context[mask].copy()
    part["prediction"] = prediction[mask]
    return part, trace


def prediction_metrics(part):
    actual, pred = part.target_return.to_numpy(), part.prediction.to_numpy()
    mse, zero_mse = float(np.mean((pred - actual) ** 2)), float(np.mean(actual ** 2))
    return {"mse": mse, "zero_forecast_mse": zero_mse, "mse_ratio_to_zero": mse / max(zero_mse, 1e-15),
            "mae": float(np.mean(abs(pred - actual))),
            "directional_accuracy": float(np.mean((pred > 0) == (actual > 0))), "rows": len(part)}


def evaluate_candidate(dev, bars, manifest, params, seed, max_epochs, cfg, device, folds=(0, 1)):
    records, predictions = [], []
    for fold in folds:
        bounds = manifest["split"]["folds"][fold]
        part, trace = fit_predict(dev, bounds, params, seed, max_epochs, cfg, device)
        _, _, strategy = evaluate_strategy(bars, part, part.prediction, THRESHOLD,
                                           bounds["validation_start"], bounds["validation_end"], cfg)
        records.append({"fold": fold, "prediction": prediction_metrics(part), "strategy": strategy, **trace})
        predictions.append(part[["symbol", "decision_time", "target_return", "prediction"]].assign(fold=fold))
    return {"params": params, "seed": seed, "folds": records,
            "objective": float(np.mean([r["prediction"]["mse_ratio_to_zero"] for r in records])),
            "mean_sharpe_diagnostic": float(np.mean([r["strategy"]["sharpe_hourly_estimate"] for r in records]))}, pd.concat(predictions, ignore_index=True)


def suggest_params(trial):
    family = trial.suggest_categorical("family", ["ridge", "xgboost", "tcn", "gru"])
    if family == "ridge":
        return {"family": family, "alpha": trial.suggest_float("alpha", 0.1, 100, log=True)}
    if family == "xgboost":
        return {"family": family, "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True)}
    return {"family": family, "lookback": trial.suggest_categorical("lookback", [24, 48]),
            "width": trial.suggest_categorical("width", [16, 32]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.15, 0.3]),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512])}


def load_snapshot(snapshot):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("contains_final_holdout") is not False or manifest["feature_columns"] != FEATURES:
        raise ValueError("Expected development-only snapshot with matching features")
    for name in ("development.parquet", "bars.parquet", "train.parquet", "validation.parquet"):
        if sha256(snapshot / name) != manifest["files"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    dev, bars = pd.read_parquet(snapshot / "development.parquet"), pd.read_parquet(snapshot / "bars.parquet")
    end = pd.Timestamp(manifest["split"]["test_start"])
    if (dev.label_end_time >= end).any() or (dev.decision_time >= end).any():
        raise ValueError("Final holdout leaked into development")
    if (bars.timestamp + pd.Timedelta(hours=1) >= end).any():
        raise ValueError("Final holdout leaked into bars")
    if dev.duplicated(["symbol", "decision_time"]).any():
        raise ValueError("Duplicate sample keys")
    cfg = Config(**{**manifest["config"], "symbols": tuple(manifest["config"]["symbols"])})
    return manifest, dev, bars, cfg


def runtime_info(device):
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "xgboost", "xgboost-cpu", "optuna"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": packages, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}


def run_search(snapshot, output, n_trials=12, max_epochs=8, timeout_seconds=7200, device="auto"):
    """Run sequentially; persist trials, freeze selection, inspect fold 2 once.

    Timeout limits starting search trials, not total runtime; stability/refit can
    run afterwards. Reuse output to resume, not a new folder to erase history.
    """
    import optuna
    if not 4 <= n_trials <= 24 or not 1 <= max_epochs <= 12:
        raise ValueError("Budget: 4..24 total trials, 1..12 maximum epochs")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, dev, bars, cfg = load_snapshot(snapshot)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = {"protocol": PROTOCOL, "snapshot_manifest_sha256": sha256(Path(snapshot) / "manifest.json"),
            "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "source_control": manifest.get("source_control"), "n_trials": n_trials,
            "max_epochs": max_epochs, "timeout_seconds": timeout_seconds, "seeds": list(SEEDS),
            "tuning_folds": [0, 1], "diagnostic_fold": 2, "threshold": THRESHOLD,
            "objective": "mean fold MSE / zero-return forecast MSE; minimize",
            "runtime": runtime_info(device), "contains_final_holdout": False}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen plan/runtime changed. Preserve this experiment; do not overwrite it.")
    write_json(plan_path, plan)
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    if (output / "diagnostic.started.json").exists():
        raise ValueError("Post-selection diagnostic already started. Preserve partial evidence; do not rerun it.")
    started = time.monotonic()
    storage = "sqlite:///" + str(output / "study.sqlite3")
    study = optuna.create_study(study_name=PROTOCOL, storage=storage, load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4))
    if not study.trials:
        for baseline in ({"family": "ridge", "alpha": 10.0},
                         {"family": "xgboost", "max_depth": 3, "learning_rate": 0.04},
                         {"family": "tcn"}, {"family": "gru"}):
            study.enqueue_trial(baseline)

    def objective(trial):
        params = suggest_params(trial)
        report, pred = evaluate_candidate(dev, bars, manifest, params, 42, max_epochs, cfg, device)
        trial.set_user_attr("resolved_params", params)
        path = output / "trials" / f"trial-{trial.number:03}"
        write_json(path / "report.json", report)
        write_parquet(path / "predictions.parquet", pred)
        return report["objective"]

    # Enqueued WAITING trials count only once they start; resumed attempts count
    # against the fixed total even if failed/interrupted. Save Version to keep DB.
    attempts = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    remaining = max(0, n_trials - attempts)
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout_seconds, n_jobs=1)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 4 or len({t.user_attrs["resolved_params"]["family"] for t in complete}) < 4:
        raise ValueError("Need completed trials from all four families. Resume the same output/study.")
    leaders = sorted(complete, key=lambda t: (t.value, t.number))[:2]
    stability = []
    for trial in leaders:
        params = trial.user_attrs["resolved_params"]
        reports = []
        for seed in SEEDS:
            path = output / "stability" / f"trial-{trial.number:03}-seed-{seed}.json"
            if path.exists():
                report = json.loads(path.read_text())
            elif seed == 42:
                report = json.loads((output / "trials" / f"trial-{trial.number:03}" / "report.json").read_text())
                write_json(path, report)
            else:
                report, _ = evaluate_candidate(dev, bars, manifest, params, seed, max_epochs, cfg, device)
                write_json(path, report)
            reports.append(report)
        stability.append({"trial": trial.number, "params": params,
                          "mean_objective": float(np.mean([r["objective"] for r in reports])),
                          "std_objective": float(np.std([r["objective"] for r in reports])), "seeds": reports})
    selected = min(stability, key=lambda r: (r["mean_objective"], r["trial"]))
    write_json(output / "selection.json", {"selected": selected, "shortlist": stability,
                                           "frozen_at": utcnow(), "threshold": THRESHOLD})
    # Lock BEFORE reading the last development fold for this candidate. Even a
    # crash must not silently turn repeated diagnostics into another search loop.
    with (output / "diagnostic.started.json").open("x") as handle:
        json.dump({"selection_sha256": sha256(output / "selection.json"), "created_at": utcnow()}, handle)
    diagnostic, pred = evaluate_candidate(dev, bars, manifest, selected["params"], 42,
                                          max_epochs, cfg, device, folds=(2,))
    write_parquet(output / "diagnostic_predictions.parquet", pred)
    bounds = manifest["split"]["folds"][2]
    sensitivity = []
    for multiplier in (1.0, 1.5, 2.0):
        stressed = replace(cfg, fee_bps=cfg.fee_bps * multiplier, slippage_bps=cfg.slippage_bps * multiplier)
        _, _, summary = evaluate_strategy(bars, pred, pred.prediction, THRESHOLD,
                                           bounds["validation_start"], bounds["validation_end"], stressed)
        sensitivity.append({"cost_multiplier": multiplier, **summary})
    write_json(output / "diagnostic.json", {"report": diagnostic, "cost_sensitivity": sensitivity})
    result = {"protocol": PROTOCOL, "completed_at": utcnow(),
              "attempted_trials": sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials),
              "completed_trials": len(complete), "selection": selected, "diagnostic": diagnostic,
              "cost_sensitivity": sensitivity, "elapsed_seconds": time.monotonic() - started,
              "deployment_status": "research_only_no_promotion", "contains_final_holdout": False,
              "limitations": ["Development folds, including fold 2, were historically observed in run 001.",
                              "Repeated search can overfit development; metrics are not an independent alpha test.",
                              "MSE selection need not produce a profitable trading strategy.",
                              "Sharpe is an unadjusted hourly estimate; trade count may be small.",
                              "Optuna resumes saved parameters/trials; TPE RNG restarts on process restart.",
                              "No checkpoint refit or serving promotion is performed by this search."]}
    write_json(output / "result.json", result)
    return result
