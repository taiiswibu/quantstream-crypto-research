"""Chronological data-to-model research with a fixed whole-job budget."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .common import Config, git_provenance, sha256, utcnow, write_json, write_parquet
from .features import FEATURES, build_features, fold_data


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise BudgetExpired(
                "Whole-job time budget exhausted; partial evidence retained"
            )


def runtime_record(device):
    import importlib
    from .search import runtime_info

    result = runtime_info(device)
    result["loaded_modules"] = {}
    for name in ("torch", "numpy", "pandas", "sklearn", "xgboost", "optuna", "pyarrow"):
        module = importlib.import_module(name)
        result["loaded_modules"][name] = {
            "version": str(module.__version__),
            "path": str(module.__file__),
        }
    return result


def prepare_dataset(root, cfg=None):
    from .data import download_history, validate_bars
    from .terminal import fetch_symbol

    root = Path(root)
    cfg = cfg or replace(Config(), position_fraction=0.1, drawdown_halt=0.1)
    frozen_end = pd.Timestamp("2026-11-14T00:00:00Z")
    contract = {
        "config": cfg.to_dict(),
        "frozen_end": str(frozen_end),
        "features": FEATURES,
        "feature_code_sha256": sha256(Path(__file__).with_name("features.py")),
        "test_is_independent": False,
        "test_note": "Retrospective historical diagnostic; historical periods have been observed before.",
    }
    contract = json.loads(json.dumps(contract))
    plan_path = root / "dataset_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != contract:
        raise ValueError("Dataset plan changed; preserve existing experiment")
    write_json(plan_path, contract)
    prepared = root / "data/processed/features.parquet"
    if prepared.exists():
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if sha256(root / name) != digest:
                raise ValueError("Dataset file changed: " + name)
        return manifest
    download_history(root, cfg)
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    end_month = pd.Timestamp(cfg.end_month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)
    extra = [fetch_symbol(s, end_month, frozen_end) for s in cfg.symbols]
    combined = pd.concat([bars, *extra], ignore_index=True)
    combined = combined[combined.timestamp + pd.Timedelta(hours=1) <= frozen_end]
    clean, bad, quality = validate_bars(combined, now=frozen_end)
    write_parquet(root / "data/processed/bars.parquet", clean)
    if len(bad):
        write_parquet(root / "data/quarantine/bars.parquet", bad)
    write_json(root / "data/processed/data_quality.json", quality)
    features = build_features(clean, cfg)
    labelled = features.dropna(subset=["target_return", "label_end_time"])
    split = make_split(labelled)
    write_parquet(prepared, features)
    write_json(root / "split.json", split)
    manifest = {
        **contract,
        "created_at": utcnow(),
        "source_control": git_provenance(Path(__file__).resolve().parents[1]),
        "files": {
            str(p.relative_to(root)): sha256(p)
            for p in [
                root / "data/processed/bars.parquet",
                prepared,
                root / "split.json",
            ]
        },
        "rows": len(features),
        "split": split,
        "provider": "Binance monthly archives plus public REST",
        "synthetic": False,
    }
    write_json(root / "dataset_manifest.json", manifest)
    return manifest


def make_split(features):
    times = np.sort(features.decision_time.unique())
    if len(times) < 1200:
        raise ValueError("Need at least 1,200 timestamps")

    def cut(f):
        return str(pd.Timestamp(times[int(len(times) * f)]))

    return {
        "tuning_folds": [
            {"validation_start": cut(0.5), "validation_end": cut(0.65)},
            {"validation_start": cut(0.65), "validation_end": cut(0.8)},
        ],
        "test_start": cut(0.8),
        "end": str(features.label_end_time.max() + pd.Timedelta(hours=1)),
        "purge": "train.label_end_time < next block decision_time",
        "test_is_independent": False,
    }


def suggest(trial):
    family = trial.suggest_categorical("family", ["ridge", "xgboost", "tcn", "gru"])
    if family == "ridge":
        return {
            "family": family,
            "alpha": trial.suggest_float("alpha", 0.1, 100, log=True),
        }
    if family == "xgboost":
        return {
            "family": family,
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [120, 300, 600]),
        }
    parameters = {
        "family": family,
        "lookback": 48,
        "width": trial.suggest_categorical("width", [16, 32]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.15, 0.3]),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": 256,
    }

    if family == "tcn":
        parameters["activation"] = trial.suggest_categorical(
            "activation", ["gelu", "relu", "silu"]
        )
    return parameters


def neural_train(
    train, params, seed, epochs, cfg, device, budget, folder, validation=None
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from .deep import Windows, predict_tcn
    from .search import SequenceRegressor

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg.threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    mean = train[FEATURES].mean().to_numpy(np.float32)
    std = train[FEATURES].std().clip(lower=1e-6).to_numpy(np.float32)
    write_json(
        Path(folder) / "scaler.json",
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit_rows": len(train),
            "last_label": str(train.label_end_time.max()),
        },
    )
    model = SequenceRegressor(params).to(device)
    dataset = Windows(train.reset_index(drop=True), mean, std, params["lookback"])
    if not len(dataset):
        raise ValueError("No contiguous train windows")
    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loss_name = params.get("loss", "huber")
    if loss_name not in ("huber", "mse"):
        raise ValueError("Expected huber or mse loss")
    loss_fn = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()
    schedule = params.get("lr_schedule", "constant")
    if schedule not in ("constant", "cosine"):
        raise ValueError("Expected constant or cosine learning rate")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("schedule_epochs", epochs),
            eta_min=params["lr"] * 0.05,
        )
        if schedule == "cosine"
        else None
    )
    history = []
    best = None
    best_mse = float("inf")
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        budget.check()
        model.train()
        loss_sum = 0.0
        count = 0
        grad = []
        for x, y, _ in loader:
            budget.check()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
            grad.append(float(norm))
        row = {
            "epoch": epoch,
            "train_loss_scaled": loss_sum / count,
            "loss_name": loss_name,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_mean": float(np.mean(grad)),
        }
        if loss_name == "huber":
            row["train_huber_scaled"] = row["train_loss_scaled"]
        if validation is not None:
            prediction = predict_tcn(
                model, validation, mean, std, params["lookback"], device
            )
            mask = (validation.is_validation & validation.eligible_window).to_numpy()
            if not mask.any() or not np.isfinite(prediction[mask]).all():
                raise ValueError("No validation windows")
            mse = float(
                np.mean(
                    (prediction[mask] - validation.target_return.to_numpy()[mask]) ** 2
                )
            )
            row["inner_validation_mse"] = mse
            if mse < best_mse:
                best_mse, best_epoch, stale = mse, epoch, 0
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                stale += 1
        history.append(row)
        pd.DataFrame(history).to_csv(Path(folder) / "history.csv", index=False)
        print(Path(folder).name, row, flush=True)
        if scheduler is not None:
            scheduler.step()
        if validation is not None and stale >= params.get("patience", 3):
            break
    if best is not None:
        model.load_state_dict(best)
    return model, mean, std, best_epoch


def fit_model(train, params, seed, epochs, cfg, device, budget, folder):
    """Only train labels enter fitting/scaling or the inner early-stop tail."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    budget.check()
    days = params.get("train_window_days", 0)
    if days:
        train = train[
            train.decision_time.ge(train.decision_time.max() - pd.Timedelta(days=days))
        ].copy()
    if train.empty:
        raise ValueError("Empty training window")
    epochs = min(epochs, params.get("epoch_cap", epochs))
    write_json(
        folder / "fit.json",
        {
            "params": params,
            "seed": seed,
            "rows": len(train),
            "last_label": str(train.label_end_time.max()),
            "first_decision": str(train.decision_time.min()),
        },
    )
    if params["family"] in ("tcn", "gru"):
        from .search import inner_split, with_context

        fit, stop, cut = inner_split(train)
        context = with_context(train, stop, cut)
        pilot = folder / "early-stop"
        pilot.mkdir(exist_ok=True)
        _, _, _, chosen = neural_train(
            fit, params, seed, epochs, cfg, device, budget, pilot, context
        )
        final = folder / "refit"
        final.mkdir(exist_ok=True)
        model, mean, std, _ = neural_train(
            train, params, seed, chosen, cfg, device, budget, final
        )
        return {
            "model": model,
            "mean": mean,
            "std": std,
            "epochs": chosen,
            "params": params,
            "device": device,
            "training_rows": len(train),
            "training_first_decision": str(train.decision_time.min()),
            "training_last_label": str(train.label_end_time.max()),
        }
    if params["family"] == "ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        model = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=30,
            reg_lambda=params.get("reg_lambda", 10),
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            n_jobs=cfg.threads,
            random_state=seed,
        )
    model.fit(train[FEATURES], train.target_return)
    budget.check()
    if params["family"] == "ridge":
        scaler = model.named_steps["standardscaler"]
        write_json(
            folder / "scaler.json",
            {
                "mean": scaler.mean_.tolist(),
                "std": scaler.scale_.tolist(),
                "fit_rows": len(train),
                "last_label": str(train.label_end_time.max()),
            },
        )
    return {
        "model": model,
        "params": params,
        "epochs": None,
        "training_rows": len(train),
        "training_first_decision": str(train.decision_time.min()),
        "training_last_label": str(train.label_end_time.max()),
    }


def predict(fitted, frame):
    if fitted["params"]["family"] in ("tcn", "gru"):
        from .deep import predict_tcn

        return predict_tcn(
            fitted["model"], frame, fitted["mean"], fitted["std"], 48, fitted["device"]
        )
    return fitted["model"].predict(frame[FEATURES])


def forecast_metrics(rows):
    from .search import prediction_metrics

    result = prediction_metrics(rows)
    result["rmse"] = float(np.sqrt(result["mse"]))
    zero_mae = float(np.mean(abs(rows.target_return)))
    result["mae_ratio_to_zero"] = result["mae"] / max(zero_mae, 1e-15)
    return result


def evaluate(features, bars, split, params, seed, epochs, cfg, device, budget, folder):
    from .search import with_context
    from .backtest import evaluate_strategy

    records = []
    parts = []
    for fold, bounds in enumerate(split["tuning_folds"]):
        budget.check()
        train, val = fold_data(
            features, bounds["validation_start"], bounds["validation_end"]
        )
        context = with_context(features, val, bounds["validation_start"])
        fitted = fit_model(
            train,
            params,
            seed,
            epochs,
            cfg,
            device,
            budget,
            Path(folder) / f"fold-{fold}",
        )
        values = predict(fitted, context)
        mask = (context.is_validation & context.eligible_window).to_numpy()
        if not np.isfinite(values[mask]).all():
            raise ValueError("Missing shared-population forecast")
        part = context[mask].copy()
        part["prediction"] = values[mask]
        part["fold"] = fold
        _, _, strategy = evaluate_strategy(
            bars,
            part,
            part.prediction,
            0.006,
            bounds["validation_start"],
            bounds["validation_end"],
            cfg,
        )
        records.append(
            {
                "fold": fold,
                "prediction": forecast_metrics(part),
                "strategy": strategy,
                "epochs": fitted["epochs"],
            }
        )
        parts.append(
            part[["symbol", "decision_time", "target_return", "prediction", "fold"]]
        )
        del fitted
    report = {
        "params": params,
        "seed": seed,
        "folds": records,
        "objective": float(
            np.mean([r["prediction"]["mse_ratio_to_zero"] for r in records])
        ),
    }
    write_json(Path(folder) / "report.json", report)
    write_parquet(
        Path(folder) / "predictions.parquet", pd.concat(parts, ignore_index=True)
    )
    return report


def export_model(
    root, fitted, cfg, manifest, selection, test_report, context, run_name="run006"
):
    import torch
    from .workflow_model import check_parity

    output = Path(root) / "deployment"
    output.mkdir(exist_ok=True)
    params = fitted["params"]
    family = params["family"]
    model = fitted["model"]
    if family == "ridge":
        path = output / "model.json"
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        write_json(
            path,
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": ridge.coef_.tolist(),
                "intercept": float(ridge.intercept_),
            },
        )
    elif family == "xgboost":
        path = output / "model.ubj"
        model.save_model(path)
    else:
        path = output / "model.pt"
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "mean": fitted["mean"].tolist(),
                "std": fitted["std"].tolist(),
                "lookback": 48,
            },
            path,
        )
    metadata = {
        "format": "quantstream-notebook-v1",
        "created_at": utcnow(),
        "type": "portable",
        "model": run_name + "_" + family,
        "research_run": run_name,
        "hyperparameters": params,
        "features": FEATURES,
        "threshold": 0.006,
        "config": cfg.to_dict(),
        "feature_code_sha256": manifest["feature_code_sha256"],
        "sequence_code_sha256": sha256(Path(__file__).with_name("search.py")),
        "source_control": git_provenance(Path(__file__).resolve().parents[1]),
        "training_rows": fitted["training_rows"],
        "training_first_decision": fitted.get("training_first_decision"),
        "training_last_label": fitted.get("training_last_label"),
        "selected_epochs": fitted["epochs"],
        "training_runtime": runtime_record(fitted.get("device", "cpu")),
        "training_features_sha256": manifest["files"][
            "data/processed/features.parquet"
        ],
        "validation": selection["validation"],
        "validation_gate": selection.get("validation_gate"),
        "test": test_report,
        "test_is_independent": False,
        "deployment_status": "research_only_paper",
        "model_file": path.name,
        "sha256": sha256(path),
    }
    if family in ("tcn", "gru"):
        metadata["normalization"] = {
            "mean": fitted["mean"].tolist(),
            "std": fitted["std"].tolist(),
            "lookback": 48,
        }
    windows = []
    expected = []
    for _, g in context.groupby(["symbol", "segment"]):
        if len(g) < 48:
            continue
        piece = g.tail(48).copy().reset_index(drop=True)
        windows.append(piece[FEATURES].to_numpy(np.float32).tolist())
        expected.append(float(predict(fitted, piece)[-1]))
        if len(windows) == 2:
            break
    if not windows:
        raise ValueError("No export parity windows")
    parity = {"windows": windows, "predictions": expected, "atol": 1e-5, "rtol": 1e-5}
    check_parity(path, metadata, parity)
    write_json(output / "parity.json", parity)
    write_json(output / "metadata.json", metadata)
    import shutil

    for trace in (Path(root) / "final-fit").glob("*/history.csv"):
        shutil.copy2(trace, output / ("history_" + trace.parent.name + ".csv"))
    write_json(
        output / "bundle_manifest.json",
        {
            "format": "quantstream-notebook-v1",
            "files": {
                p.name: sha256(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "bundle_manifest.json"
            },
        },
    )
    return str(output)


def run(
    root, n_trials=24, max_epochs=12, seconds=7200, device="auto", protocol="run006"
):
    import optuna
    import torch
    from .search import with_context
    from .backtest import evaluate_strategy, simulate

    root = Path(root)
    if protocol not in ("run006", "run007"):
        raise ValueError("Unknown research protocol")
    limits = (24, 12, 21600) if protocol == "run006" else (48, 60, 14400)
    if (
        not (4 if protocol == "run006" else 14) <= n_trials <= limits[0]
        or not 1 <= max_epochs <= limits[1]
        or not 1 <= seconds <= limits[2]
    ):
        raise ValueError(f"Budget outside {protocol} limits: {limits}")
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if protocol == "run007" and manifest.get("research_protocol") != "run007-depth-v1":
        raise ValueError("Prepare the three-fold Run 007 dataset before training")
    for name, digest in manifest["files"].items():
        if sha256(root / name) != digest:
            raise ValueError("Dataset changed: " + name)
    cfg = Config(
        **{**manifest["config"], "symbols": tuple(manifest["config"]["symbols"])}
    )
    if manifest["feature_code_sha256"] != sha256(
        Path(__file__).with_name("features.py")
    ):
        raise ValueError("Feature source changed after dataset preparation")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = {
        "protocol": "run006-e2e-v1" if protocol == "run006" else "run007-depth-v1",
        "trials": n_trials,
        "epochs": max_epochs,
        "seconds": seconds,
        "dataset_manifest_sha256": sha256(root / "dataset_manifest.json"),
        "source": git_provenance(Path(__file__).resolve().parents[1]),
        "source_files": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "runtime": runtime_record(device),
        "seeds": [42, 123, 2026],
        "test_is_independent": False,
    }
    plan_path = root / "training_plan.json"
    if plan_path.exists():
        raise ValueError(
            "Run already started. Keep evidence; do not rerun or overwrite this experiment"
        )
    with plan_path.open("x") as stream:
        json.dump(plan, stream, indent=2)
    budget = Budget(seconds)
    search_end = time.monotonic() + seconds * 0.6
    features = pd.read_parquet(root / "data/processed/features.parquet")
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    split = manifest["split"]
    dev = features[
        (features.label_end_time < pd.Timestamp(split["test_start"]))
        & features.target_return.notna()
    ]
    study = optuna.create_study(
        study_name=protocol,
        storage="sqlite:///" + str((root / "study.sqlite").resolve()),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    initial = [
        {"family": "ridge", "alpha": 10.0},
        {
            "family": "xgboost",
            "max_depth": 3,
            "learning_rate": 0.04,
            "n_estimators": 120,
        },
        {
            "family": "tcn",
            "width": 32,
            "dropout": 0.15,
            "lr": 0.0005,
            "weight_decay": 0.001,
        },
        {
            "family": "gru",
            "width": 32,
            "dropout": 0.15,
            "lr": 0.0005,
            "weight_decay": 0.001,
        },
    ]
    suggest_params = suggest
    if protocol == "run007":
        from .extended_training import anchors, suggest as extended_suggest

        initial, suggest_params = anchors(), extended_suggest
    for candidate in initial:
        study.enqueue_trial(candidate)

    def objective(trial):
        report = evaluate(
            dev,
            bars,
            split,
            suggest_params(trial),
            42,
            max_epochs,
            cfg,
            device,
            budget,
            root / "trials" / f"{trial.number:03}",
        )
        trial.set_user_attr("params", report["params"])
        return report["objective"]

    try:
        write_json(root / "status.json", {"status": "running", "started_at": utcnow()})
        for trial_index in range(n_trials):
            if time.monotonic() >= search_end and trial_index >= len(initial):
                break
            budget.check()
            study.optimize(objective, n_trials=1, catch=(ValueError,))
            study.trials_dataframe().to_csv(root / "trials.csv", index=False)
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len({t.user_attrs["params"]["family"] for t in completed}) < 4:
            raise ValueError(
                "Not all four model families completed; keep partial evidence"
            )
        leaders = sorted(completed, key=lambda t: (t.value, t.number))[
            : (2 if protocol == "run006" else 3)
        ]
        shortlist = []
        for trial in leaders:
            scores = []
            seed_reports = []
            for seed in [42, 123, 2026]:
                folder = root / "stability" / f"{trial.number:03}-seed-{seed}"
                if seed == 42:
                    report = json.loads(
                        (
                            root / "trials" / f"{trial.number:03}" / "report.json"
                        ).read_text()
                    )
                    write_json(folder / "report.json", report)
                    write_parquet(
                        folder / "predictions.parquet",
                        pd.read_parquet(
                            root
                            / "trials"
                            / f"{trial.number:03}"
                            / "predictions.parquet"
                        ),
                    )
                else:
                    report = evaluate(
                        dev,
                        bars,
                        split,
                        trial.user_attrs["params"],
                        seed,
                        max_epochs,
                        cfg,
                        device,
                        budget,
                        folder,
                    )
                scores.append(report["objective"])
                seed_reports.append(report)
            from .extended_training import validation_gate

            shortlist.append(
                {
                    "trial": trial.number,
                    "params": trial.user_attrs["params"],
                    "mean_objective": float(np.mean(scores)),
                    "std_objective": float(np.std(scores)),
                    "validation_gate": validation_gate(seed_reports),
                }
            )
        selected = min(shortlist, key=lambda s: (s["mean_objective"], s["trial"]))
        pred = pd.read_parquet(
            root / "trials" / f"{selected['trial']:03}" / "predictions.parquet"
        )
        selection = {
            **selected,
            "shortlist": shortlist,
            "seed": 42,
            "frozen_at": utcnow(),
            "validation": forecast_metrics(pred),
        }
        write_json(root / "selection.json", selection)
        budget.check()
        fitted = fit_model(
            dev,
            selected["params"],
            42,
            max_epochs,
            cfg,
            device,
            budget,
            root / "final-fit",
        )
        fitted.setdefault("training_rows", len(dev))
        with (root / "retrospective-test.started.json").open("x") as stream:
            json.dump(
                {
                    "selection_sha256": sha256(root / "selection.json"),
                    "started_at": utcnow(),
                },
                stream,
            )
        _, test = fold_data(features, split["test_start"], split["end"])
        context = with_context(features, test, split["test_start"])
        values = predict(fitted, context)
        mask = (context.is_validation & context.eligible_window).to_numpy()
        if not np.isfinite(values[mask]).all():
            raise ValueError("Missing test forecasts")
        rows = context[mask].copy()
        rows["prediction"] = values[mask]
        write_parquet(
            root / "retrospective_predictions.parquet",
            rows[["symbol", "decision_time", "target_return", "prediction"]],
        )
        metrics = []
        for name, p in [
            ("selected", rows.prediction),
            ("zero_return", np.zeros(len(rows))),
            ("naive_ret4", np.expm1(rows.ret_4)),
        ]:
            part = rows.assign(prediction=p)
            metrics.append({"model": name, **forecast_metrics(part)})
        write_json(root / "retrospective_metrics.json", metrics)
        asset_metrics = [
            {"symbol": symbol, **forecast_metrics(group)}
            for symbol, group in rows.groupby("symbol")
        ]
        write_json(root / "retrospective_asset_metrics.json", asset_metrics)
        curve, trades, strategy = evaluate_strategy(
            bars, rows, rows.prediction, 0.006, split["test_start"], split["end"], cfg
        )
        empty = pd.DataFrame(columns=["symbol", "decision_time", "go_long"])
        hold, _, hold_metrics = simulate(
            bars, empty, split["test_start"], split["end"], cfg, buy_hold=True
        )
        ema, _, ema_metrics = evaluate_strategy(
            bars, rows, rows.ema_gap, 0, split["test_start"], split["end"], cfg
        )
        write_parquet(
            root / "retrospective_equity.parquet",
            pd.DataFrame(
                {
                    "timestamp": curve.index,
                    "model": curve.values,
                    "buy_hold": hold.values,
                    "ema": ema.values,
                }
            ),
        )
        write_parquet(root / "retrospective_trades.parquet", trades)
        write_json(
            root / "retrospective_strategies.json",
            {"model": strategy, "buy_hold": hold_metrics, "ema": ema_metrics},
        )
        budget.check()
        deployment = export_model(
            root,
            fitted,
            cfg,
            manifest,
            selection,
            metrics[0],
            context,
            run_name=protocol,
        )
        write_json(
            root / "status.json",
            {
                "status": "completed",
                "completed_at": utcnow(),
                "deployment": deployment,
                "test_is_independent": False,
                "promotion": "paper_only_no_real_money",
                "validation_gate": selection.get("validation_gate"),
            },
        )
        return selection
    except BaseException as error:
        study.trials_dataframe().to_csv(root / "trials.csv", index=False)
        write_json(
            root / "status.json",
            {"status": "incomplete", "error": str(error), "at": utcnow()},
        )
        (root / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", choices=["run006", "run007"], default="run006")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args.root)
    else:
        run(
            args.root,
            args.trials,
            args.epochs,
            args.seconds,
            args.device,
            args.protocol,
        )
