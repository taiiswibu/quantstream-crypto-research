"""Run 007: controlled training-depth, loss and recency comparisons."""

from pathlib import Path
import json
import shutil
import zipfile

import numpy as np
import pandas as pd

from .common import sha256, write_json


def find_parent_snapshot(input_root, cache):
    """Inspect only attached Run 006 snapshots; do not extract weights or arbitrary paths."""
    input_root, cache = Path(input_root), Path(cache)
    candidates = (
        sorted(input_root.rglob("dataset_manifest.json")) if input_root.exists() else []
    )
    candidates = [
        p
        for p in candidates
        if (p.parent / "data/processed/features.parquet").is_file()
    ]
    if not candidates and input_root.exists():
        archives = sorted(input_root.rglob("QuantStream_Run006_Output*.zip"))
        for index, archive in enumerate(archives):
            target = cache / str(index)
            with zipfile.ZipFile(archive) as zipped:
                names = [
                    name
                    for name in zipped.namelist()
                    if Path(name).name == "dataset_manifest.json"
                ]
                if len(names) != 1:
                    raise ValueError("Ambiguous parent manifest in attached ZIP")
                prefix = str(Path(names[0]).parent)
                prefix = "" if prefix == "." else prefix + "/"
                allowed = [
                    "dataset_manifest.json",
                    "dataset_plan.json",
                    "split.json",
                    "data/processed/features.parquet",
                    "data/processed/bars.parquet",
                    "data/processed/source_manifest.json",
                    "data/processed/data_quality.json",
                ]
                for name in allowed:
                    member = prefix + name
                    if member not in zipped.namelist():
                        continue
                    info = zipped.getinfo(member)
                    if info.file_size > 500 * 1024 * 1024:
                        raise ValueError("Attached dataset member exceeds 500 MiB")
                    destination = target / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
            candidates.append(target / "dataset_manifest.json")
    valid = []
    for path in candidates:
        manifest = json.loads(path.read_text())
        if manifest.get("research_protocol") == "run007-depth-v1":
            continue
        if (
            manifest.get("synthetic") is False
            and manifest.get("frozen_end") == "2026-11-14 00:00:00+00:00"
        ):
            valid.append(path)
    if len({sha256(path) for path in valid}) > 1:
        raise ValueError("Multiple different parent snapshots; select one explicitly")
    return valid[0].parent if valid else None


def prepare_dataset(root, parent=None):
    """Reuse a verified snapshot when supplied; never reuse fitted weights."""
    from .workflow import prepare_dataset as collect

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("research_protocol") == "run007-depth-v1":
            for name, expected in existing["files"].items():
                if sha256(root / name) != expected:
                    raise ValueError("Dataset changed: " + name)
            return existing
    if parent is None:
        base = collect(root)
        parent_sha = sha256(manifest_path)
    else:
        parent = Path(parent)
        base = json.loads((parent / "dataset_manifest.json").read_text())
        parent_sha = sha256(parent / "dataset_manifest.json")
        allowed = {
            "data/processed/bars.parquet",
            "data/processed/features.parquet",
            "split.json",
        }
        if set(base["files"]) != allowed:
            raise ValueError("Unexpected paths in parent dataset manifest")
        for name, expected in base["files"].items():
            if sha256(parent / name) != expected:
                raise ValueError("Parent dataset changed: " + name)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != expected:
                raise ValueError("Destination dataset differs; preserve the run")
            shutil.copy2(parent / name, destination)
        for name in (
            "dataset_plan.json",
            "data/processed/data_quality.json",
            "data/processed/source_manifest.json",
        ):
            if (parent / name).is_file():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parent / name, destination)
    data = pd.read_parquet(root / "data/processed/features.parquet")
    test_start = pd.Timestamp(base["split"]["test_start"])
    dev = data[data.label_end_time.lt(test_start) & data.target_return.notna()]
    times = np.sort(dev.decision_time.unique())
    if len(times) < 300:
        raise ValueError("Not enough development timestamps for three folds")
    edges = [
        pd.Timestamp(times[int(len(times) * fraction)]) for fraction in (0.4, 0.6, 0.8)
    ]
    edges.append(test_start)
    split = {
        **base["split"],
        "tuning_folds": [
            {"validation_start": str(a), "validation_end": str(b)}
            for a, b in zip(edges[:-1], edges[1:])
        ],
    }
    write_json(root / "split.json", split)
    manifest = {
        **base,
        "research_protocol": "run007-depth-v1",
        "split": split,
        "parent_dataset_manifest_sha256": parent_sha,
        "files": {**base["files"], "split.json": sha256(root / "split.json")},
    }
    write_json(root / "reference/base_dataset_manifest.json", base)
    write_json(manifest_path, manifest)
    return manifest


def anchors():
    candidates = [
        {"family": "ridge", "alpha": 10.0, "train_window_days": 0},
        {
            "family": "xgboost",
            "max_depth": 3,
            "learning_rate": 0.04,
            "n_estimators": 120,
            "reg_lambda": 10.0,
            "train_window_days": 0,
        },
    ]
    for family in ("tcn", "gru"):
        common = {
            "family": family,
            "width": 32,
            "dropout": 0.15,
            "lr": 0.0005,
            "weight_decay": 0.001,
            "train_window_days": 0,
            "loss": "huber",
            "lr_schedule": "constant",
            "epoch_cap": 60,
        }
        if family == "tcn":
            common["activation"] = "silu"
        candidates.extend(
            [
                {**common, "epoch_cap": 12},
                common,
                {**common, "lr_schedule": "cosine"},
                {**common, "lr_schedule": "cosine", "loss": "mse"},
                {**common, "lr_schedule": "cosine", "train_window_days": 730},
                {
                    **common,
                    "lr_schedule": "cosine",
                    "loss": "mse",
                    "train_window_days": 730,
                },
            ]
        )
    return candidates


def suggest(trial):
    family = trial.suggest_categorical("family", ["ridge", "xgboost", "tcn", "gru"])
    params = {
        "family": family,
        "train_window_days": trial.suggest_categorical(
            "train_window_days", [0, 365, 730]
        ),
    }
    if family == "ridge":
        return {**params, "alpha": trial.suggest_float("alpha", 0.01, 1000, log=True)}
    if family == "xgboost":
        return {
            **params,
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [120, 300, 600]),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True),
        }
    params.update(
        {
            "lookback": 48,
            "batch_size": 256,
            "width": trial.suggest_categorical("width", [16, 32, 64]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.15, 0.3]),
            "lr": trial.suggest_float("lr", 3e-5, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "loss": trial.suggest_categorical("loss", ["huber", "mse"]),
            "lr_schedule": trial.suggest_categorical(
                "lr_schedule", ["constant", "cosine"]
            ),
            "epoch_cap": trial.suggest_categorical("epoch_cap", [12, 60]),
        }
    )
    params["patience"] = 8
    params["schedule_epochs"] = params["epoch_cap"]
    if family == "tcn":
        params["activation"] = trial.suggest_categorical(
            "activation", ["gelu", "relu", "silu"]
        )
    return params


def validation_gate(reports):
    """Freeze before the old retrospective block is read; never a real-money gate."""
    ratios = np.array(
        [
            [fold["prediction"]["mse_ratio_to_zero"] for fold in report["folds"]]
            for report in reports
        ]
    )
    mean = float(ratios.mean())
    worst = float(ratios.mean(axis=0).max())
    return {
        "mean_seed_fold_mse_ratio": mean,
        "worst_fold_mean_mse_ratio": worst,
        "beats_zero_on_mean_and_every_fold": bool(mean < 1 and worst < 1),
        "usage": "research_paper_only",
        "test_is_independent": False,
    }


def figures(root):
    import matplotlib.pyplot as plt
    from .workflow_plots import show_saved

    root = Path(root)
    trials = pd.read_csv(root / "trials.csv")
    labels = [
        "12 constant Huber",
        "60 constant Huber",
        "60 cosine Huber",
        "60 cosine MSE",
        "60 cosine Huber 730d",
        "60 cosine MSE 730d",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for family, ax in zip(("tcn", "gru"), axes):
        rows = trials[trials.params_family.eq(family) & trials.number.lt(14)]
        ax.bar(range(len(rows)), rows.value, color="#0f766e")
        ax.set(
            xticks=range(len(rows)),
            xticklabels=labels[: len(rows)],
            title=family.upper() + " · fixed seed 42 / three folds",
            ylabel="Mean MSE / zero-return MSE",
        )
        ax.tick_params(axis="x", rotation=55)
        ax.axhline(1, color="#b91c1c", ls="--")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Controlled ablations · longer training may help or hurt")
    plt.tight_layout()
    show_saved(root, "13-training-ablations")
    history = []
    for family in ("tcn", "gru"):
        rows = trials[
            trials.params_family.eq(family) & trials.state.eq("COMPLETE")
        ].dropna(subset=["value"])
        if len(rows):
            number = int(rows.sort_values("value").iloc[0].number)
            history.extend(
                (root / "trials" / f"{number:03}").glob("fold-*/early-stop/history.csv")
            )
    fig, ax = plt.subplots(figsize=(11, 4))
    for file in history:
        frame = pd.read_csv(file)
        if "learning_rate" in frame:
            ax.plot(
                frame.epoch,
                frame.learning_rate,
                label="/".join(file.relative_to(root / "trials").parts[:2]),
            )
    ax.set(
        title="Recorded learning rate used in each epoch",
        xlabel="Epoch",
        ylabel="Learning rate",
        yscale="log",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    show_saved(root, "14-learning-rates")
