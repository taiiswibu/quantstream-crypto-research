"""Small causal TCN: training and final refit, usable on Kaggle or local CPU."""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .common import Config, sha256, write_json, write_parquet, utcnow
from .features import FEATURES
from .backtest import evaluate_strategy


class CausalBlock(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel_size=3, dilation=dilation)
        self.activation = nn.GELU()
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.padding, 0)))
        return self.activation(y) + self.skip(x)


class TCN(nn.Module):
    def __init__(self, inputs=len(FEATURES), width=24):
        super().__init__()
        self.blocks = nn.Sequential(CausalBlock(inputs, width, 1), CausalBlock(width, width, 2),
                                    CausalBlock(width, width, 4), CausalBlock(width, width, 8))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.blocks(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class Windows(Dataset):
    def __init__(self, frame, mean, std, lookback, labelled=True):
        self.items = []
        self.groups = []
        self.lookback = lookback
        for _, g in frame.groupby(["symbol", "segment"], sort=True):
            g = g.sort_values("decision_time")
            x = ((g[FEATURES].to_numpy(np.float32) - mean) / std).astype(np.float32)
            y = g.target_return.to_numpy(np.float32)
            ids = g.index.to_numpy()
            times = g.decision_time.to_numpy()
            group_index = len(self.groups)
            self.groups.append((x, y))
            for end in range(lookback - 1, len(g)):
                contiguous = pd.Timestamp(times[end]) - pd.Timestamp(times[end - lookback + 1]) == pd.Timedelta(hours=lookback - 1)
                if contiguous and (not labelled or np.isfinite(y[end])):
                    self.items.append((group_index, end, int(ids[end])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        group, end, row_id = self.items[i]
        x, y = self.groups[group]
        target = float(y[end]) * 100 if np.isfinite(y[end]) else 0.0
        return torch.from_numpy(x[end - self.lookback + 1:end + 1]), torch.tensor(target), row_id


def train_loop(frame, mean, std, cfg, epochs, device, validation=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    model = TCN().to(device)
    data = Windows(frame, mean, std, cfg.lookback)
    if not len(data):
        raise ValueError("No contiguous TCN windows")
    loader = DataLoader(data, batch_size=256, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg.seed))
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_epoch, best_loss, history = None, 1, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        record = {"epoch": epoch, "train_huber": loss_sum / seen}
        if validation is not None:
            pred = predict_tcn(model, validation, mean, std, cfg.lookback, device)
            mask = np.isfinite(pred) & validation.target_return.notna().to_numpy()
            val_loss = float(np.mean((pred[mask] - validation.target_return.to_numpy()[mask]) ** 2))
            record["validation_mse"] = val_loss
            if val_loss < best_loss:
                best_loss, best_epoch, best = val_loss, epoch, copy.deepcopy(model.state_dict())
        history.append(record)
        print(record, flush=True)
    if best is not None:
        model.load_state_dict(best)
    return model, best_epoch, history


def predict_tcn(model, frame, mean, std, lookback, device="cpu"):
    model.eval()
    output = np.full(len(frame), np.nan, dtype=float)
    positions = {int(index): i for i, index in enumerate(frame.index)}
    loader = DataLoader(Windows(frame, np.asarray(mean), np.asarray(std), lookback, labelled=False), batch_size=512)
    with torch.no_grad():
        for x, _, ids in loader:
            pred = model(x.to(device)).cpu().numpy() / 100
            for i, value in zip(ids.tolist(), pred):
                output[positions[i]] = value
    return output


def predict_tcn_file(path, frame):
    # weights_only blocks arbitrary pickle classes in imported PyTorch artifacts.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TCN()
    model.load_state_dict(payload["state_dict"])
    return predict_tcn(model, frame, payload["mean"], payload["std"], payload["lookback"])


def train_snapshot(snapshot, output, epochs=8, device="auto"):
    snapshot, output = Path(snapshot), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    shutil.copy2(snapshot / "manifest.json", output / "training_snapshot_manifest.json")
    for name, expected in manifest["files"].items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    d = manifest["config"]
    d["symbols"] = tuple(d["symbols"])
    cfg = Config(**d)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = pd.read_parquet(snapshot / "development.parquet")
    bars = pd.read_parquet(snapshot / "bars.parquet")
    from .features import fold_data
    predictions, history, best_epochs = [], [], []
    for fold, bounds in enumerate(manifest["split"]["folds"]):
        train, val = fold_data(dev, bounds["validation_start"], bounds["validation_end"])
        mean = train[FEATURES].mean().to_numpy(np.float32)
        std = train[FEATURES].std().clip(lower=1e-6).to_numpy(np.float32)
        validation_start = pd.Timestamp(bounds["validation_start"])
        # Context may contain purged rows as FEATURES, but their labels are hidden.
        past = dev[dev.decision_time < validation_start].groupby("symbol").tail(cfg.lookback)
        context = pd.concat([past, val]).drop_duplicates(["symbol", "decision_time"])
        context = context.sort_values(["decision_time", "symbol"]).reset_index(drop=True)
        context["is_validation"] = context.decision_time.ge(validation_start)
        context.loc[~context.is_validation, "target_return"] = np.nan
        print(f"TCN fold {fold + 1}/{len(manifest['split']['folds'])}", flush=True)
        model, best_epoch, fold_history = train_loop(train.reset_index(drop=True), mean, std, cfg, epochs, device, context)
        pred = predict_tcn(model, context, mean, std, cfg.lookback, device)
        mask = context.is_validation.to_numpy() & np.isfinite(pred)
        part = context[mask].copy()
        part["prediction"], part["fold"] = pred[mask], fold
        predictions.append(part)
        best_epochs.append(best_epoch)
        history.append({"fold": fold, "best_epoch": best_epoch, "epochs": fold_history})
    val = pd.concat(predictions, ignore_index=True)
    options = []
    for threshold in [0.0, 0.0015, 0.003, 0.006]:
        summaries = []
        for fold, bounds in enumerate(manifest["split"]["folds"]):
            part = val[val.fold == fold]
            _, _, summary = evaluate_strategy(bars, part, part.prediction, threshold, bounds["validation_start"], bounds["validation_end"], cfg)
            summaries.append(summary)
        count = sum(s["trades"] for s in summaries)
        mean_sharpe = float(np.mean([s["sharpe_hourly_estimate"] for s in summaries]))
        options.append({"threshold": threshold, "mean_validation_sharpe": mean_sharpe,
                        "validation_trades": count, "eligible": count >= cfg.minimum_trades,
                        "mean_validation_return": float(np.mean([s["total_return"] for s in summaries])),
                        "worst_validation_drawdown": min(s["max_drawdown"] for s in summaries),
                        "last_fold": summaries[-1], "folds": summaries,
                        "selection_score": mean_sharpe if count >= cfg.minimum_trades else -1e6 + count})
    chosen = max(options, key=lambda x: (x["selection_score"], x["mean_validation_return"]))
    write_parquet(output / "validation_predictions.parquet", val[["symbol", "decision_time", "target_return", "prediction", "fold"]])
    # Hyperparameters/epochs are now frozen. Refit the chosen architecture on development only.
    mean = dev[FEATURES].mean().to_numpy(np.float32)
    std = dev[FEATURES].std().clip(lower=1e-6).to_numpy(np.float32)
    best_epoch = int(np.median(best_epochs))
    final, _, final_history = train_loop(dev.reset_index(drop=True), mean, std, cfg, best_epoch, device)
    torch.save({"state_dict": {k: v.cpu() for k, v in final.state_dict().items()},
                "mean": mean.tolist(), "std": std.tolist(), "lookback": cfg.lookback}, output / "model.pt")
    metadata = {"created_at": utcnow(), "type": "tcn", "model": "tcn", "threshold": chosen["threshold"],
                "source_control": manifest.get("source_control", {"status": "not_recorded_in_legacy_snapshot"}),
                "sha256": sha256(output / "model.pt"), "config": cfg.to_dict(), "features": FEATURES,
                "features_sha256": manifest["full_features_sha256"], "split": manifest["split"],
                "selection": chosen, "best_epoch": best_epoch, "device_used": device,
                "training_rows": len(dev), "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
                "contains_final_holdout": False, "history": history, "final_history": final_history}
    write_json(output / "tcn_metadata.json", metadata)
    return metadata


def register_tcn(root, folder):
    root, folder = Path(root), Path(folder)
    metadata = json.loads((folder / "tcn_metadata.json").read_text())
    # A later model can be used for forward paper trading, but final_evaluate will refuse
    # to reuse the already-opened holdout for it.
    if metadata["features_sha256"] != sha256(root / "data/processed/features.parquet"):
        raise ValueError("Imported model uses a different dataset snapshot")
    if sha256(folder / "model.pt") != metadata["sha256"]:
        raise ValueError("Imported checkpoint hash mismatch")
    payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    TCN().load_state_dict(payload["state_dict"])
    destination = root / "artifacts/tcn"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / "model.pt", destination / "model.pt")
    metadata["path"] = "artifacts/tcn/model.pt"
    write_json(destination / "tcn_metadata.json", metadata)
    # Explicit registration chooses the champion. Compare validation reports first.
    write_json(root / "artifacts/champion.json", metadata)
    return metadata
