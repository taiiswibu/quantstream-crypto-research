"""Chronos-2 LoRA experiment. Development-only, no automatic promotion."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Config, sha256, write_json, write_parquet, utcnow


def chronos_experiment(snapshot, output, steps=200, max_windows=512, device="auto", model_id="amazon/chronos-2", revision=None):
    import inspect
    import torch
    import peft  # Explicit import: missing PEFT must not silently fall back to full fine-tuning.
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi
    snapshot, output = Path(snapshot), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    shutil.copy2(snapshot / "manifest.json", output / "training_snapshot_manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    d = manifest["config"]
    d["symbols"] = tuple(d["symbols"])
    cfg = Config(**d)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CPU fine-tuning can be slow. Kaggle GPU is recommended; set --steps 2 for a smoke run.", flush=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    resolved_revision = revision or HfApi().model_info(model_id, timeout=30).sha
    pipe = Chronos2Pipeline.from_pretrained(model_id, revision=resolved_revision, device_map=device,
                                            torch_dtype=torch.float32)
    if "finetune_mode" not in inspect.signature(pipe.fit).parameters:
        raise RuntimeError("Installed Chronos lacks LoRA API. Install requirements-kaggle.txt.")
    bars = pd.read_parquet(snapshot / "bars.parquet")
    val = pd.read_parquet(snapshot / "validation.parquet")
    start = pd.Timestamp(manifest["split"]["folds"][-1]["validation_start"])
    training_inputs = []
    contexts, row_ids = [], []
    context_length = 128
    for symbol, data in bars.groupby("symbol"):
        data = data.sort_values("timestamp").reset_index(drop=True)
        segments = data.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _, segment in data.groupby(segments):
            train = segment[segment.timestamp + pd.Timedelta(hours=1) < start]
            if len(train) >= context_length + cfg.horizon + cfg.execution_lag_bars:
                training_inputs.append(np.log(train.close.to_numpy(np.float32)))
        eligible = val[val.symbol == symbol]
        # A fixed evenly-spaced sample is used for the exploratory before/after comparison.
        if max_windows and len(eligible) > max_windows // len(cfg.symbols):
            positions = np.linspace(0, len(eligible) - 1, max_windows // len(cfg.symbols), dtype=int)
            eligible = eligible.iloc[positions]
        for row_id, row in eligible.iterrows():
            past = data[data.timestamp < row.decision_time].tail(context_length)
            if len(past) < context_length or past.timestamp.diff().dropna().ne(pd.Timedelta(hours=1)).any():
                continue
            contexts.append(np.log(past.close.to_numpy(np.float32)))
            row_ids.append(int(row_id))
    if not training_inputs or not contexts:
        raise ValueError("Insufficient contiguous training/evaluation data")
    horizon = cfg.horizon + cfg.execution_lag_bars

    def forecast(model):
        _, points = model.predict_quantiles(contexts, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
                                            context_length=context_length, batch_size=16)
        # Entry open is unknown at decision time. Its price is approximated by the forecast
        # of the immediately preceding close; this is NOT the realised future entry price.
        forecasts = [p.detach().float().cpu().numpy().reshape(-1, horizon)[0] for p in points]
        return np.array([np.expm1(p[-1] - p[cfg.execution_lag_bars - 1]) for p in forecasts])

    before = forecast(pipe)
    tuned = pipe.fit(inputs=training_inputs, prediction_length=horizon, finetune_mode="lora",
                     context_length=context_length, learning_rate=1e-5, num_steps=steps,
                     batch_size=8, min_past=64, output_dir=output / "trainer", seed=cfg.seed,
                     report_to="none", dataloader_num_workers=0, logging_steps=max(1, steps // 5))
    tuned.save_pretrained(output / "adapter")
    after = forecast(tuned)
    scored = val.loc[row_ids, ["symbol", "decision_time", "target_return"]].copy()
    scored["zero_shot_return"], scored["finetuned_return"] = before, after
    write_parquet(output / "chronos_validation.parquet", scored)
    truth = scored.target_return.to_numpy()
    report = {"created_at": utcnow(), "model_id": model_id, "resolved_revision": resolved_revision,
              "steps": steps, "mode": "lora", "device": device, "context_length": context_length,
              "peft_version": peft.__version__,
              "evaluation_windows": len(scored), "contains_final_holdout": False,
              "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
              "zero_shot_mae": float(np.mean(np.abs(before - truth))),
              "finetuned_mae": float(np.mean(np.abs(after - truth))),
              "zero_shot_directional_accuracy": float(np.mean((before > 0) == (truth > 0))),
              "finetuned_directional_accuracy": float(np.mean((after > 0) == (truth > 0))),
              "pretraining_data_cutoff_verified": False,
              "limitations": ["Exploratory sampled validation, not final strategy performance.",
                              "Pretraining overlap with financial data is unverified.",
                              "Forecasted pre-entry close approximates future entry open.",
                              "No automatic deployment of this experimental adapter."]}
    write_json(output / "chronos_report.json", report)
    return report
