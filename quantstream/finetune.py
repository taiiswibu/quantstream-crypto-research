"""Chronos-2 LoRA experiment. Development-only, no automatic promotion."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Config, sha256, write_json, write_parquet, utcnow


def chronos_inputs(snapshot, context_length=128, max_windows=512):
    """Share the exact Run 003 sampling/order with later diagnostics."""
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("contains_final_holdout") is not False:
        raise ValueError("Development-only input required")
    for name, digest in manifest["files"].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    d = manifest["config"]
    d["symbols"] = tuple(d["symbols"])
    cfg = Config(**d)
    bars = pd.read_parquet(snapshot / "bars.parquet")
    val = pd.read_parquet(snapshot / "validation.parquet")
    start = pd.Timestamp(manifest["split"]["folds"][-1]["validation_start"])
    training_inputs = []
    contexts, row_ids = [], []
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
    scored = val.loc[row_ids, ["symbol", "decision_time", "target_return"]].copy()
    if scored.duplicated(["symbol", "decision_time"]).any() or not np.isfinite(scored.target_return).all():
        raise ValueError("Invalid evaluation sample keys/targets")
    end = pd.Timestamp(manifest["split"]["test_start"])
    selected = val.loc[row_ids]
    if (selected.decision_time >= end).any() or (selected.label_end_time >= end).any():
        raise ValueError("Final holdout leaked into validation")
    return manifest, cfg, training_inputs, contexts, scored


def chronos_experiment(snapshot, output, steps=200, max_windows=512, device="auto", model_id="amazon/chronos-2", revision=None,
                       learning_rate=1e-5, batch_size=8, context_length=128, seed=42,
                       lora_config=None, callbacks=None, logging_steps=None):
    import inspect
    import torch
    import peft  # Explicit import: missing PEFT must not silently fall back to full fine-tuning.
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi
    if steps < 1 or learning_rate <= 0 or batch_size < 1 or context_length < 64 or max_windows < 2:
        raise ValueError("Invalid fine-tuning budget or window configuration")
    snapshot, output = Path(snapshot), Path(output)
    if (output / "chronos_report.json").exists():
        raise FileExistsError("Preserve completed experiments; choose a new experiment output name.")
    output.mkdir(parents=True, exist_ok=True)
    manifest, cfg, training_inputs, contexts, scored = chronos_inputs(snapshot, context_length, max_windows)
    shutil.copy2(snapshot / "manifest.json", output / "training_snapshot_manifest.json")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CPU fine-tuning can be slow. Kaggle GPU is recommended; set --steps 2 for a smoke run.", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg.threads)
    resolved_revision = revision or HfApi().model_info(model_id, timeout=30).sha
    pipe = Chronos2Pipeline.from_pretrained(model_id, revision=resolved_revision, device_map=device,
                                            torch_dtype=torch.float32)
    if "finetune_mode" not in inspect.signature(pipe.fit).parameters:
        raise RuntimeError("Installed Chronos lacks LoRA API. Install requirements-kaggle.txt.")
    horizon = cfg.horizon + cfg.execution_lag_bars

    def forecast(model):
        # no_grad alone does not disable dropout after Trainer.train().
        model.model.eval()
        _, points = model.predict_quantiles(contexts, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
                                            context_length=context_length, batch_size=16)
        # Entry open is unknown at decision time. Its price is approximated by the forecast
        # of the immediately preceding close; this is NOT the realised future entry price.
        forecasts = [p.detach().float().cpu().numpy().reshape(-1, horizon)[0] for p in points]
        return np.array([np.expm1(p[-1] - p[cfg.execution_lag_bars - 1]) for p in forecasts])

    before = forecast(pipe)
    tuned = pipe.fit(inputs=training_inputs, prediction_length=horizon, finetune_mode="lora",
                     context_length=context_length, learning_rate=learning_rate, num_steps=steps,
                     batch_size=batch_size, min_past=64, output_dir=output / "trainer", seed=seed,
                     lora_config=lora_config, callbacks=callbacks,
                     report_to="none", dataloader_num_workers=0,
                     logging_steps=logging_steps or max(1, steps // 5))
    tuned.save_pretrained(output / "adapter")
    after = forecast(tuned)
    scored["zero_shot_return"], scored["finetuned_return"] = before, after
    write_parquet(output / "chronos_validation.parquet", scored)
    truth = scored.target_return.to_numpy()
    report = {"created_at": utcnow(), "model_id": model_id, "resolved_revision": resolved_revision,
              "source_control": manifest.get("source_control", {"status": "not_recorded_in_legacy_snapshot"}),
              "steps": steps, "mode": "lora", "device": device, "context_length": context_length,
              "learning_rate": learning_rate, "batch_size": batch_size, "seed": seed,
              "inference_mode": "eval",
              "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
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
