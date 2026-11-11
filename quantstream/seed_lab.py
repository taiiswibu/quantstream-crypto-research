"""Run 004: fixed-budget seed diagnostic, with a separate Kaggle preflight gate.

No strategy selection, final holdout access, model registration or order placement.
Optional training dependencies are imported only inside the training paths.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import traceback
import zipfile

import numpy as np
import pandas as pd

from .common import git_provenance, sha256, utcnow, write_json
from .finetune import chronos_experiment, chronos_inputs

MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT_SHA = "c8e0112916e6cea733338ead4412d6e9553e4c5844a2d200d1fd38e31c747fd0"
SEEDS = (42, 123, 2026)
STEPS, CONTEXT, BATCH, WINDOWS = 200, 128, 8, 512
PROTOCOL = "run-004-seed-stability-v2"
RUN003_REFERENCE = Path(__file__).resolve().parents[1] / "reports/experiments/run-003-lora-lab/training_snapshot_manifest.json"
EXPORT_METADATA = {"created_at", "source_control", "code_files"}
TARGETS = ["self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
           "output_patch_embedding.output_layer"]


def normalized(value):
    """Convert PEFT enums/sets to stable, JSON-serializable configuration."""
    def default(item):
        if isinstance(item, set):
            return sorted(item)
        if hasattr(item, "value"):
            return item.value
        raise TypeError(type(item).__name__)
    return json.loads(json.dumps(value, default=default, sort_keys=True, allow_nan=False))


def resolved_lora():
    from peft import LoraConfig
    # Chronos 2.3.2 defaults, made explicit; remaining PEFT defaults are recorded too.
    return normalized(LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS).to_dict())


def runtime():
    import torch
    packages = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "packages": dict(sorted(packages.items())),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "torchao_installed": "torchao" in packages}


def source_identity():
    root = Path(__file__).resolve().parents[1]
    source = git_provenance(root)
    if source["status"] != "available" or source["dirty"] is not False:
        raise ValueError("Run 004 requires a clean Git checkout at the pinned source SHA")
    source["files"] = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "quantstream").glob("*.py"))}
    source["protocol_sha256"] = sha256(root / "docs/experiments/run-004-seed-stability-protocol.md")
    # A detached checkout is expected on Kaggle; branch names are not runtime identity.
    source.pop("branch")
    return source


def sample_sha256(frame):
    rows = frame[["symbol", "decision_time", "target_return"]].copy()
    rows["decision_time"] = pd.to_datetime(rows.decision_time, utc=True).astype(str)
    return hashlib.sha256(rows.to_json(orient="split", index=False, double_precision=15).encode()).hexdigest()


def verify_run003_snapshot(snapshot):
    """Verify scientific identity, retaining each export's original manifest/provenance.

    Old exports can carry different dates/code metadata even when the four Parquet
    files and every data/split/config field are identical. Never rewrite a manifest
    or accept files merely because their self-reported checksums match each other.
    """
    snapshot = Path(snapshot)
    if sha256(RUN003_REFERENCE) != SNAPSHOT_SHA:
        raise ValueError("Run 003 reference manifest checksum mismatch in source checkout")
    reference = json.loads(RUN003_REFERENCE.read_text())
    path = snapshot / "manifest.json"
    current = json.loads(path.read_text())
    if current.get("contains_final_holdout") is not False:
        raise ValueError("Development-only input required")
    reference_contract = {k: v for k, v in reference.items() if k not in EXPORT_METADATA}
    current_contract = {k: v for k, v in current.items() if k not in EXPORT_METADATA}
    changed = [key for key in sorted(set(reference_contract) | set(current_contract))
               if reference_contract.get(key) != current_contract.get(key)
               or (key in reference_contract) != (key in current_contract)]
    if changed:
        raise ValueError("Snapshot data contract differs from Run 003: " + ", ".join(changed))
    actual_files = {}
    for name, expected in reference["files"].items():
        actual_files[name] = sha256(snapshot / name)
        if actual_files[name] != expected:
            raise ValueError("Data file differs from Run 003: " + name)
    actual_manifest = sha256(path)
    return {"manifest_sha256": actual_manifest, "reference_manifest_sha256": SNAPSHOT_SHA,
            "manifest_relation": "exact" if actual_manifest == SNAPSHOT_SHA else "data_equivalent_metadata_diff",
            "metadata_differences": sorted(key for key in EXPORT_METADATA if current.get(key) != reference.get(key)),
            "data_file_hashes": actual_files}


def inputs_identity(snapshot):
    snapshot = Path(snapshot)
    verified = verify_run003_snapshot(snapshot)
    # Check both the development contract and the exact historical LoRA sampling path.
    from .search import load_snapshot
    load_snapshot(snapshot)
    _, cfg, training, contexts, scored = chronos_inputs(snapshot, CONTEXT, WINDOWS)
    if cfg.symbols != ("BTCUSDT", "ETHUSDT") or cfg.horizon != 4 or cfg.execution_lag_bars != 1:
        raise ValueError("Run 004 symbol/target contract changed")
    if len(scored) != WINDOWS or scored.groupby("symbol").size().to_dict() != {"BTCUSDT": 256, "ETHUSDT": 256}:
        raise ValueError("Expected the same 512 Run 003 windows (256 per asset)")
    train_hash = hashlib.sha256()
    for series in training:
        train_hash.update(str(series.shape).encode())
        train_hash.update(np.asarray(series, dtype="<f4").tobytes())
    context_hash = hashlib.sha256(np.asarray(contexts, dtype="<f4").tobytes()).hexdigest()
    return {**verified, "sample_sha256": sample_sha256(scored),
            "contexts_sha256": context_hash, "training_sha256": train_hash.hexdigest(),
            "training_series": len(training), "training_lengths": [len(s) for s in training],
            "rows": len(scored), "horizon": cfg.horizon + cfg.execution_lag_bars}


def trainer_evidence(folder):
    """Persist actual step counts, resolved TrainingArguments and loss logs on every step."""
    from transformers import TrainerCallback
    folder = Path(folder)

    class Evidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            write_json(folder / "training_args.json", normalized(args.to_dict()))
            self.save(state)

        def save(self, state):
            # TrainerState is serializable; use our atomic writer to survive interrupts.
            payload = json.loads(state.to_json_string()) if hasattr(state, "to_json_string") else None
            if payload is None:
                from dataclasses import asdict
                payload = asdict(state)
            write_json(folder / "trainer_state.json", payload)

        def on_step_end(self, args, state, control, **kwargs):
            self.save(state)

        def on_log(self, args, state, control, **kwargs):
            self.save(state)

        def on_train_end(self, args, state, control, **kwargs):
            self.save(state)

    return Evidence()


def adapter_identity(folder):
    folder = Path(folder)
    config = json.loads((folder / "adapter_config.json").read_text())
    if (config["peft_type"] != "LORA" or config["r"] != 8 or config["lora_alpha"] != 16
            or sorted(config["target_modules"]) != sorted(TARGETS) or config.get("lora_dropout", 0) != 0):
        raise ValueError("Serialized adapter does not match the frozen LoRA configuration")
    weights = sorted(folder.glob("*.safetensors"))
    if not weights:
        raise ValueError("Missing adapter weights")
    from safetensors import safe_open
    tensors = 0
    for path in weights:
        with safe_open(path, framework="numpy") as handle:
            for key in handle.keys():
                values = handle.get_tensor(key)
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite adapter tensor")
                tensors += 1
    return {"tensors": tensors, "files": {p.name: sha256(p) for p in [folder / "adapter_config.json", *weights]}}


def smoke_pipeline(pipe, folder, lora, device):
    """One synthetic optimizer step, never financial training or a seed trial."""
    import torch
    folder = Path(folder)
    synthetic = [np.sin(np.arange(256, dtype=np.float32) / 12),
                 np.cos(np.arange(256, dtype=np.float32) / 17)]
    tuned = pipe.fit(inputs=synthetic, prediction_length=5, finetune_mode="lora", lora_config=lora,
                     context_length=CONTEXT, learning_rate=1e-5, num_steps=1, batch_size=BATCH,
                     min_past=64, output_dir=folder / "trainer", seed=0, report_to="none",
                     dataloader_num_workers=0, logging_steps=1, callbacks=[trainer_evidence(folder)])
    tuned.save_pretrained(folder / "adapter")
    tuned.model.eval()
    _, points = tuned.predict_quantiles(synthetic, prediction_length=5, context_length=CONTEXT,
                                        batch_size=16, quantile_levels=[0.1, 0.5, 0.9])
    if not all(torch.isfinite(p).all() for p in points):
        raise ValueError("Non-finite smoke forecast")
    state = json.loads((folder / "trainer_state.json").read_text())
    if state["global_step"] != 1 or not any("loss" in row for row in state["log_history"]):
        raise ValueError("Smoke did not complete the recorded optimizer step")
    evidence = {"global_step": state["global_step"], "adapter": adapter_identity(folder / "adapter"),
                "synthetic_only": True, "device": device, "forecast_finite": True,
                "inference_model_training": tuned.model.training}
    del tuned
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evidence


def preflight_run004(snapshot, output, device="cuda"):
    import torch
    from chronos import Chronos2Pipeline
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never erase an earlier attempt.
    write_json(output / "status.json", {"status": "started", "created_at": utcnow(), "scope": "preflight_only"})
    try:
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("The Run 004 lock requires a smoke test on Kaggle CUDA")
        env = runtime()
        write_json(output / "environment.json", env)
        if env["torchao_installed"]:
            raise ValueError("Remove optional torchao in a fresh session before importing PEFT")
        if env["packages"].get("chronos-forecasting") != "2.3.2" or env["packages"].get("peft") != "0.21.2":
            raise ValueError("Use the declared Chronos 2.3.2 / PEFT 0.21.2 setup")
        # Version-only requirements: no editable/private URLs or credentials in the evidence.
        (output / "requirements-environment.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in env["packages"].items()), encoding="utf-8")
        shutil.copy2(Path(snapshot) / "manifest.json", output / "input_snapshot_manifest.json")
        shutil.copy2(RUN003_REFERENCE, output / "reference_snapshot_manifest.json")
        identity, source, lora = inputs_identity(snapshot), source_identity(), resolved_lora()
        write_json(output / "preflight-plan.json", {"protocol": PROTOCOL, "source": source,
            "inputs": identity, "lora": lora, "model_id": MODEL, "revision": REVISION,
            "synthetic_only": True, "optimizer_steps": 1})
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.set_num_threads(4)
        pipe = Chronos2Pipeline.from_pretrained(MODEL, revision=REVISION, device_map=device, torch_dtype=torch.float32)
        model_config = json.loads(pipe.model.config.to_json_string())
        write_json(output / "model_config.json", model_config)
        smoke = smoke_pipeline(pipe, output / "smoke", lora, device)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        lock = {"protocol": PROTOCOL, "source": source, "inputs": identity, "environment": env,
                "lora": lora, "model_id": MODEL, "revision": REVISION,
                "model_config_sha256": sha256(output / "model_config.json"), "smoke": smoke,
                "steps": STEPS, "seeds": list(SEEDS), "context_length": CONTEXT,
                "batch_size": BATCH, "max_windows": WINDOWS, "learning_rate": 1e-5,
                "inference_mode": "eval", "min_past": 64, "forecast_batch_size": 16,
                "forecast_quantiles": [0.1, 0.5, 0.9]}
        write_json(output / "candidate-lock.json", lock)
        write_json(output / "status.json", {"status": "completed", "scope": "preflight_only", "finished_at": utcnow(),
                                           "research_fits_started": 0})
        return lock
    except BaseException as error:
        record_failure(output, error)
        raise


def record_failure(folder, error):
    write_json(Path(folder) / "status.json", {"status": "failed_or_interrupted", "at": utcnow(),
                                             "error_type": type(error).__name__, "message": str(error)})
    (Path(folder) / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def validate_alignment(frames):
    """Reject missing rows, duplicate keys, shuffled targets and drifting baselines."""
    reference = frames[0].reset_index(drop=True)
    columns = ["target_return", "zero_shot_return", "finetuned_return"]
    for frame in frames:
        if frame.duplicated(["symbol", "decision_time"]).any() or not np.isfinite(frame[columns].to_numpy()).all():
            raise ValueError("Invalid prediction rows")
        if not reference[["symbol", "decision_time", "target_return"]].equals(
                frame[["symbol", "decision_time", "target_return"]].reset_index(drop=True)):
            raise ValueError("Seed evaluation rows/targets do not align")
        if not np.allclose(frame.zero_shot_return, reference.zero_shot_return, rtol=1e-6, atol=1e-9):
            raise ValueError("Zero-shot baseline drift between seeds")
    return reference


def metrics(actual, prediction):
    mse0, mae0 = float(np.mean(actual ** 2)), float(np.mean(abs(actual)))
    mse, mae = float(np.mean((prediction - actual) ** 2)), float(np.mean(abs(prediction - actual)))
    return {"rows": len(actual), "mae": mae, "mse": mse,
            "mae_ratio_to_zero": mae / mae0 if mae0 else None,
            "mse_ratio_to_zero": mse / mse0 if mse0 else None,
            "directional_accuracy": float(np.mean((prediction > 0) == (actual > 0)))}


def weekly_bootstrap(times, delta, repetitions=2000, seed=2026):
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    monday = times.normalize() - pd.to_timedelta(times.dayofweek, unit="D")
    codes, weeks = pd.factorize(monday, sort=True)
    sums, counts = np.bincount(codes, weights=delta), np.bincount(codes)
    result = {"blocks": len(weeks), "repetitions": repetitions, "rng_seed": seed,
              "unit": "UTC calendar week, Monday 00:00; both assets retained together",
              "estimate": float(np.mean(delta)), "percentile_95": None,
              "scope": "descriptive development diagnostic, no independent alpha inference"}
    if len(weeks) > 1:
        draws = np.random.default_rng(seed).integers(0, len(weeks), size=(repetitions, len(weeks)))
        values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result["percentile_95"] = np.quantile(values, [0.025, 0.975]).tolist()
    return result


def summarize(frames):
    if len(frames) != len(SEEDS):
        raise ValueError("Complete comparison requires all three seeds")
    reference = validate_alignment(frames)
    records, spreads = [], []
    for asset in ("ALL", "BTCUSDT", "ETHUSDT"):
        mask = np.ones(len(reference), dtype=bool) if asset == "ALL" else reference.symbol.eq(asset).to_numpy()
        if not mask.any():
            raise ValueError("Missing declared asset")
        truth = reference.target_return.to_numpy()[mask]
        baseline = reference.zero_shot_return.to_numpy()[mask]
        for name, predictions in (("zero_return", np.zeros(len(truth))), ("zero_shot", baseline)):
            records.append({"asset": asset, "model": name, "seed": None, **metrics(truth, predictions)})
        seed_rows = []
        for seed, frame in zip(SEEDS, frames):
            row = {"asset": asset, "model": "lora_B", "seed": seed,
                   **metrics(truth, frame.finetuned_return.to_numpy()[mask])}
            row["mae_delta_vs_zero_shot"] = row["mae"] - float(np.mean(abs(baseline - truth)))
            records.append(row)
            seed_rows.append(row)
        spreads.append({"asset": asset, "seeds": len(SEEDS), "std_ddof": 1,
                        **{key: {"mean": float(np.mean([r[key] for r in seed_rows])),
                                 "std": float(np.std([r[key] for r in seed_rows], ddof=1))}
                           for key in ("mae", "mse", "directional_accuracy", "mae_delta_vs_zero_shot")}})
    actual = reference.target_return.to_numpy()
    per_row = np.mean([abs(frame.finetuned_return.to_numpy() - actual) for frame in frames], axis=0)
    delta = per_row - abs(reference.zero_shot_return.to_numpy() - actual)
    changes = [r["mae_delta_vs_zero_shot"] for r in records if r["asset"] == "ALL" and r["model"] == "lora_B"]
    decision = ("reject" if all(d > 0 for d in changes) else
                "keep_for_further_research" if all(d < 0 for d in changes) else "inconclusive")
    return {"metrics": records, "seed_summary": spreads,
            "paired_mae_bootstrap": weekly_bootstrap(reference.decision_time, delta),
            "diagnostic_decision": decision, "deployment_status": "research_only_no_promotion",
            "contains_final_holdout": False,
            "limitations": ["Previously observed development windows; B was selected after Run 003.",
                            "Three seeds do not characterize the full initialization distribution.",
                            "No PnL/backtest or independent prospective result.",
                            "Pretraining overlap is unverified."]}


def validate_lock(lock, snapshot):
    expected = {"protocol": PROTOCOL, "model_id": MODEL, "revision": REVISION,
                "steps": STEPS, "seeds": list(SEEDS), "context_length": CONTEXT,
                "batch_size": BATCH, "max_windows": WINDOWS, "learning_rate": 1e-5,
                "inference_mode": "eval", "min_past": 64, "forecast_batch_size": 16,
                "forecast_quantiles": [0.1, 0.5, 0.9]}
    if any(lock.get(key) != value for key, value in expected.items()):
        raise ValueError("Run 004 budget/configuration differs from the preflight lock")
    smoke = lock["smoke"]
    if smoke["device"] != "cuda" or smoke["global_step"] != 1 or smoke["synthetic_only"] is not True:
        raise ValueError("A completed synthetic CUDA preflight is required")
    for name, current in (("environment", runtime()), ("inputs", inputs_identity(snapshot)),
                          ("source", source_identity()), ("lora", resolved_lora())):
        if lock[name] != current:
            raise ValueError(f"Frozen {name} changed; preserve evidence and review before training")


def run_seed_lab(snapshot, output, lock_path, notebook_url):
    import torch
    if not notebook_url.startswith("https://www.kaggle.com/code/"):
        raise ValueError("Record the actual Kaggle notebook URL before the research fits")
    lock_path, output = Path(lock_path), Path(output)
    lock = json.loads(lock_path.read_text())
    validate_lock(lock, snapshot)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {**lock, "preflight_lock_sha256": sha256(lock_path),
                                     "notebook_url": notebook_url, "created_at": utcnow(),
                                     "baseline_tolerance": {"rtol": 1e-6, "atol": 1e-9}})
    write_json(output / "status.json", {"status": "started", "at": utcnow()})
    frames = []
    try:
        for seed in SEEDS:
            folder = output / f"seed-{seed}"
            write_json(folder / "status.json", {"status": "started", "seed": seed, "at": utcnow()})
            try:
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                # chronos_experiment reloads the pinned base for EVERY fit; never accepts a previous adapter.
                report = chronos_experiment(snapshot, folder, steps=STEPS, max_windows=WINDOWS, device="cuda",
                    model_id=MODEL, revision=REVISION, learning_rate=1e-5, batch_size=BATCH,
                    context_length=CONTEXT, seed=seed, lora_config=lock["lora"],
                    callbacks=[trainer_evidence(folder)], logging_steps=1)
                state = json.loads((folder / "trainer_state.json").read_text())
                if state["global_step"] != STEPS or not any("loss" in row for row in state["log_history"]):
                    raise ValueError("Research fit lacks the declared step count/loss evidence")
                if report["evaluation_windows"] != WINDOWS or report["seed"] != seed:
                    raise ValueError("Research fit report does not match the plan")
                adapter = adapter_identity(folder / "adapter")
                frame = pd.read_parquet(folder / "chronos_validation.parquet")
                if len(frame) != WINDOWS or sample_sha256(frame) != lock["inputs"]["sample_sha256"]:
                    raise ValueError("Predictions differ from the locked evaluation sample")
                frames.append(frame)
                validate_alignment(frames)
                write_json(folder / "status.json", {"status": "completed", "seed": seed, "at": utcnow(),
                    "global_step": state["global_step"], "adapter": adapter,
                    "prediction_sha256": sha256(folder / "chronos_validation.parquet")})
            except BaseException as error:
                record_failure(folder, error)
                raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        result = summarize(frames)
        write_json(output / "comparison.json", result)
        pd.DataFrame(result["metrics"]).to_csv(output / "metrics.csv", index=False)
        write_json(output / "status.json", {"status": "completed", "at": utcnow(), "completed_fits": len(frames),
                                           "total_optimizer_steps": len(frames) * STEPS})
        return result
    except BaseException as error:
        record_failure(output, error)
        raise


def archive_evidence(output, destination):
    """Archive partial evidence as well; exclude duplicate/optimizer checkpoint binaries."""
    output, destination = Path(output), Path(destination)
    if not output.exists():
        return None
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if "trainer" in relative.parts and path.suffix not in {".json", ".jsonl", ".log"}:
                continue
            archive.write(path, relative)
    return str(destination)
