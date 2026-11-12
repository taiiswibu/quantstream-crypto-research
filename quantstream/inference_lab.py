"""Run 005: fixed-weight inference controls. No fitting, optimizer or deployment."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import traceback
import zipfile

import numpy as np
import pandas as pd

from .common import git_provenance, sha256, utcnow, write_json, write_parquet
from .finetune import chronos_inputs
from .seed_lab import CONTEXT, MODEL, REVISION, inputs_identity, runtime, sample_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "run-005-inference-v1"
RUN004_SHA = "0c53a0dabcd2c67e3c36993628d9495c4027d950840be5a28673f9a0b849c333"
ADAPTER_SHA = "fb1769324c59a682ef7b84a674fab1b0275e5ed0e408981cb0e27cc4a094182d"
REFERENCE_FILES = {
    "plan.json": "80be6e8b47b47520038ed451e66b87951f681df4e8723f5e7761c7d1d3dae1db",
    "status.json": "5e6f5e864343ddefc82adc8712606d5890fa040e18b6fdadbadc1f9beed8abf9",
    "seed-42/status.json": "4ba8c2d89f9e31377289d26426da29fd63bfdb8d3e7882da5f62dbfd9d000152",
    "seed-42/chronos_validation.parquet": "da063b07065b3d587cd6ac08c70e007b7406b32c366f9783f413431a859e397b",
    "seed-42/adapter/adapter_config.json": "c0c394299241fd9f1d06fd42edc0364ca50feab6f860373b5d8019bdbd2b37b8",
    "seed-42/adapter/adapter_model.safetensors": ADAPTER_SHA,
}
RTOL, ATOL = 1e-6, 1e-9
CASES = (
    ("eval_42_a", False, 42), ("eval_42_b", False, 42), ("eval_123", False, 123),
    ("dropout_42_a", True, 42), ("dropout_42_b", True, 42),
    ("dropout_123", True, 123), ("dropout_2026", True, 2026),
    ("eval_42_after", False, 42),
)


def state_fingerprint(model):
    """Hash all parameter and persistent buffer values, names, dtypes and shapes."""
    import torch
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().contiguous()
        result.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        result.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def seed_inference(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_fingerprints():
    import torch
    return {"cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda": [hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                     for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def forecast_returns(pipe, contexts, cfg):
    """Same Run 004 transformation; intentionally never change the model's mode here."""
    import torch
    horizon = cfg.horizon + cfg.execution_lag_bars
    with torch.inference_mode():
        _, points = pipe.predict_quantiles(contexts, prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9], context_length=CONTEXT, batch_size=16)
        paths = [point.detach().float().cpu().numpy().reshape(-1, horizon)[0] for point in points]
        result = np.array([np.expm1(path[-1] - path[cfg.execution_lag_bars - 1]) for path in paths])
    if len(result) != len(contexts) or not np.isfinite(result).all():
        raise ValueError("Missing/non-finite forecasts")
    return result


def difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Comparison needs the same finite prediction population")
    return {"bit_identical": bool(np.array_equal(left, right)),
            "within_tolerance": bool(np.allclose(left, right, rtol=RTOL, atol=ATOL)),
            "max_absolute_difference": float(np.max(np.abs(left - right))),
            "mean_absolute_difference": float(np.mean(np.abs(left - right)))}


def score(scored, predictions, case):
    records = []
    for asset in ("ALL", "BTCUSDT", "ETHUSDT"):
        mask = np.ones(len(scored), dtype=bool) if asset == "ALL" else scored.symbol.eq(asset).to_numpy()
        if not mask.any():
            raise ValueError("Missing declared asset")
        truth, pred = scored.target_return.to_numpy()[mask], predictions[mask]
        records.append({"case": case, "asset": asset, "rows": len(truth),
            "mae": float(np.mean(np.abs(pred - truth))), "mse": float(np.mean((pred - truth) ** 2)),
            "directional_accuracy": float(np.mean((pred > 0) == (truth > 0)))})
    return records


def controlled_passes(pipe, contexts, scored, cfg, output, reference):
    """Run the fixed eight passes; freeze values and gradients in both mode treatments."""
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pipe.model.requires_grad_(False)
    fixed_state = state_fingerprint(pipe.model)
    predictions, records, cases = {}, [], []
    try:
        for case, dropout_on, seed in CASES:
            seed_inference(seed)
            pipe.model.train(dropout_on)
            before_mode = pipe.model.training
            merged_layers = sum(bool(getattr(module, "merged_adapters", [])) for module in pipe.model.modules())
            if merged_layers:
                raise RuntimeError("Run 005 requires unmerged LoRA layers")
            active_dropout = sum(isinstance(module, torch.nn.Dropout) and module.p > 0 and module.training
                                 for module in pipe.model.modules())
            rng_before = rng_fingerprints()
            with torch.inference_mode():
                grad_enabled = torch.is_grad_enabled()
                predicted = forecast_returns(pipe, contexts, cfg)
            after_state = state_fingerprint(pipe.model)
            row = {"case": case, "dropout_requested": dropout_on, "rng_seed": seed,
                "model_training_before": before_mode, "model_training_after": pipe.model.training,
                "active_dropout_modules": active_dropout, "grad_enabled_during_prediction": grad_enabled,
                "merged_lora_layers": merged_layers,
                "trainable_parameters": sum(parameter.numel() for parameter in pipe.model.parameters()
                                            if parameter.requires_grad),
                "state_before": fixed_state, "state_after": after_state,
                "state_unchanged": after_state == fixed_state,
                "rng_before": rng_before, "rng_after": rng_fingerprints(), "optimizer_steps": 0}
            write_json(output / "passes" / f"{case}.json", row)
            if after_state != fixed_state:
                raise RuntimeError("Model parameters/buffers changed during inference; preserve evidence")
            if grad_enabled or row["trainable_parameters"]:
                raise RuntimeError("Fixed-weight inference must not enable gradients or trainable parameters")
            frame = scored.copy()
            frame["prediction"] = predicted
            write_parquet(output / "passes" / f"{case}.parquet", frame)
            predictions[case] = predicted
            records.extend(score(scored, predicted, case))
            cases.append(row)
            write_json(output / "progress.json", {"completed_passes": len(cases), "planned_passes": len(CASES),
                                                  "optimizer_steps": 0, "at": utcnow()})
            print(f"Inference {len(cases)}/{len(CASES)}: {case}; no optimizer steps", flush=True)
        checks = {
            "eval_same_rng": difference(predictions["eval_42_a"], predictions["eval_42_b"]),
            "eval_different_rng": difference(predictions["eval_42_a"], predictions["eval_123"]),
            "eval_after_dropout": difference(predictions["eval_42_a"], predictions["eval_42_after"]),
            "dropout_same_rng": difference(predictions["dropout_42_a"], predictions["dropout_42_b"]),
            "dropout_different_rng_123": difference(predictions["dropout_42_a"], predictions["dropout_123"]),
            "dropout_different_rng_2026": difference(predictions["dropout_42_a"], predictions["dropout_2026"]),
            "mode_effect_rng42": difference(predictions["eval_42_a"], predictions["dropout_42_a"]),
            "run004_eval_reference": difference(predictions["eval_42_a"], reference.finetuned_return.to_numpy()),
        }
        mode_held = all(row["model_training_before"] == row["dropout_requested"] == row["model_training_after"]
                        for row in cases)
        treatment_active = all(row["active_dropout_modules"] > 0 for row in cases if row["dropout_requested"])
        reproducible = all(checks[key]["within_tolerance"] for key in
                           ("eval_same_rng", "eval_different_rng", "eval_after_dropout", "dropout_same_rng"))
        result = {"checks": checks, "controls_passed": mode_held and treatment_active and reproducible,
            "mode_treatment_held": mode_held, "dropout_treatment_active": treatment_active,
            "mode_effect_observed": not checks["mode_effect_rng42"]["within_tolerance"],
            "rng_effect_with_dropout_observed": any(not checks[key]["within_tolerance"] for key in
                ("dropout_different_rng_123", "dropout_different_rng_2026")),
            "metrics": records, "passes": cases, "full_state_sha256": fixed_state,
            "completed_passes": len(cases), "optimizer_steps": 0,
            "deployment_status": "research_only_no_promotion",
            "limitations": ["Diagnostic on previously observed development data, not a model search or independent test.",
                "Inference RNG seeds are not new training seeds; all passes use the same seed-42 adapter.",
                "A mode effect in this locked runtime cannot reconstruct an unrecorded historical RNG/runtime state.",
                "No PnL, prospective evaluation, refit or model promotion."]}
        pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
        write_json(output / "comparison.json", result)
        return result
    finally:
        pipe.model.eval()


def prepare_inputs(input_root, work):
    """Read the existing snapshot and the exact supplied Run 004 artifact."""
    input_root, work = Path(input_root), Path(work)
    artifacts = list(input_root.rglob("QuantStream_Run004_Output.zip"))
    expanded = [p.parent for p in input_root.rglob("plan.json")
                if (p.parent / "seed-42/chronos_validation.parquet").exists()
                and (p.parent / "seed-42/adapter/adapter_model.safetensors").exists()]
    if len(artifacts) > 1 or (not artifacts and len(expanded) != 1):
        raise FileNotFoundError("Add Input đúng một output Run 004 (ZIP hoặc đã giải nén) cùng Dataset cũ")
    artifact = artifacts[0] if artifacts else expanded[0]
    archives = list(input_root.rglob("QuantStream_Kaggle_Input.zip"))
    if len(archives) > 1:
        raise ValueError("Chỉ Add Input một phiên bản Dataset snapshot")
    snapshot = work / "snapshot"
    if not snapshot.exists():
        if archives:
            with zipfile.ZipFile(archives[0]) as archive:
                for info in archive.infolist():
                    name = Path(info.filename)
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError("Unsafe snapshot archive path")
                    if name.parts and name.parts[0] == "snapshot":
                        archive.extract(info, work)
        else:
            candidates = [p.parent for p in input_root.rglob("manifest.json")
                          if (p.parent / "development.parquet").exists()]
            if len(candidates) != 1:
                raise FileNotFoundError("Add Input Dataset snapshot đã dùng cho Run 004")
            shutil.copytree(candidates[0], snapshot)
    return snapshot, artifact


def read_reference(artifact, work, output):
    artifact = Path(artifact)
    if artifact.is_file() and sha256(artifact) != RUN004_SHA:
        raise ValueError("Run 004 ZIP differs from the reviewed completed artifact")
    archive = zipfile.ZipFile(artifact) if artifact.is_file() else None
    try:
        def read(name):
            return archive.read(name) if archive else (artifact / name).read_bytes()
        verified = {}
        for name, expected in REFERENCE_FILES.items():
            verified[name] = read(name)
            if hashlib.sha256(verified[name]).hexdigest() != expected:
                raise ValueError("Run 004 reference file differs: " + name)
        status = json.loads(verified["status.json"])
        if status["status"] != "completed" or status["total_optimizer_steps"] != 600:
            raise ValueError("Need the completed Run 004 artifact")
        plan = json.loads(verified["plan.json"])
        reference = pd.read_parquet(io.BytesIO(verified["seed-42/chronos_validation.parquet"]))
        adapter = Path(work) / ("adapter-" + ADAPTER_SHA[:12])
        adapter.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            raw = verified["seed-42/adapter/" + name]
            path = adapter / name
            if path.exists() and path.read_bytes() != raw:
                raise FileExistsError("Adapter cache changed; preserve the earlier attempt")
            if not path.exists():
                path.write_bytes(raw)
        if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA:
            raise ValueError("Selected adapter weights do not match Run 004 seed 42 / Run 003 B")
        write_json(Path(output) / "run004_plan.json", plan)
        write_parquet(Path(output) / "reference_predictions.parquet", reference)
        write_json(Path(output) / "reference-files.json", {"input_form": "zip" if archive else "expanded",
            "zip_sha256_recomputed": RUN004_SHA if archive else None, "verified_file_sha256": REFERENCE_FILES})
    finally:
        if archive:
            archive.close()
    return plan, reference, adapter


def run_inference_lab(snapshot, artifact, output, device="cuda"):
    """Validate fixed inputs/runtime, then run nine predictions and zero optimizer steps."""
    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "started", "at": utcnow(), "optimizer_steps": 0})
    try:
        previous, reference, adapter = read_reference(artifact, output.parent, output)
        environment = runtime()
        write_json(output / "environment.json", environment)
        differences = {key: {"expected": previous["environment"].get(key), "actual": environment.get(key)}
                       for key in set(environment) | set(previous["environment"])
                       if environment.get(key) != previous["environment"].get(key)}
        write_json(output / "environment-diff.json", differences)
        if differences:
            raise ValueError("Environment differs from Run 004; send the diagnostic ZIP before inference")
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("This registered diagnostic requires the same Kaggle CUDA runtime")
        identity = inputs_identity(snapshot)
        scientific = ("reference_manifest_sha256", "data_file_hashes", "sample_sha256", "contexts_sha256",
                      "training_sha256", "training_lengths", "training_series", "rows", "horizon")
        if any(identity[key] != previous["inputs"][key] for key in scientific):
            raise ValueError("Run 005 data/context/sample differs from Run 004")
        _, cfg, _, contexts, scored = chronos_inputs(snapshot, CONTEXT, 512)
        if sample_sha256(reference) != identity["sample_sha256"]:
            raise ValueError("Reference prediction sample mismatch")
        pd.testing.assert_frame_equal(scored.reset_index(drop=True), reference[["symbol", "decision_time", "target_return"]])
        source = git_provenance(ROOT)
        if source["status"] != "available" or source["dirty"] is not False:
            raise ValueError("Use the clean pinned Run 005 source checkout")
        source["files"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "quantstream").glob("*.py"))}
        source["protocol_sha256"] = sha256(ROOT / "docs/experiments/run-005-inference-protocol.md")
        plan = {"protocol": PROTOCOL, "created_at": utcnow(), "source": source, "environment": environment,
            "inputs": identity, "run004_artifact_sha256": RUN004_SHA, "adapter_sha256": ADAPTER_SHA,
            "adapter_training_seed": 42, "selection_reason": "Historical byte-identical adapter, not the lowest Run 004 MAE",
            "model_id": MODEL, "model_revision": REVISION, "loading_mode": "peft_unmerged",
            "cases": [{"id": name, "dropout_on": mode, "inference_rng_seed": seed} for name, mode, seed in CASES],
            "base_reference_passes": 1, "adapter_passes": 8, "windows_per_pass": 512,
            "context_length": CONTEXT, "forecast_batch_size": 16, "rtol": RTOL, "atol": ATOL,
            "optimizer_steps": 0, "model_merge_operations": 0}
        write_json(output / "plan.json", plan)
        shutil.copy2(Path(snapshot) / "manifest.json", output / "input_snapshot_manifest.json")
        torch.set_num_threads(cfg.threads)
        seed_inference(42)
        pipe = Chronos2Pipeline.from_pretrained(MODEL, revision=REVISION, device_map=device, torch_dtype=torch.float32)
        write_json(output / "base_model_config.json", json.loads(pipe.model.config.to_json_string()))
        if sha256(output / "base_model_config.json") != previous["model_config_sha256"]:
            raise ValueError("Base model configuration differs from Run 004")
        pipe.model.requires_grad_(False)
        pipe.model.eval()
        base_state = state_fingerprint(pipe.model)
        zero_shot = forecast_returns(pipe, contexts, cfg)
        if state_fingerprint(pipe.model) != base_state:
            raise RuntimeError("Base state changed during zero-shot control")
        baseline_check = difference(zero_shot, reference.zero_shot_return.to_numpy())
        write_parquet(output / "base_control.parquet", scored.assign(prediction=zero_shot))
        pipe.model = PeftModel.from_pretrained(pipe.model, str(adapter), is_trainable=False)
        pipe.inner_model = pipe.model
        pipe.model.requires_grad_(False)
        result = controlled_passes(pipe, contexts, scored, cfg, output, reference)
        result["zero_shot_reference"] = baseline_check
        result["reference_reproduced"] = baseline_check["within_tolerance"] and result["checks"]["run004_eval_reference"]["within_tolerance"]
        result["ready_within_diagnostic_scope"] = result["controls_passed"] and result["reference_reproduced"]
        if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA:
            raise RuntimeError("Adapter file changed during inference")
        write_json(output / "comparison.json", result)
        write_json(output / "status.json", {"status": "completed", "at": utcnow(), "optimizer_steps": 0,
            "completed_adapter_passes": 8, "completed_base_passes": 1,
            "controls_passed": result["controls_passed"], "reference_reproduced": result["reference_reproduced"]})
        return result
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed_or_interrupted", "at": utcnow(), "optimizer_steps": 0,
            "error_type": type(error).__name__, "message": str(error)})
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
