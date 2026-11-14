"""Run 005 bootstrap repair for reviewed duplicate NumPy metadata. No training."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/experiments/run-005-runtime-probe/verification.json"
PINS = ("torch", "pandas", "pyarrow", "scikit-learn", "chronos-forecasting", "peft", "transformers",
        "accelerate", "datasets", "huggingface-hub", "safetensors", "tokenizers")
MODULES = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "scipy": "scipy",
           "sklearn": "scikit-learn", "matplotlib": "matplotlib", "torch": "torch",
           "chronos": "chronos-forecasting", "peft": "peft", "transformers": "transformers",
           "accelerate": "accelerate", "datasets": "datasets", "huggingface_hub": "huggingface-hub",
           "safetensors": "safetensors", "tokenizers": "tokenizers"}


def inventory():
    candidates = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            candidates.setdefault(name.lower().replace("_", "-"), []).append(
                {"version": dist.version, "root": str(dist.locate_file(""))})
    return {"legacy_packages": {name: rows[-1]["version"] for name, rows in candidates.items()},
            "candidates": dict(sorted(candidates.items())),
            "duplicates": {name: rows for name, rows in candidates.items() if len(rows) > 1}}


def package_diff(expected, actual):
    return {name: {"expected": expected.get(name), "actual": actual.get(name)}
            for name in sorted(set(expected) | set(actual)) if expected.get(name) != actual.get(name)}


def approved_repairs(expected, actual):
    """Only the two differences established by the uploaded probe are repairable."""
    differences = package_diff(expected, actual)
    for name, row in differences.items():
        allowed = (name == "chronos-forecasting" and row == {"expected": "2.3.2", "actual": None}) or (
            name == "torchao" and row == {"expected": None, "actual": "0.10.0"})
        if not allowed:
            raise ValueError("Unreviewed package drift before setup: " + name)
    return differences


def validate_pins(expected, candidates, allow_missing_chronos=False):
    for name in PINS:
        try:
            current = metadata.version(name)
        except metadata.PackageNotFoundError:
            current = None
        if name == "chronos-forecasting" and current is None and allow_missing_chronos:
            continue
        if current != expected[name]:
            raise ValueError(f"Active metadata pin differs: {name}: {current} != {expected[name]}")
        versions = {row["version"] for row in candidates.get(name, [])}
        if versions != {expected[name]}:
            raise ValueError("Conflicting core metadata not reviewed: " + name)


def dependency_errors(roots=PINS):
    """Check active dependencies of the research stack, not unrelated Kaggle apps."""
    from packaging.requirements import Requirement
    errors, visited, pending = [], set(), list(roots) + ["numpy"]
    while pending:
        name = pending.pop().lower().replace("_", "-")
        if name in visited:
            continue
        visited.add(name)
        try:
            requirements = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            errors.append({"package": name, "error": "missing distribution"})
            continue
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(requirement.name)
                if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                    errors.append({"package": name, "requires": raw, "actual": version})
                pending.append(requirement.name)
            except metadata.PackageNotFoundError:
                errors.append({"package": name, "requires": raw, "actual": None})
    return errors


def loaded_runtime():
    modules = {}
    for name, distribution in MODULES.items():
        module = importlib.import_module(name)
        modules[name] = {"version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None),
                         "metadata_version": metadata.version(distribution)}
    import numpy as np
    import torch
    return {"python": platform.python_version(), "modules": modules,
            "numpy_bridge_sum": float(torch.from_numpy(np.ones(2, dtype=np.float32)).sum()),
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}


def validate_loaded(actual, expected, audit):
    if actual["python"] != expected["python"] or not actual["cuda_available"]:
        raise ValueError("Reviewed Python and CUDA runtime required")
    for name, version in audit["actual_module_versions"].items():
        if actual["modules"][name]["version"] != version:
            raise ValueError("Imported module differs from reviewed probe: " + name)
    for key in ("cuda", "cudnn", "gpu", "gpu_capability"):
        if actual[key] != expected[key]:
            raise ValueError("Reviewed device/runtime differs: " + key)
    if actual["numpy_bridge_sum"] != 2.0:
        raise ValueError("NumPy/PyTorch bridge failed")


def synthetic_adapter_forward():
    import torch
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model
    if not callable(Chronos2Pipeline):
        raise RuntimeError("Chronos import failed")
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)),
                          LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
    model.requires_grad_(False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.ones(2, 4))
    if output.shape != (2, 4) or not torch.isfinite(output).all():
        raise RuntimeError("Synthetic PEFT forward failed")
    return {"synthetic_only": True, "device": "cpu", "finite_forward": True,
            "optimizer_steps": 0, "model_downloads": 0}


def loaded_conflicts():
    return [name for name in ("peft", "chronos", "transformers", "torchao")
            if any(key == name or key.startswith(name + ".") for key in sys.modules)]


def kernel_numpy_version():
    return getattr(sys.modules["numpy"], "__version__", None) if "numpy" in sys.modules else None


def prepare_runtime(expected, evidence):
    """Preserve NumPy/Torch; remove reviewed torchao and install only missing Chronos."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    report = {"revision": "runtime-fix-001", "status": "started", "actions": [], "optimizer_steps": 0,
              "probe_artifact_sha256": audit["artifact_sha256"], "model_downloads": 0}

    def save():
        (evidence / "setup.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def pip(args):
        record = {"args": args}
        report["actions"].append(record)
        save()
        result = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True, timeout=180)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        save()
        result.check_returncode()

    try:
        save()
        if platform.python_version() != expected["python"]:
            raise RuntimeError("Python base image differs from Run 004")
        loaded = loaded_conflicts()
        if loaded:
            raise RuntimeError("Restart Session before setup: " + ", ".join(loaded))
        before = inventory()
        report["before"] = before
        report["approved_repairs"] = approved_repairs(expected["packages"], before["legacy_packages"])
        validate_pins(expected["packages"], before["candidates"], allow_missing_chronos=True)
        numpy_candidates = before["candidates"].get("numpy", [])
        if numpy_candidates != audit["numpy_metadata_candidates"] or metadata.version("numpy") != "2.1.3":
            raise ValueError("NumPy metadata resolution differs from reviewed probe")
        if kernel_numpy_version() not in (None, "2.1.3"):
            raise ValueError("Kernel already loaded a different NumPy; preserve evidence and restart")
        save()
        if "torchao" in before["legacy_packages"]:
            pip(["uninstall", "-y", "torchao"])
        if "chronos-forecasting" not in before["legacy_packages"]:
            pip(["install", "--no-deps", "--only-binary=:all:", "chronos-forecasting==2.3.2"])
        importlib.invalidate_caches()
        if importlib.util.find_spec("torchao") is not None:
            raise ValueError("torchao remains discoverable after removal")
        after = inventory()
        report["after"] = after
        report["legacy_inventory_diff"] = package_diff(expected["packages"], after["legacy_packages"])
        save()
        if report["legacy_inventory_diff"]:
            raise ValueError("Full Run 004 metadata inventory differs after the two approved repairs")
        validate_pins(expected["packages"], after["candidates"])
        report["research_dependency_errors"] = dependency_errors()
        save()
        if report["research_dependency_errors"]:
            raise ValueError("Research stack has unsatisfied active dependencies; see setup.json")
        report["loaded_runtime"] = loaded_runtime()
        validate_loaded(report["loaded_runtime"], expected, audit)
        report["smoke"] = synthetic_adapter_forward()
        report["status"] = "completed"
        report["historical_loaded_numpy_recorded"] = False
        return report
    except BaseException:
        report["status"] = "setup_failed_or_interrupted"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()
