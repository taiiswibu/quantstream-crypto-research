"""Read-only Run 005 runtime probe: no installs, model downloads, fits or input changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
import zipfile

PROTOCOL = "run-005-runtime-probe-v1"
MODULES = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "torch")
CRITICAL_PACKAGES = ("numpy", "pandas", "torch", "chronos-forecasting", "peft", "transformers",
                     "accelerate", "datasets", "pyarrow", "scikit-learn", "torchao")

CHILD_PROBE = r'''
import importlib, json, sys, traceback
result={"python":sys.version,"executable":sys.executable,"modules":{}}
for name in ("numpy","pandas","pyarrow","scipy","sklearn","matplotlib","torch"):
    try:
        module=importlib.import_module(name)
        row={"version":getattr(module,"__version__",None),"file":getattr(module,"__file__",None)}
        if name=="numpy":
            row["dot_check"]=float(module.dot(module.ones(2),module.ones(2)))
        if name=="torch":
            row.update({"cuda":module.version.cuda,"cuda_available":module.cuda.is_available()})
            if row["cuda_available"]:
                row["gpu"]=module.cuda.get_device_name(0)
                row["gpu_capability"]=list(module.cuda.get_device_capability(0))
                row["cudnn"]=module.backends.cudnn.version()
            try:
                np=importlib.import_module("numpy")
                row["numpy_bridge_sum"]=float(module.from_numpy(np.ones(2,dtype=np.float32)).sum())
            except Exception:
                row["numpy_bridge_error"]=traceback.format_exc()
        result["modules"][name]=row
    except Exception:
        result["modules"][name]={"error":traceback.format_exc()}
print("QUANTSTREAM_PROBE_JSON="+json.dumps(result,sort_keys=True))
'''


def package_inventory(distributions=None):
    """Keep every metadata candidate: version() picks first, old lock used last."""
    distributions = metadata.distributions() if distributions is None else distributions
    inventory = {}
    for dist in distributions:
        name = dist.metadata.get("Name")
        if not name:
            continue
        name = name.lower().replace("_", "-")
        inventory.setdefault(name, []).append({"version": dist.version, "root": str(dist.locate_file(""))})
    first = {}
    for name in CRITICAL_PACKAGES:
        try:
            first[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            first[name] = None
    return {"all_candidates": dict(sorted(inventory.items())),
            "duplicates": {name: rows for name, rows in inventory.items() if len(rows) > 1},
            "legacy_last_wins_versions": {name: rows[-1]["version"] for name, rows in inventory.items()},
            "metadata_version_first_match": first}


def command_probe(args, timeout=90):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error_type": type(error).__name__, "error": str(error)}


def run_probe(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(tempfile.mkdtemp(prefix=f"QuantStream_Run005_Runtime_Probe_{stamp}_", dir=destination))
    archive = folder.with_suffix(".zip")
    report = {"protocol": PROTOCOL, "optimizer_steps": 0, "model_downloads": 0,
              "package_mutations": 0, "status": "started", "created_at_utc": stamp}

    def save():
        (folder / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    try:
        save()
        report["kernel"] = {"python": platform.python_version(), "version": sys.version,
                            "executable": sys.executable, "platform": platform.platform(),
                            "sys_path": list(sys.path)}
        report["kernel_already_loaded"] = {
            name: {"version": getattr(sys.modules[name], "__version__", None),
                   "file": getattr(sys.modules[name], "__file__", None)}
            for name in MODULES if name in sys.modules}
        report["inventory"] = package_inventory()
        save()
        child = command_probe([sys.executable, "-c", CHILD_PROBE])
        report["fresh_process"] = child
        for line in child.get("stdout", "").splitlines():
            if line.startswith("QUANTSTREAM_PROBE_JSON="):
                report["fresh_process_modules"] = json.loads(line.split("=", 1)[1])
        report["pip_check"] = command_probe([sys.executable, "-m", "pip", "check"], timeout=45)
        report["nvidia_smi"] = command_probe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                                              "--format=csv,noheader"], timeout=15)
        report["status"] = "probe_completed"
        report["scope"] = "Environment evidence only. Import or pip-check errors remain evidence, not fit failures."
        return archive, report
    except BaseException:
        report["status"] = "probe_failed_or_interrupted"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as saved:
            saved.write(folder / "diagnostic.json", "diagnostic.json")
        print("Runtime probe ZIP:", archive, flush=True)
        print("No packages installed; no checkpoint downloaded; optimizer_steps=0", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, _ = run_probe(args.output)
    print("ZIP SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())
