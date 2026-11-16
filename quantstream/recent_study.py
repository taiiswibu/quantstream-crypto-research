"""Reproducible screening of recent forecasting methods, with a whole-job budget."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .common import git_provenance, sha256, utcnow, write_json
from .modern_models import MODELS, check_model_files, prepare_model


def run(snapshot, output, seconds=1800, model_cache=None):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    code = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    source_files = [
        *sorted((code / "quantstream").glob("*.py")),
        *sorted((code / "scripts/modern_studies").glob("*.py")),
    ]
    manifest = json.loads((snapshot / "dataset_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(snapshot / name) != digest:
            raise ValueError("Snapshot changed: " + name)
    contract = {
        "protocol": "recent-methods-screen-v1",
        "seconds": seconds,
        "features_sha256": sha256(snapshot / "data/processed/features.parquet"),
        "source_files": {str(p.relative_to(code)): sha256(p) for p in source_files},
        "models": MODELS,
        "source": git_provenance(code),
        "target": "unchanged four-hour return; execution delay one hour",
        "foundation_cadence": "00:00 UTC daily; screening only",
        "old_test_used": False,
        "independent_test": False,
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != contract:
            raise ValueError("Study contract changed; keep the earlier evidence")
        status = json.loads((output / "status.json").read_text())
        if status.get("status") != "completed":
            raise ValueError(
                "Incomplete study retained; use a new output folder for a deliberate retry"
            )
        return status
    write_json(plan_path, contract)
    versions = {}
    for package in [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "tirex-2",
        "timestral",
        "xgboost-cpu",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(
        output / "runtime.json",
        {"python": sys.version, "device": "cpu", "packages": versions},
    )
    write_json(output / "status.json", {"status": "running", "started_at": utcnow()})
    started = time.monotonic()
    try:
        for name in MODELS:
            destination = output / "models" / name
            if model_cache:
                cached = Path(model_cache) / ("tirex-2" if name == "TiRex-2" else name)
                if cached.is_dir() and not destination.exists():
                    check_model_files(cached, name)
                    shutil.copytree(cached, destination)
            prepare_model(destination, name, deadline=started + seconds)
        if model_cache:
            cached_external = Path(model_cache) / "external"
            if cached_external.is_dir() and not (output / "external").exists():
                shutil.copytree(cached_external, output / "external")
        environment = {
            **os.environ,
            "QUANTSTREAM_SNAPSHOT": str(snapshot),
            "QUANTSTREAM_MODERN_OUTPUT": str(output),
            "PYTHONPATH": str(code) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        for script in [
            "download_exogenous",
            "enriched_study",
            "foundation_study",
            "timestral_study",
            "interval_study",
        ]:
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Whole-study budget exhausted")
            write_json(
                output / "progress.json",
                {"stage": script, "elapsed_seconds": time.monotonic() - started},
            )
            with (output / (script + ".log")).open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(code / "scripts/modern_studies" / (script + ".py")),
                    ],
                    cwd=code,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=True,
                )
        for name, expected in [("enriched", 6), ("foundation", 2), ("timestral", 4)]:
            actual = json.loads((output / name / "results.json").read_text())
            if len(actual) != expected:
                raise ValueError("Incomplete results: " + name)
        status = {
            "status": "completed",
            "completed_at": utcnow(),
            "elapsed_seconds": time.monotonic() - started,
            "conditions": 12,
            "old_test_used": False,
            "promotion": "none; evidence for research and interval monitoring only",
        }
        write_json(output / "status.json", status)
        return status
    except BaseException as error:
        write_json(
            output / "status.json",
            {"status": "incomplete", "error": str(error), "at": utcnow()},
        )
        (output / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.snapshot, args.output, args.seconds, args.model_cache), indent=2
        )
    )
