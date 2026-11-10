"""Keep configuration, feature data and temporal splits bound to one snapshot."""
from __future__ import annotations

import json
from pathlib import Path

from . import features
from .common import sha256


def verify_feature_snapshot(root, cfg):
    """Fail before training/export if a prepared dataset is stale or modified."""
    root = Path(root)
    folder = root / "data/processed"
    manifest = json.loads((folder / "feature_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((folder / "split_plan.json").read_text(encoding="utf-8"))
    def canonical(value):
        return json.dumps(value, sort_keys=True)
    if canonical(manifest["config"]) != canonical(cfg.to_dict()):
        raise ValueError("Config changed after feature preparation. Run prepare first.")
    if sha256(folder / "features.parquet") != manifest["features_sha256"]:
        raise ValueError("Prepared feature data changed. Run prepare first.")
    if canonical(plan) != canonical(manifest["split"]):
        raise ValueError("Temporal split changed after preparation. Run prepare first.")
    if manifest["feature_columns"] != features.FEATURES:
        raise ValueError("Feature schema changed. Run prepare first.")
    if sha256(Path(features.__file__)) != manifest["feature_code_sha256"]:
        raise ValueError("Feature implementation changed. Run prepare first.")
    return manifest, plan
