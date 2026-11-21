"""Import a trained TCN into the terminal without fitting or replacing its source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from filelock import FileLock
import numpy as np
import pandas as pd

from .common import Config, sha256, utcnow, write_json
from .features import FEATURES

IDENTITY_KEYS = [
    "type",
    "model",
    "sha256",
    "config",
    "features",
    "features_sha256",
    "snapshot_manifest_sha256",
    "split",
    "threshold",
    "best_epoch",
    "training_rows",
    "contains_final_holdout",
    "device_used",
    "source_control",
    "history",
    "final_history",
    "created_at",
    "selection",
]
# Public reference identities. Unknown checkpoints need their original source manifest.
KNOWN = {
    "0be961ff47c2d5516bf4f4bed4e2459d9e1d41f6164a3b270deebd00dca35fe6": {
        "kind": "reference_cpu",
        "label": "TCN · reference CPU",
        "metadata_identity": "df0fd41b565d97fe292ca8d6130a7f05597e46c7f9f350b895c561033284248f",
    },
    "59f1c095d253b3ce5a4b1eb839b59088826d0401b45d756a22fdf87ddb5f4952": {
        "kind": "kaggle_run001",
        "label": "TCN · Kaggle Run 001",
        "metadata_identity": "5cabe16f2561484238c1fcf617e13ff787217b6996117f76a4642c4bdbe76b12",
    },
}
FEATURE_SOURCE_SHA = "2944b185c86dc2f53f24a0dabc858b9d6b80e1550d38d662464ec33e1377a388"


def identity(metadata):
    core = {key: metadata[key] for key in IDENTITY_KEYS if key in metadata}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inspect_checkpoint(folder, runtime_cfg):
    folder = Path(folder).expanduser().resolve()
    model_path, metadata_path = folder / "model.pt", folder / "tcn_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise ValueError("Cần model.pt và tcn_metadata.json trong cùng thư mục")
    if model_path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Checkpoint lớn hơn giới hạn của TCN nhỏ hiện tại")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256(model_path)
    if digest != metadata.get("sha256"):
        raise ValueError(
            "Checkpoint SHA không khớp metadata; giữ nguyên file để kiểm tra"
        )
    if metadata.get("type") != "tcn" or metadata.get("features") != FEATURES:
        raise ValueError("Sai loại model hoặc thứ tự feature TCN")
    if metadata.get("contains_final_holdout") is not False:
        raise ValueError("Metadata phải xác nhận không train trên final holdout")
    training = dict(metadata["config"])
    training["symbols"] = tuple(training["symbols"])
    trained_cfg = Config(**training)
    for key in ("symbols", "interval", "horizon", "execution_lag_bars", "lookback"):
        if getattr(trained_cfg, key) != getattr(runtime_cfg, key):
            raise ValueError("Cấu hình inference khác model đã train: " + key)
    if not np.isfinite(metadata["threshold"]) or metadata["threshold"] < 0:
        raise ValueError("Threshold không hợp lệ")
    if digest in KNOWN:
        anchor = KNOWN[digest]
        if identity(metadata) != anchor["metadata_identity"]:
            raise ValueError("Metadata khác bản đã đối chiếu của checkpoint này")
        if sha256(Path(__file__).with_name("features.py")) != FEATURE_SOURCE_SHA:
            raise ValueError("Feature code đã đổi so với checkpoint reference")
        origin = {
            "kind": anchor["kind"],
            "label": anchor["label"],
            "status": "matches_public_reference",
        }
    else:
        manifest_path = folder / "training_snapshot_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Checkpoint chưa có trong reference; cần training_snapshot_manifest.json gốc"
            )
        if sha256(manifest_path) != metadata.get("snapshot_manifest_sha256"):
            raise ValueError("Training manifest SHA không khớp metadata")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("full_features_sha256") != metadata.get("features_sha256")
            or manifest.get("config") != metadata["config"]
        ):
            raise ValueError("Training manifest khác dữ liệu hoặc cấu hình đã ghi")
        expected = manifest.get("code_files", {}).get("quantstream/features.py")
        if expected != sha256(Path(__file__).with_name("features.py")):
            raise ValueError("Chưa xác minh được source tạo feature của checkpoint")
        origin = {
            "kind": "supplied_manifest",
            "label": "TCN nhập từ checkpoint",
            "status": "source_contract_verified_training_not_independently_audited",
        }
    return metadata, origin, model_path


def load_checked_payload(path, metadata):
    try:
        import torch
    except ImportError as error:
        raise ValueError(
            "Cần PyTorch CPU. Xem hướng dẫn cài phần dl trong docs/VAN_HANH_VI.md"
        ) from error
    major, minor = (int(v) for v in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 6):
        raise ValueError("Cần PyTorch >= 2.6 để nạp checkpoint weights_only")
    from .deep import TCN

    payload = torch.load(path, map_location="cpu", weights_only=True)
    mean, std = (
        np.asarray(payload["mean"], dtype=float),
        np.asarray(payload["std"], dtype=float),
    )
    if (
        mean.shape != (len(FEATURES),)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Scaler trong checkpoint không hợp lệ")
    if not isinstance(payload["lookback"], int) or not 1 <= payload["lookback"] <= 1024:
        raise ValueError("Lookback trong checkpoint không hợp lệ")
    if payload["lookback"] != metadata["config"]["lookback"]:
        raise ValueError("Lookback khác metadata")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()
    ):
        raise ValueError("Trọng số TCN không hợp lệ")
    model = TCN()
    model.load_state_dict(state, strict=True)
    model.eval()
    return payload


def import_checkpoint(root, folder):
    from . import terminal

    root = Path(root)
    cfg = terminal.config(root)
    with FileLock(str(root / "terminal.lock"), timeout=5):
        original, origin, source = inspect_checkpoint(folder, cfg)
        payload = load_checked_payload(source, original)
        digest = original["sha256"]
        model_id = "tcn_" + digest[:12]
        registry_path = root / "artifacts/local_registry.json"
        registry = terminal.read_json(registry_path, {})
        if model_id in registry:
            existing = registry[model_id]
            if existing.get("source_metadata_identity") != identity(original):
                raise ValueError("Đã có checkpoint cùng SHA với metadata khác")
            if sha256(root / existing["path"]) != digest:
                raise ValueError("Bản checkpoint đã nhập bị thay đổi")
            return existing
        destination = root / "artifacts/imported" / digest
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, destination / "model.pt")
        shutil.copy2(
            source.with_name("tcn_metadata.json"),
            destination / "original_metadata.json",
        )
        manifest = source.with_name("training_snapshot_manifest.json")
        if manifest.exists():
            shutil.copy2(manifest, destination / manifest.name)
        if sha256(destination / "model.pt") != digest:
            raise ValueError("Checkpoint đã đổi trong lúc copy")
        metadata = {
            "type": "tcn",
            "model": model_id,
            "sha256": digest,
            "path": str((destination / "model.pt").relative_to(root)),
            "created_at": original["created_at"],
            "imported_at": utcnow(),
            "threshold": original["threshold"],
            "features": FEATURES,
            "feature_code_sha256": sha256(Path(__file__).with_name("features.py")),
            "config": cfg.to_dict(),
            "training_config": original["config"],
            "source_control": original.get(
                "source_control", {"status": "not_recorded"}
            ),
            "training_rows": original["training_rows"],
            "origin": origin,
            "source_metadata_identity": identity(original),
            "source_metadata_path": str(
                (destination / "original_metadata.json").relative_to(root)
            ),
            "source_metadata_sha256": sha256(destination / "original_metadata.json"),
            "training_features_sha256": original["features_sha256"],
            "validation": None,
            "validation_is_independent_final_test": False,
            "artifact_bytes": (destination / "model.pt").stat().st_size,
            "deployment_status": "imported_checkpoint_paper_only",
            "import_optimizer_steps": 0,
            "normalization": {
                "mean": list(payload["mean"]),
                "std": list(payload["std"]),
                "lookback": payload["lookback"],
            },
        }
        registry[model_id] = metadata
        write_json(registry_path, registry)
        # Registration never changes the active model or the current paper session.
        return metadata


def historical_report(root, metadata):
    path = Path(root) / metadata["source_metadata_path"]
    if sha256(path) != metadata["source_metadata_sha256"]:
        raise ValueError("Metadata gốc đã nhập bị thay đổi")
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_window(root, metadata, symbol):
    from . import terminal
    from .features import build_features

    bars = pd.read_parquet(Path(root) / "data/processed/bars.parquet")
    features = build_features(bars, terminal.config(root))
    group = features[features.symbol.eq(symbol)]
    if group.empty:
        raise ValueError("Chưa đủ dữ liệu cho " + symbol)
    group = group[group.segment.eq(group.segment.iloc[-1])].tail(
        metadata["normalization"]["lookback"]
    )
    if len(group) != metadata["normalization"]["lookback"]:
        raise ValueError("Chưa đủ nến liên tục cho cửa sổ TCN")
    values = (
        group[FEATURES].to_numpy() - np.asarray(metadata["normalization"]["mean"])
    ) / np.asarray(metadata["normalization"]["std"])
    return pd.DataFrame(values, index=group.decision_time, columns=FEATURES)


def import_bundle(root, folder):
    from . import terminal
    from .workflow_model import check_parity

    root, folder = Path(root), Path(folder).expanduser().resolve()
    cfg = terminal.config(root)
    with FileLock(str(root / "terminal.lock"), timeout=5):
        manifest = terminal.read_json(folder / "bundle_manifest.json")
        if not manifest or manifest.get("format") != "quantstream-notebook-v1":
            raise ValueError(
                "Chọn thư mục model đã giải nén có bundle_manifest.json"
            )
        for name, expected in manifest.get("files", {}).items():
            if (
                Path(name).name != name
                or name in (".", "..")
                or sha256(folder / name) != expected
            ):
                raise ValueError("Sai đường dẫn hoặc SHA trong model bundle")
        if not {"metadata.json", "parity.json"} <= set(manifest["files"]):
            raise ValueError("Bundle thiếu metadata hoặc parity")
        original = terminal.read_json(folder / "metadata.json")
        if original.get("features") != FEATURES or original.get("type") != "portable":
            raise ValueError("Sai định dạng model hoặc thứ tự feature")
        if original.get("feature_code_sha256") != sha256(
            Path(__file__).with_name("features.py")
        ):
            raise ValueError("Source feature khác notebook đã train")
        family = original["hyperparameters"]["family"]
        if original.get("forecast_contract_sha256") and original[
            "forecast_contract_sha256"
        ] != sha256(Path(__file__).with_name("forecast_contract.py")):
            raise ValueError("Forecast-unit contract differs from the exported model")
        if family not in ("ridge", "xgboost", "tcn", "gru", "zero_return"):
            raise ValueError("Chưa hỗ trợ họ model này")
        if family in ("tcn", "gru") and original.get("sequence_code_sha256") != sha256(
            Path(__file__).with_name("search.py")
        ):
            raise ValueError("Source kiến trúc neural khác notebook đã train")
        for key in ("symbols", "interval", "horizon", "execution_lag_bars", "lookback"):
            if json.dumps(original["config"][key]) != json.dumps(cfg.to_dict()[key]):
                raise ValueError("Cấu hình inference khác notebook: " + key)
        name = original["model_file"]
        if name not in manifest["files"] or Path(name).name != name:
            raise ValueError("Bundle thiếu trọng số model")
        if sha256(folder / name) != original["sha256"]:
            raise ValueError("SHA trọng số khác metadata notebook")
        parity_delta = check_parity(
            folder / name, original, terminal.read_json(folder / "parity.json")
        )
        model_id = "notebook_" + family + "_" + original["sha256"][:10]
        registry = terminal.read_json(root / "artifacts/local_registry.json", {})
        if model_id in registry:
            existing = registry[model_id]
            if (
                existing.get("source_metadata_sha256")
                != sha256(folder / "metadata.json")
                or sha256(root / existing["path"]) != original["sha256"]
            ):
                raise ValueError("Bản model đã nhập khác bundle này")
            return existing
        destination = root / "artifacts/imported" / model_id
        destination.mkdir(parents=True, exist_ok=False)
        for file in manifest["files"]:
            shutil.copy2(folder / file, destination / file)
        shutil.copy2(
            folder / "bundle_manifest.json", destination / "bundle_manifest.json"
        )
        metadata = {
            **original,
            "model": model_id,
            "path": str((destination / name).relative_to(root)),
            "imported_at": utcnow(),
            "config": cfg.to_dict(),
            "training_config": original["config"],
            "source_metadata_path": str(
                (destination / "metadata.json").relative_to(root)
            ),
            "source_metadata_sha256": sha256(destination / "metadata.json"),
            "origin": {
                "kind": "notebook_bundle",
                "label": (
                    "Notebook 11"
                    if original.get("research_run") == "run008"
                    else (
                        "Notebook 10"
                        if original.get("research_run") == "run007"
                        else "Notebook 09"
                    )
                )
                + " · "
                + family.upper(),
                "status": "hash_and_prediction_parity_verified",
            },
            "parity_max_abs_delta": parity_delta,
            "artifact_bytes": (destination / name).stat().st_size,
            "import_optimizer_steps": 0,
        }
        registry[model_id] = metadata
        write_json(root / "artifacts/local_registry.json", registry)
        return metadata
