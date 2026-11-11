"""Accept metadata-only export differences, never changed data or scientific settings."""
import copy
import json

import pytest

from quantstream import seed_lab as lab
from quantstream.common import sha256, write_json


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    original = json.loads(lab.RUN003_REFERENCE.read_text())
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in original["files"]:
        # Real file bytes are rehashed by the validator; Parquet parsing is a later gate.
        (snapshot / name).write_bytes(("synthetic test bytes: " + name).encode())
        original["files"][name] = sha256(snapshot / name)
    reference = tmp_path / "reference.json"
    write_json(reference, original)
    write_json(snapshot / "manifest.json", original)
    monkeypatch.setattr(lab, "RUN003_REFERENCE", reference)
    monkeypatch.setattr(lab, "SNAPSHOT_SHA", sha256(reference))
    return snapshot, original, reference


def test_exact_and_metadata_only_exports_keep_actual_manifest_identity(snapshots):
    snapshot, original, reference = snapshots
    exact = lab.verify_run003_snapshot(snapshot)
    assert exact["manifest_relation"] == "exact"
    current = copy.deepcopy(original)
    current["created_at"] = "2026-11-10T16:15:47.939870+00:00"
    current["source_control"] = {"commit": "earlier export, not executing source"}
    current.pop("code_files")
    write_json(snapshot / "manifest.json", current)
    before = (snapshot / "manifest.json").read_bytes()
    result = lab.verify_run003_snapshot(snapshot)
    assert result["manifest_relation"] == "data_equivalent_metadata_diff"
    assert result["metadata_differences"] == ["code_files", "created_at", "source_control"]
    assert result["manifest_sha256"] == sha256(snapshot / "manifest.json")
    assert result["reference_manifest_sha256"] == sha256(reference)
    assert result["manifest_sha256"] != result["reference_manifest_sha256"]
    assert result["data_file_hashes"] == original["files"]
    assert (snapshot / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("field", ["config", "split", "feature_columns", "full_features_sha256",
                                    "files", "contains_final_holdout", "unknown_setting"])
def test_metadata_acceptance_does_not_relax_data_contract(snapshots, field):
    snapshot, original, _ = snapshots
    current = copy.deepcopy(original)
    if field == "config":
        current[field]["horizon"] = 8
    elif field == "split":
        current[field]["folds"][-1]["validation_start"] = "2025-09-01T00:00:00+00:00"
    elif field == "feature_columns":
        current[field].reverse()
    elif field == "files":
        (snapshot / "bars.parquet").write_bytes(b"changed data with a self-consistent manifest")
        current[field]["bars.parquet"] = sha256(snapshot / "bars.parquet")
    elif field == "contains_final_holdout":
        current[field] = True
    else:
        current[field] = "changed"
    write_json(snapshot / "manifest.json", current)
    with pytest.raises(ValueError, match="contract|Development-only"):
        lab.verify_run003_snapshot(snapshot)


def test_file_bytes_are_rehashed_against_reference_not_just_manifest_claims(snapshots):
    snapshot, _, _ = snapshots
    (snapshot / "validation.parquet").write_bytes(b"corrupted without editing manifest")
    with pytest.raises(ValueError, match="Data file differs.*validation.parquet"):
        lab.verify_run003_snapshot(snapshot)


def test_reference_cannot_be_silently_replaced(snapshots):
    snapshot, _, reference = snapshots
    reference.write_bytes(reference.read_bytes() + b" ")
    with pytest.raises(ValueError, match="reference manifest checksum mismatch"):
        lab.verify_run003_snapshot(snapshot)


def test_committed_reference_matches_audited_run003_manifest():
    assert sha256(lab.RUN003_REFERENCE) == "c8e0112916e6cea733338ead4412d6e9553e4c5844a2d200d1fd38e31c747fd0"
