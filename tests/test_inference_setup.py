"""Bootstrap regression: duplicate metadata must not downgrade imported NumPy."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantstream import inference_setup as setup

ROOT = Path(__file__).resolve().parents[1]


def reviewed_environment():
    expected = json.loads((ROOT / "reports/experiments/run-004-seed-stability/kaggle-preflight-lock-rerun-001.json").read_text())["environment"]
    audit = json.loads(setup.AUDIT.read_text())
    return expected, audit


def test_reviewed_probe_explains_first_and_last_numpy_metadata():
    expected, audit = reviewed_environment()
    assert audit["artifact_sha256"] == "547404c9cc40228e0578b54726a506f8a54e183cd476a0aa1f3cb21c5eb20347"
    assert audit["numpy_metadata_first"] == audit["actual_module_versions"]["numpy"] == "2.1.3"
    assert audit["numpy_metadata_legacy_last"] == expected["packages"]["numpy"] == "1.26.4"
    assert audit["old_inventory_count"] == audit["current_inventory_count"] == 937
    assert set(audit["legacy_inventory_diff"]) == {"chronos-forecasting", "torchao"}


def test_only_reviewed_setup_repairs_are_allowed():
    expected, _ = reviewed_environment()
    before = {**expected["packages"], "torchao": "0.10.0"}
    del before["chronos-forecasting"]
    assert set(setup.approved_repairs(expected["packages"], before)) == {"chronos-forecasting", "torchao"}
    assert setup.approved_repairs(expected["packages"], expected["packages"]) == {}
    for name, version in (("numpy", "2.1.3"), ("torchao", "0.11.0"), ("chronos-forecasting", "2.2.0")):
        drift = {**before, name: version}
        with pytest.raises(ValueError, match="Unreviewed package drift"):
            setup.approved_repairs(expected["packages"], drift)


def test_bootstrap_restores_legacy_inventory_without_touching_numpy_torch(tmp_path, monkeypatch):
    expected, audit = reviewed_environment()
    current = {**expected["packages"], "torchao": "0.10.0"}
    del current["chronos-forecasting"]
    actions = []
    def inventory():
        candidates = {name: [{"version": version, "root": "/fixture"}] for name, version in current.items()}
        candidates["numpy"] = audit["numpy_metadata_candidates"]
        return {"legacy_packages": current.copy(), "candidates": candidates, "duplicates": {"numpy": candidates["numpy"]}}
    def version(name):
        if name == "numpy":
            return "2.1.3"
        if name not in current:
            raise setup.metadata.PackageNotFoundError(name)
        return current[name]
    def pip(args, **kwargs):
        actions.append(args)
        assert args[:3] == [setup.sys.executable, "-m", "pip"]
        if "uninstall" in args:
            assert args[3:] == ["uninstall", "-y", "torchao"]
            del current["torchao"]
        else:
            assert args[3:] == ["install", "--no-deps", "--only-binary=:all:", "chronos-forecasting==2.3.2"]
            current["chronos-forecasting"] = "2.3.2"
        return SimpleNamespace(returncode=0, stdout="fixture", stderr="", check_returncode=lambda: None)
    effective = {"python": expected["python"], "cuda_available": True, "numpy_bridge_sum": 2.0,
                 **{key: expected[key] for key in ("cuda", "cudnn", "gpu", "gpu_capability")},
                 "modules": {name: {"version": value} for name, value in audit["actual_module_versions"].items()}}
    monkeypatch.setattr(setup.platform, "python_version", lambda: expected["python"])
    monkeypatch.setattr(setup, "inventory", inventory)
    monkeypatch.setattr(setup.metadata, "version", version)
    monkeypatch.setattr(setup, "loaded_conflicts", lambda: [])
    monkeypatch.setattr(setup, "kernel_numpy_version", lambda: "2.1.3")
    monkeypatch.setattr(setup.subprocess, "run", pip)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(setup, "dependency_errors", lambda: [])
    monkeypatch.setattr(setup, "loaded_runtime", lambda: effective)
    monkeypatch.setattr(setup, "synthetic_adapter_forward", lambda: {"optimizer_steps": 0, "finite_forward": True})
    report = setup.prepare_runtime(expected, tmp_path / "setup")
    assert report["status"] == "completed" and report["legacy_inventory_diff"] == {}
    assert len(actions) == 2 and current == expected["packages"]
    assert current["numpy"] == "1.26.4" and report["loaded_runtime"]["modules"]["numpy"]["version"] == "2.1.3"
    assert all("numpy" not in args and "torch" not in args for args in actions)
    assert report["optimizer_steps"] == report["model_downloads"] == 0
    assert json.loads((tmp_path / "setup/setup.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        setup.prepare_runtime(expected, tmp_path / "setup")


def test_loaded_runtime_contract_detects_a_real_numpy_change():
    expected, audit = reviewed_environment()
    effective = {"python": expected["python"], "cuda_available": True, "numpy_bridge_sum": 2.0,
                 **{key: expected[key] for key in ("cuda", "cudnn", "gpu", "gpu_capability")},
                 "modules": {name: {"version": value} for name, value in audit["actual_module_versions"].items()}}
    setup.validate_loaded(effective, expected, audit)
    changed = copy.deepcopy(effective)
    changed["modules"]["numpy"]["version"] = "2.2.0"
    with pytest.raises(ValueError, match="Imported module differs.*numpy"):
        setup.validate_loaded(changed, expected, audit)


def test_dependency_check_uses_active_versions_and_ignores_unused_extras(monkeypatch):
    requirements = {"root": ["dep>=2", "optional; extra == 'dev'"], "dep": [], "numpy": []}
    monkeypatch.setattr(setup.metadata, "requires", lambda name: requirements[name])
    monkeypatch.setattr(setup.metadata, "version", lambda name: {"dep": "1.0", "numpy": "2.1.3"}[name])
    errors = setup.dependency_errors(("root",))
    assert errors == [{"package": "root", "requires": "dep>=2", "actual": "1.0"}]


def test_synthetic_adapter_check_has_no_backward_or_fit():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("chronos")
    result = setup.synthetic_adapter_forward()
    assert result["finite_forward"] and result["synthetic_only"]
    assert result["optimizer_steps"] == result["model_downloads"] == 0
