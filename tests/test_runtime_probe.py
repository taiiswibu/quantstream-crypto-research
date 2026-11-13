"""Preserve conflicting package metadata and archive read-only diagnostic evidence."""
import ast
import json
from pathlib import Path
import runpy
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_probe_preserves_duplicate_metadata_without_picking_a_true_runtime(monkeypatch):
    probe = runpy.run_path(str(ROOT / "scripts/diagnose_run005_runtime.py"))
    class Distribution:
        metadata = {"Name": "numpy"}
        def __init__(self, version):
            self.version = version
        def locate_file(self, path):
            return "/fixture/site-packages"
    monkeypatch.setattr(probe["metadata"], "version", lambda name: "2.1.3")
    result = probe["package_inventory"]([Distribution("2.1.3"), Distribution("1.26.4")])
    assert [row["version"] for row in result["duplicates"]["numpy"]] == ["2.1.3", "1.26.4"]
    assert result["metadata_version_first_match"]["numpy"] == "2.1.3"
    assert result["legacy_last_wins_versions"]["numpy"] == "1.26.4"


def test_probe_keeps_fresh_import_errors_and_does_not_overwrite_earlier_attempt(tmp_path, monkeypatch):
    probe = runpy.run_path(str(ROOT / "scripts/diagnose_run005_runtime.py"))
    original = tmp_path / "QuantStream_Run005_Output.zip"
    original.write_bytes(b"earlier setup failure evidence")
    calls = []
    def command(args, timeout=90):
        calls.append(args)
        if args[-1] == probe["CHILD_PROBE"]:
            return {"returncode": 0, "stdout": 'QUANTSTREAM_PROBE_JSON={"modules":{"numpy":{"error":"binary import error"}}}\n',
                    "stderr": "fixture warning"}
        return {"returncode": 1, "stdout": "fixture missing dependency", "stderr": ""}
    monkeypatch.setitem(probe["run_probe"].__globals__, "command_probe", command)
    monkeypatch.setitem(probe["run_probe"].__globals__, "package_inventory", lambda: {"duplicates": {}})
    first, report = probe["run_probe"](tmp_path)
    second, _ = probe["run_probe"](tmp_path)
    assert first != second and first.exists() and second.exists()
    assert original.read_bytes() == b"earlier setup failure evidence"
    assert report["fresh_process_modules"]["modules"]["numpy"]["error"] == "binary import error"
    assert report["optimizer_steps"] == report["package_mutations"] == report["model_downloads"] == 0
    assert not any("install" in args or "uninstall" in args for args in calls)
    with zipfile.ZipFile(first) as archive:
        saved = json.loads(archive.read("diagnostic.json"))
        assert saved["status"] == "probe_completed" and saved["pip_check"]["returncode"] == 1


def test_probe_notebook_is_standalone_and_has_no_training_or_install_calls(tmp_path):
    builder = runpy.run_path(str(ROOT / "scripts/build_runtime_probe_notebook.py"))
    path = tmp_path / "probe.ipynb"
    builder["build"](path)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        assert not calls.intersection({"fit", "backward", "from_pretrained", "unlink", "rmtree"})
        assert "pip','install" not in cell.source and "pip', 'install" not in cell.source
