"""Regression coverage for Windows builds deployed through Git to Linux."""
import json
from pathlib import Path

import pytest

from src.common import sha256, source_sha256, write_json


def test_source_hash_tolerates_line_endings_but_detects_code_changes(tmp_path):
    lf = tmp_path / "linux.py"
    crlf = tmp_path / "windows.py"
    lf.write_bytes(b"a = 1\nprint(a)\n")
    crlf.write_bytes(b"a = 1\r\nprint(a)\r\n")
    assert sha256(lf) != sha256(crlf)
    assert source_sha256(lf) == source_sha256(crlf)
    crlf.write_bytes(b"a = 2\r\nprint(a)\r\n")
    assert source_sha256(lf) != source_sha256(crlf)


def test_json_output_always_uses_lf(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"value": 42})
    assert b"\r" not in path.read_bytes()
    assert json.loads(path.read_bytes()) == {"value": 42}


def test_runtime_accepts_crlf_checkout_and_rejects_real_edits(tmp_path, monkeypatch):
    import src.runtime as runtime
    src = tmp_path / "src"
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "data" / "processed"
    for folder in [src, artifacts, processed]:
        folder.mkdir(parents=True)
    source = src / "example.py"
    source.write_bytes(b"value = 1\n")
    artifact = artifacts / "metrics.json"
    write_json(artifact, {"auc": .7})
    manifest = {"source_hash_mode": "lf-normalized-v1", "packages": {}, "source_code_hashes": {"src/example.py": source_sha256(source)}, "outputs": {"artifacts/metrics.json": sha256(artifact)}}
    write_json(artifacts / "manifest.json", manifest)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "ARTIFACTS", artifacts)
    monkeypatch.setattr(runtime, "PROCESSED", processed)
    # Reach the load stage only after every integrity check has passed.
    class PassedChecks(Exception):
        pass
    monkeypatch.setattr(runtime.pd, "read_parquet", lambda *a: (_ for _ in ()).throw(PassedChecks()))
    source.write_bytes(b"value = 1\r\n")
    with pytest.raises(PassedChecks):
        runtime.load_artifacts()
    source.write_bytes(b"value = 2\r\n")
    with pytest.raises(ValueError, match="Source changed"):
        runtime.load_artifacts()
    source.write_bytes(b"value = 1\n")
    artifact.write_bytes(b'{"auc": 0.99}')
    with pytest.raises(ValueError, match="Artifact missing or changed"):
        runtime.load_artifacts()


def test_lf_checkout_loads_generated_build(tmp_path):
    """Simulate Git's Linux checkout without the untracked original input pack."""
    import shutil
    import subprocess
    import sys
    from src.common import ARTIFACTS, ROOT
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    for name in manifest["source_code_hashes"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    for name in list(manifest["outputs"]) + ["artifacts/manifest.json"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    result = subprocess.run([sys.executable, "-c", "from src.runtime import load_artifacts; d=load_artifacts(); assert len(d['accounts'])==14685; print('LF checkout passed')"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "LF checkout passed" in result.stdout
