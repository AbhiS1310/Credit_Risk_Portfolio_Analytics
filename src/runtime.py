"""Validate trusted local build outputs before loading persisted estimators."""
from __future__ import annotations

import importlib.metadata
import json

import joblib
import pandas as pd

from src.common import ARTIFACTS, PROCESSED, ROOT, sha256


def load_artifacts():
    manifest_path = ARTIFACTS / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Build artifacts are missing. Run python -m src.build from the repository root.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package mismatch for {name}. Install requirements.lock.txt or rebuild artifacts with the installed versions.")
    for name, expected in manifest["source_code_hashes"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Source changed since the build: {name}. Run python -m src.build.")
    for name, expected in manifest["outputs"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or changed: {name}. Run python -m src.build.")
    accounts = pd.read_parquet(PROCESSED / "accounts.parquet")
    return {"manifest": manifest, "accounts": accounts, "model": joblib.load(ARTIFACTS / "model.joblib"), "anomalies": json.loads((ARTIFACTS / "anomalies.json").read_text()), "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()), "quality": json.loads((ARTIFACTS / "quality_report.json").read_text()), "insights": json.loads((ARTIFACTS / "insights.json").read_text()), "importance": pd.read_csv(ARTIFACTS / "importance.csv")}
