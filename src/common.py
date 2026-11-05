"""Paths, versioning, and strict JSON serialization shared by build and runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
FEATURE_VERSION = "account-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert), encoding="utf-8")
