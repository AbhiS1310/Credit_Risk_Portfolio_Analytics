"""Reproducible offline build: python -m src.build."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time

import joblib
from threadpoolctl import threadpool_limits

from src.analytics import generate_insights
from src.anomalies import detect
from src.common import ARTIFACTS, FEATURE_VERSION, PROCESSED, ROOT, SEED, digest, sha256, source_sha256, write_json
from src.features import build_features
from src.ingestion import load_sources, stage_raw
from src.modeling import train_models


def main():
    started = time.monotonic()
    for directory in [PROCESSED, ARTIFACTS]:
        directory.mkdir(parents=True, exist_ok=True)
    print("1/5 Staging and validating five raw tables", flush=True)
    stage_raw()
    frames, quality = load_sources()
    source_hashes = {k: v["sha256"] for k, v in quality["tables"].items()}
    input_hash = digest(source_hashes)
    print("2/5 Building account features and reconciling joins", flush=True)
    accounts = build_features(frames, quality)
    write_json(ARTIFACTS / "quality_report.json", quality)
    for name, frame in frames.items():
        frame.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    with threadpool_limits(limits=2):
        print("3/5 Training logistic baseline and four gradient boosting candidates", flush=True)
        bundle, metrics, importance, scores = train_models(accounts)
        print("4/5 Computing leave-one-out peer anomalies and Isolation Forest", flush=True)
        anomalies, detector, anomaly_summary = detect(accounts, input_hash)
    accounts = accounts.merge(scores, on="account_id", validate="one_to_one")
    accounts.to_parquet(PROCESSED / "accounts.parquet", index=False)
    scores.to_parquet(ARTIFACTS / "scores.parquet", index=False)
    importance.to_csv(ARTIFACTS / "importance.csv", index=False, lineterminator="\n")
    joblib.dump(bundle, ARTIFACTS / "model.joblib", compress=3)
    joblib.dump(detector, ARTIFACTS / "anomaly_model.joblib", compress=3)
    write_json(ARTIFACTS / "metrics.json", metrics)
    write_json(ARTIFACTS / "anomalies.json", anomalies)
    write_json(ARTIFACTS / "anomaly_summary.json", anomaly_summary)
    write_json(ARTIFACTS / "insights.json", generate_insights(accounts))
    print("5/5 Recording versions and output integrity", flush=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "uncommitted-workspace"
    code_hashes = {p.relative_to(ROOT).as_posix(): source_sha256(p) for p in sorted((ROOT / "src").glob("*.py"))}
    outputs = sorted(PROCESSED.glob("*.parquet")) + [p for p in sorted(ARTIFACTS.iterdir()) if p.name != "manifest.json" and p.is_file()]
    manifest = {"build_id": digest({"inputs": source_hashes, "code": code_hashes, "features": FEATURE_VERSION})[:20], "feature_version": FEATURE_VERSION, "input_hash": input_hash, "source_hash_mode": "lf-normalized-v1", "source_hashes": source_hashes, "source_code_hashes": code_hashes, "revision": revision, "seed": SEED, "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "streamlit", "joblib", "pyarrow"]}, "outputs": {p.relative_to(ROOT).as_posix(): sha256(p) for p in outputs}}
    write_json(ARTIFACTS / "manifest.json", manifest)
    from src.reporting import render_documents
    render_documents()
    print(json.dumps({"build_id": manifest["build_id"], "selected": metrics["selected"], "test": metrics[metrics["selected"]]["test"], "anomalies": len(anomalies), "seconds": round(time.monotonic() - started, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
