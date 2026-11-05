from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.anomalies import detect, robust_reference
from src.reviews import ReviewRepository, export_reviews_csv, make_engine, make_review_engine, make_sqlite_engine, metadata


def test_review_persistence_idempotency_and_version_history(tmp_path):
    path = tmp_path / "review.db"
    engine = create_engine(f"sqlite:///{path}")
    metadata.create_all(engine)
    repository = ReviewRepository(engine, {("a", "v1"), ("a", "v2")})
    token = str(uuid4())
    repository.append(token, "a", "v1", "confirm", "Evidence reviewed")
    repository.append(token, "a", "v1", "confirm", "Evidence reviewed")
    engine.dispose()
    fresh = ReviewRepository(create_engine(f"sqlite:///{path}"), {("a", "v1"), ("a", "v2")})
    assert len(fresh.history("a")) == 1
    fresh.append(str(uuid4()), "a", "v2", "dismiss")
    assert {r["anomaly_version"] for r in fresh.history("a")} == {"v1", "v2"}
    with pytest.raises(ValueError):
        fresh.append(token, "a", "v1", "dismiss")
    with pytest.raises(ValueError):
        fresh.append(str(uuid4()), "unknown", "v1", "confirm")
    with pytest.raises(ValueError):
        fresh.append(str(uuid4()), "a", "v1", "invalid")
    with pytest.raises(ValueError):
        fresh.append(str(uuid4()), "a", "v1", "confirm", "a"*1001)


def test_failed_write_has_no_success_or_fallback():
    engine = create_engine("sqlite://")  # Deliberately no schema.
    repository = ReviewRepository(engine, {("a", "v1")})
    with pytest.raises(SQLAlchemyError):
        repository.append(str(uuid4()), "a", "v1", "confirm")
    with pytest.raises(ValueError, match="Postgres"):
        make_engine("sqlite://")


def test_postgres_transport_requires_tls():
    engine = make_engine("postgresql://user:password@example.com/db?sslmode=disable")
    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.url.query["sslmode"] == "require"


def test_sqlite_factory_and_recent_decision_order(tmp_path):
    from sqlalchemy import update
    from datetime import datetime
    from src.reviews import reviews
    path = tmp_path / "nested" / "reviews.db"
    engine = make_review_engine({"REVIEW_SQLITE_PATH": str(path)})
    assert engine.dialect.name == "sqlite" and path.exists()
    repo = ReviewRepository(engine, {("a", "v1")})
    first = "ffffffff-ffff-4fff-8fff-ffffffffffff"
    second = "00000000-0000-4000-8000-000000000000"
    repo.append(first, "a", "v1", "confirm")
    repo.append(second, "a", "v1", "dismiss")
    # Force an equal timestamp: insertion order must beat random UUID order.
    with engine.begin() as connection:
        connection.execute(update(reviews).values(created_at=datetime(2026, 11, 5)))
    assert repo.history("a")[0]["decision"] == "dismiss"
    engine.dispose()
    reopened = ReviewRepository(make_sqlite_engine(path), {("a", "v1")})
    assert len(reopened.history("a")) == 2
    assert reopened.history("a")[0]["event_id"] == second
    with pytest.raises(ValueError, match="REVIEW_STORAGE"):
        make_review_engine({"REVIEW_STORAGE": "invalid"})


def test_csv_export_roundtrip_and_safe_note(tmp_path):
    import csv
    import io
    repo = ReviewRepository(make_sqlite_engine(tmp_path / "reviews.db"), {("a", "v1")})
    repo.append(str(uuid4()), "a", "v1", "confirm", 'Reviewed, with a "quote"\nand a second line')
    raw = export_reviews_csv(repo.history())
    row = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))[0]
    assert row["note"] == 'Reviewed, with a "quote"\nand a second line'
    assert row["created_at_utc"].endswith("+00:00")
    repo.append(str(uuid4()), "a", "v1", "dismiss", "=HYPERLINK(\"example\")")
    row = list(csv.DictReader(io.StringIO(export_reviews_csv(repo.history()).decode("utf-8-sig"))))[0]
    assert row["note"].startswith("'=HYPERLINK")
    empty = list(csv.DictReader(io.StringIO(export_reviews_csv([]).decode("utf-8-sig"))))
    assert empty == []


def test_anomaly_rules_leave_one_out_and_target_independence():
    n = 40
    frame = pd.DataFrame({"account_id": [f"a{i}" for i in range(n)], "loan_type": ["loan"]*n, "repay_payment_ratio": np.linspace(.9, 1, n), "dpd_deterioration": np.linspace(0, 5, n), "recent_missed": [0]*n, "previous_missed": [0]*n, "bureau_utilization": np.linspace(.2, .6, n), "repay_dpd_max": np.linspace(0, 10, n), "payment_ratio_change": np.linspace(-.01, 0, n), "target_default": [0]*n})
    frame.loc[0, ["repay_payment_ratio", "dpd_deterioration", "recent_missed", "bureau_utilization"]] = [.1, 40, 3, 1.5]
    first, _, summary = detect(frame, "input")
    frame["target_default"] = 1
    second, _, _ = detect(frame, "input")
    assert first == second
    outlier = next(r for r in first if r["account_id"] == "a0")
    assert outlier["peer_count"] == 39
    assert {r["code"] for r in outlier["reasons"]} == {"LOW_PAYMENT_RATIO", "DPD_DETERIORATION", "MISSED_PAYMENT_INCREASE", "UNUSUAL_BUREAU_UTILIZATION"}
    assert outlier["anomaly_version"] == summary["version"]
    assert robust_reference([1, 1, 1]) == (1, None)
    frame["loan_type"] = ["rare"] + ["loan"]*(n-1)
    flags, _, _ = detect(frame, "input")
    assert next(r for r in flags if r["account_id"] == "a0")["peer_segment"] == "Portfolio fallback"
