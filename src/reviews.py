"""Append-only reviews: SQLite assessment default, optional managed Postgres."""
from __future__ import annotations

import argparse
import csv
import io
import os
from datetime import timezone
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import CheckConstraint, Column, DateTime, Index, MetaData, String, Table, create_engine, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url

from src.common import ROOT

metadata = MetaData()
reviews = Table(
    "anomaly_reviews", metadata,
    Column("event_id", String(36), primary_key=True),
    Column("account_id", String(40), nullable=False),
    Column("anomaly_version", String(64), nullable=False),
    Column("decision", String(10), nullable=False),
    Column("note", String(1000), nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
    CheckConstraint("decision in ('confirm', 'dismiss')", name="valid_decision"),
)
Index("review_history_lookup", reviews.c.account_id, reviews.c.anomaly_version, reviews.c.created_at, reviews.c.event_id)


def make_engine(database_url, *, testing=False):
    if not database_url:
        raise ValueError("DATABASE_URL is not configured")
    url = make_url(database_url)
    if url.get_backend_name() == "postgres":
        url = url.set(drivername="postgresql+psycopg")
    if url.get_backend_name() != "postgresql":
        if testing and url.get_backend_name() == "sqlite":
            return create_engine(url)
        raise ValueError("Production reviews require managed Postgres")
    url = url.set(drivername="postgresql+psycopg")
    query = dict(url.query)
    query["sslmode"] = "require"
    query["connect_timeout"] = "5"
    return create_engine(url.set(query=query), pool_pre_ping=True, pool_size=2, max_overflow=2)


def make_sqlite_engine(path):
    """Create the local store once; Streamlit caches the returned engine."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(make_url("sqlite://").set(database=str(path)), connect_args={"timeout": 10, "check_same_thread": False}, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def make_review_engine(settings):
    backend = settings.get("REVIEW_STORAGE") or "sqlite"
    if backend == "sqlite":
        path = Path(settings.get("REVIEW_SQLITE_PATH") or "data/reviews.db")
        if not path.is_absolute():
            path = ROOT / path
        return make_sqlite_engine(path)
    if backend == "postgres":
        return make_engine(settings.get("DATABASE_URL", ""))
    raise ValueError("REVIEW_STORAGE must be sqlite or postgres")


def export_reviews_csv(history):
    """UTF-8 CSV of committed events; notes are exported as spreadsheet-safe text."""
    output = io.StringIO(newline="")
    fields = ["event_id", "account_id", "anomaly_version", "decision", "note", "created_at_utc"]
    writer = csv.DictWriter(output, fieldnames=fields, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for event in history:
        row = {field: event[field] for field in fields[:-1]}
        note = str(row["note"])
        if note.lstrip().startswith(("=", "+", "-", "@")) or note.startswith(("\t", "\r", "\n")):
            row["note"] = "'" + note
        timestamp = event["created_at"]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        row["created_at_utc"] = timestamp.astimezone(timezone.utc).isoformat()
        writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")


class ReviewRepository:
    def __init__(self, engine, valid_flags):
        self.engine = engine
        self.valid_flags = set(valid_flags)

    def append(self, event_id, account_id, version, decision, note=""):
        UUID(event_id)
        if (account_id, version) not in self.valid_flags:
            raise ValueError("Unknown account/anomaly version")
        if decision not in {"confirm", "dismiss"} or len(note) > 1000:
            raise ValueError("Invalid decision or note")
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        values = {"event_id": event_id, "account_id": account_id, "anomaly_version": version, "decision": decision, "note": note}
        with self.engine.begin() as connection:
            connection.execute(insert(reviews).values(**values).on_conflict_do_nothing(index_elements=["event_id"]))
            saved = connection.execute(select(reviews).where(reviews.c.event_id == event_id)).mappings().one()
            if any(saved[k] != v for k, v in values.items()):
                raise ValueError("Submission ID already used for a different decision")
        return dict(saved)

    def history(self, account_id=None):
        statement = select(reviews)
        if account_id is not None:
            statement = statement.where(reviews.c.account_id == account_id)
        # SQLite CURRENT_TIMESTAMP has second precision. rowid preserves insertion
        # order when an analyst changes a decision twice within the same second.
        tie_breaker = literal_column("rowid").desc() if self.engine.dialect.name == "sqlite" else reviews.c.event_id.desc()
        statement = statement.order_by(reviews.c.created_at.desc(), tie_breaker)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", required=True)
    parser.parse_args()
    load_dotenv(ROOT / ".env")
    engine = make_review_engine({k: os.getenv(k, "") for k in ["REVIEW_STORAGE", "REVIEW_SQLITE_PATH", "DATABASE_URL"]})
    metadata.create_all(engine)
    print(f"{engine.dialect.name} review schema initialized. SQLite storage is local; export CSV before a hosted restart/redeploy.")


if __name__ == "__main__":
    main()
