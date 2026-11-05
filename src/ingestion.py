"""Read immutable inputs and report quality without silently deleting records."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import RAW, ROOT, sha256

SCHEMA = {
    "customers": "customer_id age gender marital_status education_level employment_type occupation_sector years_employed annual_income region city_tier num_dependents housing_type years_at_residence customer_since".split(),
    "loan_accounts": "account_id customer_id loan_type loan_purpose loan_amount interest_rate_pct tenure_months origination_date disbursal_channel collateral_flag target_default".split(),
    "bureau_credit": "bureau_id customer_id credit_type lender_category sanctioned_amount outstanding_amount account_status days_past_due date_opened date_closed".split(),
    "repayment_history": "account_id payment_month emi_due emi_paid days_past_due payment_status outstanding_balance".split(),
    "collections_events": "collection_event_id account_id event_date action_type outcome amount_recovered".split(),
}
KEYS = {"customers": ["customer_id"], "loan_accounts": ["account_id"], "bureau_credit": ["bureau_id"], "repayment_history": ["account_id", "payment_month"], "collections_events": ["collection_event_id"]}
DATES = {"customers": ["customer_since"], "loan_accounts": ["origination_date"], "bureau_credit": ["date_opened", "date_closed"], "repayment_history": [], "collections_events": ["event_date"]}
NUMERIC = {
    "customers": "age years_employed annual_income num_dependents years_at_residence".split(),
    "loan_accounts": "loan_amount interest_rate_pct tenure_months collateral_flag target_default".split(),
    "bureau_credit": "sanctioned_amount outstanding_amount days_past_due".split(),
    "repayment_history": "payment_month emi_due emi_paid days_past_due outstanding_balance".split(),
    "collections_events": ["amount_recovered"],
}
KNOWN = {
    "gender": ["Female", "Male"],
    "loan_type": ["Personal Loan", "Auto Loan", "Credit Card", "Mortgage", "Business Loan"],
    "region": ["North", "South", "East", "West", "Central"],
    "employment_type": ["Salaried", "Self-Employed", "Business Owner", "Contract", "Unemployed"],
    "account_status": ["Active", "Closed", "Written-Off"],
    "payment_status": ["On-Time", "Late", "Missed"],
}


def stage_raw() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    source = ROOT / "AI Case Study 1" / "Case 2" / "files (10)"
    for name in SCHEMA:
        dest = RAW / f"{name}.csv"
        original = source / dest.name
        if not dest.exists():
            if not original.exists():
                raise FileNotFoundError(f"Missing input: {dest}")
            shutil.copyfile(original, dest)
        if original.exists() and sha256(original) != sha256(dest):
            raise ValueError(f"Raw copy differs from supplied original: {dest.name}")


def load_sources(raw_dir: Path = RAW):
    frames, report = {}, {"tables": {}, "warnings": {}, "normalizations": {}}
    for name, columns in SCHEMA.items():
        path = raw_dir / f"{name}.csv"
        frame = pd.read_csv(path, dtype="string")
        if set(frame.columns) != set(columns):
            raise ValueError(f"{name}: schema mismatch")
        missing = frame.isna().sum().astype(int).to_dict()
        for col in frame:
            frame[col] = frame[col].str.strip().replace("", pd.NA)
        for col in NUMERIC[name]:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
            if np.isinf(frame[col]).any() or (frame[col].dropna() < 0).any():
                raise ValueError(f"{name}.{col}: negative or infinite value")
        for col in DATES[name]:
            frame[col] = pd.to_datetime(frame[col], format="%Y-%m-%d", errors="raise")
        keys = KEYS[name]
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: null/duplicate key {keys}")
        report["tables"][name] = {"rows": len(frame), "sha256": sha256(path), "missing": missing, "schema": {k: str(v) for k, v in frame.dtypes.items()}, "duplicate_keys": 0}
        frames[name] = frame
    c, l, b, r, e = (frames[x] for x in SCHEMA)
    aliases = c["gender"].isin(["M", "F"])
    report["normalizations"]["gender_aliases"] = int(aliases.sum())
    c["gender"] = c["gender"].replace({"M": "Male", "F": "Female"})
    for name, frame in frames.items():
        for col, allowed in KNOWN.items():
            if col in frame:
                report["warnings"][f"{name}.{col}.unknown_category"] = int((frame[col].notna() & ~frame[col].isin(allowed)).sum())
    for child, foreign, parent, key in [(l, "customer_id", c, "customer_id"), (b, "customer_id", c, "customer_id"), (r, "account_id", l, "account_id"), (e, "account_id", l, "account_id")]:
        if child[foreign].isna().any() or ~child[foreign].isin(parent[key]).all():
            raise ValueError(f"Orphan/null foreign key: {foreign}")
    for col in ["target_default", "collateral_flag"]:
        if not l[col].isin([0, 1]).all():
            raise ValueError(f"Invalid {col}")
    if not (l.tenure_months > 0).all() or not (l.tenure_months % 1 == 0).all():
        raise ValueError("Invalid loan tenure")
    if not ((r.payment_month > 0) & (r.payment_month % 1 == 0)).all():
        raise ValueError("Invalid repayment month")
    report["foreign_keys"] = "all valid"
    warnings = report["warnings"]
    warnings["closed_bureau_without_date"] = int(((b.account_status == "Closed") & b.date_closed.isna()).sum())
    warnings["active_bureau_with_close_date"] = int(((b.account_status == "Active") & b.date_closed.notna()).sum())
    warnings["bureau_closed_before_opened"] = int((b.date_closed < b.date_opened).sum())
    warnings["bureau_zero_sanctioned"] = int((b.sanctioned_amount == 0).sum())
    warnings["repayment_overpaid"] = int((r.emi_paid > r.emi_due).sum())
    warnings["status_on_time_positive_dpd"] = int(((r.payment_status == "On-Time") & (r.days_past_due > 0)).sum())
    warnings["status_late_zero_dpd"] = int(((r.payment_status == "Late") & (r.days_past_due == 0)).sum())
    warnings["status_missed_positive_payment"] = int(((r.payment_status == "Missed") & (r.emi_paid > 0)).sum())
    warnings["implausible_age"] = int(((c.age < 18) | (c.age > 100)).sum())
    warnings["employment_exceeds_adult_lifetime"] = int((c.years_employed > c.age - 18).sum())
    warnings["residence_exceeds_age"] = int((c.years_at_residence > c.age).sum())
    lc = l.merge(c[["customer_id", "customer_since"]], on="customer_id", validate="many_to_one")
    warnings["loan_before_customer_since"] = int((lc.origination_date < lc.customer_since).sum())
    ec = e.merge(l[["account_id", "origination_date"]], on="account_id", validate="many_to_one")
    warnings["collection_before_origination"] = int((ec.event_date < ec.origination_date).sum())
    bc = b.merge(l.groupby("customer_id").origination_date.max().rename("latest_loan_date"), on="customer_id", how="left", validate="many_to_one")
    warnings["bureau_opened_after_customer_latest_loan"] = int((bc.date_opened > bc.latest_loan_date).sum())
    return frames, report
