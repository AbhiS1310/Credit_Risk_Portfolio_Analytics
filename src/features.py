"""Aggregate children before joining; all outputs retain one row per account."""
from __future__ import annotations

import numpy as np
import pandas as pd


def divide(a, b):
    return a / b.replace(0, np.nan)


def bureau_features(b: pd.DataFrame) -> pd.DataFrame:
    x = b.copy()
    for status in ["Active", "Closed", "Written-Off"]:
        x["status_" + status] = (x.account_status == status).astype(int)
    for threshold in [0, 30, 90]:
        x[f"dpd_gt_{threshold}"] = (x.days_past_due > threshold).astype(int)
    g = x.groupby("customer_id", sort=True)
    out = g.agg(
        bureau_lines=("bureau_id", "size"), bureau_active=("status_Active", "sum"),
        bureau_closed=("status_Closed", "sum"), bureau_written_off=("status_Written-Off", "sum"),
        bureau_lenders=("lender_category", "nunique"), bureau_credit_types=("credit_type", "nunique"),
        bureau_sanctioned=("sanctioned_amount", lambda s: s.sum(min_count=1)),
        bureau_outstanding=("outstanding_amount", lambda s: s.sum(min_count=1)),
        bureau_dpd_max=("days_past_due", "max"), bureau_dpd_mean=("days_past_due", "mean"),
        bureau_dpd_observed=("days_past_due", "count"),
        bureau_dpd_gt_0=("dpd_gt_0", "sum"), bureau_dpd_gt_30=("dpd_gt_30", "sum"), bureau_dpd_gt_90=("dpd_gt_90", "sum"),
    )
    out["bureau_dpd_missing"] = out.bureau_lines - out.bureau_dpd_observed
    out["bureau_dpd_coverage"] = divide(out.bureau_dpd_observed, out.bureau_lines)
    out["bureau_utilization"] = divide(out.bureau_outstanding, out.bureau_sanctioned)
    # Counts with unknown DPD cannot be interpreted as zero delinquent lines.
    out.loc[out.bureau_dpd_observed == 0, [f"bureau_dpd_gt_{n}" for n in [0, 30, 90]]] = np.nan
    return out.reset_index()


def repayment_features(r: pd.DataFrame) -> pd.DataFrame:
    x = r.sort_values(["account_id", "payment_month"]).copy()
    x["zero_payment"] = (x.emi_paid == 0).astype(int)
    x["late"] = (x.payment_status == "Late").astype(int)
    x["missed"] = (x.payment_status == "Missed").astype(int)
    for n in [0, 30, 90]:
        x[f"dpd_gt_{n}"] = (x.days_past_due > n).astype(int)
    out = x.groupby("account_id").agg(
        repay_months=("payment_month", "size"), repay_due=("emi_due", lambda s: s.sum(min_count=1)),
        repay_paid=("emi_paid", lambda s: s.sum(min_count=1)), repay_zero_payments=("zero_payment", "sum"),
        repay_late=("late", "sum"), repay_missed=("missed", "sum"),
        repay_dpd_mean=("days_past_due", "mean"), repay_dpd_max=("days_past_due", "max"),
        repay_dpd_gt_0=("dpd_gt_0", "sum"), repay_dpd_gt_30=("dpd_gt_30", "sum"), repay_dpd_gt_90=("dpd_gt_90", "sum"),
    )
    latest = x.groupby("account_id").tail(1).set_index("account_id")
    out["latest_outstanding"] = latest.outstanding_balance
    out["repay_latest_dpd"] = latest.days_past_due
    out["repay_latest_month"] = latest.payment_month
    out["repay_payment_ratio"] = divide(out.repay_paid, out.repay_due)
    x["months_from_latest"] = x.groupby("account_id").payment_month.transform("max") - x.payment_month
    for label, lower, upper in [("recent", 0, 2), ("previous", 3, 5)]:
        w = x[x.months_from_latest.between(lower, upper)].groupby("account_id").agg(
            n=("payment_month", "size"), dpd=("days_past_due", "mean"),
            paid=("emi_paid", lambda s: s.sum(min_count=1)), due=("emi_due", lambda s: s.sum(min_count=1)),
            missed=("missed", "sum"),
        )
        valid = w.n == 3
        out[f"{label}_dpd"] = w.dpd.where(valid)
        out[f"{label}_payment_ratio"] = divide(w.paid, w.due).where(valid)
        out[f"{label}_missed"] = w.missed.where(valid)
    out["dpd_deterioration"] = out.recent_dpd - out.previous_dpd
    out["payment_ratio_change"] = out.recent_payment_ratio - out.previous_payment_ratio
    return out.reset_index()


def collection_features(e: pd.DataFrame) -> pd.DataFrame:
    out = e.groupby("account_id").agg(
        collection_events=("collection_event_id", "size"), collection_recovered=("amount_recovered", lambda s: s.sum(min_count=1)),
        collection_latest_date=("event_date", "max"),
    )
    for column in ["action_type", "outcome"]:
        pivot = pd.crosstab(e.account_id, e[column])
        pivot.columns = ["collection_" + column + "_" + s.lower().replace(" ", "_").replace("/", "_") for s in pivot.columns]
        out = out.join(pivot)
    return out.reset_index()


def build_features(frames, report):
    c, l, b, r, e = (frames[n] for n in ["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"])
    out = l.merge(c, on="customer_id", how="left", validate="many_to_one")
    checks = [{"join": "customers", "rows": len(out)}]
    for label, child, key, validation in [("bureau", bureau_features(b), "customer_id", "many_to_one"), ("repayment", repayment_features(r), "account_id", "one_to_one"), ("collections", collection_features(e), "account_id", "one_to_one")]:
        out = out.merge(child, on=key, how="left", validate=validation)
        if len(out) != len(l) or not out.account_id.is_unique:
            raise ValueError(f"Join cardinality failure: {label}")
        checks.append({"join": label, "rows": len(out)})
    for col in ["bureau_lines", "bureau_active", "bureau_closed", "bureau_written_off", "bureau_lenders", "bureau_credit_types", "bureau_dpd_observed", "bureau_dpd_missing"]:
        out[col] = out[col].fillna(0)
    out["has_bureau_history"] = (out.bureau_lines > 0).astype(int)
    out["has_repayment_history"] = out.repay_months.notna().astype(int)
    out["has_collection_history"] = out.collection_events.notna().astype(int)
    for col in out.filter(regex="^collection_").columns:
        if col != "collection_latest_date":
            out[col] = out[col].fillna(0)
    out["income_band"] = pd.cut(out.annual_income, [-np.inf, 25000, 50000, 100000, np.inf], right=False, labels=["<25k", "25k-<50k", "50k-<100k", "100k+"]).astype("string").fillna("Unknown")
    out["age_band"] = pd.cut(out.age, [-np.inf, 25, 35, 45, 55, np.inf], right=False, labels=["<25", "25-34", "35-44", "45-54", "55+"]).astype("string").fillna("Unknown")
    out["origination_year"] = out.origination_date.dt.year.astype("Int64").astype("string").fillna("Unknown")
    report["joins"] = checks
    report["reconciliation"] = {"accounts": len(out), "defaults": int(out.target_default.sum()), "principal": float(out.loan_amount.sum()), "latest_outstanding": float(out.latest_outstanding.sum()), "collection_recovered": float(out.collection_recovered.sum())}
    return out
