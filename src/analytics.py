"""Shared calculations for UI and Q&A. Rates retain their denominators."""
from __future__ import annotations

import pandas as pd

DIMENSIONS = ["region", "employment_type", "loan_type", "income_band", "age_band", "origination_year"]
METRICS = {
    "account_count": {"definition": "Number of in-scope loan accounts", "unit": "accounts", "fields": ["loan_accounts.account_id"]},
    "customer_count": {"definition": "Distinct customers with in-scope loans", "unit": "customers", "fields": ["loan_accounts.customer_id"]},
    "default_count": {"definition": "Sum of observed default labels", "unit": "accounts", "fields": ["loan_accounts.target_default"]},
    "default_rate": {"definition": "Defaulted labeled accounts / labeled accounts", "unit": "fraction", "fields": ["loan_accounts.target_default", "loan_accounts.account_id"]},
    "original_principal": {"definition": "Sum of original loan amounts", "unit": "USD", "fields": ["loan_accounts.loan_amount"]},
    "observed_exposure": {"definition": "Sum of latest observed outstanding per account; not a common-date balance", "unit": "USD", "fields": ["repayment_history.payment_month", "repayment_history.outstanding_balance"]},
}


def apply_filters(frame, filters):
    out = frame
    for field, values in filters.items():
        if field not in DIMENSIONS:
            raise ValueError("Unsupported filter")
        # An empty explicit list means empty intersection, never unrestricted.
        if values is not None:
            out = out[out[field].astype(str).isin([str(v) for v in values])]
    return out


def kpis(frame):
    labels = frame.target_default.dropna()
    return {
        "account_count": len(frame), "customer_count": int(frame.customer_id.nunique()),
        "default_count": int(labels.sum()), "labeled_count": len(labels),
        "default_rate": float(labels.mean()) if len(labels) else None,
        "original_principal": float(frame.loan_amount.sum()) if frame.loan_amount.notna().any() else None,
        "observed_exposure": float(frame.latest_outstanding.sum()) if frame.latest_outstanding.notna().any() else None,
        "exposure_coverage": int(frame.latest_outstanding.notna().sum()),
    }


def grouped(frame, dimensions):
    if not dimensions:
        return pd.DataFrame([kpis(frame)])
    records = []
    for key, group in frame.groupby(dimensions, dropna=False, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(dimensions, map(str, key))) | kpis(group))
    return pd.DataFrame(records, columns=dimensions + list(kpis(frame)))


def generate_insights(frame):
    """Measured descriptive statements; no causal language or invented performance."""
    results = []
    for dim, title, implication in [
        ("loan_type", "Highest observed product default rate", "Review underwriting and monitoring for this product; observed association is not causation."),
        ("employment_type", "Highest observed employment-segment default rate", "Investigate affordability and repayment patterns before changing credit policy."),
        ("region", "Largest observed exposure by region", "Prioritize regional monitoring in proportion to outstanding exposure."),
        ("age_band", "Highest observed age-band default rate", "Use this descriptive cut for monitoring; do not infer an age-based lending rule."),
    ]:
        g = grouped(frame, [dim]); g = g[g.account_count >= 30]
        metric = "observed_exposure" if dim == "region" else "default_rate"
        row = g.sort_values([metric, dim], ascending=[False, True]).iloc[0]
        results.append({"title": title, "segment": {dim: str(row[dim])}, "metric": metric, "value": float(row[metric]), "defaults": int(row.default_count), "accounts": int(row.account_count), "so_what": implication, "source_fields": METRICS[metric]["fields"], "caveat": "Synthetic retrospective sample; at least 30 accounts per ranked cell."})
    g = grouped(frame, ["employment_type", "income_band"])
    row = g[g.account_count >= 30].sort_values(["default_rate", "employment_type", "income_band"], ascending=[False, True, True]).iloc[0]
    results.append({"title": "Highest observed employment-by-income default rate", "segment": {"employment_type": str(row.employment_type), "income_band": str(row.income_band)}, "metric": "default_rate", "value": float(row.default_rate), "defaults": int(row.default_count), "accounts": int(row.account_count), "so_what": "Investigate the joint affordability/employment segment; aggregate employment averages may hide this concentration.", "source_fields": METRICS["default_rate"]["fields"] + ["customers.employment_type", "customers.annual_income"], "caveat": "Descriptive cross-cut, not a causal effect; at least 30 accounts."})
    return results
