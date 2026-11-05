import numpy as np
import pandas as pd
import pytest

from src.common import RAW, ROOT, sha256
from src.features import build_features, bureau_features, repayment_features
from src.ingestion import load_sources


def test_originals_unchanged():
    source = ROOT / "AI Case Study 1" / "Case 2" / "files (10)"
    for path in RAW.glob("*.csv"):
        assert sha256(path) == sha256(source / path.name)


def test_real_data_reconciliation(accounts):
    frames, report = load_sources()
    assert [len(frames[n]) for n in frames] == [10000, 14685, 22979, 176220, 8635]
    assert len(accounts) == accounts.account_id.nunique() == 14685
    assert accounts.target_default.sum() == 2364
    assert report["normalizations"]["gender_aliases"] == 600
    r = frames["repayment_history"]
    assert r.groupby("account_id").payment_month.nunique().eq(12).all()
    assert accounts.loan_amount.sum() == pytest.approx(frames["loan_accounts"].loan_amount.sum())
    latest = r.sort_values("payment_month").groupby("account_id").tail(1)
    assert accounts.latest_outstanding.sum() == pytest.approx(latest.outstanding_balance.sum())
    assert accounts.collection_recovered.sum() == pytest.approx(frames["collections_events"].amount_recovered.sum())


def test_children_do_not_multiply_accounts():
    frames, report = load_sources()
    account = frames["loan_accounts"].iloc[0]
    frames["loan_accounts"] = frames["loan_accounts"].iloc[:1]
    frames["customers"] = frames["customers"][frames["customers"].customer_id == account.customer_id]
    frames["repayment_history"] = frames["repayment_history"][frames["repayment_history"].account_id == account.account_id]
    frames["collections_events"] = frames["collections_events"][frames["collections_events"].account_id == account.account_id]
    # Two bureau lines and two collection actions make the raw fan-out unambiguous.
    b = frames["bureau_credit"].iloc[:2].copy(); b["customer_id"] = account.customer_id
    e = frames["collections_events"].iloc[:1]
    if e.empty:
        e = load_sources()[0]["collections_events"].iloc[:1].copy()
    e = pd.concat([e, e], ignore_index=True); e["account_id"] = account.account_id
    frames["bureau_credit"], frames["collections_events"] = b, e
    out = build_features(frames, report)
    assert len(out) == 1
    assert out.iloc[0].bureau_lines == 2
    assert out.iloc[0].collection_events == 2
    assert out.iloc[0].repay_due == pytest.approx(frames["repayment_history"].emi_due.sum())


def test_missing_bureau_dpd_and_zero_limit():
    b = load_sources()[0]["bureau_credit"].iloc[:2].copy()
    b["customer_id"] = "same"; b["days_past_due"] = np.nan; b["sanctioned_amount"] = 0
    row = bureau_features(b).iloc[0]
    assert pd.isna(row.bureau_dpd_max) and pd.isna(row.bureau_utilization)
    assert row.bureau_dpd_missing == 2 and row.bureau_dpd_coverage == 0
    assert pd.isna(row.bureau_dpd_gt_30)


def test_latest_balance_is_selected_by_month_and_short_windows_unknown():
    r = pd.DataFrame({"account_id": ["a"]*3, "payment_month": [1, 2, 3], "emi_due": [10]*3, "emi_paid": [10]*3, "days_past_due": [0]*3, "payment_status": ["On-Time"]*3, "outstanding_balance": [100, 80, 90]})
    row = repayment_features(r).iloc[0]
    assert row.latest_outstanding == 90
    assert pd.isna(row.dpd_deterioration)


@pytest.mark.parametrize("problem", ["duplicate", "orphan", "target", "date"])
def test_critical_data_failures(tmp_path, problem):
    import shutil
    for path in RAW.glob("*.csv"):
        shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / "loan_accounts.csv"
    loans = pd.read_csv(path)
    if problem == "duplicate":
        loans = pd.concat([loans, loans.iloc[:1]], ignore_index=True)
    elif problem == "orphan":
        loans.loc[0, "customer_id"] = "missing"
    elif problem == "target":
        loans.loc[0, "target_default"] = 2
    else:
        loans.loc[0, "origination_date"] = "invalid"
    loans.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_sources(tmp_path)
