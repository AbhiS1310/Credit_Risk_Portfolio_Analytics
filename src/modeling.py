"""Customer-disjoint retrospective classification with train-only preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.common import SEED

CATEGORICAL = "employment_type education_level housing_type region loan_type loan_purpose disbursal_channel".split()
BASE_NUMERIC = "age annual_income num_dependents years_employed years_at_residence loan_amount interest_rate_pct tenure_months collateral_flag".split()
BEHAVIOR_NUMERIC = "bureau_lines bureau_active bureau_closed bureau_written_off bureau_lenders bureau_credit_types bureau_sanctioned bureau_outstanding bureau_dpd_max bureau_dpd_mean bureau_dpd_observed bureau_dpd_missing bureau_dpd_coverage bureau_dpd_gt_0 bureau_dpd_gt_30 bureau_dpd_gt_90 bureau_utilization repay_months repay_due repay_paid repay_zero_payments repay_late repay_missed repay_dpd_mean repay_dpd_max repay_dpd_gt_0 repay_dpd_gt_30 repay_dpd_gt_90 latest_outstanding repay_latest_dpd recent_dpd previous_dpd recent_payment_ratio previous_payment_ratio recent_missed previous_missed dpd_deterioration payment_ratio_change repay_payment_ratio has_bureau_history has_repayment_history".split()
NUMERIC = BASE_NUMERIC + BEHAVIOR_NUMERIC
FEATURES = NUMERIC + CATEGORICAL


def model_input(frame):
    x = frame[FEATURES].copy()
    x[NUMERIC] = x[NUMERIC].astype(float)
    for c in CATEGORICAL:
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return x


def grouped_split(frame):
    split = np.full(len(frame), "train", dtype=object)
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(frame, groups=frame.customer_id))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(frame.iloc[trainval], groups=frame.iloc[trainval].customer_id))
    split[test] = "test"; split[trainval[val_local]] = "validation"
    for name in ["train", "validation", "test"]:
        if frame.loc[split == name, "target_default"].nunique() != 2:
            raise ValueError(f"Both target classes required in {name}")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(frame.loc[split == a, "customer_id"]) & set(frame.loc[split == b, "customer_id"]):
            raise ValueError("Customer leakage across splits")
    return split


def make_pipeline(kind, **params):
    numeric = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if kind == "logistic":
        numeric.append(("scale", StandardScaler()))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        estimator = LogisticRegression(max_iter=2000, random_state=SEED)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        estimator = HistGradientBoostingClassifier(random_state=SEED, max_iter=200, min_samples_leaf=30, early_stopping=False, **params)
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", encoder)])
    preprocess = ColumnTransformer([("num", Pipeline(numeric), NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocess), ("estimator", estimator)])


def evaluate(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    auc = float(roc_auc_score(y, scores))
    return {"auc": auc, "ks": float(np.max(tpr - fpr)), "gini": 2 * auc - 1, "accounts": len(y), "defaults": int(y.sum()), "default_rate": float(y.mean())}


def train_models(frame):
    split = grouped_split(frame)
    x, y = model_input(frame), frame.target_default.astype(int)
    train, val, test = (split == n for n in ["train", "validation", "test"])
    logistic = make_pipeline("logistic").fit(x[train], y[train])
    candidates = []
    for lr in [0.05, 0.1]:
        for leaves in [15, 31]:
            pipe = make_pipeline("histogram", learning_rate=lr, max_leaf_nodes=leaves).fit(x[train], y[train])
            candidates.append((roc_auc_score(y[val], pipe.predict_proba(x[val])[:, 1]), {"learning_rate": lr, "max_leaf_nodes": leaves}, pipe))
    best_auc, best_params, histogram = max(candidates, key=lambda c: c[0])
    logistic_auc = roc_auc_score(y[val], logistic.predict_proba(x[val])[:, 1])
    selected = "logistic" if logistic_auc >= best_auc - 0.005 else "histogram"
    models = {"logistic": logistic, "histogram": histogram}
    results = {name: {"validation": evaluate(y[val], pipe.predict_proba(x[val])[:, 1]), "test": evaluate(y[test], pipe.predict_proba(x[test])[:, 1])} for name, pipe in models.items()}
    results.update({"selected": selected, "histogram_parameters": best_params, "search": [{"validation_auc": float(a), **p} for a, p, _ in candidates], "seed": SEED, "limitation": "Retrospective classification only; no outcome horizon or timestamped bureau snapshot. Temporal leakage remains possible.", "split_counts": {n: {"accounts": int((split == n).sum()), "customers": int(frame.loc[split == n, "customer_id"].nunique())} for n in ["train", "validation", "test"]}})
    importance = permutation_importance(models[selected], x[val], y[val], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    importance_frame = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean, "std": importance.importances_std}).sort_values("importance", ascending=False)
    references = {c: float(x.loc[train, c].median()) if x.loc[train, c].notna().any() else 0.0 for c in NUMERIC}
    references.update({c: x.loc[train, c].mode().iloc[0] if x.loc[train, c].notna().any() else "Unknown" for c in CATEGORICAL})
    bundle = {"pipeline": models[selected], "name": selected, "features": FEATURES, "references": references}
    scores = pd.DataFrame({"account_id": frame.account_id, "split": split, "risk_score": models[selected].predict_proba(x)[:, 1]})
    return bundle, results, importance_frame, scores


def account_drivers(bundle, row):
    x = model_input(row)
    pipeline = bundle["pipeline"]
    if bundle["name"] == "logistic":
        prep = pipeline.named_steps["preprocess"]
        contribution = prep.transform(x)[0] * pipeline.named_steps["estimator"].coef_[0]
        return pd.DataFrame({"feature": prep.get_feature_names_out(), "contribution": contribution}).assign(magnitude=lambda f: f.contribution.abs()).sort_values("magnitude", ascending=False).head(8), "Signed logistic log-odds contributions (not causal effects)"
    score = pipeline.predict_proba(x)[0, 1]
    rows = []
    for feature in FEATURES:
        changed = x.copy(); changed[feature] = bundle["references"][feature]
        rows.append({"feature": feature, "contribution": float(score - pipeline.predict_proba(changed)[0, 1])})
    return pd.DataFrame(rows).assign(magnitude=lambda f: f.contribution.abs()).sort_values("magnitude", ascending=False).head(8), "Score sensitivity to replacing one feature with its training reference (not SHAP or causality)"
