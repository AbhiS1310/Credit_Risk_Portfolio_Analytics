"""Unsupervised flags with explicit, leave-one-out peer comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import SEED, digest

ANOMALY_FEATURES = ["repay_payment_ratio", "dpd_deterioration", "recent_missed", "bureau_utilization", "repay_dpd_max", "payment_ratio_change"]
CONFIG = {"version": "peer-forest-v1", "trees": 200, "contamination": 0.05, "seed": SEED, "peer_minimum": 30, "robust_deviations": 3}


def robust_reference(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    median = float(np.median(values))
    scale = 1.4826 * np.median(np.abs(values - median))
    if scale == 0:
        scale = (np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
    return median, float(scale) if scale > 0 else None


def detect(frame, input_hash):
    x = frame[ANOMALY_FEATURES].astype(float)
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("forest", IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED, n_jobs=1))])
    pipeline.fit(x)
    raw_score = -pipeline.score_samples(x)
    percentile = pd.Series(raw_score).rank(pct=True).to_numpy()
    forest_flag = pipeline.predict(x) == -1
    version = digest({"input": input_hash, "configuration": CONFIG})[:20]
    groups = {k: np.array(v) for k, v in frame.reset_index(drop=True).groupby("loan_type").groups.items()}
    values = x.to_numpy()
    all_indices = np.arange(len(frame))
    records = []
    for i, row in enumerate(frame.itertuples(index=False)):
        peers = groups[row.loan_type]
        fallback = len(peers) - 1 < 30
        if fallback:
            peers = all_indices
        peers = peers[peers != i]
        reasons, deviations = [], []
        for j, feature in enumerate(ANOMALY_FEATURES):
            value = values[i, j]
            median, scale = robust_reference(values[peers, j])
            if not np.isfinite(value):
                continue
            deviation = (value - median) / scale if median is not None and scale else None
            if deviation is not None:
                deviations.append({"code": "PEER_DEVIATION", "feature": feature, "observed": float(value), "peer_median": median, "robust_deviations": float(deviation), "definition": "Feature comparison explaining unusualness, not a causal forest attribution."})
            code = None
            if feature == "repay_payment_ratio" and deviation is not None and deviation < -3:
                code = "LOW_PAYMENT_RATIO"
            if feature == "dpd_deterioration" and value >= 15 and deviation is not None and deviation > 3:
                code = "DPD_DETERIORATION"
            if feature == "recent_missed" and value >= 2 and np.isfinite(row.previous_missed) and value > row.previous_missed:
                code = "MISSED_PAYMENT_INCREASE"
            if feature == "bureau_utilization" and (value > 1 or (deviation is not None and deviation > 3)):
                code = "UNUSUAL_BUREAU_UTILIZATION"
            if code:
                reasons.append({"code": code, "feature": feature, "observed": float(value), "peer_median": median, "robust_deviations": float(deviation) if deviation is not None else None, "previous_missed": float(row.previous_missed) if feature == "recent_missed" else None, "definition": "Rule threshold met; compare against account history and leave-one-out peers."})
        rule_flag = bool(reasons)
        if forest_flag[i] and not reasons:
            reasons = sorted(deviations, key=lambda d: abs(d["robust_deviations"]), reverse=True)[:3]
            if not reasons:
                reasons = [{"code": "MULTIVARIATE_PATTERN", "definition": "Isolation Forest flagged a multivariate pattern; no reliable individual robust deviation.", "observations": {c: float(values[i, j]) if np.isfinite(values[i, j]) else None for j, c in enumerate(ANOMALY_FEATURES)}}]
        if rule_flag or forest_flag[i]:
            records.append({"account_id": row.account_id, "anomaly_version": version, "anomaly_percentile": float(percentile[i]), "forest_score": float(raw_score[i]), "forest_flag": bool(forest_flag[i]), "rule_flag": rule_flag, "peer_segment": "Portfolio fallback" if fallback else row.loan_type, "peer_count": len(peers), "feature_coverage": int(np.isfinite(values[i]).sum()), "feature_total": len(ANOMALY_FEATURES), "reasons": reasons})
    return records, pipeline, {"version": version, "config": CONFIG, "flagged_accounts": len(records), "forest_flagged": int(forest_flag.sum()), "accounts": len(frame)}
