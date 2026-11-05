import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src.common import ARTIFACTS
from src.modeling import FEATURES, account_drivers, evaluate, grouped_split, model_input
from src.runtime import load_artifacts


def test_customer_disjoint_and_deterministic(accounts):
    split = grouped_split(accounts)
    assert np.array_equal(split, grouped_split(accounts))
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(accounts.loc[split == a, "customer_id"]) & set(accounts.loc[split == b, "customer_id"])
    assert np.array_equal(split, accounts["split"].to_numpy())


def test_metric_known_example():
    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert metrics["auc"] == .75
    assert metrics["ks"] == .5
    assert metrics["gini"] == .5


def test_features_exclude_leakage_and_saved_predictions_match(accounts):
    assert not {"target_default", "account_id", "customer_id", "gender", "risk_score"} & set(FEATURES)
    assert not any(x.startswith("collection") for x in FEATURES)
    data = load_artifacts()
    pipeline = data["model"]["pipeline"]
    with threadpool_limits(limits=2):
        assert np.allclose(pipeline.predict_proba(model_input(accounts.iloc[:50]))[:, 1], accounts.risk_score.iloc[:50])
        drivers, caption = account_drivers(data["model"], accounts.iloc[:1])
    assert len(drivers) == 8 and drivers.contribution.notna().all()
    # Fitted numeric imputers must exactly reflect training-only medians.
    from src.modeling import NUMERIC
    train = accounts[accounts["split"] == "train"]
    stats = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"].statistics_
    assert np.allclose(stats, train[NUMERIC].median().to_numpy())
