"""Contrats HTTP du service."""

import pytest
from fastapi.testclient import TestClient

from afrorisk.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

VALID = {
    "amt_income_total": 150000,
    "amt_credit": 400000,
    "amt_annuity": 24000,
    "amt_goods_price": 360000,
    "age_years": 34,
    "employment_years": 5,
    "ext_source_1": 0.4,
    "ext_source_2": 0.55,
    "ext_source_3": 0.33,
    "name_contract_type": "Cash loans",
    "name_education_type": "Higher education",
    "name_family_status": "Married",
    "name_income_type": "Working",
    "cnt_children": 1,
    "flag_own_realty": "Y",
    "flag_own_car": "N",
    "region_rating_client": 2,
}


def test_health_and_docs(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["model_loaded"] is True
    assert body["data_source"] in {"synthetic_fixture", "home_credit"}
    docs = client.get("/docs")
    assert docs.status_code == 200


def test_predict_valid_and_invalid(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["predicted_class"] in {0, 1}
    assert body["risk_level"] in {"faible", "modéré", "élevé"}
    assert body["predicted_label"] in {"défaut", "non-défaut"}
    assert 1 <= len(body["factors"]) <= 6
    assert {"feature", "label", "contribution", "direction"} <= set(body["factors"][0])
    assert "placeholder" in body["model"]

    missing_gender = dict(VALID)
    missing_gender["code_gender"] = None
    assert client.post("/predict", json=missing_gender).status_code == 200

    rejected = client.post("/predict", json={**VALID, "age_years": 12})
    assert rejected.status_code == 422
    rejected = client.post("/predict", json={**VALID, "amt_income_total": -5})
    assert rejected.status_code == 422


def test_metrics_summary_contract(client):
    response = client.get("/metrics-summary")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["placeholder"], bool)
    assert body["data_source"] in {"synthetic_fixture", "home_credit"}
    served = body["served"]
    assert 0.0 <= served["metrics"]["roc_auc"] <= 1.0
    assert len(served["comparison"]) == 5
    assert len(served["global_factors"]) >= 1
    names = {item["model"] for item in served["comparison"]}
    assert names == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
    }
    assert len(body["research"]["comparison"]) == 5
