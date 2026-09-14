import pytest
from fastapi.testclient import TestClient

from api.main import app
from database import get_recent_predictions
from src.model_service import get_prediction_service
from src.risk_engine import classify_fraud_risk


@pytest.fixture
def valid_transaction():
    return {
        "Time": 12345.0,
        "V1": -1.2,
        "V2": 0.45,
        "V3": 1.2,
        "V4": -0.3,
        "V5": 0.1,
        "V6": -0.2,
        "V7": 0.25,
        "V8": -0.1,
        "V9": 0.05,
        "V10": 0.2,
        "V11": -0.2,
        "V12": 1.0,
        "V13": -0.6,
        "V14": 0.4,
        "V15": -0.3,
        "V16": 0.2,
        "V17": -0.1,
        "V18": 0.05,
        "V19": 0.02,
        "V20": -0.01,
        "V21": 0.03,
        "V22": -0.02,
        "V23": 0.1,
        "V24": 0.4,
        "V25": -0.1,
        "V26": 0.02,
        "V27": -0.04,
        "V28": 0.01,
        "Amount": 250.5,
    }


def test_model_loads_successfully():
    service = get_prediction_service()
    assert service.model is not None


def test_expected_feature_list_loads():
    service = get_prediction_service()
    assert isinstance(service.feature_names, list)
    assert len(service.feature_names) > 0
    assert "Time" in service.feature_names
    assert "Amount" in service.feature_names


def test_prediction_returns_valid_output(valid_transaction):
    result = get_prediction_service().predict_transaction(valid_transaction)
    assert set(result.keys()) == {"prediction", "fraud_probability", "risk_level", "recommended_action"}
    assert result["prediction"] in {"NORMAL", "FRAUD"}
    assert isinstance(result["fraud_probability"], float)
    assert 0 <= result["fraud_probability"] <= 1
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["recommended_action"] in {"APPROVE", "REVIEW", "BLOCK"}


def test_risk_classification_works():
    low = classify_fraud_risk(0.12)
    medium = classify_fraud_risk(0.52)
    high = classify_fraud_risk(0.91)

    assert low.risk_level == "LOW"
    assert low.recommended_action == "APPROVE"
    assert medium.risk_level == "MEDIUM"
    assert medium.recommended_action == "REVIEW"
    assert high.risk_level == "HIGH"
    assert high.recommended_action == "BLOCK"


def test_api_health_works():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_api_predict_works_with_valid_input(valid_transaction):
    client = TestClient(app)
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"NORMAL", "FRAUD"}
    assert 0 <= float(body["fraud_probability"]) <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["recommended_action"] in {"APPROVE", "REVIEW", "BLOCK"}


def test_api_rejects_incomplete_input():
    client = TestClient(app)
    invalid_payload = {"Time": 1000.0, "V1": 1.0, "V2": 2.0}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_database_records_are_created(valid_transaction):
    client = TestClient(app)
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 200
    history = get_recent_predictions(limit=1)
    assert len(history) >= 1
    assert history[0]["prediction"] in {"NORMAL", "FRAUD"}
    assert history[0]["recommended_action"] in {"APPROVE", "REVIEW", "BLOCK"}


def test_normal_and_fraud_responses_have_expected_structure():
    service = get_prediction_service()
    normal_response = service.predict_transaction({
        "Time": 1000.0,
        "V1": -0.2,
        "V2": 0.1,
        "V3": 0.4,
        "V4": 0.2,
        "V5": 0.1,
        "V6": -0.3,
        "V7": 0.1,
        "V8": -0.2,
        "V9": 0.05,
        "V10": 0.0,
        "V11": -0.2,
        "V12": 0.4,
        "V13": 0.1,
        "V14": -0.3,
        "V15": 0.2,
        "V16": 0.1,
        "V17": -0.1,
        "V18": 0.05,
        "V19": -0.02,
        "V20": 0.01,
        "V21": -0.03,
        "V22": 0.02,
        "V23": -0.1,
        "V24": 0.0,
        "V25": 0.1,
        "V26": -0.02,
        "V27": 0.04,
        "V28": -0.05,
        "Amount": 15.0,
    })
    fraud_response = service.predict_transaction({
        "Time": 1000.0,
        "V1": -5.0,
        "V2": 5.0,
        "V3": 4.0,
        "V4": -3.0,
        "V5": 2.0,
        "V6": -1.5,
        "V7": 1.2,
        "V8": -0.8,
        "V9": 0.3,
        "V10": 0.2,
        "V11": -0.4,
        "V12": 1.1,
        "V13": -0.7,
        "V14": 0.9,
        "V15": -0.5,
        "V16": 0.4,
        "V17": -0.3,
        "V18": 0.2,
        "V19": -0.1,
        "V20": 0.05,
        "V21": -0.04,
        "V22": 0.02,
        "V23": 0.1,
        "V24": -0.12,
        "V25": 0.14,
        "V26": -0.08,
        "V27": 0.05,
        "V28": -0.04,
        "Amount": 850.0,
    })

    for response in (normal_response, fraud_response):
        assert set(response.keys()) == {"prediction", "fraud_probability", "risk_level", "recommended_action"}
        assert response["prediction"] in {"NORMAL", "FRAUD"}
        assert 0 <= response["fraud_probability"] <= 1
        assert response["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert response["recommended_action"] in {"APPROVE", "REVIEW", "BLOCK"}
