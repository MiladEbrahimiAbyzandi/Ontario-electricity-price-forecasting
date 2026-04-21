from fastapi.testclient import TestClient
import pytest
from api.main import app
import numpy as np
from src.config import SEQ_LENGTH, INPUT_SIZE


client = TestClient(app)


#---health input------------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

#---predict endpoint - happy path ----------------------

def test_predict_valid_request():
    payload = {
        "recent_data": np.random.randn(SEQ_LENGTH, INPUT_SIZE).tolist(),
        "horizon": 1
            }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    
    body = response.json()

    assert "horizon" in body
    assert "predicted_hoep" in body
    assert body["horizon"] == 1
    assert isinstance(body["predicted_hoep"], float)

def test_predict_all_horizons():
    
    data = np.random.randn(SEQ_LENGTH, INPUT_SIZE).tolist()

    for h in [1,2,3]:
        response = client.post("/predict", json={"recent_data": data, "horizon": h})
        
        assert response.status_code == 200
        assert response.json()["horizon"] == h


##---- predict endpoint - validation error -------------------------


def test_predict_wrong_number_of_rows():
    payload = {
        "recent_data": np.random.randn(10, INPUT_SIZE).tolist(),  # wrong rows
        "horizon": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_number_of_features():
    payload = {
        "recent_data": np.random.randn(SEQ_LENGTH, 5).tolist(),  # wrong features
        "horizon": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_horizon():
    payload = {
        "recent_data": np.random.randn(SEQ_LENGTH, INPUT_SIZE).tolist(),
        "horizon": 99,  # invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    # sending request with no horizon field
    payload = {
        "recent_data": np.random.randn(SEQ_LENGTH, INPUT_SIZE).tolist(),
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422




