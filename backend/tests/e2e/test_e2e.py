import requests

BASE_URL = "http://localhost:8000"

def test_predict_endpoint_e2e():
    """Test the full prediction flow."""
    response = requests.post(f"{BASE_URL}/predict", json={
        "area": 1200, "bedrooms": 2, "bathrooms": 1, "stories": 1,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "no",
        "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished"
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_fetch_predictions_e2e():
    """Test fetching predictions."""
    response = requests.get(f"{BASE_URL}/predictions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_save_and_fetch_prediction_e2e():
    """Test save and fetch prediction flow."""
    response = requests.post(f"{BASE_URL}/predict", json={
        "area": 1500, "bedrooms": 3, "bathrooms": 2, "stories": 2,
        "mainroad": "yes", "guestroom": "yes", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "yes",
        "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"
    })
    prediction = response.json()
    assert "predicted_price" in prediction

    fetch_response = requests.get(f"{BASE_URL}/predictions")
    assert fetch_response.status_code == 200
    assert any(pred["predicted_value"] == prediction["predicted_price"] for pred in fetch_response.json())
