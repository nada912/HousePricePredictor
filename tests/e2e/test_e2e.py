import requests
from unittest.mock import patch

BASE_URL = "http://localhost:8000"

def test_predict_endpoint_e2e():
    """Test the /predict endpoint with mocked model prediction."""
    with patch("backend.api.loaded_model") as mock_model:
        mock_model.predict.return_value = [150000]  # Mocked prediction

        response = requests.post(f"{BASE_URL}/predict", json={
            "area": 1200, "bedrooms": 2, "bathrooms": 1, "stories": 1,
            "mainroad": "yes", "guestroom": "no", "basement": "no",
            "hotwaterheating": "no", "airconditioning": "no",
            "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished"
        })

        assert response.status_code == 200
        assert response.json()["predicted_price"] == 150000

def test_fetch_predictions_e2e():
    """Test fetching predictions."""
    response = requests.get(f"{BASE_URL}/predictions")
    
    # Assert the response status code
    assert response.status_code == 200

    # Parse the JSON response
    response_data = response.get_json()

    # Assert the response data structure
    assert response_data is not None
    assert isinstance(response_data, list)

    # Check if each prediction in the response has the required keys
    if response_data:  # Ensure there are predictions to validate
        for prediction in response_data:
            assert "id" in prediction
            assert "input_data" in prediction
            assert "predicted_value" in prediction
    

def test_save_and_fetch_prediction_e2e():
    """Test the save and fetch flow with mocks."""
    mock_predictions = [
        {"input_data": {"area": 1500, "bedrooms": 3}, "predicted_value": 200000}
    ]

    with patch("backend.api.loaded_model") as mock_model, \
         patch("backend.api.supabase") as mock_supabase:
        
        # Mock the model prediction
        mock_model.predict.return_value = [200000]

        # Mock saving to Supabase
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = mock_predictions
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = mock_predictions

        # Step 1: Predict
        response_predict = requests.post(f"{BASE_URL}/predict", json={
            "area": 1500, "bedrooms": 3, "bathrooms": 2, "stories": 2,
            "mainroad": "yes", "guestroom": "yes", "basement": "no",
            "hotwaterheating": "no", "airconditioning": "yes",
            "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"
        })

        assert response_predict.status_code == 200
        assert response_predict.json()["predicted_price"] == 200000

        # Step 2: Fetch Predictions
        response_fetch = requests.get(f"{BASE_URL}/predictions")
        assert response_fetch.status_code == 200
        assert response_fetch.json() == mock_predictions