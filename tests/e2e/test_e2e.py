import requests

BASE_URL = "http://localhost:8000"  # Replace with the actual base URL of your running backend

def test_predict_endpoint_e2e():
    """Test the /predict endpoint with correctly formatted input."""
    # Raw input data matching what preprocess_input expects
    input_data = {
        "area": 1200,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "prefarea": "no",
        "furnishingstatus": "semi-furnished",
        "bathrooms": 2,
        "stories": 1,
        "parking": 1,
        "bedrooms": 2
    }

    # Make POST request to /predict
    response = requests.post(f"{BASE_URL}/predict", json=input_data)

    # Validate response
    assert response.status_code == 200, f"Status code was {response.status_code}, expected 200."
    response_json = response.json()
    assert "predicted_price" in response_json, "Response JSON does not contain 'predicted_price'."
    assert isinstance(response_json["predicted_price"], int), "Predicted price is not an integer."


def test_fetch_predictions_e2e():
    """Test the /predictions endpoint."""
    # Make GET request to /predictions
    response = requests.get(f"{BASE_URL}/predictions")

    # Validate response
    assert response.status_code == 200, f"Status code was {response.status_code}, expected 200."
    response_json = response.json()
    assert isinstance(response_json, list), "Predictions response is not a list."


def test_save_and_fetch_prediction_e2e():
    """Test saving a prediction and retrieving it."""
    # Raw input data matching what preprocess_input expects
    input_data = {
        "area": 1500,
        "mainroad": "yes",
        "guestroom": "yes",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "furnished",
        "bathrooms": 3,
        "stories": 2,
        "parking": 2,
        "bedrooms": 3
    }

    # Step 1: Make a prediction
    response_predict = requests.post(f"{BASE_URL}/predict", json=input_data)
    assert response_predict.status_code == 200, f"Prediction failed with status code {response_predict.status_code}."
    prediction_json = response_predict.json()
    assert "predicted_price" in prediction_json, "Response JSON does not contain 'predicted_price'."
    predicted_price = prediction_json["predicted_price"]
    assert isinstance(predicted_price, int), "Predicted price is not an integer."

    # Step 2: Fetch saved predictions
    response_predictions = requests.get(f"{BASE_URL}/predictions")
    assert response_predictions.status_code == 200, f"Fetching predictions failed with status code {response_predictions.status_code}."
    predictions = response_predictions.json()
    assert any(pred.get("predicted_value") == predicted_price for pred in predictions), "Saved prediction not found in fetched predictions."
