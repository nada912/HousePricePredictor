import requests
import pytest

# Base URL for the Flask application
BASE_URL = "http://localhost:8000"  # Update if using a different host or port

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Fixture to ensure the server is running before tests."""
    print(f"Ensure the server is running on {BASE_URL}")
    yield  # No teardown needed as the server remains running


def test_predict_endpoint_e2e():
    """Test the /predict endpoint."""
    # Input data for the prediction
    input_data = {
        "area": 1200,
        "bedrooms": 2,
        "bathrooms": 1,
        "stories": 1,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "parking": 1,
        "prefarea": "no",
        "furnishingstatus": "semi-furnished"
    }

    # Make POST request to /predict
    response = requests.post(f"{BASE_URL}/predict", json=input_data)

    # Validate response
    assert response.status_code == 200, f"Status code was {response.status_code}, expected 200."
    response_data = response.json()

    # Check if the response contains the 'predicted_price' key
    assert "predicted_price" in response_data, "Response is missing 'predicted_price'."
    assert isinstance(response_data["predicted_price"], int), "Predicted price should be an integer."

    print(f"Predicted price: {response_data['predicted_price']}")


def test_fetch_predictions_e2e():
    """Test the /predictions endpoint."""
    # Make GET request to /predictions
    response = requests.get(f"{BASE_URL}/predictions")

    # Validate response
    assert response.status_code == 200, f"Status code was {response.status_code}, expected 200."
    response_data = response.json()

    # Ensure the response is a list
    assert isinstance(response_data, list), "Predictions response should be a list."

    # Validate the structure of at least one prediction if available
    if response_data:
        prediction = response_data[0]
        assert "input_data" in prediction, "Prediction missing 'input_data' field."
        assert "predicted_value" in prediction, "Prediction missing 'predicted_value' field."

    print(f"Fetched predictions: {response_data}")


def test_save_and_fetch_prediction_e2e():
    """Test saving a prediction and retrieving it."""
    # Input data for the prediction
    input_data = {
        "area": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "yes",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }

    # Step 1: Make a prediction
    response_predict = requests.post(f"{BASE_URL}/predict", json=input_data)
    assert response_predict.status_code == 200, "Prediction failed."
    response_predict_data = response_predict.json()
    assert "predicted_price" in response_predict_data, "Prediction response missing 'predicted_price'."
    predicted_price = response_predict_data["predicted_price"]

    # Step 2: Fetch saved predictions
    response_fetch = requests.get(f"{BASE_URL}/predictions")
    assert response_fetch.status_code == 200, "Fetching predictions failed."
    response_fetch_data = response_fetch.json()

    # Ensure the prediction is saved
    found = any(
        pred.get("predicted_value") == predicted_price and pred.get("input_data") == input_data
        for pred in response_fetch_data
    )
    assert found, "Saved prediction not found in fetched predictions."

    print("Saved and fetched prediction successfully.")
