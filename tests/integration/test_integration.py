import pytest
import supabase
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.api import save_to_supabase
from backend.api import app
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def test_save_to_supabase():
    """Test saving a prediction to Supabase with input validation."""
    from supabase import create_client

    # Initialize Supabase client (mocked in tests)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Input data and predicted value
    input_data = {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "YES",
        "furnishingstatus": "furnished"
    }
    predicted_value = 13300000

    # Validate input structure
    required_keys = [
        "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
        "basement", "hotwaterheating", "airconditioning", "parking",
        "prefarea", "furnishingstatus"
    ]

    for key in required_keys:
        if key not in input_data:
            pytest.fail(f"Input data is missing required field: {key}")
        if not isinstance(input_data[key], (int, str)):
            pytest.fail(f"Invalid type for field {key}: {type(input_data[key])}")

    # Save the prediction to Supabase
    save_to_supabase(input_data, predicted_value)



def test_fetch_predictions(client):
    """Test fetching predictions from the /predictions endpoint."""
    # Make a GET request to the /predictions endpoint
    response = client.get("/predictions")
    
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



from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    """Fixture for Flask test client."""
    from backend.api import app
    app.testing = True
    return app.test_client()

@patch("backend.api.supabase")  # Ensure this matches the actual import path in your code
@patch("backend.api.preprocess_input")
@patch("backend.api.loaded_model")
def test_predict_and_save(mock_loaded_model, mock_preprocess_input, mock_supabase_client, client):
    """Test prediction and saving with Supabase."""
    # Mock the model's predict method
    mock_loaded_model.predict.return_value = [200000]
    mock_preprocess_input.return_value = [[1500, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]]  # Example processed data

    # Mock the Supabase insert
    mock_supabase_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data={"id": 51})

    # Define the input data
    input_data = {
        "area": 1500,
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

    # Make a POST request
    response = client.post("/predict", json=input_data)

    # Assertions
    assert response.status_code == 200
    response_data = response.get_json()
    assert "predicted_price" in response_data
    assert response_data["predicted_price"] == 200000

    # Ensure Supabase insert was called
    mock_supabase_client.table.assert_called_once_with("predictions")
    mock_supabase_client.table.return_value.insert.assert_called_once()
