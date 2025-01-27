import pytest
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
from supabase import create_client
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import *
from backend.api import app, preprocess_input

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None  # No connection in test mode

# Global variable to store the preloaded model
loaded_model = None

def test_load_model_from_dagshub():
    """
    Test loading the model from MLflow instance hosted on DagsHub.
    """
    
    global loaded_model

    # Set up MLflow tracking URI and credentials
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    DAGSHUB_REPO_NAME
    DAGSHUB_USERNAME

    client = MlflowClient()

    # fetch the model in production stage
    registered_models = client.search_registered_models()
    
    for model in registered_models:
        # Check all versions of the model for the Production stage
        production_versions = client.get_latest_versions(model.name, stages=["Production"])
        print(f"Model: {model.name}, Versions: {production_versions}")
        
        if production_versions:
            # Get the latest version saved as Production
            latest_version = max(production_versions, key=lambda v: v.version)
            print(f"Latest Production Version: {latest_version.version}")
            model_name = model.name
            print(f"Model Name: {model_name}")
            model_uri = latest_version.source  # Use the source URI directly
            print(f"Model URI: {model_uri}")

    # Load the model and check if it's functional
    model_uri = f"models:/{model_name}/Production"
    print(f"Model URI to load: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    assert loaded_model is not None


from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    app.testing = True
    return app.test_client()


@patch("backend.api.loaded_model")
@patch("backend.api.preprocess_input")
@patch("backend.api.save_to_supabase")
def test_predict_route(mock_save_to_supabase, mock_preprocess_input, mock_loaded_model, client):
    """Test the predict route with valid input."""
    # Mock the model and preprocess_input function
    mock_loaded_model.predict.return_value = [150000]
    mock_preprocess_input.return_value = [[1200, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]]  # Example transformed input

    # Define the input data
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

    # Send POST request to /predict
    response = client.post("/predict", json=input_data)

    # Assert response status code
    assert response.status_code == 200

    # Assert response JSON structure and content
    response_data = response.get_json()
    assert "predicted_price" in response_data
    assert response_data["predicted_price"] == 150000

    # Verify the preprocess_input function was called with the right data
    mock_preprocess_input.assert_called_once_with(input_data)

    # Verify the model's predict function was called with the preprocessed data
    mock_loaded_model.predict.assert_called_once_with(mock_preprocess_input.return_value)

    # Verify save_to_supabase was called with the input data and prediction
    mock_save_to_supabase.assert_called_once_with(input_data, 150000)

    
def test_predict_route_invalid_input():
    """Test the predict route with invalid input."""
    client = app.test_client()
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


"""
# test needs more work 
def test_predict_route(client, load_model):
    #Test the predict route with valid input using the real loaded model.
    
    global loaded_model

    # Ensure the model is loaded before testing
    if loaded_model is None:
        raise ValueError("Model is not loaded. Please check if model loading is successful.")

    # Define the input data
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
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }

    input_features = preprocess_input(input_data)  # Preprocess the input using the same method as in the backend
    print(f"Processed Features: {input_features.head(10)}")

    # Convert the DataFrame to a dictionary
    input_features_dict = input_features.to_dict(orient='records')[0]

    # Send POST request to /predict
    response = client.post("/predict", json=input_features_dict)
    print(response.get_data(as_text=True))

    # Assert response status code
    assert response.status_code == 200

    # Assert response JSON structure and content
    response_data = response.get_json()
    assert "predicted_price" in response_data
"""


"""
# test works on local only
def test_load_model_from_mlflow():

    #Test loading the model from MLflow.

    # set the MLflow tracking URI from environment variables
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "PolynomialRegressionModel"
    client = MlflowClient()

    # Get the latest version of the model in the Production stage
    latest_version = client.get_latest_versions(model_name, stages=["Production"])[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        pytest.fail(f"The model could not be loaded from MLflow. Error : {e}")
    
    assert model is not None, "The model should not be None"
"""