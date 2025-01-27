import pytest
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
from backend.app import app

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

def test_load_model_from_mlflow():
    """
    Test loading the model from MLflow.
    """
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


def test_predict_route():
    """Test the predict route with valid input."""
    client = app.test_client()
    response = client.post('/predict', json={
        "area": 1200, "bedrooms": 2, "bathrooms": 1, "stories": 1,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "no",
        "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished"
    })
    assert response.status_code == 200
    assert "predicted_price" in response.get_json()


def test_predict_route_invalid_input():
    """Test the predict route with invalid input."""
    client = app.test_client()
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()
