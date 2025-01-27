import pytest
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from backend.app import app
from supabase import create_client

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None  # No connection in test mode


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


from unittest.mock import patch, Mock

@patch("backend.connect_supabase.create_client")
def test_predict_route(mock_create_client):
    mock_client = Mock()
    mock_create_client.return_value = mock_client
    mock_client.table.return_value.insert.return_value.execute.return_value = {"data": "mock_response"}

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
