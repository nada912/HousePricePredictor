import pytest
import supabase
# exit the current directory and point to the backend directory
import sys
sys.path.append('backend')
from app import save_to_supabase


def test_save_to_supabase():
    """Test saving a prediction to Supabase."""
    test_data = {"area": 1200, "bedrooms": 2}
    predicted_value = 50000
    try:
        save_to_supabase(test_data, predicted_value)
    except Exception as e:
        pytest.fail(f"Saving to Supabase failed: {str(e)}")

def test_fetch_predictions():
    """Test fetching predictions from Supabase."""
    response = supabase.table("predictions").select("*").execute()
    assert response.data is not None
    assert isinstance(response.data, list)

def test_predict_and_save():
    """Test prediction and saving to Supabase together."""
    from app import app
    client = app.test_client()
    response = client.post('/predict', json={
        "area": 1200, "bedrooms": 2, "bathrooms": 1, "stories": 1,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "no",
        "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished"
    })
    assert response.status_code == 200
    prediction = response.get_json()
    assert "predicted_price" in prediction
    save_to_supabase({"input": "test"}, prediction["predicted_price"])
