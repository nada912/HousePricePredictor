from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from flask_cors import CORS
from supabase import create_client

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MLflow Tracking URI and model name from .env
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
MODEL_NAME = "LinearRegressionModel"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set supabase URL and key
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_to_supabase(input_data, predicted_value):
    """Save prediction to Supabase."""
    try:
        response = supabase.table("predictions").insert({
            "input_data": input_data,
            "predicted_value": predicted_value
        }).execute()
        print("Prediction saved to Supabase:", response.data)
    except Exception as e:
        print(f"Error saving to Supabase: {e}")


# Global variable to store the preloaded model
loaded_model = None

# Define feature names in the expected order
FEATURE_NAMES = [
    'area', 'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'prefarea', 'furnishingstatus_semi_furnished',
    'furnishingstatus_unfurnished', 'bathrooms_2', 'bathrooms_3',
    'bathrooms_4', 'stories_2', 'stories_3', 'stories_4', 'parking_1',
    'parking_2', 'parking_3', 'bedrooms_2', 'bedrooms_3', 'bedrooms_4',
    'bedrooms_5', 'bedrooms_6'
]


def preload_model():
    """Load the model from the local directory."""
    global loaded_model
    try:
        # Locate the model in the models folder
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "models")
        model_name = os.listdir(local_path)[1]
        print(f"Model Name: {model_name}")

        # Load the model using MLflow
        loaded_model = mlflow.sklearn.load_model(local_path + "/" + model_name)
        print("Model loaded successfully")

    except Exception as e:
        print(f"Error loading the model: {e}")


def preprocess_input(data):
    """
    Preprocess the input data from the front end to match the 23 features expected by the model.
    Ensures that all values (except 'area') are either True (1) or False (0).
    """
    # Initialize all features with default values
    features = {}
    
    for feature in FEATURE_NAMES:
        if feature == 'area':
            # 'area' is treated as a numeric value
            features[feature] = int(data.get(feature, 0))
        else:
            # For other features, treat them as True/False
            feature_value = data.get(feature, '').lower()
            if feature_value in ['yes', '1', 'true']:
                features[feature] = 1
            else:
                features[feature] = 0

    # Convert the feature dictionary to a pandas DataFrame
    input_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    return input_df


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        global loaded_model

        # Check if the model is loaded
        if loaded_model is None:
            raise ValueError("Model not loaded. Please check server logs for errors during model initialization.")

        # Get the data from the request
        data = request.get_json()

        # Preprocess the data
        input_features = preprocess_input(data)

        # Make prediction using the model
        predicted_price = loaded_model.predict(input_features)[0]

        # Ensure the prediction is returned as an integer
        predicted_price_int = int(round(predicted_price))

        # Save prediction to Supabase
        save_to_supabase(data, predicted_price_int)

        # Return the predicted price as JSON
        return jsonify({"predicted_price": predicted_price_int})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Preload the model when the server starts
    preload_model()
    app.run(debug=True, host='0.0.0.0', port=8000)