from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: Supabase URL or Key is not set. Running in mock mode.")
    supabase = None
else:
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
