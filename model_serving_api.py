import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
from typing import List
import os
import sys

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring Model API",
    description="Serve a trained credit scoring machine-learning model for predictions.",
    version="1.0"
)

# Load the trained model 
MODEL_PATH = "notebooks/transaction_level_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to accept a CSV file and return predictions.
    """
    try:
        # Read the uploaded CSV file into a DataFrame
        data = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the CSV file: {e}")

    from scripts.feature_engineering import FeatureEngineering

    # Create an instance of FeatureEngineering
    features = FeatureEngineering(data)

    # Step 1: Extract Date Features
    data = features.extract_date_features()

    # Step 2: Encode Categorical Variables
    data = features.encode_categorical_variables(method='label')

    # Step 3: Scale Numerical Features
    data = features.scale_numerical_features(method='standardize')

    # Validate required columns **after preprocessing**
    required_columns = [
        'Amount', 'Value', 'transaction_hour', 'transaction_day', 'transaction_month',
        'transaction_year', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory'
    ]

    if not set(required_columns).issubset(data.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessed data must contain the following columns: {required_columns}"
        )

    # Step 4: Ensure only required columns are passed to the model
    input_data = data[required_columns]

    # Step 5: Make predictions
    predictions = model.predict(input_data)

    # Return predictions as a JSON response
    return {"predictions": predictions.tolist()}


@app.get("/")
def read_root():
    """
    Root endpoint for API health check.
    """
    return {"message": "Welcome to the Credit Scoring Model API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
