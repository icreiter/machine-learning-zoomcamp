import os
import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Constants and Model Loading ---
PIPELINE_FILE = "pipeline_v1.bin"

# Ensure the model file exists before attempting to load
if not os.path.exists(PIPELINE_FILE):
    print(
        f"Error: {PIPELINE_FILE} not found. Please ensure the model file is in the same directory."
    )
    exit(1)

with open(PIPELINE_FILE, "rb") as f_in:
    # Load the complete scikit-learn pipeline
    pipeline = pickle.load(f_in)


# --- Prediction Function (Wrapper) ---
def predict_conversion_proba(record, pipeline):
    """
    Transforms the record and predicts the conversion probability.
    """
    # Convert Pydantic model to a dict list for the pipeline
    X_record = [record.dict()]

    # Predict the probability of the positive class (index 1)
    y_pred_proba = pipeline.predict_proba(X_record)[:, 1]
    return float(y_pred_proba[0])


# --- 2. Define Pydantic Data Model for Request Body ---
class Lead(BaseModel):
    """Data model for a single lead record."""

    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


# --- 3. Initialize FastAPI App and Endpoint ---
app = FastAPI()


@app.post("/predict")
def predict_lead_conversion(lead: Lead):
    """
    Endpoint to predict the conversion probability for a lead.
    """
    probability = predict_conversion_proba(lead, pipeline)

    result = {"conversion_probability": probability}
    return result


# --- 4. Main Execution for Uvicorn ---
if __name__ == "__main__":
    # Uvicorn will handle the blocking process here
    uvicorn.run(app, host="0.0.0.0", port=9695)
