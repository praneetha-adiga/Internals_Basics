from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from datetime import datetime
import json
import os

app = FastAPI()

# ------------------------
# Load trained model
# ------------------------
model = joblib.load("models/best_model.pkl")


# ------------------------
# Input validation schema
# ------------------------
class InputData(BaseModel):
    prompt_token_count: int = Field(ge=10, le=1000)
    system_prompt_length: int = Field(ge=50, le=2000)
    temperature: float = Field(ge=0, le=1.5)
    is_few_shot: int = Field(ge=0, le=1)


# ------------------------
# Health check endpoint
# ------------------------
@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "operational",
        "service": "PromptLab API"
    }


# ------------------------
# Prediction endpoint + logging
# ------------------------
@app.post("/predict")
def predict(data: InputData):

    # Prepare input for model
    X = np.array([[
        data.prompt_token_count,
        data.system_prompt_length,
        data.temperature,
        data.is_few_shot
    ]])

    # Prediction
    prediction = model.predict(X)[0]

    # ------------------------
    # LOGGING (IMPORTANT FOR TASK 3)
    # ------------------------
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "prompt_token_count": data.prompt_token_count,
            "system_prompt_length": data.system_prompt_length,
            "temperature": data.temperature,
            "is_few_shot": data.is_few_shot
        },
        "prediction": float(prediction)
    }

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # Append log in JSONL format
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Return response
    return {
        "prediction": float(prediction)
    }