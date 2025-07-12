from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn
import os

# Define input schema
class WeatherInput(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    city_name: int  # encoded city name
    precipitation_sum: float
    rain_sum: float
    snowfall_sum: float
    wind_speed_max: float
    shortwave_radiation_sum: float
    month: int
    day: int
    dayofweek: int

# Load model
model_path = "models/xgboost_model.json"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Please run train_xgboost.py first.")

model = xgb.XGBRegressor()
model.load_model(model_path)
print("✅ Model loaded")

# Create FastAPI app
app = FastAPI(title="Weather Prediction API")

@app.post("/predict")
def predict_temperature(data: WeatherInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    # Predict
    prediction = model.predict(input_df)[0]
    return {"predicted_temperature": round(float(prediction), 2)}

# Run the server (if this file is run directly)
if __name__ == "__main__":
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=True)
