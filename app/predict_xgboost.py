import pandas as pd
import xgboost as xgb  # ✅ use XGBoost
import joblib          # for loading saved models

# ✅ Load the correct XGBoost model
model = joblib.load('models/xgboost_weather_model.pkl')

# Sample input (make sure it matches the features used in training!)
sample_input = pd.DataFrame([{
    'weathercode': 2,
    'temperature_2m_min': 22.6,
    'temperature_2m_mean': 26.1,
    'apparent_temperature_max': 34.4,
    'apparent_temperature_min': 24.9,
    'apparent_temperature_mean': 29.1,
    'shortwave_radiation_sum': 25.3,
    'precipitation_sum': 0.4,
    'rain_sum': 0.4,
    'snowfall_sum': 0.0,
    'precipitation_hours': 1,
    'windspeed_10m_max': 11.7,
    'windgusts_10m_max': 27.4,
    'winddirection_10m_dominant': 20,
    'et0_fao_evapotranspiration': 4.61,
    'latitude': 6.9,
    'longitude': 79.9,
    'elevation': 10,
    'day': 3,
    'month': 7,
    'year': 2025,
    'dayofweek': 3  # Wednesday
}])

# Ensure the input columns match training features
# (Optional: load saved feature names if needed)

# Make prediction
prediction = model.predict(sample_input)

print("Predicted result:", prediction[0])
