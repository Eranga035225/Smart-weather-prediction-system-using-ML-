import os
import csv
import pandas as pd
import joblib
import datetime

# Load trained models
temp_model = joblib.load('models/lightgbm_temp_model.pkl')
rain_model = joblib.load('models/lightgbm_rain_model.pkl')

# Define district-wise info
district_info = {
    "Colombo": {"location_id": 0, "latitude": 6.9244, "longitude": 79.9072, "elevation": 4},
    "Kandy": {"location_id": 3, "latitude": 7.2759, "longitude": 80.6266, "elevation": 499},
    "Jaffna": {"location_id": 9, "latitude": 9.6661, "longitude": 80.0349, "elevation": 8},
    # Add more districts as needed
}

# 1. Accept user inputs
district_name = input("Enter district name (e.g., Colombo): ").strip()
input_date_str = input("Enter date (YYYY-MM-DD): ").strip()

# Validate district
if district_name not in district_info:
    print("❌ Invalid district. Please check spelling.")
    exit()

# Validate date format
try:
    input_date = datetime.datetime.strptime(input_date_str, "%Y-%m-%d")
except ValueError:
    print("❌ Invalid date format. Use YYYY-MM-DD.")
    exit()

info = district_info[district_name]

# Build input sample (dummy weather values)
sample = pd.DataFrame([{
    'weather_code (wmo code)': 1,
    'temperature_2m_min (°C)': 22,
    'temperature_2m_mean (°C)': 26,
    'apparent_temperature_max (°C)': 30,
    'apparent_temperature_min (°C)': 24,
    'apparent_temperature_mean (°C)': 27,
    'daylight_duration (s)': 40000,
    'sunshine_duration (s)': 35000,
    'precipitation_sum (mm)': 0.2,
    'precipitation_hours (h)': 2,
    'wind_speed_10m_max (km/h)': 15,
    'wind_gusts_10m_max (km/h)': 30,
    'wind_direction_10m_dominant (°)': 180,
    'shortwave_radiation_sum (MJ/m²)': 20,
    'et0_fao_evapotranspiration (mm)': 4,
    'latitude': info['latitude'],
    'longitude': info['longitude'],
    'elevation': info['elevation'],
    'year': input_date.year,
    'month': input_date.month,
    'day': input_date.day,
    'daylight_hours': 12
}])

# 3. Predict
pred_temp = temp_model.predict(sample)[0]
pred_rainfall = rain_model.predict(sample)[0]

# Ensure folder exists
os.makedirs("prediction_results", exist_ok=True)

# Save result
result_file = "prediction_results/predictions.csv"
file_exists = os.path.isfile(result_file)

with open(result_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["district", "date", "predicted_max_temp", "predicted_rainfall"])
    writer.writerow([district_name, input_date.strftime("%Y-%m-%d"), round(pred_temp, 2), round(pred_rainfall, 2)])

print("\n✅ Prediction saved to 'prediction_results/predictions.csv'")
print("🔸 District:", district_name)
print("🔸 Date:", input_date.strftime("%Y-%m-%d"))
print("🔸 Predicted Maximum Temperature:", round(pred_temp, 2))
print("🔸 Predicted Rainfall:", round(pred_rainfall, 2))

print("\n✅ Prediction saved to 'prediction_results/predictions.csv'")