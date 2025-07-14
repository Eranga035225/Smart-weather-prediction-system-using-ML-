import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
import os
import numpy as np

# Load cleaned dataset
df = pd.read_csv('data/cleaned_weather.csv')

# Rename important columns for simplicity
df.rename(columns={
    'temperature_2m_max (°C)': 'temperature_max',
    'rain_sum (mm)': 'rain_sum'
}, inplace=True)

# Drop non-feature columns
drop_cols = ['temperature_max', 'rain_sum', 'city_name', 'sunrise_hour', 'sunset_hour', 'location_id']
X = df.drop(columns=drop_cols, errors='ignore')

# Targets
y_temp = df['temperature_max']
y_rain = df['rain_sum']

# Train-test split
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)

# Initialize LightGBM models
temp_model = lgb.LGBMRegressor()
rain_model = lgb.LGBMRegressor()

# Train models
temp_model.fit(X_train_temp, y_train_temp)
rain_model.fit(X_train_rain, y_train_rain)

# Evaluate
print("🔸 Temperature MAE:", mean_absolute_error(y_test_temp, temp_model.predict(X_test_temp)))
print("🔸 Temperature RMSE:", np.sqrt(mean_squared_error(y_test_temp, temp_model.predict(X_test_temp))))
print("🔹 Rainfall MAE:", mean_absolute_error(y_test_rain, rain_model.predict(X_test_rain)))
print("🔹 Rainfall RMSE:", np.sqrt(mean_squared_error(y_test_rain, rain_model.predict(X_test_rain))))

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(temp_model, "models/lightgbm_temp_model.pkl")
joblib.dump(rain_model, "models/lightgbm_rain_model.pkl")

print("✅ Models saved in 'models/' folder.")
