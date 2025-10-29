from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from datetime import datetime
import re

# ---------------- FastAPI App ----------------
app = FastAPI()

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# ---------------- Templates & Static ----------------
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------- Load Model ----------------
model = CatBoostRegressor()
model.load_model(os.path.join(BASE_DIR, "models/catboost_model.cbm"))

# ---------------- Load Weather Data ----------------
df_weather = pd.read_csv(os.path.join(BASE_DIR, "data/weatherData.csv"))
df_weather['date'] = pd.to_datetime(df_weather['date'], errors='coerce')
df_weather = df_weather.dropna(subset=['date']).drop_duplicates().reset_index(drop=True)
df_weather['dayofyear'] = df_weather['date'].dt.dayofyear
df_weather['weekday'] = df_weather['date'].dt.weekday

# Rename columns
rename_map = {
    'temperature_2m_mean (°C)': 'temp_mean',
    'precipitation_sum (mm)': 'precip_sum',
    'rain_sum (mm)': 'rain_sum',
    'wind_speed_10m_max (km/h)': 'wind_speed_max',
    'weather_code (wmo code)': 'weather_code'
}
df_weather = df_weather.rename(columns=rename_map)

# Targets
targets = ['rain_sum', 'precip_sum', 'temp_mean', 'wind_speed_max']

# Historical averages
for target in targets:
    hist = df_weather.groupby(['location_id', 'dayofyear'])[target].mean().reset_index()
    hist.rename(columns={target: f'{target}_hist'}, inplace=True)
    df_weather = df_weather.merge(hist, on=['location_id', 'dayofyear'], how='left')

# Feature & categorical
features = model.feature_names_
cat_features = ['location_id', 'weather_code']

# Load City Mapping & clean city names
df_cities = pd.read_csv(os.path.join(BASE_DIR, "data/locationData.csv"))

def clean_city_name(name):
    return re.sub(r'\[.*?\]', '', name).strip().lower()

city_to_id = {clean_city_name(row['city_name']): row['location_id'] for _, row in df_cities.iterrows()}

# ---------------- Helper Function ----------------
def make_prediction(location_id: int, date: str):
    date_obj = pd.to_datetime(date)
    dayofyear = date_obj.dayofyear
    weekday = date_obj.weekday()
    month = date_obj.month

    # Cyclic features
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365)
    dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    # Most frequent weather_code
    weather_series = df_weather[
        (df_weather['location_id']==location_id) & 
        (df_weather['dayofyear']==dayofyear)
    ]['weather_code'].mode()
    weather_code = int(weather_series[0]) if len(weather_series) > 0 else 1

    # Historical averages
    hist_values = {}
    for target in targets:
        hist = df_weather[
            (df_weather['location_id']==location_id) & 
            (df_weather['dayofyear']==dayofyear)
        ][f'{target}_hist'].mean()
        hist_values[f'{target}_hist'] = hist if not np.isnan(hist) else 0

    # Input DataFrame
    X_input = pd.DataFrame([{
        'location_id': location_id,
        'year': date_obj.year,
        'month': month,
        'weather_code': weather_code,
        'weekday': weekday,
        'dayofyear': dayofyear,
        'dayofyear_sin': dayofyear_sin,
        'dayofyear_cos': dayofyear_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'weekday_sin': weekday_sin,
        'weekday_cos': weekday_cos,
        'temp_mean_hist': hist_values['temp_mean_hist'],
        'rain_sum_hist': hist_values['rain_sum_hist'],
        'precip_sum_hist': hist_values['precip_sum_hist'],
        'wind_speed_max_hist': hist_values['wind_speed_max_hist']
    }])

    # Convert categorical features
    X_input = X_input.astype({'location_id':'category','weather_code':'category'})

    # Ensure all features present
    for f in features:
        if f not in X_input.columns:
            X_input[f] = 0

    X_input = X_input[features]
    pool = Pool(X_input, cat_features=cat_features)
    y_pred = model.predict(pool)

    # Handle 2D / 1D prediction output
    if y_pred.ndim == 1:
        return dict(zip(targets, np.round(y_pred, 2)))
    return dict(zip(targets, np.round(y_pred[0], 2)))

# ---------------- Routes ----------------
@app.get("/")
def home(request: Request):
    default_city = "Jaffna"
    default_date = datetime.today().strftime('%Y-%m-%d')
    default_location_id = city_to_id.get(clean_city_name(default_city), 9)
    prediction = make_prediction(default_location_id, default_date)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_city": default_city,
        "default_date": default_date,
        "default_location_id": default_location_id,
        "prediction": prediction
    })

@app.post("/predict")
async def predict(city_name: str = Form(...), date: str = Form(...)):
    city_clean = clean_city_name(city_name)
    if city_clean not in city_to_id:
        return JSONResponse({"error": f"City '{city_name}' not found."}, status_code=400)
    location_id = city_to_id[city_clean]
    prediction = make_prediction(location_id, date)
    return JSONResponse({"prediction": prediction, "location_id": location_id})
