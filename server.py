import os
import time
import traceback
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Paths
MODELS_DIR = 'models'
DATA_DIR = 'data'
LOCATIONS_PATH = os.path.join(DATA_DIR, 'locations.csv')
CLEANED_WEATHER_PATH = os.path.join(DATA_DIR, 'cleaned_weather.csv')

# Load ML models on startup
models = {}
try:
    print("⏳ Loading machine learning models...")
    models['lgbm'] = joblib.load(os.path.join(MODELS_DIR, 'lgbm_weather_model.pkl'))
    models['xgboost'] = joblib.load(os.path.join(MODELS_DIR, 'xgboost_weather_model.pkl'))
    models['rf'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    traceback.print_exc()

# Load cities and pre-compute monthly averages
cities_list = []
monthly_defaults = {}  # keyed by (location_id, month)
historical_trends = {}  # keyed by location_id

def load_data_and_precompute():
    global cities_list, monthly_defaults, historical_trends
    if not os.path.exists(LOCATIONS_PATH) or not os.path.exists(CLEANED_WEATHER_PATH):
        print("⚠️ Data files locations.csv or cleaned_weather.csv are missing!")
        return
        
    try:
        print("⏳ Loading datasets and calculating historical statistics...")
        # Load locations
        locations_df = pd.read_csv(LOCATIONS_PATH)
        # Clean Kilinochchi[1] or other city names if they contain wiki references
        locations_df['city_name'] = locations_df['city_name'].apply(lambda x: x.split('[')[0].strip())
        cities_list = locations_df.to_dict(orient='records')
        
        # Load weather
        weather_df = pd.read_csv(CLEANED_WEATHER_PATH)
        
        # Calculate monthly statistics for each location
        # Group by location_id and month
        grouped = weather_df.groupby(['location_id', 'month'])
        
        # Compute means
        means = grouped.mean()
        
        # Populate defaults
        for (loc_id, m), row in means.iterrows():
            # Clean column names for mapping
            monthly_defaults[(int(loc_id), int(m))] = {
                'weathercode': float(row.get('weather_code (wmo code)', 2.0)),
                'temperature_2m_min': float(row.get('temperature_2m_min (°C)', 22.0)),
                'temperature_2m_mean': float(row.get('temperature_2m_mean (°C)', 26.0)),
                'apparent_temperature_max': float(row.get('apparent_temperature_max (°C)', 32.0)),
                'apparent_temperature_min': float(row.get('apparent_temperature_min (°C)', 24.0)),
                'apparent_temperature_mean': float(row.get('apparent_temperature_mean (°C)', 28.0)),
                'daylight_duration': float(row.get('daylight_duration (s)', 42200.0)),
                'sunshine_duration': float(row.get('sunshine_duration (s)', 35000.0)),
                'precipitation_sum': float(row.get('precipitation_sum (mm)', 0.5)),
                'rain_sum': float(row.get('rain_sum (mm)', 0.5)),
                'precipitation_hours': float(row.get('precipitation_hours (h)', 1.0)),
                'windspeed_10m_max': float(row.get('wind_speed_10m_max (km/h)', 12.0)),
                'windgusts_10m_max': float(row.get('wind_gusts_10m_max (km/h)', 25.0)),
                'winddirection_10m_dominant': float(row.get('wind_direction_10m_dominant (°)', 180.0)),
                'shortwave_radiation_sum': float(row.get('shortwave_radiation_sum (MJ/m²)', 18.0)),
                'et0_fao_evapotranspiration': float(row.get('et0_fao_evapotranspiration (mm)', 4.0))
            }
            
        # Pre-compute 12-month historical trends for charting
        for loc in cities_list:
            loc_id = int(loc['location_id'])
            loc_weather = weather_df[weather_df['location_id'] == loc_id]
            monthly_grouped = loc_weather.groupby('month').mean()
            
            trends = {
                'months': list(range(1, 13)),
                'temp_max': [float(monthly_grouped.loc[m, 'temperature_2m_max (°C)']) if m in monthly_grouped.index else 30.0 for m in range(1, 13)],
                'temp_min': [float(monthly_grouped.loc[m, 'temperature_2m_min (°C)']) if m in monthly_grouped.index else 22.0 for m in range(1, 13)],
                'rain': [float(monthly_grouped.loc[m, 'rain_sum (mm)']) if m in monthly_grouped.index else 2.0 for m in range(1, 13)],
                'wind': [float(monthly_grouped.loc[m, 'wind_speed_10m_max (km/h)']) if m in monthly_grouped.index else 10.0 for m in range(1, 13)],
                'radiation': [float(monthly_grouped.loc[m, 'shortwave_radiation_sum (MJ/m²)']) if m in monthly_grouped.index else 15.0 for m in range(1, 13)]
            }
            historical_trends[loc_id] = trends
            
        print("✅ Data pre-computation complete!")
    except Exception as e:
        print(f"⚠️ Error during pre-computation: {e}")
        traceback.print_exc()

# Initialize stats
load_data_and_precompute()

@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify(cities_list)

@app.route('/api/defaults', methods=['GET'])
def get_defaults():
    city_id = request.args.get('city_id', type=int)
    month = request.args.get('month', type=int)
    
    if city_id is None or month is None:
        return jsonify({'error': 'Missing city_id or month parameter'}), 400
        
    defaults = monthly_defaults.get((city_id, month))
    if defaults is None:
        # Fallback to general averages or first available city
        defaults = monthly_defaults.get((0, month)) or list(monthly_defaults.values())[0]
        
    return jsonify(defaults)

@app.route('/api/historical', methods=['GET'])
def get_historical():
    city_id = request.args.get('city_id', type=int)
    if city_id is None:
        return jsonify({'error': 'Missing city_id parameter'}), 400
        
    trends = historical_trends.get(city_id)
    if trends is None:
        trends = list(historical_trends.values())[0]  # fallback
        
    return jsonify(trends)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models.get('lgbm') or not models.get('xgboost') or not models.get('rf'):
        return jsonify({'error': 'Machine learning models are not loaded on server.'}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Parse inputs
        city_id = int(data.get('city_id', 0))
        lat = float(data.get('latitude', 6.9))
        lon = float(data.get('longitude', 79.9))
        elevation = float(data.get('elevation', 10.0))
        
        date_str = data.get('date', '2026-07-05')
        date_parsed = pd.to_datetime(date_str)
        year = date_parsed.year
        month = date_parsed.month
        day = date_parsed.day
        dayofweek = date_parsed.dayofweek
        
        weathercode = float(data.get('weathercode', 2.0))
        temp_min = float(data.get('temperature_2m_min', 22.0))
        temp_mean = float(data.get('temperature_2m_mean', 26.0))
        app_temp_max = float(data.get('apparent_temperature_max', 32.0))
        app_temp_min = float(data.get('apparent_temperature_min', 24.0))
        app_temp_mean = float(data.get('apparent_temperature_mean', 28.0))
        
        precip_sum = float(data.get('precipitation_sum', 0.0))
        rain_sum = float(data.get('rain_sum', 0.0))
        precip_hours = float(data.get('precipitation_hours', 0.0))
        
        wind_max = float(data.get('windspeed_10m_max', 12.0))
        wind_gust = float(data.get('windgusts_10m_max', 25.0))
        wind_dir = float(data.get('winddirection_10m_dominant', 180.0))
        
        rad_sum = float(data.get('shortwave_radiation_sum', 18.0))
        et0 = float(data.get('et0_fao_evapotranspiration', 4.0))
        
        daylight_duration = float(data.get('daylight_duration', 42200.0))
        sunshine_duration = float(data.get('sunshine_duration', 35000.0))
        
        # 1. Prepare LGBM/XGB Input Dataframe
        lgb_xgb_features = [
            'weathercode', 'temperature_2m_min', 'temperature_2m_mean',
            'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
            'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max',
            'winddirection_10m_dominant', 'et0_fao_evapotranspiration', 'latitude',
            'longitude', 'elevation', 'day', 'month', 'year', 'dayofweek'
        ]
        
        input_row = {
            'weathercode': weathercode,
            'temperature_2m_min': temp_min,
            'temperature_2m_mean': temp_mean,
            'apparent_temperature_max': app_temp_max,
            'apparent_temperature_min': app_temp_min,
            'apparent_temperature_mean': app_temp_mean,
            'shortwave_radiation_sum': rad_sum,
            'precipitation_sum': precip_sum,
            'rain_sum': rain_sum,
            'snowfall_sum': 0.0,  # snowfall is always 0 in Sri Lanka
            'precipitation_hours': precip_hours,
            'windspeed_10m_max': wind_max,
            'windgusts_10m_max': wind_gust,
            'winddirection_10m_dominant': wind_dir,
            'et0_fao_evapotranspiration': et0,
            'latitude': lat,
            'longitude': lon,
            'elevation': elevation,
            'day': day,
            'month': month,
            'year': year,
            'dayofweek': dayofweek
        }
        
        df_lgb_xgb = pd.DataFrame([input_row])[lgb_xgb_features]
        
        # Run LightGBM Prediction
        t_start = time.time()
        lgbm_pred = float(models['lgbm'].predict(df_lgb_xgb)[0])
        lgbm_latency = (time.time() - t_start) * 1000
        
        # Run XGBoost Prediction
        t_start = time.time()
        xgb_pred = float(models['xgboost'].predict(df_lgb_xgb)[0])
        xgb_latency = (time.time() - t_start) * 1000
        
        # 2. Prepare Random Forest Input Dataframe
        # Use average predicted temperature max as 'Temp_Max'
        avg_temp_max = (lgbm_pred + xgb_pred) / 2.0
        
        rf_features = [
            'location_id', 'weather_code (wmo code)', 'Temp_Max', 'Temp_Min', 'Temp_Mean',
            'Apparent_Temp_Max', 'Apparent_Temp_Min', 'Apparent_Temp_Mean',
            'Daylight_Duration', 'Sunshine_Duration', 'Precip_Hours', 'Wind_Speed_Max',
            'Wind_Gust_Max', 'Wind_Dir_Dominant', 'Radiation_Sum', 'Evapotranspiration',
            'elevation', 'Year', 'Month', 'Day'
        ]
        
        input_rf = {
            'location_id': float(city_id),
            'weather_code (wmo code)': weathercode,
            'Temp_Max': avg_temp_max,
            'Temp_Min': temp_min,
            'Temp_Mean': temp_mean,
            'Apparent_Temp_Max': app_temp_max,
            'Apparent_Temp_Min': app_temp_min,
            'Apparent_Temp_Mean': app_temp_mean,
            'Daylight_Duration': daylight_duration,
            'Sunshine_Duration': sunshine_duration,
            'Precip_Hours': precip_hours,
            'Wind_Speed_Max': wind_max,
            'Wind_Gust_Max': wind_gust,
            'Wind_Dir_Dominant': wind_dir,
            'Radiation_Sum': rad_sum,
            'Evapotranspiration': et0,
            'elevation': elevation,
            'Year': float(year),
            'Month': float(month),
            'Day': float(day)
        }
        
        df_rf = pd.DataFrame([input_rf])[rf_features]
        
        # Run Random Forest Prediction
        t_start = time.time()
        rf_pred = int(models['rf'].predict(df_rf)[0])
        rf_prob = float(models['rf'].predict_proba(df_rf)[0][1]) if hasattr(models['rf'], 'predict_proba') else (1.0 if rf_pred == 1 else 0.0)
        rf_latency = (time.time() - t_start) * 1000
        
        # Format response
        results = {
            'lgbm': {
                'temp_max': round(lgbm_pred, 2),
                'latency_ms': round(lgbm_latency, 2)
            },
            'xgboost': {
                'temp_max': round(xgb_pred, 2),
                'latency_ms': round(xgb_latency, 2)
            },
            'random_forest': {
                'rain_predicted': rf_pred == 1,
                'rain_prob': round(rf_prob * 100, 1),
                'latency_ms': round(rf_latency, 2)
            },
            'summary': {
                'avg_temp_max': round(avg_temp_max, 2),
                'will_rain': rf_pred == 1,
                'weather_condition': 'Stormy' if (rf_pred == 1 and wind_max > 25) else ('Rainy' if rf_pred == 1 else ('Cloudy' if weathercode > 2 else 'Sunny'))
            }
        }
        
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    model_choice = request.form.get('model', 'xgboost')  # xgboost, lgbm, or compare
    
    if file.filename == '':
        return jsonify({'error': 'Empty file selection'}), 400
        
    try:
        df = pd.read_csv(file)
        
        # Basic validation of columns
        required_cols = [
            'weathercode', 'temperature_2m_min', 'temperature_2m_mean',
            'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
            'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max',
            'winddirection_10m_dominant', 'et0_fao_evapotranspiration', 'latitude',
            'longitude', 'elevation', 'day', 'month', 'year'
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            # Try to map friendly or spaces/units column names
            # Map columns to model-friendly keys
            mapper = {
                'weather_code (wmo code)': 'weathercode',
                'temperature_2m_min (°C)': 'temperature_2m_min',
                'temperature_2m_mean (°C)': 'temperature_2m_mean',
                'apparent_temperature_max (°C)': 'apparent_temperature_max',
                'apparent_temperature_min (°C)': 'apparent_temperature_min',
                'apparent_temperature_mean (°C)': 'apparent_temperature_mean',
                'shortwave_radiation_sum (MJ/m²)': 'shortwave_radiation_sum',
                'precipitation_sum (mm)': 'precipitation_sum',
                'rain_sum (mm)': 'rain_sum',
                'precipitation_hours (h)': 'precipitation_hours',
                'wind_speed_10m_max (km/h)': 'windspeed_10m_max',
                'wind_gusts_10m_max (km/h)': 'windgusts_10m_max',
                'wind_direction_10m_dominant (°)': 'winddirection_10m_dominant',
                'et0_fao_evapotranspiration (mm)': 'et0_fao_evapotranspiration',
            }
            df = df.rename(columns=mapper)
            
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return jsonify({'error': f'Uploaded file is missing required columns: {missing}'}), 400
                
        # Fill missing features
        if 'snowfall_sum' not in df.columns:
            df['snowfall_sum'] = 0.0
            
        if 'dayofweek' not in df.columns:
            # compute dayofweek
            try:
                dates = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
                df['dayofweek'] = dates.dt.dayofweek
            except:
                df['dayofweek'] = 0
                
        # Run predictions
        results_df = df.copy()
        
        lgb_xgb_features = [
            'weathercode', 'temperature_2m_min', 'temperature_2m_mean',
            'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
            'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max',
            'winddirection_10m_dominant', 'et0_fao_evapotranspiration', 'latitude',
            'longitude', 'elevation', 'day', 'month', 'year', 'dayofweek'
        ]
        
        # Predict max temperatures
        df_for_model = results_df[lgb_xgb_features].fillna(0)
        
        if model_choice == 'lgbm' or model_choice == 'compare':
            results_df['Predicted_TempMax_LightGBM'] = models['lgbm'].predict(df_for_model)
        if model_choice == 'xgboost' or model_choice == 'compare':
            results_df['Predicted_TempMax_XGBoost'] = models['xgboost'].predict(df_for_model)
            
        # Predict rain using Random Forest
        # We need predicted Temp_Max for RF
        predicted_temp_max = results_df['Predicted_TempMax_XGBoost'] if 'Predicted_TempMax_XGBoost' in results_df.columns else results_df['Predicted_TempMax_LightGBM']
        
        rf_features = [
            'location_id', 'weather_code (wmo code)', 'Temp_Max', 'Temp_Min', 'Temp_Mean',
            'Apparent_Temp_Max', 'Apparent_Temp_Min', 'Apparent_Temp_Mean',
            'Daylight_Duration', 'Sunshine_Duration', 'Precip_Hours', 'Wind_Speed_Max',
            'Wind_Gust_Max', 'Wind_Dir_Dominant', 'Radiation_Sum', 'Evapotranspiration',
            'elevation', 'Year', 'Month', 'Day'
        ]
        
        # Prepare inputs for Random Forest
        rf_df = pd.DataFrame()
        rf_df['location_id'] = results_df.get('location_id', 0.0)
        rf_df['weather_code (wmo code)'] = results_df['weathercode']
        rf_df['Temp_Max'] = predicted_temp_max
        rf_df['Temp_Min'] = results_df['temperature_2m_min']
        rf_df['Temp_Mean'] = results_df['temperature_2m_mean']
        rf_df['Apparent_Temp_Max'] = results_df['apparent_temperature_max']
        rf_df['Apparent_Temp_Min'] = results_df['apparent_temperature_min']
        rf_df['Apparent_Temp_Mean'] = results_df['apparent_temperature_mean']
        rf_df['Daylight_Duration'] = results_df.get('daylight_duration (s)', 42200.0)
        rf_df['Sunshine_Duration'] = results_df.get('sunshine_duration (s)', 35000.0)
        rf_df['Precip_Hours'] = results_df['precipitation_hours']
        rf_df['Wind_Speed_Max'] = results_df['windspeed_10m_max']
        rf_df['Wind_Gust_Max'] = results_df['windgusts_10m_max']
        rf_df['Wind_Dir_Dominant'] = results_df['winddirection_10m_dominant']
        rf_df['Radiation_Sum'] = results_df['shortwave_radiation_sum']
        rf_df['Evapotranspiration'] = results_df['et0_fao_evapotranspiration']
        rf_df['elevation'] = results_df['elevation']
        rf_df['Year'] = results_df['year'].astype(float)
        rf_df['Month'] = results_df['month'].astype(float)
        rf_df['Day'] = results_df['day'].astype(float)
        
        rf_df_clean = rf_df[rf_features].fillna(0)
        
        results_df['Predicted_Rain_Today'] = models['rf'].predict(rf_df_clean)
        if hasattr(models['rf'], 'predict_proba'):
            results_df['Rain_Probability_%'] = np.round(models['rf'].predict_proba(rf_df_clean)[:, 1] * 100, 1)
            
        # Return summary of first 20 rows and success
        records = results_df.head(50).to_dict(orient='records')
        
        # Save output temporarily to serve it
        os.makedirs('static/downloads', exist_ok=True)
        out_filename = f"batch_prediction_{int(time.time())}.csv"
        out_path = f"static/downloads/{out_filename}"
        results_df.to_csv(out_path, index=False)
        
        return jsonify({
            'success': True,
            'preview': records,
            'total_rows': len(results_df),
            'download_url': f"/downloads/{out_filename}"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory('static/downloads', filename, as_attachment=True)

# Serve Frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        # Fallback to index.html for SPA routing
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Run server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
