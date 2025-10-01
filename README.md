# 🌦️ Smart Weather Prediction System  

check this demo video to check the functionality of our weather prediciton system

## 📌 Project Overview  
This is a **Weather Prediction Web Application** built with **Django and Machine Learning**.  
- Fetches **current weather data** from an external API.  
- Uses a trained ML model to **predict rain and other conditions**.  
- Provides a user-friendly **web interface** for forecasting.  

---

## ⚙️ Features  
- 🌐 Live weather data from [OpenWeatherMap API / your API].  
- 🤖 Machine Learning model trained on historical datasets.  
- 📊 Predicts rainfall, humidity, and temperature trends.  
- 🖥️ Django-based web app with interactive UI.  

---

## 🛠️ Tech Stack  
- **Backend:** Django (Python 3.12)  
- **Frontend:** HTML, CSS, Bootstrap  
- **Machine Learning:** scikit-learn, pandas, numpy  
- **API:** OpenWeatherMap (or your chosen API)  
- **Database:** SQLite  

---

## 📂 Project Structure  
For the WeatherPredictor app check follwing path in the repo
│── WeatherPredictor/forecast # Django app (views, models, ML integration)


---

## 🚀 Installation & Setup  

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/Smart-weather-prediction-system.git
cd Smart-weather-prediction-system

python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell

pip install -r requirements.txt

python manage.py migrate

python manage.py runserver

Open in browser
```


📊 Machine Learning Model

1) Trained on historical weather dataset (temperature, humidity, wind, rainfall).

2) Model used: Logistic Regression / Random Forest 

3) Predictions depend on dataset quality and real-time API inputs.

⚠️ Limitations

1) Predictions are not always 100% accurate due to dataset constraints.

2) Weather forecasting is influenced by complex factors not in the dataset.

3) This is a university project, focusing on ML + API integration.







Happy Predicting! ☕️





