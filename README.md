🌦️ Smart Weather Prediction System

Check our demo video of our Smart Weather Prediciton System


📌 Project Overview

This project is a Weather Prediction Web Application built with Django and Machine Learning.

Fetches current weather conditions from an external API.

Uses a trained ML model to predict rain and other conditions.

Provides a user-friendly web interface for weather forecasting.
⚙️ Features

🌐 Fetches live weather data from [OpenWeatherMap API / your chosen API].

🤖 Machine Learning model trained on historical weather datasets.

📊 Predicts rain / humidity / temperature trends.

🖥️ Django-based web application with interactive UI.

📈 Graphical representation of results (optional).

🛠️ Tech Stack

Backend: Django (Python 3.12)

Frontend: HTML, CSS, Bootstrap (or your stack)

Machine Learning: scikit-learn, pandas, numpy

API: OpenWeatherMap (or whichever you used)

Database: SQLite (default in Django)

📂 Project Structure

WeatherPredictor/
│── forecast/           # Django app (views, models, ML integration)
│── templates/          # HTML templates
│── static/             # CSS, JS, images
│── venv/               # Virtual environment
│── manage.py
│── requirements.txt    # Dependencies


#🚀 Installation & Setup

Clone the repository

git clone https://github.com/your-username/Smart-weather-prediction-system.git
cd Smart-weather-prediction-system


Create & activate virtual environment

python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell


Install dependencies

pip install -r requirements.txt


Run migrations

python manage.py migrate


Start the development server

python manage.py runserver


Open in browser: http://127.0.0.1:8000

📊 Machine Learning Model

Trained on historical weather dataset (temperature, humidity, wind, rainfall).

Model used: Logistic Regression / Random Forest / your chosen model.

Predictions may vary depending on dataset quality and API fluctuations.

⚠️ Limitations

Predictions are not always 100% accurate due to dataset constraints.

Real-world weather is influenced by many complex factors not covered in the dataset.

This is a university project – accuracy is secondary to demonstrating ML integration.



Happy Predicting! ☕️




