# 🌦️ Smart Weather Prediction System using Machine Learning

This project aims to build an intelligent weather prediction system using machine learning techniques. It leverages historical weather data to forecast parameters like temperature, humidity, rainfall, etc.

---
# 🐋 How to Run the Smart Weather Prediction System Using Docker

This guide lets you run your Smart Weather Prediction System project inside a Docker container using a Jupyter Notebook environment.

---

## ✅ 1. Prerequisites

- **Install Docker Desktop**:\
  [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

- Verify installation:

  ```bash
  docker --version
  ```

---

## 📁 2. Ensure the Following Project Structure

```
Smart-weather-prediction-system-using-ML/
├── data/                    # CSV files
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build config (create this)
```

---

## 🧱 3. Create a Dockerfile

In the project root, create a file named `Dockerfile` and paste:

```Dockerfile
FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install notebook jupyter

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

---

## 📦 4. Build the Docker Image

From the project root:

```bash
docker build -t smart-weather-predictor .
```

---

## ▶️ 5. Run the Docker Container

```bash
docker run -it -p 8888:8888 smart-weather-predictor
```

You will see a terminal output with a URL like:

```
http://127.0.0.1:8888/?token=your_token_here
```

Copy this URL and paste it into your browser.

---

## 📅 6. Open and Run the Notebook

In Jupyter UI:

- Navigate to: `notebooks/01_lightgbm_model.ipynb`
- Run the notebook cells to execute the model.

---

## ♻️ Rebuild if Code Changes

If you change any dependencies or major code:

```bash
docker build -t smart-weather-predictor .
```

---

Happy Predicting! ☕️



