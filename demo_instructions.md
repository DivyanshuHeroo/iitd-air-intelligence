# Demo Instructions

## Environment Setup
First, create your virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Data Pipeline & Training
Run the data extraction, preprocessing, and training pipeline in order:
```bash
python scripts/00_load_data.py
python scripts/01_prepare_data.py
python scripts/02_train_models.py
python scripts/03_evaluate_models.py
```

## 2. Running Tests
Make sure everything is working as expected:
```bash
pytest tests/
```

## 3. Launching the Dashboard
Launch the Streamlit app to interact with the model:
```bash
streamlit run app/streamlit_app.py
```
This will open `http://localhost:8501`. Here you can try predicting PM2.5 at various South Delhi locations and analyze route exposures.

## 4. Launching the API
To start the FastAPI backend:
```bash
uvicorn app.api:app --reload
```
This will open `http://127.0.0.1:8000`. You can also visit `http://127.0.0.1:8000/docs` to see the OpenAPI UI.

## 5. API Usage
To test the API, you can use `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_pm25' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "lat": 28.545,
  "long": 77.1926,
  "temperature": 30.0,
  "humidity": 45.0,
  "pressure": 1005.0,
  "hour": 14,
  "day": 10,
  "dayofweek": 2,
  "month": 5,
  "is_weekend": 0,
  "pm2_5_lag_1": 60.5
}'
```

You should receive a JSON response with the predicted PM2.5 and AQI category.

## 6. 60-Second Video Demo Script
Read the following script during your video presentation:

*“This is IITD Air Intelligence, an end-to-end ML project for hyperlocal PM2.5 prediction in Delhi-NCR. I used an IITD-linked air pollution dataset, engineered temporal, spatial, meteorological, lag, and rolling features, benchmarked multiple models using time-based validation, and deployed the best model through a Streamlit dashboard and FastAPI service. I also added a route exposure advisor around IIT Delhi and South Delhi.”*
