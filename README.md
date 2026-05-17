# IITD Air Intelligence: Hyperlocal PM2.5 Prediction for Delhi-NCR

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20LightGBM-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![IIT Delhi](https://img.shields.io/badge/IIT%20Delhi-Air%20Quality-blueviolet.svg)

IITD Air Intelligence is an end-to-end machine learning project that predicts hyperlocal PM2.5 levels across Delhi-NCR using an IITD-linked air pollution dataset. It combines feature engineering, time-aware validation, baseline benchmarking, model deployment, and a route exposure advisor focused on IIT Delhi and South Delhi.

![Dashboard Overview](reports/screenshots/dashboard_overview.png)

## 1. Recruiter / Interview Material
- [Recruiter Summary](RECRUITER_SUMMARY.md)
- [Resume Bullets](RESUME.md)
- [Interview Q&A](INTERVIEW_QA.md)
- [Demo Video Script](DEMO_VIDEO_SCRIPT.md)
- [Project One-Pager](PROJECT_ONE_PAGER.md)
- [LinkedIn Post Draft](LINKEDIN_POST.md)

## 2. Why this project?
Air pollution is highly localized. While city-wide Air Quality Indices (AQI) give a broad picture, they fail to provide actionable intelligence for individual neighborhoods or commute routes. This project bridges that gap by modeling PM2.5 at specific coordinates based on meteorology and historical lag data, converting raw ML predictions into practical exposure advice.

## 3. IIT Delhi Relevance
Utilizing the IITD-linked `AirDelhi` dataset, the dashboard natively supports demo locations across South Delhi (e.g., Hauz Khas, Green Park, SDA Market). The embedded "Route Exposure Advisor" demonstrates a localized application tailored to the IIT Delhi community.

## 4. Key Features
- **Zero-Leakage Engineering**: Shifted lag and rolling features designed explicitly for tabular time-series tasks.
- **Robust Validation**: 80/20 time-based split to ensure out-of-time generalization.
- **Multiple Interfaces**: A responsive Streamlit dashboard and a high-performance FastAPI service.
- **Exposure Advisor**: Calculates estimated PM2.5 exposure along a designated geographical route.

## 5. Architecture

```text
Raw Air Pollution Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Time-Based Split
        ↓
Baseline + ML Models
        ↓
Evaluation
        ↓
Best Model
        ↓
Streamlit Dashboard + FastAPI
        ↓
IIT Delhi Prediction + Route Exposure Advisor
```

## 6. Dataset
- **Source**: `sachin-iitd/DelhiPollDataset` (Hugging Face mirror of the AirDelhi dataset).
- **Features Used**: Spatial (Lat/Long), Temporal (Hour/Day), Meteorological (Temp/Humidity/Pressure), and Historical Lags (PM2.5/PM10).

## 7. ML Pipeline
1. Extract data programmatically via Hugging Face.
2. Clean impossible/anomalous weather values.
3. Engineer cyclical time encodings and Haversine distances.
4. Train and benchmark Ridge Regression against complex trees (Random Forest, XGBoost, LightGBM) and a naive Persistence Baseline.

## 8. Results
Based on strict out-of-time validation, **Ridge Regression** proved to be the most reliable forecaster. Complex boosting models overfitted the earlier distribution blocks and failed to generalize as well to unseen temporal periods. The detailed model leaderboard is logged in `reports/metrics/model_results.csv`.

## 9. Dashboard
The Streamlit dashboard (`app/streamlit_app.py`) provides interactive prediction forms, Folium map visualizations, and model interpretation tabs.
![Route Exposure Advisor](reports/screenshots/route_exposure_advisor.png)

## 10. API
The FastAPI backend (`app/api.py`) exposes `/predict_pm25`, returning the continuous prediction and interpreted AQI category. It auto-computes necessary distance and cyclical features from minimal input.
![FastAPI Docs](reports/screenshots/fastapi_docs.png)

## 11. Docker
The repository includes a `Dockerfile` for simple API deployment:
```bash
docker build -t iitd-air-intelligence .
docker run -p 8000:8000 iitd-air-intelligence
```

## 12. Deployment
Please refer to `DEPLOYMENT.md` for detailed instructions on hosting via Streamlit Community Cloud and Render/Railway. 

## 13. How to Run
To run the full pipeline locally:
```bash
# 1. Setup Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run Pipeline Commands (Or use `make all`)
python scripts/00_load_data.py
python scripts/01_prepare_data.py
python scripts/02_train_models.py
python scripts/03_evaluate_models.py

# 3. Launch Dashboard
streamlit run app/streamlit_app.py

# 4. Launch API
uvicorn app.api:app --reload
```
*Note: If the dataset is not downloaded, you can test features via the sample data provided in `data/sample/`.*

## 14. Repository Structure
```
iitd-air-intelligence/
├── app/                  # Streamlit dashboard and FastAPI backend
├── data/                 # Raw, processed, and sample datasets
├── models/               # Serialized models and feature configurations
├── notebooks/            # Programmatically generated Jupyter notebooks
├── reports/              # Plots, metrics, error analysis, screenshots, model card
├── scripts/              # Pipeline execution scripts
├── src/                  # Core modules (preprocessing, features, train, predict)
└── tests/                # Pytest suite
```

## 15. Limitations
- **Educational Use**: This model is for portfolio demonstration and should not be used as an official health advisory.
- **Autoregressive Dependency**: Predicting without recent lag data significantly reduces model accuracy.

## 16. Future Work
- Integrate live traffic data streams (e.g., Google Maps API).
- Implement recurrent neural networks (LSTMs) for long-horizon multi-step forecasting.

## 17. Resume Bullets
- Built an end-to-end ML system for hyperlocal PM2.5 prediction in Delhi-NCR using an IITD-linked air pollution dataset.
- Engineered temporal, spatial, meteorological, lag, and rolling-window features while avoiding time-series leakage.
- Benchmarked persistence, linear, tree-based, and boosting models using time-based validation.
- Deployed the best model through a Streamlit dashboard and FastAPI inference service.
- Added an IIT Delhi / South Delhi route exposure advisor to convert predictions into actionable decisions.

## 18. Interview Pitch
“I built IITD Air Intelligence, an end-to-end machine learning system for hyperlocal PM2.5 prediction in Delhi-NCR using an IITD-linked air pollution dataset. I engineered spatial, temporal, meteorological, lag, and rolling-window features, used time-based validation to avoid leakage, benchmarked multiple models against a persistence baseline, and deployed the best model through a Streamlit dashboard and FastAPI service. I also added a route exposure advisor around IIT Delhi and South Delhi, so the project goes beyond prediction and turns ML output into a practical decision-support tool.”
