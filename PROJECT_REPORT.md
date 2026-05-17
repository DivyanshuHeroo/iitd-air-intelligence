# IITD Air Intelligence - Project Report

## 1. Abstract
Air pollution in Delhi-NCR is a severe and persistent problem, impacting the health of millions. This project presents a hyperlocal PM2.5 forecasting model using machine learning, allowing individuals to evaluate expected air quality at a granular level, specifically focusing on the IIT Delhi campus and surrounding South Delhi areas. The end-to-end system includes robust data preprocessing, leakage-free feature engineering, time-series model validation, a deployed Streamlit dashboard for real-time inference, and a FastAPI backend.

## 2. Architecture Diagram

```
Raw Air Pollution Data (Hugging Face)
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering (Spatial, Temporal, Lags)
        ↓
Time-Based Train/Test Split (80/20)
        ↓
Model Training & Baseline Comparison
        ↓
Evaluation & Error Analysis
        ↓
Best Model Saved (Ridge Regression)
        ↓
Streamlit Dashboard + FastAPI Service
        ↓
IIT Delhi / South Delhi Prediction & Route Exposure Advisor
```

## 3. Problem Statement
City-wide Air Quality Indices (AQI) provide a generalized view of pollution but fail to capture hyperlocal variations caused by traffic patterns, local emissions, and microclimates. A hyperlocal predictive model empowers individuals to make informed decisions regarding their daily routes and outdoor activities.

## 4. Dataset
The project utilizes the `sachin-iitd/DelhiPollDataset` hosted on Hugging Face. This IITD-linked dataset contains hourly PM values and meteorological data (temperature, humidity, pressure) across various coordinates in the Delhi region over several months.

## 5. Preprocessing
- **Missing Values:** Rows with missing target (PM2.5) were discarded.
- **Constraints:** Imposed strict bounds to remove anomalous/impossible values (e.g., negative pollution values, invalid lat/long).
- **Time Sorting:** The entire dataset was sorted chronologically to prepare for sequential lag generation.

## 6. Feature Engineering
Engineered features strictly without looking into the future:
- **Temporal Components:** Hour, Day, Month, Weekend indicator.
- **Cyclical Encoding:** Sine and Cosine transformations for hour of day and day of week to represent the continuous nature of time.
- **Spatial Components:** Calculated the Haversine distance from IIT Delhi coordinates.
- **Auto-regressive Lags:** PM2.5 and PM10 values shifted by 1, 3, and 6 hours.
- **Rolling Windows:** 3-hour and 6-hour rolling averages, applied *after* shifting to prevent data leakage.

## 7. Validation Strategy
To avoid look-ahead bias, a **Time-Based Split** (80% training, 20% testing) was utilized. This ensures the model is evaluated on future unseen data relative to its training corpus, accurately reflecting real-world deployment performance. 

## 8. Models Benchmarked
We benchmarked several models against a **Persistence Baseline** (where the prediction for time $t$ is simply the value at time $t-1$).
- Ridge Regression (Linear Baseline)
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

## 9. Results
On the out-of-time test set, **Ridge Regression** consistently outperformed the Persistence Baseline and the tree-based models. While Gradient Boosting methods (XGBoost/LightGBM) achieved near-zero error on the training set, they struggled to generalize to the temporal shifts in the unseen test set, indicating overfitting to early seasonal patterns. Ridge Regression proved more robust and reliable.

## 10. Error Analysis
Error analysis (saved in `reports/metrics/error_analysis.csv`) reveals that the largest absolute errors occur during sudden, extreme pollution spikes (e.g., festival nights, sudden inversions). Historical lags alone cannot predict these without external context.

## 11. Deployment (Dashboard & API)
- **FastAPI Service:** Handles high-throughput model inference requests.
- **Streamlit Dashboard:** Provides an interactive UI for users to predict PM2.5, visualize AQI categories, and utilize the unique **Route Exposure Advisor**, which calculates average pollution exposure along common routes in South Delhi.

## 12. Limitations
- **Educational Use:** This model is a portfolio demonstration and should not be used as an official medical or regulatory air quality advisory.
- **Cold Start:** Real-time prediction heavily relies on access to recent past values (`lag_1`).
- **Missing Modalities:** The model does not currently incorporate live traffic data, satellite AOD (Aerosol Optical Depth), or industrial emission reports.

## 13. Future Work
- Integrate live API feeds (e.g., CPCB or OpenWeather) for real-time lag data.
- Explore sequence-to-sequence Deep Learning models (LSTMs or Transformers).
- Expand route exposure to include dynamic routing via Google Maps API.

## 14. Interview Explanation
“I built IITD Air Intelligence, an end-to-end machine learning system for hyperlocal PM2.5 prediction in Delhi-NCR using an IITD-linked air pollution dataset. I engineered spatial, temporal, meteorological, lag, and rolling-window features, used time-based validation to avoid leakage, benchmarked multiple models against a persistence baseline, and deployed the best model through a Streamlit dashboard and FastAPI service. I also added a route exposure advisor around IIT Delhi and South Delhi, so the project goes beyond prediction and turns ML output into a practical decision-support tool.”
