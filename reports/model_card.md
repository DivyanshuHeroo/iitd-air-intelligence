# Model Card: IITD Air Intelligence PM2.5 Forecaster

## Model Name
IITD Air Intelligence - Hyperlocal PM2.5 Predictor (Ridge Regression)

## Purpose
The primary purpose of this model is to demonstrate a complete, end-to-end Machine Learning pipeline for an internship portfolio. It forecasts hyperlocal PM2.5 values in Delhi-NCR (specifically around IIT Delhi) 1-3 hours ahead based on historical pollution and meteorology.

## Intended Use
- **Primary Use Case**: Predicting short-term PM2.5 levels at specific latitude-longitude coordinates to assist in personal route-level exposure analysis.

## Not Intended Use
- **Out-of-Scope**: Not designed for long-term climate prediction or predicting exact AQI out of the Delhi-NCR region.

## Target Variable
- **Continuous PM2.5 concentration** (µg/m³).

## Input Features
The model relies on spatial, temporal, and historical weather/pollution variables:
- **Temporal**: Hour, Day, Month, Weekend indicator, cyclical hour/day sine/cosine encodings.
- **Spatial**: Latitude, Longitude, Haversine distance from IIT Delhi.
- **Meteorological**: Temperature, Humidity, Pressure.
- **Lags & Rolling**: Past PM2.5 and PM10 values at 1, 3, and 6 hours, plus rolling averages.

## Training Data
- Source: `sachin-iitd/DelhiPollDataset`
- Period: Hourly snapshots over several months in 2020/2021.
- Validated via a strict Time-Based Split (80% training / 20% testing).

## Evaluation Metrics
- Evaluated on **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R2)**.

## Ethical Considerations & Limitations
- **Educational Use Only:** This model is for educational and portfolio demonstration purposes only. It should not be used as an official air quality warning, medical, or regulatory system.
- The model assumes the future behaves largely like the past. 
- It heavily relies on the nearest past readings (`lag_1`), struggling to predict sudden, unprecedented spikes without external real-time data like traffic or industrial accidents.

## Deployment Notes
- Lightweight inference suitable for CPU edge deployment.
- Accessible via a FastAPI REST endpoint and a Streamlit frontend.
