# IITD Air Intelligence — Project One-Pager

## Title
**IITD Air Intelligence: Hyperlocal PM2.5 Prediction & Exposure Decision Support**

## Problem
City-wide Air Quality Indices (AQI) in Delhi-NCR provide a broad, generalized view of pollution but fail to capture hyperlocal variations caused by traffic patterns, local emissions, and microclimates. Students and residents need localized predictions to make actionable exposure decisions regarding their daily routes.

## Solution
An end-to-end Machine Learning system that models PM2.5 levels at highly specific geographic coordinates based on meteorology and historical lag data. The system translates raw ML predictions into practical exposure advice via a Streamlit Dashboard and a FastAPI backend.

## Dataset
- **Source**: `sachin-iitd/DelhiPollDataset` (Hugging Face mirror of the IITD-linked AirDelhi dataset).
- **Features**: Hourly PM values and meteorological data across Delhi-NCR.

## ML Approach
- **Feature Engineering**: Extensive, leakage-free spatial mapping (Haversine distance), cyclical temporal encodings, auto-regressive lags, and rolling averages.
- **Validation**: Strict 80/20 time-based split to simulate real-world forward-in-time forecasting and prevent look-ahead bias.
- **Benchmarking**: Compared Ridge Regression, Random Forest, XGBoost, and LightGBM against a naive Persistence Baseline.

## Deployment
- **FastAPI**: REST API for fast, programmatic inference. Auto-computes spatial and cyclical features on the fly.
- **Streamlit**: Interactive user dashboard featuring a "Route Exposure Advisor" built with Folium maps.
- **Docker**: Included `Dockerfile` for containerized cloud deployment.

## Results
**Ridge Regression** outperformed the complex tree-based models on the out-of-time test set. While gradient boosting methods overfit the early distribution blocks, the regularized linear model generalized well and reliably beat the persistence baseline.

## IIT Delhi Relevance
The user interface and default logic prioritize the IIT Delhi campus. The Route Exposure Advisor maps commutes specific to the community, such as traveling from IIT Delhi to Hauz Khas Metro or SDA Market.

## Limitations
- **Cold Start**: The model relies on recent PM2.5 history (`lag_1`). Predictions without immediate past context have higher error margins.
- **Static Externalities**: The model does not currently absorb live anomalies like sudden traffic jams or local fires.

## Future Work
- Integrate live API feeds (e.g., CPCB or OpenWeather) for real-time lag and weather data.
- Explore Temporal Convolutional Networks (TCNs) for longer-horizon forecasts.

## Links
- **GitHub Repository**: [Add Link Here]
- **Demo Video**: [Add Link Here]
