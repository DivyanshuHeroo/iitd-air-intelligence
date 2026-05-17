# IITD Air Intelligence — Recruiter Summary

## One-Line Description
IITD Air Intelligence is an end-to-end ML system for hyperlocal PM2.5 prediction in Delhi-NCR, with a Streamlit dashboard, FastAPI backend, and route exposure advisor focused on IIT Delhi and South Delhi.

## Why it Matters
Air pollution is highly localized; city-wide AQI does not reflect individual exposure. This project models PM2.5 at specific GPS coordinates, translating machine learning predictions into practical, route-level decision support for the local community.

## Technical Skills Demonstrated
- Python (Pandas, NumPy, Scikit-Learn)
- Tree-based & Linear ML Models (Random Forest, XGBoost, LightGBM, Ridge)
- Advanced Feature Engineering (Time cyclical, Spatial, Auto-regressive Lags, Rolling Windows)
- Time-series Validation
- Deployment (FastAPI, Streamlit, Docker)
- Unit Testing (`pytest`)
- Professional MLOps and GitHub documentation

## ML Skills Demonstrated
- Baseline comparison (Persistence baseline)
- Time-aware validation to prevent look-ahead bias
- Leakage-safe lag and rolling feature computation
- Model evaluation and residual analysis
- Feature importance visualization

## Engineering Skills Demonstrated
- Modular, well-architected project structure
- Automated training and evaluation pipelines
- Interactive UI Dashboard
- REST API service
- Containerization and deployment readiness

## IIT Delhi Relevance
The project utilizes the IITD-linked AirDelhi dataset, natively tracking coordinates around the IIT Delhi campus (e.g., Hauz Khas, SDA Market) to simulate real-world utility for students and faculty navigating the area.

## Best Model and Metrics
**Ridge Regression** generalized best on the strict time-split, avoiding the temporal overfitting seen in gradient boosting models.
- **MAE**: 40.75
- **RMSE**: 58.29
- **R²**: 0.498

## Demo Features
- Live PM2.5 Prediction API
- South Delhi Route Exposure Advisor (Folium Map)
- Visual Model Leaderboard and Evaluation

## Resume Bullet Highlight
Built an end-to-end ML system for hyperlocal PM2.5 prediction in Delhi-NCR using an IITD-linked air pollution dataset; engineered temporal, spatial, meteorological, lag, and rolling-window features, benchmarked models with time-based validation, and deployed the best model via Streamlit and FastAPI.

## Interview Pitch
“I built IITD Air Intelligence, an end-to-end machine learning system for hyperlocal PM2.5 prediction in Delhi-NCR using an IITD-linked air pollution dataset. I engineered spatial, temporal, meteorological, lag, and rolling-window features, used time-based validation to avoid leakage, benchmarked multiple models against a persistence baseline, and deployed the best model through a Streamlit dashboard and FastAPI service. I also added a route exposure advisor around IIT Delhi and South Delhi, so the project goes beyond prediction and turns ML output into a practical decision-support tool.”
