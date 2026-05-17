# Interview Q&A Guide

Use these questions and answers to prepare for data science / ML engineering interviews during the IIT Delhi internship season.

### 1. Why did you choose this project?
"I wanted to build an end-to-end ML system that went beyond just training a model in a notebook. Air pollution is a severe, real-world issue in Delhi, so I wanted to create a project that takes raw data and turns it into a practical decision-support tool (like a Route Exposure Advisor) while demonstrating my software engineering and MLOps skills."

### 2. Why is this relevant to IIT Delhi?
"I specifically used the IITD-linked AirDelhi dataset and configured the deployment to focus on the IIT Delhi campus and surrounding South Delhi areas. The route advisor actively demonstrates how students could use ML to minimize exposure when traveling to places like Hauz Khas or SDA Market."

### 3. What dataset did you use?
"I used the `sachin-iitd/DelhiPollDataset` from Hugging Face, which contains hourly meteorological variables and particulate matter readings across various coordinates in Delhi-NCR."

### 4. What was the target variable?
"The target variable was continuous PM2.5 concentration, measured in µg/m³."

### 5. What features did you engineer?
"I engineered over 25 features: 
- Spatial features: Haversine distance from IIT Delhi.
- Temporal features: Hour, day, month, and cyclical sine/cosine encodings to capture the continuous nature of time.
- Lags & Rolling: Auto-regressive features like 1-hour, 3-hour, and 6-hour lags for PM2.5 and PM10, as well as rolling averages."

### 6. How did you avoid data leakage?
"Data leakage is the biggest trap in time-series prediction. First, I grouped by location and strictly used a `.shift()` operation *before* calculating any rolling windows, ensuring the model never saw current or future pollution data. Second, I used a strict time-based split rather than random K-Fold cross-validation."

### 7. Why did you use a time-based split?
"Because random splits can leak future information in time-dependent pollution data. If I randomly split the data, the model could use data from Wednesday to predict Tuesday, which is impossible in the real world. A time-based split (training on the past 80%, testing on the future 20%) proves the model can actually forecast the future."

### 8. What baseline did you use?
"I used a Persistence Baseline—where the prediction for time *t* is simply the exact value from time *t-1*. Because PM2.5 is highly auto-correlated, this is actually a very strong baseline to beat."

### 9. Which model performed best?
"Interestingly, Ridge Regression performed the best on the out-of-time test set. While complex tree models like XGBoost and LightGBM achieved lower error on the training set, they overfit the early temporal patterns and struggled to generalize on the unseen future block. Ridge was the most robust and reliably beat the persistence baseline."

### 10. Why not just use deep learning?
"Deep learning (like LSTMs or Transformers) requires massive amounts of sequential data and heavy compute. For tabular forecasting with specific geographical coordinates, tree-based models and regularized linear models are much faster to train, easier to interpret, and serve as excellent, highly-performant baselines. The dashboard makes the project usable, not just a notebook."

### 11. What does the route exposure advisor do?
"It takes a route (like IIT Delhi to AIIMS), breaks it into coordinate points, queries the FastAPI model for predictions at each point, and plots the route on an interactive map with the calculated average PM2.5 exposure. It turns a standard ML output into a user decision."

### 12. What are the limitations?
"The model is educational and not an official health advisory. Technically, it suffers from a 'cold start' problem—because it relies heavily on lag features, predicting pollution without knowing the immediate past (lag_1) results in lower accuracy. It also doesn't account for sudden real-time events like traffic jams or fires."

### 13. How would you improve this in the future?
"I would integrate live traffic density APIs (like Google Maps) and satellite Aerosol Optical Depth (AOD) data to provide external context. I would also experiment with Temporal Convolutional Networks (TCNs)."

### 14. How would you make it real-time?
"I would set up an automated cron job or Airflow pipeline to ingest live CPCB or OpenWeather API data into a feature store every hour, ensuring the model always has the latest lag features available for inference."

### 15. What engineering practices did you follow?
"I modularized the codebase (separating configs, preprocessing, features, and training), added unit tests with `pytest`, centralized configuration variables to avoid hardcoded paths, and built RESTful endpoints using FastAPI."

### 16. How would you deploy it?
"I've written a Dockerfile for the FastAPI backend, which can be deployed to AWS ECS, Render, or Railway. The Streamlit dashboard can be hosted easily on Streamlit Community Cloud."

### 17. What did you learn?
"I learned that complex models don't always win. The tree-based models overfit the time-series split, teaching me the critical importance of rigorous, leakage-free validation strategies over just throwing XGBoost at a dataset."

### 18. How is this different from a Kaggle notebook?
"A Kaggle notebook usually stops at `model.predict()`. This project puts the model in a FastAPI service, serves it to a Streamlit frontend, containerizes the environment, tests the logic, and focuses heavily on the end-user product (the Route Advisor)."
