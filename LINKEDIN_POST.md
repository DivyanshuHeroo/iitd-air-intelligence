# LinkedIn Posts for IITD Air Intelligence

## Option 1: Main Post (Recommended)
Built a new ML project: **IITD Air Intelligence** 🌫️

Over the past few days, I built an end-to-end Machine Learning project for hyperlocal PM2.5 prediction in Delhi-NCR.

Since I am at IIT Delhi, I wanted to build something locally relevant instead of another generic ML project. Delhi-NCR air pollution is a real-world problem, and pollution levels can vary significantly across time and location. 

I set up a challenging task: predicting *next-hour* PM2.5 levels. To solve this, I built a complete ML engineering pipeline:

✅ **Data cleaning and validation**
✅ **Advanced feature engineering:** temporal, spatial, meteorological, lag, and expanding-window features
✅ **Time-based cross-validation** to strictly avoid temporal leakage
✅ **Persistence baseline comparison** (My best Ridge model improved RMSE by 19.6% over baseline)
✅ **Secondary Classification Task:** Predicting PM2.5 categories (Good, Satisfactory, Moderate, etc.) with ~62% performance
✅ **Streamlit dashboard** for interactive predictions
✅ **FastAPI backend** for inference
✅ **Route exposure advisor** comparing paths around IIT Delhi and South Delhi

The part I enjoyed most was converting a prediction model into a decision-support tool. Instead of only asking, “What is the predicted PM2.5?”, the dashboard also compares routes and estimates which one may have lower predicted exposure.

This project helped me practice the full ML workflow: data processing, feature engineering, evaluation, deployment, testing, documentation, and product thinking. 

*(Note: This is an educational ML project and not an official health advisory, but it was a great way to connect machine learning with a real Delhi-NCR problem).*

**Live Demo**: <add live demo link>
**GitHub**: <add GitHub repo link>

#MachineLearning #DataScience #IITDelhi #AirQuality #Python #FastAPI #Streamlit #MLProjects #Delhi #OpenSource


## Option 2: Short Post
I recently built **IITD Air Intelligence** 🌫️, an end-to-end Machine Learning project for hyperlocal PM2.5 prediction in Delhi-NCR!

Since I'm at IIT Delhi, I wanted to build a project connected to a real local issue. I engineered temporal, spatial, meteorological, lag, and expanding-window features to predict next-hour PM2.5 levels. 

To ensure the project was robust, I:
🔹 Used strictly time-based validation (no data leakage)
🔹 Benchmarked multiple models against a persistence baseline (achieving a ~20% RMSE improvement)
🔹 Deployed the best model via a Streamlit dashboard and FastAPI backend
🔹 Built a "Route Exposure Advisor" to compare predicted pollution on different paths around South Delhi

It was a great experience practicing the full ML engineering lifecycle from data to deployment! *(Educational demo only).*

Check out the live dashboard here: <add live demo link>
Code is available here: <add GitHub repo link>

#MachineLearning #DataScience #IITDelhi #Python #Streamlit


## Option 3: Technical Post
Just deployed **IITD Air Intelligence** 🌫️, a hyperlocal PM2.5 prediction and routing engine for Delhi-NCR. 

A lot of ML portfolio projects stop at a Jupyter notebook. For this one, I wanted to focus on robust evaluation and deployment. 

**Technical Highlights:**
🛠 **Target**: Next-hour PM2.5 prediction (a notoriously noisy, autoregressive time-series task).
🛠 **Features**: Engineered Haversine distances, cyclical time (sine/cosine), expanding means, and rolling statistics.
🛠 **Validation**: Strictly time-based (no random splits) to prevent temporal leakage. 
🛠 **Evaluation**: Benchmarked Random Forest, LightGBM, HistGradientBoosting, and Ridge. The best model (Ridge Regression) improved RMSE by 19.6% over a persistence baseline. 
🛠 **Deployment**: FastAPI handles the inference backend (with automatic feature computation), and Streamlit serves the UI, which includes a Route Exposure Advisor mapping tool built with Folium.

It was a fantastic exercise in avoiding data leakage and treating an ML model like a software product. 

**Live Demo**: <add live demo link>
**GitHub**: <add GitHub repo link>

#MachineLearning #MLOps #DataScience #IITDelhi #FastAPI #Streamlit #Python
