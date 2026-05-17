# LinkedIn Posts for IITD Air Intelligence

## Option 1: Main Post (Recommended)
Built a new ML project: **IITD Air Intelligence** 🌫️

Over the past few days, I built IITD Air Intelligence — an end-to-end Machine Learning project for hyperlocal PM2.5 prediction and air-quality category interpretation in Delhi-NCR.

Since I’m at IIT Delhi, I wanted to build something locally relevant instead of another generic ML project. Delhi-NCR air pollution is a real-world problem, and pollution levels can vary significantly across time and location.

For this project, I built a complete ML engineering pipeline:

✅ Data cleaning and validation
✅ Temporal, spatial, meteorological, lag, and rolling-window features
✅ Next-hour PM2.5 category prediction
✅ Time-based train/test split to avoid leakage
✅ Baseline comparison and model benchmarking
✅ Streamlit dashboard for interactive predictions
✅ FastAPI backend for inference
✅ Route exposure advisor around IIT Delhi and South Delhi

One important learning: predicting the exact fine-grained PM2.5 category is challenging because air-quality categories are threshold-based and PM2.5 can spike suddenly.

So I evaluated the model at multiple levels:

📌 6-class exact category prediction: ~59.3% accuracy
📌 Adjacent-category accuracy: ~95.6%
📌 3-class severity prediction: ~84.9% accuracy

The adjacent-category result was especially interesting: even when the model missed the exact category, it was usually off by only one neighboring severity level.

The part I enjoyed most was turning ML predictions into a decision-support tool. Instead of only asking, “What is the predicted PM2.5?”, the dashboard also compares routes and estimates which one may have lower predicted exposure.

This project helped me practice the full ML workflow: data processing, feature engineering, leakage-safe validation, evaluation, deployment, testing, documentation, and product thinking.

This is an educational ML project and not an official health advisory.

**Live Demo**: <add live demo link>
**GitHub**: <add GitHub repo link>

#MachineLearning #DataScience #IITDelhi #AirQuality #Python #FastAPI #Streamlit #MLProjects #Delhi #OpenSource


## Option 2: Short Post
I recently built **IITD Air Intelligence** 🌫️, an end-to-end Machine Learning project for hyperlocal PM2.5 prediction in Delhi-NCR!

Since I'm at IIT Delhi, I wanted to build a project connected to a real local issue. I engineered temporal, spatial, meteorological, lag, and rolling-window features to predict next-hour PM2.5 categories. 

To ensure the project was robust, I:
🔹 Used strictly time-based validation (no data leakage)
🔹 Benchmarked multiple models, achieving ~84.9% 3-class severity accuracy and ~95.6% adjacent-category accuracy
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
🛠 **Target**: Next-hour PM2.5 category prediction (a notoriously noisy, autoregressive time-series task).
🛠 **Features**: Engineered Haversine distances, cyclical time (sine/cosine), expanding means, and rolling statistics.
🛠 **Validation**: Strictly time-based (no random splits) to prevent temporal leakage. 
🛠 **Evaluation**: Predicting exact 6-class fine-grained categories is notoriously difficult due to continuous thresholds (achieving ~59.3% accuracy). However, the model achieved ~95.6% adjacent-category accuracy and ~84.9% accuracy on a broader 3-class severity task.
🛠 **Deployment**: FastAPI handles the inference backend (with automatic feature computation), and Streamlit serves the UI, which includes a Route Exposure Advisor mapping tool built with Folium.

It was a fantastic exercise in avoiding data leakage and treating an ML model like a software product. *(Educational demo only).*

**Live Demo**: <add live demo link>
**GitHub**: <add GitHub repo link>

#MachineLearning #MLOps #DataScience #IITDelhi #FastAPI #Streamlit #Python
