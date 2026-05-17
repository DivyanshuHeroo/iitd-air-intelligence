# Demo Video Scripts

Use these scripts while recording a screen-capture demo of your project for LinkedIn, GitHub, or an interview presentation.

## 30-Second Version (Elevator Pitch)
"Hi! I'm Divyanshu, and this is IITD Air Intelligence. It's a complete machine learning system that predicts hyperlocal PM2.5 levels around IIT Delhi. I engineered leakage-free lag features, evaluated multiple models using strict time-based splits, and deployed the best one here in Streamlit. It includes a live prediction form and a Route Exposure Advisor that maps out the cleanest commutes around South Delhi. It's fully backed by a FastAPI inference service!"

## 60-Second Version (Standard Pitch)
"Hi, this is IITD Air Intelligence, an end-to-end machine learning project for hyperlocal PM2.5 prediction in Delhi-NCR. I built this around an IITD-linked air pollution dataset to make the project locally relevant and technically meaningful.

The pipeline starts with data loading and cleaning, then engineers temporal, spatial, meteorological, lag, and rolling-window features. I used a time-based train/test split to avoid leakage and benchmarked multiple models against a persistence baseline.

The best model is deployed in two ways: a Streamlit dashboard for interactive predictions and a FastAPI service for programmatic inference. The dashboard also includes an IIT Delhi and South Delhi route exposure advisor, which compares routes based on predicted PM2.5 exposure.

This project demonstrates not just model training, but a complete ML engineering workflow: data processing, evaluation, deployment, testing, documentation, and practical decision support."

## 2-Minute Version (Deep Dive)
"Hi everyone, I want to walk you through IITD Air Intelligence, an end-to-end ML project I built for hyperlocal PM2.5 prediction in Delhi-NCR. 

*[(Show the Streamlit Dashboard Overview)]*
I started with an IITD-linked air pollution dataset because city-wide AQI isn't enough—we need hyperlocal data for real decision-making. 

*[(Show the terminal or GitHub repo architecture)]*
Instead of just a simple Kaggle notebook, I built a modular software pipeline. The data preparation scripts engineer over 25 features: including spatial distances, cyclical time encodings, and auto-regressive lags. A major engineering focus was preventing data leakage—I used strict shift-before-rolling operations and a strict 80/20 time-based train-test split.

*[(Show Model Results Tab)]*
I evaluated Ridge Regression, Random Forest, LightGBM, and XGBoost against a naive persistence baseline. Interestingly, Ridge Regression generalized the best, as the complex tree models overfit the early temporal patterns and struggled on unseen future blocks.

*[(Show FastAPI Docs)]*
The best model is deployed via a Dockerized FastAPI service that handles real-time inference and auto-computes necessary spatial features on the fly. 

*[(Show Streamlit Route Advisor)]*
Finally, I built this Streamlit dashboard. The coolest feature is the Route Exposure Advisor. It takes standard routes around South Delhi—like IIT Delhi to AIIMS—queries the model at different coordinate points, and visualizes the average exposure on a Folium map. It turns ML predictions into an actual decision-support tool."
