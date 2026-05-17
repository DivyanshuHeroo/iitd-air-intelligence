import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import folium
from streamlit_folium import st_folium
import os
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import load_model_and_features
from src.features import haversine_distance
from src import config

st.set_page_config(page_title="IITD Air Intelligence", layout="wide", page_icon="🌬️")

# Helper functions
def classify_pm25_category(pm25):
    if pm25 <= 30: return "Good"
    elif pm25 <= 60: return "Satisfactory"
    elif pm25 <= 90: return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else: return "Severe"

def get_category_interpretation(category):
    interpretations = {
        "Good": "Minimal impact.",
        "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
        "Moderate": "May cause breathing discomfort to people with lung disease.",
        "Poor": "May cause breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "May cause respiratory illness on prolonged exposure.",
        "Severe": "May cause respiratory impact even on healthy people."
    }
    return interpretations.get(category, "")

def build_model_input(lat, long, temperature, humidity, pressure, hour, pm2_5_lag_1, pm10_lag_1, pm2_5_rolling_3, pm10_rolling_3):
    dist = haversine_distance(lat, long, config.IITD_LAT, config.IITD_LONG)
    return {
        "lat": lat, "long": long, "temperature": temperature, "humidity": humidity, "pressure": pressure,
        "hour": hour, "day": 15, "dayofweek": 3, "month": 6, "is_weekend": 0,
        "distance_from_iitd_km": dist,
        "pm2_5_lag_1": pm2_5_lag_1, "pm2_5_lag_3": pm2_5_lag_1, "pm2_5_lag_6": pm2_5_lag_1,
        "pm10_lag_1": pm10_lag_1, "pm10_lag_3": pm10_lag_1, "pm10_lag_6": pm10_lag_1,
        "pm2_5_rolling_3": pm2_5_rolling_3, "pm2_5_rolling_6": pm2_5_rolling_3,
        "pm10_rolling_3": pm10_rolling_3, "pm10_rolling_6": pm10_rolling_3,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofweek_sin": 0.433, "dayofweek_cos": -0.900
    }

def predict_pm25(input_data):
    try:
        model, feature_cols = load_model_and_features()
    except Exception as e:
        raise Exception("Model not loaded. Please train the model first.")
        
    df = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    pred = model.predict(df)[0]
    return float(pred)

def calculate_route_exposure(points, locations):
    total_pm25 = 0
    predictions = []
    route_coords = []
    m = folium.Map(location=[config.IITD_LAT, config.IITD_LONG], zoom_start=14)
    
    for pt in points:
        lat, long = locations[pt]
        route_coords.append((lat, long))
        input_data = build_model_input(
            lat, long, 25.0, 50.0, 1010.0, 12, 40.0, 90.0, 40.0, 90.0
        )
        try:
            pred = predict_pm25(input_data)
        except:
            pred = 50.0  # fallback
            
        predictions.append(pred)
        total_pm25 += pred
        
        folium.Marker(
            [lat, long],
            popup=f"{pt}: {pred:.1f} PM2.5",
            tooltip=pt,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
    folium.PolyLine(route_coords, color="red", weight=2.5, opacity=0.8).add_to(m)
    avg_pm25 = total_pm25 / len(points)
    
    return avg_pm25, m

# Locations
LOCATIONS = {
    "IIT Delhi": (28.5450, 77.1926),
    "Hauz Khas": (28.5494, 77.2001),
    "Green Park": (28.5580, 77.2067),
    "AIIMS": (28.5672, 77.2100),
    "SDA Market": (28.5466, 77.1973),
    "Safdarjung": (28.5700, 77.2000),
    "Lajpat Nagar": (28.5677, 77.2433),
    "Custom Location": (28.5500, 77.2000)
}

# Sidebar
st.sidebar.title("IITD Air Intelligence")
st.sidebar.markdown("Hyperlocal PM2.5 Prediction for Delhi-NCR")

nav_selection = st.sidebar.radio("Navigation", [
    "Overview", 
    "Live PM2.5 Prediction", 
    "IIT Delhi Demo Locations", 
    "Route Exposure Advisor",
    "Model Performance", 
    "Feature Importance", 
    "About Dataset",
    "Limitations"
])

st.sidebar.markdown("---")
st.sidebar.caption("**Note:** Uses IITD-linked AirDelhi dataset via Hugging Face.")
st.sidebar.caption("**Disclaimer:** This is an educational ML demo and should not be used as an official health advisory.")

if nav_selection == "Overview":
    st.title("🌬️ IITD Air Intelligence")
    st.subheader("Hyperlocal PM2.5 Prediction for Delhi-NCR")
    st.markdown("""
    This project converts Delhi-NCR air pollution data into an end-to-end ML decision-support system. 
    Instead of only predicting pollution, it also demonstrates how predictions can support route-level exposure decisions around IIT Delhi and South Delhi.
    
    *Built autonomously using Python, FastAPI, Streamlit, and modern ML practices.*
    """)
    st.info("👈 Use the sidebar to navigate through the dashboard.")
    
    model_exists = os.path.exists(config.BEST_MODEL_PATH)
    if not model_exists:
        st.warning("⚠️ Trained model not found! Please run the training script locally to generate the model files.")

elif nav_selection == "Live PM2.5 Prediction":
    st.header("Live PM2.5 Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loc_name = st.selectbox("Location", list(LOCATIONS.keys()))
        if loc_name == "Custom Location":
            lat = st.number_input("Latitude", value=28.5500)
            long = st.number_input("Longitude", value=77.2000)
        else:
            lat, long = LOCATIONS[loc_name]
            st.text(f"Latitude: {lat}")
            st.text(f"Longitude: {long}")
            
        temperature = st.number_input("Temperature (C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        
    with col2:
        pressure = st.number_input("Pressure (mb)", value=1010.0)
        hour = st.slider("Hour of Day", 0, 23, 12)
        prev_pm25 = st.number_input("Previous PM2.5 (Lag 1)", value=40.0)
        prev_pm10 = st.number_input("Previous PM10 (Lag 1)", value=90.0)
        
    if st.button("Predict PM2.5", type="primary"):
        input_data = build_model_input(
            lat, long, temperature, humidity, pressure, hour, 
            prev_pm25, prev_pm10, prev_pm25, prev_pm10
        )
        
        try:
            pm25_val = predict_pm25(input_data)
            cat = classify_pm25_category(pm25_val)
            interpretation = get_category_interpretation(cat)
            
            color_map = {
                "Good": "green", "Satisfactory": "yellow", "Moderate": "orange",
                "Poor": "red", "Very Poor": "purple", "Severe": "maroon"
            }
            color = color_map.get(cat, "gray")
            
            st.markdown(f"### Predicted PM2.5: <span style='color:{color}'>{pm25_val:.2f} µg/m³</span>", unsafe_allow_html=True)
            st.markdown(f"### Category: <span style='color:{color}'>{cat}</span>", unsafe_allow_html=True)
            st.caption(interpretation)
        except Exception as e:
            st.error(f"Error predicting: {e}")

elif nav_selection == "IIT Delhi Demo Locations":
    st.header("IIT Delhi Demo Locations")
    m = folium.Map(location=[config.IITD_LAT, config.IITD_LONG], zoom_start=14)
    for loc, coords in LOCATIONS.items():
        if loc != "Custom Location":
            folium.Marker(
                [coords[0], coords[1]],
                popup=loc,
                tooltip=loc
            ).add_to(m)
    st_folium(m, width=800, height=400)

elif nav_selection == "Route Exposure Advisor":
    st.header("Route Exposure Advisor")
    
    routes = {
        "IIT Delhi to Hauz Khas Metro": ["IIT Delhi", "SDA Market", "Hauz Khas"],
        "IIT Delhi to Green Park": ["IIT Delhi", "Hauz Khas", "Green Park"],
        "IIT Delhi to AIIMS": ["IIT Delhi", "Green Park", "AIIMS"],
        "IIT Delhi to SDA Market": ["IIT Delhi", "SDA Market"]
    }
    
    route_name = st.selectbox("Compare Routes", list(routes.keys()))
    
    if st.button("Analyze Route Exposure", type="primary"):
        points = routes[route_name]
        avg_pm25, m = calculate_route_exposure(points, LOCATIONS)
        cat = classify_pm25_category(avg_pm25)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Average Exposure (PM2.5)", value=f"{avg_pm25:.2f}")
        with col2:
            st.metric(label="Exposure Category", value=cat)
            
        st_folium(m, width=800, height=400)

elif nav_selection == "Model Performance":
    st.header("Model Performance")
    if config.MODEL_RESULTS_PATH.exists():
        df_metrics = pd.read_csv(config.MODEL_RESULTS_PATH)
        st.dataframe(df_metrics.style.format(precision=2), use_container_width=True)
        
        ml_models = df_metrics[df_metrics["Model"] != "Persistence_Baseline"]
        if not ml_models.empty:
            best_model = ml_models.iloc[0]
            st.success(f"**Best Model**: {best_model['Model']}")
            
            imp_rmse = best_model.get("RMSE_improvement_percent", 0)
            imp_mae = best_model.get("MAE_improvement_percent", 0)
            st.info(f"The model improves over a simple persistence baseline by **{imp_rmse:.1f}%** in RMSE and **{imp_mae:.1f}%** in MAE.")
        
        fig_path = config.FIGURE_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Model Comparison (MAE)")
            
        class_path = config.METRICS_DIR / "category_classification_results.csv"
        if class_path.exists():
            df_class = pd.read_csv(class_path)
            st.write("### Secondary Task: PM2.5 Category Classification")
            
            summary_path = config.METRICS_DIR / "best_classification_summary.md"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    st.markdown(f.read())
            
            st.dataframe(df_class.style.format(precision=3), use_container_width=True)
            
            st.write("### Classification Confusion Matrices")
            col1, col2 = st.columns(2)
            with col1:
                cm_6_path = config.FIGURE_DIR / "category_confusion_matrix_6class.png"
                if cm_6_path.exists():
                    st.image(str(cm_6_path), caption="6-Class Confusion Matrix")
            with col2:
                cm_3_path = config.FIGURE_DIR / "category_confusion_matrix_3class.png"
                if cm_3_path.exists():
                    st.image(str(cm_3_path), caption="3-Class Confusion Matrix")
    else:
        st.warning("Model results not found. Run training script first.")

elif nav_selection == "Feature Importance":
    st.header("Feature Importance")
    feat_fig_path = config.FIGURE_DIR / "feature_importance.png"
    if feat_fig_path.exists():
        st.image(str(feat_fig_path), caption="Feature Importance")
    else:
        st.info("Feature importance plot not generated for this model (e.g., linear models without feature_importances_).")
        st.write("Top feature groups typically include:")
        st.write("- Lags (PM2.5 at t-1)")
        st.write("- Time of day (Hour, cyclical hour)")
        st.write("- Meteorological conditions")

elif nav_selection == "About Dataset":
    st.header("About Dataset")
    st.markdown("""
    - **Source**: `sachin-iitd/DelhiPollDataset` (Hugging Face)
    - **Description**: This is an IITD-linked / AirDelhi dataset containing hourly PM values and meteorology data across Delhi.
    - **Variables**: `dateTime`, `lat`, `long`, `pressure`, `temperature`, `humidity`, `pm1_0`, `pm2_5`, `pm10`.
    """)

elif nav_selection == "Limitations":
    st.header("Limitations")
    st.markdown("""
    - **Educational/Portfolio Use**: This model is for educational and portfolio demonstration purposes only. It is not an official health advisory.
    - **Historical Data**: The historical dataset may not reflect current real-time pollution anomalies.
    - **Pollution Spikes**: The model attempts the challenging task of predicting next-hour PM2.5 levels. While the model captures meaningful signal and improves over a baseline, it still struggles with sudden pollution spikes, which are difficult to predict from historical and meteorological features alone.
    - **External Factors**: Does not explicitly account for sudden events like traffic jams or stubble burning.
    """)
