# Manual Screenshot Guide for LinkedIn

Because Streamlit Community Cloud and local UI rendering requires a visual browser context, please manually capture the following screenshots and save them exactly as named below in this folder (`linkedin_assets/screenshots/`):

1. **`website_home.png`**
   - Open your deployed Streamlit app (or run `streamlit run app/streamlit_app.py`).
   - Navigate to the "Overview" section.
   - Capture the top of the page showing the project title "🌬️ IITD Air Intelligence".

2. **`iitd_prediction.png`**
   - Navigate to "Live PM2.5 Prediction" in the sidebar.
   - Select "IIT Delhi" as the location.
   - Click "Predict PM2.5".
   - Capture the section showing the sliders/inputs and the colorful "Predicted PM2.5" / "Category" result.

3. **`route_exposure_advisor.png`**
   - Navigate to "Route Exposure Advisor".
   - Select "IIT Delhi to Hauz Khas Metro" (or similar).
   - Click "Analyze Route Exposure".
   - Capture the map with the red route line and the average exposure metrics above it.

4. **`model_performance.png`**
   - Navigate to "Model Performance".
   - Capture the leaderboard table showing Ridge Regression as the best model.
   - Alternatively, you can use the generated image in `linkedin_assets/performance/model_results_table.png`.

5. **`fastapi_docs.png`**
   - Start the FastAPI backend locally: `uvicorn app.api:app --reload`
   - Open your browser to `http://127.0.0.1:8000/docs`.
   - Capture the Swagger UI showing the `/predict_pm25` POST endpoint.
