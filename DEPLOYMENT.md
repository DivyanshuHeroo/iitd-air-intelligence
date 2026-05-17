# Deployment Guide

This project can be easily deployed via Docker, Streamlit Community Cloud, or standard PaaS providers. **No API keys are required to deploy.**

## 1. Local Development
For testing out the platform locally:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 2. Streamlit Community Cloud (Primary)
Deploying the dashboard via [Streamlit Cloud](https://streamlit.io/cloud) is straightforward:
1. Push your repository to GitHub.
2. Log into Streamlit Community Cloud and click **New app**.
3. Select this repository and set the **Main file path** to `app/streamlit_app.py`.
4. Click **Deploy!** Streamlit will automatically install dependencies from `requirements.txt`.
*(Note: Ensure your trained model `models/best_pm25_model.pkl` is committed, or use a sample dataset/trigger a training run during initialization).*

## 3. Hugging Face Spaces (Fallback)
If Streamlit Community Cloud is unavailable, you can host the ML demo app on Hugging Face Spaces:
- **SDK**: Streamlit
- **App file**: `app/streamlit_app.py`
- **Requirements**: Use root `requirements.txt`

## 4. Render / Railway for FastAPI (Optional)
If you want to host the FastAPI inference service:
1. Connect your GitHub repository to Render or Railway.
2. Set the Build Command: `pip install -r requirements.txt`
3. Set the Start Command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
