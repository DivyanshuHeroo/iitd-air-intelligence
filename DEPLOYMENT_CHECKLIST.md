# Deployment Checklist

## Before Deployment
- [x] `requirements.txt` exists and contains only necessary packages.
- [x] `app/streamlit_app.py` runs locally.
- [x] Model (`models/best_pm25_model.pkl`) exists and is committed (it's small enough).
- [x] `models/feature_columns.json` exists and is committed.
- [x] No raw full data committed.
- [x] No secrets committed.
- [x] GitHub repo is public.
- [x] `README.md` has a live demo section.

## Streamlit Cloud Settings
- **Repository**: `DivyanshuHeroo/iitd-air-intelligence`
- **Branch**: `main`
- **Main file path**: `app/streamlit_app.py`
- **Python dependencies**: Automatically installed via `requirements.txt`

## Post-Deployment Checks
- [ ] App opens on the public URL.
- [ ] Prediction works without crashing.
- [ ] Route advisor maps correctly.
- [ ] `README.md` live demo link updated.
- [ ] LinkedIn post includes live link.
