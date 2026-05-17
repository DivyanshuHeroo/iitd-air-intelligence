# LinkedIn Post Draft

*(Attach a nice screenshot of your Streamlit Route Advisor Map or a short video demo!)*

Hi network! 👋 As the internship season approaches at IIT Delhi, I wanted to share a recent machine learning project I built: **IITD Air Intelligence**. 

Air pollution in Delhi-NCR is highly localized, meaning your exposure can change drastically depending on your neighborhood or commute. To explore this, I built an end-to-end ML system that forecasts hyperlocal PM2.5 levels, translating raw predictions into actionable, route-level exposure insights around South Delhi.

Instead of just running a simple Kaggle notebook, I challenged myself to build the full ML engineering workflow from scratch:
✨ **Feature Engineering**: Extracted temporal, spatial, and meteorological features, along with auto-regressive lags and rolling windows. 
🛡️ **Validation**: Implemented a strict time-based split to completely avoid look-ahead bias and data leakage (a common trap in time-series tabular modeling).
📊 **Benchmarking**: Compared Ridge Regression, Random Forest, LightGBM, and XGBoost against a baseline.
🚀 **Deployment**: Packaged the best model into a beautiful interactive Streamlit dashboard and a production-ready FastAPI backend.

The dashboard even includes a "Route Exposure Advisor" allowing users to map their journey (e.g., IIT Delhi to Hauz Khas) and compare predicted PM2.5 exposures.

While it's currently an educational portfolio piece rather than an official health advisory, building out the data ingestion, validation, and deployment infrastructure was a fantastic engineering exercise. 

Check out the GitHub repo here: [Link to your GitHub Repo]

#MachineLearning #DataScience #IITDelhi #AirQuality #Python #FastAPI #Streamlit #MLProjects
