import nbformat as nbf
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR

notebooks_dir = ROOT_DIR / "notebooks"
notebooks_dir.mkdir(exist_ok=True)

# 01_eda.ipynb
nb_eda = nbf.v4.new_notebook()
nb_eda.cells = [
    nbf.v4.new_markdown_cell("# Exploratory Data Analysis"),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from src import config

# Load data
df = pd.read_csv(config.PROCESSED_DATA_DIR / 'clean_pollution_data.csv')
df['dateTime'] = pd.to_datetime(df['dateTime'])
df.head()"""),
    nbf.v4.new_code_cell("""df.info()"""),
    nbf.v4.new_code_cell("""# PM2.5 Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['pm2_5'], bins=50)
plt.title('PM2.5 Distribution')
plt.show()"""),
    nbf.v4.new_code_cell("""# Daily Trend
df_daily = df.groupby(df['dateTime'].dt.date)['pm2_5'].mean().reset_index()
plt.figure(figsize=(12,6))
plt.plot(df_daily['dateTime'], df_daily['pm2_5'])
plt.xticks(rotation=45)
plt.title('Daily PM2.5 Trend')
plt.show()""")
]
nbf.write(nb_eda, str(notebooks_dir / "01_eda.ipynb"))

# 02_model_training.ipynb
nb_train = nbf.v4.new_notebook()
nb_train.cells = [
    nbf.v4.new_markdown_cell("# Model Training"),
    nbf.v4.new_code_cell("""import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from src import config
from src.train import train_and_evaluate

df = pd.read_csv(config.PROCESSED_DATA_DIR / 'features_pollution_data.csv')
df.head()"""),
    nbf.v4.new_code_cell("""feature_cols = [
    "hour", "day", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
    "lat", "long", "distance_from_iitd_km",
    "pressure", "temperature", "humidity",
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6",
    "pm2_5_rolling_3", "pm2_5_rolling_6"
]
feature_cols = [c for c in feature_cols if c in df.columns]

metrics, models, test_preds = train_and_evaluate(df, 'pm2_5', feature_cols)
metrics""")
]
nbf.write(nb_train, str(notebooks_dir / "02_model_training.ipynb"))

# 03_results_analysis.ipynb
nb_results = nbf.v4.new_notebook()
nb_results.cells = [
    nbf.v4.new_markdown_cell("# Results Analysis"),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from src import config

metrics_df = pd.read_csv(config.METRICS_DIR / 'model_results.csv')
metrics_df"""),
    nbf.v4.new_code_cell("""plt.figure(figsize=(10,6))
sns.barplot(x='MAE', y='Model', data=metrics_df)
plt.title('Model MAE Comparison')
plt.show()""")
]
nbf.write(nb_results, str(notebooks_dir / "03_results_analysis.ipynb"))

print("Notebooks created successfully.")
