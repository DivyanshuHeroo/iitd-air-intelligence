import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create model results table image
df = pd.read_csv('reports/metrics/model_results.csv')
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

# Format dataframe for display
display_df = df.round(2)
display_df = display_df.rename(columns={'Model': 'Model', 'MAE': 'MAE (µg/m³)', 'RMSE': 'RMSE (µg/m³)', 'R2': 'R² Score', 'MAPE': 'MAPE (%)'})

table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 2.0)

plt.title('Model Performance Leaderboard', fontsize=20, pad=20, fontweight='bold')
plt.savefig('linkedin_assets/performance/model_results_table.png', bbox_inches='tight', dpi=300)
plt.close()

print("Model results table generated!")
