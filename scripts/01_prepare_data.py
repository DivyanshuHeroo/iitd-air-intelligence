import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.preprocessing import clean_data
from src.features import create_features

def main():
    print("Loading raw data...")
    raw_path = config.RAW_DATA_DIR / "raw_pollution_data.csv"
    if not raw_path.exists():
        print(f"Error: Raw data not found at {raw_path}")
        print("Please run scripts/00_load_data.py first.")
        sys.exit(1)
        
    df = pd.read_csv(raw_path)
    print("Cleaning data...")
    clean_df = clean_data(df)
    
    clean_path = config.PROCESSED_DATA_DIR / "clean_pollution_data.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved cleaned data to {clean_path}")

    print("Engineering features...")
    features_df = create_features(clean_df)
    
    features_path = config.PROCESSED_DATA_DIR / "features_pollution_data.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Saved featured data to {features_path}")
    print(f"Features shape: {features_df.shape}")

if __name__ == "__main__":
    main()
