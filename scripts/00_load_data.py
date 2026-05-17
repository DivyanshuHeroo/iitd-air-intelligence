import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
import pandas as pd
from src import config

def main():
    print("Loading dataset sachin-iitd/DelhiPollDataset from Hugging Face...")
    try:
        dataset = load_dataset("sachin-iitd/DelhiPollDataset")
        print("Available splits:", dataset.keys())
        
        # Taking 'train' split if available, otherwise taking the first split
        split_name = 'train' if 'train' in dataset.keys() else list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
        
        print("Dataset columns:", df.columns.tolist())
        
        raw_path = config.RAW_DATA_DIR / "raw_pollution_data.csv"
        df.to_csv(raw_path, index=False)
        print(f"Saved raw data to {raw_path}")
        
        # Save a smaller sample for app testing
        sample_path = config.SAMPLE_DATA_DIR / "sample_pollution_data.csv"
        df.head(1000).to_csv(sample_path, index=False)
        print(f"Saved sample data to {sample_path}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please check your internet connection or the dataset name.")
        sys.exit(1)

if __name__ == "__main__":
    main()
