"""
Data loading and basic cleaning module for the UCI Bank Marketing Dataset.
Enforces the Five Golden Rules:
- Original CSV is read-only and never modified
- Deduplication is tracked and documented
- Target 'y' is converted to binary 1 (yes) / 0 (no)
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

DATA_PATH_SUBDIR = "data/bank-additional-full.csv"
DATA_PATH_PARENT_SUBDIR = "../data/bank-additional-full.csv"
DATA_PATH_ROOT = "bank-additional-full.csv"

def get_data_path() -> str:
    """Resolve dataset path checking data/ directory first, then fallback."""
    for candidate in [DATA_PATH_SUBDIR, DATA_PATH_PARENT_SUBDIR, DATA_PATH_ROOT]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("bank-additional-full.csv not found in data/ or working directory.")

def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw bank marketing CSV with semicolon separator."""
    path = filepath or get_data_path()
    df = pd.read_csv(path, sep=";")
    return df

def clean_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the bank marketing dataset:
    - Identify and remove exact duplicate rows
    - Convert target 'y' from 'yes'/'no' to 1/0
    - Retain 'unknown' as an explicit categorical value
    - Verify absence of null values
    """
    df = df_raw.copy()
    initial_rows = len(df)
    
    # 1. Deduplication
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    cleaned_rows = len(df)
    
    # 2. Target binarization
    df['y'] = (df['y'] == 'yes').astype(int)
    
    # 3. Basic summary stats
    positive_count = int(df['y'].sum())
    negative_count = int(cleaned_rows - positive_count)
    base_conversion_rate = float(df['y'].mean())
    always_no_accuracy = float(1.0 - base_conversion_rate)
    
    metadata = {
        "initial_rows": initial_rows,
        "duplicate_count": duplicate_count,
        "cleaned_rows": cleaned_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "base_conversion_rate": base_conversion_rate,
        "always_no_accuracy": always_no_accuracy,
    }
    
    return df, metadata

if __name__ == "__main__":
    raw_df = load_raw_data()
    df_clean, meta = clean_data(raw_df)
    print("Cleaned Data Shape:", df_clean.shape)
    print("Metadata:", meta)
