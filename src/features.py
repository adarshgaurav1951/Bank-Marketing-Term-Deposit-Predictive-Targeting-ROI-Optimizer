"""
Feature Engineering and Preprocessing Pipeline for Bank Marketing.
Strictly enforces Golden Rule #1 (Drop duration from production feature set)
and Golden Rule #3 (Fit preprocessing only on training data).
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = [
    'age',
    'campaign',
    'pdays_cleaned',
    'previous',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed',
    'previously_contacted'
]

CATEGORICAL_FEATURES = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'day_of_week',
    'poutcome',
    'age_group'
]

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer leak-free features:
    - previously_contacted (1 if pdays != 999 else 0)
    - pdays_cleaned (0 if pdays == 999 else pdays)
    - age_group (<30, 30-39, 40-49, 50-59, 60+)
    - drops 'duration' if present to prevent lookahead leakage
    """
    def __init__(self, drop_duration: bool = True):
        self.drop_duration = drop_duration

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. pdays transformation
        if 'pdays' in df.columns:
            df['previously_contacted'] = (df['pdays'] != 999).astype(int)
            df['pdays_cleaned'] = np.where(df['pdays'] == 999, 0, df['pdays'])
        elif 'pdays_cleaned' not in df.columns:
            df['previously_contacted'] = 0
            df['pdays_cleaned'] = 0

        # 2. Age group discretization
        if 'age' in df.columns:
            bins = [0, 29, 39, 49, 59, 120]
            labels = ['<30', '30-39', '40-49', '50-59', '60+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True).astype(str)
        elif 'age_group' not in df.columns:
            df['age_group'] = '30-39'

        # 3. Ensure 'unknown' is clean string
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')

        # 4. Strict leakage enforcement: Drop duration for production
        if self.drop_duration and 'duration' in df.columns:
            df = df.drop(columns=['duration'])

        return df

def get_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_feature_data(df: pd.DataFrame, include_duration_benchmark: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract X and y from cleaned dataframe.
    If include_duration_benchmark is False (default), duration is strictly removed.
    """
    y = df['y']
    cols_to_drop = ['y']
    if not include_duration_benchmark and 'duration' in df.columns:
        cols_to_drop.append('duration')
        
    X = df.drop(columns=cols_to_drop)
    return X, y
