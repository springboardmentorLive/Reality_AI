import pandas as pd
import numpy as np

def clean_tabular_data(df):
    """
    Basic cleaning for tabular data.
    """
    # Fill missing values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill missing values with mode for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def preprocess_features(df):
    """
    Feature engineering (Mock).
    """
    # Example: Create a 'House Age' column if YearBuilt exists
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2024 - df['YearBuilt']
    return df
