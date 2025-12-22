import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df_encoded = df.copy()
    
    # Create label encoders
    location_encoder = LabelEncoder()
    furnished_encoder = LabelEncoder()
    
    # Fit and transform
    df_encoded['Location_Encoded'] = location_encoder.fit_transform(df['Location'])
    df_encoded['Furnished_Encoded'] = furnished_encoder.fit_transform(df['Furnished'])
    
    return df, df_encoded, location_encoder, furnished_encoder
    

def train_and_save_model(df_encoded, model, location_encoder, furnished_encoder, df):
    feature_columns = ['Location_Encoded', 'Size_sqft', 'BHK', 'Furnished_Encoded']
    X = df_encoded[feature_columns]
    y = df_encoded['Monthly_Rent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/rent_predictor_model.pkl')
    
    # Save encoders
    joblib.dump(location_encoder, 'models/location_encoder.pkl')
    joblib.dump(furnished_encoder, 'models/furnished_encoder.pkl')
    
    # Save unique values for UI dropdowns
    metadata = {
        'locations': df['Location'].unique().tolist(),
        'furnished_options': df['Furnished'].unique().tolist(),
        'min_size': int(df['Size_sqft'].min()),
        'max_size': int(df['Size_sqft'].max()),
        'min_bhk': int(df['BHK'].min()),
        'max_bhk': int(df['BHK'].max())
    }
    joblib.dump(metadata, 'models/metadata.pkl')


def main():
    # Load & Encode data
    df, df_encoded, location_encoder, furnished_encoder = load_and_preprocess_data('house_rent_data.csv')
    
    # Train model
    model = train_and_save_model(df_encoded, model, location_encoder, furnished_encoder, df)



if __name__ == "__main__":
    main()
