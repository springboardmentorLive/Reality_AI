import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("rent_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
furnishing_encoder = joblib.load("furnishing_encoder.pkl")

st.title("🏠 House Rent Prediction")

# User inputs
location = st.selectbox("Select Location", location_encoder.classes_)
size = st.slider("House Size (sq ft)", 300, 3000, 800)
bhk = st.slider("Number of BHK", 1, 5, 2)
furnishing = st.selectbox("Furnishing Type", furnishing_encoder.classes_)

if st.button("Predict Rent"):
    loc_encoded = location_encoder.transform([location])[0]
    furn_encoded = furnishing_encoder.transform([furnishing])[0]

    input_data = np.array([[loc_encoded, size, bhk, furn_encoded]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Monthly Rent: ₹{int(prediction[0])}")
