import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# Page Config
st.set_page_config(page_title="RealityAI", layout="wide")
st.title("RealtyAI: Smart Real Estate Insight Platform")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/real_estate_prices.csv")
    df.columns = df.columns.str.strip()
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error("Error: 'data/real_estate_prices.csv' not found.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Market Trends", "Price Predictor", "Property Analysis"])

# --- TAB 1: MARKET TRENDS & FORECASTING ---
with tab1:
    st.header("Real Estate Market Trends")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical Price Trend")
        st.line_chart(data.set_index("year")["price"])
        
    with col2:
        st.subheader("Future Forecast")
        # Time Series Forecasting Logic
        X = data[['year']]
        y = data['price']
        
        forecaster = LinearRegression()
        forecaster.fit(X, y)
        
        future_years = [[2025], [2026], [2027], [2028], [2029], [2030]]
        future_prices = forecaster.predict(future_years)
        
        forecast_df = pd.DataFrame({
            "Year": [y[0] for y in future_years],
            "Predicted Price": future_prices.astype(int)
        })
        
        st.dataframe(forecast_df)
        
    st.info("Forecast based on simple Linear Regression of historical data.")

# --- TAB 2: PRICE PREDICTOR ---
with tab2:
    st.header("Smart Price Predictor")
    
    # Train Model
    X_price = data[['area', 'rooms']]
    y_price = data['price']
    
    price_model = LinearRegression()
    price_model.fit(X_price, y_price)
    
    col1, col2 = st.columns(2)
    
    with col1:
        area_input = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1600)
        rooms_input = st.number_input("Number of Rooms", min_value=1, max_value=10, value=4)
        
    with col2:
        st.markdown("### Prediction")
        if st.button("Estimate Price"):
            input_data = [[area_input, rooms_input]]
            prediction = price_model.predict(input_data)[0]
            st.metric(label="Estimated Property Price", value=f"${int(prediction):,}")

# --- TAB 3: PROPERTY ANALYSIS ---
with tab3:
    st.header("Property Condition Analysis")
    
    st.write("Upload a property image or use the default for segmentation and condition classification.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        # Load default image if it exists
        try:
            img = cv2.imread("data/images/house.jpg")
            if img is None:
                st.warning("Default image not found. Please upload an image.")
        except Exception:
            img = None
            
    if img is not None:
        col1, col2, col3 = st.columns(3)
        
        # Original Image
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        # Segmentation Logic
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        with col2:
            st.subheader("Segmentation")
            st.image(seg, channels="GRAY", use_container_width=True)
            
        # Classification Logic
        # Train dummy classifier as per property_classification.py
        X_train = [[100], [150], [200]] 
        y_train = ["Old", "Average", "New"]
        classifier = SVC()
        classifier.fit(X_train, y_train)
        
        avg_brightness = int(np.mean(img))
        condition = classifier.predict([[avg_brightness]])[0]
        
        with col3:
            st.subheader("AI Assessment")
            st.metric("Detected Condition", condition)
            st.caption(f"Average Brightness: {avg_brightness}")
            
    else:
        st.info("Waiting for image...")