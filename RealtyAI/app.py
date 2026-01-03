import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from buy_decision_prediction import predict_buy_decision

# Page configuration
st.set_page_config(
    page_title="LiveMeetingAI",
    layout="wide"
)

# Title
st.title("RealtyAI – Smart Real Estate Insight Platform")

# Load dataset
data = pd.read_csv("data/real_estate_prices.csv")

# Line chart
st.subheader("Property Price Trend")
st.line_chart(data.set_index("year")["price"])

# Data table
st.subheader("Property Data")
st.dataframe(data)

# Image segmentation section
st.subheader("Satellite Image Segmentation")

# Read image
img = cv2.imread("images/house.jpg")

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, seg = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY
    )

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(seg, cmap="gray")
    ax.set_title("Segmented Image")
    ax.axis("off")

    st.pyplot(fig)
else:
    st.error("Image not found. Please check the image path.")

st.subheader("Smart Buy / Do Not Buy Recommendation")

price = st.number_input("Property Price", min_value=1000000, step=500000)
area = st.number_input("Property Area (sq ft)", min_value=500, step=100)
year = st.number_input("Construction Year", min_value=2000, max_value=2025)

if st.button("Get Recommendation"):
    decision = predict_buy_decision(price, area, year)

    if decision == "Buy":
        st.success("Recommendation: BUY ✅")
    else:
        st.error("Recommendation: DO NOT BUY ❌")

# Success message
st.success("AI Analysis Completed Successfully")