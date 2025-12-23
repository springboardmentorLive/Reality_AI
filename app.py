import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt  # Fixed typo: 'pit' changed to 'plt'

st.set_page_config(page_title="RealityAI", layout="wide")
st.title("RealtyAI: Smart Real Estate Insight Platform")

# Fix 1: Added missing '=' assignment operator
data = pd.read_csv("data/real_estate_prices.csv")

# Fix 2: Corrected 'st. (function)' syntax to 'st.subheader' and 'st.line_chart'
st.subheader("Price Trend") 
st.line_chart(data.set_index("year")["price"])

st.subheader("Property Data")
st.dataframe(data)

st.subheader("Satellite Image Segmentation")
img = cv2.imread("images/imageshouse.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fix 3: cv2.threshold returns a tuple (ret, thresh); use '_' for the first value
_, seg = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

fig, ax = plt.subplots()
ax.imshow(seg, cmap="gray")
ax.set_title("Segmented Image")  # Fixed comma to dot: 'ax.set_title'
ax.axis("off")                  # Fixed 'sx' to 'ax'

st.pyplot(fig)
st.success("AI Analysis Completed Successfully") # Fixed 'st, success' to 'st.success'