import streamlit as st  
import pandas as pd
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="house_price_prediction", layout="wide")

st.title("House Price Prediction-using Reality_AI")

data = pd.read_csv('data\house_rent_prediction_2010_2024.csv')

st.subheader("property price over the years")
st.line_chart(data.set_index('Year')['price'])

st.subheader("data preview")
st.dataframe(data)
st.subheader('stailight segmentation example')
img = cv2.imread('images\OIP (6).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #simple segmentation
fig,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('segmented  Image')
ax[0].axis('off')
st.pyplot(fig)
st.subheader("future price prediction")