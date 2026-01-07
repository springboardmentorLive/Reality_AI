import streamlit as st
from PIL import Image

st.title("Upload and Display Image")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption = "Uploaded Image", use_container_width=True)